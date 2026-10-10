# What each of the eight founder parents' kernels is predicted to look like,
# from the seed model, and how well that prediction holds up.
#
# The root version, generate_parent_archetypes.py, generates each founder by
# setting every locus to that founder. This does the same for kernels and adds
# two checks the roots never had, because kernels are measured without a
# segmenter and there are far more lines:
#   - real kernels for comparison: for each founder, the real lines that carry
#     the most of that founder's genome, beside its generated archetype;
#   - an additive expectation: each kernel trait regressed on the share of the
#     genome from each founder, across all imaged lines. The regression's value
#     at 100% of one founder is what that founder "should" score if founders
#     add up; the generated archetype is compared against it trait by trait.
# Pure founders do not exist in the population, so the out-of-distribution
# check from the root script is kept: how far each pure founder sits from real
# genotypes in the encoder's PCA space.
#
# Writes to results/seeds/parent_archetypes/: parents_side_by_side.png,
# parents_multi_seed.png, parents_vs_real_kernels.png, purity_sweep.png,
# out_of_distribution.png, additive_vs_generated.png,
# traits.csv, additive_expectation.csv, summary.json.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    SEED_DIFFUSION_DIR, SEED_RESULTS_DIR, SEED_SCALED_DIR, SEED_SCALED_METADATA,
    SEED_SNP_PARQUET, apply_overrides, find_latest_checkpoint, pick_device,
    resolve_input, resolve_output,
)

from kernel_traits.measure_kernel_traits import TRAITS, load_rgb, measure_kernel, scale_from_metadata
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import generate_batch
from latent_diffusion.generation.generate_parent_archetypes import (
    enriched_vector, ood_distance, pca_reconstruction_error, pure_parent_vector,
    save_ood_figure, save_purity_sweep,
)
from latent_diffusion.utils.dataset_inputs import checkpoint_dataset, load_decoder, load_snp_table

# Traits shown in the additive comparison figure: size, shape and the colour
# traits that differ most between founders.
FIGURE_TRAITS = ['area_mm2', 'aspect_ratio', 'lightness', 'red_green', 'yellow_blue', 'hue_deg']


# Founder names in code order. load_seed_snp_data_from_parquet numbers the
# founders by their sorted names, so the same sort gives the names back.
def seed_founder_names(parquet):
    calls = pd.read_parquet(parquet, columns=['genotype'])['genotype'].astype(str)
    return sorted(calls.str[:calls.str.len().iloc[0] // 2].unique())


# Share of each line's called loci from each founder (uncalled loci are -1).
def founder_shares(snp_matrix, founders):
    called = (snp_matrix > 0).sum(axis=1).clip(min=1)
    return np.stack([(snp_matrix == f).sum(axis=1) / called for f in founders], axis=1)


# Additive expectation for a pure founder: least squares of a trait on the
# eight founder shares, no intercept. The shares sum to 1, so the coefficient
# of founder k is the fitted value at 100% founder k.
def additive_expectation(shares, values):
    ok = np.isfinite(values)
    coef, *_ = np.linalg.lstsq(shares[ok], values[ok], rcond=None)
    fitted = shares[ok] @ coef
    r2 = 1 - ((values[ok] - fitted) ** 2).sum() / ((values[ok] - values[ok].mean()) ** 2).sum()
    return coef, float(r2)


# cell_labels, when given, names every image underneath it (same shape as
# images), for grids whose cells are not one thing per column.
def save_grid(images, row_labels, col_labels, save_path, title, cell_labels=None):
    n_rows, n_cols = len(images), len(images[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 1.95 * n_rows), squeeze=False)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r][c]
            img = images[r][c]
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if cell_labels is not None:
                ax.set_xlabel(cell_labels[r][c] or '', fontsize=8.5)
            elif r == 0:
                ax.set_title(col_labels[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=8.5, rotation=0, ha='right', va='center', labelpad=8)
    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_additive_figure(expect, generated, real_lines, names, r2, save_path):
    fig, axes = plt.subplots(1, len(FIGURE_TRAITS), figsize=(3.3 * len(FIGURE_TRAITS), 3.6))
    for ax, t in zip(axes, FIGURE_TRAITS):
        x, y = expect[t].to_numpy(), generated[t].to_numpy()
        lo = np.nanpercentile(real_lines[t], 2.5); hi = np.nanpercentile(real_lines[t], 97.5)
        ax.axhspan(lo, hi, color='0.92', zorder=0)
        ax.axvspan(lo, hi, color='0.92', zorder=0)
        ax.scatter(x, y, color='#1f77b4', zorder=3)
        for xi, yi, n in zip(x, y, names):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(n, (xi, yi), textcoords='offset points', xytext=(4, 3), fontsize=7.5)
        both = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)], [lo, hi]])
        pad = 0.08 * (both.max() - both.min() or 1)
        ax.plot([both.min() - pad, both.max() + pad], [both.min() - pad, both.max() + pad],
                ls='--', color='crimson', lw=1)
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else float('nan')
        ax.set_title(f"{TRAITS[t]}\nr = {r:.2f}   additive R² = {r2[t]:.2f}", fontsize=9)
        ax.set_xlabel('additive expectation', fontsize=8.5)
        ax.set_ylabel('generated archetype', fontsize=8.5)
    fig.suptitle('Generated founder archetypes against what the real lines predict for a pure founder\n'
                 'grey band = middle 95% of real lines; dashed = exact agreement', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/generation/generate_seed_parent_archetypes.py
    class cfg:
        # None -> the newest numbered checkpoint in SEED_DIFFUSION_DIR.
        checkpoint = None
        litevae_checkpoint = None        # None -> the seed LiteVAE in paths.py
        snp_parquet = SEED_SNP_PARQUET
        metadata_path = SEED_SCALED_METADATA
        image_dir = SEED_SCALED_DIR
        output_dir = SEED_RESULTS_DIR / 'parent_archetypes'

        seeds = [0, 1, 2, 3, 4]
        # Fraction of loci set to the founder, from a population-typical line.
        # None skips the sweep.
        purities = [0.125, 0.25, 0.5, 0.75, 1.0]
        # Real lines shown per founder, most founder-rich first.
        n_real_per_founder = 3

        sampling_steps = 50
        latent_size = 32
        batch_size = 8
        device = pick_device()

    apply_overrides(cfg, overrides)
    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    (out / 'images').mkdir(parents=True, exist_ok=True)

    checkpoint = Path(resolve_input(cfg.checkpoint, 'seed checkpoint') if cfg.checkpoint
                      else find_latest_checkpoint(resolve_input(SEED_DIFFUSION_DIR, 'seed model folder')))
    if checkpoint_dataset(checkpoint) != 'seeds':
        raise SystemExit(f"{checkpoint.name} is not a seed model")
    print(f"Checkpoint: {checkpoint}\nOutput: {out}")

    names, _, snp_matrix = load_snp_table('seeds', cfg.snp_parquet)
    names, snp_matrix = list(names), np.asarray(snp_matrix)
    founder_names = seed_founder_names(resolve_input(cfg.snp_parquet, 'seed SNP parquet'))
    founders = list(range(1, len(founder_names) + 1))
    label = {f: founder_names[f - 1] for f in founders}
    row = {n: i for i, n in enumerate(names)}
    shares = founder_shares(snp_matrix, founders)
    print("Founders: " + ", ".join(f"{f}={label[f]}" for f in founders))
    print(f"Largest single-founder share in a real line: {shares.max():.0%}")

    # Real kernels, measured exactly as generated ones will be.
    meta = pd.read_csv(resolve_input(cfg.metadata_path, 'seed metadata'))
    px_per_mm = scale_from_metadata(meta)
    image_dir = Path(resolve_input(cfg.image_dir, 'seed image folder'))
    real_rows = []
    for r in meta.itertuples():
        p = image_dir / r.filename
        if p.exists():
            real_rows.append({'genotype': r.genotype, 'filename': r.filename,
                              **measure_kernel(load_rgb(p), px_per_mm)})
    real = pd.DataFrame(real_rows)
    lines = real[real.genotype.isin(row) & real.detected].groupby('genotype')[list(TRAITS)].mean()
    print(f"Real kernels measured: {len(real)}; lines with genotype data: {len(lines)}")

    # Additive expectation per founder and trait.
    line_shares = shares[[row[g] for g in lines.index]]
    expect, r2 = {}, {}
    for t in TRAITS:
        coef, r2[t] = additive_expectation(line_shares, lines[t].to_numpy())
        expect[t] = coef
    expect = pd.DataFrame(expect, index=[label[f] for f in founders])
    expect.to_csv(out / 'additive_expectation.csv')

    # Out of distribution, as in the root script.
    snp_encoder, unet, unet_cfg = load_model(checkpoint, snp_matrix, device)
    unet.set_store_attention(False)
    # The PCA distance check only applies to PCA models; a founder-window model
    # has no PCA, so its founders are labelled without a distance.
    pca = snp_encoder.pca
    real_rms, real_recon = float('nan'), np.array([np.nan])
    ood = pd.DataFrame({'parent': founders, 'founder': [label[f] for f in founders],
                        'rms_z': np.nan, 'max_abs_z': np.nan, 'pca_reconstruction_error': np.nan})
    if pca is not None:
        population = pca.transform(snp_matrix)
        mean, std = population.mean(axis=0), np.where(population.std(axis=0) > 0, population.std(axis=0), 1)
        real_rms = float(np.mean(np.sqrt((((population - mean) / std) ** 2).mean(axis=1))))
        real_recon = pca_reconstruction_error(pca, snp_matrix)
        rows = []
        for f in founders:
            vec = pure_parent_vector(snp_matrix.shape[1], f)
            rms, mx = ood_distance(pca, population, vec)
            rows.append({'parent': f, 'founder': label[f], 'rms_z': rms, 'max_abs_z': mx,
                         'pca_reconstruction_error': float(pca_reconstruction_error(pca, vec)[0])})
        ood = pd.DataFrame(rows)
        ood.to_csv(out / 'out_of_distribution.csv', index=False)
        save_ood_figure(ood, real_rms, out / 'out_of_distribution.png')

    # Real reference images: no founder was imaged, so the nearest real thing
    # is the lines that carry the most of each founder's genome.
    def real_image(g):
        f = real.loc[real.genotype == g, 'filename']
        return load_rgb(image_dir / f.iloc[0]) if len(f) else None
    imaged = [g for g in lines.index]
    top = {f: sorted(imaged, key=lambda g: -shares[row[g], f - 1])[:cfg.n_real_per_founder] for f in founders}

    try:
        decoder = load_decoder('seeds', device, unet_cfg['latent_channels'], cfg.litevae_checkpoint)
    except (SystemExit, FileNotFoundError) as exc:
        decoder = None
        print(f"\nNo seed LiteVAE ({exc}); writing the parts that need no images and stopping.")

    summary = {'checkpoint': str(checkpoint), 'founders': label, 'px_per_mm': px_per_mm,
               'largest_real_founder_share': float(shares.max()),
               'mean_real_genotype_rms_z': real_rms,
               'real_pca_reconstruction_error_mean': float(real_recon.mean()),
               'pure_founder_rms_z': dict(zip(ood.founder, ood.rms_z)),
               'additive_r2': r2,
               'most_founder_rich_lines': {label[f]: [(g, float(shares[row[g], f - 1])) for g in top[f]]
                                           for f in founders}}
    if decoder is None:
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
        return

    scheduler = DiffusionScheduler()
    for a in ('betas', 'alphas', 'alpha_bars'):
        setattr(scheduler, a, getattr(scheduler, a).to(device))
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    def generate(vectors, seeds):
        imgs = []
        t = torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)
        for s in range(0, len(vectors), cfg.batch_size):
            imgs.extend(generate_batch(snp_encoder, unet, scheduler, decoder, t[s:s + cfg.batch_size],
                                       seeds[s:s + cfg.batch_size], device, latent_shape, cfg.sampling_steps))
        return imgs

    # Pure founders: the same seeds for every founder, so a row differs only
    # by founder and a column only by noise.
    pure = [pure_parent_vector(snp_matrix.shape[1], f) for f in founders]
    by_seed, trait_rows = {}, []
    for seed in cfg.seeds:
        by_seed[seed] = generate(pure, [seed] * len(pure))
        for f, img in zip(founders, by_seed[seed]):
            Image.fromarray(img).save(out / 'images' / f'parent_{label[f]}_seed{seed}.png')
            trait_rows.append({'founder': label[f], 'seed': seed, **measure_kernel(img, px_per_mm)})
    traits = pd.DataFrame(trait_rows)
    traits.to_csv(out / 'traits.csv', index=False)
    generated = traits[traits.detected].groupby('founder')[list(TRAITS)].mean().reindex(expect.index)

    names_row = [label[f] for f in founders]
    sub = [f"{label[f]}\n{ood.loc[ood.parent == f, 'rms_z'].iloc[0]:.1f} sd from population"
           if pca is not None else label[f] for f in founders]
    save_grid([by_seed[cfg.seeds[0]]], [f'seed {cfg.seeds[0]}'], sub, out / 'parents_side_by_side.png',
              'Founder archetypes from the seed model (every locus set to one founder)')
    save_grid([by_seed[s] for s in cfg.seeds], [f'seed {s}' for s in cfg.seeds], names_row,
              out / 'parents_multi_seed.png',
              'Founder archetypes across noise seeds: down a column is noise, across a row is the founder')

    # Generated archetype on top, the most founder-rich real lines below it.
    rows_img = [by_seed[cfg.seeds[0]]]
    row_labels = ['generated']
    for i in range(cfg.n_real_per_founder):
        rows_img.append([real_image(top[f][i]) for f in founders])
        row_labels.append(f'real, #{i + 1}\nmost {""}founder-rich')
    col_labels = [f"{label[f]}\n" + ", ".join(f"{shares[row[g], f - 1]:.0%}" for g in top[f]) for f in founders]
    save_grid(rows_img, row_labels, col_labels, out / 'parents_vs_real_kernels.png',
              'Generated founder archetypes above the real lines richest in that founder '
              '(share of genome from it under each name)')

    save_additive_figure(expect, generated, lines, list(expect.index), r2, out / 'additive_vs_generated.png')

    if cfg.purities:
        rng = np.random.default_rng(cfg.seeds[0])
        base = snp_matrix[rng.integers(len(snp_matrix))]
        sweep = {}
        for f in founders:
            vecs = [enriched_vector(base, f, p, rng) for p in cfg.purities]
            sweep[f] = generate(vecs, [cfg.seeds[0]] * len(vecs))
        save_purity_sweep(sweep, founders, cfg.purities, out / 'purity_sweep.png')

    agree = {}
    for t in TRAITS:
        x, y = expect[t].to_numpy(), generated[t].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        agree[t] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float('nan')
    summary.update({'detected_share': float(traits.detected.mean()),
                    'generated_vs_additive_r': agree,
                    'generated_traits': generated.round(3).to_dict(orient='index'),
                    'additive_expectation': expect.round(3).to_dict(orient='index')})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nKernel detected in {traits.detected.mean():.0%} of generated founder images")
    print("Generated archetype vs additive expectation, r across the 8 founders:")
    for t in TRAITS:
        print(f"  {TRAITS[t]:<28} r {agree[t]:+.2f}   additive R² {r2[t]:.2f}")
    print(f"Wrote figures and tables to {out}")


if __name__ == '__main__':
    main()
