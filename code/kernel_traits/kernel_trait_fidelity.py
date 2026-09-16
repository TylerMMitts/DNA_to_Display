# Tests whether generated kernels have the size and colour their genotype should give.
#
# The seed counterpart of genetic_fidelity_test.py. Every genotype is generated
# from several noise seeds, each kernel is measured from its pixels, and the
# genotype's mean generated trait is compared with its real kernel - separately
# for genotypes the model trained on and ones it never saw. Only the held-out
# comparison says whether the model learned how genotype sets size and colour;
# the trained one can be matched by recall alone.
#
# Real kernels are also passed through the seed LiteVAE and measured, which is
# the ceiling: a trait the autoencoder cannot carry, the diffusion model cannot
# produce either.
#
# Writes per_sample.csv, genotype_means.csv, trait_statistics.csv,
# fidelity_scatter.png, fidelity_summary.png and summary.json, and keeps every
# generated kernel in generated/ so an interrupted run resumes.

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    SEED_DIFFUSION_MODEL, SEED_LITEVAE_MODEL, SEED_RESULTS_DIR, SEED_SCALED_DIR,
    SEED_SCALED_METADATA, SEED_SNP_PARQUET, apply_overrides, pick_device,
    resolve_input, resolve_output,
)

from kernel_traits.measure_kernel_traits import (
    TRAITS, load_rgb, measure_kernel, scale_from_metadata,
)
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import generate_batch
from latent_diffusion.models.snp_encoder import load_seed_snp_data_from_parquet
from latent_diffusion.training.train_seeds import load_litevae, make_transforms

SPLITS = ('trained', 'held out')
SPLIT_COLORS = {'trained': '#9A9A9A', 'held out': '#E07B39'}


# Agreement between real and generated genotype values for one trait.
#
# chance is the 95th percentile of r with genotype labels shuffled, and p the
# share of shuffles reaching the observed r, so both answer "could an unrelated
# pairing do this well". var_ratio near 0 means the model gives every genotype
# much the same kernel; bias_sd is the average offset in real standard deviations.
def agreement(real, generated, n_permutations, rng):
    ok = ~(np.isnan(real) | np.isnan(generated))
    real, generated = real[ok], generated[ok]
    row = {'n_genotypes': int(len(real))}
    # ptp rather than std: identical values can still give a std a hair above
    # zero, and a correlation over them is noise.
    if len(real) < 3 or np.ptp(real) == 0 or np.ptp(generated) == 0:
        return {**row, 'r': np.nan, 'rho': np.nan, 'chance_r': np.nan, 'p_value': np.nan,
                'var_ratio': np.nan, 'bias_sd': np.nan}
    r = float(np.corrcoef(real, generated)[0, 1])
    null = np.array([np.corrcoef(real, rng.permutation(generated))[0, 1]
                     for _ in range(n_permutations)])
    return {**row, 'r': r, 'rho': float(stats.spearmanr(real, generated)[0]),
            'chance_r': float(np.percentile(np.abs(null), 95)),
            'p_value': float((1 + (null >= r).sum()) / (1 + n_permutations)),
            'var_ratio': float(generated.std() / real.std()),
            'bias_sd': float((generated.mean() - real.mean()) / real.std())}


# Share of the variation between generated kernels that comes from genotype
# rather than noise seed, corrected for the seed noise left in each genotype's
# mean. Near 0 means the seed decides the kernel and the genotype barely matters.
def genotype_share(per_seed):
    groups = [g.dropna().to_numpy() for _, g in per_seed]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 3:
        return np.nan
    k = np.mean([len(g) for g in groups])
    within = np.mean([g.var(ddof=1) for g in groups])
    between = max(np.var([g.mean() for g in groups], ddof=1) - within / k, 0.0)
    return float(between / (between + within)) if between + within > 0 else np.nan


def main(overrides=None):
    # Edit these values, then run:
    #     python code/kernel_traits/kernel_trait_fidelity.py
    class cfg:
        checkpoint = SEED_DIFFUSION_MODEL
        litevae_checkpoint = SEED_LITEVAE_MODEL
        snp_parquet = SEED_SNP_PARQUET
        metadata_path = SEED_SCALED_METADATA
        image_dir = SEED_SCALED_DIR
        pca_cache = SEED_RESULTS_DIR / 'pca.pkl'
        output_dir = SEED_RESULTS_DIR / 'kernel_trait_fidelity'

        # Kernels generated per genotype. There is one real kernel per genotype,
        # so the generated side is averaged to keep one unlucky noise draw from
        # standing in for what the model predicts for that genotype.
        seeds = [0, 1, 2, 3]

        # None -> every genotype. A small number is a quick check that runs the
        # same code on fewer genotypes.
        max_genotypes = None

        measure_reconstructions = True
        n_permutations = 1000
        batch_size = 32
        sampling_steps = 50
        image_size = 256
        latent_size = 32
        seed = 0
        device = pick_device()

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    generated_dir = out / 'generated'
    generated_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_input(cfg.checkpoint, 'seed diffusion checkpoint')
    print(f"Device: {device}\nCheckpoint: {checkpoint_path}\nOutput: {out}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if ckpt.get('dataset') != 'seeds':
        raise SystemExit(f"{checkpoint_path.name} is not a seed model (dataset="
                         f"{ckpt.get('dataset')!r}); train_seeds.py writes dataset='seeds'")
    if 'val_genotypes' not in ckpt:
        raise SystemExit(f"{checkpoint_path.name} records no validation split")
    held_out = set(ckpt['val_genotypes'])
    epoch = ckpt.get('epoch')
    del ckpt

    names, _, snp_matrix = load_seed_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'seed SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    row_of = {n: i for i, n in enumerate(names)}

    image_dir = resolve_input(cfg.image_dir, 'scaled seed image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'seed scaled metadata'))
    px_per_mm = scale_from_metadata(metadata)
    samples = [(r.genotype, image_dir / r.filename) for r in metadata.itertuples()
               if r.genotype in row_of and (image_dir / r.filename).exists()]
    if cfg.max_genotypes is not None:
        samples = samples[:cfg.max_genotypes]
    if not samples:
        raise SystemExit("no kernels have both an image and SNP data")
    split_of = {g: 'held out' if g in held_out else 'trained' for g, _ in samples}
    for split in SPLITS:
        print(f"  {split:9s} {sum(s == split for s in split_of.values()):4d} genotypes")

    rows = []
    for genotype, path in samples:
        rows.append({'source': 'real', 'genotype': genotype, 'split': split_of[genotype],
                     'seed': np.nan, **measure_kernel(load_rgb(path), px_per_mm)})

    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    unet.set_store_attention(False)
    encoder, decoder = load_litevae(
        resolve_input(cfg.litevae_checkpoint, 'seed LiteVAE checkpoint'), device,
        unet_cfg['latent_channels'])

    if cfg.measure_reconstructions:
        print("\nMeasuring LiteVAE reconstructions (the ceiling)...")
        transform = make_transforms(cfg.image_size, False)
        with torch.no_grad():
            for start in range(0, len(samples), cfg.batch_size):
                chunk = samples[start:start + cfg.batch_size]
                batch = torch.stack([transform(Image.open(p).convert('RGB')) for _, p in chunk])
                z = encoder(batch.to(device), save_steps=False)[0]
                images = torch.clamp((decoder(z, save_steps=False) + 1) / 2, 0, 1)
                images = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
                for (genotype, _), image in zip(chunk, images):
                    rows.append({'source': 'reconstruction', 'genotype': genotype,
                                 'split': split_of[genotype], 'seed': np.nan,
                                 **measure_kernel(image, px_per_mm)})

    scheduler = DiffusionScheduler()
    # Plain tensors rather than module buffers, so they are moved by hand.
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    # The same seeds for every genotype, so two genotypes' kernels start from
    # identical noise and differ only through the genotype.
    jobs = [(g, s, generated_dir / f'{g}_seed{s}.png') for g, _ in samples for s in cfg.seeds]
    todo = [j for j in jobs if not j[2].exists()]
    print(f"\nGenerating {len(todo)} kernels ({len(jobs) - len(todo)} already on disk)...")
    start_time = time.time()
    for start in range(0, len(todo), cfg.batch_size):
        chunk = todo[start:start + cfg.batch_size]
        snp_batch = torch.tensor(np.stack([snp_matrix[row_of[g]] for g, _, _ in chunk]),
                                 dtype=torch.float32, device=device)
        images = generate_batch(snp_encoder, unet, scheduler, decoder, snp_batch,
                                [s for _, s, _ in chunk], device, latent_shape,
                                cfg.sampling_steps)
        for (_, _, path), image in zip(chunk, images):
            Image.fromarray(image).save(path)
        done = start + len(chunk)
        print(f"  {done}/{len(todo)}  elapsed {(time.time() - start_time) / 60:.1f} min")

    for genotype, seed, path in jobs:
        rows.append({'source': 'generated', 'genotype': genotype, 'split': split_of[genotype],
                     'seed': seed, **measure_kernel(load_rgb(path), px_per_mm)})

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(out / 'per_sample.csv', index=False)
    detected = per_sample.groupby('source')['detected'].mean()
    print("\nKernel found in the image:")
    for source, rate in detected.items():
        print(f"  {source:15s} {rate:.1%}")

    # Averaged per genotype on every side. There is one real kernel per genotype
    # today, but averaging means a second image per genotype would still work.
    real = per_sample[per_sample.source == 'real'].groupby('genotype')
    recon = per_sample[per_sample.source == 'reconstruction'].groupby('genotype')
    generated = per_sample[per_sample.source == 'generated']
    means = real[['split']].first()
    for t in TRAITS:
        means[f'real_{t}'] = real[t].mean()
        if cfg.measure_reconstructions:
            means[f'reconstruction_{t}'] = recon[t].mean()
        grouped = generated.groupby('genotype')[t]
        means[f'generated_{t}'] = grouped.mean()
        means[f'generated_sd_{t}'] = grouped.std()
    means.to_csv(out / 'genotype_means.csv')

    rng = np.random.default_rng(cfg.seed)
    stat_rows = []
    for t in TRAITS:
        if cfg.measure_reconstructions:
            stat_rows.append({'trait': t, 'comparison': 'reconstruction', 'split': 'all',
                              **agreement(means[f'real_{t}'].to_numpy(),
                                          means[f'reconstruction_{t}'].to_numpy(),
                                          cfg.n_permutations, rng),
                              'genotype_share': np.nan})
        for split in SPLITS:
            sel = means[means.split == split]
            gen_split = generated[generated.split == split]
            stat_rows.append({'trait': t, 'comparison': 'generated', 'split': split,
                              **agreement(sel[f'real_{t}'].to_numpy(),
                                          sel[f'generated_{t}'].to_numpy(),
                                          cfg.n_permutations, rng),
                              'genotype_share': genotype_share(gen_split.groupby('genotype')[t])})
    statistics = pd.DataFrame(stat_rows)
    statistics.to_csv(out / 'trait_statistics.csv', index=False)

    def stat(t, comparison, split, column):
        sel = statistics[(statistics.trait == t) & (statistics.comparison == comparison)
                         & (statistics.split == split)]
        return float(sel[column].iloc[0]) if len(sel) else np.nan

    print(f"\n{'trait':28s} {'LiteVAE r':>9s} {'trained r':>9s} {'held-out r':>10s} "
          f"{'chance':>7s} {'p':>6s} {'var ratio':>9s} {'genotype share':>14s}")
    for t in TRAITS:
        print(f"{t:28s} {stat(t, 'reconstruction', 'all', 'r'):9.2f} "
              f"{stat(t, 'generated', 'trained', 'r'):9.2f} "
              f"{stat(t, 'generated', 'held out', 'r'):10.2f} "
              f"{stat(t, 'generated', 'held out', 'chance_r'):7.2f} "
              f"{stat(t, 'generated', 'held out', 'p_value'):6.3f} "
              f"{stat(t, 'generated', 'held out', 'var_ratio'):9.2f} "
              f"{stat(t, 'generated', 'held out', 'genotype_share'):14.2f}")
    print("\nr         real kernel vs mean generated kernel, across genotypes")
    print("LiteVAE r real vs its own reconstruction - the most generation could reach")
    print("chance    95th percentile of |r| with genotypes shuffled (held out)")
    print("genotype share  how much of the generated variation follows genotype rather")
    print("          than noise seed (held out); near 0 means the seed decides")

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax, (t, label) in zip(axes.ravel(), TRAITS.items()):
        for split in SPLITS:
            sel = means[means.split == split]
            ax.scatter(sel[f'real_{t}'], sel[f'generated_{t}'], s=10, alpha=0.6,
                       color=SPLIT_COLORS[split],
                       label=f"{split} r={stat(t, 'generated', split, 'r'):.2f}")
        values = pd.concat([means[f'real_{t}'], means[f'generated_{t}']]).dropna()
        if len(values):
            lo, hi = values.min(), values.max()
            ax.plot([lo, hi], [lo, hi], color='#444444', linewidth=0.8, linestyle='--')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('real', fontsize=8)
        ax.set_ylabel('generated (mean of seeds)', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(f'Generated vs real kernels by genotype  ({checkpoint_path.name})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / 'fidelity_scatter.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.4))
    x = np.arange(len(TRAITS))
    bars = [('reconstruction', 'all', '#4C78A8', 'LiteVAE reconstruction (ceiling)'),
            ('generated', 'trained', SPLIT_COLORS['trained'], 'generated, trained genotypes'),
            ('generated', 'held out', SPLIT_COLORS['held out'], 'generated, held-out genotypes')]
    for k, (comparison, split, colour, label) in enumerate(bars):
        ax.bar(x + (k - 1) * 0.27, [stat(t, comparison, split, 'r') for t in TRAITS],
               width=0.27, color=colour, label=label)
    ax.scatter(x + 0.27, [stat(t, 'generated', 'held out', 'chance_r') for t in TRAITS],
               marker='_', s=300, color='black', zorder=3, label='held-out chance level')
    ax.axhline(0, color='#444444', linewidth=0.8)
    ax.set_xticks(x, list(TRAITS.values()), rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('r, real vs generated')
    # Above the axes, where it cannot cover a bar whatever the values turn out to be.
    ax.legend(fontsize=8, frameon=False, ncol=4, loc='lower center',
              bbox_to_anchor=(0.5, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.suptitle('Does the genotype set kernel size and colour?', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'fidelity_summary.png', dpi=130)
    plt.close(fig)

    summary = {
        'checkpoint': str(checkpoint_path), 'epoch': epoch,
        'n_genotypes': len(samples),
        'n_held_out_genotypes': sum(s == 'held out' for s in split_of.values()),
        'seeds': list(cfg.seeds), 'sampling_steps': cfg.sampling_steps,
        'px_per_mm': px_per_mm,
        'detection_rate': {k: float(v) for k, v in detected.items()},
        'held_out_r': {t: stat(t, 'generated', 'held out', 'r') for t in TRAITS},
        'held_out_chance_r': {t: stat(t, 'generated', 'held out', 'chance_r') for t in TRAITS},
        'trained_r': {t: stat(t, 'generated', 'trained', 'r') for t in TRAITS},
        'reconstruction_r': {t: stat(t, 'reconstruction', 'all', 'r') for t in TRAITS},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote per_sample.csv, genotype_means.csv, trait_statistics.csv, "
          f"fidelity_scatter.png, fidelity_summary.png and summary.json to {out}")


if __name__ == '__main__':
    main()
