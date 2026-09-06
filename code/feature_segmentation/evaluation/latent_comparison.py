# How close a generated latent sits to real latents of the same genotype,
# measured against two baselines: real replicates of that genotype, and
# unrelated genotypes.

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
    CROPPED_IMAGES_DIR, IMAGE_METADATA, LITEVAE_MODEL, RESULTS_DIR,
    resolve_input, resolve_output,
)

from feature_segmentation.evaluation.latent_average_test import (
    load_litevae, load_image, encode,
)


# Comparison metrics

def latent_metrics(z_real, z_gen):
    # Per-image agreement between two [C, H, W] latents.
    a = z_real.reshape(-1).astype(np.float64)
    b = z_gen.reshape(-1).astype(np.float64)

    diff = b - a
    l2 = float(np.linalg.norm(diff))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(a @ b / denom) if denom > 0 else np.nan

    ac, bc = a - a.mean(), b - b.mean()
    dc = np.linalg.norm(ac) * np.linalg.norm(bc)
    pearson = float(ac @ bc / dc) if dc > 0 else np.nan

    out = {
        'l2_distance': l2,
        'rmse': float(np.sqrt((diff ** 2).mean())),
        'cosine': cosine,
        'pearson': pearson,
        'real_norm': float(np.linalg.norm(a)),
        'gen_norm': float(np.linalg.norm(b)),
    }
    for c in range(z_real.shape[0]):
        out[f'real_ch{c}_mean'] = float(z_real[c].mean())
        out[f'gen_ch{c}_mean'] = float(z_gen[c].mean())
        out[f'real_ch{c}_std'] = float(z_real[c].std())
        out[f'gen_ch{c}_std'] = float(z_gen[c].std())
    return out


def pair_distances(latents, genotypes, same_genotype, n_pairs, seed=0):
    # Samples real-vs-real latent distances, within or across genotypes.
    rng = np.random.default_rng(seed)
    by_genotype = {}
    for i, g in enumerate(genotypes):
        by_genotype.setdefault(g, []).append(i)

    eligible = [g for g, idx in by_genotype.items() if len(idx) >= 2]
    if same_genotype and not eligible:
        return np.array([])

    distances = []
    attempts = 0
    while len(distances) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        if same_genotype:
            g = eligible[rng.integers(len(eligible))]
            i, j = rng.choice(by_genotype[g], size=2, replace=False)
        else:
            i, j = rng.integers(len(latents), size=2)
            if genotypes[i] == genotypes[j]:
                continue
        distances.append(float(np.linalg.norm(
            latents[j].reshape(-1) - latents[i].reshape(-1))))
    return np.array(distances)


# Figures

def save_latent_figure(name, genotype, real_img, gen_img, z_real, z_gen,
                       metrics, save_path):
    n_ch = z_real.shape[0]
    fig = plt.figure(figsize=(3.0 * (n_ch + 1) + 3.4, 8.2))
    grid = fig.add_gridspec(3, n_ch + 2, width_ratios=[1.15] + [1] * n_ch + [1.5],
                            hspace=0.18, wspace=0.10)

    for row, (img, title) in enumerate(((real_img, 'Real image'),
                                        (gen_img, 'Generated from SNPs'))):
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    ax = fig.add_subplot(grid[2, 0])
    ax.text(0.5, 0.5, 'latent difference\n(generated - real)', ha='center',
            va='center', fontsize=10, color='0.35', transform=ax.transAxes)
    ax.axis('off')

    for c in range(n_ch):
        # One shared colour scale per channel across both rows, so the real and
        # generated maps are directly comparable; independent scaling would make
        # very different latents look alike.
        lo = float(min(z_real[c].min(), z_gen[c].min()))
        hi = float(max(z_real[c].max(), z_gen[c].max()))

        for row, z in ((0, z_real), (1, z_gen)):
            ax = fig.add_subplot(grid[row, c + 1])
            ax.imshow(z[c], cmap='viridis', vmin=lo, vmax=hi, interpolation='nearest')
            if row == 0:
                ax.set_title(f'channel {c}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        diff = z_gen[c] - z_real[c]
        limit = float(np.abs(diff).max()) or 1.0
        ax = fig.add_subplot(grid[2, c + 1])
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-limit, vmax=limit,
                       interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[:, n_ch + 1])
    ax.axis('off')
    lines = [
        f"{'L2 distance':<16}{metrics['l2_distance']:>10.2f}",
        f"{'RMSE':<16}{metrics['rmse']:>10.4f}",
        f"{'cosine':<16}{metrics['cosine']:>10.3f}",
        f"{'pearson':<16}{metrics['pearson']:>10.3f}",
        f"{'|z| real':<16}{metrics['real_norm']:>10.2f}",
        f"{'|z| generated':<16}{metrics['gen_norm']:>10.2f}",
        '',
        f"{'ch':<4}{'real mu':>9}{'gen mu':>9}",
    ]
    for c in range(n_ch):
        lines.append(f"{c:<4}{metrics[f'real_ch{c}_mean']:>9.3f}"
                     f"{metrics[f'gen_ch{c}_mean']:>9.3f}")
    lines.append('')
    lines.append(f"{'ch':<4}{'real sd':>9}{'gen sd':>9}")
    for c in range(n_ch):
        lines.append(f"{c:<4}{metrics[f'real_ch{c}_std']:>9.3f}"
                     f"{metrics[f'gen_ch{c}_std']:>9.3f}")

    ax.text(0.0, 0.98, '\n'.join(lines), family='monospace', fontsize=9.5,
            va='top', ha='left', transform=ax.transAxes)

    # Figure footer rather than inside the stats column, which would collide
    # with the difference row's colorbars.
    fig.text(0.5, 0.015,
             'The generated image samples this genotype, not this photo, so a nonzero '
             'distance is expected. Read it against the same- and different-genotype '
             'baselines in summary.png.',
             fontsize=8.5, color='0.35', ha='center', va='bottom')

    fig.suptitle(f'{name}   genotype {genotype}', fontsize=11, y=0.965)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def pca_latent_space(z_reals, z_gens, n_components=10, seed=0):
    # Projects real and generated latents into a shared PCA space.
    #
    # Fitted on real and generated stacked together, so neither side defines the
    # axes. Fitting on the real latents alone and projecting the generated ones in
    # would answer a different question - how generated images look along the
    # axes of real variation - and would understate any generated variation that
    # lies in directions the real data does not span.
    from sklearn.decomposition import PCA

    real_flat = z_reals.reshape(len(z_reals), -1)
    gen_flat = z_gens.reshape(len(z_gens), -1)
    combined = np.concatenate([real_flat, gen_flat], axis=0)

    n_components = int(min(n_components, combined.shape[0] - 1, combined.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(combined)

    real_c, gen_c = coords[:len(real_flat)], coords[len(real_flat):]

    # Separation of the two clouds in the FULL latent space, not just the two
    # plotted components - a multivariate standardised distance between the
    # centroids. Reading separation off the scatter alone would miss any offset
    # that lives in the components not being shown.
    centroid_real = real_flat.mean(axis=0)
    centroid_gen = gen_flat.mean(axis=0)
    centroid_distance = float(np.linalg.norm(centroid_gen - centroid_real))
    pooled = float(np.sqrt(
        (real_flat.var(axis=0, ddof=1).sum() + gen_flat.var(axis=0, ddof=1).sum()) / 2))
    separation = centroid_distance / pooled if pooled > 0 else float('nan')

    return pca, real_c, gen_c, {
        'centroid_distance': centroid_distance,
        'centroid_separation': separation,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    }


def save_pca_figure(pca, real_c, gen_c, stats, save_path):
    ev = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.scatter(real_c[:, 0], real_c[:, 1], s=22, alpha=0.55,
               label='real', color='seagreen', edgecolor='none')
    ax.scatter(gen_c[:, 0], gen_c[:, 1], s=22, alpha=0.55,
               label='generated', color='steelblue', edgecolor='none')
    ax.scatter(*real_c[:, :2].mean(axis=0), s=220, marker='X',
               color='darkgreen', edgecolor='white', linewidth=1.5, zorder=5)
    ax.scatter(*gen_c[:, :2].mean(axis=0), s=220, marker='X',
               color='navy', edgecolor='white', linewidth=1.5, zorder=5)
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% var)' if len(ev) > 1 else 'PC2')
    ax.set_title('Latent space, real vs generated\n(X = centroid)')
    ax.legend(fontsize=9)

    ax = axes[1]
    for coords, label, color in ((real_c, 'real', 'seagreen'),
                                 (gen_c, 'generated', 'steelblue')):
        ax.hist(coords[:, 0], bins=35, alpha=0.55, label=label,
                color=color, density=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('density')
    ax.set_title('PC1 marginal distribution')
    ax.legend(fontsize=9)

    ax = axes[2]
    k = len(ev)
    ax.bar(np.arange(1, k + 1), ev * 100, color='0.6')
    ax.set_xlabel('component')
    ax.set_ylabel('% variance explained')
    ax.set_title(f'Scree\ncentroid separation = {stats["centroid_separation"]:.3f}')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_summary_figure(df, same_d, diff_d, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    bins = 40
    if len(same_d):
        ax.hist(same_d, bins=bins, alpha=0.6, label='real vs real, same genotype',
                color='seagreen', density=True)
    if len(diff_d):
        ax.hist(diff_d, bins=bins, alpha=0.6, label='real vs real, diff genotype',
                color='0.6', density=True)
    ax.hist(df['l2_distance'].dropna(), bins=bins, alpha=0.6,
            label='real vs generated', color='steelblue', density=True)
    ax.set_xlabel('latent L2 distance')
    ax.set_ylabel('density')
    ax.set_title('Where does real-vs-generated sit?')
    ax.legend(fontsize=8)

    ax = axes[1]
    labels, values, colors = [], [], []
    if len(same_d):
        labels.append('same genotype\n(floor)'); values.append(same_d.mean()); colors.append('seagreen')
    labels.append('real vs\ngenerated'); values.append(df['l2_distance'].mean()); colors.append('steelblue')
    if len(diff_d):
        labels.append('diff genotype\n(ceiling)'); values.append(diff_d.mean()); colors.append('0.6')
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('mean latent L2 distance')
    ax.set_title('Mean distance vs. both baselines')

    ax = axes[2]
    ax.hist(df['cosine'].dropna(), bins=40, color='darkorange', alpha=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('cosine similarity, real vs generated latent')
    ax.set_ylabel('images')
    ax.set_title('Latent direction agreement')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/latent_comparison.py
    class cfg:
        # Generated images cached by genetic_fidelity_test.py.
        generated_dir = RESULTS_DIR / 'genetic_fidelity/generated'
        # Same real images that comparison used - cropped roots, not
        # dataset/images (see the note in genetic_fidelity_test.py).
        image_dir = CROPPED_IMAGES_DIR
        metadata_path = IMAGE_METADATA
        litevae_checkpoint = LITEVAE_MODEL

        output_dir = RESULTS_DIR / 'latent_comparison'

        # Per-image figures are capped; the statistics always use everything.
        # None -> a figure for every image.
        max_figures = 120

        n_baseline_pairs = 2000   # sampled real-vs-real pairs per baseline
        pca_components = 10       # components fitted for the latent-space PCA figure
        batch_size = 16
        imgsz = 256
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    (out / 'comparisons').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    gen_dir = Path(resolve_input(cfg.generated_dir, 'generated image directory'))
    image_dir = Path(resolve_input(cfg.image_dir, 'real image directory'))
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))

    genotype_by_stem = {Path(r['new_filename']).stem: r['genotype']
                        for _, r in metadata.iterrows()}
    original_by_stem = {Path(r['new_filename']).stem: image_dir / r['new_filename']
                        for _, r in metadata.iterrows()}

    generated_files = sorted(p for p in gen_dir.iterdir()
                             if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not generated_files:
        raise SystemExit(
            f"no generated images in {gen_dir}.\n"
            f"Run genetic_fidelity_test.py first - it writes that cache.")

    pairs, n_missing = [], 0
    for gen_path in generated_files:
        stem = gen_path.stem
        real_path = original_by_stem.get(stem)
        if real_path is None or not real_path.exists():
            n_missing += 1
            continue
        pairs.append({'stem': stem, 'genotype': genotype_by_stem.get(stem, 'unknown'),
                      'real_path': real_path, 'gen_path': gen_path})

    print(f"Generated images: {len(generated_files)}")
    if n_missing:
        print(f"  skipped (no matching real image): {n_missing}")
    print(f"  paired: {len(pairs)}")

    encoder, _ = load_litevae(resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                              device)

    # encode both sides
    records, z_reals, z_gens, n_figures = [], [], [], 0
    max_figs = len(pairs) if cfg.max_figures is None else cfg.max_figures

    for start in range(0, len(pairs), cfg.batch_size):
        batch = pairs[start:start + cfg.batch_size]

        real_t = torch.stack([load_image(p['real_path'], cfg.imgsz) for p in batch]).to(device)
        gen_t = torch.stack([load_image(p['gen_path'], cfg.imgsz) for p in batch]).to(device)

        z_real = encode(encoder, real_t).cpu().numpy()
        z_gen = encode(encoder, gen_t).cpu().numpy()

        for i, p in enumerate(batch):
            m = latent_metrics(z_real[i], z_gen[i])
            records.append({'image': p['stem'], 'genotype': p['genotype'], **m})
            z_reals.append(z_real[i])
            z_gens.append(z_gen[i])

            if n_figures < max_figs:
                save_latent_figure(
                    p['stem'], p['genotype'],
                    np.array(Image.open(p['real_path']).convert('RGB')
                             .resize((cfg.imgsz, cfg.imgsz), Image.LANCZOS)),
                    np.array(Image.open(p['gen_path']).convert('RGB')
                             .resize((cfg.imgsz, cfg.imgsz), Image.LANCZOS)),
                    z_real[i], z_gen[i], m,
                    out / 'comparisons' / f'{p["stem"]}.png')
                n_figures += 1

        done = start + len(batch)
        if done % (cfg.batch_size * 5) == 0 or done == len(pairs):
            print(f"  {done}/{len(pairs)}")

    df = pd.DataFrame(records)
    df.to_csv(out / 'latent_metrics.csv', index=False)

    # baselines from the real images alone
    z_reals = np.stack(z_reals)
    z_gens = np.stack(z_gens)
    genotypes = df['genotype'].tolist()
    same_d = pair_distances(z_reals, genotypes, True, cfg.n_baseline_pairs, cfg.seed)
    diff_d = pair_distances(z_reals, genotypes, False, cfg.n_baseline_pairs, cfg.seed)

    save_summary_figure(df, same_d, diff_d, out / 'summary.png')

    # PCA of the latent space, real vs generated
    pca, real_c, gen_c, pca_stats = pca_latent_space(
        z_reals, z_gens, cfg.pca_components, cfg.seed)
    save_pca_figure(pca, real_c, gen_c, pca_stats, out / 'pca_latent_space.png')

    n_plot = real_c.shape[1]
    pca_df = pd.concat([
        pd.DataFrame({'image': df['image'], 'genotype': df['genotype'], 'kind': 'real',
                      **{f'PC{i+1}': real_c[:, i] for i in range(n_plot)}}),
        pd.DataFrame({'image': df['image'], 'genotype': df['genotype'], 'kind': 'generated',
                      **{f'PC{i+1}': gen_c[:, i] for i in range(n_plot)}}),
    ], ignore_index=True)
    pca_df.to_csv(out / 'pca_coordinates.csv', index=False)

    gen_mean = float(df['l2_distance'].mean())
    same_mean = float(same_d.mean()) if len(same_d) else float('nan')
    diff_mean = float(diff_d.mean()) if len(diff_d) else float('nan')

    # 0 = as close as two real replicates of the same genotype; 1 = no closer
    # than two unrelated genotypes.
    span = diff_mean - same_mean
    position = (gen_mean - same_mean) / span if np.isfinite(span) and span > 0 else float('nan')

    # How well latent distance separates same-genotype from different-genotype
    # pairs at all, as a standardised effect size. This has to be checked before
    # `position` means anything: position divides by the gap between the two
    # baselines, so if the baselines overlap heavily the denominator is mostly
    # noise and the ratio swings wildly for reasons that have nothing to do with
    # the model. A small effect size here is a statement about LiteVAE's latent
    # space - genotype accounts for little of the distance between images - not
    # about the diffusion model.
    if len(same_d) and len(diff_d):
        pooled = np.sqrt((same_d.var(ddof=1) + diff_d.var(ddof=1)) / 2)
        separation = float((diff_mean - same_mean) / pooled) if pooled > 0 else float('nan')
    else:
        separation = float('nan')

    summary = {
        'n_pairs': len(df),
        'n_genotypes': int(df['genotype'].nunique()),
        'mean_l2_real_vs_generated': gen_mean,
        'mean_l2_same_genotype': same_mean,
        'mean_l2_different_genotype': diff_mean,
        'position_between_baselines': position,
        'baseline_separation_cohens_d': separation,
        'mean_cosine': float(df['cosine'].mean()),
        'mean_pearson': float(df['pearson'].mean()),
        'mean_norm_real': float(df['real_norm'].mean()),
        'mean_norm_generated': float(df['gen_norm'].mean()),
        'pca_centroid_distance': pca_stats['centroid_distance'],
        'pca_centroid_separation': pca_stats['centroid_separation'],
        'pca_explained_variance_ratio': pca_stats['explained_variance_ratio'],
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Latent distance ({len(df)} images, {summary['n_genotypes']} genotypes) ===")
    print(f"  same genotype, real vs real   {same_mean:>10.2f}   (floor)")
    print(f"  real vs generated             {gen_mean:>10.2f}")
    print(f"  diff genotype, real vs real   {diff_mean:>10.2f}   (ceiling)")
    print(f"\n  baseline separation (Cohen's d) {separation:>8.3f}")
    if np.isfinite(separation) and abs(separation) < 0.2:
        print("     WARNING: the two baselines barely separate, so latent distance")
        print("     hardly distinguishes genotype at all in this latent space.")
        print("     'position' below divides by that near-zero gap - treat it as")
        print("     unreliable, and prefer the trait-level analysis instead.")

    print(f"\n  position between baselines    {position:>10.3f}")
    print("     0.0 = generated sits as close to its genotype as a real replicate")
    print("     1.0 = no closer than an unrelated genotype (conditioning unused)")
    print(f"\n  mean cosine  {summary['mean_cosine']:.3f}    "
          f"mean pearson {summary['mean_pearson']:.3f}")
    print(f"  mean |z| real {summary['mean_norm_real']:.2f}   "
          f"generated {summary['mean_norm_generated']:.2f}")

    ev = pca_stats['explained_variance_ratio']
    print(f"\n=== Latent-space PCA (real + generated fitted together) ===")
    print(f"  PC1 {ev[0]*100:.1f}% variance" +
          (f", PC2 {ev[1]*100:.1f}%" if len(ev) > 1 else ''))
    print(f"  centroid separation           {pca_stats['centroid_separation']:>10.3f}")
    print("     distance between the real and generated centroids, in units of")
    print("     the clouds' own spread. Near 0 = the two distributions sit on top")
    print("     of each other; large = generated latents occupy a different region")

    print(f"\nWrote {n_figures} comparison figures, latent_metrics.csv, "
          f"pca_coordinates.csv, summary.png, pca_latent_space.png and "
          f"summary.json to {out}")


if __name__ == '__main__':
    main()
