# Does a generated root actually carry its genotype's traits?
#
# The end-to-end test of the whole idea. Generates an image for every real
# image, measures both with the segmentation model, and correlates the
# genotype means - against a shuffled-label null, since some correlation
# arises by chance.

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
    CROPPED_IMAGES_DIR, DIFFUSION_ONEHOT_DIR, IMAGE_METADATA, LITEVAE_MODEL,
    RESULTS_DIR, SEGMENTATION_MODEL, SNP_PARQUET, find_latest_checkpoint,
    resolve_input, resolve_output,
)

from ultralytics import YOLO

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.generation.generate_from_dataset import (
    generate_batch, load_original,
)
from feature_segmentation.evaluation.reconstruction_fidelity_test import (
    measure, overlay,
)

TRAITS = [
    ('root_diameter_px', 'Root diameter', '{:.1f}'),
    ('stele_diameter_px', 'Stele diameter', '{:.1f}'),
    ('vessel_total_area_px', 'Vessel area', '{:.0f}'),
    ('vessel_count_cc', 'Vessel count', '{:.0f}'),
    ('stele_root_diameter_ratio', 'Stele/root ratio', '{:.3f}'),
]


def repeatability(values, groups):

    df = pd.DataFrame({'value': values, 'group': groups}).dropna()
    if df['group'].nunique() < 2:
        return np.nan, np.nan, np.nan

    grand = df['value'].mean()
    counts = df.groupby('group')['value'].size()
    means = df.groupby('group')['value'].mean()

    k = len(counts)
    N = int(counts.sum())
    if N <= k:
        return np.nan, np.nan, np.nan

    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float(((df['value'] - df['group'].map(means)) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)

    # Average group size correction for unbalanced designs.
    n0 = (N - (counts ** 2).sum() / N) / (k - 1)
    var_between = max((ms_between - ms_within) / n0, 0.0)
    var_within = ms_within
    total = var_between + var_within
    icc = var_between / total if total > 0 else np.nan
    return icc, var_between, var_within


def permutation_null(real_means, gen_means, n_perm=1000, seed=0):

    real = np.asarray(real_means, dtype=float)
    gen = np.asarray(gen_means, dtype=float)
    good = np.isfinite(real) & np.isfinite(gen)
    real, gen = real[good], gen[good]
    if len(real) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(real, rng.permutation(gen))[0, 1]
                     for _ in range(n_perm)])
    return float(np.nanmean(np.abs(null))), float(np.nanpercentile(np.abs(null), 95))


def safe_corr(a, b, method='pearson'):
    s = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(s) < 3 or s['a'].nunique() < 2 or s['b'].nunique() < 2:
        return np.nan
    return float(s['a'].corr(s['b'], method=method))


def save_pair_figure(name, genotype, real_img, real_masks, real_traits,
                     gen_img, gen_masks, gen_traits, save_path):
    fig = plt.figure(figsize=(13, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.06)

    for col, (img, masks, title) in enumerate((
            (real_img, real_masks, 'Real image'),
            (gen_img, gen_masks, 'Generated from SNPs only'))):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(overlay(img, masks))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    ax = fig.add_subplot(grid[0, 2])
    ax.axis('off')

    header = f"{'Trait':<19}{'Real':>9}{'Gen':>10}{'Diff':>9}{'%':>8}"
    lines = [header]
    for key, label, fmt in TRAITS:
        r, g = real_traits[key], gen_traits[key]
        if not (np.isfinite(r) and np.isfinite(g)):
            lines.append(f'{label:<19}{"n/a":>9}{"n/a":>10}{"":>9}{"":>8}')
            continue
        diff = g - r
        pct = (diff / r * 100) if r else np.nan
        lines.append(f'{label:<19}{fmt.format(r):>9}{fmt.format(g):>10}'
                     f'{diff:>+9.1f}{pct:>+8.1f}')

    ax.text(0.0, 0.97, '\n'.join(lines), family='monospace', fontsize=9.5,
            va='top', ha='left', transform=ax.transAxes)
    ax.text(0.0, 0.34, 'red = root   green = stele   blue = vessel',
            fontsize=8.5, color='0.35', va='top', transform=ax.transAxes)
    ax.text(0.0, 0.02,
            'The generated image is a sample for this genotype,\n'
            'not a reconstruction of this specific photo, so\n'
            'per-image difference is expected to be nonzero.\n'
            'Genotype-level tracking is the real test.',
            fontsize=8.5, color='0.35', va='bottom', transform=ax.transAxes)

    fig.suptitle(f'{name}   genotype {genotype}', fontsize=10, y=0.99)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_genotype_scatter(geno_df, stats, save_path):
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(4.0 * len(TRAITS), 4.2))
    for ax, (key, label, _) in zip(np.atleast_1d(axes), TRAITS):
        real = geno_df[f'real_{key}'].to_numpy(dtype=float)
        gen = geno_df[f'gen_{key}'].to_numpy(dtype=float)
        good = np.isfinite(real) & np.isfinite(gen)
        ax.scatter(real[good], gen[good], s=26, alpha=0.75, edgecolor='none')

        if good.sum():
            lo = float(min(real[good].min(), gen[good].min()))
            hi = float(max(real[good].max(), gen[good].max()))
            pad = 0.05 * (hi - lo or 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls='--',
                    color='crimson', lw=1.2, label='y = x')
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.legend(fontsize=8)

        s = stats.get(key, {})
        ax.set_title(f"{label}\nr = {s.get('genotype_pearson', float('nan')):.2f}   "
                     f"ICC = {s.get('repeatability_real', float('nan')):.2f}", fontsize=10)
        ax.set_xlabel('real genotype mean')
        ax.set_ylabel('generated genotype mean')
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_summary_figure(stats, save_path):

    labels = [label for key, label, _ in TRAITS]
    corr = [abs(stats.get(k, {}).get('genotype_pearson', np.nan)) for k, _, _ in TRAITS]
    ceiling = [stats.get(k, {}).get('repeatability_real', np.nan) for k, _, _ in TRAITS]
    null95 = [stats.get(k, {}).get('permutation_null_p95', np.nan) for k, _, _ in TRAITS]
    varratio = [stats.get(k, {}).get('variance_ratio', np.nan) for k, _, _ in TRAITS]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(labels))
    w = 0.27

    ax = axes[0]
    ax.bar(x - w, corr, w, label='|r| real vs generated', color='steelblue')
    ax.bar(x, ceiling, w, label='ICC ceiling (real data)', color='seagreen')
    ax.bar(x + w, null95, w, label='chance (95th pct)', color='0.7')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('correlation')
    ax.set_title('Genotype-level tracking vs. ceiling and chance')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, varratio, 0.5, color='darkorange')
    ax.axhline(1.0, color='crimson', ls='--', lw=1.2, label='matches real spread')
    ax.axhline(0.0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('std(generated) / std(real)')
    ax.set_title('Between-genotype spread\n(near 0 = conditioning ignored)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():

    class cfg:
        # find_latest_checkpoint picks the highest numbered checkpoint here.
        checkpoint_dir = DIFFUSION_ONEHOT_DIR
        litevae_checkpoint = LITEVAE_MODEL
        seg_weights = SEGMENTATION_MODEL
        snp_parquet = SNP_PARQUET
        metadata_path = IMAGE_METADATA
        image_dir = CROPPED_IMAGES_DIR
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'

        output_dir = RESULTS_DIR / 'genetic_fidelity'

        # None -> every image whose genotype has SNP data
        max_images = None
        # Per-image comparison figures are only written for this many images;
        # the dataset-level statistics always use everything measured.
        max_pair_figures = 60

        reuse_cached_generations = True

        batch_size = 8
        sampling_steps = 50
        seed = 0
        imgsz = 256
        latent_size = 32
        conf = 0.25
        min_vessel_px = 4
        connectivity = 2
        n_permutations = 1000
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    (out / 'comparisons').mkdir(parents=True, exist_ok=True)
    gen_cache = out / 'generated'
    gen_cache.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    # Data
    image_dir = resolve_input(cfg.image_dir, 'image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))
    sample_names, _, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    name_to_row = {n: i for i, n in enumerate(sample_names)}

    rows, n_no_snp, n_no_file = [], 0, 0
    for idx, row in metadata.iterrows():
        if row['genotype'] not in name_to_row:
            n_no_snp += 1
            continue
        path = image_dir / row['new_filename']
        if not path.exists():
            n_no_file += 1
            continue
        rows.append({'row_index': idx, 'genotype': row['genotype'],
                     'filename': row['new_filename'], 'image_path': path})

    print(f"\nMetadata rows: {len(metadata)}")
    print(f"  skipped (no SNP data): {n_no_snp}   (image missing): {n_no_file}")
    print(f"  eligible: {len(rows)}")
    if cfg.max_images is not None:
        rows = rows[:cfg.max_images]
        print(f"  limited to: {len(rows)}")

    # Models
    seg_path = resolve_input(cfg.seg_weights, 'segmentation weights')
    seg = YOLO(str(seg_path))
    print(f"Segmentation weights: {seg_path}")

    checkpoint_path = find_latest_checkpoint(
        resolve_input(cfg.checkpoint_dir, 'checkpoint directory'))
    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    from latent_diffusion.diffusion.scheduler import DiffusionScheduler
    from litevae.models import LiteVAEDecoder

    scheduler = DiffusionScheduler()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    vae_ckpt = torch.load(resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                          map_location=device, weights_only=False)
    decoder = LiteVAEDecoder(latent_channels=unet_cfg['latent_channels'],
                             output_channels=3, base_channels=512, num_res_blocks=2)
    decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
    decoder.to(device).eval()

    # Generate and measure
    records, n_figures = [], 0
    for start in range(0, len(rows), cfg.batch_size):
        batch = rows[start:start + cfg.batch_size]

        cached = []
        for r in batch:
            p = gen_cache / f"{Path(r['filename']).stem}.png"
            cached.append(np.array(Image.open(p).convert('RGB'))
                          if (cfg.reuse_cached_generations and p.exists()) else None)

        todo = [i for i, c in enumerate(cached) if c is None]
        if todo:
            snp_batch = torch.tensor(
                np.stack([snp_matrix[name_to_row[batch[i]['genotype']]] for i in todo]),
                dtype=torch.float32, device=device)
            seeds = [cfg.seed + batch[i]['row_index'] for i in todo]
            produced = generate_batch(snp_encoder, unet, scheduler, decoder,
                                      snp_batch, seeds, device, latent_shape,
                                      cfg.sampling_steps)
            for slot, img in zip(todo, produced):
                cached[slot] = img
                Image.fromarray(img).save(
                    gen_cache / f"{Path(batch[slot]['filename']).stem}.png")

        for r, gen_img in zip(batch, cached):
            real_img = load_original(r['image_path'], cfg.imgsz)

            res_real = seg.predict(real_img[:, :, ::-1], conf=cfg.conf,
                                   imgsz=cfg.imgsz, device=cfg.device, verbose=False)[0]
            res_gen = seg.predict(gen_img[:, :, ::-1], conf=cfg.conf,
                                  imgsz=cfg.imgsz, device=cfg.device, verbose=False)[0]

            t_real, m_real = measure(res_real, cfg.imgsz, cfg.min_vessel_px, cfg.connectivity)
            t_gen, m_gen = measure(res_gen, cfg.imgsz, cfg.min_vessel_px, cfg.connectivity)

            if n_figures < cfg.max_pair_figures:
                save_pair_figure(Path(r['filename']).stem, r['genotype'],
                                 real_img, m_real, t_real, gen_img, m_gen, t_gen,
                                 out / 'comparisons' / f"{Path(r['filename']).stem}.png")
                n_figures += 1

            rec = {'filename': r['filename'], 'genotype': r['genotype']}
            rec.update({f'real_{k}': v for k, v in t_real.items()})
            rec.update({f'gen_{k}': v for k, v in t_gen.items()})
            records.append(rec)

        done = start + len(batch)
        if done % (cfg.batch_size * 5) == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}")

    df = pd.DataFrame(records)
    df.to_csv(out / 'per_image_measurements.csv', index=False)

    # Dataset-level analysis
    geno = df.groupby('genotype').agg(
        n_images=('filename', 'size'),
        **{f'{side}_{k}': (f'{side}_{k}', 'mean')
           for k, _, _ in TRAITS for side in ('real', 'gen')}).reset_index()
    geno.to_csv(out / 'genotype_means.csv', index=False)

    stats = {}
    for key, label, _ in TRAITS:
        icc, var_b, var_w = repeatability(df[f'real_{key}'], df['genotype'])
        null_mean, null_p95 = permutation_null(
            geno[f'real_{key}'], geno[f'gen_{key}'], cfg.n_permutations, cfg.seed)

        real_std = float(np.nanstd(geno[f'real_{key}']))
        gen_std = float(np.nanstd(geno[f'gen_{key}']))
        real_all = df[f'real_{key}'].to_numpy(dtype=float)
        gen_all = df[f'gen_{key}'].to_numpy(dtype=float)
        good = np.isfinite(real_all) & np.isfinite(gen_all) & (real_all != 0)

        detect_real = float(np.isfinite(real_all).mean())
        detect_gen = float(np.isfinite(gen_all).mean())

        def _nanmean(a):
            return float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')

        stats[key] = {
            'detection_rate_real': detect_real,
            'detection_rate_generated': detect_gen,
            'genotype_pearson': safe_corr(geno[f'real_{key}'], geno[f'gen_{key}']),
            'genotype_spearman': safe_corr(geno[f'real_{key}'], geno[f'gen_{key}'], 'spearman'),
            'repeatability_real': icc,
            'between_genotype_var_real': var_b,
            'within_genotype_var_real': var_w,
            'permutation_null_mean': null_mean,
            'permutation_null_p95': null_p95,
            'variance_ratio': (gen_std / real_std) if real_std > 0 else np.nan,
            'real_mean': _nanmean(real_all),
            'gen_mean': _nanmean(gen_all),
            'mean_signed_pct_error': float(np.mean(
                (gen_all[good] - real_all[good]) / real_all[good] * 100)) if good.any() else np.nan,
            'per_image_pearson': safe_corr(df[f'real_{key}'], df[f'gen_{key}']),
        }

    pd.DataFrame(stats).T.to_csv(out / 'trait_statistics.csv')
    save_genotype_scatter(geno, stats, out / 'genotype_scatter.png')
    save_summary_figure(stats, out / 'summary.png')

    with open(out / 'summary.json', 'w') as f:
        json.dump({'checkpoint': str(checkpoint_path),
                   'seg_weights': str(seg_path),
                   'n_images': len(df), 'n_genotypes': len(geno),
                   'skipped_no_snp': n_no_snp, 'skipped_no_file': n_no_file,
                   'traits': stats}, f, indent=2, default=float)

    # Report
    print(f"\nDetection rate (fraction of images where the trait was measurable)")
    print(f"{'trait':<20}{'real':>8}{'generated':>12}")
    for key, label, _ in TRAITS:
        s = stats[key]
        flag = '   <-- generated images often lack this structure' \
            if s['detection_rate_generated'] < 0.8 else ''
        print(f"{label:<20}{s['detection_rate_real']:>8.2f}"
              f"{s['detection_rate_generated']:>12.2f}{flag}")

    print(f"\nGenotype-level tracking ({len(geno)} genotypes, {len(df)} images)")
    print(f"{'trait':<20}{'r':>7}{'rho':>7}{'ICC':>7}{'chance':>8}{'var ratio':>11}{'bias %':>9}")
    for key, label, _ in TRAITS:
        s = stats[key]
        print(f"{label:<20}{s['genotype_pearson']:>7.2f}{s['genotype_spearman']:>7.2f}"
              f"{s['repeatability_real']:>7.2f}{s['permutation_null_p95']:>8.2f}"
              f"{s['variance_ratio']:>11.2f}{s['mean_signed_pct_error']:>+9.1f}")

    print("\nr / rho  correlation of genotype means, real vs generated")
    print("ICC      share of real variance that is between genotypes - the ceiling;")
    print("         a trait with low ICC has little genetic signal here to predict")
    print("chance   95th percentile of |r| under shuffled genotype labels")
    print("var ratio  spread of generated genotype means / spread of real;")
    print("         near 0 means the model is ignoring its SNP conditioning")
    print(f"\nWrote {n_figures} comparison figures, per_image_measurements.csv, "
          f"genotype_means.csv, trait_statistics.csv, genotype_scatter.png, "
          f"summary.png to {out}")


if __name__ == '__main__':
    main()
