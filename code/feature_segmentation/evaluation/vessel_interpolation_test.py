# Does LiteVAE's latent space actually encode vessel count as a smooth axis?
#
# Picks pairs of real roots with very different vessel counts (say 4 and 12),
# blends their latents at a range of weights, decodes each blend, and counts
# the vessels that come out. If vessel number is a real, continuously encoded
# property of the latent space, the midpoint should decode to a root with an
# intermediate number of vessels - roughly 8 for a 4-vs-12 pair - and the count
# should move monotonically from one endpoint to the other.
#
# If instead the count jumps abruptly, stays pinned at one endpoint, or bounces
# around, then vessel number is not something the latent represents
# continuously, and no amount of conditioning the diffusion model on genetics
# will let it control that trait: the space it generates into cannot express
# it.
#
# What the interpolation is measured against
# The expected line runs between the counts of the two RECONSTRUCTIONS (the
# alpha = 0 and alpha = 1 decodes), not the counts of the two original photos.
# LiteVAE already miscounts vessels somewhat on a plain round trip, and that error
# belongs to the autoencoder, not to the blending. Anchoring on the originals
# would charge reconstruction loss to the interpolation and make a working latent
# space look broken. Both are recorded so the two effects stay separable.
#
# The blur control
# Averaging in latent space produces visibly softer decodes than the endpoints do.
# That is a problem for this specific measurement, because blur can merge adjacent
# vessels into one connected component or push faint ones below the detector's
# threshold - which would look exactly like "the latent lost the vessels" while
# really being a decode-sharpness artifact. Image sharpness is therefore tracked
# across the sweep alongside the counts, so a dip in the middle can be attributed
# correctly. A count dip that coincides with a sharpness dip is suspect; one
# without it is a real property of the latent space.
#
# Usage
# Edit the CONFIG block in main(), then run:
#
# python code/feature_segmentation/evaluation/vessel_interpolation_test.py

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
from scipy import ndimage

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    CROPPED_IMAGES_DIR, LITEVAE_MODEL, RESULTS_DIR, SEGMENTATION_MODEL,
    resolve_input, resolve_output,
)

from ultralytics import YOLO

from feature_segmentation.evaluation.latent_average_test import (
    load_litevae, load_image, to_uint8, encode,
)
from feature_segmentation.evaluation.reconstruction_fidelity_test import (
    measure, overlay,
)


def sharpness(image_rgb):
    # Variance of the Laplacian - the standard cheap focus measure.
    #
    # Used only as a control: it separates 'the latent stopped encoding vessels'
    # from 'the decode got soft enough that the segmenter merged them'.
    gray = np.asarray(image_rgb, dtype=np.float64).mean(axis=2)
    return float(ndimage.laplace(gray).var())


def spearman(x, y):
    s = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(s) < 3 or s['x'].nunique() < 2 or s['y'].nunique() < 2:
        return np.nan
    return float(s['x'].corr(s['y'], method='spearman'))


# Pair selection

def survey_pool(seg, paths, imgsz, conf, min_vessel_px, connectivity, device, batch_log=25):
    # Segments a pool of real images and records each one's vessel count.
    rows = []
    for i, path in enumerate(paths):
        img = np.array(Image.open(path).convert('RGB').resize((imgsz, imgsz), Image.LANCZOS))
        result = seg.predict(img[:, :, ::-1], conf=conf, imgsz=imgsz,
                             device=device, verbose=False)[0]
        traits, _ = measure(result, imgsz, min_vessel_px, connectivity)
        rows.append({'path': str(path), 'stem': Path(path).stem,
                     'vessel_count': traits['vessel_count_cc'],
                     'vessel_area': traits['vessel_total_area_px'],
                     'root_diameter': traits['root_diameter_px']})
        if (i + 1) % batch_log == 0:
            print(f"    surveyed {i + 1}/{len(paths)}")
    return pd.DataFrame(rows)


def choose_pairs(survey, n_pairs, min_gap):
    # Pairs low-vessel images with high-vessel ones, largest contrast first.
    #
    # Sorting by count and pairing the extremes from the outside in gives the
    # widest gaps available, which is what makes an intermediate result
    # unambiguous - a 4-vs-6 pair cannot distinguish a working latent axis from
    # noise, while 4-vs-12 can.
    ranked = survey.dropna(subset=['vessel_count']).sort_values('vessel_count')
    ranked = ranked[ranked['vessel_count'] > 0].reset_index(drop=True)
    if len(ranked) < 2:
        raise SystemExit("not enough images with detected vessels to form pairs")

    pairs = []
    lo, hi = 0, len(ranked) - 1
    while lo < hi and len(pairs) < n_pairs:
        low, high = ranked.iloc[lo], ranked.iloc[hi]
        gap = high['vessel_count'] - low['vessel_count']
        if gap >= min_gap:
            pairs.append({'low': low, 'high': high, 'gap': gap})
            lo += 1
            hi -= 1
        else:
            # Remaining candidates are too similar to be informative.
            break
    return pairs


# Figures

def save_pair_figure(pair_id, low_stem, high_stem, alphas, images, masks_list,
                     counts, expected, save_path):
    n = len(alphas)
    fig = plt.figure(figsize=(2.3 * n, 8.4))
    grid = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.95], hspace=0.22, wspace=0.05)

    for i, (a, img, masks, c) in enumerate(zip(alphas, images, masks_list, counts)):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(img)
        ax.set_title(f'a = {a:.2f}', fontsize=10)
        ax.axis('off')

        ax = fig.add_subplot(grid[1, i])
        ax.imshow(overlay(img, masks))
        ax.set_xlabel(f'{c} vessels', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(grid[2, :])
    ax.plot(alphas, counts, 'o-', color='steelblue', lw=2, label='observed')
    ax.plot(alphas, expected, '--', color='crimson', lw=1.5,
            label='linear between the two reconstructions')
    ax.set_xlabel('blend weight a   (0 = low-vessel root, 1 = high-vessel root)')
    ax.set_ylabel('vessel count')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(f'pair {pair_id}:  {low_stem}  ->  {high_stem}', fontsize=11, y=0.98)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_sweep_grid(sweeps, alphas, save_path, max_rows=10):
    # Every pair's sweep stacked into one grid of segmentation overlays.
    #
    # Rows are pairs, columns are blend weights. Seeing the sweeps together is
    # what makes the shared shape obvious - the per-pair plots each look like
    # their own story, but stacked it is clear whether the counts ramp smoothly
    # or sit flat and then jump at the same place in every pair.
    sweeps = sweeps[:max_rows]
    n_rows, n_cols = len(sweeps), len(alphas)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.55 * n_cols, 1.75 * n_rows),
                             squeeze=False)

    for r, sweep in enumerate(sweeps):
        for c, a in enumerate(alphas):
            ax = axes[r][c]
            ax.imshow(overlay(sweep['images'][c], sweep['masks'][c]))
            ax.set_xticks([]); ax.set_yticks([])

            count = sweep['counts'][c]
            lo, hi = sweep['counts'][0], sweep['counts'][-1]
            # Flag decodes that leave the range spanned by the two endpoints -
            # those are not intermediates, they are invented structure.
            outside = count > max(lo, hi) or count < min(lo, hi)
            ax.text(0.5, -0.02, f'{count}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=10,
                    color='crimson' if outside else '0.15',
                    fontweight='bold' if outside else 'normal')

            if r == 0:
                ax.set_title(f'a={a:g}', fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{sweep['low_stem'][:11]}\n{int(lo)} -> {int(hi)}",
                              fontsize=8, rotation=0, ha='right', va='center',
                              labelpad=42)

    fig.suptitle('Vessel-count sweeps across latent blends\n'
                 'rows = image pairs, columns = blend weight, '
                 'number = vessels detected (red = outside the endpoint range)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_summary_figure(df, stats, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for pair_id, sub in df.groupby('pair'):
        sub = sub.sort_values('alpha')
        lo, hi = sub['count'].iloc[0], sub['count'].iloc[-1]
        span = hi - lo
        if span == 0:
            continue
        # Rescale each pair onto 0..1 so trajectories with different endpoint
        # counts can be overlaid and compared against one diagonal.
        ax.plot(sub['alpha'], (sub['count'] - lo) / span, '-', alpha=0.45,
                color='steelblue')
    ax.plot([0, 1], [0, 1], '--', color='crimson', lw=1.8, label='ideal linear')
    ax.set_xlabel('blend weight a')
    ax.set_ylabel('vessel count, rescaled per pair')
    ax.set_title('All interpolation trajectories')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(df['count_minus_expected'].dropna(), bins=25, color='darkorange', alpha=0.85)
    ax.axvline(0, color='black', lw=1.0)
    ax.set_xlabel('observed count - linear expectation')
    ax.set_ylabel('decoded images')
    ax.set_title(f"Deviation from linear\nmean abs = "
                 f"{stats['mean_abs_deviation']:.2f} vessels")

    ax = axes[2]
    by_alpha = df.groupby('alpha').agg(sharp=('sharpness', 'mean'),
                                       cnt=('count', 'mean'))
    ax.plot(by_alpha.index, by_alpha['sharp'], 'o-', color='seagreen')
    ax.set_xlabel('blend weight a')
    ax.set_ylabel('image sharpness (Laplacian variance)', color='seagreen')
    ax.tick_params(axis='y', labelcolor='seagreen')
    ax2 = ax.twinx()
    ax2.plot(by_alpha.index, by_alpha['cnt'], 's--', color='steelblue')
    ax2.set_ylabel('mean vessel count', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax.set_title('Blur control\n(a mid-sweep count dip that tracks sharpness\n'
                 'is a decode artifact, not the latent space)')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/vessel_interpolation_test.py
    class cfg:
        source_images = CROPPED_IMAGES_DIR
        litevae_checkpoint = LITEVAE_MODEL
        seg_weights = SEGMENTATION_MODEL
        output_dir = RESULTS_DIR / 'vessel_interpolation'

        # Images segmented up front to find high- and low-vessel examples.
        # Larger pool -> more extreme pairs available, at survey cost.
        survey_size = 150
        n_pairs = 10
        # Minimum vessel-count difference for a pair to be worth testing. A
        # narrow gap cannot distinguish a real latent axis from measurement
        # noise.
        min_gap = 5

        alphas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        max_pair_figures = 10

        imgsz = 256
        conf = 0.25
        min_vessel_px = 4
        connectivity = 2
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    (out / 'pairs').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    seg = YOLO(str(resolve_input(cfg.seg_weights, 'segmentation weights')))
    encoder, decoder = load_litevae(
        resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'), device)

    src = Path(resolve_input(cfg.source_images, 'source images'))
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_paths = sorted(p for p in src.iterdir() if p.suffix in exts)
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.survey_size, len(all_paths))
    pool = [all_paths[i] for i in rng.choice(len(all_paths), size=n, replace=False)]

    print(f"\nSurveying {len(pool)} images for vessel counts...")
    survey = survey_pool(seg, pool, cfg.imgsz, cfg.conf, cfg.min_vessel_px,
                         cfg.connectivity, cfg.device)
    survey.to_csv(out / 'survey.csv', index=False)
    counts = survey['vessel_count']
    print(f"  vessel count: min={counts.min()} max={counts.max()} "
          f"mean={counts.mean():.1f}")

    pairs = choose_pairs(survey, cfg.n_pairs, cfg.min_gap)
    if not pairs:
        raise SystemExit(
            f"no pairs with a vessel-count gap of at least {cfg.min_gap} were found "
            f"(range in this pool: {counts.min()} to {counts.max()}).\n"
            f"Lower min_gap or raise survey_size.")
    print(f"  selected {len(pairs)} pairs, gaps "
          f"{[int(p['gap']) for p in pairs]}")

    # interpolate
    records, sweeps, n_figures = [], [], 0
    for pair_id, pair in enumerate(pairs):
        low, high = pair['low'], pair['high']

        batch = torch.stack([
            load_image(Path(low['path']), cfg.imgsz),
            load_image(Path(high['path']), cfg.imgsz),
        ]).to(device)
        with torch.no_grad():
            z = encode(encoder, batch)
        z_low, z_high = z[0], z[1]

        blended = torch.stack([(1 - a) * z_low + a * z_high for a in cfg.alphas])
        with torch.no_grad():
            decoded = decoder(blended, save_steps=False)

        images, masks_list, obs_counts, sharps = [], [], [], []
        for i, a in enumerate(cfg.alphas):
            img = to_uint8(decoded[i])
            result = seg.predict(img[:, :, ::-1], conf=cfg.conf, imgsz=cfg.imgsz,
                                 device=cfg.device, verbose=False)[0]
            traits, masks = measure(result, cfg.imgsz, cfg.min_vessel_px, cfg.connectivity)
            images.append(img)
            masks_list.append(masks)
            obs_counts.append(traits['vessel_count_cc'])
            sharps.append(sharpness(img))

        # Anchor the expectation on the reconstructions at the two ends, not on
        # the original photos - see the module docstring.
        recon_low, recon_high = obs_counts[0], obs_counts[-1]
        expected = [(1 - a) * recon_low + a * recon_high for a in cfg.alphas]

        for a, c, e, s in zip(cfg.alphas, obs_counts, expected, sharps):
            records.append({
                'pair': pair_id,
                'low_stem': low['stem'], 'high_stem': high['stem'],
                'orig_low_count': low['vessel_count'],
                'orig_high_count': high['vessel_count'],
                'recon_low_count': recon_low, 'recon_high_count': recon_high,
                'alpha': a, 'count': c, 'expected': e,
                'count_minus_expected': c - e, 'sharpness': s,
            })

        sweeps.append({'low_stem': low['stem'], 'high_stem': high['stem'],
                       'images': images, 'masks': masks_list, 'counts': obs_counts})

        if n_figures < cfg.max_pair_figures:
            save_pair_figure(pair_id, low['stem'], high['stem'], cfg.alphas,
                             images, masks_list, obs_counts, expected,
                             out / 'pairs' / f'pair_{pair_id:02d}.png')
            n_figures += 1

        print(f"  pair {pair_id}: originals {int(low['vessel_count'])} -> "
              f"{int(high['vessel_count'])}, reconstructions {recon_low} -> "
              f"{recon_high}, sweep {obs_counts}")

    df = pd.DataFrame(records)
    df.to_csv(out / 'interpolation.csv', index=False)

    # per-pair behaviour
    pair_rows = []
    for pair_id, sub in df.groupby('pair'):
        sub = sub.sort_values('alpha')
        c = sub['count'].to_numpy(dtype=float)
        recon_low, recon_high = c[0], c[-1]
        lo, hi = min(recon_low, recon_high), max(recon_low, recon_high)
        interior = c[1:-1]

        pair_rows.append({
            'pair': pair_id,
            'recon_low_count': recon_low, 'recon_high_count': recon_high,
            'spearman_alpha_count': spearman(sub['alpha'], sub['count']),
            'mean_abs_deviation': float(np.abs(sub['count_minus_expected']).mean()),
            # Does the sweep actually pass through intermediate values, or does
            # it sit at one end and jump? This is the question the whole test
            # exists to answer.
            'fraction_interior_between_ends': (
                float(np.mean((interior >= lo) & (interior <= hi)))
                if len(interior) else np.nan),
            # How far outside the endpoint range the sweep strays. A blend
            # producing MORE vessels than either parent is not interpolation at
            # all - it means the decoder is inventing structure that neither
            # source root had, which a simple monotonicity check would miss.
            'max_overshoot': (float(max(0.0, interior.max() - hi, lo - interior.min()))
                              if len(interior) else np.nan),
            'midpoint_count': float(sub.loc[np.isclose(sub['alpha'], 0.5), 'count'].mean())
                if np.isclose(sub['alpha'], 0.5).any() else np.nan,
            'midpoint_expected': float((recon_low + recon_high) / 2),
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out / 'per_pair.csv', index=False)

    stats = {
        'n_pairs': len(pair_df),
        'mean_abs_deviation': float(np.abs(df['count_minus_expected']).mean()),
        'mean_spearman_alpha_count': float(pair_df['spearman_alpha_count'].mean()),
        'fraction_pairs_monotonic': float((pair_df['spearman_alpha_count'] > 0.7).mean()),
        'mean_fraction_interior_between_ends': float(
            pair_df['fraction_interior_between_ends'].mean()),
        'mean_max_overshoot': float(pair_df['max_overshoot'].mean()),
        'fraction_pairs_overshooting': float((pair_df['max_overshoot'] > 0).mean()),
        'mean_midpoint_count': float(pair_df['midpoint_count'].mean()),
        'mean_midpoint_expected': float(pair_df['midpoint_expected'].mean()),
        'sharpness_by_alpha': df.groupby('alpha')['sharpness'].mean().to_dict(),
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(stats, f, indent=2, default=float)

    save_summary_figure(df, stats, out / 'summary.png')
    save_sweep_grid(sweeps, cfg.alphas, out / 'sweep_grid.png')

    # report
    print(f"\n=== Vessel-count interpolation ({stats['n_pairs']} pairs) ===")
    print(f"  mean Spearman(alpha, count)        {stats['mean_spearman_alpha_count']:>7.3f}")
    print(f"  pairs moving monotonically (>0.7)  {stats['fraction_pairs_monotonic']:>7.1%}")
    print(f"  mean |observed - linear|           {stats['mean_abs_deviation']:>7.2f} vessels")
    print(f"  interior points between the ends   "
          f"{stats['mean_fraction_interior_between_ends']:>7.1%}")
    print(f"  pairs overshooting the endpoints   "
          f"{stats['fraction_pairs_overshooting']:>7.1%}  "
          f"(mean worst overshoot {stats['mean_max_overshoot']:.1f} vessels)")
    print(f"  midpoint: observed {stats['mean_midpoint_count']:.1f} vs "
          f"expected {stats['mean_midpoint_expected']:.1f}")

    sharp = stats['sharpness_by_alpha']
    end_sharp = (sharp[min(sharp)] + sharp[max(sharp)]) / 2
    mid_key = min(sharp, key=lambda a: abs(a - 0.5))
    print(f"\n  blur control: sharpness {end_sharp:.1f} at the ends vs "
          f"{sharp[mid_key]:.1f} at a={mid_key}")
    if end_sharp > 0 and sharp[mid_key] < 0.7 * end_sharp:
        print("     NOTE: blends are markedly softer than the endpoints. A count")
        print("     dip in the middle may be the segmenter losing blurred vessels")
        print("     rather than the latent space failing to encode them.")

    print("\n  Reading this: high Spearman with interior points between the ends")
    print("  means vessel count is a smooth, continuously encoded latent axis.")
    print("  Near-zero Spearman, or interior points outside the endpoint range,")
    print("  means it is not - and a genetics-conditioned model could not steer")
    print("  that trait no matter how well the conditioning worked.")
    print(f"\nWrote {n_figures} pair figures, sweep_grid.png, survey.csv, "
          f"interpolation.csv, per_pair.csv, summary.png and summary.json to {out}")


if __name__ == '__main__':
    main()
