# Do traits survive LiteVAE's encode and decode?
#
# This is the ceiling on everything downstream: a trait the autoencoder
# cannot preserve cannot be predicted from genotype either, however good the
# diffusion model is.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import CROPPED_IMAGES_DIR, LITEVAE_MODEL, RESULTS_DIR, SEGMENTATION_MODEL

from litevae.models import LiteVAEEncoder, LiteVAEDecoder
from ultralytics import YOLO

from feature_segmentation.evaluation.latent_average_test import (
    load_litevae, load_image, to_uint8, encode,
)
from feature_segmentation.vessel_counting import count_vessels

CLASS_COLORS = {0: (255, 60, 60), 1: (60, 255, 60), 2: (60, 160, 255)}


# Measurement

def class_masks(result, imgsz):
    # Returns {class_id: [mask, ...]} upsampled to the image grid.
    #
    # Ultralytics returns masks at the network's mask resolution, which need not
    # match the input size, so everything is resampled to imgsz first and areas
    # are counted in image pixels.
    out = {0: [], 1: [], 2: []}
    if result.masks is None or len(result.masks.data) == 0:
        return out, {}

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    conf_by_class = {}
    for m, c, conf in zip(masks, classes, confs):
        mt = torch.from_numpy(m)[None, None].float()
        mt = F.interpolate(mt, size=(imgsz, imgsz), mode='bilinear', align_corners=False)
        out[int(c)].append(mt[0, 0].numpy() > 0.5)
        conf_by_class.setdefault(int(c), []).append(float(conf))
    return out, conf_by_class


def measure(result, imgsz=256, min_vessel_px=4, connectivity=2,
            vessel_method='watershed', watershed_min_distance=6):
    # Extracts the trait measurements from one segmentation result.
    #
    # vessel_method defaults to watershed because plain connected components was
    # validated against the 21 hand-annotated images and undercounts badly: MAE
    # 6.5 vessels with a bias of -6.5 against a true mean of 16.4, because
    # adjacent vessels merge in the predicted mask at 256x256. Watershed with
    # min_distance=6 scored MAE 0.7 on the same images. See
    # validate_vessel_counting.py, which re-runs that comparison.
    #
    # 'connected_components' is still available for reproducing older results.
    masks, confs = class_masks(result, imgsz)

    traits = {
        'root_area_px': np.nan, 'root_diameter_px': np.nan,
        'stele_area_px': np.nan, 'stele_diameter_px': np.nan,
        'vessel_total_area_px': 0.0,
        'vessel_count_cc': 0, 'vessel_count_instances': len(masks[2]),
        'stele_root_diameter_ratio': np.nan,
    }

    # One root and one stele exist per image; if several are proposed, the most
    # confident is the intended one.
    for cls_id, key in ((0, 'root'), (1, 'stele')):
        if masks[cls_id]:
            best = int(np.argmax(confs[cls_id]))
            area = float(masks[cls_id][best].sum())
            traits[f'{key}_area_px'] = area
            # Equivalent-circle diameter: the diameter a circle of equal area
            # would have. More stable than a bounding-box side on cross-sections
            # that are round but not perfectly circular.
            traits[f'{key}_diameter_px'] = 2.0 * np.sqrt(area / np.pi)

    if masks[2]:
        # Merge every vessel detection into one binary map, then split it back
        # into individual vessels. The merge is necessary because overlapping
        # detections would otherwise double-count the same pocket; the split is
        # necessary because touching vessels merge into one blob.
        union = np.zeros((imgsz, imgsz), dtype=bool)
        for m in masks[2]:
            union |= m

        _, count, areas = count_vessels(
            union, method=vessel_method, min_area=min_vessel_px,
            connectivity=connectivity, min_distance=watershed_min_distance)
        traits['vessel_count_cc'] = int(count)
        traits['vessel_total_area_px'] = float(sum(areas))

    if traits['root_diameter_px'] > 0:
        traits['stele_root_diameter_ratio'] = (traits['stele_diameter_px'] /
                                               traits['root_diameter_px'])
    return traits, masks


def overlay(image_rgb, masks, alpha=0.45):
    canvas = image_rgb.astype(np.float32).copy()
    for cls_id in (0, 1, 2):
        for m in masks.get(cls_id, []):
            canvas[m] = (1 - alpha) * canvas[m] + alpha * np.array(CLASS_COLORS[cls_id], np.float32)
    return canvas.astype(np.uint8)


# Per-image figure

TRAIT_ROWS = [
    ('root_diameter_px', 'Root diameter', 'px', '{:.1f}'),
    ('stele_diameter_px', 'Stele diameter', 'px', '{:.1f}'),
    ('vessel_total_area_px', 'Vessel area', 'px', '{:.0f}'),
    ('vessel_count_cc', 'Vessel count', '', '{:.0f}'),
    ('stele_root_diameter_ratio', 'Stele/root diam.', '', '{:.3f}'),
]


def save_comparison(name, orig_img, orig_masks, orig_traits,
                    recon_img, recon_masks, recon_traits, save_path):
    fig = plt.figure(figsize=(13, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.06)

    for col, (img, masks, title) in enumerate((
            (orig_img, orig_masks, 'Original'),
            (recon_img, recon_masks, 'LiteVAE reconstruction'))):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(overlay(img, masks))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    ax = fig.add_subplot(grid[0, 2])
    ax.axis('off')

    header = f"{'Trait':<19}{'Orig':>9}{'Recon':>10}{'Diff':>9}{'%':>8}"
    lines = [header]
    for key, label, unit, fmt in TRAIT_ROWS:
        o, r = orig_traits[key], recon_traits[key]
        if not (np.isfinite(o) and np.isfinite(r)):
            lines.append(f'{label:<19}{"n/a":>9}{"n/a":>10}{"":>9}{"":>8}')
            continue
        diff = r - o
        pct = (diff / o * 100) if o else np.nan
        lines.append(f'{label:<19}{fmt.format(o):>9}{fmt.format(r):>10}'
                     f'{diff:>+9.1f}{pct:>+8.1f}')

    lines.append('')
    lines.append(f"vessel instances   {orig_traits['vessel_count_instances']:>9}"
                 f"{recon_traits['vessel_count_instances']:>10}")
    lines.append('(detector proposals, vs. connected')
    lines.append(' components above)')

    ax.text(0.0, 0.97, '\n'.join(lines), family='monospace', fontsize=9.5,
            va='top', ha='left', transform=ax.transAxes)
    ax.text(0.0, 0.03,
            'red = root   green = stele   blue = vessel',
            fontsize=8.5, color='0.35', va='bottom', transform=ax.transAxes)

    fig.suptitle(name, fontsize=10, y=0.99)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_agreement_figure(df, save_path):
    # Original vs reconstruction per trait, against the y = x line.
    traits = [t for t in TRAIT_ROWS if t[0] != 'stele_root_diameter_ratio']
    fig, axes = plt.subplots(1, len(traits), figsize=(4.0 * len(traits), 4.0))

    for ax, (key, label, unit, _) in zip(np.atleast_1d(axes), traits):
        o = df[f'orig_{key}'].to_numpy(dtype=float)
        r = df[f'recon_{key}'].to_numpy(dtype=float)
        good = np.isfinite(o) & np.isfinite(r)
        o, r = o[good], r[good]

        ax.scatter(o, r, s=34, alpha=0.8, edgecolor='none')
        if len(o):
            lo = float(min(o.min(), r.min()))
            hi = float(max(o.max(), r.max()))
            pad = 0.05 * (hi - lo or 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color='crimson', lw=1.2, ls='--', label='y = x')
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            bias = np.mean((r - o) / np.where(o == 0, np.nan, o)) * 100
            ax.set_title(f'{label}\nmean bias {bias:+.1f}%', fontsize=10)
        ax.set_xlabel(f'original {unit}'.strip())
        ax.set_ylabel(f'reconstruction {unit}'.strip())
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/reconstruction_fidelity_test.py
    class cfg:
        source_images = CROPPED_IMAGES_DIR
        litevae_checkpoint = LITEVAE_MODEL
        seg_weights = SEGMENTATION_MODEL
        output_dir = RESULTS_DIR / 'reconstruction_fidelity'

        n_images = 24
        seed = 0

        imgsz = 256
        conf = 0.25            # segmentation confidence threshold
        min_vessel_px = 4      # drop connected components smaller than this
        connectivity = 2       # 2 = 8-connectivity, 1 = 4-connectivity.
                               # 8 merges diagonally touching pockets, 4 keeps
                               # them separate; 8 is the usual default.
        batch_size = 8
        device = 'cpu'

    device = torch.device(cfg.device)
    out = Path(cfg.output_dir)
    (out / 'comparisons').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    for path, label in ((cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                        (cfg.seg_weights, 'segmentation weights'),
                        (cfg.source_images, 'source images')):
        if not Path(path).exists():
            raise SystemExit(f"missing {label}: {path}")

    encoder, decoder = load_litevae(cfg.litevae_checkpoint, device)
    seg = YOLO(str(cfg.seg_weights))

    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = sorted(p for p in Path(cfg.source_images).iterdir() if p.suffix in exts)
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.n_images, len(all_images))
    sources = [all_images[i] for i in rng.choice(len(all_images), size=n, replace=False)]
    print(f"Comparing {len(sources)} images from {Path(cfg.source_images).name}")

    rows = []
    for start in range(0, len(sources), cfg.batch_size):
        chunk = sources[start:start + cfg.batch_size]
        batch = torch.stack([load_image(p, cfg.imgsz) for p in chunk]).to(device)

        latents = encode(encoder, batch)
        with torch.no_grad():
            recon = decoder(latents, save_steps=False)

        for i, path in enumerate(chunk):
            orig_img = to_uint8(batch[i])
            recon_img = to_uint8(recon[i])

            # Ultralytics expects BGR for ndarray input.
            r_orig = seg.predict(orig_img[:, :, ::-1], conf=cfg.conf, imgsz=cfg.imgsz,
                                 device=cfg.device, verbose=False)[0]
            r_recon = seg.predict(recon_img[:, :, ::-1], conf=cfg.conf, imgsz=cfg.imgsz,
                                  device=cfg.device, verbose=False)[0]

            t_orig, m_orig = measure(r_orig, cfg.imgsz, cfg.min_vessel_px, cfg.connectivity)
            t_recon, m_recon = measure(r_recon, cfg.imgsz, cfg.min_vessel_px, cfg.connectivity)

            save_comparison(path.stem, orig_img, m_orig, t_orig,
                            recon_img, m_recon, t_recon,
                            out / 'comparisons' / f'{path.stem}.png')

            row = {'image': path.stem}
            row.update({f'orig_{k}': v for k, v in t_orig.items()})
            row.update({f'recon_{k}': v for k, v in t_recon.items()})
            rows.append(row)

        print(f"  {min(start + cfg.batch_size, len(sources))}/{len(sources)}")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'measurements.csv', index=False)
    save_agreement_figure(df, out / 'agreement_summary.png')

    # summary
    print("\n=== Reconstruction fidelity (paired, per image) ===")
    print(f"{'trait':<26}{'orig mean':>11}{'recon mean':>12}"
          f"{'signed %':>10}{'abs %':>8}{'exact':>8}")
    summary = {}
    for key, label, unit, _ in TRAIT_ROWS:
        o = df[f'orig_{key}'].to_numpy(dtype=float)
        r = df[f'recon_{key}'].to_numpy(dtype=float)
        good = np.isfinite(o) & np.isfinite(r) & (o != 0)
        if not good.any():
            continue
        pct = (r[good] - o[good]) / o[good] * 100
        exact = float(np.mean(np.isclose(o[good], r[good]))) * 100
        summary[key] = {
            'orig_mean': float(np.nanmean(o[good])),
            'recon_mean': float(np.nanmean(r[good])),
            'signed_pct': float(pct.mean()),
            'abs_pct': float(np.abs(pct).mean()),
            'exact_match_pct': exact,
        }
        print(f'{label:<26}{np.nanmean(o[good]):>11.1f}{np.nanmean(r[good]):>12.1f}'
              f'{pct.mean():>+10.1f}{np.abs(pct).mean():>8.1f}{exact:>7.0f}%')

    with open(out / 'summary.json', 'w') as f:
        json.dump({'n_images': len(df), 'connectivity': cfg.connectivity,
                   'source': str(cfg.source_images), 'traits': summary}, f, indent=2)

    print(f"\nWrote {len(df)} comparison figures, measurements.csv, "
          f"agreement_summary.png and summary.json to {out}")


if __name__ == '__main__':
    main()
