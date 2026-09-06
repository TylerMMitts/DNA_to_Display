# Which vessel-counting method is actually correct, measured against the hand-
# drawn annotations.
#
# The 21 annotated images carry one polygon per vessel, so their true vessel
# counts are known exactly. That makes it possible to pick a counting method
# and its parameters by measurement rather than by eye - which matters, because
# the tempting knobs here (watershed min_distance, opening radius) trade over-
# segmentation against under-segmentation, and eyeballing a few overlays cannot
# tell you where the balance sits.
#
# Two error sources are separated on purpose:
#
# segmentation error
#     the model's predicted vessel mask differs from the true mask.
#
# counting error
#     the mask is fine but the method splits or merges it wrongly.
#
# Running every method against the ground-truth MASK as well as the predicted
# one tells you which of the two is limiting you. If a method is near-perfect
# on the true mask but poor on the predicted mask, the counter is fine and the
# segmenter is the problem; if it is poor on both, the counter is at fault.
#
# Usage
#     python code/feature_segmentation/evaluation/validate_vessel_counting.py

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import RESULTS_DIR, SEGMENTATION_MODEL, resolve_input, resolve_output

from ultralytics import YOLO

from feature_segmentation.evaluation.reconstruction_fidelity_test import class_masks
from feature_segmentation.vessel_counting import count_vessels

VESSEL_CLASS = 2


def load_ground_truth(label_path, imgsz):
    # Rasterises the annotated vessel polygons.
    #
    # Returns the union mask plus the true count, which is simply the number of
    # polygons - each annotated vessel is one polygon, so this is exact and does
    # not depend on any counting method.
    union = np.zeros((imgsz, imgsz), dtype=bool)
    n = 0
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 7 or int(parts[0]) != VESSEL_CLASS:
            continue
        coords = (np.array([float(v) for v in parts[1:]]).reshape(-1, 2) * imgsz)
        img = Image.new('L', (imgsz, imgsz), 0)
        ImageDraw.Draw(img).polygon(coords.flatten().tolist(), fill=255)
        union |= np.array(img) > 127
        n += 1
    return union, n


def save_method_figure(rows, save_path):
    # Predicted count vs. true count, one panel per method.
    methods = [m for m in rows[0] if m.startswith('pred_')]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.3 * len(methods), 4.3))
    df = pd.DataFrame(rows)

    for ax, key in zip(np.atleast_1d(axes), methods):
        true = df['true_count'].to_numpy(float)
        pred = df[key].to_numpy(float)
        ax.scatter(true, pred, s=40, alpha=0.8, edgecolor='none')
        lo, hi = 0, max(true.max(), pred.max()) * 1.08
        ax.plot([lo, hi], [lo, hi], '--', color='crimson', lw=1.3, label='y = x')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        bias = np.mean(pred - true)
        mae = np.mean(np.abs(pred - true))
        ax.set_title(f"{key.replace('pred_', '')}\nMAE {mae:.1f}   bias {bias:+.1f}",
                     fontsize=10)
        ax.set_xlabel('true count (annotated)')
        ax.set_ylabel('predicted count')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/validate_vessel_counting.py
    class cfg:
        dataset_dir = 'dataset/root_features_256'
        seg_weights = SEGMENTATION_MODEL
        output_dir = RESULTS_DIR / 'vessel_counting_validation'

        imgsz = 256
        conf = 0.25
        min_area = 4

        # Watershed min_distance values to sweep, and opening radii. The point
        # of the sweep is to find where over- and under-segmentation balance,
        # rather than trusting a default.
        watershed_min_distances = [2, 3, 4, 5, 6, 8]
        opening_radii = [1, 2]

        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    ds = Path(resolve_input(cfg.dataset_dir, 'prepared dataset'))
    # Rotated copies are the same roots and would weight those images 4x.
    pairs = []
    for split in ('train', 'val'):
        for label in sorted((ds / split / 'labels').glob('*.txt')):
            if 'rot' in label.stem:
                continue
            image = ds / split / 'images' / f'{label.stem}.jpg'
            if image.exists():
                pairs.append((image, label))
    if not pairs:
        raise SystemExit(f"no annotated images found under {ds}")
    print(f"Annotated images: {len(pairs)}")

    seg = YOLO(str(resolve_input(cfg.seg_weights, 'segmentation weights')))

    rows = []
    for image_path, label_path in pairs:
        gt_mask, true_count = load_ground_truth(label_path, cfg.imgsz)
        img = np.array(Image.open(image_path).convert('RGB'))

        result = seg.predict(img[:, :, ::-1], conf=cfg.conf, imgsz=cfg.imgsz,
                             device=cfg.device, verbose=False)[0]
        masks, _ = class_masks(result, cfg.imgsz)
        pred_mask = np.zeros((cfg.imgsz, cfg.imgsz), dtype=bool)
        for m in masks[VESSEL_CLASS]:
            pred_mask |= m

        # Deliberately not named with a 'pred_'/'gt_' prefix: those prefixes
        # select the counting methods to be scored, and a mask area is not one.
        row = {'image': image_path.stem, 'true_count': true_count,
               'n_detector_instances': len(masks[VESSEL_CLASS]),
               'mask_px_true': int(gt_mask.sum()),
               'mask_px_predicted': int(pred_mask.sum())}

        # On the predicted mask - what the pipeline will actually see.
        _, n, _ = count_vessels(pred_mask, 'connected_components', cfg.min_area)
        row['pred_connected_components'] = n
        for r in cfg.opening_radii:
            _, n, _ = count_vessels(pred_mask, 'opening', cfg.min_area, radius=r)
            row[f'pred_opening_r{r}'] = n
        for d in cfg.watershed_min_distances:
            _, n, _ = count_vessels(pred_mask, 'watershed', cfg.min_area, min_distance=d)
            row[f'pred_watershed_d{d}'] = n

        # On the ground-truth mask - isolates counting error from segmentation
        # error, since here the mask is by definition correct.
        _, n, _ = count_vessels(gt_mask, 'connected_components', cfg.min_area)
        row['gt_connected_components'] = n
        for d in cfg.watershed_min_distances:
            _, n, _ = count_vessels(gt_mask, 'watershed', cfg.min_area, min_distance=d)
            row[f'gt_watershed_d{d}'] = n

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_image.csv', index=False)

    # scoring
    def score(col):
        err = df[col] - df['true_count']
        return {
            'mae': float(np.abs(err).mean()),
            'bias': float(err.mean()),
            'rmse': float(np.sqrt((err ** 2).mean())),
            'pearson': float(df[col].corr(df['true_count'])),
            'exact_pct': float((err == 0).mean() * 100),
        }

    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    gt_cols = [c for c in df.columns if c.startswith('gt_')]
    scores = {c: score(c) for c in pred_cols + gt_cols + ['n_detector_instances']}
    pd.DataFrame(scores).T.to_csv(out / 'method_scores.csv')

    best = min(pred_cols, key=lambda c: scores[c]['mae'])

    with open(out / 'summary.json', 'w') as f:
        json.dump({'n_images': len(df), 'best_method': best,
                   'true_count_mean': float(df['true_count'].mean()),
                   'scores': scores}, f, indent=2)

    save_method_figure(rows, out / 'method_comparison.png')

    print(f"\nTrue vessel count: mean {df['true_count'].mean():.1f}, "
          f"range {df['true_count'].min()}-{df['true_count'].max()}")

    print(f"\n=== On the PREDICTED mask (what the pipeline sees) ===")
    print(f"{'method':<28}{'MAE':>7}{'bias':>8}{'r':>7}{'exact':>8}")
    for c in sorted(pred_cols, key=lambda c: scores[c]['mae']):
        s = scores[c]
        print(f"{c.replace('pred_', ''):<28}{s['mae']:>7.1f}{s['bias']:>+8.1f}"
              f"{s['pearson']:>7.2f}{s['exact_pct']:>7.0f}%")
    s = scores['n_detector_instances']
    print(f"{'(raw detector instances)':<28}{s['mae']:>7.1f}{s['bias']:>+8.1f}"
          f"{s['pearson']:>7.2f}{s['exact_pct']:>7.0f}%")

    print(f"\n=== On the GROUND-TRUTH mask (isolates counting from segmentation) ===")
    print(f"{'method':<28}{'MAE':>7}{'bias':>8}{'r':>7}{'exact':>8}")
    for c in sorted(gt_cols, key=lambda c: scores[c]['mae']):
        s = scores[c]
        print(f"{c.replace('gt_', ''):<28}{s['mae']:>7.1f}{s['bias']:>+8.1f}"
              f"{s['pearson']:>7.2f}{s['exact_pct']:>7.0f}%")

    print(f"\nBest on predicted masks: {best.replace('pred_', '')} "
          f"(MAE {scores[best]['mae']:.1f}, bias {scores[best]['bias']:+.1f})")
    print("\nIf a method scores well on the ground-truth mask but poorly on the")
    print("predicted one, the counter is fine and the segmentation is the limit.")
    print("If it scores poorly on both, the counting method is the problem.")
    print(f"\nWrote per_image.csv, method_scores.csv, method_comparison.png "
          f"and summary.json to {out}")


if __name__ == '__main__':
    main()
