# Cross-validates the root/stele/vessel segmenter and the ways traits are read from it.
#
# Every earlier accuracy figure for the segmenter was measured mostly on images it
# trained on: only 7 of the 34 annotated images were ever held out. Here the 34 are
# split into folds, one model is trained per fold, and each image is scored only by
# the model that never saw it, so a change that looks better actually is.
#
# Compared on those held-out images:
#   - resolution: training and predicting at 256 px as now, against 512 px - the
#     same 256 px image upscaled. YOLOv8 predicts masks at a quarter of its input
#     size, and a 64x64 grid cannot draw the one-pixel walls between vessels.
#   - (not retina masks: from Ultralytics 8.4.146 every mask is upsampled before it
#     is cut out, which is what retina_masks used to add, and on this data the two
#     gave identical masks. That same change is why vessel area read 1.77x the
#     annotated area under 8.4.50 and 1.06x under 8.4.146 with the same weights, so
#     vessel areas are only comparable when measured under the same version.)
#   - detector confidence, and vessel counts from the detector's own instances
#     against watershed on the merged mask
# Every trait is scored against the annotations, since mask IoU says little about
# whether a diameter or a vessel count is right.
#
# Trains on the CPU only; the GPU is hidden from the process. Resumable: a fold
# that finished training is not retrained. Writes per_image.csv,
# trait_scores.csv, summary.json and cross_validation.png.

import os

# Hidden before torch loads, so neither training nor prediction can reach the GPU.
# This laptop's power supply cannot carry a GPU training load.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import DATASET_DIR, RESULTS_DIR, apply_overrides, resolve_input, resolve_output

from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version

from feature_segmentation.evaluation.reconstruction_fidelity_test import class_masks
from feature_segmentation.prepare_dataset import read_label, rot90_polygon, write_label
from feature_segmentation.vessel_counting import count_vessels

ROOT, STELE, VESSEL = 0, 1, 2
# Traits in the units the rest of the pipeline uses: pixels of a 256 px image.
TRAITS = ['root_diameter', 'stele_diameter', 'stele_root_ratio', 'vessel_area', 'vessel_count']


def annotated_images(source_dir):
    image_dir, label_dir = source_dir / 'train' / 'images', source_dir / 'train' / 'labels'
    pairs = [(p, label_dir / f'{p.stem}.txt') for p in sorted(image_dir.iterdir())
             if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    return [(i, l) for i, l in pairs if l.exists()]


def source_of(image_path):
    return 'MEMA' if image_path.name.startswith('MEMA') else 'W3'


# Fold number per image. Dealt out within each source in turn, so every fold holds
# MEMA roots - the dataset being modelled - as well as W3 roots from the other
# experiment, and a fold of only one kind cannot skew the comparison.
def assign_folds(pairs, n_folds, seed):
    rng = np.random.default_rng(seed)
    folds, dealt = np.empty(len(pairs), dtype=int), 0
    for source in ('MEMA', 'W3'):
        members = [i for i, (p, _) in enumerate(pairs) if source_of(p) == source]
        for i in rng.permutation(members):
            folds[i] = dealt % n_folds
            dealt += 1
    return folds


# The image a model at imgsz sees: squashed to 256 px first, as every crop and
# every generated image is, then upscaled. Generated roots only exist at 256 px,
# so upscaling from there is the only way a 512 px segmenter can be fed them.
def model_input(image_path, imgsz):
    with Image.open(image_path) as im:
        im = im.convert('RGB').resize((256, 256), Image.LANCZOS)
    return im if imgsz == 256 else im.resize((imgsz, imgsz), Image.BICUBIC)


def build_fold_dataset(pairs, folds, fold, imgsz, out_dir):
    yaml_path = out_dir / 'data.yaml'
    if yaml_path.exists():
        return yaml_path
    for split in ('train', 'val'):
        (out_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    for (image_path, label_path), f in zip(pairs, folds):
        split = 'val' if f == fold else 'train'
        rows = read_label(label_path)
        image = model_input(image_path, imgsz)
        image.save(out_dir / split / 'images' / f'{image_path.stem}.jpg', quality=95)
        write_label(out_dir / split / 'labels' / f'{image_path.stem}.txt', rows)
        # The same lossless 90-degree turns train_segmentation.py adds, on the
        # training split only.
        if split == 'train':
            base = np.array(image)
            for k in (1, 2, 3):
                Image.fromarray(np.rot90(base, k=k)).save(
                    out_dir / split / 'images' / f'{image_path.stem}_rot{k * 90}.jpg', quality=95)
                write_label(out_dir / split / 'labels' / f'{image_path.stem}_rot{k * 90}.txt',
                            [(c, rot90_polygon(coords, k)) for c, coords in rows])
    yaml_path.write_text(f"path: {out_dir.as_posix()}\ntrain: train/images\nval: val/images\n\n"
                         f"nc: 3\nnames: ['root', 'stele', 'vessel']\n")
    return yaml_path


# Trains one fold, or resumes it if a previous run was interrupted. A marker file
# is written only once training ends, because best.pt exists from the first epoch
# and would otherwise make a half-trained fold look finished.
def train_fold(yaml_path, run_dir, imgsz, cfg):
    weights, marker = run_dir / 'weights' / 'best.pt', run_dir / 'finished.txt'
    if marker.exists():
        return weights, 0.0
    start = time.time()
    last = run_dir / 'weights' / 'last.pt'
    if last.exists():
        YOLO(str(last)).train(resume=True, device='cpu', workers=0)
    else:
        # The augmentation train_segmentation.py uses: nothing that moves or
        # rescales the root, since size is what is being measured.
        YOLO(cfg.model).train(
            data=str(yaml_path), imgsz=imgsz, epochs=cfg.epochs, patience=cfg.patience,
            batch=cfg.batch, seed=cfg.seed, device='cpu', workers=0,
            project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
            deterministic=True, plots=False, verbose=False,
            translate=0.0, scale=0.0, mosaic=0.0, shear=0.0, perspective=0.0,
            copy_paste=0.0, mixup=0.0, degrees=0.0,
            fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3)
    marker.write_text(f'{time.time() - start:.0f} seconds\n')
    return weights, time.time() - start


def polygon_mask(coords, size):
    image = Image.new('L', (size, size), 0)
    ImageDraw.Draw(image).polygon((np.asarray(coords).reshape(-1, 2) * size).ravel().tolist(), fill=255)
    return np.array(image) > 127


def diameter(area):
    return 2.0 * np.sqrt(area / np.pi) if area > 0 else np.nan


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan


# The annotated truth for one image: traits at 256 px, masks at imgsz for IoU.
def ground_truth(label_path, imgsz):
    rows = read_label(label_path)
    truth, masks = {}, {}
    for cls in (ROOT, STELE, VESSEL):
        polys = [c for k, c in rows if k == cls]
        small = np.zeros((256, 256), bool)
        big = np.zeros((imgsz, imgsz), bool)
        for coords in polys:
            small |= polygon_mask(coords, 256)
            big |= polygon_mask(coords, imgsz)
        masks[cls] = big
        if cls == VESSEL:
            truth['vessel_area'] = float(small.sum())
            truth['vessel_count'] = len(polys)
        else:
            truth[f"{'root' if cls == ROOT else 'stele'}_diameter"] = diameter(small.sum())
    truth['stele_root_ratio'] = truth['stele_diameter'] / truth['root_diameter']
    return truth, masks


# Every way of reading traits from one prediction, one row per confidence
# threshold. Areas are scaled back to 256 px units and watershed's min_distance
# is scaled up with the image, so all resolutions report comparable numbers.
def read_traits(result, imgsz, truth_masks, confs, distances):
    masks, conf_of = class_masks(result, imgsz)
    scale = (256 / imgsz) ** 2
    rows = []
    for threshold in confs:
        kept_vessels = [m for m, c in zip(masks[VESSEL], conf_of.get(VESSEL, [])) if c >= threshold]
        best = {}
        for cls in (ROOT, STELE):
            candidates = [(c, m) for m, c in zip(masks[cls], conf_of.get(cls, [])) if c >= threshold]
            best[cls] = max(candidates, key=lambda x: x[0])[1] if candidates else np.zeros((imgsz, imgsz), bool)
        vessels = np.zeros((imgsz, imgsz), bool)
        for m in kept_vessels:
            vessels |= m
        row = {'conf': threshold,
               'root_diameter': diameter(best[ROOT].sum() * scale),
               'stele_diameter': diameter(best[STELE].sum() * scale),
               'vessel_area': float(vessels.sum() * scale),
               'vessel_count_instances': len(kept_vessels),
               'iou_root': iou(best[ROOT], truth_masks[ROOT]),
               'iou_stele': iou(best[STELE], truth_masks[STELE]),
               'iou_vessel': iou(vessels, truth_masks[VESSEL])}
        row['stele_root_ratio'] = row['stele_diameter'] / row['root_diameter']
        for d in distances:
            _, n, _ = count_vessels(vessels, method='watershed', min_area=max(4, round(4 * imgsz ** 2 / 256 ** 2)),
                                    min_distance=round(d * imgsz / 256))
            row[f'vessel_count_watershed_d{d}'] = n
        rows.append(row)
    return rows


# Error of one predicted column against its annotated trait, over held-out images.
def score(frame, predicted, truth):
    ok = frame[[predicted, truth]].dropna()
    err = ok[predicted] - ok[truth]
    # r is undefined when either side is constant - a model that finds nothing.
    varies = len(ok) > 2 and ok[predicted].nunique() > 1 and ok[truth].nunique() > 1
    return {'n': len(ok), 'mae': float(err.abs().mean()), 'bias': float(err.mean()),
            'mae_pct_of_mean': float(err.abs().mean() / ok[truth].mean() * 100),
            'r': float(ok[predicted].corr(ok[truth])) if varies else np.nan,
            'mean_ratio': float((ok[predicted] / ok[truth]).mean())}


def main(overrides=None):
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/cross_validate_segmentation.py
    class cfg:
        source_dataset = DATASET_DIR / 'root_features_new.yolov8'
        output_dir = RESULTS_DIR / 'segmentation_cross_validation'

        n_folds = 5
        imgsizes = [256, 512]
        model = 'yolov8n-seg.pt'
        epochs = 50
        patience = 10
        batch = 4
        seed = 0

        confs = [0.25, 0.4, 0.55]
        watershed_distances = [4, 5, 6, 8]

    apply_overrides(cfg, overrides)

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = annotated_images(resolve_input(cfg.source_dataset, 'annotated segmentation dataset'))
    folds = assign_folds(pairs, cfg.n_folds, cfg.seed)
    pd.DataFrame({'image': [p.stem for p, _ in pairs], 'source': [source_of(p) for p, _ in pairs],
                  'fold': folds}).to_csv(out / 'folds.csv', index=False)
    print(f"Ultralytics {ultralytics_version} (vessel areas depend on the version)")
    print(f"Annotated images: {len(pairs)} "
          f"({sum(source_of(p) == 'MEMA' for p, _ in pairs)} MEMA), {cfg.n_folds} folds\nOutput: {out}")

    rows, train_times = [], []
    for imgsz in cfg.imgsizes:
        for fold in range(cfg.n_folds):
            name = f'imgsz{imgsz}_fold{fold}'
            yaml_path = build_fold_dataset(pairs, folds, fold, imgsz, out / 'data' / name)
            print(f"\n[{name}] training on CPU "
                  f"({int((folds != fold).sum())} images x 4 rotations, "
                  f"{int((folds == fold).sum())} held out)", flush=True)
            weights, seconds = train_fold(yaml_path, out / 'runs' / name, imgsz, cfg)
            train_times.append({'run': name, 'seconds': round(seconds)})
            print(f"[{name}] {'trained in ' + f'{seconds / 60:.1f} min' if seconds else 'already trained'}",
                  flush=True)

            model = YOLO(str(weights))
            for (image_path, label_path), f in zip(pairs, folds):
                if f != fold:
                    continue
                truth, truth_masks = ground_truth(label_path, imgsz)
                pixels = np.array(Image.open(out / 'data' / name / 'val' / 'images' / f'{image_path.stem}.jpg'))
                result = model.predict(pixels[:, :, ::-1], conf=min(cfg.confs), imgsz=imgsz,
                                       device='cpu', verbose=False)[0]
                for predicted in read_traits(result, imgsz, truth_masks, cfg.confs,
                                             cfg.watershed_distances):
                    rows.append({'imgsz': imgsz, 'fold': fold,
                                 'image': image_path.stem, 'source': source_of(image_path),
                                 **{f'true_{k}': v for k, v in truth.items()},
                                 **{f'pred_{k}': v for k, v in predicted.items() if k != 'conf'},
                                 'conf': predicted['conf']})
            pd.DataFrame(rows).to_csv(out / 'per_image.csv', index=False)

    per_image = pd.DataFrame(rows)
    count_methods = ['vessel_count_instances'] + [f'vessel_count_watershed_d{d}'
                                                  for d in cfg.watershed_distances]
    score_rows = []
    for (imgsz, conf), group in per_image.groupby(['imgsz', 'conf']):
        setting = {'imgsz': imgsz, 'conf': conf}
        for trait in ['root_diameter', 'stele_diameter', 'stele_root_ratio', 'vessel_area']:
            score_rows.append({**setting, 'trait': trait, 'method': '',
                               **score(group, f'pred_{trait}', f'true_{trait}')})
        for method in count_methods:
            score_rows.append({**setting, 'trait': 'vessel_count', 'method': method.replace('vessel_count_', ''),
                               **score(group, f'pred_{method}', 'true_vessel_count')})
        for cls in ('root', 'stele', 'vessel'):
            score_rows.append({**setting, 'trait': f'iou_{cls}', 'method': '',
                               'n': len(group), 'mae': np.nan, 'bias': np.nan, 'mae_pct_of_mean': np.nan,
                               'r': np.nan, 'mean_ratio': float(group[f'pred_iou_{cls}'].mean())})
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / 'trait_scores.csv', index=False)

    # The spread of each trait across the annotated roots, for judging whether an
    # error is small relative to the differences being measured.
    truth_sd = {t: float(per_image.drop_duplicates('image')[f'true_{t}'].std()) for t in TRAITS}

    def get(frame, trait, column, method=''):
        sel = frame[(frame.trait == trait) & (frame.method == method)]
        return float(sel[column].iloc[0]) if len(sel) else np.nan

    print(f"\nHeld-out accuracy ({len(pairs)} images, each scored by a model that never saw it)")
    print(f"{'imgsz':>5} {'conf':>5} | {'root d MAE':>10} {'stele d MAE':>11} "
          f"{'ratio MAE':>9} | {'vessel area':>11} | {'count MAE: inst':>15} {'best wshed':>14} | "
          f"{'IoU r/s/v':>14}")
    settings = []
    for (imgsz, conf), s in scores.groupby(['imgsz', 'conf']):
        counts = s[s.trait == 'vessel_count'].set_index('method')
        best_ws = counts.drop('instances')['mae'].idxmin()
        settings.append({'imgsz': imgsz, 'conf': conf,
                         'count_mae_instances': counts.loc['instances', 'mae'],
                         'count_bias_instances': counts.loc['instances', 'bias'],
                         'count_mae_best_watershed': counts.loc[best_ws, 'mae'],
                         'best_watershed': best_ws,
                         'vessel_area_ratio': get(s, 'vessel_area', 'mean_ratio'),
                         'root_diameter_mae': get(s, 'root_diameter', 'mae'),
                         'stele_diameter_mae': get(s, 'stele_diameter', 'mae'),
                         'ratio_mae': get(s, 'stele_root_ratio', 'mae'),
                         'iou_root': get(s, 'iou_root', 'mean_ratio'),
                         'iou_stele': get(s, 'iou_stele', 'mean_ratio'),
                         'iou_vessel': get(s, 'iou_vessel', 'mean_ratio')})
        x = settings[-1]
        print(f"{imgsz:>5} {conf:>5.2f} | {x['root_diameter_mae']:>10.1f} "
              f"{x['stele_diameter_mae']:>11.1f} {x['ratio_mae']:>9.3f} | "
              f"{x['vessel_area_ratio']:>10.2f}x | {x['count_mae_instances']:>15.2f} "
              f"{x['count_mae_best_watershed']:>6.2f} ({best_ws.replace('watershed_', '')}) | "
              f"{x['iou_root']:.2f}/{x['iou_stele']:.2f}/{x['iou_vessel']:.2f}")
    print(f"\nFor scale, spread (SD) across annotated roots: root d {truth_sd['root_diameter']:.1f}, "
          f"stele d {truth_sd['stele_diameter']:.1f}, ratio {truth_sd['stele_root_ratio']:.3f}, "
          f"vessel area {truth_sd['vessel_area']:.0f}, vessel count {truth_sd['vessel_count']:.1f}")
    print("vessel area is predicted / annotated area; 1.00x is unbiased.")

    frame = pd.DataFrame(settings)
    frame['label'] = frame.apply(lambda r: f"{r.imgsz}px conf {r.conf:.2f}", axis=1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    x = np.arange(len(frame))
    panels = [
        (axes[0, 0], [('count_mae_instances', 'detector instances', '#4C78A8'),
                      ('count_mae_best_watershed', 'best watershed', '#F58518')],
         'Vessel count error (mean abs., vessels)', None),
        (axes[0, 1], [('vessel_area_ratio', 'predicted / annotated', '#54A24B')],
         'Vessel area (1.0 = unbiased)', 1.0),
        (axes[1, 0], [('root_diameter_mae', 'root', '#B279A2'), ('stele_diameter_mae', 'stele', '#9D7660')],
         'Diameter error (mean abs., px at 256)', None),
        (axes[1, 1], [('iou_root', 'root', '#B279A2'), ('iou_stele', 'stele', '#9D7660'),
                      ('iou_vessel', 'vessel', '#4C78A8')], 'Mask IoU', None),
    ]
    for ax, bars, title, reference in panels:
        width = 0.8 / len(bars)
        for k, (column, label, colour) in enumerate(bars):
            ax.bar(x + (k - (len(bars) - 1) / 2) * width, frame[column], width=width,
                   color=colour, label=label)
        if reference is not None:
            ax.axhline(reference, color='#444444', linewidth=0.8, linestyle='--')
        ax.set_xticks(x, frame['label'], rotation=40, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(f'Segmenter accuracy on held-out images ({cfg.n_folds}-fold, {len(pairs)} annotated roots)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / 'cross_validation.png', dpi=130)
    plt.close(fig)

    frame.drop(columns='label').to_csv(out / 'settings_summary.csv', index=False)
    (out / 'summary.json').write_text(json.dumps({
        'n_images': len(pairs), 'n_folds': cfg.n_folds, 'epochs': cfg.epochs,
        'ultralytics_version': ultralytics_version,
        'truth_sd': truth_sd, 'train_seconds': train_times,
        'settings': frame.drop(columns='label').to_dict('records'),
    }, indent=2, default=float))
    print(f"\nWrote folds.csv, per_image.csv, trait_scores.csv, settings_summary.csv, "
          f"summary.json and cross_validation.png to {out}")


if __name__ == '__main__':
    main()
