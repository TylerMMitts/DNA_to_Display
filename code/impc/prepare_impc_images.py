# IMPC whole-body dorso-ventral X-rays -> the 512 px training images the WiDiv
# pipeline reads (WIDIV_DATASET=impc_bcm). Run after fetch_impc_xray.py and
# organize_impc.py:
#
#   WIDIV_DATASET=impc_bcm python code/impc/prepare_impc_images.py [--limit N] [--out DIR]
#
# Per image:
#   1. The BCM JPEGs are a 2048 x 2048 radiograph with a grey-scale legend panel
#      appended on the right; only the first 2048 columns are kept.
#   2. The mouse: Otsu threshold, largest component, holes filled. Its trunk (head
#      and body without legs and tail) is the mask after an opening with a disc
#      wider than a leg or the tail.
#   3. Rotated so the trunk's long axis is vertical. BCM mice lie head up; the
#      rotation is the smallest one that makes the axis vertical (|angle| <= 90),
#      so a head-up mouse stays head up.
#   4. One physical scale for every mouse. The imaging setup changed over the years
#      and the JPEG width (the legend panel) tells the setups apart: wild-type
#      controls measure about 1,290 px of trunk on 2,408-2,948 px wide JPEGs
#      (2013 - Nov 2015) and about 1,527 px on 2,048-2,364 px ones, while the IMPC's
#      tibia lengths in mm stay flat over the same years - a magnification change,
#      not biology (results/impc_bcm/pipeline_check/). Pass 1 measures every mouse's
#      trunk; each JPEG-width group's scale factor is the reference group's median
#      trunk length (the largest group, 2,364 px) over its own median (all mice of
#      the group: hundreds of genes average out). Groups of fewer than MIN_GROUP
#      images cannot be calibrated and are left out. Pass 2 crops a square of
#      CANVAS_PX / factor source px around the trunk and resizes it to 512 px, so
#      every image has the reference scale (4 reference px per output px) and body
#      size stays a phenotype instead of being normalised away per mouse.
#   5. Intensities: each image is rescaled by the 1st and 99.7th percentiles of its
#      mouse pixels (the legend styles render tissue at different grey levels), then
#      inverted - white background, dark bone, as the root pipeline expects - and
#      everything outside the mouse set to pure white. Remaining date-to-date
#      differences are handled by the BATCH_ columns of the genotype matrix.
#   6. Saved as 3-channel PNG (the LiteVAE reads RGB), named <line>.png.
#
# Writes results/impc_bcm/scaled_512/, scaled_512_metadata.csv (scaled_filename,
# genotype = line, and QC: source width, scale group and factor, rotation, trunk
# length and width in source px, whether the mouse touches the canvas edge) and
# scale_groups.csv.
import os
os.environ.setdefault('WIDIV_DATASET', 'impc_bcm')
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'widiv'))
from widiv_paths import (WIDIV_IMAGE_SIZE, WIDIV_IMAGES_DIR, WIDIV_METADATA_DIR,  # noqa: E402
                         WIDIV_PLOT_PHENOTYPES, WIDIV_RESULTS_DIR, WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA)

RADIOGRAPH_PX = 2048       # the radiograph part of every BCM JPEG
CANVAS_PX = 2048           # reference-scale source px per side of the training square (4 per output px)
REFERENCE_WIDTH = 2364     # JPEG width of the largest setup group, whose scale every image is brought to
MIN_GROUP = 5              # a setup group needs this many images to estimate its scale
TRUNK_OPENING = 22         # disc radius (source px) that removes legs and tail but keeps the trunk
WORK = 4                   # segmentation runs at 1/WORK resolution for speed


def segment(src):
    im = Image.open(src)
    width = im.size[0]
    g = np.asarray(im.convert('L'), dtype=np.float32)[:, :RADIOGRAPH_PX]
    small = g[::WORK, ::WORK]
    mouse = small > threshold_otsu(small)
    lab, n = ndimage.label(mouse)
    if n == 0:
        return None
    mouse = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(ndimage.sum(mouse, lab, range(1, n + 1)))))
    trunk = ndimage.binary_opening(mouse, structure=disk(max(1, TRUNK_OPENING // WORK)))
    lab, n = ndimage.label(trunk)
    if n == 0:
        return None
    trunk = lab == 1 + int(np.argmax(ndimage.sum(trunk, lab, range(1, n + 1))))
    ys, xs = np.nonzero(trunk)
    evals, evecs = np.linalg.eigh(np.cov(np.stack([xs, ys])))
    vx, vy = evecs[:, 1]                                   # long axis
    angle = np.degrees(np.arctan2(vx, vy))                 # 0 = vertical
    angle = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
    return {'g': g, 'mouse': mouse, 'width_px': width, 'cy': ys.mean() * WORK, 'cx': xs.mean() * WORK,
            'angle': angle, 'length': 4 * np.sqrt(evals[1]) * WORK, 'trunk_width': 4 * np.sqrt(evals[0]) * WORK}


def measure(src):
    s = segment(src)
    if s is None:
        return {'source_width_px': np.nan, 'trunk_length_px': np.nan, 'error': 'no mouse found'}
    return {'source_width_px': s['width_px'], 'trunk_length_px': float(s['length']), 'error': ''}


def process(args):
    src, dst, factor = args
    s = segment(src)
    if s is None:
        return {'error': 'no mouse found'}
    g, mouse, cx, cy, angle = s['g'], s['mouse'], s['cx'], s['cy'], s['angle']
    full_mouse = np.asarray(Image.fromarray(mouse.astype(np.uint8) * 255).resize(
        (RADIOGRAPH_PX, g.shape[0]), Image.NEAREST)) > 0
    full_mouse = ndimage.binary_dilation(full_mouse, iterations=3)
    lo, hi = np.percentile(g[full_mouse], [1, 99.7])
    v = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
    neg = (np.where(full_mouse, 1 - v, 1.0) * 255).astype(np.uint8)

    # Rotate about the trunk centre (negative: background is white), then crop a
    # square that holds CANVAS_PX reference px; 8-bit padding covers any crop.
    canvas = int(round(CANVAS_PX / factor))
    pad = canvas // 2 + 64
    big = Image.fromarray(np.pad(neg, pad, constant_values=255))
    bm = Image.fromarray(np.pad(full_mouse, pad).astype(np.uint8) * 255)
    centre = (cx + pad, cy + pad)
    rot = big.rotate(-angle, resample=Image.BICUBIC, center=centre, fillcolor=255)
    mrot = bm.rotate(-angle, resample=Image.NEAREST, center=centre, fillcolor=0)
    x0, y0 = int(centre[0] - canvas // 2), int(centre[1] - canvas // 2)
    box = (x0, y0, x0 + canvas, y0 + canvas)
    crop, mcrop = rot.crop(box), np.asarray(mrot.crop(box)) > 0
    edge = bool(mcrop[0].any() or mcrop[-1].any() or mcrop[:, 0].any() or mcrop[:, -1].any())
    crop.resize((WIDIV_IMAGE_SIZE, WIDIV_IMAGE_SIZE), Image.LANCZOS).convert('RGB').save(dst)
    return {'rotation_deg': round(float(angle), 2), 'trunk_width_px': round(float(s['trunk_width']), 1),
            'canvas_source_px': canvas, 'mouse_touches_edge': edge,
            'intensity_lo': float(lo), 'intensity_hi': float(hi), 'error': ''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='first N mice only (a trial run)')
    ap.add_argument('--out', default=None, help='output folder (default: the dataset scaled folder)')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else WIDIV_SCALED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mice = pd.read_csv(WIDIV_PLOT_PHENOTYPES)
    img = pd.read_csv(WIDIV_METADATA_DIR / 'impc_xray_images.csv').drop_duplicates('observation_id')
    img = img.merge(mice[['specimen_id', 'line']], on='specimen_id')
    img['n'] = img.groupby('line').cumcount()
    # One observation per mouse in BCM: <line>.png; a second would be <line>_2.png.
    img['scaled_filename'] = np.where(img.n == 0, img.line + '.png', img.line + '_' + (img.n + 1).astype(str) + '.png')
    img['src'] = [WIDIV_IMAGES_DIR / f'{o}.jpg' for o in img.observation_id]
    img = img[[p.exists() for p in img.src]].reset_index(drop=True)
    if args.limit:
        img = img.head(args.limit)

    print(f'Pass 1: measuring {len(img):,} mice')
    with ProcessPoolExecutor(args.workers) as pool:
        m = pd.DataFrame(list(pool.map(measure, img.src, chunksize=16)))
    img = pd.concat([img, m], axis=1)
    groups = (img[img.error == ''].groupby('source_width_px')
              .agg(images=('trunk_length_px', 'size'), median_trunk_px=('trunk_length_px', 'median'),
                   first=('date_of_experiment', 'min'), last=('date_of_experiment', 'max')))
    ref = groups.loc[REFERENCE_WIDTH, 'median_trunk_px']
    groups['scale_factor'] = np.where(groups.images >= MIN_GROUP, ref / groups.median_trunk_px, np.nan)
    groups.to_csv((WIDIV_RESULTS_DIR if not args.out else out_dir.parent) / 'scale_groups.csv')
    print(groups.round(3).to_string())
    img['scale_factor'] = img.source_width_px.map(groups.scale_factor)
    img.loc[(img.error == '') & img.scale_factor.isna(), 'error'] = 'setup group too small to calibrate'
    img['trunk_length_ref_px'] = img.trunk_length_px * img.scale_factor

    todo = img[img.error == '']
    print(f'Pass 2: writing {len(todo):,} images -> {out_dir} ({(img.error != "").sum()} left out)')
    with ProcessPoolExecutor(args.workers) as pool:
        qc = pd.DataFrame(list(pool.map(process, [(r.src, out_dir / r.scaled_filename, r.scale_factor)
                                                  for r in todo.itertuples()], chunksize=16)), index=todo.index)
    img.update(qc[['error']])
    meta = img[['scaled_filename', 'line', 'observation_id', 'specimen_id', 'source_width_px', 'scale_factor',
                'trunk_length_px', 'trunk_length_ref_px', 'error']].join(qc.drop(columns='error'))
    meta = meta.rename(columns={'line': 'genotype'})
    meta = meta[meta.error == '']                          # the trainers list only usable images
    meta_path = WIDIV_SCALED_METADATA if not args.out else out_dir.parent / f'{out_dir.name}_metadata.csv'
    meta.to_csv(meta_path, index=False)
    left = img[img.error != ''][['observation_id', 'line', 'source_width_px', 'error']]
    left.to_csv(meta_path.with_name(meta_path.stem + '_left_out.csv'), index=False)
    print(f'{len(meta):,} written, {len(left)} left out; rotation |deg| median {meta.rotation_deg.abs().median():.1f}, '
          f'> 20 deg: {(meta.rotation_deg.abs() > 20).sum()}; mouse touching canvas edge: '
          f'{meta.mouse_touches_edge.mean():.0%}; trunk length (reference px) median '
          f'{meta.trunk_length_ref_px.median():.0f}')
    print(f'metadata -> {meta_path}')


if __name__ == '__main__':
    main()
