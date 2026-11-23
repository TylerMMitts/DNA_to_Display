# Shared pieces for the RIL v5 checkpoint analysis. Inference only.
# Test lines are never read here beyond their names (to exclude them): the sender
# asked that the 495 test lines are not used for anything, including choosing a
# checkpoint, so every score is on validation lines (held out of training) and,
# for comparison, trained lines.
import os
os.environ['WIDIV_DATASET'] = 'ril_v5'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
import widiv_traits as WT  # palette built from RIL image + mask pairs (WIDIV_DATASET above)
from widiv_paths import WIDIV_RESULTS_DIR, WIDIV_SCALED_METADATA, WIDIV_SPLIT

OUT = WIDIV_RESULTS_DIR / 'checkpoint_selection'
OUT.mkdir(parents=True, exist_ok=True)
TRAITS = ['root_diam_um', 'stele_diam_um', 'mx_n', 'mx_diam_um', 'rca_pct', 'mcs_layers', 'cortex_width_um']
# Same validity rule as generate_all_lines.py.
VALID_TRAITS = ['root_diam_um', 'stele_diam_um', 'mx_n', 'rca_pct']
RAYS = 180


def cortex_width(cls, root, endo, centre):
    # Colour version of cortex_span() in measure_traits.py: along 180 rays from the
    # root centre, first to last pixel that is cortex or aerenchyma (white inside
    # the root, outside the stele), median over rays.
    band = ((cls == WT.IDX[4]) | ((cls == WT.WHITE) & root)) & ~endo
    cy, cx = centre
    t = np.arange(0, 256, 0.5)
    spans = []
    for a in np.linspace(0, 2 * np.pi, RAYS, endpoint=False):
        ys = np.clip((cy - t * np.sin(a)).astype(int), 0, band.shape[0] - 1)
        xs = np.clip((cx + t * np.cos(a)).astype(int), 0, band.shape[1] - 1)
        hit = np.flatnonzero(band[ys, xs])
        spans.append(hit[-1] - hit[0] + 1 if hit.size else 0)
    return float(np.median(spans) * 0.5 * WT.UM)


def measure(rgb):
    # widiv_traits.measure minus cortical cell size, plus cortex width.
    out = {k: v for k, v in WT.measure(rgb).items() if k != 'cort_cell_um'}
    cls = WT.classify(rgb)
    white = cls == WT.WHITE
    root = ndimage.binary_opening(ndimage.binary_fill_holes(~white), iterations=2)
    stele = ndimage.binary_opening(ndimage.binary_fill_holes(np.isin(cls, [WT.IDX[l] for l in (6, 7, 8, 9)])),
                                   iterations=2)
    endo = ndimage.binary_dilation(stele, iterations=4)
    centre = ndimage.center_of_mass(root) if root.any() else (rgb.shape[0] / 2, rgb.shape[1] / 2)
    out['cortex_width_um'] = cortex_width(cls, root, endo, centre)
    return out


def load_tables():
    # Phenotypes of every non-test line only; test-line rows are dropped on read.
    meta = pd.read_csv(WIDIV_SCALED_METADATA)
    split = pd.read_csv(WIDIV_SPLIT).set_index('line').split_random
    meta = meta[meta.genotype.map(split) == 'train'].set_index('genotype')
    return meta, split
