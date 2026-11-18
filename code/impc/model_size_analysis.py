# Should the IMPC X-ray diffusion model be larger than the medium preset the RIL and
# WiDiv runs used? Nothing is trained here (training runs on Hellbender); this
# compares the three datasets on what decides model size:
#   1. data: training images, distinct genotype conditions, images per condition,
#      and parameters per training image for each size preset;
#   2. image complexity, on 600 training-format 512 px images per dataset:
#      - share of the image that is subject (not white background),
#      - effective dimensionality: PCA of 128 px grey images, components for 90% of
#        the variance and the participation ratio (sum lambda)^2 / sum lambda^2,
#      - fine detail: mean |Laplacian| over subject pixels at 512 px, and PNG bytes
#        per subject pixel (how much a lossless coder must store).
# Writes results/impc_bcm/size_analysis/size_analysis.csv.
#   python code/impc/model_size_analysis.py
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'results' / 'impc_bcm' / 'size_analysis'
N = 600
SETS = {
    'WiDiv roots': ROOT / 'results' / 'widiv' / 'scaled_512',
    'RIL v5 roots': ROOT / 'results' / 'ril_v5' / 'scaled_512',
    'IMPC BCM X-rays': OUT / 'impc_512',
}
# Training images and genotype conditions the diffusion model sees (validation and
# test held out): WiDiv ~2,930 images from ~733 lines; RIL ~1,780 lines, one image
# each; IMPC ~6,750 mice in ~1,700 conditions (gene x zygosity x sex, plus controls
# by sex), counted from the split below.
UNET = {'small': 2.7e6, 'medium': 8.8e6, 'large': 29.3e6}


def impc_conditions():
    t = pd.read_csv(ROOT / 'dataset' / 'impc_bcm' / 'metadata' / 'mouse_table.csv')
    t = t[t.split_random == 'train']
    n_train = int(round(len(t) * 0.9))                      # 10% validation mice
    cond = np.where(t.biological_sample_group == 'control', 'wildtype', t.gene_symbol + '|' + t.zygosity) + '|' + t.sex
    return n_train, len(set(cond))


def stats(folder, rng):
    files = sorted(folder.glob('*.png'))
    files = [files[i] for i in rng.choice(len(files), min(N, len(files)), replace=False)]
    small, fg, lap, png = [], [], [], []
    for f in files:
        g = np.asarray(Image.open(f).convert('L'), dtype=np.float32) / 255
        subj = g < 0.96
        fg.append(subj.mean())
        lap.append(np.abs(ndimage.laplace(g))[subj].mean() if subj.any() else 0)
        buf = io.BytesIO()
        Image.fromarray((g * 255).astype(np.uint8)).save(buf, format='PNG', optimize=True)
        png.append(buf.tell() / max(subj.sum(), 1))
        small.append(np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR),
                                dtype=np.float32).ravel() / 255)
    X = np.stack(small)
    X -= X.mean(0)
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    cum = np.cumsum(lam) / lam.sum()
    return {'images_scored': len(files), 'subject_share': np.mean(fg),
            'pcs_for_90pct': int(np.searchsorted(cum, 0.90) + 1),
            'participation_ratio': float(lam.sum() ** 2 / (lam ** 2).sum()),
            'detail_laplacian': float(np.mean(lap)), 'png_bytes_per_subject_px': float(np.mean(png))}


def main():
    rng = np.random.default_rng(0)
    n_impc, c_impc = impc_conditions()
    data = {'WiDiv roots': (2930, 733), 'RIL v5 roots': (1780, 1780), 'IMPC BCM X-rays': (n_impc, c_impc)}
    rows = []
    for name, folder in SETS.items():
        if not folder.exists():
            print(f'skip {name}: {folder} missing')
            continue
        s = stats(folder, rng)
        n, c = data[name]
        rows.append({'dataset': name, 'train_images': n, 'conditions': c, 'images_per_condition': n / c, **s,
                     **{f'params_per_image_{k}': v / n for k, v in UNET.items()}})
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / 'size_analysis.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.round(3).T.to_string())


if __name__ == '__main__':
    main()
