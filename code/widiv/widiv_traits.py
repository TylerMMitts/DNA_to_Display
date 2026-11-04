# Measures WiDiv traits from a 512 px image by tissue colour, with no segmenter.
# Each pixel goes to the nearest of the rendered tissue colours (learned from
# real image + mask pairs); traits are read off that map. validate() checks each
# trait on real images against the delivered per-plot phenotypes.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

sys.path.insert(0, r'D:\DNA_to_Display\code\widiv')
from widiv_paths import WIDIV_SCALED_DIR, WIDIV_SCALED_MASKS_DIR, WIDIV_SCALED_METADATA

UM = 5.371
TRAITS = ['root_diam_um', 'stele_diam_um', 'mx_n', 'mx_diam_um', 'rca_pct', 'mcs_layers', 'cort_cell_um']
NAMES = {1: 'epidermis', 2: 'exodermis', 3: 'sclerenchyma', 4: 'cortex', 5: 'endodermis', 6: 'pericycle',
         7: 'stele', 8: 'phloem', 9: 'protoxylem'}


def build_palette(n_images=40):
    meta = pd.read_csv(WIDIV_SCALED_METADATA)
    rows = meta[meta.scaled_has_mask].iloc[::60][:n_images]
    acc = {}
    for n in rows.scaled_filename:
        img = np.asarray(Image.open(WIDIV_SCALED_DIR / n).convert('RGB')).reshape(-1, 3)
        m = np.asarray(Image.open(WIDIV_SCALED_MASKS_DIR / n)).ravel()
        for lab in range(1, 10):
            if (m == lab).any():
                acc.setdefault(lab, []).append(img[m == lab])
    labels = sorted(acc)
    centres = np.array([np.median(np.concatenate(acc[l]), axis=0) for l in labels] + [[255, 255, 255]],
                       np.float32)
    return labels + ['white'], centres


CLASSES, CENTRES = build_palette()
IDX = {c: i for i, c in enumerate(CLASSES)}
WHITE = IDX['white']


def classify(rgb):
    d = ((rgb[..., None, :].astype(np.float32) - CENTRES) ** 2).sum(-1)
    return d.argmin(-1)


# Cell spacing in the cortex: the dominant period of the grey-level texture,
# from the radially averaged power spectrum of a patch of cortex.
def cell_period(gray, cortex, white):
    ys, xs = np.nonzero(cortex & ~white)
    if len(ys) < 400:
        return np.nan
    best, size = None, 64
    # Patches centred on cortex pixels that are cortex almost throughout.
    rng = np.random.default_rng(0)
    powers = []
    for _ in range(60):
        if len(powers) >= 12:
            break
        i = rng.integers(len(ys))
        y0, x0 = ys[i] - size // 2, xs[i] - size // 2
        if y0 < 0 or x0 < 0 or y0 + size > gray.shape[0] or x0 + size > gray.shape[1]:
            continue
        # Cortex zone throughout, with little aerenchyma in the patch.
        if cortex[y0:y0 + size, x0:x0 + size].mean() < 0.95 or white[y0:y0 + size, x0:x0 + size].mean() > 0.05:
            continue
        p = gray[y0:y0 + size, x0:x0 + size]
        p = (p - p.mean()) * np.hanning(size)[:, None] * np.hanning(size)[None]
        powers.append(np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2)
    if not powers:
        return np.nan
    P = np.mean(powers, 0)
    yy, xx = np.indices(P.shape) - size // 2
    r = np.sqrt(yy ** 2 + xx ** 2)
    radial = ndimage.mean(P, labels=r.round().astype(int), index=np.arange(size // 2))
    # Periods 2.5-10 px (13-54 um) cover the cortical cells.
    freqs = np.arange(size // 2) / size
    band = (freqs > 1 / 10) & (freqs < 1 / 2.5)
    k = np.argmax(np.where(band, radial, 0))
    return UM / freqs[k]


def measure(rgb):
    cls = classify(rgb)
    white = cls == WHITE
    root = ndimage.binary_fill_holes(~white)
    root = ndimage.binary_opening(root, iterations=2)
    stele = ndimage.binary_fill_holes(np.isin(cls, [IDX[l] for l in (6, 7, 8, 9)]))
    stele = ndimage.binary_opening(stele, iterations=2)
    inner = ndimage.binary_erosion(stele, iterations=2)
    lab, n = ndimage.label(white & inner)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else np.array([])
    # The smallest vessel (47 um) is about 58 px here; white specks in the
    # stele texture are a few px.
    vessels = sizes[sizes >= 40]
    endo = ndimage.binary_dilation(stele, iterations=4)
    cortex_zone = root & ~endo & ~ndimage.binary_dilation(~root, iterations=6)
    rca = 100 * (white & cortex_zone).sum() / max(cortex_zone.sum(), 1)
    # Sclerenchyma: dark ring under the epidermis; thickness = area / circumference.
    root_r = np.sqrt(root.sum() / np.pi)
    scl = (cls == IDX[3]) & root
    scl_thick = scl.sum() / max(2 * np.pi * root_r, 1)
    gray = rgb.astype(np.float32).mean(-1)
    return {'root_diam_um': 2 * root_r * UM,
            'stele_diam_um': 2 * np.sqrt(stele.sum() / np.pi) * UM,
            'mx_n': float(len(vessels)),
            'mx_diam_um': float(np.median(2 * np.sqrt(vessels / np.pi)) * UM) if len(vessels) else np.nan,
            'rca_pct': rca,
            'mcs_layers': scl_thick,
            'cort_cell_um': cell_period(gray, cortex_zone, white)}


def validate(n=600):
    meta = pd.read_csv(WIDIV_SCALED_METADATA)
    rows = meta.iloc[np.linspace(0, len(meta) - 1, n).astype(int)]
    got = pd.DataFrame([measure(np.asarray(Image.open(WIDIV_SCALED_DIR / f).convert('RGB')))
                        for f in rows.scaled_filename])
    out = {}
    for t in TRAITS:
        ok = got[t].notna().to_numpy()
        out[t] = float(np.corrcoef(got[t][ok], rows[t].to_numpy()[ok])[0, 1])
    return out


if __name__ == '__main__':
    print('palette classes:', CLASSES)
    print('r(measured on real 512 px image, delivered plot phenotype):')
    for t, r in validate().items():
        print(f'  {t:14s} {r:.3f}')
