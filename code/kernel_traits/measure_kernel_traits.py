# Measures size, shape and colour of a seed kernel straight from its pixels.
#
# The root segmenter has nothing to find in a kernel, and a kernel needs no
# model: every scaled image is a single kernel on a white canvas at one known
# px/mm, so the kernel is simply everything that is not white. Size comes from
# that mask in millimetres, and colour from the pixels inside it.
#
# The functions here are shared by kernel_trait_fidelity.py, so real, rebuilt and
# generated kernels are all measured identically. Run on its own, this measures
# every real kernel and writes real_kernel_traits.csv, trait_distributions.png,
# mask_check.png and summary.json.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import color, measure, segmentation

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    SEED_KERNEL_TRAITS_DIR, SEED_SCALED_DIR, SEED_SCALED_METADATA, apply_overrides,
    resolve_input, resolve_output,
)


# Every trait measured, with the label figures use. Size and shape first, then
# colour in CIE Lab, which separates lightness from hue so a darker kernel and a
# redder one show up as different traits rather than one mixed RGB shift.
TRAITS = {
    'area_mm2': 'Area (mm²)',
    'height_mm': 'Height (mm)',
    'width_mm': 'Width (mm)',
    'aspect_ratio': 'Height / width',
    'fill_fraction': 'Fill of bounding box',
    'taper': 'Taper (lower / upper width)',
    'lightness': 'Lightness L*',
    'red_green': 'Red-green a*',
    'yellow_blue': 'Yellow-blue b*',
    'chroma': 'Chroma',
    'hue_deg': 'Hue angle (°)',
    'lightness_top_minus_bottom': 'Top - bottom lightness',
}


# The kernel as a boolean mask, or None when there is no kernel to measure.
#
# The threshold is the one rescale_seed_crops.py placed kernels with, so a real
# kernel's height and width here match seed_scaled_metadata.csv. Only the largest
# connected region is kept: a generated image can carry faint specks away from
# the kernel, and those are not part of its size.
def kernel_mask(rgb, ink_threshold=245, min_area_px=200):
    gray = np.asarray(Image.fromarray(rgb).convert('L'))
    labels = measure.label(gray < ink_threshold, connectivity=2)
    if labels.max() == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    # Filled, because a pale patch inside a kernel is still kernel.
    mask = ndimage.binary_fill_holes(labels == sizes.argmax())
    if mask.sum() < min_area_px:
        return None
    # Every real kernel was placed with clear space on all sides, so a region
    # reaching the edge is not a whole kernel - most often an image with no white
    # background at all - and its size would be the canvas, not the kernel.
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return None
    return mask


# Mean width of the mask across a band of rows, given as fractions of its height.
def band_width(mask, top, height, start, stop):
    rows = mask[top + int(start * height):top + max(int(stop * height), int(start * height) + 1)]
    return float(rows.sum(axis=1).mean())


# Every trait for one image, as a dict. Undetected kernels get NaN for all of
# them, so a missing kernel is counted as missing and never measured as tiny.
def measure_kernel(rgb, px_per_mm, ink_threshold=245, min_area_px=200, erode_px=3):
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = kernel_mask(rgb, ink_threshold, min_area_px)
    if mask is None:
        return {'detected': False, **{t: np.nan for t in TRAITS}}

    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    top, height = int(rows[0]), int(rows[-1] - rows[0] + 1)
    width = int(cols[-1] - cols[0] + 1)
    area = int(mask.sum())

    # Colour from the interior only. The outermost pixels are anti-aliased
    # against the white canvas, and would pull every kernel paler than it is.
    interior = ndimage.binary_erosion(mask, iterations=erode_px)
    if not interior.any():
        interior = mask
    lab = color.rgb2lab(rgb / 255.0)
    L, a, b = (lab[..., i][interior] for i in range(3))

    # The plates draw colour as horizontal bands, usually darker at the crown,
    # so the change from top to bottom is a trait of its own.
    thirds = np.flatnonzero(interior.any(axis=1))
    third = max(len(thirds) // 3, 1)
    top_rows, bottom_rows = thirds[:third], thirds[-third:]
    L_top = lab[top_rows, :, 0][interior[top_rows]].mean()
    L_bottom = lab[bottom_rows, :, 0][interior[bottom_rows]].mean()

    return {
        'detected': True,
        'area_mm2': area / px_per_mm ** 2,
        'height_mm': height / px_per_mm,
        'width_mm': width / px_per_mm,
        'aspect_ratio': height / width,
        'fill_fraction': area / (height * width),
        # Below 1 when the kernel narrows toward its tip.
        'taper': band_width(mask, top, height, 0.65, 0.85)
                 / max(band_width(mask, top, height, 0.15, 0.35), 1.0),
        'lightness': float(L.mean()),
        'red_green': float(a.mean()),
        'yellow_blue': float(b.mean()),
        'chroma': float(np.hypot(a.mean(), b.mean())),
        'hue_deg': float(np.degrees(np.arctan2(b.mean(), a.mean()))),
        'lightness_top_minus_bottom': float(L_top - L_bottom),
        'height_px': height,
        'width_px': width,
    }


# The single px/mm every scaled image shares, read from the metadata rather than
# repeated here, so a rescale at a different scale cannot silently disagree.
def scale_from_metadata(metadata):
    values = metadata['px_per_mm'].unique()
    if len(values) != 1:
        raise SystemExit(f"seed metadata has {len(values)} different px_per_mm values; "
                         "the images are not at one common scale")
    return float(values[0])


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def main(overrides=None):
    # Edit these values, then run:
    #     python code/kernel_traits/measure_kernel_traits.py
    class cfg:
        metadata_path = SEED_SCALED_METADATA
        image_dir = SEED_SCALED_DIR
        output_dir = SEED_KERNEL_TRAITS_DIR

        # Kernels drawn with their measured outline, to check the mask by eye.
        n_mask_examples = 12
        seed = 0

    apply_overrides(cfg, overrides)

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_dir = resolve_input(cfg.image_dir, 'scaled seed image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'seed scaled metadata'))
    px_per_mm = scale_from_metadata(metadata)
    print(f"Images: {len(metadata)}   scale: {px_per_mm} px/mm\nOutput: {out}")

    rows = []
    for r in metadata.itertuples():
        traits = measure_kernel(load_rgb(image_dir / r.filename), px_per_mm)
        rows.append({'filename': r.filename, 'genotype': r.genotype, **traits})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'real_kernel_traits.csv', index=False)

    # The mask should reproduce the box rescale_seed_crops.py measured. A
    # disagreement means the mask is picking up something other than the kernel.
    merged = frame.merge(metadata[['filename', 'kernel_height_px', 'kernel_width_px']],
                         on='filename')
    height_diff = (merged.height_px - merged.kernel_height_px).abs()
    width_diff = (merged.width_px - merged.kernel_width_px).abs()
    print(f"\nDetected: {int(frame.detected.sum())}/{len(frame)}")
    print(f"Agreement with seed_scaled_metadata.csv: height off by up to "
          f"{height_diff.max():.0f} px, width by up to {width_diff.max():.0f} px "
          f"({int(((height_diff > 0) | (width_diff > 0)).sum())} kernels differ at all)")

    print(f"\n{'trait':30s} {'mean':>9s} {'sd':>8s} {'min':>9s} {'max':>9s}")
    for t in TRAITS:
        v = frame[t]
        print(f"{t:30s} {v.mean():9.3f} {v.std():8.3f} {v.min():9.3f} {v.max():9.3f}")

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, (t, label) in zip(axes.ravel(), TRAITS.items()):
        ax.hist(frame[t].dropna(), bins=30, color='#8C6D4F')
        ax.set_title(label, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(f'Real kernel traits ({int(frame.detected.sum())} kernels)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / 'trait_distributions.png', dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(cfg.seed)
    picks = rng.choice(len(metadata), min(cfg.n_mask_examples, len(metadata)), replace=False)
    cols = min(6, len(picks))
    fig, axes = plt.subplots(int(np.ceil(len(picks) / cols)), cols,
                             figsize=(2.6 * cols, 2.9 * np.ceil(len(picks) / cols)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, i in zip(axes.ravel(), picks):
        rgb = load_rgb(image_dir / metadata.filename.iloc[i])
        mask = kernel_mask(rgb)
        shown = rgb.copy()
        if mask is not None:
            shown[segmentation.find_boundaries(mask, mode='outer')] = (0, 160, 255)
        ax.imshow(shown)
        f = frame.iloc[i]
        ax.set_title(f"{f.genotype}\n{f.area_mm2:.1f} mm²  L* {f.lightness:.0f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'mask_check.png', dpi=130)
    plt.close(fig)

    summary = {
        'n_images': len(frame), 'n_detected': int(frame.detected.sum()),
        'px_per_mm': px_per_mm,
        'max_height_diff_px_vs_metadata': float(height_diff.max()),
        'max_width_diff_px_vs_metadata': float(width_diff.max()),
        'trait_means': {t: float(frame[t].mean()) for t in TRAITS},
        'trait_sds': {t: float(frame[t].std()) for t in TRAITS},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote real_kernel_traits.csv, trait_distributions.png, mask_check.png "
          f"and summary.json to {out}")


if __name__ == '__main__':
    main()
