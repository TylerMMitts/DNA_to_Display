# Crops the title and the scale bar off the seed kernel scans.
#
# The scans are rendered plates, not photographs: a genotype name at the top, the
# kernel in the middle, a scale bar at the bottom. Every plate puts those in the
# same rows, so one fixed crop works for all of them - but the crop is checked
# against each image rather than trusted, because a plate that does not match the
# expected layout would otherwise be silently cropped through the kernel.
#
# The scale bar is measured before it is discarded. Each plate was rendered to a
# fixed kernel height, so the bar is the only record of how large the kernel
# actually is; cropping it away without reading it first would make a small
# kernel and a large one indistinguishable.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    SEED_CROPPED_DIR, SEED_IMAGE_METADATA, SEED_SCANS_DIR,
    resolve_input, resolve_output,
)


# Rows holding ink, as (first, last) pairs. The plates are white paper with a few
# separated elements on them, so contiguous runs of inked rows are the elements.
def ink_blocks(gray, threshold):
    inked = (gray < threshold).any(axis=1)
    blocks, start = [], None
    for i, v in enumerate(inked):
        if v and start is None:
            start = i
        elif not v and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(inked) - 1))
    return blocks


# Confirms this plate really is laid out the way the fixed crop assumes.
#
# The kernel is the tallest inked block. It has to sit entirely inside the crop
# window, and every other element has to sit entirely outside it, or the crop is
# either cutting the kernel or leaving text behind.
def check_layout(gray, top, bottom, threshold):
    blocks = ink_blocks(gray, threshold)
    if not blocks:
        return None, 'blank image'

    kernel = max(blocks, key=lambda b: b[1] - b[0])
    if kernel[0] < top or kernel[1] >= bottom:
        return kernel, f'kernel rows {kernel[0]}-{kernel[1]} fall outside the crop'

    for b in blocks:
        if b is kernel:
            continue
        if not (b[1] < top or b[0] >= bottom):
            return kernel, f'non-kernel ink at rows {b[0]}-{b[1]} survives the crop'
    return kernel, None


# Length of the scale bar in pixels, read from the strip below the crop.
def measure_scale_bar(gray, bottom, threshold):
    strip = gray[bottom:, :] < threshold
    columns = strip.any(axis=0)
    if not columns.any():
        return 0, -1, -1
    left = int(np.argmax(columns))
    right = int(len(columns) - 1 - np.argmax(columns[::-1]))
    return int(columns.sum()), left, right


def main():
    # Edit these values, then run:
    #     python code/crop_seed_scans.py
    class cfg:
        source_dir = SEED_SCANS_DIR
        output_dir = SEED_CROPPED_DIR
        metadata_out = SEED_IMAGE_METADATA

        # Measured across all 548 plates: title ink ends by row 71, the kernel
        # occupies rows 115-704, and the earliest scale bar starts at row 743.
        # This window clears the title by 10 rows and the bar by 2, and leaves
        # about 40 rows of white either side of the widest kernel.
        crop_top = 72
        crop_bottom = 743

        # A pixel counts as ink below this grey level, on 0-255. The plates are
        # rendered on pure white, so anything short of about 250 is safe.
        # Measured: moving this between 200 and 250 changes a bar length by at
        # most 2 px out of ~320, so the anti-aliased edges do not matter here.
        ink_threshold = 245

        # What one scale bar means in millimetres. The scans are 1200 dpi, which
        # is 1200 / 2.54 = 472.4 px/cm, and the bar is drawn 236 px long there,
        # so it marks half a centimetre. Everything absolute follows from this
        # one number; the relative sizes do not depend on it at all.
        bar_mm = 5.0

        # Stop before writing anything if any plate fails the layout check,
        # rather than producing a folder where some crops are wrong.
        strict = True

    source = resolve_input(cfg.source_dir, 'seed scan directory')
    out_dir = resolve_output(cfg.output_dir)
    meta_path = resolve_output(cfg.metadata_out)

    scans = sorted(p for p in source.iterdir() if p.suffix.lower() == '.png')
    if not scans:
        raise SystemExit(f"no .png plates found in {source}")
    print(f"Source: {source}\nPlates: {len(scans)}")
    print(f"Crop:   rows {cfg.crop_top} to {cfg.crop_bottom} "
          f"({cfg.crop_bottom - cfg.crop_top} rows kept)\n")

    # Checked first, written second. A bad plate is worth knowing about before
    # half the folder has been replaced.
    rows, problems = [], []
    for path in scans:
        gray = np.asarray(Image.open(path).convert('L'))
        height, width = gray.shape

        if height <= cfg.crop_bottom:
            problems.append(f"{path.name}: only {height} rows, crop needs {cfg.crop_bottom}")
            continue

        kernel, failure = check_layout(gray, cfg.crop_top, cfg.crop_bottom,
                                       cfg.ink_threshold)
        if failure:
            problems.append(f"{path.name}: {failure}")
            continue

        bar_px, bar_left, bar_right = measure_scale_bar(gray, cfg.crop_bottom,
                                                        cfg.ink_threshold)
        if bar_px == 0:
            problems.append(f"{path.name}: no scale bar found below row {cfg.crop_bottom}")
            continue

        inked = gray[kernel[0]:kernel[1] + 1, :] < cfg.ink_threshold
        columns = inked.any(axis=0)
        rows.append({
            'filename': path.name,
            'genotype': path.stem[2:] if path.stem.startswith('K_') else path.stem,
            'source_width_px': width,
            'source_height_px': height,
            'kernel_top_px': kernel[0] - cfg.crop_top,
            'kernel_bottom_px': kernel[1] - cfg.crop_top,
            'kernel_height_px': kernel[1] - kernel[0] + 1,
            'kernel_width_px': int(columns.sum()),
            'scale_bar_px': bar_px,
            'scale_bar_left_px': bar_left,
            'scale_bar_right_px': bar_right,
            # Size relative to the bar. The plates are rendered to a fixed kernel
            # height, so these are the only size comparisons that mean anything.
            'kernel_height_per_bar': (kernel[1] - kernel[0] + 1) / bar_px,
            'kernel_width_per_bar': int(columns.sum()) / bar_px,
            # The same sizes in millimetres, once the bar is given a length.
            'mm_per_bar': cfg.bar_mm,
            'plate_px_per_mm': bar_px / cfg.bar_mm,
            'kernel_height_mm': (kernel[1] - kernel[0] + 1) / bar_px * cfg.bar_mm,
            'kernel_width_mm': int(columns.sum()) / bar_px * cfg.bar_mm,
        })

    if problems:
        print(f"{len(problems)} plate(s) did not match the expected layout:")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        if cfg.strict:
            raise SystemExit(
                "\nNothing was written. Fix the crop window for these plates, or "
                "set strict = False to skip them and crop the rest.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image = Image.open(source / row['filename'])
        image.crop((0, cfg.crop_top, image.width, cfg.crop_bottom)).save(
            out_dir / row['filename'])

    frame = pd.DataFrame(rows)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(meta_path, index=False)

    print(f"Cropped {len(rows)} plates to {out_dir}")
    if problems:
        print(f"Skipped {len(problems)}")
    print(f"Wrote {meta_path.name} to {meta_path.parent}\n")

    print(f"Scale bar:     {frame.scale_bar_px.min()} to {frame.scale_bar_px.max()} px "
          f"({frame.scale_bar_px.nunique()} distinct), taken as {cfg.bar_mm} mm")
    print(f"Kernel height: {frame.kernel_height_px.min()} to "
          f"{frame.kernel_height_px.max()} px on the plate, but "
          f"{frame.kernel_height_mm.min():.1f} to {frame.kernel_height_mm.max():.1f} mm "
          f"in real size (median {frame.kernel_height_mm.median():.1f})")
    print(f"Kernel width:  {frame.kernel_width_mm.min():.1f} to "
          f"{frame.kernel_width_mm.max():.1f} mm "
          f"(median {frame.kernel_width_mm.median():.1f})")
    print("The second range is the real one. Resize these crops without carrying "
          "scale_bar_px\nthrough and every kernel becomes the same size.")


if __name__ == '__main__':
    main()
