# Puts every seed kernel at one common scale, on a square canvas.
#
# The plates were rendered to a fixed kernel height, so a kernel's size on the
# plate says nothing about its real size - only the scale bar does. Every bar
# marks the same real length, so resizing each image by px_per_bar / its own bar
# length puts all of them in the same units, and the kernels then differ in
# pixels exactly as much as they differ in life.
#
# The canvas is filled by padding, never by resizing to fit. Resizing each image
# to the canvas is what the training transforms would do by default, and it
# would undo all of this - every kernel would come out the same size again.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    SEED_CROPPED_DIR, SEED_IMAGE_METADATA, SEED_SCALED_DIR, SEED_SCALED_METADATA,
    resolve_input, resolve_output,
)


# Bounding box of the kernel, as (left, top, right, bottom) inclusive.
def ink_box(gray, threshold):
    inked = gray < threshold
    rows, cols = inked.any(axis=1), inked.any(axis=0)
    if not rows.any():
        return None
    top, bottom = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    left, right = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return left, top, right, bottom


# Flattens onto white first. The plates are RGBA and their transparent border
# would otherwise come through as black once the alpha channel is dropped.
def load_on_white(path):
    image = Image.open(path)
    if image.mode == 'RGBA':
        canvas = Image.new('RGB', image.size, (255, 255, 255))
        canvas.paste(image, mask=image.split()[3])
        return canvas
    return image.convert('RGB')


def main():
    # Edit these values, then run:
    #     python code/rescale_seed_crops.py
    class cfg:
        source_dir = SEED_CROPPED_DIR
        metadata_in = SEED_IMAGE_METADATA
        output_dir = SEED_SCALED_DIR
        metadata_out = SEED_SCALED_METADATA

        # Pixels one scale bar becomes. This is a fixed constant on purpose: if
        # it were computed from whichever images happen to be present, adding a
        # larger kernel later would change the scale of everything and make new
        # images incomparable with ones already generated or trained on.
        #
        # 97 fits the largest kernel in the current set (2.454 bar lengths, so
        # 238 px) inside a 256 canvas with 9 px to spare.
        px_per_bar = 97.0

        # 256 is what LiteVAE and the diffusion model already take, so these
        # need no further resizing at training time.
        canvas = 256
        background = (255, 255, 255)

        # Every kernel must land inside the canvas with at least this much clear
        # space around it. Raising px_per_bar until something clips is the
        # failure this catches.
        min_margin_px = 2

        ink_threshold = 245

    source = resolve_input(cfg.source_dir, 'cropped seed image directory')
    meta = pd.read_csv(resolve_input(cfg.metadata_in, 'seed image metadata'))
    out_dir = resolve_output(cfg.output_dir)
    meta_out = resolve_output(cfg.metadata_out)

    print(f"Source: {source}\nImages: {len(meta)}")
    print(f"Scale:  {cfg.px_per_bar} px per scale bar, onto a "
          f"{cfg.canvas}x{cfg.canvas} canvas\n")

    rows, problems, pending = [], [], []
    for record in meta.itertuples():
        path = source / record.filename
        if not path.exists():
            problems.append(f"{record.filename}: missing from {source}")
            continue

        image = load_on_white(path)
        factor = cfg.px_per_bar / record.scale_bar_px
        scaled = image.resize((max(1, round(image.width * factor)),
                               max(1, round(image.height * factor))),
                              Image.LANCZOS)

        box = ink_box(np.asarray(scaled.convert('L')), cfg.ink_threshold)
        if box is None:
            problems.append(f"{record.filename}: no kernel found after scaling")
            continue
        left, top, right, bottom = box
        width, height = right - left + 1, bottom - top + 1

        limit = cfg.canvas - 2 * cfg.min_margin_px
        if width > limit or height > limit:
            problems.append(
                f"{record.filename}: kernel is {width}x{height} px at this scale, "
                f"more than the {limit} px the canvas allows")
            continue

        # Offset that lands the kernel's centre on the canvas centre. Padding is
        # whatever is left over, so a small kernel simply sits in more white.
        offset = (round(cfg.canvas / 2 - (left + right + 1) / 2),
                  round(cfg.canvas / 2 - (top + bottom + 1) / 2))

        pending.append((record.filename, scaled, offset))
        rows.append({
            'filename': record.filename,
            'genotype': record.genotype,
            'scale_bar_px': record.scale_bar_px,
            'resize_factor': factor,
            'kernel_width_px': width,
            'kernel_height_px': height,
            'kernel_width_bars': width / cfg.px_per_bar,
            'kernel_height_bars': height / cfg.px_per_bar,
            'canvas_px': cfg.canvas,
            'px_per_bar': cfg.px_per_bar,
            'fill_fraction': (width * height) / (cfg.canvas ** 2),
        })

    if problems:
        print(f"{len(problems)} image(s) could not be placed:")
        for p in problems[:20]:
            print(f"  {p}")
        raise SystemExit("\nNothing was written. Lower px_per_bar so every kernel fits.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, scaled, offset in pending:
        canvas = Image.new('RGB', (cfg.canvas, cfg.canvas), cfg.background)
        canvas.paste(scaled, offset)
        canvas.save(out_dir / filename)

    frame = pd.DataFrame(rows)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(meta_out, index=False)

    print(f"Wrote {len(frame)} images to {out_dir}")
    print(f"Wrote {meta_out.name} to {meta_out.parent}\n")
    print(f"Kernel height: {frame.kernel_height_px.min()} to "
          f"{frame.kernel_height_px.max()} px "
          f"({frame.kernel_height_px.max() / frame.kernel_height_px.min():.2f}x, "
          "and that ratio is now the real size difference)")
    print(f"Kernel width:  {frame.kernel_width_px.min()} to "
          f"{frame.kernel_width_px.max()} px")
    print(f"Canvas fill:   {frame.fill_fraction.min():.1%} to "
          f"{frame.fill_fraction.max():.1%}")


if __name__ == '__main__':
    main()
