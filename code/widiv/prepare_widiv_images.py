# Puts every WiDiv image and mask at one common scale, ready for training.
#
# The delivered images are all 0.5 um per pixel, but each canvas is sized to its
# root (about 1.07 x root diameter), so they range from 1,434 to 5,473 px.
# Resizing each one to the training size on its own would draw every root the
# same size and erase root and stele diameter, which are traits being predicted.
# Instead each image is downsampled by the same factor - the one that fits a
# 5,500 px canvas into 512 px - and centred on a white 512 px canvas, so one
# pixel is 5.37 um in every image and a bigger root is drawn bigger.
#
# Masks go through the same placement with nearest-neighbour sampling, so their
# labels stay labels; they are only needed for a segmenter, not for diffusion.
#
# Writes to results/widiv/scaled_512/, scaled_512_masks/ and
# scaled_512_metadata.csv. Skips images already written, so an interrupted run
# picks up where it stopped. Reads the delivered files only.

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from widiv_paths import (WIDIV_CANVAS_PX, WIDIV_IMAGE_SIZE, WIDIV_IMAGES_DIR, WIDIV_MASKS_DIR,
                         WIDIV_PIXEL_SIZE_UM, WIDIV_PLOT_TABLE, WIDIV_SCALED_DIR,
                         WIDIV_SCALED_MASKS_DIR, WIDIV_SCALED_METADATA)

SCALE = WIDIV_IMAGE_SIZE / WIDIV_CANVAS_PX
BACKGROUND = (255, 255, 255)      # every delivered image has a pure white border


def place(image, resample, fill):
    w, h = image.size
    size = (max(1, round(w * SCALE)), max(1, round(h * SCALE)))
    small = image.resize(size, resample)
    canvas = Image.new(image.mode, (WIDIV_IMAGE_SIZE, WIDIV_IMAGE_SIZE), fill)
    canvas.paste(small, ((WIDIV_IMAGE_SIZE - size[0]) // 2, (WIDIV_IMAGE_SIZE - size[1]) // 2))
    return canvas, size


def process(name):
    out_img = WIDIV_SCALED_DIR / name
    out_mask = WIDIV_SCALED_MASKS_DIR / name
    src_mask = WIDIV_MASKS_DIR / name
    with Image.open(WIDIV_IMAGES_DIR / name) as im:
        w = im.size[0]
        drawn_px = round(w * SCALE)
        if not out_img.exists():
            # BOX averages every source pixel under each output pixel: no ringing
            # at the sharp vessel edges, which LANCZOS would add at this ratio.
            canvas, _ = place(im.convert('RGB'), Image.BOX, BACKGROUND)
            canvas.save(out_img)
    if src_mask.exists() and not out_mask.exists():
        with Image.open(src_mask) as m:
            canvas, _ = place(m.convert('L'), Image.NEAREST, 0)
            canvas.save(out_mask)
    # The scale actually applied, after rounding the drawn size to whole pixels.
    return name, WIDIV_PIXEL_SIZE_UM * w / drawn_px, drawn_px


def main():
    WIDIV_SCALED_DIR.mkdir(parents=True, exist_ok=True)
    WIDIV_SCALED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(WIDIV_PLOT_TABLE)
    names = table.filename.tolist()
    print(f'{len(names)} images -> {WIDIV_IMAGE_SIZE} px at '
          f'{WIDIV_PIXEL_SIZE_UM / SCALE:.3f} um per pixel, into {WIDIV_SCALED_DIR}')

    done = {}
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        for i, (name, um, drawn) in enumerate(pool.imap_unordered(process, names, chunksize=8), 1):
            done[name] = (um, drawn)
            if i % 250 == 0 or i == len(names):
                print(f'  {i}/{len(names)}', flush=True)

    table['scaled_filename'] = table.filename
    table['scaled_um_per_px'] = table.filename.map(lambda n: done[n][0])
    table['scaled_canvas_drawn_px'] = table.filename.map(lambda n: done[n][1])
    table['scaled_has_mask'] = table.filename.map(lambda n: (WIDIV_SCALED_MASKS_DIR / n).exists())
    table.to_csv(WIDIV_SCALED_METADATA, index=False)
    spread = table.scaled_um_per_px
    print(f'um per pixel after rounding: {spread.min():.4f}-{spread.max():.4f} '
          f'(nominal {WIDIV_PIXEL_SIZE_UM / SCALE:.4f})')
    print(f'{int(table.scaled_has_mask.sum())} masks written; metadata in {WIDIV_SCALED_METADATA}')


if __name__ == '__main__':
    main()
