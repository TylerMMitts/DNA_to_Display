# Image helpers shared by the stages: a fast per-image survey on thumbnails,
# background and subject detection, and contact sheets for the agents to look at.
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
THUMB = 160


def open_rgb(path, max_side=None):
    im = Image.open(path)
    if max_side and im.format == 'JPEG':
        im.draft('RGB', (max_side, max_side))      # decode JPEGs at reduced size: much faster
    im = ImageOps.exif_transpose(im)
    if im.mode in ('I;16', 'I;16B', 'I', 'F'):      # 16-bit / float microscopy
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.convert('RGB')
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.BILINEAR)
    return im


def border_pixels(a, width=None):
    h, w = a.shape[:2]
    k = width or max(2, min(h, w) // 40)
    return np.concatenate([a[:k].reshape(-1, a.shape[2]), a[-k:].reshape(-1, a.shape[2]),
                           a[:, :k].reshape(-1, a.shape[2]), a[:, -k:].reshape(-1, a.shape[2])])


def otsu(values, bins=128):
    v = np.asarray(values, dtype=float).ravel()
    hist, edges = np.histogram(v, bins=bins, range=(0.0, max(float(v.max()), 1e-6)))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    between = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(edges[int(np.argmax(between)) + 1])


def objects(m):
    """Label the objects of a mask. Returns (labels, main label, panel labels). A panel is
    a solid strip along an edge (a legend, scale bar or label); the main subject is the
    largest object that is not a panel, since a legend can be bigger than the subject."""
    from scipy import ndimage
    lab, n = ndimage.label(m)
    if n == 0:
        return lab, 0, []
    h, w = m.shape
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    panels = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        tall, wide = ys.stop - ys.start, xs.stop - xs.start
        fill = sizes[k - 1] / max(tall * wide, 1)
        side = tall > 0.3 * h and wide < 0.35 * w and (xs.start < 0.2 * w or xs.stop > 0.8 * w)
        flat = wide > 0.3 * w and tall < 0.35 * h and (ys.start < 0.2 * h or ys.stop > 0.8 * h)
        if (side or flat) and fill > 0.6:
            panels.append(k)
    order = [k for k in np.argsort(-sizes) + 1 if k not in panels] or list(np.argsort(-sizes) + 1)
    return lab, int(order[0]), panels


def largest_object(m):
    """The main subject alone (see objects)."""
    lab, main, _ = objects(m)
    return lab == main if main else m


def subject_mask(a, background=None, tol=0.12):
    """Pixels that differ from the background colour (a float RGB array in 0-1)."""
    if background is None:
        background = np.median(border_pixels(a), axis=0)
    d = np.abs(a - np.asarray(background)[None, None]).max(axis=2)
    from scipy import ndimage
    # Otsu's threshold on the distance from the background, never below tol: a fixed
    # tolerance counts vignetting (X-rays brighter in the middle) as subject. A bright
    # legend panel would drag Otsu's threshold up until only the brightest parts of the
    # subject pass, so panels are found first and the threshold is recomputed without them.
    m = d > max(tol, otsu(d))
    lab, _, panels = objects(ndimage.binary_opening(m, iterations=1))
    if panels:
        excl = ndimage.binary_dilation(np.isin(lab, panels), iterations=2)
        m = (d > max(tol, otsu(d[~excl]))) & ~excl | np.isin(lab, panels)
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = sizes >= max(4, 0.002 * m.size)            # drop dust
    return np.isin(lab, np.flatnonzero(keep) + 1)


def survey_one(path):
    """Cheap statistics of one image, on a thumbnail. Never raises."""
    path = Path(path)
    out = {'path': str(path)}
    try:
        with Image.open(path) as im0:
            out.update(width=im0.size[0], height=im0.size[1], mode=im0.mode, format=im0.format)
        im = open_rgb(path, THUMB * 2)
        im.thumbnail((THUMB, THUMB))
        a = np.asarray(im, dtype=np.float32) / 255
        bg_px = border_pixels(a)
        bg = np.median(bg_px, axis=0)
        m = subject_mask(a, bg)
        grey = a.mean(axis=2)
        out.update(
            mean=float(grey.mean()), std=float(grey.std()),
            bg_r=float(bg[0]), bg_g=float(bg[1]), bg_b=float(bg[2]),
            bg_brightness=float(bg.mean()), bg_uniformity=float(bg_px.std(axis=0).mean()),
            colourfulness=float(np.abs(a - grey[..., None]).mean()),
            subject_fraction=float(m.mean()),
            # Same picture (re-saved or re-sized copies), not same silhouette: a binary
            # 16 px outline called 6,009 of 8,798 different mouse X-rays duplicates.
            thumb_hash=hashlib.md5((np.asarray(im.convert('L').resize((32, 32), 2)) // 8)
                                   .tobytes()).hexdigest(),
        )
        if m.any():
            from scipy import ndimage
            h, w = m.shape
            # The main subject and anything else: a legend, scale bar or label panel along
            # an edge (the BCM mouse X-rays carry a grey-scale legend on the right).
            lab, big, panels = objects(m)
            n = int(lab.max())
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            panel = bool(panels)
            my, mx = ndimage.find_objects(lab)[big - 1]
            out.update(n_objects=int(n), largest_share=float(sizes[big - 1] / sizes.sum()),
                       largest_fraction=float((lab == big).mean()), edge_panel=panel,
                       main_bbox_w=float((mx.stop - mx.start) / w), main_bbox_h=float((my.stop - my.start) / h))
            ys, xs = np.nonzero(m)
            out.update(
                bbox_w=float((xs.max() - xs.min() + 1) / w), bbox_h=float((ys.max() - ys.min() + 1) / h),
                centre_dx=float(xs.mean() / w - 0.5), centre_dy=float(ys.mean() / h - 0.5),
                touches_edge=bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
            )
            # Shape and brightness of the main subject only: a legend panel or label would
            # otherwise count as part of the subject.
            m = lab == big
            ys, xs = np.nonzero(m)
            # Shape: elongation and axis angle from second moments, circularity from area/perimeter.
            c = np.cov(np.vstack([xs, ys]))
            ev, evec = np.linalg.eigh(c)
            out['elongation'] = float(np.sqrt(ev[1] / max(ev[0], 1e-9)))
            out['axis_angle'] = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)
            from scipy import ndimage
            perim = (m ^ ndimage.binary_erosion(m)).sum()
            # Boundary-pixel counts underestimate a digital perimeter by about 12%, so a disc
            # would score 1.26; the factor brings a disc to about 1 (rough, fine for rules).
            out['circularity'] = float(min(4 * np.pi * m.sum() / max(perim * 1.12, 1) ** 2, 1.0))
            out['subject_mean'] = float(grey[m].mean())
            out['subject_sd'] = float(grey[m].std())
            out['subject_vs_bg'] = float(grey[m].mean() - bg.mean())
        out['ok'] = True
    except Exception as e:  # noqa: BLE001 - recorded as an unreadable image
        out['ok'] = False
        out['error'] = f'{type(e).__name__}: {e}'
    return out


def survey(paths, workers=None):
    paths = [str(p) for p in paths]
    if len(paths) < 64:
        return [survey_one(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(survey_one, paths, chunksize=32))


def contact_sheet(paths, out_path, labels=None, cols=6, cell=220, title=None):
    """A labelled grid of images, for the dashboard and the agents."""
    paths = list(paths)
    if not paths:
        return None
    rows = (len(paths) + cols - 1) // cols
    top = 28 if title else 0
    sheet = Image.new('RGB', (cols * cell, rows * (cell + 18) + top), 'white')
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((6, 6), title, fill='black')
    for i, p in enumerate(paths):
        try:
            im = open_rgb(p, cell * 2)
            im.thumbnail((cell - 6, cell - 6))
        except Exception:  # noqa: BLE001
            im = Image.new('RGB', (cell - 6, cell - 6), (230, 80, 80))
        x, y = (i % cols) * cell, top + (i // cols) * (cell + 18)
        sheet.paste(im, (x + (cell - im.size[0]) // 2, y + (cell - im.size[1]) // 2))
        label = (labels[i] if labels else Path(p).stem)[:34]
        draw.text((x + 4, y + cell), label, fill='black')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def robust_z(values, min_scale=0.0):
    """Median/MAD z-score. min_scale is the smallest spread that counts: on a feature
    that is nearly constant (colourfulness of greyscale X-rays, a perfectly white
    background) the MAD is about zero and any trace difference would score z > 5."""
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if not mad or np.isnan(mad):
        mad = np.nanstd(v)
    mad = max(mad or 0.0, min_scale)
    return (v - med) / mad if mad else np.zeros_like(v)
