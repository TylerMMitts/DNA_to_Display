# Trait measurement that works on any prepared image, real or generated.
#
# Scoring a generated image means measuring it the same way as a real one, so
# every trait here is computed by code from the 512 px training-format image
# (subject on a light background). A recipe says which measurements to make:
#
#   shape       area, equivalent diameter, major and minor axis, eccentricity,
#               solidity, perimeter of the subject
#   intensity   mean grey, its spread, mean R/G/B and texture (mean |Laplacian|)
#               inside the subject
#   inner       a region inside the subject found by Otsu's threshold (darker or
#               brighter than the rest): its area, diameter and share of the
#               subject (a stele, a nucleus, an organ)
#   blobs       small structures inside the subject, darker or brighter than
#               their surroundings, within a size range: count, mean area and
#               area share (vessels, cells, spots)
#   plugin      a trained domain segmenter, e.g. 'maize_root_cross_section'
#               (this repository's YOLOv8 root model)
#
# Sizes are converted to micrometres when the image's pixel size is known.
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

DEFAULT_RECIPE = {
    'subject_tolerance': 0.06,
    'shape': True,
    'intensity': True,
    'inner': {'enabled': True, 'polarity': 'dark'},
    'blobs': [{'name': 'dark_blobs', 'polarity': 'dark', 'min_area_px': 6, 'max_area_px': 400, 'contrast': 0.12}],
    'plugin': 'none',
    'um_per_px': 0.0,
}


def _grey(a):
    return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def subject(a, tol):
    """The subject: pixels darker than the white background, largest component, holes filled."""
    m = a.min(axis=2) < 1 - tol
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(m)


def _shape(m):
    ys, xs = np.nonzero(m)
    area = float(len(ys))
    if area < 10:
        return {}
    c = np.cov(np.vstack([xs, ys]))
    ev = np.sort(np.linalg.eigvalsh(c))
    major, minor = 4 * np.sqrt(ev[1]), 4 * np.sqrt(max(ev[0], 0))
    perim = float((m ^ ndimage.binary_erosion(m)).sum())
    # Solidity against the convex hull of the subject.
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.column_stack([xs, ys]))
        solidity = area / max(hull.volume, 1)
    except Exception:  # noqa: BLE001 - degenerate shapes
        solidity = np.nan
    return {'area_px': area, 'equivalent_diameter_px': float(2 * np.sqrt(area / np.pi)),
            'major_axis_px': float(major), 'minor_axis_px': float(minor),
            'eccentricity': float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else np.nan,
            'solidity': float(solidity), 'perimeter_px': perim}


def _otsu(v):
    hist, edges = np.histogram(v, bins=64, range=(0, 1))
    w = np.cumsum(hist)
    mu = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    total, mu_t = w[-1], mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        between = (mu_t * w / total - mu) ** 2 / (w * (total - w))
    return float(edges[int(np.nanargmax(between)) + 1])


def measure_array(a, recipe=DEFAULT_RECIPE, return_masks=False):
    """Traits of one image (float RGB 0-1, light background)."""
    out, masks = {}, {}
    m = subject(a, recipe.get('subject_tolerance', 0.06))
    masks['subject'] = m
    g = _grey(a)
    if recipe.get('shape', True):
        out.update(_shape(m))
    if m.sum() < 10:
        return (out, masks) if return_masks else out
    if recipe.get('intensity', True):
        out['mean_grey'] = float(g[m].mean())
        out['sd_grey'] = float(g[m].std())
        for i, ch in enumerate('rgb'):
            out[f'mean_{ch}'] = float(a[..., i][m].mean())
        out['texture'] = float(np.abs(ndimage.laplace(g))[ndimage.binary_erosion(m, iterations=2)].mean())
    inner = recipe.get('inner', {})
    if inner.get('enabled'):
        thr = _otsu(g[m])
        im = (g < thr) if inner.get('polarity') == 'dark' else (g > thr)
        im = ndimage.binary_opening(im & m, iterations=2)
        lab, n = ndimage.label(im)
        if n:
            # The component nearest the subject's centre, not just the largest.
            cy, cx = ndimage.center_of_mass(m)
            coms = ndimage.center_of_mass(im, lab, range(1, n + 1))
            sizes = ndimage.sum(im, lab, range(1, n + 1))
            score = [s / (1 + np.hypot(y - cy, x - cx)) for (y, x), s in zip(coms, sizes)]
            inner_m = ndimage.binary_fill_holes(lab == int(np.argmax(score)) + 1)
            masks['inner'] = inner_m
            out['inner_area_px'] = float(inner_m.sum())
            out['inner_diameter_px'] = float(2 * np.sqrt(inner_m.sum() / np.pi))
            out['inner_share'] = float(inner_m.sum() / m.sum())
    blobs = recipe.get('blobs', [])
    if blobs:
        # Local background for blob contrast: a 25 px box mean (a box filter is
        # separable and about 50x faster than a median of the same size).
        core = ndimage.binary_erosion(m, iterations=3)
        local = ndimage.uniform_filter(g, size=25)
    for b in blobs:
        diff = (local - g) if b['polarity'] == 'dark' else (g - local)
        bm = (diff > b['contrast']) & core
        lab, n = ndimage.label(bm)
        if n:
            sizes = ndimage.sum(bm, lab, range(1, n + 1))
            ok = (sizes >= b['min_area_px']) & (sizes <= b['max_area_px'])
            masks[b['name']] = np.isin(lab, np.flatnonzero(ok) + 1)
            out[f"{b['name']}_count"] = float(ok.sum())
            out[f"{b['name']}_mean_area_px"] = float(sizes[ok].mean()) if ok.any() else 0.0
            out[f"{b['name']}_area_share"] = float(sizes[ok].sum() / m.sum())
        else:
            out[f"{b['name']}_count"] = 0.0
            out[f"{b['name']}_mean_area_px"] = 0.0
            out[f"{b['name']}_area_share"] = 0.0
    um = recipe.get('um_per_px') or 0
    if um:
        for k in list(out):
            if k.endswith('_px'):
                val = out[k] * (um ** 2 if k.startswith(('area', 'inner_area')) or 'mean_area' in k else um)
                out[k[:-3] + ('_um2' if k.startswith(('area', 'inner_area')) or 'mean_area' in k else '_um')] = val
    return (out, masks) if return_masks else out


_PLUGIN = {}


def _plugin_traits(a, name):
    if name == 'maize_root_cross_section':
        import sys
        code = Path(__file__).resolve().parents[1] / 'code'   # the project's code, read only
        sys.path.insert(0, str(code))
        from paths import SEGMENTATION_MODEL, pick_device  # noqa: E402
        from feature_segmentation.evaluation.reconstruction_fidelity_test import segment  # noqa: E402
        if 'seg' not in _PLUGIN:
            from ultralytics import YOLO
            _PLUGIN['seg'] = YOLO(str(SEGMENTATION_MODEL))
            _PLUGIN['dev'] = pick_device()
        traits, _ = segment(_PLUGIN['seg'], (a * 255).astype(np.uint8), 0.25, _PLUGIN['dev'], 4, 2)
        return {f'root_{k}': float(v) for k, v in traits.items() if isinstance(v, (int, float))}
    return {}


def _measure_one(job):
    p, recipe = job
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255
    try:
        return {'image': Path(p).name, **measure_array(a, recipe)}
    except Exception:  # noqa: BLE001 - a blank or degenerate image: no values, not a failed run
        return {'image': Path(p).name}


def measure_paths(paths, recipe=DEFAULT_RECIPE):
    """Traits of many images; classical measurements run on every CPU core,
    a GPU plugin afterwards in this process."""
    paths = [str(p) for p in paths]
    jobs = [(p, recipe) for p in paths]
    # Worker processes sized from free memory, not one per core: each loads numpy, scipy
    # and an image; twelve at once ran a 16 GB laptop out of memory (resources.py).
    from . import resources
    n = resources.workers(per_worker_gb=0.6) if len(paths) > 48 else 0
    if n:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n) as ex:
            rows = list(ex.map(_measure_one, jobs, chunksize=8))
    else:
        rows = [_measure_one(j) for j in jobs]
    if recipe.get('plugin', 'none') != 'none':
        for row, p in zip(rows, paths):
            a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255
            row.update(_plugin_traits(a, recipe['plugin']))
    return pd.DataFrame(rows)


def overlay(path, recipe, out_path):
    """The image with its measured regions outlined, for the dashboard and agents."""
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    _, masks = measure_array(a, recipe, return_masks=True)
    out = (a * 255).astype(np.uint8)
    colours = {'subject': (241, 184, 45), 'inner': (47, 91, 211)}
    for name, m in masks.items():
        # 3 px outlines, so the regions stay visible in a contact-sheet thumbnail.
        edge = ndimage.binary_dilation(m ^ ndimage.binary_erosion(m), iterations=1)
        out[edge] = colours.get(name, (214, 40, 40))
    Image.fromarray(out).save(out_path)
    return str(out_path)
