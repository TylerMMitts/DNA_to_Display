# Image preprocessing: one consistent training format for any image set.
#
# What the training code needs (code/widiv/): 512 x 512 RGB, a light uniform
# background (the LiteVAE weights background pixels down and pulls their latents
# to a fixed value, so a dark or busy background must be inverted or flattened),
# and one physical scale for every image when size is a trait. That last point
# matters most: resizing each image to fill the frame erases every size trait.
# So by default each image is cut from a fixed canvas of source pixels around its
# subject and only then downsampled, exactly as prepare_widiv_images.py does.
#
# Analysis surveys every image on a thumbnail and flags what breaks training:
# unreadable or blank files, duplicates, and images unlike the rest (inverted
# polarity, a different background, a different view), which is what collapsed
# the mouse LiteVAE (four inverted X-rays among 8,800).
#
# It also looks for batch drift: whole groups of images that differ together. A
# single odd image is an outlier; a magnification change between imaging setups or
# a darker exposure one year shifts every image of that group, and the model then
# learns the date instead of the genotype (the mouse X-rays had both). Groups are
# the source image dimensions and any covariates intake found (imaging date, sex,
# plate...). The fixes are per-group scale normalisation and per-image intensity
# stretching inside the subject. Legends, scale bars and label strips along an edge
# are found as extra objects and painted over.
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..imaging import contact_sheet, largest_object, open_rgb, robust_z, subject_mask, survey
from ..project import write_json
from .base import NOTES, Stage, array, boolean, integer, number, obj, string

TARGET = 512          # the training code's image size (LiteVAE 512 px -> 4 x 64 x 64 latent)
Z_OUTLIER = 5.0     # flagged for review
Z_EXCLUDE = 8.0     # excluded by the rules: far past the tail of a large, uniform dataset
MIN_GROUP = 20      # images a group needs before its level can be estimated
DRIFT_ETA2 = 0.15   # share of a feature's variance explained by the grouping that counts as drift
SIZE_RATIO = 1.08   # and, for subject size, the largest group median over the smallest
DRIFT_FEATURES = {'size': 'subject size (px)', 'subject_mean': 'subject brightness',
                  'subject_vs_bg': 'subject-background contrast', 'bg_brightness': 'background brightness'}


def groupings(ds, stats):
    """Ways to group the images that could carry a technical batch: the source image
    dimensions, and each covariate intake wrote to line_covariates.csv."""
    g = {}
    dims = stats.width.round().astype(int).astype(str) + 'x' + stats.height.round().astype(int).astype(str)
    if dims.nunique() > 1:
        g['source_size'] = dims
    p = ds.metadata_dir / 'line_covariates.csv'
    if p.exists():
        cov = pd.read_csv(p, dtype=str).set_index('line')
        for c in cov.columns:
            g[c] = stats.line.astype(str).map(cov[c])
    return g


def subject_size(stats):
    """Linear size of the main subject in source pixels (square root of its area)."""
    frac = stats['largest_fraction'] if 'largest_fraction' in stats else stats.subject_fraction
    return np.sqrt(frac * stats.width * stats.height)


def group_effect(values, groups):
    df = pd.DataFrame({'v': np.asarray(values, float), 'g': np.asarray(groups, object)}).dropna()
    n = df.g.value_counts()
    df = df[df.g.isin(n[n >= MIN_GROUP].index)]
    if df.g.nunique() < 2:
        return None
    m = df.v.mean()
    gm = df.groupby('g').v.agg(['mean', 'median', 'size'])
    ss_t = float(((df.v - m) ** 2).sum())
    eta2 = float((gm['size'] * (gm['mean'] - m) ** 2).sum() / ss_t) if ss_t > 0 else 0.0
    lo = float(gm['median'].min())
    return {'eta2': eta2, 'groups': int(len(gm)), 'images': int(len(df)),
            'median_ratio': float(gm['median'].max() / lo) if lo > 0 else None,
            'medians': {str(k): float(v) for k, v in gm['median'].items()},
            'counts': {str(k): int(v) for k, v in gm['size'].items()}}


def drift(ds, ok):
    feats = pd.DataFrame({'size': subject_size(ok).values})
    for f in ('subject_mean', 'subject_vs_bg', 'bg_brightness'):
        if f in ok:
            feats[f] = ok[f].values
    out = []
    for gname, g in groupings(ds, ok).items():
        for f in feats.columns:
            e = group_effect(feats[f].values, g.values)
            if not e:
                continue
            flagged = e['eta2'] >= DRIFT_ETA2 and (f != 'size' or (e['median_ratio'] or 1) >= SIZE_RATIO)
            out.append({'grouping': gname, 'feature': f, 'label': DRIFT_FEATURES[f], 'flagged': bool(flagged),
                        **{k: e[k] for k in ('eta2', 'groups', 'images', 'median_ratio')},
                        '_medians': e['medians'], '_counts': e['counts']})
    out.sort(key=lambda r: -r['eta2'])
    return out


def drift_figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r['flagged']][:4]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.3 * len(rows)), dpi=100, squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        keys = sorted(r['_medians'])
        vals = [r['_medians'][k] for k in keys]
        ax.bar(range(len(keys)), vals, color='#c98a00', width=0.8)
        ax.set_title(f"{r['label']} by {r['grouping']}: {r['eta2']:.0%} of the variance lies between groups",
                     fontsize=9)
        step = max(1, len(keys) // 14)
        ax.set_xticks(range(0, len(keys), step), [keys[i] for i in range(0, len(keys), step)],
                      fontsize=7, rotation=30)
        ax.tick_params(labelsize=7)
        ax.set_ylim(min(vals) - 0.1 * abs(min(vals)), max(vals) + 0.03 * abs(max(vals)))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def scale_factors(ds, grouping):
    """Per image: the reference group's median subject size over its own group's. Groups
    smaller than MIN_GROUP cannot be calibrated and keep factor 1 (counted)."""
    if not grouping:
        return {}, None
    stats = pd.read_csv(ds.stage_dir('images') / 'survey.csv')
    stats = stats[stats.ok.astype(bool)].reset_index(drop=True)
    g = groupings(ds, stats).get(grouping)
    if g is None:
        return {}, None
    df = pd.DataFrame({'image_id': stats.image_id, 'g': g.values, 'size': subject_size(stats).values})
    n = df.g.value_counts()
    med = df[df.g.isin(n[n >= MIN_GROUP].index)].groupby('g')['size'].median()
    if med.empty:
        return {}, None
    ref = n[med.index].idxmax()
    f = (med[ref] / med).to_dict()
    df['f'] = df.g.map(f).fillna(1.0)
    main = (np.maximum(stats.main_bbox_w * stats.width, stats.main_bbox_h * stats.height)
            if 'main_bbox_w' in stats else np.maximum(stats.bbox_w * stats.width, stats.bbox_h * stats.height))
    canvas = int(math.ceil(np.nanpercentile(main * df.f, 99.5) * 1.08 / 16) * 16)
    return dict(zip(df.image_id, df.f)), {
        'grouping': grouping, 'reference_group': str(ref), 'canvas_px': canvas,
        'groups': {str(k): {'images': int(n[k]), 'median_size_px': float(med[k]), 'factor': float(f[k])}
                   for k in med.index},
        'uncalibrated_images': int((~df.g.isin(med.index)).sum())}


def _class_bg(b):
    return 'white' if b > 0.8 else 'black' if b < 0.2 else 'mid'


class Images(Stage):
    name = 'images'
    needs = ('intake',)
    effort = 'medium'

    def analyze(self, ds):
        index = pd.read_csv(ds.image_index_path)
        stats = pd.DataFrame(survey(index.source_path.tolist()))
        stats.insert(0, 'image_id', index.image_id.values)
        stats['line'] = index.line.values
        out = ds.stage_dir('images')
        out.mkdir(parents=True, exist_ok=True)
        stats.to_csv(out / 'survey.csv', index=False)

        ok = stats[stats.ok]
        bad = stats[~stats.ok]
        bgc = ok.bg_brightness.map(_class_bg)
        major_bg = bgc.mode().iat[0] if len(bgc) else 'white'
        # Batch drift first: when whole groups differ (imaging setups), an image is judged
        # against its own group, or every image of a minority setup would be an outlier.
        drift_rows = drift(ds, ok)
        by = next((r['grouping'] for r in drift_rows if r['flagged'] and r['grouping'] == 'source_size'),
                  next((r['grouping'] for r in drift_rows if r['flagged']), None))
        groups = groupings(ds, ok).get(by) if by else None
        if groups is not None:
            n = groups.value_counts()
            groups = groups.where(groups.isin(n[n >= MIN_GROUP].index), '_other')
        else:
            groups = pd.Series('_all', index=ok.index)
        # Outliers: unlike the rest of their group on any of these, or a different background class.
        feats = ['bg_brightness', 'subject_fraction', 'mean', 'colourfulness', 'subject_vs_bg', 'bg_uniformity']
        # Every feature is on a 0-1 scale; a spread below 0.02 (subject fraction 0.01)
        # is invisible, so it is not allowed to turn trace differences into outliers.
        floor = {'subject_fraction': 0.01}
        z = pd.DataFrame(index=ok.index)
        for f in feats:
            if f not in ok:
                continue
            col = ok[f].fillna(ok[f].median())
            z[f] = pd.concat([pd.Series(robust_z(col[idx], floor.get(f, 0.02)), index=idx)
                              for idx in groups.groupby(groups).groups.values()]).reindex(ok.index)
        group_bg = bgc.groupby(groups).agg(lambda s: s.mode().iat[0])
        sv_all = ok.subject_vs_bg.dropna() if 'subject_vs_bg' in ok else pd.Series(dtype=float)
        # The usual polarity, when nearly every image agrees on it; otherwise no polarity test.
        polarity = (np.sign(sv_all.median()) if len(sv_all) and (np.sign(sv_all) == np.sign(sv_all.median())).mean() > 0.97
                    else np.nan)
        reasons, severe = {}, set()
        for idx, row in ok.iterrows():
            zi = z.loc[idx]
            r = [f'{f} z={zi[f]:+.1f}' for f in z.columns if abs(zi[f]) > Z_OUTLIER]
            # Excluded by the rules only when clearly another kind of image: colour among
            # greyscale (or the reverse) far past the tail, or the opposite background
            # polarity (an inverted X-ray). Exposure and background shade are flagged for
            # review; normalisation handles them.
            if 'colourfulness' in z.columns and abs(zi['colourfulness']) > Z_EXCLUDE:
                severe.add(row.image_id)
            own = group_bg.get(groups.loc[idx], major_bg)
            if bgc.loc[idx] != own:
                r.insert(0, f'{bgc.loc[idx]} background (most are {own})')
                if {bgc.loc[idx], own} == {'white', 'black'}:
                    severe.add(row.image_id)
            # Polarity: the subject is darker than its background where almost every other
            # image has it lighter (or the reverse). Catches an inverted image whatever the
            # background shade of its group.
            sv = row.get('subject_vs_bg')
            if pd.notna(polarity) and sv is not None and pd.notna(sv) and abs(sv) > 0.4 and np.sign(sv) != polarity:
                r.insert(0, 'inverted: subject ' + ('darker' if sv < 0 else 'lighter') + ' than its background, '
                         'unlike the rest')
                severe.add(row.image_id)
            if row.get('subject_fraction', 0) < 0.003:
                r.insert(0, 'blank: no subject found')
                severe.add(row.image_id)
            if r:
                reasons[row.image_id] = '; '.join(r)
        dup = ok[ok.duplicated('thumb_hash', keep=False)]
        # The same picture twice: filed under different lines, nobody knows which line it
        # shows, so every copy goes; under one line, the extra copies go.
        for _, g in dup.groupby('thumb_hash'):
            mixed = g.line.astype(str).nunique() > 1
            for k, (_, row) in enumerate(g.iterrows()):
                if mixed or k > 0:
                    why = ('identical picture filed under different lines (' + ', '.join(map(str, g.line.unique()[:3])) + ')'
                           if mixed else f'duplicate of {g.image_id.iloc[0]}')
                    reasons[row.image_id] = why + ('; ' + reasons[row.image_id] if row.image_id in reasons else '')
                    severe.add(row.image_id)
        outliers = [{'image_id': k, 'reason': v, 'severe': k in severe} for k, v in reasons.items()]
        outliers += [{'image_id': r.image_id, 'reason': 'unreadable: ' + str(r.get('error')), 'severe': True}
                     for _, r in bad.iterrows()]

        # Orientation: an elongated subject lying at a consistent angle has an "up".
        ang = np.radians(ok.axis_angle.dropna() * 2)
        axis_consistency = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean())) if len(ang) else 0.0
        elong = float(ok.elongation.median()) if 'elongation' in ok else 1.0
        circ = float(ok.circularity.median()) if 'circularity' in ok else 0.0
        # Subject size in source pixels, to set a canvas holding every subject.
        side = np.maximum(ok.bbox_w * ok.width, ok.bbox_h * ok.height)
        main = np.maximum(ok.main_bbox_w * ok.width, ok.main_bbox_h * ok.height) if 'main_bbox_w' in ok else side
        clean = ~ok.image_id.isin(reasons)

        def canvas_of(s):
            return int(math.ceil(np.nanpercentile(s[clean], 99.5) * 1.08 / 16) * 16) if clean.any() else 0
        canvas, canvas_main = canvas_of(side), canvas_of(main)
        size_cv = float(side[clean].std() / side[clean].mean()) if clean.any() else 0.0
        edge_panels = float(ok.edge_panel.fillna(False).astype(bool).mean()) if 'edge_panel' in ok else 0.0
        single = float(ok.largest_share.median()) if 'largest_share' in ok else 1.0
        drift_fig = drift_figure(drift_rows, out / 'drift.png')

        sample = ok.sample(min(24, len(ok)), random_state=0)
        sheets = {'typical': contact_sheet(index.set_index('image_id').loc[sample.image_id].source_path,
                                           out / 'typical.png', labels=list(sample.line), title='Typical images')}
        if drift_fig:
            sheets['drift'] = drift_fig
        if outliers:
            ids = [o['image_id'] for o in outliers[:24]]
            src = index.set_index('image_id').source_path
            sheets['outliers'] = contact_sheet([src[i] for i in ids if i in src], out / 'outliers.png',
                                               labels=[reasons.get(i, 'unreadable')[:34] for i in ids if i in src],
                                               title='Flagged images')
        return {
            'n_images': int(len(stats)), 'unreadable': int(len(bad)),
            'source_size_px': {'width_median': float(ok.width.median()), 'height_median': float(ok.height.median()),
                               'width_range': [int(ok.width.min()), int(ok.width.max())],
                               'height_range': [int(ok.height.min()), int(ok.height.max())]},
            'modes': ok['mode'].value_counts().to_dict(),
            'background': {'majority': major_bg, 'counts': bgc.value_counts().to_dict(),
                           'brightness_median': float(ok.bg_brightness.median()),
                           'uniformity_median': float(ok.bg_uniformity.median())},
            'subject': {'fraction_median': float(ok.subject_fraction.median()),
                        'touches_edge_share': float(ok.touches_edge.mean()) if 'touches_edge' in ok else 0.0,
                        'circularity_median': circ, 'elongation_median': elong,
                        'axis_consistency': axis_consistency,
                        'size_px_median': float(np.nanmedian(side)), 'size_cv': size_cv,
                        'subject_darker_than_background': bool(ok.subject_vs_bg.median() < 0)},
            'colourfulness_median': float(ok.colourfulness.median()),
            'canvas_px_suggested': canvas,
            'canvas_px_main_object': canvas_main,
            'objects': {'edge_panel_share': edge_panels, 'largest_object_share_median': single,
                        'detail': 'edge panels are extra objects along an image edge: legends, scale bars, '
                                  'label strips'},
            'groupings': sorted({r['grouping'] for r in drift_rows}),
            'outliers_judged_within': by or 'all images',
            'drift': [{k: v for k, v in r.items() if not k.startswith('_')} for r in drift_rows[:20]],
            'drift_flagged': [f"{r['label']} shifts with {r['grouping']} ({r['eta2']:.0%} of its variance"
                              + (f", group medians differ up to {r['median_ratio']:.2f}x" if r['feature'] == 'size'
                                 else '') + ')' for r in drift_rows if r['flagged']],
            'duplicates': int(len(dup)),
            'n_flagged': len(outliers),
            'flagged': outliers[:60],
            '_flagged_all': outliers,
            '_figures': sheets,
        }

    def needs_agent(self, ds, report, default):
        flagged = report.get('_flagged_all', report['flagged'])
        n_severe = sum(1 for f in flagged if f.get('severe'))
        if n_severe:
            return True, f"{n_severe} images look like a different kind of image"
        if len(flagged) > 0.01 * report['n_images']:
            return True, f"{len(flagged)} images flagged for review"
        if report.get('drift_flagged'):
            return True, 'whole groups of images shift together: ' + '; '.join(report['drift_flagged'][:2])
        if (report.get('objects') or {}).get('edge_panel_share', 0) > 0.2:
            return True, 'many images carry a panel along an edge (legend, scale bar or label)'
        s = report['subject']
        if 0.3 < s['axis_consistency'] < 0.7 or report['background']['majority'] == 'mid':
            return True, 'orientation or background is borderline'
        return False, 'no flagged images, clear background and orientation'

    def digest(self, ds, report):
        d = {k: report.get(k) for k in ('n_images', 'unreadable', 'source_size_px', 'background', 'subject',
                                        'colourfulness_median', 'canvas_px_suggested', 'canvas_px_main_object',
                                        'objects', 'drift_flagged', 'duplicates', 'n_flagged')}
        d['drift_top'] = report.get('drift', [])[:6]
        d['flagged_first'] = report['flagged'][:10]
        return d

    def attachments(self, ds, report):
        figs = report.get('_figures', {})
        return [p for k, p in figs.items() if p and k in ('typical', 'outliers', 'drift')]

    def schema(self, ds, report):
        return obj({
            'placement': string('fixed_canvas keeps one physical scale (size traits survive); '
                                'fit_each scales every subject to fill the frame (erases size).',
                                enum=['fixed_canvas', 'fit_each']),
            'canvas_px': integer('Square of source pixels cut around each subject before resizing '
                                 'to 512 px (fixed_canvas).'),
            'center_on_subject': boolean(),
            'invert': boolean('Invert intensities, e.g. X-rays or fluorescence on black, so the '
                              'background becomes light.'),
            'flatten_background': boolean('Replace everything outside the subject with plain white.'),
            'colour': string(enum=['rgb', 'grey']),
            'augmentation': string('dihedral8: any rotation or flip is a valid view (cross-sections, '
                                   'cells, top-down views). hflip: there is an up (whole organisms, '
                                   'side views). none: neither.', enum=['dihedral8', 'hflip', 'none']),
            'rescale_to_um_per_px': number('Bring every image to this pixel size first (needs per-image '
                                           'pixel sizes); 0 to skip.'),
            'keep_objects': string('all: every object in the frame. largest: only the main subject; legends, '
                                   'scale bars and labels are painted over with background.',
                                   enum=['all', 'largest']),
            'normalise_scale_by': string('A grouping whose groups differ in magnification, not biology: each '
                                         "group's subjects are scaled so its median size matches the largest "
                                         'group. "" for none. Only for technical groupings (imaging setup, '
                                         'camera, date), never one that is the genotype itself.',
                                         enum=[''] + list(report.get('groupings', []))),
            'normalise_intensity': string('per_image: stretch each image between the 1st and 99.7th percentile '
                                          'of its subject pixels; removes exposure and display-style '
                                          'differences, and with them absolute brightness as a trait.',
                                          enum=['none', 'per_image']),
            'exclude': array(obj({'image_id': string(), 'reason': string()})),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        s = report['subject']
        upright = s['elongation_median'] > 1.5 and s['axis_consistency'] > 0.6
        dark_bg = report['background']['majority'] == 'black'
        flagged = report.get('_flagged_all', report['flagged'])
        # Only clear breaks: a different background, blank, unreadable, or far past the
        # tail. Big or small subjects (mean, subject_fraction) are genetic signal.
        exclude = [{'image_id': f['image_id'], 'reason': f['reason']} for f in flagged
                   if f.get('severe') and not f['reason'].startswith(('mean', 'subject_fraction'))]
        objs = report.get('objects') or {}
        keep = ('largest' if objs.get('edge_panel_share', 0) > 0.2
                or objs.get('largest_object_share_median', 0) > 0.95 else 'all')
        flagged_drift = [r for r in report.get('drift', []) if r['flagged']]
        # Scale is normalised automatically only for a change in source image dimensions, a
        # physical setup change; a size shift with a date or other covariate may be biology.
        scale_by = 'source_size' if any(r['grouping'] == 'source_size' and r['feature'] == 'size'
                                        for r in flagged_drift) else ''
        intensity = 'per_image' if any(r['feature'] != 'size' for r in flagged_drift) else 'none'
        canvas = report.get('canvas_px_main_object') if keep == 'largest' else report['canvas_px_suggested']
        return {
            'placement': 'fixed_canvas',
            'canvas_px': int(canvas or report['source_size_px']['width_median']),
            'center_on_subject': True,
            'invert': bool(dark_bg and report['colourfulness_median'] < 0.03),
            'flatten_background': report['background']['uniformity_median'] > 0.05,
            'colour': 'grey' if report['colourfulness_median'] < 0.02 else 'rgb',
            'augmentation': 'hflip' if upright else 'dihedral8',
            'rescale_to_um_per_px': 0.0,
            'keep_objects': keep,
            'normalise_scale_by': scale_by,
            'normalise_intensity': intensity,
            'exclude': exclude,
            'notes': (f"Fixed {canvas} px canvas so subject size is kept; "
                      f"{'mirror-only augmentation (subjects have an up)' if upright else 'any rotation or flip'}; "
                      f"{len(exclude)} flagged images excluded"
                      + (f'; subject size normalised per {scale_by} group' if scale_by else '')
                      + ('; intensities stretched per image' if intensity == 'per_image' else '')
                      + ('; only the main object kept' if keep == 'largest' else '') + '.'),
        }

    def prompt(self, ds, report):
        return ('Choose how every image becomes a 512 x 512 training image. Look at the typical '
                'and flagged contact sheets. Exclude images that are a different kind of image '
                '(inverted, another view, another stain, blank, corrupt); keep images that are '
                'simply large or small subjects, since that is genetic signal. If whole groups of '
                'images shift together (drift_flagged, drift.png), decide whether that is technical '
                '(imaging setup, exposure, date) and should be normalised, or biology that must stay.')

    def apply(self, ds, d):
        index = pd.read_csv(ds.image_index_path)
        excluded = {e['image_id']: e['reason'] for e in d['exclude']}
        keep = index[~index.image_id.isin(excluded)]
        out_dir = ds.scaled_dir(TARGET)
        out_dir.mkdir(parents=True, exist_ok=True)
        px = keep['pixel_size_um'] if 'pixel_size_um' in keep else pd.Series(np.nan, index=keep.index)
        factors, calib = scale_factors(ds, d.get('normalise_scale_by', ''))
        d = dict(d)
        if calib:
            d['canvas_px'] = calib['canvas_px'] or d['canvas_px']
        jobs = [(r.source_path, str(out_dir / f'{r.image_id}.png'), d,
                 float(p) if pd.notna(p) else 0.0, float(factors.get(r.image_id, 1.0)))
                for r, p in zip(keep.itertuples(), px)]
        with ProcessPoolExecutor(max_workers=_workers(index)) as ex:
            results = list(ex.map(_prepare_one, jobs, chunksize=16))
        # Failures in the pool are usually memory pressure; one retry each, one at a time.
        for k, r in enumerate(results):
            if not r['ok']:
                results[k] = _prepare_one(jobs[k])
        meta = keep[['image_id', 'line', 'relpath']].copy()
        meta.insert(0, 'scaled_filename', meta.image_id + '.png')
        meta = meta.rename(columns={'line': 'genotype'})
        meta['fits_canvas'] = [r['fits'] for r in results]
        meta['scale_factor'] = [j[4] for j in jobs]
        meta['ok'] = [r['ok'] for r in results]
        meta['error'] = [r.get('error', '') for r in results]
        failed = meta[~meta.ok]
        meta = meta[meta.ok].drop(columns=['ok', 'error'])
        meta.to_csv(ds.scaled_metadata_path(TARGET), index=False)
        pd.DataFrame([{'image_id': k, 'reason': v} for k, v in excluded.items()] +
                     [{'image_id': r.image_id, 'reason': 'failed to process: ' + str(r.error)} for r in failed.itertuples()],
                     columns=['image_id', 'reason']).to_csv(ds.stage_dir('images') / 'excluded.csv', index=False)
        sample = meta.sample(min(24, len(meta)), random_state=0)
        contact_sheet([out_dir / f for f in sample.scaled_filename], ds.stage_dir('images') / 'prepared.png',
                      labels=list(sample.genotype), title='Training images (512 px)')
        if calib:
            write_json(ds.stage_dir('images') / 'scale_groups.json', calib)
        ds.update_config(images={'target_size': TARGET, 'canvas_px': d['canvas_px'],
                                 'placement': d['placement'], 'invert': d['invert'],
                                 'colour': d['colour'], 'augmentation': d['augmentation'],
                                 'keep_objects': d.get('keep_objects', 'all'),
                                 'normalise_scale_by': d.get('normalise_scale_by', ''),
                                 'normalise_intensity': d.get('normalise_intensity', 'none')},
                         pipeline={'upright': d['augmentation'] != 'dihedral8', 'image_size': TARGET})
        n_over = int((~meta.fits_canvas).sum())
        return {'summary': f'{len(meta)} training images written, {len(excluded)} excluded'
                           + (f', {n_over} subjects larger than the canvas' if n_over else '')
                           + (f"; scale normalised over {len(calib['groups'])} {calib['grouping']} groups"
                              if calib else ''),
                'written': len(meta), 'excluded': len(excluded), 'larger_than_canvas': n_over,
                'scale_groups': calib['groups'] if calib else None}


def _workers(index):
    """Worker processes for image preparation, bounded by free memory: each holds a few
    float copies of a full-size source image (about 0.4 GB for a 3,000 px X-ray)."""
    import os
    cores = os.cpu_count() or 4
    try:
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [('len', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                        ('avail', ctypes.c_ulonglong), ('pt', ctypes.c_ulonglong), ('pa', ctypes.c_ulonglong),
                        ('vt', ctypes.c_ulonglong), ('va', ctypes.c_ulonglong), ('ve', ctypes.c_ulonglong)]
        m = MEM(); m.len = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        free_gb = m.avail / 2 ** 30
    except Exception:  # noqa: BLE001 - not Windows
        try:
            free_gb = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
        except (ValueError, OSError, AttributeError):
            free_gb = 8.0
    try:
        with Image.open(index.source_path.iloc[0]) as im:
            mp = im.size[0] * im.size[1] / 1e6
    except Exception:  # noqa: BLE001
        mp = 4.0
    per_worker = max(0.15, mp * 0.07)          # GB: float32 RGB plus working copies
    # The memory guard's count (commit memory, memory level) when it is the smaller.
    from .. import resources
    guarded = max(1, resources.workers(per_worker))
    return int(max(1, min(cores, (free_gb - 1.5) // per_worker, guarded)))


def _prepare_one(job):
    src, dst, d, px_um, factor = job
    try:
        im = open_rgb(src)
        f = factor
        if d['rescale_to_um_per_px'] and px_um:
            f *= px_um / d['rescale_to_um_per_px']
        if abs(f - 1) > 0.01:
            im = im.resize((max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32) / 255
        if d['invert']:
            a = 1 - a
        small = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (max(1, a.shape[1] // 8), max(1, a.shape[0] // 8))), dtype=np.float32) / 255
        m = subject_mask(small)
        largest = d.get('keep_objects') == 'largest'
        if largest and m.any():
            m = largest_object(m)
        bg = np.array([1.0, 1.0, 1.0]) if d['flatten_background'] else np.median(
            np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3)]), axis=0)
        full = None
        if m.any() and (d['flatten_background'] or largest or d.get('normalise_intensity') == 'per_image'):
            full = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (a.shape[1], a.shape[0]), Image.NEAREST)) > 0
        if full is not None and d.get('normalise_intensity') == 'per_image' and full.sum() > 100:
            # Stretch between the subject's own 1st and 99.7th percentiles; the background,
            # outside that range, clips to one end.
            lo, hi = np.percentile(a.mean(axis=2)[full], (1, 99.7))
            a = np.clip((a - lo) / max(hi - lo, 1e-3), 0, 1)
            if not d['flatten_background']:
                bg = np.clip((bg - lo) / max(hi - lo, 1e-3), 0, 1)
        if full is not None and (d['flatten_background'] or largest):
            a = np.where(full[..., None], a, bg)
        h, w = a.shape[:2]
        if m.any() and d['center_on_subject']:
            ys, xs = np.nonzero(m)
            cy, cx = (ys.min() + ys.max() + 1) * 4, (xs.min() + xs.max() + 1) * 4
            extent = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 8
        else:
            cy, cx, extent = h // 2, w // 2, max(h, w)
        if d['placement'] == 'fixed_canvas':
            side = int(d['canvas_px'])
            fits = extent <= side
            if not fits:
                side = int(extent * 1.05)       # never cut the subject; this image's scale differs
        else:
            side, fits = int(extent * 1.08), True
        canvas = np.ones((side, side, 3), dtype=np.float32) * bg
        y0, x0 = cy - side // 2, cx - side // 2
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(h, y0 + side), min(w, x0 + side)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
        out = Image.fromarray((np.clip(canvas, 0, 1) * 255).round().astype(np.uint8)).resize(
            (TARGET, TARGET), Image.LANCZOS)
        if d['colour'] == 'grey':
            out = out.convert('L').convert('RGB')
        out.save(dst)
        return {'ok': True, 'fits': bool(fits)}
    except Exception as e:  # noqa: BLE001 - counted and listed in excluded.csv
        return {'ok': False, 'fits': False, 'error': f'{type(e).__name__}: {e}'[:200]}
