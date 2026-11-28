# Segmentation and traits: decide what to measure on every image.
#
# Generated images can only be scored by measuring them, so the pipeline always
# needs an image measurement recipe (measure.py), even when the dataset comes
# with traits. Imported traits then do two jobs: they are the truth the
# generated images are compared with, and they validate the recipe (a measured
# trait that tracks an imported one is measuring the right thing).
#
# Analysis runs a broad trial recipe on a sample of training images and reports
# for every candidate measurement how much it varies, how repeatable it is within
# a line (when lines have several images: the between-line share of variance),
# and how well it tracks each imported trait. It draws overlays of the measured
# regions so the agent can see whether "inner region" is a stele or a shadow.
# The decision picks the recipe, names the traits, and says what generated
# images are scored against.
import numpy as np
import pandas as pd

from ..imaging import contact_sheet
from ..measure import DEFAULT_RECIPE, measure_paths, overlay
from .base import NOTES, Stage, array, boolean, integer, number, obj, string

TRIAL_RECIPE = {
    'subject_tolerance': 0.06, 'shape': True, 'intensity': True,
    'inner': {'enabled': True, 'polarity': 'dark'},
    'blobs': [{'name': 'dark_small', 'polarity': 'dark', 'min_area_px': 4, 'max_area_px': 120, 'contrast': 0.10},
              {'name': 'dark_large', 'polarity': 'dark', 'min_area_px': 120, 'max_area_px': 2500, 'contrast': 0.10},
              {'name': 'bright_small', 'polarity': 'bright', 'min_area_px': 4, 'max_area_px': 120, 'contrast': 0.10}],
    'plugin': 'none', 'um_per_px': 0.0,
}
SAMPLE = 300


def between_line_share(df, col):
    """Share of a trait's variance that lies between lines (needs replicate images)."""
    g = df.groupby('line')[col]
    if (g.size() > 1).sum() < 5:
        return np.nan
    total = df[col].var()
    within = g.var().mean()
    return float(max(0.0, 1 - within / total)) if total else np.nan


def _share(c):
    v = c.get('between_line_share')
    return None if v is None or np.isnan(v) else v


class Traits(Stage):
    name = 'traits'
    needs = ('images',)
    effort = 'medium'

    def analyze(self, ds):
        meta = pd.read_csv(ds.scaled_metadata_path())
        scaled = ds.scaled_dir()
        sample = meta.sample(min(SAMPLE, len(meta)), random_state=0)
        cfg = ds.config()
        um = _scaled_um_per_px(cfg)
        trial = dict(TRIAL_RECIPE, um_per_px=um)
        m = measure_paths([scaled / f for f in sample.scaled_filename], trial)
        m.insert(0, 'line', sample.genotype.values)
        out = ds.stage_dir('traits')
        out.mkdir(parents=True, exist_ok=True)
        m.to_csv(out / 'trial_measurements.csv', index=False)

        imported = pd.read_csv(ds.traits_path) if ds.traits_path.exists() else None
        imp_cols = [c for c in (imported.columns if imported is not None else []) if c not in ('line', 'image_id')]
        line_means = m.groupby('line').mean(numeric_only=True)
        if imported is not None and 'image_id' in imported:
            imp_line = imported.groupby('line')[imp_cols].mean()
        elif imported is not None:
            imp_line = imported.groupby('line')[imp_cols].mean()
        else:
            imp_line = None

        candidates = []
        for c in [c for c in m.columns if c not in ('line', 'image')]:
            v = m[c].astype(float)
            row = {'trait': c, 'mean': float(v.mean()), 'cv': float(v.std() / abs(v.mean())) if v.mean() else np.nan,
                   'missing': float(v.isna().mean()), 'between_line_share': between_line_share(m, c)}
            if imp_line is not None:
                both = line_means[[c]].join(imp_line, how='inner')
                if len(both) >= 10:
                    r = {t: float(both[c].corr(both[t])) for t in imp_cols}
                    best = max(r, key=lambda t: abs(r[t]) if not np.isnan(r[t]) else -1)
                    row['best_match'] = best
                    row['r_with_best_match'] = r[best]
            candidates.append(row)

        ov = []
        for f in sample.scaled_filename[:12]:
            ov.append(overlay(scaled / f, trial, out / f'overlay_{f}'))
        sheet = contact_sheet(ov, out / 'overlays.png', title='Trial recipe: subject (gold), inner (blue), blobs (red)')
        for p in ov:
            __import__('os').remove(p)

        imported_summary = None
        if imported is not None:
            imported_summary = [{'trait': c, 'n': int(imported[c].notna().sum()), 'mean': float(imported[c].mean()),
                                 'sd': float(imported[c].std())} for c in imp_cols]
        domain = _domain_guess(ds)
        return {
            'n_measured': int(len(m)), 'um_per_px': um, 'images_per_line': float(meta.groupby('genotype').size().mean()),
            'imported_traits': imported_summary, 'imported_level': ('image' if imported is not None and 'image_id' in imported
                                                                    else 'line' if imported is not None else 'none'),
            'candidates': candidates, 'domain_plugin_available': domain,
            '_figures': {'overlays': sheet},
        }

    def needs_agent(self, ds, report, default):
        if report['domain_plugin_available'] != 'none':
            return True, 'a domain segmenter could apply; the agent should confirm the image type'
        return True, 'choosing meaningful traits needs a look at the overlays'

    def digest(self, ds, report):
        keep = ('trait', 'cv', 'between_line_share', 'best_match', 'r_with_best_match')
        return {'n_measured': report['n_measured'], 'um_per_px': report['um_per_px'],
                'images_per_line': report['images_per_line'], 'imported_level': report['imported_level'],
                'imported_traits': [t['trait'] for t in (report['imported_traits'] or [])],
                'domain_plugin_available': report['domain_plugin_available'],
                'candidates': [{k: (round(c[k], 3) if isinstance(c.get(k), float) else c.get(k)) for k in keep if k in c}
                               for c in report['candidates']]}

    def attachments(self, ds, report):
        f = report.get('_figures', {}).get('overlays')
        return [f] if f else []

    def schema(self, ds, report):
        blob = obj({'name': string(), 'polarity': string(enum=['dark', 'bright']),
                    'min_area_px': integer(), 'max_area_px': integer(), 'contrast': number()})
        return obj({
            'recipe': obj({
                'subject_tolerance': number(), 'shape': boolean(), 'intensity': boolean(),
                'inner': obj({'enabled': boolean(), 'polarity': string(enum=['dark', 'bright'])}),
                'blobs': array(blob), 'plugin': string(enum=['none', 'maize_root_cross_section']),
            }),
            'traits': array(obj({'measured_name': string('Column the recipe produces, e.g. inner_diameter_um.'),
                                 'label': string('Readable name, e.g. "Stele diameter".'),
                                 'unit': string(),
                                 'imported_match': string('Imported trait it corresponds to, "" if none.')}),
                            'The traits to report and score, most important first (at most 12).'),
            'score_against': string('imported: compare generated line means with the imported traits '
                                    '(the truth); measured: with the same measurement on real images.',
                                    enum=['imported', 'measured']),
            'needs_custom_segmenter': boolean('True if the important structures cannot be measured '
                                              'with this recipe and need an annotated segmentation model.'),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        cands = report['candidates']
        # between_line_share is missing (None after the JSON round trip) with one image per line.
        share = _share
        good = [c for c in cands if c['missing'] < 0.1 and (c['cv'] or 0) > 0.03
                and (share(c) is None or share(c) > 0.2)]
        has_imported = any('r_with_best_match' in c for c in cands)
        if has_imported:
            good.sort(key=lambda c: -abs(c.get('r_with_best_match') or 0))
        else:
            good.sort(key=lambda c: -(share(c) or 0))
        # One measurement per underlying trait: with imported traits, the best match for
        # each; otherwise skip any measurement that is 0.95-correlated with one already kept.
        trial = pd.read_csv(ds.stage_dir('traits') / 'trial_measurements.csv')
        chosen, matched = [], set()
        for c in good:
            if has_imported and abs(c.get('r_with_best_match') or 0) > 0.5:
                if c['best_match'] in matched:
                    continue
                matched.add(c['best_match'])
            elif any(abs(trial[c['trait']].corr(trial[k['trait']])) > 0.95 for k in chosen):
                continue
            chosen.append(c)
            if len(chosen) == 12:
                break
        traits = [{'measured_name': c['trait'], 'label': c['trait'].replace('_', ' '),
                   'unit': 'um' if c['trait'].endswith('_um') else 'um2' if c['trait'].endswith('_um2')
                   else 'px' if c['trait'].endswith('_px') else '',
                   'imported_match': c.get('best_match', '') if abs(c.get('r_with_best_match') or 0) > 0.5 else ''}
                  for c in chosen]
        recipe = {k: TRIAL_RECIPE[k] for k in ('subject_tolerance', 'shape', 'intensity', 'inner', 'blobs')}
        recipe['plugin'] = report['domain_plugin_available']
        return {'recipe': recipe, 'traits': traits,
                'score_against': 'imported' if report['imported_traits'] else 'measured',
                'needs_custom_segmenter': False,
                'notes': f'Kept {len(traits)} measurements that vary between images and are not missing.'}

    def prompt(self, ds, report):
        return ('Choose what to measure on every image, real and generated. Look at the overlays: '
                'keep measurements whose regions outline real structures, drop ones that trace '
                'noise or background. Prefer traits that vary between lines and that track the '
                'imported traits when there are any. Give each kept trait a readable label.')

    def apply(self, ds, d):
        meta = pd.read_csv(ds.scaled_metadata_path())
        recipe = dict(d['recipe'], um_per_px=_scaled_um_per_px(ds.config()))
        m = measure_paths([ds.scaled_dir() / f for f in meta.scaled_filename], recipe)
        m.insert(0, 'line', meta.genotype.values)
        out = ds.results_dir / 'traits'
        out.mkdir(parents=True, exist_ok=True)
        m.to_csv(out / 'real_image_traits.csv', index=False)
        cols = [t['measured_name'] for t in d['traits'] if t['measured_name'] in m]
        m.groupby('line')[cols].mean().to_csv(out / 'real_line_means.csv')
        ds.update_config(traits={'recipe': recipe, 'traits': d['traits'], 'score_against': d['score_against']})
        return {'summary': f'{len(cols)} traits measured on {len(m)} images',
                'traits': cols, 'images': len(m)}


def _scaled_um_per_px(cfg):
    """Pixel size of the 512 px training images, if the source pixel size is known."""
    src = (cfg.get('pipeline') or {}).get('pixel_size_um')
    canvas = (cfg.get('images') or {}).get('canvas_px')
    if src and canvas and cfg.get('images', {}).get('placement') == 'fixed_canvas':
        return float(src) * canvas / 512
    return 0.0


def _domain_guess(ds):
    """'maize_root_cross_section' when the repo's root segmenter exists and the
    images look like round cross-sections; 'none' otherwise."""
    from ..project import PROJECT_ROOT as ROOT
    rep = ds.report('images') or {}
    s = rep.get('subject', {})
    seg = ROOT / 'models' / 'feature_segmentation' / 'feature_segmentation_best.pt'
    if seg.exists() and s.get('circularity_median', 0) > 0.6 and s.get('elongation_median', 9) < 1.3:
        return 'maize_root_cross_section'
    return 'none'
