# Statistical analysis: how well generated images carry each line's traits.
#
# Mostly code, little AI. For the chosen checkpoint (the QC stage's
# recommendation, else the best-validation-loss one):
#   1. generate n seeds for every validation line (the trainer's own validation
#      lines, recorded in the checkpoint); test lines are only generated when the
#      decision says this is the final check, the same rule as the RIL results;
#   2. measure every generated image with the traits recipe;
#   3. per trait: correlation of generated line means with the truth (imported
#      traits, or the same measurement on real images), the spread ratio, and a
#      genomic-prediction baseline (ridge on all markers, trained on training lines'
#      real values: the GBLUP equivalent);
#   4. write trait_accuracy.csv, scatter_points.csv and a scatter figure, the same
#      layout as RIL_v5_results/01_best_model_accuracy/, so the existing analysis
#      scripts and the dashboard read them directly.
import numpy as np
import pandas as pd

from ..generate import find_checkpoint, generate
from ..measure import measure_paths
from .base import NOTES, Stage, array, boolean, obj, string
from .genetics import load_genotypes, standardise

N_SEEDS = 4


def truth_table(ds):
    cfg = ds.config()
    tr = cfg.get('traits') or {}
    traits = tr.get('traits', [])
    real = pd.read_csv(ds.results_dir / 'traits' / 'real_line_means.csv', index_col=0)
    real.index = real.index.astype(str)
    if tr.get('score_against') == 'imported' and ds.traits_path.exists():
        imp = pd.read_csv(ds.traits_path)
        imp['line'] = imp['line'].astype(str)
        cols = [c for c in imp.columns if c not in ('line', 'image_id')]
        imp = imp.groupby('line')[cols].mean()
        # A trait with an imported match is scored against it; one without is scored
        # against the same measurement on the real images, rather than dropped.
        own = [t['measured_name'] for t in traits if not t['imported_match'] and t['measured_name'] in real]
        truth = imp.join(real[own].add_prefix('real: '), how='outer')
        pairs = [(t['measured_name'], t['imported_match'] or f"real: {t['measured_name']}") for t in traits]
        return real, truth, pairs
    return real, real, [(t['measured_name'], t['measured_name']) for t in traits]


def evaluate(ds, role, n_seeds=N_SEEDS, checkpoint=None, progress=None):
    ckpt = checkpoint or find_checkpoint(ds, (ds.config().get('qc') or {}).get('recommended_checkpoint', ''))
    if ckpt is None:
        raise SystemExit('no trained diffusion checkpoint in the workspace yet (models/diffusion_<dataset>*/)')
    out = ds.results_dir / 'analysis'
    gen_dir = out / 'generated' / role
    index = generate(ds, ckpt, role, n_seeds, gen_dir, progress=progress)
    recipe = (ds.config().get('traits') or {}).get('recipe')
    measured_path = out / f'generated_traits_{role}.csv'
    have = pd.read_csv(measured_path) if measured_path.exists() else pd.DataFrame(columns=['image'])
    todo = index[~index.image.isin(have.image)]
    if len(todo):
        new = measure_paths([gen_dir / f for f in todo.image], recipe)
        have = pd.concat([have, new.merge(todo, on='image')], ignore_index=True)
        have.to_csv(measured_path, index=False)
    gen = have.merge(index[['image', 'line', 'seed']], on=['image'], suffixes=('', '_i')) if 'line' not in have else have
    gen['line'] = gen['line'].astype(str)
    real, truth, pairs = truth_table(ds)
    baseline = genomic_baseline(ds, truth, [t for _, t in pairs if t], set(gen.line))
    labels = {x['measured_name']: x.get('label') or x['measured_name']
              for x in (ds.config().get('traits') or {}).get('traits', [])}
    rows, points = [], []
    for measured, truth_col in pairs:
        if measured not in gen or not truth_col or truth_col not in truth:
            continue
        g = gen.groupby('line')[measured].mean()
        both = pd.DataFrame({'generated': g, 'truth': truth[truth_col]}).dropna()
        b = baseline.get(truth_col)
        r_b = float(pd.Series(b).reindex(both.index).corr(both.truth)) if b is not None else np.nan
        rows.append({'trait': measured, 'label': labels.get(measured, measured), 'truth_column': truth_col,
                     'n_lines': len(both),
                     'r_generated': float(both.generated.corr(both.truth)) if len(both) > 2 else np.nan,
                     'r_genomic_prediction': r_b,
                     'sd_ratio': float(both.generated.std() / both.truth.std()) if both.truth.std() else np.nan})
        for line, rr in both.iterrows():
            points.append({'trait': measured, 'line': line, 'truth': rr.truth, 'generated_mean': rr.generated,
                           'genomic_prediction': (b or {}).get(line, np.nan)})
    acc = pd.DataFrame(rows)
    acc.to_csv(out / f'trait_accuracy_{role}.csv', index=False)
    pd.DataFrame(points).to_csv(out / f'scatter_points_{role}.csv', index=False)
    fig = _scatter(pd.DataFrame(points), acc, out / f'scatter_{role}.png') if points else None
    extra = extended(ds, role, ckpt, gen, gen_dir, truth, pairs, acc, out) if len(acc) else {}
    # Genes behind each trait (GWAS on training lines, or the dataset's ground truth) and
    # where they act in the generated image (d2d/gene_maps.py). Validation lines only.
    figs = {'scatter': str(fig)} if fig else {}
    if role == 'validation' and len(acc):
        from .. import gene_maps
        try:
            extra.update(gene_maps.run(ds, ckpt, role))
            figs.update({k: str(out / f'{k}_{role}.png') for k in ('gwas', 'gene_maps') if (out / f'{k}_{role}.png').exists()})
        except Exception as e:  # noqa: BLE001 - the accuracy results still stand
            import traceback
            ds.log('analysis', 'gene maps failed: ' + traceback.format_exc(), level='warning')
            extra['gene_maps_error'] = f'{type(e).__name__}: {e}'
    return {'checkpoint': ckpt.name, 'role': role, 'lines': int(gen.line.nunique()), 'images': int(len(gen)),
            'traits': acc.to_dict('records'),
            'mean_r_generated': float(acc.r_generated.mean()) if len(acc) else None,
            'mean_r_genomic_prediction': float(acc.r_genomic_prediction.mean()) if len(acc) else None,
            **extra,
            '_figures': figs}


def same_units(truth_col, measured):
    """Truth and generated values share units when the truth is the same measurement on
    real images; an imported trait (mm, counts...) is compared on standardised scales."""
    return truth_col == measured or truth_col == f'real: {measured}'


def extended(ds, role, ckpt, gen, gen_dir, truth, pairs, acc, out):
    """Reliability, distributions, line matching, memorisation and covariate baselines
    (d2d/analysis_extra.py). Writes extras_<role>.json plus one CSV per table and returns
    the headline numbers for the stage report."""
    import json
    from .. import analysis_extra as X
    from ..project import _clean
    measured = [m for m, _ in pairs if m in gen]
    target = sorted(set(gen.line))
    rel = X.reliability(gen, measured)
    dist, hist = pd.DataFrame(), {}
    rip = ds.results_dir / 'traits' / 'real_image_traits.csv'
    if rip.exists():
        ri = pd.read_csv(rip)
        ri['line'] = ri.line.astype(str)
        dist, hist = X.distributions(gen, ri[ri.line.isin(target)], measured)
    gm = gen.groupby('line')[measured].mean()
    tm = pd.DataFrame({m: truth[tc] for m, tc in pairs if tc in truth and m in gm})
    match = X.line_matching(gm[list(tm.columns)], tm) if len(tm.columns) else {}
    split = pd.read_csv(ds.split_path)
    held = set(split.line[split.split_random == 'test'].astype(str))
    held |= {p.stem.rsplit('_seed', 1)[0] for p in (out / 'generated' / 'validation').glob('*_seed0.png')}
    mem = X.memorisation(ds, ckpt.name, sorted(gen_dir.glob('*_seed*.png')), target, held,
                         out / f'memorisation_pairs_{role}.png')
    base_df, cov_cols = X.covariate_baselines(ds, truth, pairs, target, dict(zip(acc.trait, acc.r_generated)))
    for name, df in (('reliability', rel), ('distributions', dist), ('baselines', base_df)):
        if len(df):
            df.to_csv(out / f'{name}_{role}.csv', index=False)
    (out / f'extras_{role}.json').write_text(json.dumps(_clean({
        'reliability': rel.to_dict('records'), 'distributions': dist.to_dict('records'), 'histograms': hist,
        'line_matching': match, 'memorisation': mem, 'baselines': base_df.to_dict('records'),
        'covariates': cov_cols, 'same_units': {m: same_units(tc, m) for m, tc in pairs}})))
    return {'memorisation': mem.get('verdict'),
            'line_matching_top1': match.get('top1'), 'line_matching_chance': match.get('chance_top1'),
            'mean_reliability_line_mean': float(rel.reliability_line_mean.mean()) if len(rel) else None,
            'covariates_in_genotype_table': len(cov_cols)}


def genomic_baseline(ds, truth, cols, target_lines):
    """Ridge on all markers, fitted on training lines, predicting target lines."""
    from sklearn.linear_model import RidgeCV
    lines, _, x = load_genotypes(ds)
    split = pd.read_csv(ds.split_path)
    test = set(split.line[split.split_random == 'test'].astype(str))
    z, _ = standardise(x)
    idx = {l: i for i, l in enumerate(lines)}
    out = {}
    for c in cols:
        if c not in truth:
            continue
        y = truth[c].dropna()
        fit = [l for l in y.index if l in idx and l not in test and l not in target_lines]
        tgt = [l for l in target_lines if l in idx]
        if len(fit) < 30 or not tgt:
            continue
        m = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit(z[[idx[l] for l in fit]], y[fit].to_numpy())
        out[c] = dict(zip(tgt, m.predict(z[[idx[l] for l in tgt]])))
    return out


def _scatter(points, acc, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traits = list(acc.trait)
    n = len(traits)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), dpi=110, squeeze=False)
    z = lambda s: (s - s.mean()) / s.std()
    for ax, (_, a) in zip(axes.ravel(), acc.iterrows()):
        p = points[points.trait == a.trait]
        raw = same_units(a.truth_column, a.trait)
        x, yb, yg = (p.truth, p.genomic_prediction, p.generated_mean) if raw else \
            (z(p.truth), z(p.genomic_prediction), z(p.generated_mean))
        ax.scatter(x, yb, s=10, c='#9a9a9a', label='genomic prediction')
        ax.scatter(x, yg, s=12, c='#c98a00', edgecolor='#7a5c00', lw=0.3, label='generated')
        lo, hi = np.nanmin([x.min(), yb.min(), yg.min()]), np.nanmax([x.max(), yb.max(), yg.max()])
        ax.plot([lo, hi], [lo, hi], ls='--', lw=0.8, c='#444444', zorder=0, label='ideal (y = x)')
        ax.set_title(f"{a.trait}\nr generated {a.r_generated:.2f} | GP {a.r_genomic_prediction:.2f}", fontsize=8)
        ax.set_xlabel('truth' if raw else 'truth (standardised)', fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


class Analysis(Stage):
    name = 'analysis'
    needs = ('traits', 'training')
    effort = 'low'

    def analyze(self, ds):
        return evaluate(ds, 'validation')

    def needs_agent(self, ds, report, default):
        return False, 'the numbers speak for themselves; ask for a write-up with --force-agent'

    def digest(self, ds, report):
        return {k: v for k, v in report.items() if not k.startswith('_')}

    def schema(self, ds, report):
        return obj({
            'headline': string('One sentence: how well the generated images carry the traits.'),
            'evaluate_test_lines': boolean('Generate and score the sealed test lines now. Only for the '
                                           'final check of an already-chosen model.'),
            'per_trait': array(obj({'trait': string(), 'comment': string()})),
            'caveats': array(string()),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        mg, mb = report.get('mean_r_generated'), report.get('mean_r_genomic_prediction')
        head = (f"Mean r {mg:.2f} over {len(report['traits'])} traits on {report['lines']} validation lines "
                f"(genomic prediction {mb:.2f})." if mg is not None and mb is not None else 'No traits scored.')
        return {'headline': head, 'evaluate_test_lines': False,
                'per_trait': [{'trait': t['trait'], 'comment': f"r {t['r_generated']:.2f} vs GP {t['r_genomic_prediction']:.2f}"
                               if t['r_generated'] is not None and t['r_genomic_prediction'] is not None else ''}
                              for t in report['traits']],
                'caveats': ['Validation lines only; the test lines stay sealed until the final check.'],
                'notes': head}

    def prompt(self, ds, report):
        return 'Summarise the accuracy of the generated images for the user, trait by trait.'

    def apply(self, ds, d):
        result = {'summary': d['headline']}
        if d['evaluate_test_lines']:
            test = evaluate(ds, 'test')
            result['test'] = {k: v for k, v in test.items() if not k.startswith('_')}
            result['summary'] += f" Test lines: mean r {test['mean_r_generated']:.2f}."
        return result
