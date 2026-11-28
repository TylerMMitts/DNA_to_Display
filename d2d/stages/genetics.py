# The genetic encoding stage: how the genotype matrix becomes the model's gene tokens.
#
# The trainer (code/widiv/train_widiv.py) offers three encodings:
#   pca       standardised dosages -> PCA to a variance target -> MLP -> 8 tokens.
#             Good for diverse panels where traits are spread over many loci.
#   windows   a few PCs for relatedness plus gated windows of neighbouring markers
#             (fixed size or LD blocks), so one locus of large effect keeps its own
#             signal instead of being averaged into a PC.
#   qtl_only  named columns read directly (standardised). For sparse designs such
#             as knockout panels (one gene per line), or very few markers.
#
# Analysis measures what decides between them: marker count, missingness, minor
# allele frequencies, sparsity (one non-zero column per line means a knockout or
# introgression design), LD decay, the PCA variance spectrum and, when traits
# exist, how well each encoding's features predict the traits in 5-fold
# cross-validation on the training lines (ridge regression: a cheap proxy for
# how much trait signal each encoding keeps).
import numpy as np
import pandas as pd

from ..project import write_json
from .base import NOTES, Stage, integer, number, obj, string

CV_FOLDS = 5
MAX_TRAITS = 8


def load_genotypes(ds):
    cache = ds.state_dir / 'genotypes.npz'
    src = ds.genotypes_path
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        c = np.load(cache, allow_pickle=False)
        return list(c['lines']), list(c['snps']), c['x']
    df = pd.read_csv(src, index_col=0)
    x = df.to_numpy(dtype=np.int8)
    lines, snps = [str(i) for i in df.index], [str(c) for c in df.columns]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, lines=np.array(lines), snps=np.array(snps), x=x)
    return lines, snps, x


def standardise(x):
    xf = x.astype(np.float32)
    miss = xf < 0
    xf[miss] = np.nan
    mean = np.nanmean(xf, axis=0)
    xf = np.where(miss, mean[None], xf)
    sd = xf.std(axis=0)
    keep = sd > 0
    return (xf[:, keep] - mean[keep]) / sd[keep], keep


def cv_r(features, y, folds=CV_FOLDS, seed=0):
    from sklearn.linear_model import RidgeCV
    ok = ~np.isnan(y)
    f, y = features[ok], y[ok]
    if len(y) < 30:
        return np.nan
    idx = np.random.default_rng(seed).permutation(len(y))
    pred = np.empty(len(y))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(f[train], y[train])
        pred[test] = m.predict(f[test])
    return float(np.corrcoef(pred, y)[0, 1])



def _num(v, missing=np.nan):
    """A report number; NaN is stored as null in JSON and comes back as None."""
    return missing if v is None or (isinstance(v, float) and np.isnan(v)) else v

class Genetics(Stage):
    name = 'genetics'
    needs = ('intake',)
    effort = 'low'

    def analyze(self, ds):
        lines, snps, x = load_genotypes(ds)
        out = ds.stage_dir('genetics')
        out.mkdir(parents=True, exist_ok=True)
        n, p = x.shape
        miss = float((x < 0).mean())
        xf = np.where(x < 0, np.nan, x).astype(np.float32)
        af = np.nanmean(xf, axis=0) / 2
        maf = np.minimum(af, 1 - af)
        nonzero_per_line = (x > 0).sum(axis=1)
        sparse = bool(np.median(nonzero_per_line) <= 3 and p > 20)
        z, keep = standardise(x)

        # PCA spectrum on the standardised matrix.
        k = int(min(n - 1, z.shape[1], 400))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k, svd_solver='randomized', random_state=0).fit(z)
        cum = np.cumsum(pca.explained_variance_ratio_)
        total = float(cum[-1])
        pcs_for = {f'{int(t * 100)}%': int(np.searchsorted(cum, t) + 1) if cum[-1] >= t else None
                   for t in (0.5, 0.8, 0.9, 0.95)}

        # LD decay: r^2 between markers a given number of positions apart.
        snp_map = pd.read_csv(ds.snp_map_path) if ds.snp_map_path.exists() else None
        ld = []
        rng = np.random.default_rng(0)
        for lag in (1, 2, 5, 10, 25, 50, 100):
            if lag >= z.shape[1]:
                break
            j = rng.choice(z.shape[1] - lag, size=min(1500, z.shape[1] - lag), replace=False)
            a, b = z[:, j], z[:, j + lag]
            r = (a * b).mean(axis=0)
            ld.append({'lag_markers': lag, 'median_r2': float(np.median(r ** 2))})

        # Cross-validated prediction of traits from each encoding (training lines only).
        split = pd.read_csv(ds.split_path) if ds.split_path.exists() else None
        test = set(split.line[split.split_random == 'test'].astype(str)) if split is not None else set()
        traits = _line_traits(ds)
        cv = []
        if traits is not None:
            rows = [i for i, l in enumerate(lines) if l not in test and l in traits.index]
            if len(rows) >= 30:
                zt = z[rows]
                feats = {'all_markers': zt}
                pcs = pca.transform(zt)
                for lab, t in (('pca_80', 0.8), ('pca_95', 0.95)):
                    kk = pcs_for.get(f'{int(t * 100)}%') or pcs.shape[1]
                    feats[lab] = pcs[:, :kk]
                w = 200
                if zt.shape[1] >= 4 * w:
                    wins = [zt[:, s:s + w].mean(axis=1) for s in range(0, zt.shape[1], w)]
                    feats['windows_200'] = np.column_stack([pcs[:, :32]] + [np.column_stack(wins)])
                for trait in traits.columns[:MAX_TRAITS]:
                    y = traits.loc[[lines[i] for i in rows], trait].to_numpy(dtype=float)
                    cv.append({'trait': trait, **{k2: round(cv_r(f, y), 3) for k2, f in feats.items()}})
            pd.DataFrame(cv).to_csv(out / 'encoding_cv.csv', index=False)

        figs = _figures(out, cum, ld)
        return {
            'n_lines': n, 'n_markers': p, 'missing_rate': round(miss, 4),
            'maf_quantiles': {q: round(float(np.nanquantile(maf, q)), 3) for q in (0.1, 0.5, 0.9)},
            'monomorphic': int((~keep).sum()),
            'nonzero_columns_per_line_median': float(np.median(nonzero_per_line)),
            'sparse_design': sparse,
            'pca_variance_captured_by_400': round(total, 3), 'pcs_for_variance': pcs_for,
            'ld_decay': ld, 'has_positions': bool(snp_map is not None and snp_map['pos'].nunique() > 1
                                                  and (snp_map['chr'] != 0).any()),
            'trait_cv': cv,
            '_figures': figs,
        }

    def needs_agent(self, ds, report, default):
        if report['sparse_design'] or report['n_markers'] <= 500:
            return False, 'sparse or very small marker set: the encoding is clear'
        cv = report['trait_cv']
        if cv:
            gap = np.nanmean([_num(c.get('windows_200')) - max(_num(c.get('pca_80'), -1), _num(c.get('pca_95'), -1))
                              for c in cv])
            if not np.isnan(gap) and abs(gap) > 0.05:
                return False, f'cross-validation clearly favours one encoding (gap {gap:+.2f})'
        return True, 'the encodings are close or there are no traits to compare them on'

    def digest(self, ds, report):
        return {k: v for k, v in report.items() if not k.startswith('_')}

    def schema(self, ds, report):
        return obj({
            'encoding': string(enum=['pca', 'windows', 'qtl_only']),
            'pca_variance': number('PCA variance target, e.g. 0.8 or 0.95 (pca; also the relatedness '
                                   'PCs of windows).'),
            'window_size': integer('Markers per window (windows, fixed windows).'),
            'window_pcs': integer('Relatedness PCs alongside the windows.'),
            'window_sparsity': number('Gate sparsity penalty, e.g. 1e-6.'),
            'window_ld_r2': number('0 for fixed windows; e.g. 0.5 for LD-block windows (needs marker '
                                   'positions).'),
            'notes': NOTES,
        })

    def default_decision(self, ds, report):
        base = {'pca_variance': 0.95, 'window_size': 200, 'window_pcs': 32, 'window_sparsity': 1e-6,
                'window_ld_r2': 0.0}
        if report['sparse_design'] or report['n_markers'] <= 500:
            return dict(base, encoding='qtl_only',
                        notes='Each line differs at only a few columns (or there are few markers), so '
                              'every column is read directly instead of through a PCA.')
        cv = report['trait_cv']
        if cv:
            win = np.nanmean([_num(c.get('windows_200')) for c in cv])
            p80 = np.nanmean([_num(c.get('pca_80')) for c in cv])
            p95 = np.nanmean([_num(c.get('pca_95')) for c in cv])
            allm = np.nanmean([_num(c.get('all_markers')) for c in cv])
            if not np.isnan(win) and win > max(p80, p95) + 0.03:
                return dict(base, encoding='windows',
                            notes=f'Windows predict traits better in cross-validation (r {win:.2f} vs '
                                  f'PCA {max(p80, p95):.2f}).')
            target = 0.8 if p80 >= allm - 0.02 else 0.95
            return dict(base, encoding='pca', pca_variance=target,
                        notes=f'PCA at {target:.0%} variance keeps the trait signal (r {max(p80, p95):.2f}, '
                              f'all markers {allm:.2f}).')
        strong_ld = report['ld_decay'] and report['ld_decay'][0]['median_r2'] > 0.5
        if strong_ld and report['n_markers'] > 20000:
            return dict(base, encoding='windows', window_ld_r2=0.5 if report['has_positions'] else 0.0,
                        notes='Many markers in strong LD and no traits to compare on: LD-block windows.')
        return dict(base, encoding='pca', pca_variance=0.9,
                    notes='No traits to compare encodings on; PCA at 90% variance is the safe default.')

    def prompt(self, ds, report):
        return ('Choose the genetic encoding for the diffusion model from the genotype statistics '
                'and, when present, the cross-validated trait prediction of each encoding.')

    def apply(self, ds, d):
        env = {'WIDIV_SNP_ENCODING': d['encoding']}
        if d['encoding'] in ('pca', 'windows'):
            env['WIDIV_PCA_VARIANCE'] = str(d['pca_variance'])
        if d['encoding'] == 'windows':
            env.update(WIDIV_WINDOW_SIZE=str(d['window_size']), WIDIV_WINDOW_PCS=str(d['window_pcs']),
                       WIDIV_WINDOW_SPARSITY=str(d['window_sparsity']))
            if d['window_ld_r2']:
                env['WIDIV_WINDOW_LD_R2'] = str(d['window_ld_r2'])
        if d['encoding'] == 'qtl_only':
            # Every column is read directly: list them as the "QTL" table.
            _, snps, _ = load_genotypes(ds)
            cols = ds.genetic_dir / 'encoder_columns.csv'
            pd.DataFrame({'snp': snps}).to_csv(cols, index=False)
            env['WIDIV_QTL_TABLE'] = str(cols.relative_to(ds.root)).replace('\\', '/')
        ds.update_config(genetics={'encoding': d['encoding'], 'env': env},
                         pipeline={'pca_variance': d['pca_variance']})
        return {'summary': f"encoding {d['encoding']}", 'env': env}


def _line_traits(ds):
    """Line means of the traits to compare encodings on: imported, else measured."""
    if ds.traits_path.exists():
        t = pd.read_csv(ds.traits_path)
        t['line'] = t['line'].astype(str)
        cols = [c for c in t.columns if c not in ('line', 'image_id')]
        return t.groupby('line')[cols].mean() if cols else None
    measured = ds.results_dir / 'traits' / 'real_line_means.csv'
    if measured.exists():
        return pd.read_csv(measured, index_col=0)
    return None


def _figures(out, cum, ld):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=110)
    ax[0].plot(np.arange(1, len(cum) + 1), cum, color='#c98a00')
    for t in (0.8, 0.95):
        ax[0].axhline(t, color='0.7', lw=0.8, ls='--')
    ax[0].set_xlabel('principal components'); ax[0].set_ylabel('variance explained')
    ax[0].set_title('PCA spectrum')
    if ld:
        ax[1].plot([r['lag_markers'] for r in ld], [r['median_r2'] for r in ld], 'o-', color='#2f5bd3')
        ax[1].set_xscale('log')
    ax[1].set_xlabel('markers apart'); ax[1].set_ylabel('median r$^2$'); ax[1].set_title('LD decay')
    fig.tight_layout()
    path = out / 'genetics.png'
    fig.savefig(path)
    plt.close(fig)
    return {'genetics': str(path)}
