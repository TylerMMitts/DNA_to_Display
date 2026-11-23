# How much of each model's genetic variation in generated roots traces back to the
# designed QTL, vs the truth. 198 validation lines, line means of valid generated roots.
#   g = g_Q (designed QTL part, from the least-squares rebuild of g) + g_B (background)
#   between-line variance of generated means, minus seed noise (within-line var / n)
#   share_Q = variance explained by the trait's designed QTL terms / between-line variance
#   per-QTL share = beta_k^2 var(x_k) / between-line variance, against the truth's
# Answer key used for scoring only.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import load_widiv_genotypes

AK = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/'
GT = ['mx_n', 'stele_diam_um', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
MODELS = {'normal e800': 'windows_e800', 'normal e1200': 'windows_e1200',
          'QTL removed e800': 'windowsorth_e800', 'QTL only e800': 'qtlonly_e800'}
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
lines = list(lines)
col = {s: i for i, s in enumerate(snps)}
row_of = {g: i for i, g in enumerate(lines)}
tr = [g for g in lines if split.get(g) == 'train']
Dtr = dosage[[row_of[g] for g in tr]].astype(float)
Dva = dosage[[row_of[g] for g in val]].astype(float)
G = pd.read_csv(AK + 'genetic_values.csv').set_index('line')
qt = pd.read_csv(AK + 'qtl_and_scan_baseline__qtl.csv')
bg = pd.read_csv(AK + 'qtl_and_scan_baseline__background_qtl.csv')
dq = pd.read_csv(AK + 'designed_qtl_markers.csv')
pct = qt.set_index(['trait', 'qtl']).pct_genetic_variance


def qtl_terms(D, t):
    # Designed QTL terms for trait t: (names, matrix).
    rows = dq[dq.trait.str.split(';').apply(lambda ts: t in ts)]
    names = list(rows.qtl)
    X = D[:, [col[s] for s in rows.snp]] - 1.0
    if t == 'mx_diam_um':
        names.append('mx_diam AxA')
        X = np.c_[X, (D[:, col['5_2477']] - 1) * (D[:, col['8_2175']] - 1)]
    return names, X


def ols(X, y):
    A = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return b, A @ b


# Truth: full rebuild on training lines, then the designed-QTL part of it.
truth_parts = {}
for t in GT:
    mk = sorted({col[m] for m in pd.concat([qt[qt.trait == t].nearest_marker, bg[bg.trait == t].nearest_marker])})
    names, Xq_tr = qtl_terms(Dtr, t)
    others = [c for c in mk if c not in {col[s] for s in dq.snp}]
    A = np.c_[np.ones(len(tr)), Xq_tr, Dtr[:, others]]
    b = np.linalg.lstsq(A, G.loc[tr, f'g_{t}'].to_numpy(), rcond=None)[0]
    _, Xq_va = qtl_terms(Dva, t)
    g_va = G.loc[val, f'g_{t}'].to_numpy()
    gq = Xq_va @ b[1:1 + len(names)] if names else np.zeros(len(val))
    gq = gq - gq.mean()
    truth_parts[t] = (names, Xq_va, g_va, gq, g_va - g_va.mean() - gq, b[1:1 + len(names)])

rng = np.random.default_rng(0)
B = 1000
boot_idx = [rng.integers(0, len(val), len(val)) for _ in range(B)]
summary, perqtl = [], []
for label, stem in [('truth (genetic value)', None)] + list(MODELS.items()):
    if stem is not None:
        f = OUT / f'traits_{stem}.csv'
        if not f.exists():
            print('missing', f.name); continue
        d = pd.read_csv(f)
        d = d[d.valid & (d.role == 'validation')]
    for t in GT:
        names, Xq, g, gq, gb, bq_true = truth_parts[t]
        if stem is None:
            m, noise = g.copy(), 0.0
        else:
            grp = d.groupby('genotype')[t]
            m = grp.mean().reindex(val).to_numpy()
            n = grp.size().reindex(val).to_numpy()
            noise = np.nanmean(grp.var().reindex(val).to_numpy() / n)
        ok = np.isfinite(m)

        def stats(ix):
            mm, XX = m[ix], Xq[ix]
            vb = np.var(mm, ddof=1) - noise
            if vb <= 0.02 * np.var(mm, ddof=1):
                # No between-line variance beyond seed noise: nothing to attribute.
                return {'between_var': max(vb, 0.0), 'share_qtl': np.nan, 'share_bg': np.nan, 'share_rest': np.nan,
                        'coef': np.full(len(names), np.nan), 'per': [np.nan] * len(names),
                        'r_g': np.corrcoef(mm, g[ix])[0, 1],
                        'r_gq': np.corrcoef(mm, gq[ix])[0, 1] if names else np.nan,
                        'r_gb': np.corrcoef(mm, gb[ix])[0, 1] if np.std(gb) > 1e-6 * np.std(g) else np.nan}
            out = {'between_var': vb}
            has_bg = np.std(gb) > 1e-6 * np.std(g)
            # Sequential: between-line variance = what the designed-QTL terms explain
            # + what the true background value adds + the rest (model error).
            k = len(names)
            fq = ols(XX, mm)[1] if names else np.full(len(mm), mm.mean())
            parts = ([XX] if names else []) + ([gb[ix][:, None]] if has_bg else [])
            fit = ols(np.hstack(parts), mm)[1]
            out['share_qtl'] = np.var(fq, ddof=1) / vb if names else np.nan
            out['share_bg'] = (np.var(fit, ddof=1) - np.var(fq, ddof=1)) / vb if has_bg else np.nan
            out['share_rest'] = max(1 - np.var(fit, ddof=1) / vb, 0.0)
            if names:
                bq, _ = ols(XX, mm)
                out['coef'] = bq[1:]
                out['per'] = [bq[1 + j] ** 2 * np.var(XX[:, j], ddof=1) / vb for j in range(k)]
            out['r_g'] = np.corrcoef(mm, g[ix])[0, 1]
            out['r_gq'] = np.corrcoef(mm, gq[ix])[0, 1] if names else np.nan
            out['r_gb'] = np.corrcoef(mm, gb[ix])[0, 1] if has_bg else np.nan
            return out

        idx = np.flatnonzero(ok)
        s0 = stats(idx)
        bs = [stats(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(300)] if stem else []
        ci = lambda k: (np.percentile([b[k] for b in bs], 2.5), np.percentile([b[k] for b in bs], 97.5)) if bs else (np.nan, np.nan)
        row = {'model': label, 'trait': t, 'n_lines': int(ok.sum()),
               'r_g': s0['r_g'], 'r_qtl_part': s0['r_gq'], 'r_background_part': s0['r_gb'],
               'sd_between': np.sqrt(s0['between_var']), 'seed_noise_sd': np.sqrt(noise)}
        for key in ('share_qtl', 'share_bg', 'share_rest'):
            row[key] = s0[key]
            row[key + '_lo'], row[key + '_hi'] = ci(key)
        if names:
            row['qtl_sd'] = np.sqrt(s0['share_qtl'] * s0['between_var'])
            for k, nm in enumerate(names):
                if t == 'rca_pct':
                    continue   # repulsion pair: only the pair's share (share_qtl) is meaningful
                pk = [b['per'][k] for b in bs]
                perqtl.append({'model': label, 'trait': t, 'qtl': nm, 'share': s0['per'][k],
                               'share_lo': np.percentile(pk, 2.5) if bs else np.nan,
                               'share_hi': np.percentile(pk, 97.5) if bs else np.nan,
                               'effect_per_dose': s0['coef'][k], 'true_effect_per_dose': bq_true[k]})
        summary.append(row)

S = pd.DataFrame(summary)
P = pd.DataFrame(perqtl)
S.to_csv(OUT / 'qtl_variance_summary.csv', index=False)
P.to_csv(OUT / 'qtl_variance_per_qtl.csv', index=False)
pd.set_option('display.width', 250)
print(S.round(3).to_string(index=False))
print()
print(P.round(3).to_string(index=False))
