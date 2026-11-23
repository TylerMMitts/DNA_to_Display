# Linear probe: ridge from k PCA scores (fitted on trained lines) to phenotypes,
# scored on validation lines, for the 0.95-variance PCA the RIL models use (124)
# and for more components. Same comparison as the WiDiv SNP-encoding analysis.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, load_tables
import numpy as np
import pandas as pd
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import DosagePCAProjector, load_widiv_genotypes

meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
fit_lines = sorted(g for g in split.index[split == 'train'] if g not in set(val) and g in row_of)
proj = DosagePCAProjector(n_components=None, target_variance=1.0)
proj.fit(dosage[[row_of[g] for g in fit_lines]], snps, verbose=False)
tr = [g for g in fit_lines if g in meta.index]
S_tr = proj.transform(dosage[[row_of[g] for g in tr]])
S_va = proj.transform(dosage[[row_of[g] for g in val]])
ev = S_tr.var(0)
cum = np.cumsum(ev) / ev.sum()
rng = np.random.default_rng(0)
folds = rng.permutation(np.arange(len(tr)) % 5)
rows = []
for k in (int(np.searchsorted(cum, 0.80)) + 1, int(np.searchsorted(cum, 0.95)) + 1, 300, 600, 1000, S_tr.shape[1]):
    A, B = S_tr[:, :k], S_va[:, :k]
    row = {'components': k, 'variance': float(cum[k - 1])}
    for t in TRAITS:
        y = meta.loc[tr, t].to_numpy(float)
        best, best_lam = -1, None
        for lam in (1, 10, 100, 1000, 1e4, 1e5):
            pred = np.empty_like(y)
            for f in range(5):
                a, b = folds != f, folds == f
                w = np.linalg.solve(A[a].T @ A[a] + lam * np.eye(k), A[a].T @ (y[a] - y[a].mean()))
                pred[b] = y[a].mean() + A[b] @ w
            r = np.corrcoef(pred, y)[0, 1]
            if r > best:
                best, best_lam = r, lam
        w = np.linalg.solve(A.T @ A + best_lam * np.eye(k), A.T @ (y - y.mean()))
        row[t] = float(np.corrcoef(y.mean() + B @ w, meta.loc[val, t])[0, 1])
    row['mean_r'] = float(np.mean([row[t] for t in TRAITS]))
    rows.append(row)
    print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
gb = pd.read_csv(OUT / 'gblup_baseline.csv').set_index('trait').r_val
print('GBLUP', {t: round(gb[t], 3) for t in TRAITS}, 'mean', round(gb.mean(), 3))
pd.DataFrame(rows).to_csv(OUT / 'pca_components_probe.csv', index=False)
