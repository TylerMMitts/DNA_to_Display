# How much of each designed QTL can still be read from the markers left after
# removing those within +/- X cM (and after orthogonalising every marker against
# the QTL genotypes). Ridge from remaining markers -> QTL dosage, fitted on
# trained lines, R^2 on validation lines. Also the genome removed at each radius.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SNP_MAP
from widiv_snp_encoding import load_widiv_genotypes

meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X = dosage[[row_of[g] for g in tr + val]].astype(np.float64)
Z = (X - X[:len(tr)].mean(0)) / (X[:len(tr)].std(0) + 1e-6)
n_tr = len(tr)
smap = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[snps]
q = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/qtl_and_scan_baseline__qtl.csv')
qm = q.drop_duplicates('nearest_marker')
col = {s: i for i, s in enumerate(snps)}
qcols = [col[m] for m in qm.nearest_marker]
bg = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/qtl_and_scan_baseline__background_qtl.csv')
print(f'{len(qcols)} designed QTL markers; {bg.nearest_marker.nunique()} background QTL markers '
      f'({bg.nearest_marker.isin(set(qm.nearest_marker)).sum()} shared with designed)')


def r2_from(cols_keep, target):
    A = Z[:, cols_keep]
    K = A @ A.T / A.shape[1]
    y = Z[:, target]
    best = -9
    for lam in (0.01, 0.1, 1):
        al = np.linalg.solve(K[:n_tr, :n_tr] + lam * np.eye(n_tr), y[:n_tr] - y[:n_tr].mean())
        p = y[:n_tr].mean() + K[n_tr:, :n_tr] @ al
        r2 = 1 - ((y[n_tr:] - p) ** 2).sum() / ((y[n_tr:] - y[n_tr:].mean()) ** 2).sum()
        best = max(best, r2)
    return best


rows = []
chrom, cm = smap.chr.to_numpy(), smap.pos_cM.to_numpy()
for radius in (0, 2, 5, 10, 20, 40):
    remove = np.zeros(len(snps), bool)
    for c in qcols:
        remove |= (chrom == chrom[c]) & (np.abs(cm - cm[c]) <= radius)
    keep = np.flatnonzero(~remove)
    r2 = [r2_from(keep, c) for c in qcols]
    rows.append({'removal': f'+/-{radius} cM', 'markers_removed': int(remove.sum()),
                 'share_of_genome_removed': float(remove.mean()), 'median_qtl_r2': float(np.median(r2)),
                 'max_qtl_r2': float(np.max(r2)), **dict(zip(qm.qtl, np.round(r2, 3)))})
    print(rows[-1]['removal'], rows[-1]['markers_removed'], round(rows[-1]['median_qtl_r2'], 3),
          round(rows[-1]['max_qtl_r2'], 3), flush=True)
# Orthogonalised: every marker's residual after regressing on the QTL genotypes (fitting lines).
Q = Z[:, qcols]
B = np.linalg.lstsq(Q[:n_tr], Z[:n_tr], rcond=None)[0]
Zr = Z - Q @ B
Zr[:, qcols] = 0
Zs = Z
Z = Zr
keep = np.array([i for i in range(len(snps)) if i not in set(qcols)])
Z = np.concatenate([Zr, Zs[:, qcols]], 1)   # targets appended at the end
targets = list(range(Zr.shape[1], Z.shape[1]))
r2 = [r2_from(keep, t) for t in targets]
rows.append({'removal': 'orthogonalised to the QTL', 'markers_removed': len(qcols), 'share_of_genome_removed': 0.0,
             'median_qtl_r2': float(np.median(r2)), 'max_qtl_r2': float(np.max(r2)), **dict(zip(qm.qtl, np.round(r2, 3)))})
print('orthogonalised', round(float(np.median(r2)), 3), round(float(np.max(r2)), 3))
pd.DataFrame(rows).to_csv(OUT / 'qtl_leak_by_radius.csv', index=False)
