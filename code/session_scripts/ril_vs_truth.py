# Generated roots (validation lines, valid images, 4 seeds) and GBLUP against the
# answer key's true genetic values. No test line is read.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import load_widiv_genotypes

meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv')
G = G[G.line.map(split) == 'train'].set_index('line')
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']

# GBLUP on trained lines' observed phenotypes -> validation lines.
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X = dosage[[row_of[g] for g in tr + val]].astype(np.float64)
Z = (X - X[:len(tr)].mean(0)) / (X[:len(tr)].std(0) + 1e-6)
K = Z @ Z.T / Z.shape[1]
lam = pd.read_csv(OUT / 'gblup_baseline.csv').set_index('trait')['lambda']
rows = []
for t in T:
    y = meta.loc[tr, t].to_numpy(float)
    a = np.linalg.solve(K[:len(tr), :len(tr)] + lam[t] * np.eye(len(tr)), y - y.mean())
    p = y.mean() + K[len(tr):, :len(tr)] @ a
    g = G.loc[val, f'g_{t}']
    row = {'trait': t, 'gblup_vs_g': np.corrcoef(p, g)[0, 1],
           'phenotype_vs_g': np.corrcoef(meta.loc[val, t], g)[0, 1]}
    for run in ('pca_e400', 'pca_e800', 'windows_e800'):
        d = pd.read_csv(OUT / f'traits_{run}.csv')
        d = d[d.valid & (d.role == 'validation')]
        m = d.groupby('genotype')[t].mean().reindex(val)
        ok = m.notna()
        row[f'{run}_vs_g'] = np.corrcoef(m[ok], g[ok])[0, 1]
        row[f'{run}_spread'] = m.std() / g.std()
    rows.append(row)
out = pd.DataFrame(rows).set_index('trait')
out.loc['mean'] = out.mean()
out.to_csv(OUT / 'vs_genetic_values.csv')
pd.set_option('display.width', 220)
print(out.round(3).to_string())
