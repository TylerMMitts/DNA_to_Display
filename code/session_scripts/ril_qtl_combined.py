# Do the QTL-only and QTL-removed models add up to the normal model? Validation-line
# means of the two, summed (each centred), against the true g.
import sys; sys.path.insert(0, '.')
from ril_common import OUT
import numpy as np, pandas as pd
GT = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line').loc[val]
M = {}
for k in ('windows_e800', 'windowsorth_e800', 'qtlonly_e800'):
    d = pd.read_csv(OUT / f'traits_{k}.csv'); d = d[d.valid & (d.role == 'validation')]
    M[k] = d.groupby('genotype')[GT].mean().reindex(val)
rows = []
for t in GT:
    g = G[f'g_{t}']
    q, o, n = M['qtlonly_e800'][t], M['windowsorth_e800'][t], M['windows_e800'][t]
    s = (q - q.mean()) + (o - o.mean())
    ok = s.notna() & n.notna()
    rows.append({'trait': t, 'normal': np.corrcoef(n[ok], g[ok])[0, 1], 'qtl_only': np.corrcoef(q[ok], g[ok])[0, 1],
                 'qtl_removed': np.corrcoef(o[ok], g[ok])[0, 1], 'sum_of_both': np.corrcoef(s[ok], g[ok])[0, 1]})
R = pd.DataFrame(rows); R.loc[len(R)] = ['mean'] + list(R.iloc[:, 1:].mean())
R.to_csv(OUT / 'qtl_models_combined.csv', index=False)
print(R.round(3).to_string(index=False))
