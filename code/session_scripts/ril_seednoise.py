# Generated-trait r on validation lines with seed noise removed: line means of n
# seeds have reliability n*ICC/(1+(n-1)ICC); r / sqrt(reliability) estimates the r
# of the model's expected root (infinite seeds). Also trained-line r and spread.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, load_tables
import numpy as np, pandas as pd
meta, _ = load_tables()
gb = pd.read_csv(OUT / 'gblup_baseline.csv').set_index('trait')
rows = []
for run in ('pca_e400', 'pca_e800', 'windows_e800'):
    d = pd.read_csv(OUT / f'traits_{run}.csv')
    d = d[d.valid]
    for role in ('validation', 'trained'):
        v = d[d.role == role]
        for t in TRAITS:
            m = v.groupby('genotype')[t].agg(['mean', 'count'])
            obs = meta.loc[m.index, t]
            r = np.corrcoef(m['mean'], obs)[0, 1]
            # ICC by one-way ANOVA
            n = m['count'].mean()
            msb = (m['count'] * (m['mean'] - v[t].mean()) ** 2).sum() / (len(m) - 1)
            msw = ((v[t] - v.genotype.map(m['mean'])) ** 2).sum() / (len(v) - len(m))
            icc = max((msb - msw) / (msb + (n - 1) * msw), 1e-6)
            rel = n * icc / (1 + (n - 1) * icc)
            rows.append({'run': run, 'role': role, 'trait': t, 'r': r, 'icc': icc, 'reliability': rel,
                         'r_noise_free': r / np.sqrt(rel), 'gblup': gb.loc[t, 'r_val'],
                         'spread_ratio': m['mean'].std() / obs.std()})
out = pd.DataFrame(rows)
out.to_csv(OUT / 'seed_noise.csv', index=False)
pd.set_option('display.width', 200)
print(out[out.role == 'validation'].round(2).to_string(index=False))
print(out.groupby(['run', 'role'])[['r', 'r_noise_free', 'icc', 'spread_ratio']].mean().round(3))
