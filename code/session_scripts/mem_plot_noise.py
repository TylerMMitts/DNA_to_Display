# Plot-noise memorization. A real line mean = true genetic value + the mean of its plots'
# simulated noise, and that noise has no genetic basis. Regress the generated line mean on
# both (trait units): the noise coefficient is the share of the line's own plot noise the
# images reproduce. Only memorizing those images can produce it; on test lines it must be ~0.
#     python mem_plot_noise.py <run>_e<epoch> [...]
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_paths import WIDIV_GENETIC_VALUES, WIDIV_METADATA_DIR, WIDIV_RESULTS_DIR

T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'rca_pct', 'mcs_layers']
G = WIDIV_RESULTS_DIR / 'generated_all_lines'
real = pd.read_csv(WIDIV_METADATA_DIR / 'synthetic_phenotypes_widiv_hard__genotype_means.csv').set_index('genotype')
gv = pd.read_csv(WIDIV_GENETIC_VALUES).set_index('genotype')
rng = np.random.default_rng(0)


def line_means(run):
    g = pd.read_csv(G / run / 'traits.csv')
    if 'valid' in g:
        g = g[g.valid].copy()
    common = g.seed.map(g.seed.value_counts()) >= 30
    for t in T:
        g[t] = g[t] - np.where(common, g.groupby('seed')[t].transform('mean'), g[t].mean())
    role = g.groupby('genotype').role.first()
    return g.groupby('genotype')[T].mean(), role


def fit(y, gval, noise):
    x = np.c_[np.ones(len(y)), gval, noise]
    return np.linalg.lstsq(x, y, rcond=None)[0][1:]


rows = []
for run in sys.argv[1:]:
    gen, role = line_means(run)
    for r in ('trained', 'validation', 'test'):
        lines = role.index[role == r].intersection(real.index)
        shares = []
        for t in T:
            ok = lines[gen.loc[lines, t].notna() & real.loc[lines, t].notna()]
            y = gen.loc[ok, t].to_numpy()
            gval = gv.loc[ok, f'units_{t}'].to_numpy()
            noise = real.loc[ok, t].to_numpy() - gval
            b_g, b_n = fit(y, gval, noise)
            boot = []
            for _ in range(500):
                s = rng.integers(0, len(ok), len(ok))
                boot.append(fit(y[s], gval[s], noise[s]))
            boot = np.array(boot)
            rows.append({'run': run, 'role': r, 'trait': t, 'lines': len(ok), 'genetic_share': b_g,
                         'noise_share': b_n, 'noise_lo': np.percentile(boot[:, 1], 2.5),
                         'noise_hi': np.percentile(boot[:, 1], 97.5),
                         'noise_over_genetic': b_n / b_g if b_g > 0.05 else np.nan})
res = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(res.round(3).to_string(index=False))
print('\nMean over the 5 traits:')
print(res.groupby(['run', 'role'], sort=False)[['genetic_share', 'noise_share', 'noise_over_genetic']].mean().round(3).to_string())
if len(sys.argv) > 1:
    out = WIDIV_RESULTS_DIR / 'noqtl_diagnosis' / f"plot_noise_memorization_{'__'.join(sys.argv[1:])}.csv"
    res.to_csv(out, index=False)
