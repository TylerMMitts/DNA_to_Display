# Genetic value of each line's generated roots, rather than the raw 4-seed mean:
# one-way random model over valid images (line effect + seed noise).
#   sigma2_seed = pooled within-line variance, sigma2_line = var(means) - mean(sigma2_seed / n)
#   BLUP_i = mu + sigma2_line / (sigma2_line + sigma2_seed / n_i) * (mean_i - mu)
#   r_noise_free = r(mean, g) / sqrt(sigma2_line / var(means)): the r the model would reach
#   with unlimited seeds (correlation corrected for seed noise in the means).
# Windows e1200, 198 validation lines. Answer key for scoring only.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from widiv_paths import WIDIV_RESULTS_DIR

FIG = WIDIV_RESULTS_DIR / 'figures'
# GBLUP r per trait (trained lines' phenotypes -> validation lines), from ril_best_model_figures.py.
GBLUP = pd.read_csv(FIG / 'best_model_trait_r.csv').set_index('trait').gblup
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
NAME = {'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma (RCA)', 'mcs_layers': 'Sclerenchyma (MCS)'}
UNIT = {'stele_diam_um': 'µm', 'mx_n': 'vessels', 'mx_diam_um': 'µm', 'cortex_width_um': 'µm',
        'rca_pct': '% (measured)', 'mcs_layers': 'band width (measured)'}
# IMAGES=8valid: the best 8 seeds by validity (ril_seed_screen.py) with invalid images
# replaced, 8 valid images per line; default: seeds 0-3.
import os
MODE = os.environ.get('IMAGES', '')   # '', '8valid' or '16seeds'
SUF = f'_{MODE}' if MODE else ''
d = pd.read_csv(OUT / {'8valid': 'traits_windows_e1200_8valid.csv', '16seeds': 'traits_windows_e1200_16seeds.csv'}
                .get(MODE, 'traits_windows_e1200.csv'))
d = d[d.valid & (d.role == 'validation')]
IMG_NOTE = {'8valid': '8 valid images per line: best 8 seeds by validity, invalid images replaced by the next-best seed',
            '16seeds': 'its valid generated roots from 16 seeds (0-15)'}.get(MODE, 'valid generated roots from 4 seeds')
rows, blups = [], {}
for t in T:
    grp = d.groupby('genotype')[t]
    m, n, v = grp.mean().reindex(val), grp.size().reindex(val), grp.var().reindex(val)
    s2_seed = float(np.nansum(v * (n - 1)) / np.nansum(n - 1))
    s2_line = float(np.var(m, ddof=1) - np.mean(s2_seed / n))
    mu = float(m.mean())
    h = s2_line / (s2_line + s2_seed / n)
    b = mu + h * (m - mu)
    blups[t] = b
    g = G.loc[val, f'g_{t}']
    r_mean, r_blup = np.corrcoef(m, g)[0, 1], np.corrcoef(b, g)[0, 1]
    rel = s2_line / np.var(m, ddof=1)
    rows.append({'trait': NAME[t], 'seed_sd': np.sqrt(s2_seed), 'line_sd': np.sqrt(s2_line),
                 'reliability_of_4_seed_mean': rel, 'r_mean': r_mean, 'r_blup': r_blup,
                 'r_noise_free': r_mean / np.sqrt(rel), 'r_1_seed_expected': r_mean / np.sqrt(rel) * np.sqrt(
                     s2_line / (s2_line + s2_seed))})
R = pd.DataFrame(rows)
R.to_csv(FIG / f'best_model_trait_r_blup{SUF}.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
print('mean r: 4-seed mean %.3f | BLUP %.3f | noise-free %.3f' % (R.r_mean.mean(), R.r_blup.mean(), R.r_noise_free.mean()))

BLUE, INK, MUTED = '#2a6fdb', '#1f1f1f', '#6b6b6b'
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
for ax, t, rr in zip(axes.ravel(), T, rows):
    g = G.loc[val, f'g_{t}'].to_numpy()
    b = blups[t].to_numpy()
    ax.scatter(g, b, s=14, color=BLUE, alpha=0.65, linewidths=0)
    c = np.polyfit(g, b, 1)
    xs = np.linspace(g.min(), g.max(), 2)
    ax.plot(xs, np.polyval(c, xs), color=INK, lw=1)
    ax.set_title(NAME[t], fontsize=11, color=INK, loc='left')
    # r of each predictor with the true genetic value, upper left (empty for positive r).
    gb = GBLUP[NAME[t]]
    ax.text(0.03, 0.97, f"generated   r = {rr['r_blup']:.2f}\nnoise-free  r = {rr['r_noise_free']:.2f}\n"
            f"GBLUP       r = {gb:.2f}", transform=ax.transAxes, va='top', ha='left', fontsize=9,
            color=INK, family='monospace', linespacing=1.5)
    ax.set_xlabel('true genetic value', fontsize=9, color=MUTED)
    ax.set_ylabel(f'genetic value of generated roots, {UNIT[t]}', fontsize=9, color=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c8c8c8')
    ax.tick_params(colors=MUTED, labelsize=8)
fig.suptitle('Genetic value of generated roots vs true genetic value, 198 validation lines (windows, epoch 1200)',
             x=0.01, ha='left', fontsize=13, color=INK)
fig.text(0.01, 0.005, f'Each dot = one line: BLUP of its line effect from {IMG_NOTE}.\n'
         'Line + seed-noise random model. Noise-free r = r corrected for seed noise (unlimited seeds). '
         'GBLUP = SNP-only prediction of the same 198 lines from the trained lines\' phenotypes.\n'
         'All r are with the true genetic value from the answer key, never seen in training.',
         fontsize=8, color=MUTED, linespacing=1.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(FIG / f'best_model_trait_scatter_blup{SUF}.png', dpi=150)
print('wrote', FIG / f'best_model_trait_scatter_blup{SUF}.png')

