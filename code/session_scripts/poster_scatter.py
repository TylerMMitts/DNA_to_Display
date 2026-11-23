# Poster scatter: per trait, true genetic value (x) vs prediction (y, standardised so both
# predictors share an axis): generated roots (genetic value from 8 valid images, best seeds)
# in Mizzou gold, GBLUP in grey. 198 validation lines. Fonts sized for a 34 x 46 in poster.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes

GOLD, BLACK, GREY, MUTED = '#F1B82D', '#111111', '#A7A9AC', '#555555'
T = ['mx_n', 'mx_diam_um', 'stele_diam_um', 'cortex_width_um', 'mcs_layers', 'rca_pct']
NAME = {'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma', 'mcs_layers': 'Sclerenchyma'}
CSVNAME = {'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number', 'mx_diam_um': 'Metaxylem diameter',
           'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma (RCA)', 'mcs_layers': 'Sclerenchyma (MCS)'}
UNIT = {'stele_diam_um': 'µm', 'mx_n': 'vessels', 'mx_diam_um': 'µm', 'cortex_width_um': 'µm', 'rca_pct': 'liability',
        'mcs_layers': 'liability'}
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')
P = pd.read_csv(WIDIV_RESULTS_DIR / 'figures' / 'best_model_scatter_points_8valid.csv')

# GBLUP exactly as for the doc (kernel ridge on trained lines' observed phenotypes).
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X = dosage[[row_of[g] for g in tr + val]].astype(np.float64)
Z = (X - X[:len(tr)].mean(0)) / (X[:len(tr)].std(0) + 1e-6)
K = Z @ Z.T / Z.shape[1]
lam = pd.read_csv(OUT / 'gblup_baseline.csv').set_index('trait')['lambda']

plt.rcParams.update({'font.family': 'Arial'})
fig, axes = plt.subplots(2, 3, figsize=(17.6, 11.8), dpi=200)
z = lambda v: (v - v.mean()) / v.std()
for ax, t in zip(axes.ravel(), T):
    y = meta.loc[tr, t].to_numpy(float)
    a = np.linalg.solve(K[:len(tr), :len(tr)] + lam[t] * np.eye(len(tr)), y - y.mean())
    gb = pd.Series(y.mean() + K[len(tr):, :len(tr)] @ a, index=val)
    p = P[P.trait == CSVNAME[t]].set_index('line').reindex(val).dropna()
    g = G.loc[p.index, f'g_{t}']
    r_gen = np.corrcoef(p.generated_mean, g)[0, 1]
    r_gb = np.corrcoef(gb.loc[p.index], g)[0, 1]
    ax.scatter(g, z(gb.loc[p.index]), s=34, color=GREY, alpha=0.75, linewidths=0, zorder=2)
    ax.scatter(g, z(p.generated_mean), s=46, color=GOLD, edgecolors=BLACK, linewidths=0.6, zorder=3)
    ax.set_title(NAME[t], loc='left', fontsize=26, fontweight='bold', color=BLACK, pad=62)
    ax.text(0.0, 1.035, f'Generated r = {r_gen:.2f}', transform=ax.transAxes, fontsize=21, fontweight='bold',
            color=BLACK, va='bottom', ha='left', bbox=dict(boxstyle='round,pad=0.25', fc=GOLD, ec='none'))
    ax.text(1.0, 1.035, f'GBLUP r = {r_gb:.2f}', transform=ax.transAxes, fontsize=21, color='white',
            va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.25', fc='#6E7073', ec='none'))
    ax.set_xlabel(f'true genetic value ({UNIT[t]})', fontsize=18, color=MUTED)
    ax.tick_params(labelsize=16, colors=MUTED)
    ax.set_yticks([])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#BBBBBB')
for ax in axes[:, 0]:
    ax.set_ylabel('prediction (standardised)', fontsize=18, color=MUTED)
fig.tight_layout(h_pad=2.2, w_pad=1.6)
out = Path(r'E:/D2D/DNA_to_Display\presentation\poster_assets\poster_scatter.png')
fig.savefig(out, dpi=200, facecolor='white')
print('wrote', out)
