# Validation figure: every designed QTL's contribution map + 6 controls (one scale, starting at
# the controls' median), and the compartment-enrichment matrix with the expected compartment(s)
# from the answer key outlined. Writes results/ril_v5/gene_maps/gene_map_validation.png.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle

GM = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/'
R = pd.read_csv(GM + 'gene_map_validation_rescored.csv')
z = np.load(GM + 'gene_maps_all.npz')
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
NAME = {'shared_stele_mx': 'Shared stele/vessel gene', 'mx_diam_epi': 'Vessel-width, interacting', 'mx_diam_add': 'Vessel-width, additive',
        'lac_q': 'Lacuna gene (blind trait)', 'rca_linked1': 'Aerenchyma gene 1', 'rca_linked2': 'Aerenchyma gene 2',
        **{f'ctx_q{k}': f'Cortex gene {k} ({p}%)' for k, p in zip(range(1, 9), (25, 15, 10, 7, 5, 3, 2, 1))},
        **{f'control{k}': f'Control region {k}' for k in range(1, 7)}}
order = list(R[~R.key.str.startswith('control')].sort_values('pct_var', ascending=False).key) + [f'control{k}' for k in range(1, 7)]
R = R.set_index('key').loc[order]
cmap = LinearSegmentedColormap.from_list('mizzou', ['#111111', '#4A3600', '#B8860B', '#F1B82D', '#FFF4D6'])
yy, xx = np.indices((256, 256))
inside = np.hypot(yy - 127.5, xx - 127.5) <= 1.04 * 256 / 2 / 1.3
vmin = float(np.median(np.concatenate([z[f'control{k}'][inside] for k in range(1, 7)])))
vmax = float(np.percentile(z['shared_stele_mx'][inside], 99))
norm = PowerNorm(0.8, vmin=vmin, vmax=vmax)

plt.rcParams.update({'font.family': 'Arial'})
fig = plt.figure(figsize=(24, 12.5), dpi=150)
gs = fig.add_gridspec(4, 7, width_ratios=[1, 1, 1, 1, 1, 1.15, 2.6], wspace=0.08, hspace=0.32)
for i, k in enumerate(order):
    ax = fig.add_subplot(gs[i // 5, i % 5])
    ax.imshow(np.where(inside, z[k], 0), cmap=cmap, norm=norm)
    r = R.loc[k]
    mark = '' if k.startswith('control') else (' (right)' if r.correct else ' (wrong)')
    ax.set_title(NAME[k], fontsize=11, fontweight='bold', loc='left')
    sub = (f"→ {r.points_to}{mark}" if not k.startswith('control') else 'no designed gene')
    ax.text(0.02, 0.03, sub, transform=ax.transAxes, fontsize=10.5, color='white', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
# enrichment matrix
ax = fig.add_subplot(gs[:, 6])
E = R[[f'enrich_{c}' for c in COMP]].to_numpy()
im = ax.imshow(E, cmap='RdBu_r', vmin=-5, vmax=5, aspect='auto')
ax.set_xticks(range(len(COMP)), COMP, fontsize=12)
ax.set_yticks(range(len(order)), [NAME[k] for k in order], fontsize=11)
ax.xaxis.tick_top()
for i, k in enumerate(order):
    exp = R.loc[k, 'expected']
    if isinstance(exp, str):
        for c in exp.split(';'):
            j = COMP.index(c)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='black', lw=2.5))
    for j in range(len(COMP)):
        ax.text(j, i, f'{E[i, j]:.1f}', ha='center', va='center', fontsize=9.5, color='black')
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label('enrichment vs controls (z)', fontsize=11)
fig.suptitle('Contribution maps for all 14 designed QTL vs 6 control regions: where in the root each map concentrates its change '
             '(box = compartment the QTL\'s trait lives in)', x=0.01, ha='left', fontsize=14, fontweight='bold')
fig.savefig(GM + 'gene_map_validation.png', facecolor='white', bbox_inches='tight')
print('wrote', GM + 'gene_map_validation.png')
