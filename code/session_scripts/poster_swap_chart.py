# Poster chart: for each designed gene, the share of its true effect that the generated root
# reproduces when the gene's chromosome region (±10 cM) is swapped to the other parent's version.
# 48 validation lines x 2 seeds; 95% intervals over lines (qtl_swap_summary.csv, windows model).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GOLD, BLACK, MUTED = '#F1B82D', '#111111', '#555555'
W = pd.read_csv(OUT / 'qtl_swap_summary.csv')
W = W[(W.model == 'windows10cM') & (W.edit == 'block') & W.on_target]
rows = [('shared_stele_mx', 'mx_n', 'Shared gene → metaxylem number'), ('shared_stele_mx', 'stele_diam_um', 'Shared gene → stele diameter'),
        ('mx_diam_add', 'mx_diam_um', 'Metaxylem diameter, additive gene'), ('mx_diam_epi', 'mx_diam_um', 'Metaxylem diameter, interacting gene'),
        ('ctx_q1', 'cortex_width_um', 'Cortex gene, 25% of variance'), ('ctx_q2', 'cortex_width_um', 'Cortex gene, 15%'),
        ('ctx_q3', 'cortex_width_um', 'Cortex gene, 10%'), ('ctx_q4', 'cortex_width_um', 'Cortex gene, 7%'),
        ('ctx_q5', 'cortex_width_um', 'Cortex gene, 5%')]
plt.rcParams.update({'font.family': 'Arial'})
fig, ax = plt.subplots(figsize=(13.9, 4.0), dpi=200)
for i, (q, t, label) in enumerate(rows):
    r = W[(W.qtl == q) & (W.trait == t)].iloc[0]
    y = len(rows) - 1 - i
    ax.plot([r.slope_lo_rel, r.slope_hi_rel], [y, y], color='#8A8A8A', lw=2.5, zorder=1)
    ax.scatter(r.slope_rel, y, s=190, color=GOLD, edgecolors=BLACK, linewidths=1.2, zorder=3)
    ax.text(r.slope_lo_rel - 0.02, y, f'{r.slope_rel:.0%}', va='center', ha='right', fontsize=21, fontweight='bold', color=BLACK)
ax.axvline(1, color=BLACK, lw=1.5, ls=(0, (4, 3)))
ax.text(1.0, len(rows) - 0.35, 'full effect', ha='center', va='bottom', fontsize=20, color=BLACK)
ax.set_yticks(range(len(rows)), [r[2] for r in rows[::-1]], fontsize=21, color=BLACK)
ax.set_xlim(0.45, 1.4)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_xticks([0.5, 0.75, 1.0, 1.25], ['50%', '75%', '100%', '125%'], fontsize=19, color=MUTED)
ax.tick_params(left=False)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#BBBBBB')
fig.tight_layout()
out = Path(r'E:/D2D/DNA_to_Display\presentation\poster_assets\poster_swap_chart.png')
fig.savefig(out, dpi=200, facecolor='white')
print('wrote', out)
