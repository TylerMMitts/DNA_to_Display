# SNP density along the genome with the strict condition's +-1 Mb QTL windows marked,
# plus a zoom on the sparsest and densest windows showing each SNP.
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_QTL, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes

_, snps, _ = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
s = pd.Series(snps).str.extract(r'S(\d+)_(\d+)').astype(int)
s.columns = ['chr', 'pos']
q = pd.read_csv(WIDIV_QTL).drop_duplicates('snp')
q['n'] = [int(((s.chr == r.chr) & ((s.pos - r.pos).abs() <= 1e6)).sum()) for r in q.itertuples()]

INK, MUTED, GRID, ACCENT = '#1f2328', '#6b7280', '#e5e7eb', '#2563eb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(11, 10.5))
gs = fig.add_gridspec(12, 2, height_ratios=[1] * 10 + [0.9, 2.2], hspace=0.35, wspace=0.18, top=0.93, bottom=0.05)
fig.suptitle('SNPs are spaced almost evenly; small gaps make the 2 Mb QTL windows hold 78\u201396 each',
             x=0.06, ha='left', fontsize=13, color=INK, y=0.985)
fig.text(0.06, 0.957, 'WiDiv panel, 90,790 SNPs. Grey: SNPs per 1 Mb bin. Blue bands: the strict condition\'s windows '
         '(1 Mb either side of each of the 40 QTL SNPs), 3,399 SNPs removed in total.', fontsize=9, color=MUTED)

ymax = 0
bins = {}
for c in range(1, 11):
    p = s.pos[s.chr == c].to_numpy()
    edges = np.arange(0, p.max() + 1e6, 1e6)
    h, _ = np.histogram(p, edges)
    bins[c] = (edges, h)
    ymax = max(ymax, h.max())
xmax = max(e[-1] for e, _ in bins.values()) / 1e6
for c in range(1, 11):
    ax = fig.add_subplot(gs[c - 1, :])
    edges, h = bins[c]
    ax.fill_between(edges[:-1] / 1e6, h, step='post', color=MUTED, alpha=0.55, linewidth=0)
    for r in q[q.chr == c].itertuples():
        ax.axvspan((r.pos - 1e6) / 1e6, (r.pos + 1e6) / 1e6, color=ACCENT, alpha=0.35, linewidth=0)
        ax.plot([r.pos / 1e6], [ymax * 1.02], marker='v', ms=4, color=ACCENT, clip_on=False)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax * 1.05)
    ax.set_yticks([0, 100]); ax.tick_params(length=2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(-0.012, 0.45, f'Chr {c}', transform=ax.transAxes, ha='right', va='center', color=INK)
    if c < 10:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Position (Mb)')
fig.text(0.015, 0.62, 'SNPs per Mb', rotation=90, va='center', color=MUTED)

for k, r in enumerate([q.sort_values('n').iloc[0], q.sort_values('n').iloc[-1]]):
    ax = fig.add_subplot(gs[11, k])
    p = s.pos[(s.chr == r.chr) & ((s.pos - r.pos).abs() <= 1.5e6)].to_numpy()
    inside = np.abs(p - r.pos) <= 1e6
    ax.axvspan(-1, 1, color=ACCENT, alpha=0.12, linewidth=0)
    ax.vlines((p[~inside] - r.pos) / 1e6, 0.2, 0.8, color=MUTED, linewidth=0.8)
    ax.vlines((p[inside] - r.pos) / 1e6, 0.2, 0.8, color=ACCENT, linewidth=0.8)
    ax.plot([0], [0.92], marker='v', ms=7, color=INK)
    ax.text(0, 1.0, r.snp, ha='center', va='bottom', color=INK, fontsize=8)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 1.15); ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('Distance from QTL (Mb)')
    label = 'Sparsest window' if k == 0 else 'Densest window'
    ax.set_title(f'{label}: {int(inside.sum())} SNPs removed (blue), each line one SNP', loc='left', fontsize=10, color=INK)

out = WIDIV_RESULTS_DIR / 'gwas' / 'noqtl_comparison' / 'snp_density_qtl_windows.png'
fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
print(out)
