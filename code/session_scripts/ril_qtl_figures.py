# Figures for the QTL-only / QTL-removed comparison (198 validation lines):
#   qtl_variance_partition.png  between-line variance of each model's generated roots
#                               split into designed QTL / background genetics / not genetic
#   qtl_per_qtl_share.png       share of variance per designed QTL: truth vs models
#   qtl_swap_effect.png         haplotype-swap effect per QTL relative to the true effect
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from widiv_paths import WIDIV_RESULTS_DIR

FIG = WIDIV_RESULTS_DIR / 'figures'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#d9d9d7', '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
NAME = {'mx_n': 'Metaxylem number', 'stele_diam_um': 'Stele diameter', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma', 'mcs_layers': 'Sclerenchyma'}
ORDER = ['mx_n', 'stele_diam_um', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
MODELS = ['truth (genetic value)', 'normal e800', 'QTL only e800', 'QTL removed e800']
LABEL = {'truth (genetic value)': 'Truth', 'normal e800': 'Normal model', 'QTL only e800': 'QTL only',
         'QTL removed e800': 'QTL removed'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.facecolor': SURF, 'figure.facecolor': SURF})


def clean(ax, left=False):
    for s in ('top', 'right') + (('left',) if not left else ()):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


# 1. Variance partition.
S = pd.read_csv(OUT / 'qtl_variance_summary.csv')
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
for ax, t in zip(axes.ravel(), ORDER):
    for i, m in enumerate(MODELS):
        r = S[(S.model == m) & (S.trait == t)].iloc[0]
        y = len(MODELS) - 1 - i
        if np.isnan(r.share_rest) and m != MODELS[0]:
            ax.text(0.01, y, 'no genetic variation beyond seed noise', va='center', fontsize=8.5, color=MUTED)
            continue
        q, b = np.nan_to_num(r.share_qtl), np.nan_to_num(r.share_bg)
        rest = max(0.0, 1 - q - b) if m != MODELS[0] else 0.0
        tot = q + b + rest
        q, b, rest = q / tot, b / tot, rest / tot   # shares can sum a little over 1 from sampling
        x = 0
        for w, c in ((q, BLUE), (b, AQUA), (rest, GRID)):
            if w > 0:
                ax.barh(y, w, left=x, height=0.62, color=c, edgecolor=SURF, linewidth=2)
                if w >= 0.09:
                    ax.text(x + w / 2, y, f'{w:.0%}', ha='center', va='center', fontsize=8,
                            color='white' if c != GRID else INK)
            x += w
    ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS[::-1]], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ['0%', '50%', '100%'])
    ax.set_title(NAME[t], loc='left', fontsize=11, color=INK)
    ax.tick_params(left=False)
    clean(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA, GRID)]
fig.legend(handles, ['designed QTL', 'background genetics', 'not genetic (model error)'], loc='upper right',
           ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.985))
fig.suptitle('Where each model\'s line-to-line variation comes from', x=0.01, y=0.975, ha='left', fontsize=13,
             color=INK)
fig.text(0.01, 0.01, 'Between-line variance of generated roots (198 validation lines, seed noise removed), split '
         'sequentially: designed QTL terms first, then what the true background value adds, the rest is not '
         'genetic.\nTruth = the answer key\'s genetic values. Sclerenchyma has no designed QTL.',
         fontsize=8, color=MUTED, linespacing=1.5)
fig.tight_layout(rect=(0, 0.06, 1, 0.93))
fig.savefig(FIG / 'qtl_variance_partition.png', dpi=150)
plt.close(fig)

# 2. Per-QTL share of between-line variance.
P = pd.read_csv(OUT / 'qtl_variance_per_qtl.csv')
P = P[(P.qtl != 'mx_diam_epi')]
rowsets = [('mx_n', 'shared_stele_mx', 'Shared QTL → metaxylem number'),
           ('stele_diam_um', 'shared_stele_mx', 'Shared QTL → stele diameter'),
           ('mx_diam_um', 'mx_diam_add', 'Metaxylem diam. additive'),
           ('mx_diam_um', 'mx_diam AxA', 'Metaxylem diam. A×A interaction')] + \
          [('cortex_width_um', f'ctx_q{k}', f'Cortex QTL {k} ({p}% designed)')
           for k, p in zip(range(1, 9), (25, 15, 10, 7, 5, 3, 2, 1))]
fig, ax = plt.subplots(figsize=(9, 6.4))
styles = [('truth (genetic value)', INK, 'D', 'Truth'), ('normal e800', BLUE, 'o', 'Normal model'),
          ('QTL only e800', ORANGE, 'o', 'QTL only'), ('QTL removed e800', AQUA, 'o', 'QTL removed')]
for i, (t, q, lab) in enumerate(rowsets):
    y = len(rowsets) - 1 - i
    for j, (m, c, mk, _) in enumerate(styles):
        r = P[(P.model == m) & (P.trait == t) & (P.qtl == q)].iloc[0]
        yy = y + (j - 1.5) * 0.17
        if not np.isnan(r.share_lo):
            ax.plot([r.share_lo, r.share_hi], [yy, yy], color=c, lw=1.2, alpha=0.6)
        ax.scatter(r.share, yy, s=34 if mk == 'o' else 30, color=c, marker=mk, zorder=3, edgecolor=SURF,
                   linewidth=1)
ax.set_yticks(range(len(rowsets)), [r[2] for r in rowsets[::-1]], fontsize=9, color=INK)
ax.set_xlabel('share of between-line variance in that trait', fontsize=9, color=MUTED)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlim(-0.01, 1.05)
for x in (0.25, 0.5, 0.75, 1.0):
    ax.axvline(x, color=GRID, lw=0.6, zorder=0)
ax.tick_params(left=False)
clean(ax)
for m, c, mk, lab in styles:
    ax.scatter([], [], color=c, marker=mk, label=lab, s=34)
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.set_title('Variance traced to each designed QTL (198 validation lines, 95% intervals)', loc='left',
             fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(FIG / 'qtl_per_qtl_share.png', dpi=150)
plt.close(fig)

# 3. Haplotype-swap effect per QTL, relative to the true effect.
W = pd.read_csv(OUT / 'qtl_swap_summary.csv')
W = W[W.on_target]
lab = {'shared_stele_mx': 'Shared QTL', 'mx_diam_add': 'Mx diam. additive', 'mx_diam_epi': 'Mx diam. epistatic',
       'rca_linked1': 'Aerenchyma QTL 1', 'rca_linked2': 'Aerenchyma QTL 2',
       **{f'ctx_q{k}': f'Cortex QTL {k}' for k in range(1, 9)}}
rows = [('shared_stele_mx', 'mx_n'), ('shared_stele_mx', 'stele_diam_um'), ('mx_diam_add', 'mx_diam_um'),
        ('mx_diam_epi', 'mx_diam_um')] + [(f'ctx_q{k}', 'cortex_width_um') for k in range(1, 9)] + \
       [('rca_linked1', 'rca_pct'), ('rca_linked2', 'rca_pct')]
series = [('qtlonly', 'marker', ORANGE, 'QTL only: flip the QTL marker'),
          ('windows', 'marker', '#86b6ef', 'Normal: flip the QTL marker'),
          ('windows', 'block', BLUE, 'Normal: swap ±2 cM haplotype'),
          ('windows10cM', 'block', '#184f95', 'Normal: swap ±10 cM haplotype'),
          ('windowsorth', 'block', AQUA, 'QTL removed: swap ±2 cM haplotype')]
series = [s for s in series if ((W.model == s[0]) & (W.edit == s[1])).any()]
fig, ax = plt.subplots(figsize=(10, 8))
for i, (q, t) in enumerate(rows):
    y = len(rows) - 1 - i
    for j, (m, e, c, _) in enumerate(series):
        r = W[(W.model == m) & (W.edit == e) & (W.qtl == q) & (W.trait == t)]
        # +-10 cM around either aerenchyma QTL holds both (5 cM apart, repulsion): they cancel.
        if not len(r) or (m == 'windows10cM' and t == 'rca_pct'):
            continue
        r = r.iloc[0]
        yy = y + (j - (len(series) - 1) / 2) * 0.15
        ax.plot([r.slope_lo_rel, r.slope_hi_rel], [yy, yy], color=c, lw=1.1, alpha=0.6)
        ax.scatter(r.slope_rel, yy, s=26, color=c, zorder=3, edgecolor=SURF, linewidth=0.8)
ax.axvline(0, color=MUTED, lw=0.8)
ax.axvline(1, color=INK, lw=0.8, ls=(0, (3, 2)))
ax.text(1.02, len(rows) - 0.4, 'full effect', fontsize=8, color=INK)
ax.set_yticks(range(len(rows)), [f'{lab[q]} → {NAME[t].lower()}' for q, t in rows[::-1]], fontsize=8.5, color=INK)
ax.set_xlim(-1.6, 2.6)
ax.set_xlabel('change in generated root ÷ true change, relative to the model\'s overall scale', fontsize=9,
              color=MUTED)
ax.tick_params(left=False)
clean(ax)
for m, e, c, l in series:
    ax.scatter([], [], color=c, label=l, s=26)
ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.4, -0.08), ncol=3)
ax.set_title('Swapping one QTL: how much of its true effect the generated root shows', loc='left', fontsize=12,
             color=INK)
fig.text(0.01, 0.01, '48 validation lines × 2 seeds, same seed before and after the edit. True change from the '
         'answer key\'s QTL effects\n(including background QTL inside the swapped block). 1 = the model\'s full '
         'scale (its overall slope of generated on true values).\n95% intervals over lines. ±10 cM aerenchyma '
         'swaps omitted: the block holds both linked QTL, whose effects cancel.',
         fontsize=7.5, color=MUTED, linespacing=1.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / 'qtl_swap_effect.png', dpi=150)
plt.close(fig)
print('wrote', FIG)
