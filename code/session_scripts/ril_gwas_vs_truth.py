# Scores the GWAS (ril_gwas.py, no answer key) against the answer key: for each designed
# QTL, was a locus found within 5 cM (the sender's criterion), how far off, and how much
# phenotypic variance it explains vs the truth (pct of genetic variance x trait h2) and vs
# the sender's R/qtl2 baseline scan. Every other GWAS locus is checked against the
# background QTL list. Plus a Manhattan figure. Answer key used for scoring only.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes

AK = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/'
GW = WIDIV_RESULTS_DIR / 'gwas'
FIG = WIDIV_RESULTS_DIR / 'figures'
WIN = 5.0
NAME = {'mx_n': 'Metaxylem number', 'stele_diam_um': 'Stele diameter', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma', 'mcs_layers': 'Sclerenchyma'}
TRAITS = list(NAME)
L = pd.read_csv(GW / 'gwas_loci.csv')
S = pd.read_csv(GW / 'gwas_scan.csv')
qt = pd.read_csv(AK + 'qtl_and_scan_baseline__qtl.csv')
qt = qt[qt.trait != 'n_lacunae']
bg = pd.read_csv(AK + 'qtl_and_scan_baseline__background_qtl.csv')
h2 = pd.read_csv(AK + 'qtl_and_scan_baseline__traits.csv').set_index('trait').h2

# Aerenchyma pair: the two loci are in repulsion, so their variance is only meaningful
# jointly (as the truth reports it). Refit both on the same phenotype the GWAS used.
meta, split = load_tables()
val = set(pd.read_csv(OUT / 'validation_lines.csv').genotype)
fit_lines = sorted(g for g in meta.index if g not in val)
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
col = {s: i for i, s in enumerate(snps)}
row_of = {g: i for i, g in enumerate(lines)}
Xf = dosage[[row_of[g] for g in fit_lines]].astype(float) - 1


def pair_pct(trait, markers):
    y = meta.loc[fit_lines, trait].to_numpy(float)
    if trait == 'rca_pct':
        y = stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))
    allm = L[(L.trait == trait) & (L.kind == 'main')].lead_snp.tolist()
    A = np.c_[np.ones(len(y)), Xf[:, [col[s] for s in allm]]]
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    k = [allm.index(s) for s in markers]
    return 100 * (A[:, [1 + i for i in k]] @ b[[1 + i for i in k]]).var(ddof=1) / y.var(ddof=1)


rows, used = [], set()
for _, q in qt.iterrows():
    kind = 'interaction' if q.qtl == 'mx_diam_epi' else 'main'
    c = L[(L.trait == q.trait) & (L.kind == kind) & (L.chr == q.chr) & ((L.pos_cM - q.pos_cM).abs() <= WIN)]
    row = {'trait': q.trait, 'qtl': q.qtl, 'type': q.type, 'chr': q.chr, 'true_pos_cM': q.pos_cM,
           'true_pct_genetic': q.pct_genetic_variance,
           'true_pct_phenotypic': q.pct_genetic_variance * h2[q.trait],
           'qtl2_found': q.qtl2_found, 'qtl2_pct_phenotypic': q.qtl2_pct_variance}
    if len(c):
        best = c.loc[(c.pos_cM - q.pos_cM).abs().idxmin()]
        used.add(best.name)
        row.update(gwas_found=True, gwas_marker=best.lead_snp, gwas_pos_cM=best.pos_cM,
                   distance_cM=abs(best.pos_cM - q.pos_cM), gwas_logp=best.logp,
                   gwas_pct_phenotypic=best.pct_phenotypic_var)
    else:
        row.update(gwas_found=False)
    rows.append(row)
D = pd.DataFrame(rows)
# repulsion pair: joint variance for both rows
rp = D[D.qtl.str.startswith('rca_linked')]
if rp.gwas_found.all():
    v = pair_pct('rca_pct', rp.gwas_marker.tolist())
    D.loc[rp.index, 'gwas_pct_phenotypic'] = v
    D.loc[rp.index, 'note'] = 'pair, joint variance (truth: 50% of genetic variance for the pair)'
    D.loc[rp.index, 'true_pct_phenotypic'] = 50 * h2['rca_pct']
D.to_csv(GW / 'gwas_vs_designed_qtl.csv', index=False)

# Every other GWAS locus vs the background QTL list.
other = []
inter_snps = set(zip(L[L.kind == 'interaction'].trait, L[L.kind == 'interaction'].lead_snp))
for i, l in L.drop(index=list(used)).iterrows():
    # skip interactions and the main-effect term of an interaction partner (part of that locus)
    if l.kind != 'main' or (l.trait, l.lead_snp) in inter_snps:
        continue
    b = bg[(bg.trait == l.trait) & (bg.chr == l.chr) & ((bg.pos_cM - l.pos_cM).abs() <= WIN)]
    big = b.pct_genetic_variance.max() if len(b) else 0.0
    tot = b.pct_genetic_variance.sum() if len(b) else 0.0
    other.append({'trait': l.trait, 'lead_snp': l.lead_snp, 'chr': l.chr, 'pos_cM': l.pos_cM, 'logp': l.logp,
                  'gwas_pct_phenotypic': l.pct_phenotypic_var, 'n_background_qtl_within_5cM': len(b),
                  'largest_background_pct_genetic': big, 'sum_background_pct_genetic': tot,
                  'verdict': ('background QTL (>=1% of genetic var nearby)' if big >= 1 else
                              'small background QTL nearby (<1%)' if len(b) else 'no known QTL (false positive)')})
O = pd.DataFrame(other)
O.to_csv(GW / 'gwas_other_loci_vs_background.csv', index=False)

pd.set_option('display.width', 250)
print(D[['trait', 'qtl', 'true_pct_genetic', 'true_pct_phenotypic', 'qtl2_found', 'qtl2_pct_phenotypic', 'gwas_found',
         'gwas_marker', 'distance_cM', 'gwas_logp', 'gwas_pct_phenotypic']].round(2).to_string(index=False))
print()
print(O.round(2).to_string(index=False))
print()
print(O.groupby(['trait', 'verdict']).size().to_string())
# Background QTL >= 2% of genetic variance: how many did the GWAS find?
big_bg = bg[(bg.pct_genetic_variance >= 2) & (bg.trait != 'n_lacunae')]   # lacunae: blind, no phenotype
hit = [((L.trait == b.trait) & (L.chr == b.chr) & ((L.pos_cM - b.pos_cM).abs() <= WIN)).any() for _, b in big_bg.iterrows()]
print(f'\nbackground QTL with >= 2% of genetic variance: {len(big_bg)}, found by GWAS: {sum(hit)}')
print(big_bg.assign(found=hit)[['trait', 'chr', 'pos_cM', 'pct_genetic_variance', 'found']].to_string(index=False))

# Manhattan figure.
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#d9d9d7', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
CAP = 60
chr_len = S.groupby('chr').pos_cM.max()
off = pd.Series(np.r_[0, np.cumsum(chr_len.values[:-1] + 10)], index=chr_len.index)
fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True, facecolor=SURF)
thr = -np.log10(0.05 / 10000)
for ax, t in zip(axes, TRAITS):
    ax.set_facecolor(SURF)
    s = S[S.trait == t]
    x = s.pos_cM + s.chr.map(off)
    ax.scatter(x, s.logp.clip(upper=CAP), s=3, c=np.where(s.chr % 2, '#9a9a96', '#c8c8c4'), linewidths=0)
    ax.axhline(thr, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    for _, q in qt[qt.trait == t].iterrows():
        ax.axvline(q.pos_cM + off[q.chr], color=ORANGE, lw=1.2, alpha=0.8, zorder=0)
    for _, l in L[(L.trait == t) & ~((L.kind == 'main') & L.logp.isna())].iterrows():
        xx = l.pos_cM + off[l.chr]
        yy = min(l.logp, CAP) if np.isfinite(l.logp) else CAP
        ax.scatter(xx, yy, s=40, facecolor='none' if l.kind == 'interaction' else BLUE, edgecolor=BLUE, lw=1.5,
                   zorder=3)
        if l.logp > CAP:
            ax.text(xx + 8, CAP - 4, f'{l.logp:.0f}', fontsize=7.5, color=INK, va='top')
    ax.set_ylim(0, CAP + 4)
    ax.set_title(NAME[t], loc='left', fontsize=10.5, color=INK, pad=2)
    ax.set_ylabel('−log10 p', fontsize=8.5, color=MUTED)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[-1].set_xticks(off.values + chr_len.values / 2, [f'chr {c}' for c in chr_len.index])
h = [plt.Line2D([], [], color=ORANGE, lw=1.5), plt.Line2D([], [], marker='o', color=BLUE, ls='', ms=6),
     plt.Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor=BLUE, ls='', ms=6),
     plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 2)))]
fig.legend(h, ['designed QTL (answer key)', 'GWAS locus', 'GWAS interaction locus', 'Bonferroni 5% threshold'],
           loc='upper right', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 0.995))
fig.suptitle('GWAS on the 1,780 training lines vs the designed QTL', x=0.01, y=0.995, ha='left', fontsize=13,
             color=INK)
fig.text(0.01, 0.005, 'Points: single-marker mixed-model scan (LOCO kinship). Blue: loci kept by the multi-locus '
         'and interaction scans (marker at its entry p). −log10 p capped at 60; true value printed.',
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.015, 1, 0.975))
fig.savefig(FIG / 'gwas_vs_truth_manhattan.png', dpi=140)
print('wrote', FIG / 'gwas_vs_truth_manhattan.png')
