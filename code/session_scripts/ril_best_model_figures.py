# Trait correlation figures for the best RIL model (windows e1200): generated
# validation-line means (4 seeds, valid roots) against the TRUE genetic values,
# next to GBLUP. Writes PNGs to results/ril_v5/figures/ and two CSVs for the doc.
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

FIG = WIDIV_RESULTS_DIR / 'figures'
FIG.mkdir(exist_ok=True)
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv')
G = G[G.line.map(split) == 'train'].set_index('line')
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
NAME = {'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma (RCA)', 'mcs_layers': 'Sclerenchyma (MCS)'}
UNIT = {'stele_diam_um': 'µm', 'mx_n': 'vessels', 'mx_diam_um': 'µm', 'cortex_width_um': 'µm',
        'rca_pct': '% (measured)', 'mcs_layers': 'band width (measured)'}
d = pd.read_csv(OUT / 'traits_windows_e1200.csv')
d = d[d.valid & (d.role == 'validation')]
gen = d.groupby('genotype')[T].mean().reindex(val)

# GBLUP on trained lines' observed phenotypes -> validation lines (as gblup_baseline.csv).
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X = dosage[[row_of[g] for g in tr + val]].astype(np.float64)
Z = (X - X[:len(tr)].mean(0)) / (X[:len(tr)].std(0) + 1e-6)
K = Z @ Z.T / Z.shape[1]
lam = pd.read_csv(OUT / 'gblup_baseline.csv').set_index('trait')['lambda']
gblup = {}
for t in T:
    y = meta.loc[tr, t].to_numpy(float)
    a = np.linalg.solve(K[:len(tr), :len(tr)] + lam[t] * np.eye(len(tr)), y - y.mean())
    gblup[t] = y.mean() + K[len(tr):, :len(tr)] @ a

rows_r, rows_pts = [], []
for t in T:
    g = G.loc[val, f'g_{t}'].to_numpy()
    m = gen[t].to_numpy()
    ok = np.isfinite(m)
    rows_r.append({'trait': NAME[t], 'slug': t, 'generated': round(float(np.corrcoef(m[ok], g[ok])[0, 1]), 3),
                   'gblup': round(float(np.corrcoef(gblup[t], g)[0, 1]), 3), 'n_lines': int(ok.sum())})
    for line, gi, mi in zip(val, g, m):
        if np.isfinite(mi):
            rows_pts.append({'trait': NAME[t], 'line': line, 'true_genetic_value': round(float(gi), 3),
                             'generated_mean': round(float(mi), 3)})
r = pd.DataFrame(rows_r)
r.to_csv(FIG / 'best_model_trait_r.csv', index=False)
pd.DataFrame(rows_pts).to_csv(FIG / 'best_model_scatter_points.csv', index=False)
print(r.to_string(index=False))

# PNG 1: scatter per trait.
BLUE, GREY, INK, MUTED = '#2a6fdb', '#9a9a9a', '#1f1f1f', '#6b6b6b'
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
for ax, t, rr in zip(axes.ravel(), T, rows_r):
    g = G.loc[val, f'g_{t}'].to_numpy()
    m = gen[t].to_numpy()
    ok = np.isfinite(m)
    ax.scatter(g[ok], m[ok], s=14, color=BLUE, alpha=0.65, linewidths=0)
    b = np.polyfit(g[ok], m[ok], 1)
    xs = np.linspace(g.min(), g.max(), 2)
    ax.plot(xs, np.polyval(b, xs), color=INK, lw=1)
    ax.set_title(f"{NAME[t]}   r = {rr['generated']:.2f}  (GBLUP {rr['gblup']:.2f})", fontsize=11, color=INK,
                 loc='left')
    ax.set_xlabel('true genetic value', fontsize=9, color=MUTED)
    ax.set_ylabel(f'generated mean, {UNIT[t]}', fontsize=9, color=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c8c8c8')
    ax.tick_params(colors=MUTED, labelsize=8)
fig.suptitle('Generated roots vs true genetic value, 198 validation lines (windows model, epoch 1200)',
             x=0.01, ha='left', fontsize=13, color=INK)
fig.text(0.01, 0.005, 'Each dot = one line: mean of its valid generated roots (4 seeds) measured by tissue colour. '
         'Genetic values from the answer key; never seen in training.', fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(FIG / 'best_model_trait_scatter.png', dpi=150)
plt.close(fig)

# PNG 2: r per trait, generated vs GBLUP (dot plot).
fig, ax = plt.subplots(figsize=(8, 4.2))
order = r.sort_values('generated').reset_index(drop=True)
for i, row in order.iterrows():
    ax.plot([row.generated, row.gblup], [i, i], color='#d9d9d9', lw=2, zorder=1)
    ax.scatter(row.gblup, i, s=60, color=GREY, zorder=2)
    ax.scatter(row.generated, i, s=60, color=BLUE, zorder=3)
    ax.text(row.generated + (0.012 if row.generated > row.gblup else -0.012), i, f'{row.generated:.2f}',
            va='center', ha='left' if row.generated > row.gblup else 'right', fontsize=9, color=INK)
ax.set_yticks(range(len(order)), order.trait, fontsize=10, color=INK)
ax.set_xlim(0.4, 1.0)
ax.set_xlabel('r with true genetic value (198 validation lines)', fontsize=9, color=MUTED)
ax.scatter([], [], color=BLUE, label='generated roots (windows e1200)')
ax.scatter([], [], color=GREY, label='GBLUP on SNPs')
ax.legend(frameon=False, fontsize=9, loc='lower right')
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=9, left=False)
ax.set_title(f"Generated roots reach r {r.generated.mean():.2f} on average; GBLUP {r.gblup.mean():.2f}",
             loc='left', fontsize=12, color=INK)
fig.tight_layout()
fig.savefig(FIG / 'best_model_trait_r.png', dpi=150)
plt.close(fig)
print('wrote', FIG)
