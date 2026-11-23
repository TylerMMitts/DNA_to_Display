# Signed tissue maps (where a tissue is gained or lost when a gene is swapped toward the
# allele coded 2), drawn (1) for the poster: 6 genes + control + labelled average root, and
# (2) as the full validation figure: all 14 designed QTL + 6 controls, each showing the tissue
# its map points to, with right/wrong against the answer key and the method comparison.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

GM = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/'
PA = 'E:/D2D/DNA_to_Display/presentation/poster_assets/'
z = np.load(GM + 'gene_maps_methods.npz')
R = pd.read_csv(GM + 'gene_map_methods.csv')
R = R[R.method == 'tissue'].set_index('key')
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
tmpl = z['template']
root = tmpl.sum(0) > 0.5
outline = np.zeros(root.shape, bool)
from scipy import ndimage
for j in range(5):
    m = tmpl.argmax(0) == j
    outline |= (m & root) ^ ndimage.binary_erosion(m & root)
cmap = LinearSegmentedColormap.from_list('gainloss', ['#2F5BD3', '#9DB4EE', '#FFFFFF', '#F7D27A', '#C98A00'])
BLACK, MUTED = '#111111', '#555555'
plt.rcParams.update({'font.family': 'Arial'})


def tissue_map(key, comp):
    return z[f'{key}__tissue'][COMP.index(comp)]


def draw(ax, m, vmax):
    # square-root stretch (same for every panel, control included) so small effects show
    d = np.sign(m) * np.sqrt(np.clip(np.abs(m) / vmax, 0, 1))
    rgb = cmap(TwoSlopeNorm(0, -1, 1)(np.where(root, d, 0)))[..., :3]
    rgb[~root] = 1.0
    rgb[outline] = rgb[outline] * 0.55 + 0.45 * np.array([0.55, 0.55, 0.55])
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------- poster figure ----------
panels = [('shared_stele_mx', 'stele', 'Shared gene', 'stele tissue'), ('mx_diam_add', 'vessels', 'Vessel-width gene', 'vessel tissue'),
          ('ctx_q1', 'cortex', 'Cortex gene', 'cortex tissue'), ('rca_linked1', 'aerenchyma', 'Aerenchyma gene', 'air-channel tissue'),
          ('lac_q', 'aerenchyma', 'Lacuna gene', 'trait never measured'), ('control6', None, 'Control region', 'no designed gene')]
vmax = 0.25
fig, axes = plt.subplots(2, 4, figsize=(13.9, 8.1), dpi=200)
ax = axes[0, 0]
t_rgb = np.ones(root.shape + (3,))
cols = {'vessels': (0.35, 0.35, 0.35), 'stele': (0.82, 0.74, 0.62), 'aerenchyma': (1, 1, 1), 'outer': (0.45, 0.5, 0.4), 'cortex': (0.83, 0.88, 0.8)}
for j, c in enumerate(COMP):
    t_rgb[(tmpl.argmax(0) == j) & root] = cols[c]
ax.imshow(t_rgb)
C = 128
for txt, (px, py), (tx, ty) in [('stele', (C, C), (C, C)), ('vessels', (C + 36, C - 14), (C + 74, C - 104)),
                                ('cortex', (C - 66, C + 30), (C - 118, C + 112)), ('outer layers', (C + 66, C + 70), (C + 40, C + 124))]:
    if txt == 'stele':
        ax.text(px, py, txt, ha='center', va='center', fontsize=20, fontweight='bold', color=BLACK)
    else:
        ax.annotate(txt, (px, py), (tx, ty), fontsize=20, fontweight='bold', color=BLACK, ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=BLACK, lw=1.6))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('Average root', loc='left', fontsize=25, fontweight='bold', color=BLACK, pad=38)
ax.text(0, 1.02, 'tissue layout', transform=ax.transAxes, fontsize=20, color=MUTED, va='bottom')
for ax, (key, comp, title, sub) in zip(list(axes.ravel()[1:7]), panels):
    if comp is None:     # control: its strongest tissue, so it is not shown at a disadvantage
        comp = R.loc[key, 'points_to']
    draw(ax, tissue_map(key, comp), vmax)
    ax.set_title(title, loc='left', fontsize=25, fontweight='bold', color=BLACK, pad=38)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=20, color='#B33A3A' if key == 'lac_q' else MUTED,
            fontweight='bold' if key == 'lac_q' else 'normal', va='bottom')
ax = axes[1, 3]
ax.axis('off')
cax = ax.inset_axes([0.05, 0.42, 0.9, 0.12])
cax.imshow(np.linspace(-vmax, vmax, 256)[None, :].repeat(18, 0), cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
cax.set_xticks([0, 128, 255], ['lost', 'no change', 'gained'], fontsize=18)
cax.set_yticks([])
for s in cax.spines.values():
    s.set_visible(False)
ax.text(0.05, 0.98, 'Where the gene\'s own\ntissue is gained or lost\nwhen only it is swapped', transform=ax.transAxes,
        fontsize=19, fontweight='bold', va='top', color=BLACK)
ax.text(0.05, 0.2, '96 pairs per gene, unseen\nlines, same noise before\nand after', transform=ax.transAxes, fontsize=18, va='top', color=MUTED)
fig.tight_layout(w_pad=0.6, h_pad=1.2)
fig.savefig(PA + 'poster_gene_maps.png', dpi=200, facecolor='white')
print('poster figure written')

# ---------- full validation figure ----------
NAME = {'shared_stele_mx': 'Shared stele/vessel gene', 'mx_diam_epi': 'Vessel-width, interacting', 'mx_diam_add': 'Vessel-width, additive',
        'lac_q': 'Lacuna gene (blind trait)', 'rca_linked1': 'Aerenchyma gene 1', 'rca_linked2': 'Aerenchyma gene 2',
        **{f'ctx_q{k}': f'Cortex gene {k} ({p}%)' for k, p in zip(range(1, 9), (25, 15, 10, 7, 5, 3, 2, 1))},
        **{f'control{k}': f'Control region {k}' for k in range(1, 7)}}
order = list(R[~R.index.str.startswith('control')].sort_values('pct_var', ascending=False).index) + [f'control{k}' for k in range(1, 7)]
S = pd.read_csv(GM + 'gene_map_methods_summary.csv')
fig = plt.figure(figsize=(24, 12.5), dpi=150)
gs = fig.add_gridspec(4, 7, width_ratios=[1, 1, 1, 1, 1, 0.25, 2.4], wspace=0.08, hspace=0.36)
for i, k in enumerate(order):
    ax = fig.add_subplot(gs[i // 5, i % 5])
    r = R.loc[k]
    draw(ax, tissue_map(k, r.points_to), vmax)
    ax.set_title(NAME[k], fontsize=11, fontweight='bold', loc='left')
    if k.startswith('control'):
        sub = f'strongest: {r.points_to} (z {r.top_z:.1f}), not called'
    else:
        verdict = 'right' if r.correct else 'wrong'
        sub = f'→ {r.points_to} (z {r.top_z:.1f}) {verdict}' + ('' if r.called else ', below 3')
    ax.text(0.0, -0.08, sub, transform=ax.transAxes, fontsize=10, color=BLACK, va='top')
ax = fig.add_subplot(gs[:2, 6])
ax.axis('off')
labels = {'abs': 'Absolute colour change', 'signed': 'Signed colour change', 'tissue': 'Signed tissue maps',
          'tissue_norm': 'Signed tissue, size removed'}
cells = [[labels[r.method], r.correct_all, f'{r.correct_called}/{r.called}', r.cortex_genes_correct, r.structure_genes_correct,
          str(r['controls_called (false +)'])] for _, r in S.iterrows()]
HDR = ['Method', 'Right\ntissue', 'Right /\ncalled', 'Cortex\ngenes', 'Structure\ngenes', 'False\ncalls']
tb = ax.table(cellText=cells, colLabels=HDR,
              loc='upper left', cellLoc='center', colWidths=[0.40, 0.12, 0.12, 0.12, 0.13, 0.11])
tb.auto_set_font_size(False); tb.set_fontsize(12); tb.scale(1, 2.8)
for j in range(6):
    tb[(3, j)].set_facecolor('#FDF3DA')
ax.set_title('Which map type points to the right tissue? (ground truth: 14 designed QTL)', loc='left', fontsize=13, fontweight='bold')
fig.suptitle('Signed tissue maps for every designed QTL and 6 control regions: tissue gained (gold) or lost (blue) when only that region is swapped',
             x=0.01, ha='left', fontsize=14, fontweight='bold')
fig.savefig(GM + 'gene_map_methods.png', facecolor='white', bbox_inches='tight')
print('validation figure written')
