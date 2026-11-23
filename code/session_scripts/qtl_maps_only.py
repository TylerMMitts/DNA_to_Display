# The signed tissue map of every designed QTL, labelled with the QTL and its trait - images
# and labels only. Each map shows the tissue the QTL's trait lives in; gold = gained,
# blue = lost when the region is swapped toward the allele coded 2. Same stretch as the poster.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import ndimage

GM = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/'
z = np.load(GM + 'gene_maps_methods.npz')
q = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/designed_qtl_markers.csv').set_index('qtl')
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
tmpl = z['template']
root = tmpl.sum(0) > 0.5
outline = np.zeros(root.shape, bool)
for j in range(5):
    m = (tmpl.argmax(0) == j) & root
    outline |= m ^ ndimage.binary_erosion(m)
cmap = LinearSegmentedColormap.from_list('gainloss', ['#2F5BD3', '#9DB4EE', '#FFFFFF', '#F7D27A', '#C98A00'])
panels = [  # (qtl, tissue shown, label line 1, label line 2)
    ('shared_stele_mx', 'stele', 'Shared QTL', 'stele diameter + metaxylem number'),
    ('mx_diam_add', 'vessels', 'Metaxylem diameter QTL', 'additive'),
    ('mx_diam_epi', 'vessels', 'Metaxylem diameter QTL', 'interacting (A×A)'),
    ('rca_linked1', 'aerenchyma', 'Aerenchyma QTL 1', 'aerenchyma'),
    ('rca_linked2', 'aerenchyma', 'Aerenchyma QTL 2', 'aerenchyma'),
    ('lac_q', 'aerenchyma', 'Lacuna QTL', 'lacuna number (blind trait)'),
    ('ctx_q1', 'cortex', 'Cortex QTL 1', 'cortex width, 25%'),
    ('ctx_q2', 'cortex', 'Cortex QTL 2', 'cortex width, 15%'),
    ('ctx_q3', 'cortex', 'Cortex QTL 3', 'cortex width, 10%'),
    ('ctx_q4', 'cortex', 'Cortex QTL 4', 'cortex width, 7%'),
    ('ctx_q5', 'cortex', 'Cortex QTL 5', 'cortex width, 5%'),
    ('ctx_q6', 'cortex', 'Cortex QTL 6', 'cortex width, 3%'),
    ('ctx_q7', 'cortex', 'Cortex QTL 7', 'cortex width, 2%'),
    ('ctx_q8', 'cortex', 'Cortex QTL 8', 'cortex width, 1%'),
]
vmax = 0.25
plt.rcParams.update({'font.family': 'Arial'})
fig, axes = plt.subplots(2, 7, figsize=(24, 8.2), dpi=150)
for ax, (key, tissue, l1, l2) in zip(axes.ravel(), panels):
    m = z[f'{key}__tissue'][COMP.index(tissue)]
    d = np.sign(m) * np.sqrt(np.clip(np.abs(m) / vmax, 0, 1))
    rgb = cmap(TwoSlopeNorm(0, -1, 1)(np.where(root, d, 0)))[..., :3]
    rgb[~root] = 1.0
    rgb[outline] = rgb[outline] * 0.55 + 0.45 * np.array([0.55, 0.55, 0.55])
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'{l1}\n{l2}\n{q.loc[key, "snp"]}  ·  {tissue} shown', fontsize=13, loc='left', linespacing=1.35)
    if key == 'lac_q':
        ax.title.set_color('#B33A3A')
fig.tight_layout(w_pad=0.8, h_pad=1.6)
fig.subplots_adjust(right=0.91)
cax = fig.add_axes([0.93, 0.16, 0.014, 0.66])
cax.imshow(np.linspace(1, -1, 256)[:, None].repeat(12, 1), cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect='auto')
cax.set_xticks([])
cax.yaxis.tick_right()
cax.set_yticks([0, 127.5, 255], ['tissue\ngained', 'no change', 'tissue\nlost'], fontsize=14)
cax.tick_params(length=0, pad=8)
for s in cax.spines.values():
    s.set_visible(False)
fig.savefig(GM + 'qtl_tissue_maps.png', facecolor='white', bbox_inches='tight')
print('wrote', GM + 'qtl_tissue_maps.png')
