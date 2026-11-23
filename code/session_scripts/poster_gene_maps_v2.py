# Poster figure, v2: eight signed tissue maps (seven designed genes + a control) in a 2 x 4
# grid, with the gained/lost scale running vertically down the right side. Replaces the
# labelled average-root panel and the horizontal legend panel of tissue_map_figs.py; the
# maps themselves are drawn exactly as there (same data, stretch, colours and vmax).
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import ndimage

GM = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/'
PA = 'E:/D2D/DNA_to_Display/presentation/poster_assets/'
z = np.load(GM + 'gene_maps_methods.npz')
R = pd.read_csv(GM + 'gene_map_methods.csv')
R = R[R.method == 'tissue'].set_index('key')
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
tmpl = z['template']
root = tmpl.sum(0) > 0.5
outline = np.zeros(root.shape, bool)
for j in range(5):
    m = tmpl.argmax(0) == j
    outline |= (m & root) ^ ndimage.binary_erosion(m & root)
cmap = LinearSegmentedColormap.from_list('gainloss', ['#2F5BD3', '#9DB4EE', '#FFFFFF', '#F7D27A', '#C98A00'])
BLACK, MUTED = '#111111', '#555555'
plt.rcParams.update({'font.family': 'Arial'})
VMAX = 0.25


def draw(ax, m):
    # square-root stretch (same for every panel, control included) so small effects show
    d = np.sign(m) * np.sqrt(np.clip(np.abs(m) / VMAX, 0, 1))
    rgb = cmap(TwoSlopeNorm(0, -1, 1)(np.where(root, d, 0)))[..., :3]
    rgb[~root] = 1.0
    rgb[outline] = rgb[outline] * 0.55 + 0.45 * np.array([0.55, 0.55, 0.55])
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# (key, tissue drawn, title, subtitle). Every map shows the tissue the gene was designed
# to act on; the control shows its own strongest tissue, so it is not at a disadvantage.
panels = [('shared_stele_mx', 'stele', 'Shared gene', 'stele tissue'),
          ('mx_diam_add', 'vessels', 'Vessel-width gene', 'vessel tissue'),
          ('mx_diam_epi', 'vessels', 'Interacting gene', 'vessel tissue'),
          ('ctx_q1', 'cortex', 'Cortex gene 1', 'cortex tissue'),
          ('ctx_q2', 'cortex', 'Cortex gene 2', 'cortex tissue'),
          ('rca_linked1', 'aerenchyma', 'Aerenchyma gene', 'air-channel tissue'),
          ('lac_q', 'aerenchyma', 'Lacuna gene', 'trait never measured'),
          ('control6', None, 'Control region', 'no designed gene')]

fig = plt.figure(figsize=(13.9, 7.7), dpi=200)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.16], wspace=0.06, hspace=0.32,
                      left=0.005, right=0.93, top=0.89, bottom=0.01)
for i, (key, comp, title, sub) in enumerate(panels):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    if comp is None:
        comp = R.loc[key, 'points_to']
    draw(ax, z[f'{key}__tissue'][COMP.index(comp)])
    ax.set_title(title, loc='left', fontsize=23, fontweight='bold', color=BLACK, pad=36)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=20, va='bottom',
            color='#B33A3A' if key == 'lac_q' else MUTED, fontweight='bold' if key == 'lac_q' else 'normal')

# Vertical scale down the right side, gained at the top.
cax = fig.add_subplot(gs[:, 4])
pos = cax.get_position()
cax.set_position([pos.x0 + pos.width * 0.25, pos.y0 + pos.height * 0.12, pos.width * 0.4, pos.height * 0.76])
cax.imshow(np.linspace(VMAX, -VMAX, 256)[:, None].repeat(12, 1), cmap=cmap,
           norm=TwoSlopeNorm(0, -VMAX, VMAX), aspect='auto')
cax.set_xticks([])
cax.yaxis.tick_right()
cax.set_yticks([0, 127.5, 255], ['gained', 'no\nchange', 'lost'], fontsize=19)
cax.tick_params(axis='y', length=0, pad=6)
for s in cax.spines.values():
    s.set_visible(False)
fig.savefig(PA + 'poster_gene_maps.png', dpi=200, facecolor='white')
w, h = fig.get_size_inches() * 200
print('poster figure written', int(w), int(h))
