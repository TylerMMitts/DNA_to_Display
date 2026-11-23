# Poster figure: genotype-to-phenotype maps. Average aligned root (labelled) + one map per
# gene: mean |change| in the generated image when that gene's chromosome region is swapped,
# 96 line-seed pairs each, one shared black-to-gold scale. Sized for the poster's right column.
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

P = 'E:/D2D/DNA_to_Display/presentation/poster_assets/'
z = np.load(P + 'gene_heatmaps.npz')
info = json.load(open(P + 'gene_heatmaps.json'))
GOLD, BLACK = '#F1B82D', '#111111'
cmap = LinearSegmentedColormap.from_list('mizzou', ['#111111', '#4A3600', '#B8860B', '#F1B82D', '#FFF4D6'])
panels = [('shared', 'Shared gene', 'stele size, vessel number'), ('mxdiam', 'Vessel-width gene', 'metaxylem diameter'),
          ('cortex', 'Cortex gene', 'cortex width'), ('rca', 'Aerenchyma gene', 'air channels'),
          ('lacuna', 'Lacuna gene', 'trait never measured'), ('control', 'Control region', 'no designed gene')]
C = 256 / 2
yy, xx = np.indices((256, 256))
r = np.hypot(yy - C + 0.5, xx - C + 0.5)
R = 256 / 2 / 1.3                      # root radius in the aligned frame
inside = r <= 1.04 * R
# Scale starts at the control region's typical change (the noise floor any edit causes), so
# only change beyond what an unrelated region produces lights up; one scale for every panel.
vmin = float(np.median(z['control'][inside]))
vmax = float(np.percentile(z['shared'][inside], 99.0))
norm = PowerNorm(gamma=0.8, vmin=vmin, vmax=vmax)

plt.rcParams.update({'font.family': 'Arial'})
fig, axes = plt.subplots(2, 4, figsize=(13.9, 8.1), dpi=200)
# reference panel
ax = axes[0, 0]
tmpl = z['template'].astype(np.uint8).copy()
tmpl[~inside] = 255
ax.imshow(tmpl)
for txt, (px, py), (tx, ty) in [('stele', (C, C), (C, C)), ('vessels', (C + 36, C - 14), (C + 74, C - 104)),
                                ('cortex', (C - 66, C + 30), (C - 118, C + 112)), ('outer layers', (C + 66, C + 70), (C + 40, C + 124))]:
    if txt == 'stele':
        ax.text(px, py, txt, ha='center', va='center', fontsize=20, fontweight='bold', color=BLACK)
    else:
        ax.annotate(txt, (px, py), (tx, ty), fontsize=20, fontweight='bold', color=BLACK, ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color=BLACK, lw=1.6))
ax.set_title('Average root', loc='left', fontsize=25, fontweight='bold', color=BLACK, pad=38)
ax.text(0, 1.02, 'what each region is', transform=ax.transAxes, fontsize=20, color='#555555', va='bottom')
for ax, (key, title, subtitle) in zip(list(axes.ravel()[1:]), panels):
    m = np.where(inside, z[key], 0)
    ax.imshow(m, cmap=cmap, norm=norm)
    ax.set_title(title, loc='left', fontsize=25, fontweight='bold', color=BLACK, pad=38)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=20, color='#B33A3A' if key == 'lacuna' else '#555555',
            fontweight='bold' if key == 'lacuna' else 'normal', va='bottom')
# legend panel
ax = axes[1, 3]
ax.axis('off')
grad = np.linspace(vmin, vmax, 256)[None, :].repeat(18, 0)
cax = ax.inset_axes([0.05, 0.42, 0.9, 0.12])
cax.imshow(grad, cmap=cmap, norm=norm, aspect='auto')
cax.set_xticks([0, 255], ['control level', 'large change'], fontsize=18)
cax.set_yticks([])
for s in cax.spines.values():
    s.set_visible(False)
ax.text(0.05, 0.98, 'How much the image\nchanges when only\nthis gene is swapped', transform=ax.transAxes, fontsize=19,
        fontweight='bold', va='top', color=BLACK)
ax.text(0.05, 0.2, f'{info["shared"]["n_pairs"]} line-seed pairs per gene,\nunseen lines, same noise\nbefore and after', transform=ax.transAxes,
        fontsize=18, va='top', color='#555555')
for ax in axes.ravel()[:7]:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(w_pad=0.6, h_pad=1.2)
fig.savefig(P + 'poster_gene_maps.png', dpi=200, facecolor='white')
print('wrote', P + 'poster_gene_maps.png', 'vmax', vmax)
