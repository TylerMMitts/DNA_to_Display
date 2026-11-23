# Figures for ril_real_tissue_maps.py: (1) designed QTL effect sizes vs the same effects in the
# original images' phenotypes; (2) tissue maps from the original images, same layout/scale as
# qtl_tissue_maps.png; (3) original vs generated, map above map.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import ndimage

GM = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/'
AK = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/'
plt.rcParams.update({'font.family': 'Arial'})
INK, MUTED, GOLD = '#111111', '#666666', '#C98A00'
E = pd.read_csv(GM + 'qtl_effect_sizes_original.csv')
qt = pd.read_csv(AK + 'qtl_and_scan_baseline__qtl.csv')
pct = {(r.qtl, r.trait): r.pct_genetic_variance for _, r in qt.iterrows()}
# Trait heritability (answer key); a QTL's heritability = its share of the genetic variance x h2.
H2 = pd.read_csv(AK + 'qtl_and_scan_baseline__traits.csv').set_index('trait').h2.to_dict()
qh2 = lambda k, t: pct[(k, t)] * H2[t]
# A x A: going 0 -> 2 at the interacting QTL changes the coded x by 2, so the effect is 2 x the term.
E.loc[E.term == 'AxA', ['designed_effect', 'designed_effect_image_units', 'original_images_effect']] *= 2
eff = lambda k, t: float(E[(E.term == k) & (E.trait == t)].designed_effect_image_units.iloc[0])
INFO = {'shared_stele_mx': (f'stele {eff("shared_stele_mx", "stele_diam_um"):+.0f} µm, MX no. {eff("shared_stele_mx", "mx_n"):+.1f}',
                            f'h² stele {qh2("shared_stele_mx", "stele_diam_um"):.0f}%, MX no. {qh2("shared_stele_mx", "mx_n"):.0f}%'),
        'mx_diam_add': (f'MX diameter {eff("mx_diam_add", "mx_diam_um"):+.1f} µm', f'h² {qh2("mx_diam_add", "mx_diam_um"):.0f}%'),
        'mx_diam_epi': (f'MX diameter ±{abs(eff("AxA", "mx_diam_um")):.1f} µm (A×A)', f'h² {qh2("mx_diam_epi", "mx_diam_um"):.0f}%'),
        'lac_q': (f'lacunae {eff("lac_q", "n_lacunae"):+.1f}', f'h² {qh2("lac_q", "n_lacunae"):.0f}%')}
for k in ('rca_linked1', 'rca_linked2'):
    INFO[k] = (f'aerenchyma {eff(k, "rca_pct"):+.1f} %-points', f'h² {qh2(k, "rca_pct"):.0f}% (for the pair)')
for i in range(1, 9):
    k = f'ctx_q{i}'
    INFO[k] = (f'cortex width {eff(k, "cortex_width_um"):+.0f} µm', f'h² {qh2(k, "cortex_width_um"):.1f}%')

# ---- 1. effect sizes ---------------------------------------------------------------------
PAN = [('Stele diameter (µm)', 'stele_diam_um', [('shared_stele_mx', 'Shared QTL')]),
       ('Metaxylem number', 'mx_n', [('shared_stele_mx', 'Shared QTL')]),
       ('Metaxylem diameter (µm)', 'mx_diam_um', [('mx_diam_add', 'Additive QTL'), ('AxA', 'Interacting QTL (A×A)')]),
       ('Cortex width (µm)', 'cortex_width_um', [(f'ctx_q{i}', f'Cortex QTL {i}') for i in range(1, 9)]),
       ('Aerenchyma (% of cortex)', 'rca_pct', [('rca_linked1', 'Aerenchyma QTL 1'), ('rca_linked2', 'Aerenchyma QTL 2')]),
       ('Lacuna number', 'n_lacunae', [('lac_q', 'Lacuna QTL')])]
fig, axes = plt.subplots(len(PAN), 1, figsize=(9, 10.5), dpi=150,
                         gridspec_kw={'height_ratios': [len(p[2]) + 0.9 for p in PAN]})
for ax, (title, t, items) in zip(axes, PAN):
    d = E[E.trait == t].set_index('term')
    des = np.array([d.loc[k, 'designed_effect_image_units'] for k, _ in items])
    org = np.array([d.loc[k, 'original_images_effect'] for k, _ in items])
    y = np.arange(len(items))[::-1]
    ax.barh(y, des, height=0.55, color='#F1B82D', edgecolor='none', zorder=2)
    if np.isfinite(org).any():
        ax.scatter(org, y, s=60, color=INK, edgecolor='white', linewidth=1.5, zorder=3)
    lim = max(np.nanmax(np.abs(np.r_[des, org])), 1e-9) * 1.18
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color='#999999', lw=1, zorder=1)
    labs = []
    for k, lab in items:
        kk = 'mx_diam_epi' if k == 'AxA' else k
        labs.append(f'{lab}  (h² {qh2(kk, t):.3g}%)' if (kk, t) in pct else lab)
    ax.set_yticks(y, labs, fontsize=11, color=INK)
    ax.set_title(title, loc='left', fontsize=12.5, fontweight='bold', color=INK)
    ax.tick_params(axis='x', labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#E6E6E6', lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if t == 'n_lacunae':
        ax.text(0.01, -0.62, 'not measured on images (blind trait)', transform=ax.transAxes, fontsize=9, color=MUTED, style='italic')
h1 = plt.Rectangle((0, 0), 1, 1, color='#F1B82D')
h2 = plt.Line2D([], [], marker='o', ls='', color=INK, markeredgecolor='white', markersize=8)
fig.legend([h1, h2], ['Designed effect (answer key)', 'Measured on the original images'], loc='upper center',
           ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.55, 1.0))
fig.text(0.55, 0.004, 'Change in the trait when the QTL goes from dosage 0 to 2 (A×A: when the partner QTL carries allele 2; opposite sign with allele 0).\n'
         'h² = share of the phenotypic variance from that QTL (its share of genetic variance × trait heritability; aerenchyma: the linked pair).\n'
         'Aerenchyma designed effects converted from the liability scale to % with a linear fit. 1,978 non-test lines.',
         ha='center', fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.055, 1, 0.965), h_pad=1.2)
fig.savefig(GM + 'qtl_effect_sizes.png', facecolor='white', bbox_inches='tight')
print('wrote effect sizes')

# ---- 2/3. tissue maps --------------------------------------------------------------------
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
q = pd.read_csv(AK + 'designed_qtl_markers.csv').set_index('qtl')
R = np.load(GM + 'qtl_maps_original_images.npz')
G = np.load(GM + 'gene_maps_methods.npz')
cmap = LinearSegmentedColormap.from_list('gainloss', ['#2F5BD3', '#9DB4EE', '#FFFFFF', '#F7D27A', '#C98A00'])
panels = [('shared_stele_mx', 'stele', 'Shared QTL', 'stele diameter + metaxylem number'),
          ('mx_diam_add', 'vessels', 'Metaxylem diameter QTL', 'additive'),
          ('mx_diam_epi', 'vessels', 'Metaxylem diameter QTL', 'interacting (A×A)'),
          ('rca_linked1', 'aerenchyma', 'Aerenchyma QTL 1', 'aerenchyma'),
          ('rca_linked2', 'aerenchyma', 'Aerenchyma QTL 2', 'aerenchyma'),
          ('lac_q', 'aerenchyma', 'Lacuna QTL', 'lacuna number (blind trait)')] + \
         [(f'ctx_q{i}', 'cortex', f'Cortex QTL {i}', f'cortex width, {p}%') for i, p in zip(range(1, 9), (25, 15, 10, 7, 5, 3, 2, 1))]
VMAX = 0.25


def outline_of(tmpl):
    root = tmpl.sum(0) > 0.5
    o = np.zeros(root.shape, bool)
    for j in range(5):
        m = (tmpl.argmax(0) == j) & root
        o |= m ^ ndimage.binary_erosion(m)
    return root, o


def render(m, root, outline):
    d = np.sign(m) * np.sqrt(np.clip(np.abs(m) / VMAX, 0, 1))
    rgb = cmap(TwoSlopeNorm(0, -1, 1)(np.where(root, d, 0)))[..., :3]
    rgb[~root] = 1.0
    rgb[outline] = rgb[outline] * 0.55 + 0.45 * np.array([0.55, 0.55, 0.55])
    return rgb


def colorbar(fig, rect):
    cax = fig.add_axes(rect)
    cax.imshow(np.linspace(1, -1, 256)[:, None].repeat(12, 1), cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), aspect='auto')
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 127.5, 255], ['tissue\ngained', 'no change', 'tissue\nlost'], fontsize=14)
    cax.tick_params(length=0, pad=8)
    for s in cax.spines.values():
        s.set_visible(False)


rroot, rout = outline_of(R['template'])
groot, gout = outline_of(G['template'])
fig, axes = plt.subplots(2, 7, figsize=(24, 9.4), dpi=150)
for ax, (key, tissue, l1, l2) in zip(axes.ravel(), panels):
    ax.imshow(render(R[f'{key}__tissue'][COMP.index(tissue)], rroot, rout))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    short = {'shared_stele_mx': 'Shared stele/metaxylem QTL', 'mx_diam_add': 'Metaxylem diam. QTL (add.)',
             'mx_diam_epi': 'Metaxylem diam. QTL (A×A)'}.get(key, l1)
    ax.set_title(f'{short}\n{INFO[key][0]}\n{INFO[key][1]}\n{tissue} shown', fontsize=12.5, loc='left', linespacing=1.35)
fig.tight_layout(w_pad=0.8, h_pad=1.6)
fig.subplots_adjust(right=0.91)
colorbar(fig, [0.93, 0.16, 0.014, 0.66])
fig.savefig(GM + 'qtl_tissue_maps_original_images.png', facecolor='white', bbox_inches='tight')
print('wrote original-image maps')

fig, axes = plt.subplots(4, 7, figsize=(24, 17.5), dpi=130)
for i, (key, tissue, l1, l2) in enumerate(panels):
    r, c = 2 * (i // 7), i % 7
    j = COMP.index(tissue)
    axes[r, c].imshow(render(R[f'{key}__tissue'][j], rroot, rout))
    axes[r + 1, c].imshow(render(G[f'{key}__tissue'][j], groot, gout))
    short = {'shared_stele_mx': 'Shared stele/metaxylem QTL', 'mx_diam_add': 'Metaxylem diam. QTL (add.)',
             'mx_diam_epi': 'Metaxylem diam. QTL (A×A)'}.get(key, l1)
    axes[r, c].set_title(f'{short}\n{INFO[key][0]}\n{INFO[key][1]}\n{tissue} shown', fontsize=12.5, loc='left', linespacing=1.3)
    for a in (axes[r, c], axes[r + 1, c]):
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
for r in (0, 2):
    axes[r, 0].set_ylabel('Original images', fontsize=14, fontweight='bold')
    axes[r + 1, 0].set_ylabel('Generated (swap)', fontsize=14, fontweight='bold')
fig.tight_layout(w_pad=0.6, h_pad=0.8)
fig.subplots_adjust(right=0.91)
colorbar(fig, [0.93, 0.3, 0.012, 0.4])
fig.savefig(GM + 'qtl_tissue_maps_original_vs_generated.png', facecolor='white', bbox_inches='tight')
print('wrote comparison')
