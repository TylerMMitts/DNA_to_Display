# Paper-style architecture figure of the RIL v5 windows SNP encoder (best model, windows
# epoch 1200). Dimensions are read from the checkpoint. Writes PNG (300 dpi), PDF and SVG to
# results/ril_v5/figures/window_encoder_paper.*
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

sys.path.insert(0, r'D:\DNA_to_Display\code')
sys.path.insert(0, r'D:\DNA_to_Display\code\widiv')
from widiv_window_encoding import GatedDosageWindowEncoder

CK = 'D:/Downloads/d2dimages/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
OUT = Path(r'D:\DNA_to_Display\results\ril_v5\figures\window_encoder_paper')
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = ck['snp_encoder_config']
evr = float(np.sum(ck['snp_projector']['pca']['explained_variance_ratio']))
lam = ck['window_settings'][3]
NS, NP, NW, WW, WD = cfg['n_snps'], cfg['n_pcs'], cfg['n_windows'], cfg['window_width'], cfg['window_dim']
NIN, HID, NT, ED = NP + NW * WD, cfg['hidden_dim'], cfg['num_tokens'], cfg['embedding_dim']

plt.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'stixsans', 'font.size': 7})
INK, GREY, LGREY, LEARN, LEARN_D = '#1A1A1A', '#8C8C8C', '#E9E9E9', '#F6D88E', '#C98A00'
fig = plt.figure(figsize=(7.2, 3.15))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-7, 181); ax.set_ylim(-3, 75); ax.axis('off')


def block(x, y, w, h, learned, r=1.2):
    if learned:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=LEARN, ec=LEARN_D, lw=0.8))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc='white', ec=GREY, lw=0.8, ls=(0, (3, 2))))


def arr(p0, p1, lw=0.7, ms=6, color=INK, cs='arc3'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 connectionstyle=cs))


def t(x, y, s, size=7, ha='center', va='center', color=INK, **k):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, linespacing=1.25, **k)


def vec(x, y, w, h, n, shade_every=None, fc='white', ec=INK):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.7))
    for i in range(1, n):
        yy = y + h - i * h / n
        ax.plot([x, x + w], [yy, yy], color=GREY if shade_every is None or i % shade_every else INK, lw=0.25 if shade_every and i % shade_every else 0.5)


Y0, Y1 = 9, 63                      # vertical extent of the genotype vectors

# ---- input x --------------------------------------------------------------------------
vec(4, Y0, 4, Y1 - Y0, 10)
for c in range(10):
    if c % 2 == 0:
        ax.add_patch(Rectangle((4, Y1 - (c + 1) * (Y1 - Y0) / 10), 4, (Y1 - Y0) / 10, fc=LGREY, ec='none', zorder=0))
t(6, Y1 + 4, r'$\mathbf{x}$', 9)
t(6, Y0 - 3.5, r'$\{0,1,2\}^{%s}$' % f'{NS:,}'.replace(',', r'\,'), 6.5)
t(2.2, Y1 - 2.7, 'chr 1', 5.5, ha='right', color=GREY)
t(2.2, Y0 + 2.7, 'chr 10', 5.5, ha='right', color=GREY)

# ---- standardise ------------------------------------------------------------------------
arr((8, 36), (12, 36))
block(12, 29, 15, 14, False)
t(19.5, 39, 'Standardise', 7)
t(19.5, 33.5, r'$\mathbf{z}=(\mathbf{x}-\boldsymbol{\mu})\,/\,\boldsymbol{\sigma}$', 6.5)
arr((27, 36), (31, 36))

# ---- z, split into windows --------------------------------------------------------------
ZX, ZW = 31, 4
vec(ZX, Y0, ZW, Y1 - Y0, NW, shade_every=5)
t(ZX + ZW / 2, Y1 + 4, r'$\mathbf{z}$', 9)
wh = (Y1 - Y0) / NW


def win_y(k):                      # centre of window k (1-based), top = window 1
    return Y1 - (k - 0.5) * wh


# ---- branch 1: PCA ----------------------------------------------------------------------
PY = 63
ax.plot([ZX + ZW + 0.6, ZX + ZW + 1.6, ZX + ZW + 1.6, ZX + ZW + 0.6], [Y1, Y1, Y0, Y0], color=GREY, lw=0.5)
arr((ZX + ZW + 1.6, PY), (46, PY))
t(40.8, PY + 1.8, 'all SNPs', 5.3, color=GREY)
block(46, PY - 7, 30, 14, False)
t(61, PY + 3.2, f'PCA projection ({NP} PCs)', 7)
t(61, PY - 2.3, r'$\mathbf{p}=U\mathbf{z},\; U\in\mathbb{R}^{%d\times%s}$' % (NP, f'{NS:,}'.replace(',', r'\,')), 6.5)
t(61, PY - 9.6, f'{evr * 100:.1f}% of genotype variance', 5.8, color=GREY)

# ---- branch 2: windows ------------------------------------------------------------------
rows = [(1, 44.5), (2, 33.5), (NW, 15.5)]
for k, ry in rows:
    ax.add_patch(Rectangle((ZX, win_y(k) - wh / 2), ZW, wh, fc=LEARN_D, ec='none'))
    ax.plot([ZX + ZW, 46], [win_y(k), ry], color=GREY, lw=0.45)
    lab = str(k)
    block(46, ry - 4, 13, 8, True, r=0.8)
    t(52.5, ry + 0.9, r'$W_{%s}$' % lab, 7.5)
    t(52.5, ry - 2.3, r'$%d\!\rightarrow\!%d$' % (WW, WD), 5.5)
    arr((59, ry), (64.2, ry))
    ax.add_patch(Circle((67, ry), 2.8, fc=LEARN, ec=LEARN_D, lw=0.8))
    t(67, ry, r'$g_{%s}$' % lab, 6.5)
    arr((69.8, ry), (74, ry))
    for j in range(WD):
        ax.add_patch(Rectangle((74 + j * 2.2, ry - 1.4), 2.2, 2.8, fc='white', ec=INK, lw=0.5))
    t(78.4, ry - 4.4, r'$\mathbf{h}_{%s}$' % lab, 6.5)
t(52.5, 24.8, r'$\vdots$', 9)
t(67, 24.8, r'$\vdots$', 9)
t(78.4, 24.8, r'$\vdots$', 9)
t(62.5, 2.0, f'{NW} windows of {WW} adjacent SNPs, none crossing a chromosome\n'
            r'$\mathbf{h}_k=g_k\,(W_k\mathbf{z}_k+\mathbf{b}_k)$,  hard-concrete gate $g_k$, $L_0$ penalty $\lambda=10^{-6}$',
  5.8, color=GREY)

# ---- concatenate ------------------------------------------------------------------------
CX, CW = 92, 4
ch = Y1 - Y0
ph = ch * NP / NIN
ax.add_patch(Rectangle((CX, Y1 - ph), CW, ph, fc='white', ec=INK, lw=0.7))
ax.add_patch(Rectangle((CX, Y0), CW, ch - ph, fc=LEARN, ec=INK, lw=0.7))
for i in range(1, NW):
    yy = Y0 + i * (ch - ph) / NW
    ax.plot([CX, CX + CW], [yy, yy], color=LEARN_D, lw=0.2)
t(CX + CW / 2, Y1 + 4, 'concat', 6.5)
t(CX + CW + 1.2, Y1 - ph / 2, r'$\mathbf{p}$', 6.5, ha='left')
t(CX + CW + 1.2, (Y0 + Y1 - ph) / 2, r'$\mathbf{h}_{1..%d}$' % NW, 6.5, ha='left')
t(CX + CW / 2, Y0 - 3.5, r'$\mathbb{R}^{%d}$' % NIN, 6.5)
arr((76, PY), (CX, Y1 - ph / 2), cs='arc3,rad=0')
for k, ry in rows:
    cy = Y0 + (ch - ph) * (1 - (k - 0.5) / NW)
    arr((83.2, ry), (CX, cy))

# ---- MLP --------------------------------------------------------------------------------
arr((CX + CW + 6, 36), (104, 36))
block(104, 18, 27, 36, True)
t(117.5, 50, 'MLP', 7.5, weight='bold')
layers = [(r'Linear $%d\!\rightarrow\!%d$' % (NIN, HID), 'LayerNorm, SiLU, Dropout'),
          (r'Linear $%d\!\rightarrow\!%d$' % (HID, HID // 2), 'LayerNorm, SiLU, Dropout'),
          (r'Linear $%d\!\rightarrow\!%s$' % (HID // 2, f'{NT * ED:,}'.replace(',', r'\,')), None)]
for i, (a, b) in enumerate(layers):
    yy = 43.5 - i * 9.5
    ax.add_patch(Rectangle((105.5, yy - 3.6), 24, 7.2 if b else 5.2, fc='white', ec=LEARN_D, lw=0.5))
    t(117.5, yy + (1.4 if b else 0.0) - (0 if b else 1), a, 6.2)
    if b:
        t(117.5, yy - 1.8, b, 5.0, color=GREY)

# ---- tokens -----------------------------------------------------------------------------
arr((131, 36), (142, 36))
t(136.5, 40.4, 'reshape\n' + r'$+\,E_{\mathrm{pos}}$', 5.5, color=GREY)
TX, TY, TW, TH = 142, 22, 11, 28
for i in range(NT):
    ax.add_patch(Rectangle((TX, TY + TH - (i + 1) * TH / NT), TW, TH / NT, fc=LEARN if i % 2 == 0 else '#FBEBC4', ec=INK, lw=0.5))
t(TX + TW / 2, TY + TH + 4, r'$C\in\mathbb{R}^{%d\times%d}$' % (NT, ED), 6.5)
t(TX + TW / 2, TY - 3.5, f'{NT} tokens', 6, color=GREY)

# ---- U-Net ------------------------------------------------------------------------------
arr((TX + TW, 36), (158, 36))
ax.add_patch(FancyBboxPatch((158, 22), 22, 28, boxstyle='round,pad=0,rounding_size=1.2', fc=LGREY, ec=INK, lw=0.8))
t(169, 44, 'Denoising\nU-Net', 7.5, weight='bold')
t(169, 32.5, 'cross-attention\n' + r'keys, values $\leftarrow C$' + '\nat every step $t$', 5.8)

# ---- legend -----------------------------------------------------------------------------
lx, ly = 106, 6.5
ax.add_patch(FancyBboxPatch((lx, ly - 1.6), 5, 3.2, boxstyle='round,pad=0,rounding_size=0.6', fc='white', ec=GREY, lw=0.8, ls=(0, (3, 2))))
t(lx + 6.5, ly, 'fixed, fitted on training lines', 6, ha='left')
ax.add_patch(FancyBboxPatch((lx + 37, ly - 1.6), 5, 3.2, boxstyle='round,pad=0,rounding_size=0.6', fc=LEARN, ec=LEARN_D, lw=0.8))
t(lx + 43.5, ly, 'learned (denoising loss)', 6, ha='left')

for ext, dpi in (('png', 300), ('pdf', None), ('svg', None)):
    fig.savefig(OUT.with_suffix('.' + ext), dpi=dpi, facecolor='white')
print('wrote', OUT)
