# Diagram of the RIL v5 windows SNP encoder (best model, windows epoch 1200): how one line's
# 10,000 SNPs become the 8 tokens the denoising U-Net reads. Numbers are read from the
# checkpoint so the figure stays true to it.
# Writes results/ril_v5/figures/window_encoder_diagram.png
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

CK = 'D:/Downloads/d2dimages/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
OUT = Path(r'D:\DNA_to_Display\results\ril_v5\figures\window_encoder_diagram.png')
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = ck['snp_encoder_config']
evr = np.asarray(ck['snp_projector']['pca']['explained_variance_ratio'])
la = ck['snp_encoder_state_dict']['log_alpha']
# open probability as GatedDosageWindowEncoder.open_probability (hard-concrete constants)
sys.path.insert(0, r'D:\DNA_to_Display\code')
sys.path.insert(0, r'D:\DNA_to_Display\code\widiv')
from widiv_window_encoding import GatedDosageWindowEncoder
enc = GatedDosageWindowEncoder(**cfg)
enc.load_state_dict(ck['snp_encoder_state_dict'], strict=False)
n_open = int((enc.open_probability().detach().numpy() > 0.5).sum())
NS, NP, NW, WW, WD = cfg['n_snps'], cfg['n_pcs'], cfg['n_windows'], cfg['window_width'], cfg['window_dim']
NIN = NP + NW * WD
HID = cfg['hidden_dim']
NT, ED = cfg['num_tokens'], cfg['embedding_dim']

BLACK, GOLD, GOLD_L, GOLD_T, GREY, MUTED, INK = '#111111', '#F1B82D', '#F7D27A', '#FDF3DA', '#F2F2F2', '#666666', '#111111'
plt.rcParams.update({'font.family': 'Arial'})
fig, ax = plt.subplots(figsize=(18, 8.2), dpi=160)
ax.set_xlim(0, 180); ax.set_ylim(0, 82); ax.axis('off')


def box(x, y, w, h, fc=GREY, ec='none', lw=0, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, label=None, dy=1.6, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=18, lw=2, color=BLACK, ls=ls,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha='center', va='bottom', fontsize=10.5, color=MUTED)


def text(x, y, s, size=12, weight='normal', color=INK, ha='left', va='top', **k):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.35, **k)


def genome_bar(x, y, w, h, split_windows=False, highlight=None):
    cw = w / 10
    for c in range(10):
        fc = GOLD if c % 2 == 0 else GOLD_L
        ax.add_patch(Rectangle((x + c * cw, y), cw, h, fc=fc, ec='white', lw=1.5))
        if split_windows:
            for j in range(1, 5):
                ax.plot([x + c * cw + j * cw / 5] * 2, [y, y + h], color='white', lw=0.9)
    if highlight is not None:
        c, j = highlight
        ax.add_patch(Rectangle((x + c * cw + j * cw / 5, y), cw / 5, h, fc=BLACK, ec=BLACK, lw=1))
    return cw


# ---- 1. genotype -------------------------------------------------------------------------
box(2, 26, 30, 30)
text(4, 53.5, 'Genotype of one line', 14, 'bold')
text(4, 48.6, f'{NS:,} SNPs on 10 chromosomes\ncoded 0 / 1 / 2', 11.5, color=MUTED)
genome_bar(4, 36.5, 26, 4)
for c in range(10):
    text(4 + 2.6 * c + 1.3, 35.6, str(c + 1), 8.5, color=MUTED, ha='center')
rng = np.random.default_rng(3)
text(4, 33.4, ' '.join(str(v) for v in rng.choice([0, 2], 12)) + ' …', 10.5, family='monospace')
text(4, 30.0, "standardised with the training\nlines' mean and SD", 9.5, color=MUTED)
arrow(32, 41, 40, 41)
# split
ax.plot([40, 40], [17, 66], color=BLACK, lw=2)
arrow(40, 66, 46, 66)
arrow(40, 17, 46, 17)

# ---- 2a. PCs ----------------------------------------------------------------------------
box(46, 52, 58, 27, fc=GOLD_T)
text(48.5, 77, 'Path 1 · Population structure', 14, 'bold')
text(48.5, 72.2, f'{NP} principal components (PCs)', 12.5, 'bold')
text(48.5, 68.3, f'Each PC is a weighted sum of all\n{NS:,} SNPs, not a group of them.\n'
                 f'Together the {NP} PCs hold {evr.sum() * 100:.0f}% of\n'
                 'the genotype variance: they say\nwhich lines are related to which.', 11, color=INK)
# mini scree
bx, by, bw, bh = 84, 56, 18, 12
mx = evr.max()
for i, v in enumerate(evr):
    ax.add_patch(Rectangle((bx + i * bw / NP, by), bw / NP * 0.78, v / mx * bh, fc=GOLD, ec='none'))
text(bx, by - 0.6, 'PC 1', 8.5, color=MUTED)
text(bx + bw, by - 0.6, f'PC {NP}', 8.5, color=MUTED, ha='right')
text(bx + bw / 2, by + bh + 2.6, 'variance held by each PC', 9, color=MUTED, ha='center')
text(48.5, 56.2, f'→ {NP} numbers', 12, 'bold')

# ---- 2b. windows ------------------------------------------------------------------------
box(46, 3, 58, 46)
text(48.5, 47, 'Path 2 · Local genes', 14, 'bold')
text(48.5, 42.2, f'{NW} windows of {WW} neighbouring SNPs', 12.5, 'bold')
text(48.5, 37.8, '5 per chromosome, never crossing one', 11, color=MUTED)
gx, gy, gw = 48.5, 29.5, 53
cw = genome_bar(gx, gy, gw, 4, split_windows=True, highlight=(4, 2))
hx = gx + 4 * cw + 2 * cw / 5
# zoom on one window
zx, zy = 52, 12.5
ax.plot([hx, zx], [gy, zy + 9], color=BLACK, lw=0.8, ls=':')
ax.plot([hx + cw / 5, zx + 16], [gy, zy + 9], color=BLACK, lw=0.8, ls=':')
box(zx, zy, 16, 9, fc='white', ec=BLACK, lw=1.2, r=0.8)
text(zx + 8, zy + 7.6, f'one window\n{WW} SNPs', 10.5, ha='center')
arrow(zx + 16, zy + 4.5, zx + 24, zy + 4.5)
box(zx + 24, zy, 13, 9, fc='white', ec=BLACK, lw=1.2, r=0.8)
text(zx + 30.5, zy + 7.6, f'its own\nweights\n{WW} → {WD}', 10.5, ha='center')
arrow(zx + 37, zy + 4.5, zx + 42, zy + 4.5)
# gate symbol
ax.add_patch(plt.Circle((zx + 45, zy + 4.5), 2.6, fc=GOLD, ec=BLACK, lw=1.2))
text(zx + 45, zy + 5.7, 'gate', 9.5, 'bold', ha='center')
text(zx + 45, zy - 0.8, 'on / off', 9, color=MUTED, ha='center')
text(48.5, 9.2, f'Every SNP has its own weight inside its window, so one\n'
                f'gene can steer the image directly. The gates can close\n'
                f'windows that do not help; here all {n_open} of {NW} stayed open.', 10.5, color=INK)
text(48.5, 0.6, '', 1)
text(88.5, 26, f'→ {NW} × {WD} = {NW * WD} numbers', 12, 'bold', ha='center')

# ---- 3. merge + network -----------------------------------------------------------------
arrow(104, 66, 112, 47)
arrow(104, 26, 112, 37)
box(112, 30, 26, 22)
text(125, 49.5, 'Combine', 14, 'bold', ha='center')
text(125, 44.5, f'{NP} + {NW * WD} = {NIN} numbers', 11.5, ha='center')
text(125, 39.5, 'shared network', 11.5, 'bold', ha='center')
text(125, 35.5, f'{NIN} → {HID} → {HID // 2} → {NT * ED:,}', 11.5, ha='center', color=MUTED)
arrow(138, 41, 145, 41)

# ---- 4. tokens --------------------------------------------------------------------------
for i in range(NT):
    ax.add_patch(Rectangle((145.5, 31.5 + i * 2.4), 9, 1.8, fc=GOLD if i % 2 == 0 else GOLD_L, ec='none'))
text(150, 54.5, f'{NT} tokens', 13, 'bold', ha='center')
text(150, 30.5, f'each {ED} numbers', 10, color=MUTED, ha='center')
arrow(155, 41, 161, 41)

# ---- 5. U-Net ---------------------------------------------------------------------------
box(161, 28, 17, 26, fc=BLACK)
text(169.5, 49.5, 'Denoising\nU-Net', 14, 'bold', color=GOLD, ha='center')
text(169.5, 40, 'reads the tokens\nat every one of\nthe 50 steps\n(cross-attention)', 10, color='white', ha='center')

fig.savefig(OUT, facecolor='white', bbox_inches='tight')
print('wrote', OUT)
