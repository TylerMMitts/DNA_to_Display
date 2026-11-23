# What the windows encoder's combined input looks like: for a few validation lines, the 32 PC
# scores and the 50 x 4 gated window features, concatenated into the 232 numbers the MLP
# reads. Best RIL v5 model (windows epoch 1200).
# Writes results/ril_v5/figures/window_encoder_input_vectors.png
import os
os.environ['WIDIV_DATASET'] = 'ril_v5'
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, r'D:\DNA_to_Display\code')
sys.path.insert(0, r'D:\DNA_to_Display\code\widiv')
from widiv_paths import WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

CK = 'D:/Downloads/d2dimages/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
OUT = WIDIV_RESULTS_DIR / 'figures' / 'window_encoder_input_vectors.png'
N_LINES = 10
ck = torch.load(CK, map_location='cpu', weights_only=False)
enc, proj = load_widiv_encoder(ck)
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
val = pd.read_csv(WIDIV_RESULTS_DIR / 'checkpoint_selection' / 'validation_lines.csv').genotype.tolist()[:N_LINES]
D = torch.tensor(dosage[[row_of[g] for g in val]])
with torch.no_grad():
    z = enc.standardize(D)
    pcs = (z @ enc.pca_components.T).numpy()
    win = (enc.window_features(z) * enc.gates()[None, :, None]).numpy()          # (n, 50, 4)
NP, NW, WD = pcs.shape[1], win.shape[1], win.shape[2]
X = np.concatenate([pcs, win.reshape(len(val), -1)], 1)                          # (n, 232)
print('PC scores sd', pcs.std().round(2), ' window features sd', win.std().round(2))
assert X.shape[1] == enc.net[0].in_features

# window -> chromosome, and which windows hold a designed QTL (for the labels only)
chrom = np.array(proj.window_chromosome)
qtl = pd.read_csv('D:/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/designed_qtl_markers.csv')
woc = proj.window_of_column()
qwin = {int(woc[list(snps).index(s)]) for s in qtl.snp}

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})
cmap = LinearSegmentedColormap.from_list('bwg', ['#2F5BD3', '#FFFFFF', '#C98A00'])
fig = plt.figure(figsize=(14, 4.6), dpi=170)
gs = fig.add_gridspec(1, 2, width_ratios=[NP, NW * WD], wspace=0.03, left=0.07, right=0.93, top=0.78, bottom=0.2)
axp, axw = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
# each block on its own symmetric scale: PC scores are much larger numbers than window features
for ax, M, title in ((axp, pcs, f'{NP} PC scores\n(population structure)'),
                     (axw, win.reshape(len(val), -1), f'{NW} windows × {WD} features = {NW * WD}\n(local genes)')):
    lim = np.quantile(np.abs(M), 0.98)
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=10, loc='left', pad=6)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
axp.set_yticks(range(len(val)), val, fontsize=8)
axp.set_xticks([0, NP - 1], ['PC 1', f'PC {NP}'], fontsize=8)
# chromosome boundaries and labels under the window block
for c in range(1, 10):
    axw.axvline(c * 5 * WD - 0.5, color='#111111', lw=0.8)
axw.set_xticks([(c + 0.5) * 5 * WD - 0.5 for c in range(10)], [f'chr {c + 1}' for c in range(10)], fontsize=8)
for k in range(1, NW):
    if k % 5:
        axw.axvline(k * WD - 0.5, color='#BBBBBB', lw=0.3)
n = len(val)
for k in sorted(qwin):
    axw.plot(k * WD + 1.5, n + 1.25, marker='^', color='#111111', ms=4.5, clip_on=False)
axw.text(NW * WD - 0.5, n + 2.6, '▲ window holding a designed QTL', ha='right', va='top', fontsize=7.5, color='#555555')
axw.tick_params(axis='x', pad=14, length=0)
axp.tick_params(axis='x', pad=14, length=0)
fig.text(0.07, 0.93, f'The {NP + NW * WD} numbers the MLP reads for {len(val)} validation lines: '
         f'[{NP} PC scores | {NW * WD} window features], one line per row', fontsize=11.5, fontweight='bold')
fig.text(0.07, 0.06, 'Colour: gold = above zero, blue = below zero. Each block is on its own scale '
         f'(PC scores spread {pcs.std():.1f}, window features {win.std():.2f}).', fontsize=8.5, color='#555555')
fig.savefig(OUT, facecolor='white')
print('wrote', OUT)
