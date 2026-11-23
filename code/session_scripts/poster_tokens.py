# Poster "gene tokens" picture, redrawn so it reads as what it is: 8 tokens, each a
# list of numbers. Same checkpoint and genotype as the pipeline diagram
# (make_slide_assets.diagram_pipeline_overview: DIFFUSION_ONEHOT_MODEL, genotype 0).
# All 256 numbers per token cannot be shown legibly at 2.6 in, so each token shows the
# 14 dimensions that vary most between genotypes (z-scored across all 200 genotypes,
# so colour = above or below the average genotype), then an ellipsis.
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
from model_toolkit import load_session

OUT = r'E:/D2D/DNA_to_Display\presentation\poster_assets\pipe_tokens_v2.png'
N_SHOW, GENOTYPE = 10, 0
# The poster's pipeline figure: medium model, epoch 300 (make_talk_assets.py), genotype 0.
s = load_session(300)
names = list(s.genotypes)
T = np.asarray(s.embed(names))                                   # (genotypes, 8, 256)
n_tok, D = T.shape[1:]
Z = (T - T.mean(0)) / (T.std(0) + 1e-6)
print('genotype', names[GENOTYPE], 'tokens', T.shape)

# Dimensions shown: per token, the N_SHOW with the largest between-genotype spread,
# kept in their original order.
spread = T.std(0)                                               # (8, D)
dims = np.sort(np.argsort(-spread, axis=1)[:, :N_SHOW], axis=1)
vals = np.take_along_axis(Z[GENOTYPE], dims, axis=1).clip(-2, 2)

cmap = LinearSegmentedColormap.from_list('slate_gold', ['#3E5C8A', '#FFFFFF', '#E0A100'])
W, H = 2.6, 1.98
fig = plt.figure(figsize=(W, H), dpi=400)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
row_h, gap = 0.205, 0.04
top = (H - (n_tok * row_h + (n_tok - 1) * gap)) / 2
cell = 0.215
x0 = (W - (N_SHOW * cell + 0.3)) / 2
for t in range(n_tok):
    y = H - top - (t + 1) * row_h - t * gap
    for k in range(N_SHOW):
        ax.add_patch(FancyBboxPatch((x0 + k * cell + 0.012, y + 0.012), cell - 0.024, row_h - 0.024,
                                    boxstyle='round,pad=0,rounding_size=0.025',
                                    facecolor=cmap((vals[t, k] + 2) / 4), edgecolor='#1A1A1A', linewidth=0.45))
    ax.text(x0 + N_SHOW * cell + 0.06, y + row_h / 2 - 0.01, '…', ha='left', va='center', fontsize=12, color='#1A1A1A')
fig.savefig(OUT, transparent=True)
print('wrote', OUT)
