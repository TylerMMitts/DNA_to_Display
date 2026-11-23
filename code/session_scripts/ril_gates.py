# Window gates of the RIL windows model vs each window's share of the genetic
# signal (GBLUP marker effects fitted on trained lines' phenotypes).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, load_tables
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

meta, split = load_tables()
val = set(pd.read_csv(OUT / 'validation_lines.csv').genotype)
tr = sorted(g for g in meta.index if g not in val)
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X = dosage[[row_of[g] for g in tr]].astype(np.float64)
Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
K = Z @ Z.T / Z.shape[1]
rows = {}
for e in (400, 800):
    ck = torch.load(rf'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_{e}.pt',
                    map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck)
    with torch.no_grad():
        rows[e] = enc.gates().numpy()
    win = proj.windows
cols = {}
for t in TRAITS:
    y = meta.loc[tr, t].to_numpy(float)
    y = (y - y.mean()) / y.std()
    alpha = np.linalg.solve(K + 1.0 * np.eye(len(y)), y)
    beta = Z.T @ alpha / Z.shape[1]
    cols[t] = [np.var(Z[:, w] @ beta[w]) for w in win]
imp = pd.DataFrame(cols)
imp = imp / imp.sum(0)
imp['mean_share'] = imp[TRAITS].mean(1)
imp['gate_e400'], imp['gate_e800'] = rows[400], rows[800]
imp.to_csv(OUT / 'window_gates.csv', index_label='window')
g = imp.gate_e800
print(f'{len(g)} windows; gate e800 min {g.min():.3f} median {g.median():.3f} max {g.max():.3f}; '
      f'closed (0): {(g <= 0).sum()}; e400 range {imp.gate_e400.min():.3f}-{imp.gate_e400.max():.3f}')
print(f'gate vs window genetic share (Spearman): e400 {spearmanr(imp.gate_e400, imp.mean_share)[0]:.2f}, '
      f'e800 {spearmanr(g, imp.mean_share)[0]:.2f}')
for t in TRAITS:
    print(f'  {t:16s} rho {spearmanr(g, imp[t])[0]:+.2f}')
print('window genetic share: max', round(imp.mean_share.max(), 3), 'min', round(imp.mean_share.min(), 4))
