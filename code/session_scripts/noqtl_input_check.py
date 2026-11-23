# How much QTL information is left in each model's genotype input?
# For each checkpoint's projector, fit a ridge regression from the projected scores to
# (a) each QTL's dosage and (b) each trait's QTL part / background part of the true
# genetic value, on training lines, and score it on test lines (out-of-sample r).
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_data import load_split
from widiv_paths import WIDIV_GENETIC_VALUES, WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_QTL, WIDIV_SPLIT
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'rca_pct', 'mcs_layers']
MODELS = {'full': 'random_vaev3', 'literal': 'random_noqtl0kb_vaev3', 'strict': 'random_noqtl1000kb_orth_vaev3'}
lines, snps, d = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
names = np.array(lines)
col = {s: i for i, s in enumerate(snps)}
qtl = pd.read_csv(WIDIV_QTL)
split = load_split(WIDIV_SPLIT).reindex(names).split_random.to_numpy()
tr, te = split == 'train', split == 'test'
gv = pd.read_csv(WIDIV_GENETIC_VALUES).set_index('genotype').loc[lines]

targets = {}
for r in qtl.itertuples():
    targets[f'dosage {r.trait} {r.snp}'] = d[:, col[r.snp]].astype(float)
for t in T:
    part = sum(r.effect_per_alt_allele_sd * d[:, col[r.snp]] for r in qtl[qtl.trait == t].itertuples())
    targets[f'qtl_part {t}'] = part
    targets[f'background {t}'] = gv[f'g_{t}'].to_numpy() - part


def ridge_r(x, y):
    # lambda chosen by 5-fold CV inside the training lines, then fit on all of them.
    xm, ym = x[tr].mean(0), y[tr].mean()
    xt, yt = x[tr] - xm, y[tr] - ym
    folds = np.arange(tr.sum()) % 5
    best = None
    for lam in 10.0 ** np.arange(-1, 5):
        err = 0
        for f in range(5):
            a, b = folds != f, folds == f
            w = np.linalg.solve(xt[a].T @ xt[a] + lam * np.eye(x.shape[1]), xt[a].T @ yt[a])
            err += ((xt[b] @ w - yt[b]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, lam)
    w = np.linalg.solve(xt.T @ xt + best[1] * np.eye(x.shape[1]), xt.T @ yt)
    pred = (x[te] - xm) @ w
    return float(np.corrcoef(pred, y[te])[0, 1])


rows = {}
for label, run in MODELS.items():
    ck = torch.load(f'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_widiv_medium_{run}_epoch_400.pt', map_location='cpu',
                    weights_only=False)
    _, projector = load_widiv_encoder(ck, 'cpu')
    x = np.asarray(projector.transform(d), dtype=float)
    del ck
    rows[label] = {k: ridge_r(x, y) for k, y in targets.items()}
    print(label, x.shape, flush=True)

res = pd.DataFrame(rows)
res.to_csv(r'C:\Users\TylerM\AppData\Local\Temp\claude\5cea7d8aa087c68d\196eda90-cf5d-4266-a917-5c0f45e4b1a0\scratchpad\noqtl_input_check.csv')
pd.set_option('display.width', 200)
dos = res[res.index.str.startswith('dosage')]
print('\nQTL dosage recoverable from input (test-line r), over 30 designed QTL:')
print(dos.describe().loc[['mean', '50%', 'min', 'max']].round(3))
print('\nGenetic value parts recoverable from input (test-line r):')
print(res[~res.index.str.startswith('dosage')].round(3))
