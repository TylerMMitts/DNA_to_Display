# Reference: how much QTL effect can a plain predictor recover on test lines from each model's
# genotype input? Fit ridge (linear) and RBF kernel ridge from the model's projector features
# to real training-line means, predict test lines, and run the predictions through the same
# joint regression as compare_noqtl.py (QTL share, background share, QTL / background).
# A relatedness-style predictor reproduces QTL in proportion to background even with no QTL
# marker in the input, so this sets what "removed" can mean for a model of this kind.
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV

S = r'C:/Users/TylerM/AppData/Local/Temp/claude/5cea7d8aa087c68d/196eda90-cf5d-4266-a917-5c0f45e4b1a0/scratchpad/compare_noqtl.py'
exec(open(S).read().split('per_qtl, pooled, hat = [], [], []')[0])
from widiv_window_encoding import load_widiv_encoder

CKD = 'E:/D2D/DNA_to_Display/models/from_hellbender'
CK = {'full': 'random_vaev3', 'literal': 'random_noqtl0kb_vaev3', 'strict': 'random_noqtl1000kb_orth_vaev3'}
real = tables['real']
train_idx = np.flatnonzero(split == 'train')
test_idx = np.flatnonzero(split == 'test')

preds = {}
for lab, run in CK.items():
    ck = torch.load(f'{CKD}/diffusion_widiv_medium_{run}_epoch_{EP}.pt', map_location='cpu', weights_only=False)
    _, proj = load_widiv_encoder(ck, 'cpu')
    del ck
    X = np.asarray(proj.transform(d), dtype=np.float64)
    X = (X - X[train_idx].mean(0)) / (X[train_idx].std(0) + 1e-8)
    sq = ((X[train_idx[:300], None] - X[None, train_idx[:300]]) ** 2).sum(-1)
    gamma = 1 / np.median(sq[sq > 0])
    for kind in ('ridge', 'kernel'):
        out = pd.DataFrame(index=names, columns=T, dtype=float)
        for t in T:
            tr = train_idx[real.reindex(names[train_idx])[t].notna().to_numpy()]
            y = real.loc[names[tr], t].to_numpy(float)
            if kind == 'ridge':
                m = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(X[tr], y)
            else:
                m = GridSearchCV(KernelRidge(kernel='rbf', gamma=gamma), {'alpha': np.logspace(-3, 2, 11)},
                                 cv=5, scoring='r2').fit(X[tr], y - y.mean()).best_estimator_
                m.offset = y.mean()
            out[t] = m.predict(X) + getattr(m, 'offset', 0)
        preds[f'{kind} {lab}'] = out
    print('done', lab, flush=True)

rng = np.random.default_rng(1)
rows = []
eff_all = np.concatenate([design[t][2] for t in T])
for source, table in preds.items():
    idxs = {t: test_idx[real.reindex(names[test_idx])[t].notna().to_numpy()] for t in T}
    ys = {t: table.loc[names[idxs[t]], t].to_numpy(float) / plot_sd[t] for t in T}
    res = []
    for b in range(301):
        qs, bs = [], []
        for t in T:
            s = np.arange(len(idxs[t])) if b == 0 else rng.integers(0, len(idxs[t]), len(idxs[t]))
            q, bb = fit(idxs[t][s], ys[t][s], t); qs.append(q); bs.append(bb)
        qs = eff_all @ np.concatenate(qs) / (eff_all @ eff_all)
        res.append((qs, np.mean(bs), qs / np.mean(bs)))
    res = np.array(res)
    rows.append({'predictor': source, 'qtl_share': res[0, 0], 'background_share': res[0, 1],
                 'qtl_over_background': res[0, 2], 'lo': np.percentile(res[1:, 2], 2.5),
                 'hi': np.percentile(res[1:, 2], 97.5),
                 'r_real_test': np.mean([np.corrcoef(table.loc[names[idxs[t]], t], real.loc[names[idxs[t]], t])[0, 1] for t in T])})
R = pd.DataFrame(rows)
print(R.round(3).to_string(index=False))
R.to_csv(OUT / 'predictor_reference.csv', index=False)
