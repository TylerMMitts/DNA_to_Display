# Does the strict input carry the QTL nonlinearly? From each model's projector features of
# the training lines, predict the dosage of each of the 30 designed QTL with ridge, k-nearest
# neighbours (k = 1, 5, 20) and a small MLP, and score r with the true dosage on test lines.
# Also kNN on real line means (relatedness-style prediction) through the joint regression.
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

S = r'C:/Users/TylerM/AppData/Local/Temp/claude/5cea7d8aa087c68d/196eda90-cf5d-4266-a917-5c0f45e4b1a0/scratchpad/compare_noqtl.py'
exec(open(S).read().split('per_qtl, pooled, hat = [], [], []')[0])
from widiv_window_encoding import load_widiv_encoder

CKD = 'E:/D2D/DNA_to_Display/models/from_hellbender'
CK = {'full': 'random_vaev3', 'literal': 'random_noqtl0kb_vaev3', 'strict': 'random_noqtl1000kb_orth_vaev3'}
tr, te = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'test')
Q = np.concatenate([design[t][1] for t in T], 1)          # (lines, 30) QTL dosages
real = tables['real']
eff_all = np.concatenate([design[t][2] for t in T])


def r_cols(a, b):
    return np.nanmean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(b.shape[1]) if b[:, j].std() > 0])


rows, share_rows = [], []
for lab, run in CK.items():
    ck = torch.load(f'{CKD}/diffusion_widiv_medium_{run}_epoch_{EP}.pt', map_location='cpu', weights_only=False)
    _, proj = load_widiv_encoder(ck, 'cpu')
    del ck
    X = np.asarray(proj.transform(d), dtype=np.float64)
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)
    models = {'ridge': RidgeCV(alphas=np.logspace(-1, 5, 25)),
              'knn1': KNeighborsRegressor(1), 'knn5': KNeighborsRegressor(5), 'knn20': KNeighborsRegressor(20),
              'mlp': MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-2, max_iter=400, early_stopping=True, random_state=0)}
    for name, m in models.items():
        p = m.fit(X[tr], Q[tr]).predict(X[te])
        rows.append({'input': lab, 'predictor': name, 'qtl_dosage_r_test': r_cols(p, Q[te])})
    # relatedness-style trait prediction: mean of the k nearest training lines' real means
    for k in (1, 5, 20):
        pred = pd.DataFrame(index=names, columns=T, dtype=float)
        for t in T:
            ok = tr[real.reindex(names[tr])[t].notna().to_numpy()]
            m = KNeighborsRegressor(k).fit(X[ok], real.loc[names[ok], t].to_numpy(float))
            pred[t] = m.predict(X)
        qs, bs = [], []
        for t in T:
            idx = te[real.reindex(names[te])[t].notna().to_numpy()]
            q, b = fit(idx, pred.loc[names[idx], t].to_numpy(float) / plot_sd[t], t); qs.append(q); bs.append(b)
        qs = eff_all @ np.concatenate(qs) / (eff_all @ eff_all)
        share_rows.append({'input': lab, 'predictor': f'knn{k} traits', 'qtl_share': qs,
                           'background_share': np.mean(bs), 'qtl_over_background': qs / np.mean(bs)})
    print('done', lab, flush=True)

pd.set_option('display.width', 200)
A = pd.DataFrame(rows).pivot(index='predictor', columns='input', values='qtl_dosage_r_test')
print('\nr between predicted and true QTL dosage on test lines (mean over 30 QTL):')
print(A.round(3).to_string())
B = pd.DataFrame(share_rows)
print('\nkNN trait prediction on test lines through the joint regression:')
print(B.round(3).to_string(index=False))
A.to_csv(OUT / 'nonlinear_qtl_leak.csv')
B.to_csv(OUT / 'knn_trait_reference.csv', index=False)
