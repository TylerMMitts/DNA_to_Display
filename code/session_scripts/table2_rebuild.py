# Rebuilt Table 2. One predictor per QTL SNP (40), trained on the 733 training lines, scored on
# the 237 test lines: r between predicted and true dosage, for ridge, kNN (k = 5), MLP and
# gradient boosting, from the full, literal, strict-without-orthogonalization and strict inputs.
# Also: flexible trait predictors (boosting, MLP) from each input to real training-line means,
# put through the joint regression on test lines (QTL share, background share, ratio, CI).
import numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
exec(open(r'E:/D2D/DNA_to_Display/results/widiv/gwas/noqtl_comparison/compare_noqtl.py').read().split('per_qtl, pooled, hat = [], [], []')[0])
from widiv_window_encoding import load_widiv_encoder

tr, te = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'test')
Q = sorted(set(qtl.snp)); Y = np.stack([d[:, col[s]].astype(float) for s in Q], 1)
MK = {'Ridge': lambda: RidgeCV(alphas=np.logspace(-1, 5, 25)), 'kNN (k = 5)': lambda: KNeighborsRegressor(5),
      'MLP': lambda: MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-2, max_iter=400, early_stopping=True, random_state=0),
      'Gradient boosting': lambda: HistGradientBoostingRegressor(random_state=0)}
inputs = {}
for lab, run in (('Full', 'random_vaev3'), ('Literal', 'random_noqtl0kb_vaev3'), ('Strict', 'random_noqtl1000kb_orth_vaev3')):
    ck = torch.load(f'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_widiv_medium_{run}_epoch_{EP}.pt', map_location='cpu', weights_only=False)
    _, proj = load_widiv_encoder(ck, 'cpu'); del ck
    if lab == 'Strict':
        inputs['Strict, not orthogonalized'] = np.asarray(proj._standardize(d) @ proj.components_.T, float)
    inputs[lab] = np.asarray(proj.transform(d), float)
order = ['Full', 'Literal', 'Strict, not orthogonalized', 'Strict']
inputs = {k: (inputs[k] - inputs[k][tr].mean(0)) / (inputs[k][tr].std(0) + 1e-8) for k in order}

rows = []
for lab, X in inputs.items():
    for name, mk in MK.items():
        r = np.array([np.corrcoef(mk().fit(X[tr], Y[tr, j]).predict(X[te]), Y[te, j])[0, 1] for j in range(len(Q))])
        rows.append({'input': lab, 'predictor': name, 'mean_r': r.mean(), 'n_above_0.5': int((r > 0.5).sum())})
    print('dosage done', lab, flush=True)
A = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(A.pivot(index='predictor', columns='input', values='mean_r')[order].round(2).to_string())
print(A.pivot(index='predictor', columns='input', values='n_above_0.5')[order].to_string())
A.to_csv(OUT / 'table2_per_qtl_dosage.csv', index=False)

# flexible trait predictors through the joint regression
real = tables['real']
eff_all = np.concatenate([design[t][2] for t in T])
rng = np.random.default_rng(3)
out = []
for lab in ('Full', 'Literal', 'Strict'):
    X = inputs[lab]
    for name in ('Gradient boosting', 'MLP'):
        pred = pd.DataFrame(index=names, columns=T, dtype=float)
        for t in T:
            ok = tr[real.reindex(names[tr])[t].notna().to_numpy()]
            yv = real.loc[names[ok], t].to_numpy(float)
            pred[t] = MK[name]().fit(X[ok], (yv - yv.mean()) / yv.std()).predict(X) * yv.std() + yv.mean()
        idxs = {t: te[real.reindex(names[te])[t].notna().to_numpy()] for t in T}
        ys = {t: pred.loc[names[idxs[t]], t].to_numpy(float) / plot_sd[t] for t in T}
        res = []
        for b in range(301):
            qs, bs = [], []
            for t in T:
                s = np.arange(len(idxs[t])) if b == 0 else rng.integers(0, len(idxs[t]), len(idxs[t]))
                q, bb = fit(idxs[t][s], ys[t][s], t); qs.append(q); bs.append(bb)
            qv = eff_all @ np.concatenate(qs) / (eff_all @ eff_all)
            res.append((qv, np.mean(bs), qv / np.mean(bs)))
        res = np.array(res)
        lo, hi = np.percentile(res[1:, 2], [2.5, 97.5])
        out.append({'input': lab, 'predictor': name, 'qtl_share': res[0, 0], 'background_share': res[0, 1],
                    'ratio': res[0, 2], 'lo': lo, 'hi': hi})
        print('trait done', lab, name, flush=True)
B = pd.DataFrame(out)
print(B.round(3).to_string(index=False))
B.to_csv(OUT / 'flexible_trait_reference.csv', index=False)
