import sys, numpy as np, pandas as pd
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_paths import WIDIV_RESULTS_DIR
G = WIDIV_RESULTS_DIR / 'generated_all_lines'
import os
EP = os.environ.get('NOQTL_EPOCH', '400')
print(f'First-pass broken rate (seeds 0-3) by role, epoch {EP}:')
for lab, run in (('full', f'random_vaev3_e{EP}'), ('literal', f'random_noqtl0kb_vaev3_e{EP}'), ('strict', f'random_noqtl1000kb_orth_vaev3_e{EP}')):
    ok = pd.read_csv(G / run / 'traits.csv'); bad = pd.read_csv(G / f'{run}_rejected' / 'traits.csv')
    if 'valid' in ok: ok = ok[ok.valid]
    a = pd.concat([ok.assign(broken=False), bad.assign(broken=True)])
    a = a[a.seed < 4]
    print(f'  {lab:8s}', a.groupby('role').broken.mean().round(3).to_dict(), ' overall %.3f' % a.broken.mean())
# QTL share relative to background share, paired bootstrap (rerun the joint fit)
exec(open(r'C:/Users/TylerM/AppData/Local/Temp/claude/5cea7d8aa087c68d/196eda90-cf5d-4266-a917-5c0f45e4b1a0/scratchpad/compare_noqtl.py').read().split('per_qtl, pooled, hat = [], [], []')[0])
rng = np.random.default_rng(1)
rows = []
for source in ['real', 'full', 'literal', 'strict']:
    table = tables[source]
    for subset, mask in sets.items():
        res = []
        idxs = {t: np.flatnonzero(mask & np.isin(names, table.index[table[t].notna()])) for t in T}
        ys = {t: table.loc[names[idxs[t]], t].to_numpy(float) / plot_sd[t] for t in T}
        eff_all = np.concatenate([design[t][2] for t in T])
        for b in range(501):
            qs, bs = [], []
            for t in T:
                s = np.arange(len(idxs[t])) if b == 0 else rng.integers(0, len(idxs[t]), len(idxs[t]))
                q, bb = fit(idxs[t][s], ys[t][s], t); qs.append(q); bs.append(bb)
            res.append((eff_all @ np.concatenate(qs) / (eff_all @ eff_all)) / np.mean(bs))
        rows.append({'source': source, 'set': subset, 'qtl_over_background': round(res[0], 2),
                     'lo': round(np.percentile(res[1:], 2.5), 2), 'hi': round(np.percentile(res[1:], 97.5), 2)})
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(OUT / 'qtl_over_background.csv', index=False)
