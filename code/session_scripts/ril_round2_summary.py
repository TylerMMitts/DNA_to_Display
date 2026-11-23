# Every scored RIL checkpoint side by side: validation loss, genotype gain and
# memorisation ratio, valid roots, r of generated validation-line means with the
# TRUE genetic values per trait, and noise copying on trained lines (r of
# generated trait with the non-genetic part of that line's training phenotype).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd

meta, split = load_tables()
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv')
G = G[G.line.map(split) == 'train'].set_index('line')
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
summ = pd.read_csv(OUT / 'summary.csv')
rows = []
for f in sorted(OUT.glob('traits_*_e*.csv')):
    run, e = f.stem.replace('traits_', '').rsplit('_e', 1)
    d = pd.read_csv(f)
    s = summ[(summ.run == run) & (summ.epoch == int(e))]
    row = {'model': run, 'epoch': int(e)}
    if len(s):
        s = s.iloc[0]
        row.update(val_loss=s.val_loss, gain_val=s.gain_val, ratio=s.gain_trained / s.gain_val)
    row['valid'] = d[d.role == 'validation'].valid.mean()
    v = d[d.valid]
    mv = v[v.role == 'validation'].groupby('genotype')[T].mean()
    mt = v[v.role == 'trained'].groupby('genotype')[T].mean()
    rg, cp = [], []
    for t in T:
        g = G.loc[mv.index, f'g_{t}']
        row[f'r_{t}'] = np.corrcoef(mv[t], g)[0, 1]
        rg.append(row[f'r_{t}'])
        gt, ot = G.loc[mt.index, f'g_{t}'], meta.loc[mt.index, t]
        noise = ot - np.polyval(np.polyfit(gt, ot, 1), gt)
        cp.append(np.corrcoef(mt[t], noise)[0, 1])
        row[f'copy_{t}'] = cp[-1]
    row['r_mean'], row['copy_mean'] = np.mean(rg), np.mean(cp)
    rows.append(row)
r = pd.DataFrame(rows).sort_values(['model', 'epoch'])
r.to_csv(OUT / 'round2_comparison.csv', index=False)
pd.set_option('display.width', 260)
short = {'stele_diam_um': 'stele', 'mx_n': 'mx_n', 'mx_diam_um': 'mx_diam', 'cortex_width_um': 'cortex',
         'rca_pct': 'rca', 'mcs_layers': 'mcs'}
cols = ['model', 'epoch', 'val_loss', 'gain_val', 'ratio', 'valid'] + [f'r_{t}' for t in T] + ['r_mean', 'copy_mean']
print(r[cols].rename(columns={f'r_{t}': short[t] for t in T}).round(3).to_string(index=False))
print('\nnoise copying by trait (trained lines):')
print(r[['model', 'epoch'] + [f'copy_{t}' for t in T]].rename(columns={f'copy_{t}': short[t] for t in T}).round(2).to_string(index=False))
