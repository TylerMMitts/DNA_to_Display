# Summarises qtl_swap_*.csv: per model, QTL, edit and trait, the paired change
# (edited - unedited image, same line and seed, both valid) against the true change
# in g. slope = sum(dgen * dtrue) / sum(dtrue^2): 1 = the model reproduces the QTL's
# full effect, 0 = it ignores it. Bootstrap over lines. Off-target traits (true change
# ~0) are summarised as the mean |change| in units of the trait's seed noise.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, VALID_TRAITS, load_tables
import numpy as np
import pandas as pd

GT = ['mx_n', 'stele_diam_um', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
meta, _ = load_tables()
lo, hi = meta[VALID_TRAITS].quantile(0.005), meta[VALID_TRAITS].quantile(0.995)
dq = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/designed_qtl_markers.csv')
target = {r.qtl: r.trait.split(';') for _, r in dq.iterrows()}
rng = np.random.default_rng(0)
rows = []
for f in sorted(p for p in OUT.glob('qtl_swap_*.csv') if p.stem != 'qtl_swap_summary'):
    model = f.stem.replace('qtl_swap_', '')
    d = pd.read_csv(f)
    d['valid'] = np.logical_and.reduce([(d[t] >= lo[t]) & (d[t] <= hi[t]) for t in VALID_TRAITS])
    base = d[d.edit == 'base'].set_index(['genotype', 'seed'])
    for (qtl, edit), e in d[d.edit != 'base'].groupby(['qtl', 'edit']):
        e = e.set_index(['genotype', 'seed'])
        b = base.loc[e.index]
        ok = (e.valid.to_numpy() & b.valid.to_numpy())
        lines_ = e.index.get_level_values(0).to_numpy()[ok]
        for t in GT:
            dgen = (e[t].to_numpy() - b[t].to_numpy())[ok]
            dtrue = (e[f'g_{t}'].to_numpy() - b[f'g_{t}'].to_numpy())[ok]
            on = t in target[qtl]
            row = {'model': model, 'qtl': qtl, 'edit': edit, 'trait': t, 'on_target': on, 'n_pairs': int(ok.sum()),
                   'mean_abs_true_change': float(np.mean(np.abs(dtrue))),
                   'mean_abs_gen_change': float(np.mean(np.abs(dgen)))}
            if np.mean(dtrue ** 2) > 1e-9:
                slope = lambda ix: np.sum(dgen[ix] * dtrue[ix]) / np.sum(dtrue[ix] ** 2)
                u = np.unique(lines_)
                bs = []
                for _ in range(500):
                    pick = rng.choice(u, len(u))
                    ix = np.concatenate([np.flatnonzero(lines_ == g) for g in pick])
                    bs.append(slope(ix))
                row.update(slope=slope(np.arange(len(dgen))), slope_lo=np.percentile(bs, 2.5),
                           slope_hi=np.percentile(bs, 97.5),
                           r_change=np.corrcoef(dgen, dtrue)[0, 1] if np.std(dtrue) > 0 else np.nan)
            rows.append(row)
R = pd.DataFrame(rows)
# Calibration: slope of the normal model's (windows e800) validation-line means on the
# true g, i.e. generated units per true unit across natural variation. slope_rel = 1:
# the QTL moves the root as much as genotype differences do in general (needed for
# rca, where g is on the liability scale and the image is in %).
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')
w = pd.read_csv(OUT / 'traits_windows_e800.csv')
w = w[w.valid & (w.role == 'validation')].groupby('genotype')[GT].mean().reindex(val)
cal = {t: np.polyfit(G.loc[val, f'g_{t}'], w[t], 1)[0] for t in GT}
print('calibration (generated units per true unit):', {t: round(c, 3) for t, c in cal.items()})
R['calibration'] = R.trait.map(cal)
for c in ('slope', 'slope_lo', 'slope_hi'):
    R[c + '_rel'] = R[c] / R.calibration
R.to_csv(OUT / 'qtl_swap_summary.csv', index=False)
pd.set_option('display.width', 250)
on = R[R.on_target].copy()
print(on[['model', 'edit', 'qtl', 'trait', 'n_pairs', 'mean_abs_true_change', 'mean_abs_gen_change', 'slope',
          'slope_lo', 'slope_hi', 'slope_rel']].round(3).sort_values(['trait', 'qtl', 'model', 'edit']).to_string(index=False))
