# Scatter points for the doc: per validation line and trait, the true genetic value and the
# BLUP genetic value of its 8 valid generated roots (best seeds), same model as ril_line_blup.py.
import sys; sys.path.insert(0, '.')
from ril_common import OUT
import numpy as np, pandas as pd
from widiv_paths import WIDIV_RESULTS_DIR
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
NAME = {'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma (RCA)', 'mcs_layers': 'Sclerenchyma (MCS)'}
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')
d = pd.read_csv(OUT / 'traits_windows_e1200_8valid.csv'); d = d[d.valid]
rows = []
for t in T:
    grp = d.groupby('genotype')[t]
    m, n, v = grp.mean().reindex(val), grp.size().reindex(val), grp.var().reindex(val)
    s2s = float(np.nansum(v * (n - 1)) / np.nansum(n - 1)); s2l = float(np.var(m, ddof=1) - np.mean(s2s / n))
    b = m.mean() + s2l / (s2l + s2s / n) * (m - m.mean())
    for g in val:
        if np.isfinite(b[g]):
            rows.append({'trait': NAME[t], 'line': g, 'true_genetic_value': round(float(G.loc[g, f'g_{t}']), 3),
                         'generated_mean': round(float(b[g]), 3)})
out = WIDIV_RESULTS_DIR / 'figures' / 'best_model_scatter_points_8valid.csv'
P = pd.DataFrame(rows); P.to_csv(out, index=False)
print(out, len(P)); print(P.groupby('trait').apply(lambda x: round(np.corrcoef(x.true_genetic_value, x.generated_mean)[0, 1], 3)))
