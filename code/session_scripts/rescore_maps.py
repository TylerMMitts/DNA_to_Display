# Rescore gene-map validation: detection = overall change vs controls (z_root);
# location = where change is concentrated: each compartment's share of the map's change,
# compared with the controls' shares (z of the share). Map points to the most enriched compartment.
import numpy as np, pandas as pd
P = 'E:/D2D/DNA_to_Display/results/ril_v5/gene_maps/gene_map_validation.csv'
R = pd.read_csv(P)
COMP = ['vessels', 'stele', 'aerenchyma', 'outer', 'cortex']
for c in COMP:
    R[f'share_{c}'] = R[f'mean_{c}'] / R.mean_root
ctrl = R[R.key.str.startswith('control')]
for c in COMP:
    R[f'enrich_{c}'] = (R[f'share_{c}'] - ctrl[f'share_{c}'].mean()) / ctrl[f'share_{c}'].std(ddof=1)
R['points_to'] = R[[f'enrich_{c}' for c in COMP]].idxmax(axis=1).str.replace('enrich_', '')
R['enrich_top'] = R[[f'enrich_{c}' for c in COMP]].max(axis=1)
R['detected'] = R.z_root > 3
R['correct'] = [p in e.split(';') if isinstance(e, str) and e else np.nan for p, e in zip(R.points_to, R.expected)]
R.to_csv(P.replace('.csv', '_rescored.csv'), index=False)
pd.set_option('display.width', 250)
cols = ['key', 'trait', 'pct_var', 'expected', 'z_root', 'detected', 'points_to', 'correct'] + [f'enrich_{c}' for c in COMP]
print(R.sort_values('pct_var', ascending=False)[cols].round(2).to_string(index=False))
d = R[~R.key.str.startswith('control')]
print(f"\ndetected (overall change z > 3): {int(d.detected.sum())}/{len(d)}")
print(f"right compartment, detected maps: {int(d[d.detected].correct.sum())}/{int(d.detected.sum())}; all designed QTL: {int(d.correct.sum())}/{len(d)}")
c = R[R.key.str.startswith('control')]
print('controls: max z_root %.2f; top enrichment range %.2f..%.2f' % (c.z_root.max(), c.enrich_top.min(), c.enrich_top.max()))
