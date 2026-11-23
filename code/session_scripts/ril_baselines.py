# 1. How well the colour measurement reads real RIL images (vs the delivered
#    phenotypes), on 300 non-test lines; cortex width is checked as the sender asked.
# 2. SNP-only genomic prediction (GBLUP / kernel ridge) fitted on trained lines'
#    phenotypes, scored on the 198 validation lines: the yardstick for the models.
# 3. Each validation line's real image measured the same way as generated ones.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, load_tables, measure
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SCALED_DIR
from widiv_snp_encoding import load_widiv_genotypes

meta, split = load_tables()
ck = torch.load(r'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_vaev3_epoch_400.pt', map_location='cpu',
                weights_only=False)
val = sorted(g for g in ck['val_genotypes'] if g in meta.index)
ck2 = torch.load(r'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_400.pt',
                 map_location='cpu', weights_only=False)
assert sorted(ck2['val_genotypes']) == sorted(ck['val_genotypes']), 'the two runs have different validation lines'
trained = sorted(g for g in meta.index if g not in set(val))
del ck, ck2
print(f'{len(trained)} trained lines, {len(val)} validation lines (test lines not loaded)')

# 1 + 3. Real images through the colour measurement.
rng = np.random.default_rng(0)
sample = sorted(set(rng.choice(trained, 300, replace=False)) | set(val))
rows = []
for g in sample:
    rgb = np.asarray(Image.open(WIDIV_SCALED_DIR / meta.loc[g, 'scaled_filename']).convert('RGB'))
    rows.append({'genotype': g, 'role': 'validation' if g in set(val) else 'trained', **measure(rgb)})
real = pd.DataFrame(rows).set_index('genotype')
real.to_csv(OUT / 'real_measured.csv')
check = []
for t in TRAITS:
    ok = real[t].notna()
    m, ref = real.loc[ok, t], meta.loc[real.index[ok], t]
    check.append({'trait': t, 'r': float(np.corrcoef(m, ref)[0, 1]), 'mean_diff': float((m - ref).mean()),
                  'median_abs_diff': float((m - ref).abs().median()), 'n': int(ok.sum())})
check = pd.DataFrame(check)
check.to_csv(OUT / 'measurement_check.csv', index=False)
print('\ncolour measurement on real 512 px images vs delivered phenotypes:')
print(check.round(3).to_string(index=False))

# 2. GBLUP-style kernel ridge on standardised dosage, lambda by 5-fold CV on trained lines.
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X_tr = dosage[[row_of[g] for g in trained]].astype(np.float32)
X_va = dosage[[row_of[g] for g in val]].astype(np.float32)
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
Z_tr, Z_va = (X_tr - mu) / sd, (X_va - mu) / sd
K = Z_tr @ Z_tr.T / Z_tr.shape[1]
K_va = Z_va @ Z_tr.T / Z_tr.shape[1]
folds = np.arange(len(trained)) % 5
rng.shuffle(folds)
LAMBDAS = [0.03, 0.1, 0.3, 1, 3, 10]
gb = []
for t in TRAITS:
    y = meta.loc[trained, t].to_numpy(float)
    cv = {}
    for lam in LAMBDAS:
        pred = np.empty_like(y)
        for f in range(5):
            a, b = folds != f, folds == f
            alpha = np.linalg.solve(K[np.ix_(a, a)] + lam * np.eye(a.sum()), y[a] - y[a].mean())
            pred[b] = y[a].mean() + K[np.ix_(b, a)] @ alpha
        cv[lam] = np.corrcoef(pred, y)[0, 1]
    lam = max(cv, key=cv.get)
    alpha = np.linalg.solve(K + lam * np.eye(len(y)), y - y.mean())
    p = y.mean() + K_va @ alpha
    obs = meta.loc[val, t].to_numpy(float)
    gb.append({'trait': t, 'lambda': lam, 'r_cv_trained': cv[lam], 'r_val': float(np.corrcoef(p, obs)[0, 1]),
               'sd_ratio_val': float(p.std() / obs.std())})
gb = pd.DataFrame(gb)
gb.to_csv(OUT / 'gblup_baseline.csv', index=False)
print('\nGBLUP on trained lines -> validation lines (r vs observed phenotype):')
print(gb.round(3).to_string(index=False))
pd.Series(val).to_csv(OUT / 'validation_lines.csv', index=False, header=['genotype'])
