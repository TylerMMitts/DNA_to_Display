# Fits the full matched PCA on the RIL fitting lines exactly as train_widiv.py would
# (trained lines = split_random train minus validation), no training. Reports the
# number of components, shrink factors and validation / fitting score spread.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import DosagePCAProjector, load_widiv_genotypes

meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
train = sorted(g for g in split.index[split == 'train'] if g not in set(val) and g in row_of)
proj = DosagePCAProjector(n_components=None, target_variance=1.0, match_heldout=True)
proj.fit(dosage[[row_of[g] for g in train]], snps, line_names=train)
fit = proj.project_lines(dosage[[row_of[g] for g in train]], train)
raw = proj.transform(dosage[[row_of[g] for g in train]])
va = proj.project_lines(dosage[[row_of[g] for g in val]], val)
k = fit.shape[1]
print(f'{len(train)} fitting lines -> {k} components')
s = np.asarray(proj.fit_scale_).ravel()
for a, b in ((0, 10), (10, 50), (50, 150), (150, 500), (500, k)):
    if a >= k:
        break
    b = min(b, k)
    print(f'  PCs {a + 1}-{b}: shrink {s[a:b].mean():.2f}, val/fit variance raw '
          f'{(va[:, a:b].var(0) / raw[:, a:b].var(0)).mean():.2f}, after matching '
          f'{(va[:, a:b].var(0) / fit[:, a:b].var(0)).mean():.2f}')
print(f'score-vector length val/fit: raw {np.linalg.norm(va, axis=1).mean() / np.linalg.norm(raw, axis=1).mean():.2f}, '
      f'matched {np.linalg.norm(va, axis=1).mean() / np.linalg.norm(fit, axis=1).mean():.2f}')
print(f'loadings size {proj.components_.nbytes / 1e6:.0f} MB')
