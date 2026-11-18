# Turn the downloaded IMPC X-ray records (fetch_impc_xray.py) into the dataset layout
# the WiDiv pipeline reads (code/widiv/widiv_paths.py, WIDIV_DATASET=impc_bcm).
#
#   WIDIV_DATASET=impc_bcm python code/impc/organize_impc.py
#
# A "line" is one mouse (M00001, ...). Its genotype row has one column per
# knocked-out gene - 0 wild type, 1 heterozygous, 2 homozygous or hemizygous - a
# SEX_male column (0 female, 2 male) and one BATCH_<YYYY-MM> column per imaging
# month (see below), so the model is conditioned on sex and imaging batch as well
# as genotype. Every mouse carries at most one knockout, so the gene part of a row
# is one-hot; wild-type controls have none.
#
# Split: a mouse-level random split, stratified by group (gene x zygosity x sex, and
# controls by sex): 15% of each group's mice are test, never read by training. A
# group with fewer than 4 mice stays wholly in train, so every genotype the test set
# holds was also seen in training - the model cannot generate a gene it never saw.
#
# Writes into dataset/impc_bcm/:
#   genetic/genotypes_012.csv.gz      line x [genes..., SEX_male, BATCH_<month>...]
#   genetic/snp_map.csv               column, chromosome, position (gene start; sex on X)
#   genetic/encoder_columns.csv       every column, for snp_encoding='qtl_only'
#   metadata/train_test_split.csv     line, split_random
#   metadata/mouse_table.csv          line, specimen, gene, zygosity, sex, dates, and
#                                     the IMPC X-ray measurements (scoring only)
#   metadata/genotype_list.csv
import os
os.environ.setdefault('WIDIV_DATASET', 'impc_bcm')
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'widiv'))
from widiv_paths import (WIDIV_DIR, WIDIV_GENOTYPES, WIDIV_GENOTYPE_LIST, WIDIV_METADATA_DIR,  # noqa: E402
                         WIDIV_PLOT_PHENOTYPES, WIDIV_QTL, WIDIV_SNP_MAP, WIDIV_SPLIT)

TEST_FRACTION, MIN_GROUP, SEED = 0.15, 4, 0
DOSE = {'wildtype': 0, 'heterozygote': 1, 'homozygote': 2, 'hemizygote': 2}

img = pd.read_csv(WIDIV_METADATA_DIR / 'impc_xray_images.csv')
img = img[img.zygosity.isin(DOSE)].copy()
# One row per mouse. The BCM index lists every image twice under one observation_id
# (identical pixels), so a mouse has one image; one with two observations keeps its
# first here and prepare_impc_images.py uses each observation once.
mice = img.sort_values('observation_id').groupby('specimen_id', as_index=False).first()
mice = mice.sort_values('specimen_id').reset_index(drop=True)
mice['line'] = [f'M{i + 1:05d}' for i in range(len(mice))]
mice['dose'] = mice.zygosity.map(DOSE)
genes = sorted(set(mice.loc[mice.biological_sample_group == 'experimental', 'gene_symbol']))
print(f'{len(mice):,} mice ({(mice.biological_sample_group == "control").sum():,} wild-type controls), '
      f'{len(genes)} knocked-out genes')

# Genotype matrix: knockout columns, SEX_male, and one BATCH_<YYYY-MM> column per
# imaging month (2 for the mouse's month). Imaging date is a strong batch effect -
# among wild-type controls alone, the month explains 75% of trunk-length and 81% of
# X-ray-darkness variance (results/impc_bcm/pipeline_check/) - and a knockout line
# is imaged within a few weeks, so without these columns the model would learn each
# line's imaging date as its gene effect. With them, a gene's effect is its mice
# against the controls of the same months, as in the IMPC's own analysis; generate
# with a chosen month to compare genes on equal terms.
mice['month'] = pd.to_datetime(mice.date_of_experiment).dt.strftime('%Y-%m')
months = sorted(mice.month.unique())
G = np.zeros((len(mice), len(genes) + 1 + len(months)), np.int8)
col = {g: i for i, g in enumerate(genes)}
exp = mice.biological_sample_group == 'experimental'
G[np.flatnonzero(exp), mice.loc[exp, 'gene_symbol'].map(col).to_numpy()] = mice.loc[exp, 'dose'].to_numpy()
G[:, len(genes)] = np.where(mice.sex == 'male', 2, 0)
G[np.arange(len(mice)), len(genes) + 1 + mice.month.map({m: i for i, m in enumerate(months)}).to_numpy()] = 2
columns = genes + ['SEX_male'] + [f'BATCH_{m}' for m in months]
WIDIV_GENOTYPES.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(G, columns=columns).assign(line=mice.line).set_index('line').to_csv(WIDIV_GENOTYPES, compression='gzip')

# Column map: gene start where the IMPC gene index has it, else chromosome 0
pos = pd.read_csv(WIDIV_METADATA_DIR / 'impc_gene_positions.csv') if (WIDIV_METADATA_DIR / 'impc_gene_positions.csv').stat().st_size > 5 else pd.DataFrame()
pos = pos.drop_duplicates('marker_symbol').set_index('marker_symbol') if len(pos) else pos
rows = []
for g in genes:
    c = str(pos.loc[g, 'chr_name']) if g in getattr(pos, 'index', []) and pd.notna(pos.loc[g, 'chr_name']) else '0'
    p = int(pos.loc[g, 'seq_region_start']) if g in getattr(pos, 'index', []) and pd.notna(pos.loc[g, 'seq_region_start']) else 0
    rows.append((g, c, p))
rows.append(('SEX_male', 'X', 0))
rows += [(f'BATCH_{m}', 'batch', i) for i, m in enumerate(months)]
pd.DataFrame(rows, columns=['snp', 'chr', 'pos']).to_csv(WIDIV_SNP_MAP, index=False)
pd.DataFrame({'snp': columns}).to_csv(WIDIV_QTL, index=False)

# Split
rng = np.random.default_rng(SEED)
mice['group'] = np.where(exp, mice.gene_symbol + '|' + mice.zygosity, 'wildtype') + '|' + mice.sex
split = pd.Series('train', index=mice.line)
for _, g in mice.groupby('group'):
    if len(g) >= MIN_GROUP:
        k = max(1, int(round(TEST_FRACTION * len(g))))
        split[rng.choice(g.line.to_numpy(), k, replace=False)] = 'test'
split.rename('split_random').rename_axis('line').reset_index().to_csv(WIDIV_SPLIT, index=False)
pd.DataFrame({'genotype_id': mice.line}).to_csv(WIDIV_GENOTYPE_LIST, index=False)

# Per-mouse table with the X-ray measurements (numeric ones as columns; categorical
# calls as 'normal' / 'abnormal' text), for scoring only.
meas = pd.read_csv(WIDIV_METADATA_DIR / 'impc_xray_measurements.csv', low_memory=False)
num = meas[meas.observation_type == 'unidimensional'].pivot_table(
    index='specimen_id', columns='parameter_name', values='data_point', aggfunc='mean')
cat = meas[meas.observation_type == 'categorical'].pivot_table(
    index='specimen_id', columns='parameter_name', values='category', aggfunc='first')
cat.columns = [f'call: {c}' for c in cat.columns]
table = (mice[['line', 'specimen_id', 'external_sample_id', 'gene_symbol', 'zygosity', 'dose', 'sex',
               'biological_sample_group', 'colony_id', 'litter_id', 'date_of_birth', 'date_of_experiment',
               'age_in_days', 'metadata_group']]
         .merge(num, left_on='specimen_id', right_index=True, how='left')
         .merge(cat, left_on='specimen_id', right_index=True, how='left'))
table['split_random'] = table.line.map(split)
table.to_csv(WIDIV_PLOT_PHENOTYPES, index=False)

print(f'genotype matrix {G.shape[0]:,} x {G.shape[1]:,} -> {WIDIV_GENOTYPES}')
print(f'split: {int((split == "train").sum()):,} train, {int((split == "test").sum()):,} test mice')
print(f'measurements: {num.shape[1]} numeric ({", ".join(num.columns[:6])}...), {cat.shape[1]} categorical; '
      f'{table[num.columns].notna().any(axis=1).sum():,} mice measured')
print(f'mouse table -> {WIDIV_PLOT_PHENOTYPES}')
