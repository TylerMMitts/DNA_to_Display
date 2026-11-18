# Does the IMPC BCM dataset work for this pipeline? Run after prepare_impc_images.py.
# Nothing is trained (training runs on Hellbender); this checks, on the laptop:
#   A. plumbing - every file the trainers read exists and has the shape they expect:
#      genotypes, column map, split, validation lines, scaled images (512 px RGB, white
#      background), the QTL-only projector over all 719 columns, one forward pass of
#      the genotype encoder (8 tokens of 256), the X-ray foreground mask, and that
#      augmentation never turns a mouse sideways;
#   B. coverage - every test mouse's genotype condition also occurs in training;
#   C. signal - whether genotype changes the images at all. Three image features per
#      mouse (silhouette area, trunk length, mean darkness inside the mouse = how much
#      tissue the X-ray passes through), and the share of their variance that the
#      genotype condition explains (eta squared), against a permutation null
#      (condition labels shuffled within sex, 200 times). Also the knockouts the IMPC
#      itself calls for body composition (DEXA) or the skeleton (X-ray): are their
#      mice further from the controls than other knockouts' mice?
# Writes results/impc_bcm/pipeline_check/check.json and features.csv.
import os
os.environ['WIDIV_DATASET'] = 'impc_bcm'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code' / 'widiv'))
from widiv_paths import (WIDIV_DIR, WIDIV_GENOTYPES, WIDIV_PLOT_PHENOTYPES, WIDIV_QTL,  # noqa: E402
                         WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA, WIDIV_SNP_MAP, WIDIV_SPLIT, WIDIV_UPRIGHT)
from widiv_data import choose_val_lines, load_split, make_transforms  # noqa: E402
from widiv_snp_encoding import DosagePCAEncoder, QTLOnlyProjector, load_widiv_genotypes, qtl_columns  # noqa: E402

OUT = ROOT / 'results' / 'impc_bcm' / 'pipeline_check'
OUT.mkdir(parents=True, exist_ok=True)
report = {}

# ---- A. plumbing -----------------------------------------------------------------
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, ROOT / 'results' / 'cache' / 'impc_bcm_genotypes.npz')
split = load_split(WIDIV_SPLIT)
val = choose_val_lines(split, 'split_random', 0.10, 42)
train = set(split.index[split.split_random == 'train']) - val
test = set(split.index[split.split_random == 'test'])
assert not (val & test) and not (train & test)
_, cols = qtl_columns(snps, WIDIV_QTL, WIDIV_SNP_MAP)
row = {g: i for i, g in enumerate(lines)}
proj = QTLOnlyProjector().fit(dosage[[row[g] for g in sorted(train)]], snps, cols, verbose=False)
enc = DosagePCAEncoder(input_dim=proj.output_dim, embedding_dim=256, num_tokens=8, hidden_dim=512).eval()
with torch.no_grad():
    tok = enc(torch.tensor(proj.transform(dosage[:16])))
meta = pd.read_csv(WIDIV_SCALED_METADATA)
present = [(WIDIV_SCALED_DIR / f).exists() for f in meta.scaled_filename]
probe = Image.open(WIDIV_SCALED_DIR / meta.scaled_filename.iloc[0])
corners = np.asarray(probe)[[0, 0, -1, -1], [0, -1, 0, -1]]
x = torch.rand(3, 8, 8)
t = make_transforms(True).transforms[-1]
outs = {tuple(t(x).flatten()[:4].tolist()) for _ in range(50)}
allowed = {tuple(x.flatten()[:4].tolist()), tuple(torch.flip(x, dims=(-1,)).flatten()[:4].tolist())}
report['plumbing'] = {
    'mice_x_columns': list(dosage.shape), 'encoder_columns': int(len(cols)),
    'train_val_test_mice': [len(train), len(val), len(test)],
    'scaled_images_listed': int(len(meta)), 'scaled_images_present': int(np.sum(present)),
    'image_size_mode': [list(probe.size), probe.mode], 'corner_pixels_white': bool((corners == 255).all()),
    'encoder_tokens_shape': list(tok.shape), 'upright_mirror_only': bool(WIDIV_UPRIGHT and outs <= allowed),
    'mouse_touches_edge_share': float(meta.mouse_touches_edge.mean()),
    'rotation_abs_median_deg': float(meta.rotation_deg.abs().median()),
    'failed_images': int((meta.error.fillna('') != '').sum()),
}

# ---- B. coverage -------------------------------------------------------------------
mt = pd.read_csv(WIDIV_PLOT_PHENOTYPES)
mt['condition'] = np.where(mt.biological_sample_group == 'control', 'wildtype',
                           mt.gene_symbol + '|' + mt.zygosity) + '|' + mt.sex
cond_of = dict(zip(mt.line, mt.condition))
train_conds = {cond_of[g] for g in train}
report['coverage'] = {'test_mice_with_condition_in_train': float(np.mean([cond_of[g] in train_conds for g in test])),
                      'conditions_in_train': len(train_conds)}

# ---- C. signal ----------------------------------------------------------------------
feat = []
for f, g in zip(meta.scaled_filename, meta.genotype):
    a = np.asarray(Image.open(WIDIV_SCALED_DIR / f).convert('L'), dtype=np.float32) / 255
    m = a < 0.96
    feat.append({'line': g, 'area_px': float(m.sum()), 'darkness': float(1 - a[m].mean()) if m.any() else np.nan})
F = pd.DataFrame(feat).merge(meta[['genotype', 'trunk_length_ref_px']].rename(
    columns={'genotype': 'line', 'trunk_length_ref_px': 'trunk_length_px'}), on='line')
F = F.merge(mt[['line', 'condition', 'sex', 'gene_symbol', 'biological_sample_group', 'split_random',
                'date_of_experiment']], on='line')
F['month'] = pd.to_datetime(F.date_of_experiment).dt.strftime('%Y-%m')
F.to_csv(OUT / 'features.csv', index=False)
F = F[F.split_random == 'train'].dropna()
# Batch removed the way the IMPC does it: each mouse against the wild-type controls of
# the same imaging month and sex. Mice whose month x sex has fewer than 3 controls are
# dropped from the test (their batch cannot be separated from their gene).
FEATURES = ('area_px', 'trunk_length_px', 'darkness')
ctrl_ref = F[F.biological_sample_group == 'control'].groupby(['month', 'sex'])[list(FEATURES)].agg(['mean', 'count'])
for k in FEATURES:
    key = list(zip(F.month, F.sex))
    mean = pd.Series(ctrl_ref[(k, 'mean')].reindex(key).to_numpy(), index=F.index)
    n = pd.Series(ctrl_ref[(k, 'count')].reindex(key).to_numpy(), index=F.index)
    F[f'{k}_vs_batch'] = np.where(n >= 3, F[k] - mean, np.nan)
F = F.dropna(subset=[f'{k}_vs_batch' for k in FEATURES])
counts = F.condition.value_counts()
F = F[F.condition.map(counts) >= 3]


def eta2(y, groups):
    gm = y.groupby(groups).transform('mean')
    return float(((gm - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


rng = np.random.default_rng(0)
signal = {}
for k in FEATURES:
    # Before and after removing batch; the null shuffles conditions within month x sex,
    # so it keeps every batch effect and only breaks the gene labels.
    for label, y in (('raw (sex removed)', F[k] - F.groupby('sex')[k].transform('mean')),
                     ('vs same-month controls', F[f'{k}_vs_batch'])):
        obs = eta2(y, F.condition)
        null = []
        for _ in range(200):
            perm = F.groupby(['month', 'sex']).condition.transform(
                lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
            null.append(eta2(y, perm))
        null = np.array(null)
        signal[f'{k} | {label}'] = {'eta2_genotype': obs, 'eta2_null_mean': float(null.mean()),
                                    'excess_over_null': obs - float(null.mean()),
                                    'p_perm': float((null >= obs).mean() + 1 / 201)}
report['signal_train_mice'] = signal
report['signal_mice_used'] = int(len(F))

calls = pd.read_csv(WIDIV_DIR / 'answer_key' / 'impc_significant_calls.csv')
dexa = set(calls.loc[calls.procedure_name.str.contains('DEXA', na=False), 'marker_symbol'])
xray = set(calls.loc[calls.procedure_stable_id.str.contains('XRY', na=False), 'marker_symbol'])
ctrl = F[F.biological_sample_group == 'control']
z = {}
for k in ('area_px', 'darkness'):
    sd = ctrl.groupby('sex')[f'{k}_vs_batch'].std()
    F[f'z_{k}'] = F[f'{k}_vs_batch'] / F.sex.map(sd)
ko = F[F.biological_sample_group == 'experimental']
gene_dist = ko.groupby('gene_symbol')[['z_area_px', 'z_darkness']].mean().abs()
for name, s in (('DEXA-called genes', dexa), ('X-ray-called genes', xray)):
    inside = gene_dist[gene_dist.index.isin(s)]
    outside = gene_dist[~gene_dist.index.isin(s)]
    z[name] = {'n_genes': int(len(inside)),
               'mean_abs_z_area_called': float(inside.z_area_px.mean()), 'mean_abs_z_area_other': float(outside.z_area_px.mean()),
               'mean_abs_z_darkness_called': float(inside.z_darkness.mean()), 'mean_abs_z_darkness_other': float(outside.z_darkness.mean())}
report['ground_truth_genes'] = z
json.dump(report, open(OUT / 'check.json', 'w'), indent=1)
print(json.dumps(report, indent=1))
