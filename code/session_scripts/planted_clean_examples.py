# Cleaner planted-gene ("gene dial") examples for the poster. Inference only.
# Same model and procedure as the original example (planted_gene_examples.py, epoch 350):
# one line, one noise seed, only the 4,000-marker planted block changed (codes 1..8).
#   1. Screen many lines x seeds at codes 1 and 8 (wider than the original 12 lines x 4 seeds).
#   2. Score each image's stele segmentation for cleanness: one stele region, solidity
#      (area / convex-hull area) and circularity of its outline.
#   3. Candidates: both images clean, root size about unchanged, ranked by the stele change.
#   4. Render codes 1..8 for the top candidates and a contact sheet to choose from.
# Segmentation is run on the image upscaled to 512 px (exactly what segment() feeds the
# 512 px segmenter), so the masks and outlines are at 512 px; diameters are divided by 2.
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
from paths import SEGMENTATION_MODEL, SYNTHETIC_STELE_DIR, SYNTHETIC_STELE_MODEL_DIR, SYNTHETIC_STELE_RESULTS_DIR

from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import generate_batch
from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.utils.dataset_inputs import load_decoder
from ultralytics import YOLO

K = 4000
CKPT = SYNTHETIC_STELE_MODEL_DIR / 'diffusion_onehot_synthetic_stele_epoch_350.pt'
OUT = SYNTHETIC_STELE_RESULTS_DIR / 'planted_gene_clean_examples_epoch_350'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(8))
N_PER_SPLIT = 16
TOP = 6
device = 'cuda'

names, snp_names, matrix = load_snp_data_from_parquet(SYNTHETIC_STELE_DIR / f'MEMA_gene_matrix_synthetic_stele_k{K}.parquet')
names, matrix = list(names), np.asarray(matrix)
assert all(s.startswith('SYN_STELE_') for s in list(snp_names)[:K])
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
held = set(ckpt['val_genotypes']); founders = list(ckpt['founders'])
del ckpt
row = {g: i for i, g in enumerate(names)}
encoder, unet, unet_cfg = load_model(CKPT, matrix, torch.device(device))
unet.set_store_attention(False)
decoder = load_decoder('roots', torch.device(device), unet_cfg['latent_channels'])
scheduler = DiffusionScheduler()
for attr in ('betas', 'alphas', 'alpha_bars'):
    setattr(scheduler, attr, getattr(scheduler, attr).to(device))
latent_shape = (unet_cfg['latent_channels'], 32, 32)
seg = YOLO(str(SEGMENTATION_MODEL))


def generate(rows, seeds):
    imgs = []
    for b in range(0, len(rows), 8):
        snp = torch.tensor(np.stack(rows[b:b + 8]), dtype=torch.float32, device=device)
        with torch.no_grad():
            imgs.extend(generate_batch(encoder, unet, scheduler, decoder, snp, seeds[b:b + 8],
                                       torch.device(device), latent_shape, 50))
    return imgs


def with_code(g, c):
    v = matrix[row[g]].copy(); v[:K] = c
    return v


def measure(img):
    up = np.array(Image.fromarray(img).resize((512, 512), Image.BICUBIC))
    traits, masks = segment(seg, up, 0.25, device, 4, 2)
    st = masks.get(1, [])
    out = {'stele_px': traits['stele_diameter_px'] / 2, 'root_px': traits['root_diameter_px'] / 2,
           'n_stele_masks': len(st), 'solidity': np.nan, 'circularity': np.nan, 'n_parts': 0, 'contour': None}
    if st:
        m = max(st, key=lambda x: x.sum()).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        out.update(solidity=a / max(cv2.contourArea(cv2.convexHull(c)), 1), circularity=4 * np.pi * a / max(p * p, 1),
                   n_parts=len([x for x in cs if cv2.contourArea(x) > 30]), contour=c)
    return out, up


# 1-2. Screen
codes = pd.read_csv(SYNTHETIC_STELE_DIR / f'synthetic_codes_k{K}.csv')
codes = codes[codes['measured'] & codes['genotype'].isin(row)].sort_values('stele_diameter_mean')
picks = []
for split, df in (('trained', codes[~codes.genotype.isin(held)]), ('held out', codes[codes.genotype.isin(held)])):
    idx = np.unique(np.linspace(0, len(df) - 1, N_PER_SPLIT).round().astype(int))
    picks += [(g, split) for g in df.iloc[idx]['genotype']]
jobs = [(g, split, s, c) for g, split in picks for s in SEEDS for c in (founders[0], founders[-1])]
print(f'screening {len(picks)} lines x {len(SEEDS)} seeds x 2 codes = {len(jobs)} images', flush=True)
imgs = generate([with_code(j[0], j[3]) for j in jobs], [j[2] for j in jobs])
rec = []
for j, img in zip(jobs, imgs):
    m, _ = measure(img)
    rec.append({'genotype': j[0], 'split': j[1], 'seed': j[2], 'code': j[3],
                **{k: v for k, v in m.items() if k != 'contour'}})
screen = pd.DataFrame(rec)
screen.to_csv(OUT / 'screen_codes_1_and_8.csv', index=False)
w = screen.pivot_table(index=['genotype', 'split', 'seed'], columns='code',
                       values=['stele_px', 'root_px', 'solidity', 'circularity', 'n_stele_masks', 'n_parts']).dropna()
lo, hi = founders[0], founders[-1]
w['d_stele'] = w[('stele_px', hi)] - w[('stele_px', lo)]
w['d_root'] = w[('root_px', hi)] - w[('root_px', lo)]
w['min_solidity'] = w[[('solidity', lo), ('solidity', hi)]].min(axis=1)
w['min_circ'] = w[[('circularity', lo), ('circularity', hi)]].min(axis=1)
w['one_region'] = (w[('n_parts', lo)] == 1) & (w[('n_parts', hi)] == 1)
stats = {'n_pairs': int(len(w)), 'median_stele_change_px': float(w.d_stele.median()),
         'median_stele_change_pct': float((w.d_stele / w[('stele_px', lo)]).median() * 100),
         'share_positive': float((w.d_stele > 0).mean()), 'median_root_change_px': float(w.d_root.median()),
         'median_min_solidity': float(w.min_solidity.median())}
print(json.dumps(stats, indent=1), flush=True)

# 3. Candidates
cand = w[w.one_region & (w.min_solidity >= 0.95) & (w.min_circ >= 0.80) & (w.d_root.abs() <= 5)].copy()
cand['pct'] = cand.d_stele / cand[('stele_px', lo)] * 100
cand = cand.sort_values('d_stele', ascending=False)
print(f'{len(cand)} clean candidates', flush=True)
print(cand[['d_stele', 'pct', 'd_root', 'min_solidity', 'min_circ']].head(12).round(3).to_string(), flush=True)
cand = cand[~cand.index.get_level_values(0).duplicated()]          # one seed per line

# 4. Full dials for the top candidates
AMBER = (242, 166, 90)
chosen = []
for (g, split, seed) in cand.index[:TOP]:
    dial = generate([with_code(g, c) for c in founders], [seed] * len(founders))
    ms = [measure(im) for im in dial]
    d = OUT / f'{g}_seed{seed}'
    d.mkdir(exist_ok=True)
    for c, im, (m, up) in zip(founders, dial, ms):
        Image.fromarray(im).save(d / f'code{c}_raw.png')
        big = up.copy()
        if m['contour'] is not None:
            cv2.drawContours(big, [m['contour']], -1, AMBER, 4, lineType=cv2.LINE_AA)
        Image.fromarray(big).save(d / f'code{c}_outlined.png')
    info = {'genotype': g, 'split': split, 'seed': int(seed),
            'stele_px': {int(c): round(m['stele_px'], 1) for c, (m, _) in zip(founders, ms)},
            'root_px': {int(c): round(m['root_px'], 1) for c, (m, _) in zip(founders, ms)},
            'solidity': {int(c): round(m['solidity'], 3) for c, (m, _) in zip(founders, ms)},
            'n_parts': {int(c): int(m['n_parts']) for c, (m, _) in zip(founders, ms)}}
    info['change_pct_1_to_8'] = round((info['stele_px'][hi] / info['stele_px'][lo] - 1) * 100, 1)
    chosen.append(info)
    print(json.dumps(info), flush=True)

# Contact sheet: one row per candidate, codes 1, 3, 6, 8
SHOW = [1, 3, 6, 8]
tiles = []
for info in chosen:
    d = OUT / f'{info["genotype"]}_seed{info["seed"]}'
    tiles.append(np.concatenate([np.array(Image.open(d / f'code{c}_outlined.png').resize((256, 256))) for c in SHOW], 1))
Image.fromarray(np.concatenate(tiles, 0)).save(OUT / 'candidates_sheet.png')
json.dump({'screen': stats, 'candidates': chosen}, open(OUT / 'summary.json', 'w'), indent=2)
print('wrote', OUT)
