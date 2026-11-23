# One line, one noise seed, the planted stele gene set to each code 1..8 -
# so the only thing that differs between images is the planted gene. Inference
# only (no training). Screens several lines and seeds at codes 1 and 8 first,
# reports how big the 1 -> 8 change is across all of them, and renders the
# clearest one with its measured stele outlined.
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

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
OUT = SYNTHETIC_STELE_RESULTS_DIR / 'planted_gene_examples_epoch_350'
OUT.mkdir(parents=True, exist_ok=True)
EA = Path(r'E:/D2D/DNA_to_Display\presentation\edit_assets')
SEEDS = [0, 1, 2, 3]
N_PER_SPLIT = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

names, snp_names, matrix = load_snp_data_from_parquet(SYNTHETIC_STELE_DIR / f'MEMA_gene_matrix_synthetic_stele_k{K}.parquet')
names, matrix = list(names), np.asarray(matrix)
assert all(s.startswith('SYN_STELE_') for s in list(snp_names)[:K])
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
assert int(ckpt['snp_projector']['n_loci']) == matrix.shape[1]
held = set(ckpt['val_genotypes']); founders = list(ckpt['founders']); epoch = ckpt.get('epoch')
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


def stele_mask(masks):
    if not masks.get(1):
        return None
    return max(masks[1], key=lambda m: m.sum())


def measure(img):
    traits, masks = segment(seg, img, 0.25, device, 4, 2)
    return traits['stele_diameter_px'], traits['root_diameter_px'], stele_mask(masks)


def with_code(g, c):
    v = matrix[row[g]].copy(); v[:K] = c
    return v


# Screen: codes 1 and 8 for evenly spaced trained and held-out lines.
codes = pd.read_csv(SYNTHETIC_STELE_DIR / f'synthetic_codes_k{K}.csv')
codes = codes[codes['measured'] & codes['genotype'].isin(row)].sort_values('stele_diameter_mean')
picks = []
for split, df in (('trained', codes[~codes.genotype.isin(held)]), ('held out', codes[codes.genotype.isin(held)])):
    idx = np.linspace(0, len(df) - 1, N_PER_SPLIT).round().astype(int)
    picks += [(g, split, c) for g, c in zip(df.iloc[idx]['genotype'], df.iloc[idx]['code'])]
jobs = [(g, split, own, s, c) for g, split, own in picks for s in SEEDS for c in (founders[0], founders[-1])]
imgs = generate([with_code(j[0], j[4]) for j in jobs], [j[3] for j in jobs])
rec = []
for j, img in zip(jobs, imgs):
    st, rt, _ = measure(img)
    rec.append({'genotype': j[0], 'split': j[1], 'own_code': j[2], 'seed': j[3], 'code': j[4],
                'stele_px': st, 'root_px': rt})
screen = pd.DataFrame(rec)
screen.to_csv(OUT / 'screen_codes_1_and_8.csv', index=False)
wide = screen.pivot_table(index=['genotype', 'split', 'seed'], columns='code', values=['stele_px', 'root_px']).dropna()
wide['d_stele'] = wide[('stele_px', founders[-1])] - wide[('stele_px', founders[0])]
wide['d_root'] = wide[('root_px', founders[-1])] - wide[('root_px', founders[0])]
d = wide['d_stele']
stats = {'checkpoint_epoch': epoch, 'n_pairs': int(len(d)), 'median_stele_change_px': float(d.median()),
         'mean_stele_change_px': float(d.mean()), 'share_positive': float((d > 0).mean()),
         'median_root_change_px': float(wide['d_root'].median())}
print(json.dumps(stats, indent=1))
print(wide.sort_values('d_stele', ascending=False).head(8).to_string())

# The example: the clearest 1 -> 8 change among pairs whose whole root
# changed little, so the slide shows the stele moving inside a root that
# stays the same size. Disclosed as chosen, with the median beside it.
cand = wide[wide['d_root'].abs() <= 4].sort_values('d_stele', ascending=False)
(g, split, seed) = cand.index[0]
print('example', g, split, seed, float(cand['d_stele'].iloc[0]))
dial = generate([with_code(g, c) for c in founders], [seed] * len(founders))
meas = [measure(im) for im in dial]

AMBER, BLUE = (242, 166, 90), (120, 170, 255)


def outlined(img, masks_colors, scale=2):
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for m, col in masks_colors:
        if m is None:
            continue
        mm = cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cs, -1, col, 3)
    return big


for c, im, (st, rt, m) in zip(founders, dial, meas):
    Image.fromarray(im).save(OUT / f'{g}_seed{seed}_code{c}_raw.png')
    Image.fromarray(outlined(im, [(m, AMBER)])).save(EA / f'planted_dial_code{c}.png')
first, last = meas[0], meas[-1]
Image.fromarray(outlined(dial[0], [(first[2], BLUE)])).save(EA / 'planted_pair_code1.png')
Image.fromarray(outlined(dial[-1], [(last[2], AMBER)])).save(EA / 'planted_pair_code8.png')
Image.fromarray(outlined(dial[-1], [(first[2], BLUE), (last[2], AMBER)])).save(EA / 'planted_pair_overlay.png')

example = {'genotype': g, 'split': split, 'seed': int(seed),
           'stele_px_by_code': {int(c): round(float(x[0]), 1) for c, x in zip(founders, meas)},
           'root_px_by_code': {int(c): round(float(x[1]), 1) for c, x in zip(founders, meas)}}
json.dump({'screen': stats, 'example': example}, open(OUT / 'summary.json', 'w'), indent=2)
json.dump({'screen': stats, 'example': example}, open(EA / 'planted_example.json', 'w'), indent=2)
print(json.dumps(example, indent=1))
