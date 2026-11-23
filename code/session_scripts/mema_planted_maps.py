# Signed tissue maps on real microscope images: MexMAGIC synthetic-stele model (epoch 350),
# whose first 4,000 loci are a planted gene coding each line's real stele size (founder code
# 1 = small ... 8 = large). Ground truth: the planted gene should map to the stele.
# For every held-out genotype x 4 seeds: the planted block set to founder 1 vs founder 8 (same
# noise), and as controls 6 blocks of real genes of the same length set to founder 1 vs 8.
# Each pair is aligned on the founder-1 root, segmented (root / stele / vessels, the project's
# segmenter), and turned into signed tissue maps (gained +, lost -, oriented 1 -> 8) for
# vessels, stele and cortex (root minus stele), plus the plain |colour change| map.
# Scoring as for the RIL: per-tissue strength, z against the controls (leave-one-out).
import sys
import json
from pathlib import Path
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from paths import SYNTHETIC_STELE_DIR, SYNTHETIC_STELE_MODEL_DIR, SYNTHETIC_STELE_RESULTS_DIR, SEGMENTATION_MODEL, pick_device
from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import generate_batch
from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.utils.dataset_inputs import load_decoder
from ultralytics import YOLO

K, SEEDS, CANON, N_CTRL = 4000, (0, 1, 2, 3), 256, 6
OUT = SYNTHETIC_STELE_RESULTS_DIR / 'planted_gene_tissue_maps'
OUT.mkdir(parents=True, exist_ok=True)
ckpt_path = SYNTHETIC_STELE_MODEL_DIR / 'diffusion_onehot_synthetic_stele_epoch_350.pt'
dev = torch.device(pick_device())
names, snp_names, matrix = load_snp_data_from_parquet(SYNTHETIC_STELE_DIR / f'MEMA_gene_matrix_synthetic_stele_k{K}.parquet')
names, matrix = list(names), np.asarray(matrix)
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
held = [g for g in sorted(ck['val_genotypes']) if g in names]
founders = list(ck['founders'])
del ck
encoder, unet, ucfg = load_model(ckpt_path, matrix, dev)
unet.set_store_attention(False)
decoder = load_decoder('roots', dev, ucfg['latent_channels'])
sched = DiffusionScheduler()
for a in ('betas', 'alphas', 'alpha_bars'):
    setattr(sched, a, getattr(sched, a).to(dev))
seg = YOLO(str(SEGMENTATION_MODEL))
latent_shape = (ucfg['latent_channels'], 32, 32)
print(f'{len(held)} held-out genotypes x {len(SEEDS)} seeds; founders {founders[0]} vs {founders[-1]}', flush=True)


def gen(rows, seeds):
    out = []
    for b in range(0, len(rows), 8):
        snp = torch.tensor(np.stack(rows[b:b + 8]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            out += list(generate_batch(encoder, unet, sched, decoder, snp, seeds[b:b + 8], dev, latent_shape, 50))
    return out


def tissues(img):
    _, m = segment(seg, img, 0.25, None, 4, 2)
    blank = np.zeros(img.shape[:2], bool)
    pick = lambda c: (max(m[c], key=lambda x: x.sum()) if m.get(c) else blank)
    root = ndimage.binary_fill_holes(pick(0) | pick(1))
    stele = ndimage.binary_fill_holes(pick(1))
    ves = np.zeros_like(blank)
    for v in m.get(2, []):
        ves |= v
    ves &= stele
    return root, np.stack([ves, stele & ~ves, root & ~stele]).astype(np.float32)   # vessels, stele, cortex


def frame(root):
    ys, xs = np.nonzero(root)
    if len(ys) == 0:
        return 128, 128, 100
    return (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2, max(ys.max() - ys.min(), xs.max() - xs.min()) / 2


def align(arr, cy, cx, R, nearest):
    half = int(round(1.3 * R))
    cy, cx = int(round(cy)) + 300, int(round(cx)) + 300
    if arr.ndim == 3:                            # tissue stack (3, H, W)
        pad = np.pad(arr, ((0, 0), (300, 300), (300, 300)), constant_values=0)
        return np.stack([np.asarray(Image.fromarray((ch * 255).astype(np.uint8)).resize((CANON, CANON), Image.NEAREST)) / 255
                         for ch in pad[:, cy - half:cy + half, cx - half:cx + half]]).astype(np.float32)
    pad = np.pad(arr, ((300, 300), (300, 300)), constant_values=0)
    c = pad[cy - half:cy + half, cx - half:cx + half]
    return np.asarray(Image.fromarray(c.astype(np.uint8)).resize((CANON, CANON), Image.NEAREST if nearest else Image.LANCZOS)).astype(np.float32)


rng = np.random.default_rng(0)
starts = sorted(rng.choice(np.arange(K, matrix.shape[1] - K), N_CTRL, replace=False))
blocks = [('planted', slice(0, K))] + [(f'control{i + 1}', slice(int(s), int(s) + K)) for i, s in enumerate(starts)]
jobs = [(g, s) for g in held for s in SEEDS]
store, tmpl = {}, np.zeros((3, CANON, CANON), np.float32)
for name, sl in blocks:
    lo_rows, hi_rows = [], []
    for g, s in jobs:
        lo, hi = matrix[names.index(g)].copy(), matrix[names.index(g)].copy()
        lo[sl], hi[sl] = founders[0], founders[-1]
        lo_rows.append(lo); hi_rows.append(hi)
    seeds = [s for _, s in jobs]
    A, B = gen(lo_rows, seeds), gen(hi_rows, seeds)
    absmap = np.zeros((CANON, CANON), np.float32)
    tis = np.zeros((3, CANON, CANON), np.float32)
    n = 0
    for a, b in zip(A, B):
        ra, ta = tissues(a)
        rb, tb = tissues(b)
        cy, cx, R = frame(ra)
        # align colour images (pad with black, LAT background)
        ca = np.stack([align(a[..., j], cy, cx, R, False) for j in range(3)], -1)
        cb = np.stack([align(b[..., j], cy, cx, R, False) for j in range(3)], -1)
        absmap += np.abs(cb - ca).mean(-1) / 255
        sa, sb = align(ta, cy, cx, R, True), align(tb, cy, cx, R, True)
        tis += sb - sa
        if name == 'planted':
            tmpl += sa
        n += 1
    store[name] = {'abs': absmap / n, 'tissue': tis / n}
    print(name, 'pairs', n, flush=True)

tmpl /= len(jobs)
root_mask = tmpl.sum(0) > 0.5
region = np.where(root_mask, tmpl.argmax(0), -1)
np.savez_compressed(OUT / 'planted_maps.npz', template=tmpl, **{f'{k}__{m}': v for k, d in store.items() for m, v in d.items()})
COMP = ['vessels', 'stele', 'cortex']
rows = []
for method in ('abs', 'tissue'):
    S = {}
    for k, d in store.items():
        if method == 'abs':
            S[k] = [float(d['abs'][region == j].mean()) for j in range(3)]
        else:
            S[k] = [float(np.abs(d['tissue'][j])[root_mask].mean()) for j in range(3)]
    ctrl = [k for k in store if k.startswith('control')]
    for k in store:
        pool = np.array([S[c] for c in ctrl if c != k])
        z = (np.array(S[k]) - pool.mean(0)) / pool.std(0, ddof=1)
        rows.append({'method': method, 'block': k, 'points_to': COMP[int(np.argmax(z))], 'top_z': float(z.max()),
                     **{f'z_{c}': float(v) for c, v in zip(COMP, z)},
                     **{f'strength_{c}': v for c, v in zip(COMP, S[k])}})
R_ = pd.DataFrame(rows)
R_.to_csv(OUT / 'planted_map_scores.csv', index=False)
# Net direction inside the stele band for the planted gene: does stele grow (gain) or shrink?
net = {c: float(store['planted']['tissue'][j][root_mask].sum()) for j, c in enumerate(COMP)}
json.dump({'held_out_genotypes': len(held), 'pairs': len(jobs), 'control_starts': [int(s) for s in starts],
           'net_tissue_change_px_planted': net}, open(OUT / 'planted_maps.json', 'w'), indent=1)
pd.set_option('display.width', 220)
print(R_.round(2).to_string(index=False))
print('net tissue change (px, founder 1 -> 8), planted gene:', {k: round(v, 1) for k, v in net.items()})
print('done')
