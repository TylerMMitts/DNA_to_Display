# Poster hero candidates from the best RIL model (windows e1200), inference only, validation
# lines only.
#   hero_mosaic_ril.png       100 validation lines, best seed, one fixed-scale crop each,
#                             ordered by generated stele diameter
#   hero_allele_swap_ril.png  one line generated with its own genotype and with the ±10 cM
#                             haplotype around the shared stele/metaxylem QTL swapped; same seed;
#                             metaxylem vessels coloured
# Writes to presentation/poster_assets/.
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT as CKOUT, VALID_TRAITS, load_tables, measure
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import widiv_traits as WT
from latent_diffusion.models.unet import DenoisingUNet
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SNP_MAP
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

OUT = Path(r'E:/D2D/DNA_to_Display\presentation\poster_assets')
OUT.mkdir(parents=True, exist_ok=True)
CK = 'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
VAE = 'E:/D2D/DNA_to_Display/models/from_hellbender/litevae_ril_v5_v3_best.pt'
SEEDS = (30, 2)
dev = torch.device('cuda')
meta, split = load_tables()
val = pd.read_csv(CKOUT / 'validation_lines.csv').genotype.tolist()
lo, hi = meta[VALID_TRAITS].quantile(0.005), meta[VALID_TRAITS].quantile(0.995)
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
smap = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[list(snps)]
_, dec_vae = load_widiv_litevae(Path(VAE), dev)
sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
sched.betas, sched.alphas, sched.alpha_bars = (sched.betas.to(dev), sched.alphas.to(dev), sched.alpha_bars.to(dev))
ck = torch.load(CK, map_location='cpu', weights_only=False)
enc, proj = load_widiv_encoder(ck, dev)
ucfg = dict(ck['unet_config']); ucfg['channel_mult'] = tuple(ucfg['channel_mult'])
unet = DenoisingUNet(**ucfg); unet.load_state_dict(ck['unet_state_dict']); unet.set_store_attention(False)
unet = unet.to(dev).eval()
sched.set_scaler(LatentScaler.from_checkpoint(ck))
decoder = ScaledDecoder(dec_vae, LatentScaler.from_checkpoint(ck))


def generate(D, names, seeds):
    out = []
    with torch.no_grad():
        for b in range(0, len(D), 8):
            cond = torch.tensor(proj.project_lines(D[b:b + 8], names[b:b + 8]), dtype=torch.float32, device=dev)
            out += list(generate_widiv_batch(enc, unet, sched, decoder, cond, seeds[b:b + 8], dev,
                                             (ucfg['latent_channels'], 64, 64), 50,
                                             latent_mean=ck['latent_channel_mean']))
    return out


def valid(m):
    return all(lo[t] <= m[t] <= hi[t] for t in VALID_TRAITS)


def root_box(im):
    # centre and half-size of the root: largest non-white component
    m = ndimage.binary_opening(im.min(-1) < 200, iterations=2)
    # close the aerenchyma channels so the whole root disc is one component
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=10))
    lab, nn = ndimage.label(m)
    if nn == 0:
        return 256, 256, 64
    big = 1 + int(np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, nn + 1))))
    ys, xs = np.nonzero(lab == big)
    return (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2, max(ys.max() - ys.min(), xs.max() - xs.min()) / 2


def crop(im, cy, cx, half):
    pad = np.pad(im, ((400, 400), (400, 400), (0, 0)), constant_values=255)
    cy, cx, half = int(round(cy)) + 400, int(round(cx)) + 400, int(round(half))
    return pad[cy - half:cy + half, cx - half:cx + half]


# 1. Mosaic.
names = val[:100]
imgs = generate(dosage[[row_of[g] for g in names]], names, [SEEDS[0]] * len(names))
ms = [measure(im) for im in imgs]
keep = [k for k in range(len(imgs)) if valid(ms[k])]
order = sorted(keep, key=lambda k: ms[k]['stele_diam_um'])[:100]
# one fixed crop size for all (fixed image scale), so root sizes stay comparable
boxes = {k: root_box(imgs[k]) for k in order}
half = max(b[2] for b in boxes.values()) + 8
n = int(np.floor(np.sqrt(len(order))))
order = order[:n * n]
tile, gap = 320, 8
mosaic = Image.new('RGB', (n * tile + (n - 1) * gap, n * tile + (n - 1) * gap), (255, 255, 255))
for i, k in enumerate(order):
    cy, cx, _ = boxes[k]
    c = Image.fromarray(crop(imgs[k], cy, cx, half)).resize((tile, tile), Image.LANCZOS)
    mosaic.paste(c, ((i % n) * (tile + gap), (i // n) * (tile + gap)))
mosaic.save(OUT / 'hero_mosaic_ril.png')
print(f'mosaic: {n}x{n} of {len(keep)} valid, crop half-size {half}px', flush=True)

# 2. Allele swap at the shared stele / metaxylem number QTL (marker 1_814), ±10 cM haplotype.
q = '1_814'
c = list(snps).index(q)
block = np.flatnonzero((smap.chr.to_numpy() == smap.loc[q, 'chr'])
                       & (np.abs(smap.pos_cM.to_numpy() - smap.loc[q, 'pos_cM']) <= 10))
cand = [g for g in val if dosage[row_of[g], c] != 1][:40]
D0 = dosage[[row_of[g] for g in cand]]
D1 = D0.copy()
for r in range(len(cand)):
    hom = block[D1[r, block] != 1]
    D1[r, hom] = 2 - D1[r, hom]
best = None
for sd in SEEDS:
    a = generate(D0, cand, [sd] * len(cand))
    b = generate(D1, cand, [sd] * len(cand))
    for g, ia, ib in zip(cand, a, b):
        ma, mb = measure(ia), measure(ib)
        if not (valid(ma) and valid(mb)):
            continue
        lowhi = (ia, ib, ma, mb) if ma['mx_n'] < mb['mx_n'] else (ib, ia, mb, ma)
        gain = lowhi[3]['mx_n'] - lowhi[2]['mx_n']
        if best is None or gain > best[0]:
            best = (gain, g, sd) + lowhi
gain, g, sd, im_lo, im_hi, m_lo, m_hi = best
print(f'allele swap: {g} seed {sd}: mx_n {m_lo["mx_n"]:.0f} -> {m_hi["mx_n"]:.0f}, stele '
      f'{m_lo["stele_diam_um"]:.0f} -> {m_hi["stele_diam_um"]:.0f} um', flush=True)


def vessels(rgb):
    cls = WT.classify(rgb)
    white = cls == WT.WHITE
    stele = ndimage.binary_opening(ndimage.binary_fill_holes(np.isin(cls, [WT.IDX[l] for l in (6, 7, 8, 9)])),
                                   iterations=2)
    inner = ndimage.binary_erosion(stele, iterations=2)
    lab, nn = ndimage.label(white & inner)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nn + 1)) if nn else np.array([])
    return np.isin(lab, 1 + np.flatnonzero(sizes >= 40))


ACC = np.array([214, 64, 69])   # warm red for the vessels
both = [im_lo, im_hi]
bx = [root_box(im) for im in both]
half = max(b[2] for b in bx) + 10
panels = []
for im, (cy, cx, _) in zip(both, bx):
    v = vessels(im)
    col = im.copy()
    col[v] = ACC
    panels.append(Image.fromarray(crop(col, cy, cx, half)).resize((1200, 1200), Image.LANCZOS))
for name, p in zip(('allele_swap_before', 'allele_swap_after'), panels):
    p.save(OUT / f'{name}.png')
W = 2 * 1200 + 160
sheet = Image.new('RGB', (W, 1200), (255, 255, 255))
sheet.paste(panels[0], (0, 0)); sheet.paste(panels[1], (1360, 0))
d = ImageDraw.Draw(sheet)
d.polygon([(1215, 560), (1330, 600), (1215, 640)], fill=(60, 60, 60))
sheet.save(OUT / 'hero_allele_swap_ril.png')
json.dump({'line': g, 'seed': int(sd), 'qtl_marker': q, 'block_cM': 10, 'mx_n': [m_lo['mx_n'], m_hi['mx_n']],
           'stele_um': [m_lo['stele_diam_um'], m_hi['stele_diam_um']]},
          open(OUT / 'allele_swap.json', 'w'), indent=1, default=float)
print('done')
