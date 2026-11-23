# Extra poster assets: real MexMAGIC photos at high resolution (to pick one for the
# anatomy figure), a DNA marker strip for one RIL line, and its encoder tokens.
import os
os.environ['WIDIV_DATASET'] = 'ril_v5'
import sys
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code')
sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pandas as pd

OUT = Path(r'E:/D2D/DNA_to_Display\presentation\poster_assets')
from model_toolkit import _metadata
from paths import CROPPED_IMAGES_DIR
meta = _metadata()
for g in ('MEMA310', 'MEMA233', 'MEMA105', 'MEMA296', 'MEMA042'):
    for f in meta.loc[meta.genotype == g, 'new_filename'][:2]:
        p = Path(CROPPED_IMAGES_DIR) / f
        if p.exists():
            im = Image.open(p).convert('RGB')
            s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))
            im.resize((1400, 1400), Image.LANCZOS).save(OUT / f'real_{p.stem}.png')
            print(p.name, im.size)

# DNA strip: RIL line G0346, 10,000 markers along the 10 chromosomes, two parental colours.
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SNP_MAP
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
lines = list(lines)
d = dosage[lines.index('G0346')]
chrom = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[list(snps)].chr.to_numpy()
A, B, H = np.array([42, 120, 214]), np.array([235, 104, 52]), np.array([200, 200, 200])
cols = np.where(d[:, None] == 0, A, np.where(d[:, None] == 2, B, H)).astype(np.uint8)
pieces = []
for c in range(1, 11):
    seg = cols[chrom == c]
    pieces.append(np.repeat(seg[None], 60, 0))
    pieces.append(np.full((60, 18, 3), 255, np.uint8))
strip = np.concatenate(pieces[:-1], 1)
Image.fromarray(strip).resize((2400, 140), Image.NEAREST).save(OUT / 'dna_strip_G0346.png')

# Encoder tokens for the same line (8 tokens x 256 values), as a heat strip.
ck = torch.load('E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt',
                map_location='cpu', weights_only=False)
enc, proj = load_widiv_encoder(ck, torch.device('cpu'))
with torch.no_grad():
    tok = enc(torch.tensor(proj.project_lines(dosage[[lines.index('G0346')]], ['G0346']), dtype=torch.float32))[0].numpy()
z = (tok - tok.mean()) / tok.std()
z = np.clip((z + 2.5) / 5, 0, 1)
import matplotlib
cm = matplotlib.colormaps['magma']
img = (cm(z)[..., :3] * 255).astype(np.uint8)            # 8 x 256
img = np.repeat(np.repeat(img, 40, 0), 4, 1)
gap = np.full((8, img.shape[1], 3), 255, np.uint8)
rows = [np.concatenate([img[i * 40:(i + 1) * 40], gap]) for i in range(8)]
Image.fromarray(np.concatenate(rows)[:-8]).save(OUT / 'tokens_G0346.png')
print('done')
