# Causal QTL test: for validation lines homozygous at a designed QTL, swap the parental
# haplotype at that QTL (dosage 0 <-> 2) and generate with the same seed; the change in
# each measured trait is the effect the model gives that QTL. Two edits:
#   marker: flip only the QTL's own marker
#   block:  flip every homozygous marker within +-BLOCK_CM of it (a realistic
#           introgression: neighbours in LD move with it)
# Truth for every edit: g(edited) - g(original) from a least-squares rebuild of g on the
# answer key's listed QTL markers (R^2 0.987-1.000). Answer key used for scoring only.
# Validation lines only (no test line is read).
# Usage: python ril_qtl_swap.py <litevae.pt> name=checkpoint.pt:edits [...]   edits = marker,block
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, load_tables, measure

import numpy as np
import pandas as pd
import torch

from latent_diffusion.models.unet import DenoisingUNet
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SNP_MAP
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

AK = 'E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/'
import os
N_LINES, SEEDS, BLOCK_CM = 48, (0, 1), float(os.environ.get('BLOCK_CM', 2.0))
GT = ['mx_n', 'stele_diam_um', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
dev = torch.device('cuda')
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
lines = list(lines)
col = {s: i for i, s in enumerate(snps)}
row_of = {g: i for i, g in enumerate(lines)}
smap = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[list(snps)]
q = pd.read_csv(AK + 'designed_qtl_markers.csv')
q = q[q.qtl != 'lac_q'].reset_index(drop=True)   # lacunae are not measured on generated roots

# Truth model: OLS of g on the listed markers (+ A x A for mx_diam), training lines only.
qt = pd.read_csv(AK + 'qtl_and_scan_baseline__qtl.csv')
bg = pd.read_csv(AK + 'qtl_and_scan_baseline__background_qtl.csv')
G = pd.read_csv(AK + 'genetic_values.csv').set_index('line')
tr = [g for g in lines if split.get(g) == 'train']
Dtr = dosage[[row_of[g] for g in tr]].astype(float)


def design(D, t, mk):
    X = np.c_[np.ones(len(D)), D[:, mk]]
    if t == 'mx_diam_um':
        X = np.c_[X, (D[:, col['5_2477']] - 1) * (D[:, col['8_2175']] - 1)]
    return X


truth = {}
for t in GT:
    mk = sorted({col[m] for m in pd.concat([qt[qt.trait == t].nearest_marker, bg[bg.trait == t].nearest_marker])})
    beta = np.linalg.lstsq(design(Dtr, t, mk), G.loc[tr, f'g_{t}'].to_numpy(), rcond=None)[0]
    truth[t] = (mk, beta)


def g_of(D, t):
    mk, beta = truth[t]
    return design(D.astype(float), t, mk) @ beta


# Edits: one shared set of N_LINES validation lines (so the unedited images are shared);
# for each QTL, the lines homozygous at it.
rng = np.random.default_rng(0)
lineset = sorted(rng.choice(val, N_LINES, replace=False))
edits = []   # (qtl, kind, line, edited dosage row)
chosen = {}
for _, r in q.iterrows():
    c = col[r.snp]
    pick = [g for g in lineset if dosage[row_of[g], c] != 1]
    chosen[r.qtl] = pick
    block = np.flatnonzero((smap.chr.to_numpy() == r.chr)
                           & (np.abs(smap.pos_cM.to_numpy() - smap.loc[r.snp, 'pos_cM']) <= BLOCK_CM))
    for g in pick:
        base = dosage[row_of[g]].copy()
        m = base.copy(); m[c] = 2 - m[c]
        b = base.copy(); hom = block[b[block] != 1]; b[hom] = 2 - b[hom]
        edits += [(r.qtl, 'marker', g, m), (r.qtl, 'block', g, b)]
base_lines = sorted({g for v in chosen.values() for g in v})
print(f'{len(q)} QTL, {len(base_lines)} distinct lines, block +-{BLOCK_CM} cM '
      f'(~{np.mean([((smap.chr == r.chr) & ((smap.pos_cM - smap.loc[r.snp, "pos_cM"]).abs() <= BLOCK_CM)).sum() for _, r in q.iterrows()]):.0f} markers)', flush=True)

enc_vae, dec_vae = load_widiv_litevae(Path(sys.argv[1]), dev)
del enc_vae
sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
sched.betas, sched.alphas, sched.alpha_bars = (sched.betas.to(dev), sched.alphas.to(dev), sched.alpha_bars.to(dev))

for spec in sys.argv[2:]:
    name, rest = spec.split('=', 1)
    path, kinds = rest.rsplit(':', 1)
    kinds = kinds.split(',')
    out_path = OUT / f'qtl_swap_{name}.csv'
    t0 = time.time()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck, dev)
    ucfg = dict(ck['unet_config']); ucfg['channel_mult'] = tuple(ucfg['channel_mult'])
    unet = DenoisingUNet(**ucfg); unet.load_state_dict(ck['unet_state_dict']); unet.set_store_attention(False)
    unet = unet.to(dev).eval()
    scaler = LatentScaler.from_checkpoint(ck)
    sched.set_scaler(scaler)
    decoder = ScaledDecoder(dec_vae, scaler)
    # project_lines with the real names: windows layouts use out-of-fold rows only for
    # trained lines, and validation lines are not trained lines, so edits project the same way.
    jobs = [('base', 'base', g, dosage[row_of[g]]) for g in base_lines] + [e for e in edits if e[1] in kinds]
    rows = []
    with torch.no_grad():
        flat = [(j, s) for j in jobs for s in SEEDS]
        for b in range(0, len(flat), 8):
            batch = flat[b:b + 8]
            D = np.stack([j[3] for j, _ in batch])
            cond = torch.tensor(proj.project_lines(D, [j[2] for j, _ in batch]), dtype=torch.float32, device=dev)
            imgs = generate_widiv_batch(enc, unet, sched, decoder, cond, [s for _, s in batch], dev,
                                        (ucfg['latent_channels'], 64, 64), 50, latent_mean=ck['latent_channel_mean'])
            for ((qtl, kind, g, d), s), im in zip(batch, imgs):
                rows.append({'qtl': qtl, 'edit': kind, 'genotype': g, 'seed': s, **measure(im),
                             **{f'g_{t}': float(g_of(d[None], t)[0]) for t in GT}})
            if b % 400 == 0:
                print(f'  {name}: {b + len(batch)}/{len(flat)} images, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'{name}: {len(rows)} images in {time.time() - t0:.0f}s -> {out_path.name}', flush=True)
    del enc, unet, ck
    torch.cuda.empty_cache()
print('done')
