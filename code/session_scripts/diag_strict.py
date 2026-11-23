# Why does the strict no-QTL model break roots? Inference only.
#  1. Denoising error on real images at fixed noise levels, true vs wrong genotype,
#     same noise for every model (the average val loss could hide a high-noise problem).
#  2. Broken-root rate at epochs 300 / 400 / 500 on the same 200 lines x seeds 0-3.
#  3. The average training genotype (all-zero scores) x 32 seeds per model: if strict
#     still breaks, the problem is in the UNet, not in particular genotype inputs.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_data import make_transforms
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import (WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_PLOT_PHENOTYPES, WIDIV_RESULTS_DIR,
                         WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA)
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder
import widiv_traits as WT
from latent_diffusion.models.unet import DenoisingUNet

D = Path('E:/D2D/DNA_to_Display/models/from_hellbender')
CK = {'full e400': 'random_vaev3_epoch_400', 'literal e400': 'random_noqtl0kb_vaev3_epoch_400',
      'strict e300': 'random_noqtl1000kb_orth_vaev3_epoch_300', 'strict e400': 'random_noqtl1000kb_orth_vaev3_epoch_400',
      'strict e500': 'random_noqtl1000kb_orth_vaev3_epoch_500'}
OUT = WIDIV_RESULTS_DIR / 'noqtl_diagnosis'
(OUT / 'images').mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda')
VALID = ['root_diam_um', 'stele_diam_um', 'mx_n', 'rca_pct']
real = pd.read_csv(WIDIV_PLOT_PHENOTYPES)
real = real[real.has_image.astype(str).str.upper() == 'TRUE']
lo, hi = real[VALID].quantile(0.005), real[VALID].quantile(0.995)
lines, _, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
latent_mean = torch.load(D / f"diffusion_widiv_medium_{CK['literal e400']}.pt", map_location='cpu',
                         weights_only=False)['latent_channel_mean']   # same LiteVAE and training latents for all


def load(name):
    ck = torch.load(D / f'diffusion_widiv_medium_{CK[name]}.pt', map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck, dev)
    cfg = dict(ck['unet_config']); cfg['channel_mult'] = tuple(cfg['channel_mult'])
    unet = DenoisingUNet(**cfg); unet.load_state_dict(ck['unet_state_dict']); unet.set_store_attention(False)
    unet = unet.to(dev).eval()
    sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
    for k in ('betas', 'alphas', 'alpha_bars'):
        setattr(sched, k, getattr(sched, k).to(dev))
    scaler = LatentScaler.from_checkpoint(ck)
    sched.set_scaler(scaler)
    meta = {k: ck[k] for k in ('val_genotypes', 'test_genotypes')}
    del ck
    return enc, proj, unet, sched, scaler, meta


# ---------- 1. denoising error by noise level ----------
md = pd.read_csv(WIDIV_SCALED_METADATA)
ck0 = torch.load(D / f"diffusion_widiv_medium_{CK['literal e400']}.pt", map_location='cpu', weights_only=False)
val, test = set(ck0['val_genotypes']), set(ck0['test_genotypes'])
del ck0
md['role'] = np.where(md.genotype.isin(test), 'test', np.where(md.genotype.isin(val), 'validation', 'trained'))
md = md[md.genotype.isin(row_of)]
pick = pd.concat([md[md.role == r].sample(96, random_state=0) for r in ('trained', 'validation', 'test')])
litevae_enc, litevae_dec = load_widiv_litevae(D / 'litevae_widiv_v3_best.pt', dev)
tf = make_transforms(False)
torch.manual_seed(0)
zs = []
with torch.no_grad():
    for i in range(0, len(pick), 6):
        ims = torch.stack([tf(Image.open(WIDIV_SCALED_DIR / f).convert('RGB')) for f in pick.scaled_filename.iloc[i:i + 6]])
        z, _, _ = litevae_enc(ims.to(dev))          # latent_target 'sample', as in training
        zs.append(z.cpu())
z_raw = torch.cat(zs)
del litevae_enc
torch.cuda.empty_cache()
geno = pick.genotype.to_numpy()
wrong = np.empty_like(geno)
rng = np.random.default_rng(0)
for r in ('trained', 'validation', 'test'):        # wrong genotype from the same set
    m = (pick.role == r).to_numpy()
    wrong[m] = rng.permutation(geno[m])
TS = [20, 100, 250, 500, 750, 900, 980]
gen = torch.Generator().manual_seed(1)
noise = {t: torch.randn(z_raw.shape, generator=gen) for t in TS}
loss_rows = []
for name in CK:
    enc, proj, unet, sched, scaler, _ = load(name)
    s_true = torch.tensor(proj.transform(dosage[[row_of[g] for g in geno]]), dtype=torch.float32)
    s_wrong = torch.tensor(proj.transform(dosage[[row_of[g] for g in wrong]]), dtype=torch.float32)
    z0 = scaler.encode(z_raw)
    with torch.no_grad():
        for t in TS:
            for cond, s in (('true', s_true), ('wrong', s_wrong)):
                errs, x0errs = [], []
                for i in range(0, len(z0), 16):
                    zb, nb = z0[i:i + 16].to(dev), noise[t][i:i + 16].to(dev)
                    tb = torch.full((len(zb),), t, device=dev, dtype=torch.long)
                    zt = sched.add_noise(zb, nb, tb)
                    pred = unet(zt, tb, enc(s[i:i + 16].to(dev)))
                    errs.append(((pred - nb) ** 2).mean((1, 2, 3)).cpu())
                    ab = sched.alpha_bars[t]
                    x0 = (zt - torch.sqrt(1 - ab) * pred) / torch.sqrt(ab)
                    x0 = torch.maximum(torch.minimum(x0, sched.clamp_hi.to(dev)), sched.clamp_lo.to(dev))
                    x0errs.append(((x0 - zb) ** 2).mean((1, 2, 3)).cpu())
                e, x = torch.cat(errs).numpy(), torch.cat(x0errs).numpy()
                for r in ('trained', 'validation', 'test'):
                    m = (pick.role == r).to_numpy()
                    loss_rows.append({'model': name, 't': t, 'genotype': cond, 'role': r,
                                      'eps_mse': float(e[m].mean()), 'x0_mse': float(x[m].mean())})
    print('losses done', name, flush=True)
    del enc, unet
    torch.cuda.empty_cache()
L = pd.DataFrame(loss_rows)
L.to_csv(OUT / 'denoise_by_noise_level.csv', index=False)

# ---------- 2 + 3. generation ----------
decoder = ScaledDecoder(litevae_dec, LatentScaler.identity(4))
sample_lines = list(np.random.default_rng(0).choice(lines, 200, replace=False))
rows = []


def run(name, enc, unet, sched, scaler, snp, jobs, tag):
    decoder.scaler = scaler
    for b in range(0, len(jobs), 8):
        part = jobs[b:b + 8]
        with torch.no_grad():
            ims = generate_widiv_batch(enc, unet, sched, decoder, snp[b:b + 8].to(dev), [s for _, s in part], dev,
                                       (4, 64, 64), 50, latent_mean=latent_mean)
        for (g, s), im in zip(part, ims):
            tr = WT.measure(im)
            ok = all(lo[t] <= tr[t] <= hi[t] for t in VALID)
            fn = f"{name.replace(' ', '_')}_{tag}_{g}_seed{s}.png"
            Image.fromarray(im).resize((256, 256)).save(OUT / 'images' / fn)
            rows.append({'model': name, 'input': tag, 'genotype': g, 'seed': s, 'valid': ok, 'image': fn,
                         **{k: v for k, v in tr.items() if k != 'cort_cell_um'}})


for name in CK:
    enc, proj, unet, sched, scaler, meta = load(name)
    zero_jobs = [('mean', s) for s in range(32)]
    run(name, enc, unet, sched, scaler, torch.zeros(32, proj.output_dim), zero_jobs, 'meaninput')
    if name in ('strict e300', 'strict e500'):
        jobs = [(g, s) for g in sample_lines for s in range(4)]
        snp = torch.tensor(proj.transform(dosage[[row_of[g] for g, _ in jobs]]), dtype=torch.float32)
        run(name, enc, unet, sched, scaler, snp, jobs, 'lines')
    pd.DataFrame(rows).to_csv(OUT / 'generated_traits.csv', index=False)
    print('generation done', name, flush=True)
    del enc, unet
    torch.cuda.empty_cache()
pd.Series(sample_lines).to_csv(OUT / 'sample_lines.csv', index=False)
print('done')
