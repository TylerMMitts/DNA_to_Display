# 16 images per validation line for windows e1200: seeds 0-3 already in
# traits_windows_e1200.csv, this adds seeds 4-15 (same pipeline as ril_eval.py:
# matched start noise, 50 DDIM steps, colour measurement, same validity rule).
# Then: actual r with n = 1, 2, 4, 8, 16 images per line (averaged over disjoint
# seed subsets) against the projection from the 4-seed variance components.
# Validation lines only. Answer key for scoring only.
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, VALID_TRAITS, load_tables, measure
import numpy as np
import pandas as pd
import torch
from latent_diffusion.models.unet import DenoisingUNet
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

CK = 'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
VAE = 'E:/D2D/DNA_to_Display/models/from_hellbender/litevae_ril_v5_v3_best.pt'
NEW_SEEDS = range(4, 16)
GT = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
dev = torch.device('cuda')
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
lo, hi = meta[VALID_TRAITS].quantile(0.005), meta[VALID_TRAITS].quantile(0.995)
out_path = OUT / 'traits_windows_e1200_16seeds.csv'

if not out_path.exists():
    lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
    row_of = {g: i for i, g in enumerate(lines)}
    _, dec_vae = load_widiv_litevae(Path(VAE), dev)
    sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
    sched.betas, sched.alphas, sched.alpha_bars = (sched.betas.to(dev), sched.alphas.to(dev), sched.alpha_bars.to(dev))
    ck = torch.load(CK, map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck, dev)
    ucfg = dict(ck['unet_config']); ucfg['channel_mult'] = tuple(ucfg['channel_mult'])
    unet = DenoisingUNet(**ucfg); unet.load_state_dict(ck['unet_state_dict']); unet.set_store_attention(False)
    unet = unet.to(dev).eval()
    scaler = LatentScaler.from_checkpoint(ck)
    sched.set_scaler(scaler)
    decoder = ScaledDecoder(dec_vae, scaler)
    cond = torch.tensor(proj.project_lines(dosage[[row_of[g] for g in val]], val), dtype=torch.float32, device=dev)
    local = {g: i for i, g in enumerate(val)}
    jobs = [(g, s) for g in val for s in NEW_SEEDS]
    rows, t0 = [], time.time()
    with torch.no_grad():
        for b in range(0, len(jobs), 8):
            batch = jobs[b:b + 8]
            imgs = generate_widiv_batch(enc, unet, sched, decoder, cond[[local[g] for g, _ in batch]],
                                        [s for _, s in batch], dev, (ucfg['latent_channels'], 64, 64), 50,
                                        latent_mean=ck['latent_channel_mean'])
            rows += [{'genotype': g, 'seed': s, 'role': 'validation', **measure(im)} for (g, s), im in zip(batch, imgs)]
            if b % 400 == 0:
                print(f'{b + len(batch)}/{len(jobs)} images, {time.time() - t0:.0f}s', flush=True)
    new = pd.DataFrame(rows)
    new['valid'] = np.logical_and.reduce([(new[t] >= lo[t]) & (new[t] <= hi[t]) for t in VALID_TRAITS])
    old = pd.read_csv(OUT / 'traits_windows_e1200.csv')
    old = old[old.role == 'validation']
    pd.concat([old, new], ignore_index=True).to_csv(out_path, index=False)
    print(f'{len(new)} new images in {time.time() - t0:.0f}s', flush=True)

d = pd.read_csv(out_path)
d = d[d.valid]
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')
proj_tab = pd.read_csv(WIDIV_RESULTS_DIR / 'figures' / 'best_model_trait_r_blup.csv')
NAME = dict(zip(GT, proj_tab.trait))
rows = []
for t in GT:
    g = G.loc[val, f'g_{t}']
    p = proj_tab.set_index('trait').loc[NAME[t]]
    for n in (1, 2, 4, 8, 16):
        # disjoint seed subsets of size n; actual r averaged over them
        rs = []
        for k in range(16 // n):
            sub = d[d.seed.isin(range(k * n, (k + 1) * n))]
            m = sub.groupby('genotype')[t].mean().reindex(val)
            ok = m.notna().to_numpy()
            rs.append(np.corrcoef(m[ok], g[ok])[0, 1])
        rel = p.line_sd ** 2 / (p.line_sd ** 2 + p.seed_sd ** 2 / n)
        rows.append({'trait': NAME[t], 'n_images': n, 'r_actual': np.mean(rs), 'r_actual_sd': np.std(rs),
                     'n_subsets': len(rs), 'r_projected': p.r_noise_free * np.sqrt(rel)})
R = pd.DataFrame(rows)
R.to_csv(OUT / 'images_per_line_check.csv', index=False)
pd.set_option('display.width', 200)
print(R.pivot(index='trait', columns='n_images', values='r_actual').round(3).to_string())
print(R.pivot(index='trait', columns='n_images', values='r_projected').round(3).to_string())
print(R.groupby('n_images')[['r_actual', 'r_projected']].mean().round(3).to_string())
