# Seed screening and invalid-image replacement for windows e1200 (validation lines).
#   A. Screen seeds 16-47 on 66 screening lines (every third validation line); seeds 0-15
#      already exist for all 198. Rank all 48 seeds by share of valid images on the
#      screening lines (validity only: no phenotype or answer key).
#   B. Keep the best 8. Every validation line gets those 8; an invalid image is replaced
#      by the next seed in the ranking until the line has 8 valid images (or the 48 run out).
#   C. r with the true genetic values on the 132 lines NOT used for screening (and all 198),
#      against seeds 0-3, the 4-seed average and all 16 seeds.
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, VALID_TRAITS, load_tables, measure
import numpy as np
import pandas as pd
import torch
from latent_diffusion.models.unet import DenoisingUNet
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

CK = 'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
VAE = 'E:/D2D/DNA_to_Display/models/from_hellbender/litevae_ril_v5_v3_best.pt'
N_KEEP, SCREEN_SEEDS = 8, range(16, 48)
GT = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
dev = torch.device('cuda')
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
screen_lines = val[::3]
eval_lines = [g for g in val if g not in set(screen_lines)]
lo, hi = meta[VALID_TRAITS].quantile(0.005), meta[VALID_TRAITS].quantile(0.995)
pool_path = OUT / 'traits_windows_e1200_seedpool.csv'
pool = pd.read_csv(pool_path) if pool_path.exists() else pd.read_csv(OUT / 'traits_windows_e1200_16seeds.csv')
have = set(zip(pool.genotype, pool.seed))

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


def generate(jobs, label):
    global pool
    jobs = [j for j in jobs if j not in have]
    if not jobs:
        return
    rows, t0 = [], time.time()
    with torch.no_grad():
        for b in range(0, len(jobs), 8):
            batch = jobs[b:b + 8]
            imgs = generate_widiv_batch(enc, unet, sched, decoder, cond[[local[g] for g, _ in batch]],
                                        [s for _, s in batch], dev, (ucfg['latent_channels'], 64, 64), 50,
                                        latent_mean=ck['latent_channel_mean'])
            rows += [{'genotype': g, 'seed': s, 'role': 'validation', **measure(im)} for (g, s), im in zip(batch, imgs)]
            if b % 400 == 0:
                print(f'  {label}: {b + len(batch)}/{len(jobs)} images, {time.time() - t0:.0f}s', flush=True)
    new = pd.DataFrame(rows)
    new['valid'] = np.logical_and.reduce([(new[t] >= lo[t]) & (new[t] <= hi[t]) for t in VALID_TRAITS])
    pool = pd.concat([pool, new], ignore_index=True)
    pool.to_csv(pool_path, index=False)
    have.update(zip(new.genotype, new.seed))
    print(f'{label}: {len(new)} images in {time.time() - t0:.0f}s', flush=True)


# A. Screen.
generate([(g, s) for s in SCREEN_SEEDS for g in screen_lines], 'screen')
sc = pool[pool.genotype.isin(screen_lines)]
rank = sc.groupby('seed').valid.mean().sort_values(ascending=False, kind='stable')
rank.rename('valid_share_screen_lines').to_csv(OUT / 'seed_ranking.csv')
print('seed ranking (valid share on screening lines):', rank.round(3).to_dict(), flush=True)
order = list(rank.index)
keep = order[:N_KEEP]
print('kept seeds:', keep, flush=True)

# B. The kept seeds for every line, then replacements for invalid images.
generate([(g, s) for s in keep for g in val], 'kept seeds')
nxt = {g: N_KEEP for g in val}
while True:
    v = pool[pool.seed.isin(order)]
    jobs = []
    for g in val:
        cand = order[:nxt[g]]
        n_valid = int(v[(v.genotype == g) & v.seed.isin(cand)].valid.sum())
        if n_valid < N_KEEP and nxt[g] < len(order):
            need = N_KEEP - n_valid
            jobs += [(g, s) for s in order[nxt[g]:nxt[g] + need]]
            nxt[g] = min(nxt[g] + need, len(order))
    if not jobs:
        break
    generate(jobs, 'replacements')
    have.update(jobs)

# Final set: per line, the first 8 valid images in ranking order.
v = pool[pool.valid].copy()
v['rank'] = v.seed.map({s: i for i, s in enumerate(order)})
final = (v.sort_values('rank').groupby('genotype').head(N_KEEP))
final.assign(replacement=~final.seed.isin(keep)).to_csv(OUT / 'traits_windows_e1200_8valid.csv', index=False)
counts = final.groupby('genotype').size().reindex(val).fillna(0)
print(f'lines with 8 valid images: {(counts == N_KEEP).sum()}/{len(val)}; replacement images: '
      f'{int((~final.seed.isin(keep)).sum())}', flush=True)

# C. r with the true genetic values.
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')


def r_set(df, group):
    m = df.groupby('genotype')[GT].mean().reindex(group)
    return {t: np.corrcoef(m[t].dropna(), G.loc[m[t].dropna().index, f'g_{t}'])[0, 1] for t in GT}


p16 = pool[pool.valid & (pool.seed < 16)]
sets = {'seeds 0-3 (valid only)': p16[p16.seed < 4],
        'all 16 seeds (valid only)': p16,
        'best 8 seeds (valid only)': pool[pool.valid & pool.seed.isin(keep)],
        'best 8 seeds + replacements (8 valid each)': final}
rows = []
for name, df in sets.items():
    for gname, group in (('132 lines not used for screening', eval_lines), ('all 198 lines', val)):
        rr = r_set(df, group)
        rows.append({'image_set': name, 'lines': gname, **rr, 'mean': np.mean(list(rr.values()))})
# Typical 4 seeds: the four disjoint sets of seeds 0-15.
for gname, group in (('132 lines not used for screening', eval_lines), ('all 198 lines', val)):
    rs = [r_set(p16[p16.seed.isin(range(4 * k, 4 * k + 4))], group) for k in range(4)]
    rr = {t: np.mean([x[t] for x in rs]) for t in GT}
    rows.append({'image_set': 'typical 4 seeds (avg of 4 sets)', 'lines': gname, **rr, 'mean': np.mean(list(rr.values()))})
R = pd.DataFrame(rows)
R.to_csv(OUT / 'seed_selection_r.csv', index=False)
pd.set_option('display.width', 220)
print(R.round(3).to_string(index=False))
print('done')
