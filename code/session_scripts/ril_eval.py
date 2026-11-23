# Checkpoint selection and genetic-signal analysis for the RIL v5 diffusion runs.
# Inference only, validation and trained lines only (no test line is touched).
# Per checkpoint:
#   1. genotype gain: denoising loss of a real image with its own genotype vs the
#      least-related line's (shared noise, t = 20-950, root region), on the 198
#      validation lines and 198 trained lines; trained/validation = memorisation.
#   2. 4 seeds (0-3, same noise for every line, matched start noise) for every
#      validation line and 100 trained lines; traits by colour; validity as in
#      generate_all_lines.py; line means of valid images vs observed phenotypes.
# Usage: python ril_eval.py <litevae.pt> <checkpoint.pt> [...]
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, TRAITS, VALID_TRAITS, load_tables, measure

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from latent_diffusion.models.unet import DenoisingUNet
from model_toolkit import save_grid
from widiv_data import make_transforms
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_KINSHIP_MATRIX, WIDIV_SCALED_DIR
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

TIMESTEPS, N_DRAWS = (20, 100, 250, 500, 750, 950), 2
SEEDS = (0, 1, 2, 3)
N_TRAIN_GAIN, N_TRAIN_GEN = 198, 100
dev = torch.device('cuda')
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
trained_all = sorted(g for g in meta.index if g not in set(val))
rng = np.random.default_rng(0)
trained = sorted(rng.choice(trained_all, N_TRAIN_GAIN, replace=False))
trained_gen = trained[:N_TRAIN_GEN]
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
lo, hi = meta[VALID_TRAITS].quantile(0.005), meta[VALID_TRAITS].quantile(0.995)
real_measured = pd.read_csv(OUT / 'real_measured.csv', index_col=0)
KIN = pd.read_csv(WIDIV_KINSHIP_MATRIX, index_col=0)

enc_vae, dec_vae = load_widiv_litevae(Path(sys.argv[1]), dev)
sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
sched.betas, sched.alphas, sched.alpha_bars = (sched.betas.to(dev), sched.alphas.to(dev),
                                               sched.alpha_bars.to(dev))
tf = make_transforms(False)


def least_related(group):
    k = KIN.loc[group, group].to_numpy().copy()
    np.fill_diagonal(k, np.inf)
    return {g: group[int(np.argmin(k[i]))] for i, g in enumerate(group)}


def encode_lines(group, seed):
    torch.manual_seed(seed)
    zs, masks = [], []
    for g in group:
        x = tf(Image.open(WIDIV_SCALED_DIR / meta.loc[g, 'scaled_filename']).convert('RGB'))[None].to(dev)
        with torch.no_grad():
            _, zm, zl = enc_vae(x)
        zs.append(zm + torch.exp(0.5 * zl) * torch.randn_like(zm))
        masks.append(F.max_pool2d((x.min(1, keepdim=True).values < 0.96).float(), 8) > 0)
    return torch.cat(zs), torch.cat(masks)


def boot(v, n=2000):
    r = np.random.default_rng(0)
    m = [np.mean(r.choice(v, len(v))) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


print('encoding real images and finding least-related partners', flush=True)
z_va, m_va = encode_lines(val, 1)
z_tr, m_tr = encode_lines(trained, 2)
p_va, p_tr = least_related(val), least_related(trained)


# cond: SNP features for val + trained lines, rows in that order (local_row).
def gain(enc, unet, cond, local_row, z, masks, group, partner):
    own_i = [local_row[g] for g in group]
    wrong_i = [local_row[partner[g]] for g in group]
    own, wrong = torch.zeros(len(group)), torch.zeros(len(group))
    with torch.no_grad():
        for t in TIMESTEPS:
            for d in range(N_DRAWS):
                noise = torch.randn(z.shape, generator=torch.Generator().manual_seed(1000 * t + d)).to(dev)
                for s in range(0, len(group), 16):
                    sl = slice(s, s + 16)
                    zb, nb, mb = z[sl], noise[sl], masks[sl].float()
                    tb = torch.full((zb.shape[0],), t, device=dev, dtype=torch.long)
                    ab = sched.alpha_bars[tb].view(-1, 1, 1, 1)
                    zt = ab.sqrt() * zb + (1 - ab).sqrt() * nb
                    for idx, acc in ((own_i, own), (wrong_i, wrong)):
                        err = (unet(zt, tb, enc(cond[idx[sl]])) - nb) ** 2
                        acc[sl] += ((err * mb).sum((1, 2, 3)) / (mb.sum((1, 2, 3)) * err.shape[1])).cpu()
    return (wrong / own - 1).numpy()


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan


summary_path = OUT / 'summary.csv'
summary = pd.read_csv(summary_path).to_dict('records') if summary_path.exists() else []
done = {(r['run'], int(r['epoch'])) for r in summary}
examples = {}
# GAIN_ONLY=1: only val loss and genotype gain, for picking which checkpoints to generate from.
GAIN_ONLY = __import__('os').environ.get('GAIN_ONLY') == '1'
gains = []
for path in map(Path, sys.argv[2:]):
    run, epoch = path.stem.replace('diffusion_ril_v5_medium_', '').rsplit('_epoch_', 1)
    run = ('qtlonly' if 'qtlonly' in run else 'windowsorth' if 'windows' in run and '_orth' in run
           else 'hybrid' if 'hybrid' in run else 'windows' if 'windows' in run else 'pca')
    # SUMMARY_ONLY=1: hybrid with its window tokens always null (what the image prediction carries alone).
    if run == 'hybrid' and __import__('os').environ.get('SUMMARY_ONLY') == '1':
        run = 'hybridsummary'
    epoch = int(epoch)
    per_path = OUT / f'traits_{run}_e{epoch}.csv'
    t0 = time.time()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck, dev)
    if run == 'hybridsummary':
        enc.drop_windows = True
    ucfg = dict(ck['unet_config']); ucfg['channel_mult'] = tuple(ucfg['channel_mult'])
    unet = DenoisingUNet(**ucfg); unet.load_state_dict(ck['unet_state_dict']); unet.set_store_attention(False)
    unet = unet.to(dev).eval()
    scaler = LatentScaler.from_checkpoint(ck)
    sched.set_scaler(scaler)
    decoder = ScaledDecoder(dec_vae, scaler)
    # SNP features for validation + trained lines only; partners come from the same groups.
    names = val + trained
    cond_t = torch.tensor(proj.project_lines(dosage[[row_of[g] for g in names]], names),
                          dtype=torch.float32, device=dev)
    row_of_local = {g: i for i, g in enumerate(names)}
    g_va = boot(gain(enc, unet, cond_t, row_of_local, z_va, m_va, val, p_va))
    g_tr = boot(gain(enc, unet, cond_t, row_of_local, z_tr, m_tr, trained, p_tr))
    t1 = time.time()
    if GAIN_ONLY:
        gains.append({'run': run, 'epoch': epoch, 'val_loss': ck['val_loss'], 'train_loss': ck['loss'],
                      'gain_val': g_va[0], 'gain_val_lo': g_va[1], 'gain_val_hi': g_va[2],
                      'gain_trained': g_tr[0], 'gain_trained_lo': g_tr[1], 'gain_trained_hi': g_tr[2],
                      'memorisation_ratio': g_tr[0] / g_va[0], 'excess_trained': g_tr[0] - g_va[0]})
        pd.DataFrame(gains).to_csv(OUT / 'gain_summary.csv', index=False)
        print(f"{run} e{epoch}: val_loss {ck['val_loss']:.5f} | gain val {g_va[0]:+.2%} [{g_va[1]:+.2%},{g_va[2]:+.2%}] "
              f"trained {g_tr[0]:+.2%} [{g_tr[1]:+.2%},{g_tr[2]:+.2%}] ratio {g_tr[0] / g_va[0]:.2f}x | {t1 - t0:.0f}s",
              flush=True)
        del enc, unet, ck
        torch.cuda.empty_cache()
        continue
    if per_path.exists():
        per = pd.read_csv(per_path)
        with torch.no_grad():
            imgs = generate_widiv_batch(enc, unet, sched, decoder, cond_t[[row_of_local[g] for g in val[:6]]],
                                        [0] * 6, dev, (ucfg['latent_channels'], 64, 64), 50,
                                        latent_mean=ck['latent_channel_mean'])
        for g, im in zip(val[:6], imgs):
            examples[(run, epoch, g)] = im
    else:
        rows = []
        jobs = [(g, s) for g in val + trained_gen for s in SEEDS]
        with torch.no_grad():
            for b in range(0, len(jobs), 8):
                batch = jobs[b:b + 8]
                imgs = generate_widiv_batch(enc, unet, sched, decoder, cond_t[[row_of_local[g] for g, _ in batch]],
                                            [s for _, s in batch], dev, (ucfg['latent_channels'], 64, 64), 50,
                                            latent_mean=ck['latent_channel_mean'])
                for (g, s), im in zip(batch, imgs):
                    rows.append({'genotype': g, 'seed': s, 'role': 'validation' if g in set(val) else 'trained',
                                 **measure(im)})
                    if s == 0 and g in val[:6]:
                        examples[(run, epoch, g)] = im
        per = pd.DataFrame(rows)
        per['valid'] = np.logical_and.reduce([(per[t] >= lo[t]) & (per[t] <= hi[t]) for t in VALID_TRAITS])
        per.to_csv(per_path, index=False)
    row = {'run': run, 'epoch': epoch, 'val_loss': ck['val_loss'], 'train_loss': ck['loss'],
           'gain_val': g_va[0], 'gain_val_lo': g_va[1], 'gain_val_hi': g_va[2],
           'gain_trained': g_tr[0], 'gain_trained_lo': g_tr[1], 'gain_trained_hi': g_tr[2],
           'valid_val': per[per.role == 'validation'].valid.mean(),
           'valid_trained': per[per.role == 'trained'].valid.mean()}
    ok = per[per.valid]
    for role, group in (('val', val), ('trained', trained_gen)):
        means = ok[ok.genotype.isin(group)].groupby('genotype')[TRAITS].mean().reindex(group)
        obs = meta.loc[group, TRAITS]
        for t in TRAITS:
            row[f'r_{role}_{t}'] = corr(means[t], obs[t])
            if role == 'val':
                row[f'sd_ratio_val_{t}'] = float(means[t].std() / real_measured.loc[group, t].std())
                row[f'r_valmeasured_{t}'] = corr(means[t], real_measured.loc[group, t])
    v = ok[ok.role == 'validation']
    for t in TRAITS:
        tot = v[t].var()
        row[f'line_share_{t}'] = float(v.groupby('genotype')[t].mean().var() / tot)
        row[f'seed_share_{t}'] = float(v.groupby('seed')[t].mean().var() / tot)
    summary = [r for r in summary if (r['run'], int(r['epoch'])) != (run, epoch)] + [row]
    pd.DataFrame(summary).sort_values(['run', 'epoch']).to_csv(summary_path, index=False)
    print(f"{run} e{epoch}: val_loss {row['val_loss']:.5f} | gain val {g_va[0]:+.2%} [{g_va[1]:+.2%},{g_va[2]:+.2%}] "
          f"trained {g_tr[0]:+.2%} ({g_tr[0] / g_va[0]:.2f}x) | valid {row['valid_val']:.0%} | r val "
          + ' '.join(f"{t.split('_')[0]}:{row[f'r_val_{t}']:.2f}" for t in TRAITS)
          + f" | {t1 - t0:.0f}s + {time.time() - t1:.0f}s", flush=True)
    del enc, unet, ck
    torch.cuda.empty_cache()

if examples:
    runs = sorted({(r, e) for r, e, _ in examples})
    grid = [[np.asarray(Image.open(WIDIV_SCALED_DIR / meta.loc[g, 'scaled_filename']).convert('RGB'))]
            + [examples[(r, e, g)] for r, e in runs] for g in val[:6]]
    save_grid(grid, OUT / 'validation_examples_seed0.png', row_labels=val[:6],
              col_labels=['real'] + [f'{r} e{e}' for r, e in runs])
print('done')
