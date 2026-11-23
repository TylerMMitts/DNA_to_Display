# Genotype-gain memorization test (as analyze_genotype_memorization.py, for WiDiv): each real
# image noised once per timestep with fixed noise, denoised with its own line and with a random
# line from the same role. Gain = wrong / own - 1. A trained gain far above the held-out gain
# means the genotype acts as a key to recall memorized images. Same images, noise, timesteps
# and partners for every model. 95% CIs by resampling lines. Inference only.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, r'E:/D2D/DNA_to_Display\code'); sys.path.insert(0, r'E:/D2D/DNA_to_Display\code\widiv')
from widiv_data import make_transforms
from widiv_latents import LatentScaler, ScaledClampScheduler
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_RESULTS_DIR, WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder
from latent_diffusion.models.unet import DenoisingUNet

D = Path('E:/D2D/DNA_to_Display/models/from_hellbender')
CK = {'full e400': 'random_vaev3_epoch_400', 'full e500': 'random_vaev3_epoch_500',
      'literal e500': 'random_noqtl0kb_vaev3_epoch_500', 'strict e500': 'random_noqtl1000kb_orth_vaev3_epoch_500'}
TS = [20, 100, 250, 500, 750, 950]
OUT = WIDIV_RESULTS_DIR / 'noqtl_diagnosis'
dev = torch.device('cuda')
lines, _, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}

ck0 = torch.load(D / f"diffusion_widiv_medium_{CK['full e500']}.pt", map_location='cpu', weights_only=False)
val, test = set(ck0['val_genotypes']), set(ck0['test_genotypes'])
del ck0
md = pd.read_csv(WIDIV_SCALED_METADATA)
md = md[md.genotype.isin(row_of)].copy()
md['role'] = np.where(md.genotype.isin(test), 'test', np.where(md.genotype.isin(val), 'validation', 'trained'))
rng = np.random.default_rng(0)
trained_lines = rng.choice(sorted(md[md.role == 'trained'].genotype.unique()), 150, replace=False)
md = md[(md.role != 'trained') | md.genotype.isin(trained_lines)].reset_index(drop=True)
# wrong partner: another line of the same role, fixed per line
partner = {}
for r in ('trained', 'validation', 'test'):
    ls = sorted(md[md.role == r].genotype.unique())
    perm = rng.permutation(ls)
    while any(a == b for a, b in zip(ls, perm)):
        perm = rng.permutation(ls)
    partner.update(dict(zip(ls, perm)))
md['partner'] = md.genotype.map(partner)
print(md.groupby('role').agg(images=('genotype', 'size'), lines=('genotype', 'nunique')).to_string(), flush=True)

enc_vae, _ = load_widiv_litevae(D / 'litevae_widiv_v3_best.pt', dev)
tf = make_transforms(False)
torch.manual_seed(0)
zs = []
with torch.no_grad():
    for i in range(0, len(md), 6):
        ims = torch.stack([tf(Image.open(WIDIV_SCALED_DIR / f).convert('RGB')) for f in md.scaled_filename.iloc[i:i + 6]])
        zs.append(enc_vae(ims.to(dev))[0].cpu())
z_raw = torch.cat(zs)
del enc_vae
torch.cuda.empty_cache()
gen = torch.Generator().manual_seed(1)
noise = {t: torch.randn(z_raw.shape, generator=gen) for t in TS}
own_rows = [row_of[g] for g in md.genotype]
wrong_rows = [row_of[g] for g in md.partner]

per_image = []
for name, f in CK.items():
    ck = torch.load(D / f'diffusion_widiv_medium_{f}.pt', map_location='cpu', weights_only=False)
    enc, proj = load_widiv_encoder(ck, dev)
    cfg = dict(ck['unet_config']); cfg['channel_mult'] = tuple(cfg['channel_mult'])
    unet = DenoisingUNet(**cfg); unet.load_state_dict(ck['unet_state_dict']); unet.set_store_attention(False)
    unet = unet.to(dev).eval()
    sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
    for k in ('betas', 'alphas', 'alpha_bars'):
        setattr(sched, k, getattr(sched, k).to(dev))
    scaler = LatentScaler.from_checkpoint(ck)
    del ck
    z0 = scaler.encode(z_raw)
    s_own = torch.tensor(proj.transform(dosage[own_rows]), dtype=torch.float32)
    s_wrong = torch.tensor(proj.transform(dosage[wrong_rows]), dtype=torch.float32)
    loss = {c: np.zeros(len(md)) for c in ('own', 'wrong')}
    with torch.no_grad():
        for t in TS:
            for i in range(0, len(md), 16):
                zb, nb = z0[i:i + 16].to(dev), noise[t][i:i + 16].to(dev)
                tb = torch.full((len(zb),), t, device=dev, dtype=torch.long)
                zt = sched.add_noise(zb, nb, tb)
                for c, s in (('own', s_own), ('wrong', s_wrong)):
                    e = ((unet(zt, tb, enc(s[i:i + 16].to(dev))) - nb) ** 2).mean((1, 2, 3)).cpu().numpy()
                    loss[c][i:i + 16] += e / len(TS)
    per_image.append(pd.DataFrame({'model': name, 'image': md.scaled_filename, 'genotype': md.genotype,
                                   'role': md.role, 'own': loss['own'], 'wrong': loss['wrong']}))
    print('done', name, flush=True)
    del enc, unet
    torch.cuda.empty_cache()

P = pd.concat(per_image)
P.to_csv(OUT / 'genotype_gain_per_image.csv', index=False)
rows = []
brng = np.random.default_rng(0)
for (name, role), g in P.groupby(['model', 'role'], sort=False):
    tab = g.groupby('genotype')[['own', 'wrong']].mean()
    gain = tab.wrong.mean() / tab.own.mean() - 1
    boot = []
    for _ in range(2000):
        pick = tab.iloc[brng.integers(0, len(tab), len(tab))]
        boot.append(pick.wrong.mean() / pick.own.mean() - 1)
    rows.append({'model': name, 'role': role, 'lines': len(tab), 'own_loss': tab.own.mean(),
                 'gain_pct': 100 * gain, 'lo': 100 * np.percentile(boot, 2.5), 'hi': 100 * np.percentile(boot, 97.5)})
S = pd.DataFrame(rows)
S.to_csv(OUT / 'genotype_gain_summary.csv', index=False)
pd.set_option('display.width', 200)
print(S.round(4).to_string(index=False))
w = S.pivot(index='model', columns='role', values='gain_pct')
print('\ntrained gain / test gain:', (w.trained / w.test).round(2).to_dict())
