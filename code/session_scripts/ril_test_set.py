# Best model (windows e1200) on the 495 TEST lines, first use of the test set.
# Same image set as best_model_trait_scatter_blup_8valid.png: the 8 best seeds by
# validity (seed_ranking.csv, ranked on validation screening lines, so nothing was
# chosen on the test lines), each invalid image replaced by the next seed in the
# ranking until the line has 8 valid images. Then the line + seed BLUP, r with the
# true genetic values, and GBLUP (trained lines' phenotypes -> test lines).
# Test-line phenotypes are not read; the answer key is used for scoring only.
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, VALID_TRAITS, load_tables, measure
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from latent_diffusion.models.unet import DenoisingUNet
from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
from widiv_litevae import load_widiv_litevae
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_RESULTS_DIR
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder

CK = 'E:/D2D/DNA_to_Display/models/from_hellbender/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_1200.pt'
VAE = 'E:/D2D/DNA_to_Display/models/from_hellbender/litevae_ril_v5_v3_best.pt'
N_KEEP = 8
FIG = WIDIV_RESULTS_DIR / 'figures'
T = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
NAME = {'stele_diam_um': 'Stele diameter', 'mx_n': 'Metaxylem number', 'mx_diam_um': 'Metaxylem diameter',
        'cortex_width_um': 'Cortex width', 'rca_pct': 'Aerenchyma (RCA)', 'mcs_layers': 'Sclerenchyma (MCS)'}
UNIT = {'stele_diam_um': 'µm', 'mx_n': 'vessels', 'mx_diam_um': 'µm', 'cortex_width_um': 'µm',
        'rca_pct': '% (measured)', 'mcs_layers': 'band width (measured)'}
dev = torch.device('cuda')
meta, split = load_tables()                      # trained + validation phenotypes only
test = sorted(split.index[split == 'test'])
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
lo, hi = meta[VALID_TRAITS].quantile(0.005), meta[VALID_TRAITS].quantile(0.995)
order = pd.read_csv(OUT / 'seed_ranking.csv').seed.tolist()
keep = order[:N_KEEP]
print(f'{len(test)} test lines; seeds {keep}', flush=True)

pool_path = OUT / 'traits_windows_e1200_test_pool.csv'
pool = pd.read_csv(pool_path) if pool_path.exists() else pd.DataFrame()
have = set(zip(pool.genotype, pool.seed)) if len(pool) else set()

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
cond = torch.tensor(proj.project_lines(dosage[[row_of[g] for g in test]], test), dtype=torch.float32, device=dev)
local = {g: i for i, g in enumerate(test)}


def save(rows):
    global pool
    new = pd.DataFrame(rows)
    new['valid'] = np.logical_and.reduce([(new[t] >= lo[t]) & (new[t] <= hi[t]) for t in VALID_TRAITS])
    pool = pd.concat([pool, new], ignore_index=True)
    pool.to_csv(pool_path, index=False)
    have.update(zip(new.genotype, new.seed))


def generate(jobs, label):
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
            rows += [{'genotype': g, 'seed': s, 'role': 'test', **measure(im)} for (g, s), im in zip(batch, imgs)]
            if len(rows) >= 400:
                save(rows); rows = []
                print(f'  {label}: {b + len(batch)}/{len(jobs)} images, {time.time() - t0:.0f}s', flush=True)
    if rows:
        save(rows)
    print(f'{label}: {len(jobs)} images in {time.time() - t0:.0f}s', flush=True)


generate([(g, s) for s in keep for g in test], 'kept seeds')
nxt = {g: N_KEEP for g in test}
while True:
    v = pool[pool.seed.isin(order)]
    jobs = []
    for g in test:
        n_valid = int(v[(v.genotype == g) & v.seed.isin(order[:nxt[g]])].valid.sum())
        if n_valid < N_KEEP and nxt[g] < len(order):
            need = N_KEEP - n_valid
            jobs += [(g, s) for s in order[nxt[g]:nxt[g] + need]]
            nxt[g] = min(nxt[g] + need, len(order))
    if not jobs:
        break
    generate(jobs, 'replacements')

v = pool[pool.valid].copy()
v['rank'] = v.seed.map({s: i for i, s in enumerate(order)})
final = v.sort_values('rank').groupby('genotype').head(N_KEEP)
final = final.assign(replacement=~final.seed.isin(keep))
final.to_csv(OUT / 'traits_windows_e1200_test_8valid.csv', index=False)
counts = final.groupby('genotype').size().reindex(test).fillna(0)
print(f'lines with 8 valid images: {(counts == N_KEEP).sum()}/{len(test)}; '
      f'replacement images: {int(final.replacement.sum())}', flush=True)

# GBLUP: trained lines' observed phenotypes -> test lines (same lambda as gblup_baseline.csv).
X = dosage[[row_of[g] for g in tr + test]].astype(np.float64)
Z = (X - X[:len(tr)].mean(0)) / (X[:len(tr)].std(0) + 1e-6)
K = Z @ Z.T / Z.shape[1]
lam = pd.read_csv(OUT / 'gblup_baseline.csv').set_index('trait')['lambda']
G = pd.read_csv('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5/genetic_values.csv').set_index('line')
rows, blups, pts = [], {}, []
for t in T:
    y = meta.loc[tr, t].to_numpy(float)
    a = np.linalg.solve(K[:len(tr), :len(tr)] + lam[t] * np.eye(len(tr)), y - y.mean())
    gb = y.mean() + K[len(tr):, :len(tr)] @ a
    grp = final.groupby('genotype')[t]
    m, n, var = grp.mean().reindex(test), grp.size().reindex(test), grp.var().reindex(test)
    s2_seed = float(np.nansum(var * (n - 1)) / np.nansum(n - 1))
    s2_line = float(np.var(m, ddof=1) - np.nanmean(s2_seed / n))
    mu = float(m.mean())
    b = mu + s2_line / (s2_line + s2_seed / n) * (m - mu)
    blups[t] = b
    g = G.loc[test, f'g_{t}']
    ok = b.notna().to_numpy()
    rel = s2_line / np.var(m, ddof=1)
    r_mean = np.corrcoef(m[ok], g[ok])[0, 1]
    rows.append({'trait': NAME[t], 'n_lines': int(ok.sum()), 'seed_sd': np.sqrt(s2_seed), 'line_sd': np.sqrt(s2_line),
                 'reliability_of_mean': rel, 'r_mean': r_mean, 'r_blup': np.corrcoef(b[ok], g[ok])[0, 1],
                 'r_noise_free': r_mean / np.sqrt(rel), 'r_gblup': np.corrcoef(gb, g)[0, 1]})
    pts += [{'trait': NAME[t], 'line': L, 'true_genetic_value': round(float(gi), 3),
             'generated_blup': round(float(bi), 3)} for L, gi, bi in zip(test, g, b) if np.isfinite(bi)]
R = pd.DataFrame(rows)
R.to_csv(FIG / 'best_model_trait_r_blup_8valid_test.csv', index=False)
pd.DataFrame(pts).to_csv(FIG / 'best_model_scatter_points_8valid_test.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
print('mean r: BLUP %.3f | noise-free %.3f | GBLUP %.3f' % (R.r_blup.mean(), R.r_noise_free.mean(), R.r_gblup.mean()))

BLUE, INK, MUTED = '#2a6fdb', '#1f1f1f', '#6b6b6b'
fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
for ax, t, rr in zip(axes.ravel(), T, rows):
    g = G.loc[test, f'g_{t}'].to_numpy()
    b = blups[t].to_numpy()
    ok = np.isfinite(b)
    ax.scatter(g[ok], b[ok], s=10, color=BLUE, alpha=0.55, linewidths=0)
    c = np.polyfit(g[ok], b[ok], 1)
    xs = np.linspace(g.min(), g.max(), 2)
    ax.plot(xs, np.polyval(c, xs), color=INK, lw=1)
    ax.set_title(NAME[t], fontsize=11, color=INK, loc='left')
    ax.text(0.03, 0.97, f"generated   r = {rr['r_blup']:.2f}\nnoise-free  r = {rr['r_noise_free']:.2f}\n"
            f"GBLUP       r = {rr['r_gblup']:.2f}", transform=ax.transAxes, va='top', ha='left', fontsize=9,
            color=INK, family='monospace', linespacing=1.5)
    ax.set_xlabel('true genetic value', fontsize=9, color=MUTED)
    ax.set_ylabel(f'genetic value of generated roots, {UNIT[t]}', fontsize=9, color=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c8c8c8')
    ax.tick_params(colors=MUTED, labelsize=8)
fig.suptitle(f'Genetic value of generated roots vs true genetic value, {len(test)} test lines (windows, epoch 1200)',
             x=0.01, ha='left', fontsize=13, color=INK)
fig.text(0.01, 0.005, 'Each dot = one line: BLUP of its line effect from 8 valid images per line: the best 8 seeds by '
         'validity (ranked on validation lines), invalid images replaced by the next-best seed.\n'
         'Line + seed-noise random model. Noise-free r = r corrected for seed noise (unlimited seeds). '
         'GBLUP = SNP-only prediction of the same test lines from the trained lines\' phenotypes.\n'
         'Test lines were held out of all training, model and seed choice. All r are with the true genetic value '
         'from the answer key, never seen in training.', fontsize=8, color=MUTED, linespacing=1.5)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig(FIG / 'best_model_trait_scatter_blup_8valid_test.png', dpi=150)
print('wrote', FIG / 'best_model_trait_scatter_blup_8valid_test.png')
