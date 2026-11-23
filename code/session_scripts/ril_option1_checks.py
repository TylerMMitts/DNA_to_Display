# Feasibility checks for option 1: condition the UNet on an out-of-fold genomic
# prediction of a rotation-invariant image summary instead of genotype tokens.
# Trained lines (1,780) fit everything; validation lines (198) are scored; test
# lines are never read. The answer key's genetic values are used only to score.
#
# Summary vector, from each 512 px real image (roots are centred):
#   polar resampling, 128 radii x 256 angles, per RGB channel
#   - radial profile: mean over angle at each radius        (3 x 128)
#   - angular spectrum: |FFT over angle|, frequencies 1-40   (3 x 128 x 40), log1p
#   standardised, then PCA fitted on trained lines.
#
# Check 1: does the summary hold the traits?  ridge summary -> observed phenotype
#          (same image), fitted on trained lines, r on validation lines.
# Check 2: is the predicted summary as informative as GBLUP?  kernel ridge SNP ->
#          summary PCs (out-of-fold for trained lines), then
#          (a) probe: ridge predicted-summary -> true g, fitted on trained, r on val
#          (b) composed: the check-1 map (summary -> trait) applied to the predicted
#              summary, r with true g; what an image drawn from it would measure
#          next to GBLUP -> g and the current windows tokens (bottleneck_probe.csv).
# Check 3: the same with epistatic kernels (Gaussian; additive + additive x additive).
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_SCALED_DIR
from widiv_snp_encoding import load_widiv_genotypes

t0 = time.time()
KEY = Path('E:/D2D/DNA_to_Display/dataset/ril_v5/genetic/answer_key_ril_v5')
OUT1 = OUT / 'option1_checks'
OUT1.mkdir(exist_ok=True)
meta, split = load_tables()
val = pd.read_csv(OUT / 'validation_lines.csv').genotype.tolist()
tr = sorted(g for g in meta.index if g not in set(val))
names = tr + val
n_tr = len(tr)
ph = pd.read_csv(KEY / 'phenotypes_with_blind_trait.csv')
ph = ph[ph.line.map(split) == 'train'].set_index('line')
G = pd.read_csv(KEY / 'genetic_values.csv')
G = G[G.line.map(split) == 'train'].set_index('line')
TRAITS = ['stele_diam_um', 'mx_n', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers', 'n_lacunae']


def log(*a):
    print(f'[{time.time() - t0:5.0f}s]', *a, flush=True)


# ---------------------------------------------------------------- summary vectors
cache = OUT1 / 'summary_raw.npy'
if cache.exists():
    S = np.load(cache)
else:
    dev = torch.device('cuda')
    R, A, KF = 128, 256, 40
    rad = torch.arange(R, dtype=torch.float32) * 2 + 0.5                  # 0.5 .. 254.5 px
    ang = torch.arange(A, dtype=torch.float32) * 2 * np.pi / A
    xs = (255.5 + rad[:, None] * torch.cos(ang)[None]) / 255.5 - 1
    ys = (255.5 - rad[:, None] * torch.sin(ang)[None]) / 255.5 - 1
    grid = torch.stack([xs, ys], -1)[None].to(dev)                       # (1, R, A, 2)
    feats = []
    for i, g in enumerate(names):
        x = torch.from_numpy(np.asarray(Image.open(WIDIV_SCALED_DIR / meta.loc[g, 'scaled_filename']).convert('RGB'))
                             .copy()).permute(2, 0, 1)[None].float().to(dev) / 255
        p = F.grid_sample(x, grid, align_corners=True)[0]                 # (3, R, A)
        radial = p.mean(-1)                                               # (3, R)
        spec = torch.log1p(torch.fft.rfft(p, dim=-1).abs()[..., 1:KF + 1])  # (3, R, KF)
        feats.append(torch.cat([radial.flatten(), spec.flatten()]).cpu().numpy())
        if i % 500 == 0:
            log(f'summary {i}/{len(names)}')
    S = np.stack(feats).astype(np.float32)
    np.save(cache, S)
log('summary', S.shape)
mu, sd = S[:n_tr].mean(0), S[:n_tr].std(0) + 1e-6
Sz = (S - mu) / sd
U, s, Vt = np.linalg.svd(Sz[:n_tr], full_matrices=False)
var = s ** 2 / (s ** 2).sum()
PCS = Sz @ Vt.T                                                           # all lines, all PCs
log('summary PCA: components for 90/95/99% variance',
    [int(np.searchsorted(np.cumsum(var), v) + 1) for v in (0.9, 0.95, 0.99)])


# ---------------------------------------------------------------- ridge helpers
def ridge_feat(Xtr, Ytr, Xte, lams=(0.1, 1, 10, 100, 1e3, 1e4, 1e5)):
    # Feature-space ridge with intercept; lambda per target by 5-fold CV on train.
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - xm, Ytr - ym
    folds = np.random.default_rng(0).permutation(np.arange(len(Xtr)) % 5)
    err = np.zeros((len(lams), Ytr.shape[1]))
    for f in range(5):
        a, b = folds != f, folds == f
        xa = Xc[a] - Xc[a].mean(0)
        u, sv, vt = np.linalg.svd(xa, full_matrices=False)
        uty = u.T @ (Yc[a] - Yc[a].mean(0))
        for li, lam in enumerate(lams):
            w = vt.T @ ((sv / (sv ** 2 + lam))[:, None] * uty)
            pred = (Xc[b] - Xc[a].mean(0)) @ w + Yc[a].mean(0)
            err[li] += ((pred - Yc[b]) ** 2).sum(0)
    u, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    uty = u.T @ Yc
    out = np.empty((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        lam = lams[int(np.argmin(err[:, j]))]
        w = vt.T @ ((sv / (sv ** 2 + lam)) * uty[:, j])
        out[:, j] = (Xte - xm) @ w + ym[j]
    return out


def kernel_ridge(K, Y, rows_fit, rows_pred, lams=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)):
    # Dual ridge on a kernel; lambda per target column by leave-one-out on rows_fit.
    Kf = K[np.ix_(rows_fit, rows_fit)]
    m = Kf.mean(0)
    Kc = Kf - m[None] - m[:, None] + m.mean()
    Kp = K[np.ix_(rows_pred, rows_fit)]
    Kp = Kp - Kp.mean(1, keepdims=True) - m[None] + m.mean()
    ev, V = np.linalg.eigh(Kc)
    ev = np.clip(ev, 0, None)
    scale = ev.mean()
    Ym = Y[rows_fit].mean(0)
    Yc = Y[rows_fit] - Ym
    VtY = V.T @ Yc
    best_err, best = np.full(Y.shape[1], np.inf), np.zeros(Y.shape[1])
    for lam in lams:
        f = ev / (ev + lam * scale)
        H = V @ (f[:, None] * VtY)
        h = (V ** 2 * f).sum(1) + 1 / len(rows_fit)
        e = (((Yc - H) / (1 - h)[:, None]) ** 2).mean(0)
        better = e < best_err
        best_err[better], best[better] = e[better], lam
    out = np.empty((len(rows_pred), Y.shape[1]))
    KpV = Kp @ V
    for lam in np.unique(best):
        cols = best == lam
        out[:, cols] = KpV @ ((1 / (ev + lam * scale))[:, None] * VtY[:, cols]) + Ym[cols]
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- check 1
obs = ph.loc[names, TRAITS].to_numpy(float)
gv = G.loc[names, [f'g_{t}' for t in TRAITS]].to_numpy(float)
check1 = []
for k in (16, 32, 64, 128, 256, 512):
    pred = ridge_feat(PCS[:n_tr, :k], obs[:n_tr], PCS[n_tr:, :k])
    check1.append({'summary_pcs': k, 'variance': float(var[:k].sum()),
                   **{t: corr(pred[:, j], obs[n_tr:, j]) for j, t in enumerate(TRAITS)}})
    log('check 1', k, {t: round(check1[-1][t], 3) for t in TRAITS})
check1 = pd.DataFrame(check1)
check1.to_csv(OUT1 / 'check1_summary_holds_traits.csv', index=False)
K_USE = 128

# ---------------------------------------------------------------- kernels on SNPs
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
X = dosage[[row_of[g] for g in names]].astype(np.float64)
Z = (X - X[:n_tr].mean(0)) / (X[:n_tr].std(0) + 1e-6)
KA = Z @ Z.T / Z.shape[1]
d2 = np.diag(KA)[:, None] + np.diag(KA)[None] - 2 * KA
KG = np.exp(-d2 / np.median(d2[:n_tr, :n_tr]))
KAA = KA * KA
KAA = KAA / np.mean(np.diag(KAA[:n_tr, :n_tr]))
KERNELS = {'additive (GBLUP)': KA, 'gaussian': KG, 'additive + AxA': KA + KAA}
log('kernels built')

tr_rows, va_rows = np.arange(n_tr), np.arange(n_tr, len(names))
folds = np.random.default_rng(1).permutation(np.arange(n_tr) % 10)
Y = PCS[:, :K_USE]
# check-1 map (summary PCs -> trait), fitted on trained lines' real summaries
W_obs = {}


def summary_to_trait(train_X, apply_X):
    return ridge_feat(train_X, obs[:n_tr], apply_X)


rows = []
for kname, K in KERNELS.items():
    # out-of-fold predicted summaries for trained lines, full-fit for validation
    P = np.zeros_like(Y)
    for f in range(10):
        fit, held = tr_rows[folds != f], tr_rows[folds == f]
        P[held] = kernel_ridge(K, Y, fit, held)
    P[va_rows] = kernel_ridge(K, Y, tr_rows, va_rows)
    # quality of the summary prediction itself
    r_pcs = [corr(P[va_rows, j], Y[va_rows, j]) for j in range(8)]
    # (a) probe: predicted summary -> g (fitted on trained OOF), r on val
    probe = ridge_feat(P[:n_tr], gv[:n_tr], P[n_tr:])
    # (b) composed: real-summary -> phenotype map applied to predicted summary
    comp = summary_to_trait(PCS[:n_tr, :K_USE], P[n_tr:])
    # direct kernel prediction of phenotypes (GBLUP-type) for reference
    direct = kernel_ridge(K, obs, tr_rows, va_rows)
    for j, t in enumerate(TRAITS):
        rows.append({'kernel': kname, 'trait': t, 'probe_r_g': corr(probe[:, j], gv[n_tr:, j]),
                     'composed_r_g': corr(comp[:, j], gv[n_tr:, j]),
                     'direct_kernel_r_g': corr(direct[:, j], gv[n_tr:, j])})
    log(f'{kname}: summary PC1-8 predicted r on val {np.round(r_pcs, 2)}')
    np.save(OUT1 / f'predicted_summary_{kname.split()[0].replace("+", "")}.npy', P)
res = pd.DataFrame(rows)
res.to_csv(OUT1 / 'check2_3_predicted_summary.csv', index=False)
tok = pd.read_csv(OUT / 'bottleneck_probe.csv').set_index(['model', 'stage'])
gen = pd.read_csv(OUT / 'vs_genetic_values.csv').set_index('trait')
tab = res.pivot_table(index='trait', columns='kernel', values=['composed_r_g', 'probe_r_g'])
tab[('windows tokens', 'probe')] = [tok.loc[('windows', 'tokens'), f'g_{t}'] for t in tab.index]
tab[('windows e800', 'generated')] = [gen.loc[t, 'windows_e800_vs_g'] if t in gen.index else np.nan for t in tab.index]
pd.set_option('display.width', 250)
print(tab.round(3).to_string())
print(tab.mean().round(3).to_string())
log('done')
