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
OUT1 = OUT / 'option1_checks_v2'
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
# Version 2: a tissue-layout fingerprint. Each pixel gets the nearest rendered
# tissue colour (widiv_traits palette, learned from image + mask pairs); the class
# map is resampled in polar coordinates around the image centre (roots are
# centred), and described without reference to rotation:
#   - root and stele radius (absolute)
#   - share of each class at 48 relative radii (0 = centre, 1 = root edge)
#   - share of each class in 24 bins of absolute depth from the edge (5.4 um each)
#   - white gaps around each relative ring: number, mean length (absolute px)
#   - angular spectrum of the white and sclerenchyma rings, freqs 1-24, per ring
import widiv_traits as WT

cache = OUT1 / 'summary_v2.npy'
if cache.exists():
    S = np.load(cache)
else:
    dev = torch.device('cuda')
    centres = torch.tensor(WT.CENTRES, device=dev)                       # (C, 3)
    NC = len(centres)
    WH = WT.WHITE
    SCL = WT.IDX[3]
    STELE = [WT.IDX[l] for l in (6, 7, 8, 9)]
    RA, A = 512, 512                                                     # 0.5 px radial steps
    rad = torch.arange(RA, dtype=torch.float32) * 0.5
    ang = torch.arange(A, dtype=torch.float32) * 2 * np.pi / A
    xs = (255.5 + rad[:, None] * torch.cos(ang)[None]) / 255.5 - 1
    ys = (255.5 - rad[:, None] * torch.sin(ang)[None]) / 255.5 - 1
    grid = torch.stack([xs, ys], -1)[None].to(dev)
    NR, ND, KF = 48, 24, 24
    feats = []
    for i, g in enumerate(names):
        x = torch.from_numpy(np.asarray(Image.open(WIDIV_SCALED_DIR / meta.loc[g, 'scaled_filename']).convert('RGB'))
                             .copy()).to(dev).float()
        cls = ((x[..., None, :] - centres) ** 2).sum(-1).argmin(-1)      # (512, 512)
        onehot = F.one_hot(cls, NC).permute(2, 0, 1)[None].float()
        P = F.grid_sample(onehot, grid, mode='nearest', align_corners=True)[0].argmax(0)   # (RA, A) class
        tissue = P != WH
        # outermost tissue sample along each angle -> root radius (median over angles)
        idx = torch.arange(RA, device=dev)[:, None].expand(RA, A)
        edge = torch.where(tissue, idx, torch.zeros_like(idx)).amax(0).float()             # (A,) in samples
        r_root = edge.median()
        st = torch.isin(P, torch.tensor(STELE, device=dev))
        r_stele = torch.where(st, idx, torch.zeros_like(idx)).amax(0).float().median()
        # relative-radius rings
        rel = (torch.arange(NR, device=dev).float() + 0.5) / NR * 1.02
        rows_rel = (rel * r_root).round().long().clamp(0, RA - 1)
        ring = P[rows_rel]                                                                  # (NR, A)
        share_rel = F.one_hot(ring, NC).float().mean(1)                                     # (NR, NC)
        # absolute depth from the edge (per angle)
        depth = torch.arange(ND, device=dev)[:, None] * 2                                   # samples = 1 px
        rows_d = (edge[None] - depth).clamp(0, RA - 1).long()
        band = P.gather(0, rows_d)                                                          # (ND, A)
        share_depth = F.one_hot(band, NC).float().mean(1)
        # white gaps around each relative ring
        w = (ring == WH).float()
        starts = (w * (1 - w.roll(1, 1))).sum(1)                                            # (NR,)
        mean_len = w.sum(1) / starts.clamp(min=1) * (rel * r_root * 0.5 * 2 * np.pi / A)   # px
        spec_w = torch.log1p(torch.fft.rfft(w, dim=1).abs()[:, 1:KF + 1])
        spec_s = torch.log1p(torch.fft.rfft((ring == SCL).float(), dim=1).abs()[:, 1:KF + 1])
        feats.append(torch.cat([torch.stack([r_root, r_stele]) / 2, share_rel.flatten(), share_depth.flatten(),
                                starts, mean_len, spec_w.flatten(), spec_s.flatten()]).cpu().numpy())
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def ridge_obs(Xtr, Xte):
    # summary -> each observed phenotype, fitted on trained lines that have it
    out = np.full((len(Xte), len(TRAITS)), np.nan)
    for j in range(len(TRAITS)):
        ok = np.isfinite(obs[:n_tr, j])
        out[:, j] = ridge_feat(Xtr[ok], obs[:n_tr][ok, j:j + 1], Xte)[:, 0]
    return out


# ---------------------------------------------------------------- check 1
obs = ph.loc[names, TRAITS].to_numpy(float)
gv = G.loc[names, [f'g_{t}' for t in TRAITS]].to_numpy(float)
check1 = []
for k in (16, 32, 64, 128, 256, 512):
    pred = ridge_obs(PCS[:n_tr, :k], PCS[n_tr:, :k])
    check1.append({'summary_pcs': k, 'variance': float(var[:k].sum()),
                   **{t: corr(pred[:, j], obs[n_tr:, j]) for j, t in enumerate(TRAITS)}})
    log('check 1', k, {t: round(check1[-1][t], 3) for t in TRAITS})
check1 = pd.DataFrame(check1)
check1.to_csv(OUT1 / 'check1_summary_holds_traits.csv', index=False)
K_USE = int(check1.assign(m=check1[TRAITS].mean(1)).sort_values('m').summary_pcs.iloc[-1])
log('using', K_USE, 'summary PCs')

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
    return ridge_obs(train_X, apply_X)


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
    direct = np.column_stack([kernel_ridge(K, obs[:, j:j + 1], tr_rows[np.isfinite(obs[:n_tr, j])], va_rows)[:, 0]
                              for j in range(len(TRAITS))])
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
