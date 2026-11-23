# GWAS on the RIL training lines, using only data the diffusion model may use:
# genotypes and the observed phenotypes of the 1,780 lines it trains on (non-test lines
# minus the 198 validation lines). Never reads the answer key.
#   - mixed model, LOCO kinship (kinship from every chromosome except the tested one),
#     variance ratio by REML per chromosome (P3D), t-test per marker
#   - multi-locus forward selection: add the best marker as a covariate and rescan until
#     nothing passes the threshold (separates linked QTL)
#   - interaction scan: each selected peak x every marker (main effects in the model),
#     finds loci that act only through an interaction
#   - threshold: Bonferroni 5% genome-wide (0.05 / 10,000 markers; 0.05 / (10,000 x peaks)
#     for interactions), conservative
# Writes results/ril_v5/gwas/: gwas_scan.csv (first scan, every marker), gwas_peaks.csv.
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ril_common import OUT, load_tables
import numpy as np
import pandas as pd
from scipy import stats
from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_RESULTS_DIR, WIDIV_SNP_MAP
from widiv_snp_encoding import load_widiv_genotypes

GW = WIDIV_RESULTS_DIR / 'gwas'
GW.mkdir(exist_ok=True)
TRAITS = ['mx_n', 'stele_diam_um', 'mx_diam_um', 'cortex_width_um', 'rca_pct', 'mcs_layers']
MAX_PEAKS, ALPHA = 15, 0.05
meta, split = load_tables()
val = set(pd.read_csv(OUT / 'validation_lines.csv').genotype)
fit_lines = sorted(g for g in meta.index if g not in val)
lines, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
row_of = {g: i for i, g in enumerate(lines)}
snps = np.asarray(snps)
smap = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[snps]
chrom, cM = smap.chr.to_numpy(), smap.pos_cM.to_numpy()
X = dosage[[row_of[g] for g in fit_lines]].astype(np.float64) - 1.0   # -1 / 0 / 1
n, m = X.shape
poly = X.std(0) > 0
print(f'{n} lines x {m} markers ({(~poly).sum()} monomorphic)', flush=True)
THR = -np.log10(ALPHA / m)
Z = (X - X.mean(0)) / np.where(poly, X.std(0), 1.0)
CHRS = sorted(set(chrom))

# LOCO kinship eigendecompositions and markers rotated into each one's basis.
t0 = time.time()
eig = {}
for c in CHRS:
    other = (chrom != c) & poly
    K = Z[:, other] @ Z[:, other].T / other.sum()
    s, U = np.linalg.eigh(K)
    cols = np.flatnonzero(chrom == c)
    eig[c] = (np.clip(s, 0, None), U, cols, U.T @ X[:, cols])
Kall = Z[:, poly] @ Z[:, poly].T / poly.sum()
s_all, U_all = np.linalg.eigh(Kall)
s_all = np.clip(s_all, 0, None)
print(f'kinship eigendecompositions {time.time() - t0:.0f}s', flush=True)
H_GRID = np.linspace(0.0, 0.99, 100)


def reml_h(s, yr, Cr):
    best, bh = -np.inf, 0.0
    for h in H_GRID:
        V = h * s + (1 - h)
        w = 1 / V
        CtW = Cr.T * w
        A = CtW @ Cr
        beta = np.linalg.solve(A, CtW @ yr)
        r = yr - Cr @ beta
        sig = (r * w) @ r / (len(yr) - Cr.shape[1])
        ll = -0.5 * ((len(yr) - Cr.shape[1]) * np.log(sig) + np.log(V).sum() + np.linalg.slogdet(A)[1])
        if ll > best:
            best, bh = ll, h
    return bh


def resid(Cw, *mats):
    Q, _ = np.linalg.qr(Cw)
    return [M - Q @ (Q.T @ M) for M in mats]


def scan(y, cov_idx, inter=None):
    # -log10 p and effect per marker, given fixed covariate markers cov_idx (+ products in inter).
    logp, beta = np.zeros(m), np.zeros(m)
    for c in CHRS:
        s, U, cols, Xr = eig[c]
        C = np.c_[np.ones(n), X[:, cov_idx]] if len(cov_idx) else np.ones((n, 1))
        if inter is not None:
            C = np.c_[C, X[:, inter[0]] * X[:, inter[1]]]
        Cr, yr = U.T @ C, U.T @ y
        h = reml_h(s, yr, Cr)
        sw = 1 / np.sqrt(h * s + (1 - h))
        yres, Mres = resid(Cr * sw[:, None], yr * sw, Xr * sw[:, None])
        mm = (Mres ** 2).sum(0)
        ok = mm > 1e-8 * n
        b = np.where(ok, Mres.T @ yres / np.where(ok, mm, 1), 0)
        df = n - C.shape[1] - 1
        rss = yres @ yres - b ** 2 * mm
        se = np.sqrt(np.maximum(rss, 1e-12) / df / np.where(ok, mm, 1))
        tt = np.where(ok, b / se, 0)
        # df ~1,770: normal tail, which stays finite where the t tail underflows
        logp[cols] = -(stats.norm.logsf(np.abs(tt)) + np.log(2)) / np.log(10)
        beta[cols] = b
    logp[np.isin(np.arange(m), cov_idx)] = 0
    return logp, beta


def interaction_scan(y, base_idx, p):
    # Test x_p * x_j for every marker j (x_j main effect in the model), LOCO on j's chromosome.
    logp = np.zeros(m)
    for c in CHRS:
        s, U, cols, Xr = eig[c]
        C = np.c_[np.ones(n), X[:, base_idx]]
        Cr, yr = U.T @ C, U.T @ y
        h = reml_h(s, yr, Cr)
        sw = 1 / np.sqrt(h * s + (1 - h))
        Ir = U.T @ (X[:, [p]] * X[:, cols])
        yres, A, B = resid(Cr * sw[:, None], yr * sw, Xr * sw[:, None], Ir * sw[:, None])
        aa = (A ** 2).sum(0)
        ab = (A * B).sum(0)
        okA = aa > 1e-8 * n
        Bp = B - A * np.where(okA, ab / np.where(okA, aa, 1), 0)
        bb = (Bp ** 2).sum(0)
        ok = bb > 1e-6 * n
        b = np.where(ok, Bp.T @ yres / np.where(ok, bb, 1), 0)
        df = n - C.shape[1] - 2
        rss = yres @ yres - np.where(okA, (A.T @ yres) ** 2 / np.where(okA, aa, 1), 0) - b ** 2 * bb
        se = np.sqrt(np.maximum(rss, 1e-12) / df / np.where(ok, bb, 1))
        tt = np.where(ok, b / se, 0)
        lp = -(stats.norm.logsf(np.abs(tt)) + np.log(2)) / np.log(10)
        # markers linked to the peak: x_p * x_j is nearly constant, not a real test
        lp[(chrom[cols] == chrom[p]) & (np.abs(cM[cols] - cM[p]) < 10)] = 0
        logp[cols] = lp
    return logp


def inverse_normal(v):
    r = stats.rankdata(v)
    return stats.norm.ppf((r - 0.5) / len(v))


scan_rows, peak_rows, locus_rows = [], [], []
for t in TRAITS:
    t1 = time.time()
    raw = meta.loc[fit_lines, t].to_numpy(float)
    y = inverse_normal(raw) if t == 'rca_pct' else raw   # aerenchyma is strongly skewed
    # genomic heritability (null model, full kinship): for "% of genetic variance"
    h2 = reml_h(s_all, U_all.T @ y, U_all.T @ np.ones((n, 1)))
    lp0, b0 = scan(y, [])
    scan_rows.append(pd.DataFrame({'trait': t, 'snp': snps, 'chr': chrom, 'pos_cM': cM, 'logp': lp0, 'effect': b0}))
    chosen = []
    lp = lp0
    while len(chosen) < MAX_PEAKS and lp.max() > THR:
        j = int(np.argmax(lp))
        chosen.append((j, float(lp[j]), 'main'))
        lp, _ = scan(y, [c for c, _, _ in chosen])
    main_idx = [c for c, _, _ in chosen]
    # interactions with each selected peak
    found_inter = []
    if main_idx:
        thr_i = -np.log10(ALPHA / (m * len(main_idx)))
        for p in main_idx:
            li = interaction_scan(y, main_idx, p)
            if li.max() > thr_i:
                j = int(np.argmax(li))
                found_inter.append((p, j, float(li.max())))
    # final multi-locus model (OLS on all terms) for effects and variance explained
    terms = [X[:, j] for j in main_idx] + [X[:, p] * X[:, j] for p, j, _ in found_inter]
    extra = [j for p, j, _ in found_inter if j not in main_idx]
    terms += [X[:, j] for j in extra]
    names = ([('main', j, None) for j in main_idx] + [('interaction', j, p) for p, j, _ in found_inter]
             + [('partner main', j, None) for j in extra])
    if terms:
        A = np.c_[np.ones(n), np.column_stack(terms)]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        vy = y.var(ddof=1)
        # Loci: main-effect peaks on the same chromosome within 2 cM are one locus (the
        # scan can pick a second marker next to a QTL that sits between markers); its
        # variance is that of the summed parts. Each interaction is its own locus.
        parts = A[:, 1:] * coef[1:]
        mains = sorted([k for k, nm in enumerate(names) if nm[0] != 'interaction'],
                       key=lambda k: (chrom[names[k][1]], cM[names[k][1]]))
        groups = []
        for k in mains:
            j = names[k][1]
            if groups and chrom[names[groups[-1][-1]][1]] == chrom[j] and cM[j] - cM[names[groups[-1][-1]][1]] <= 2:
                groups[-1].append(k)
            else:
                groups.append([k])
        groups += [[k] for k, nm in enumerate(names) if nm[0] == 'interaction']
        for grp in groups:
            lead = max(grp, key=lambda k: next((l for c, l, _ in chosen if c == names[k][1]), -1))
            kind, j, p = names[lead]
            entry = (max([l for c, l, _ in chosen if c in [names[k][1] for k in grp]], default=np.nan)
                     if kind != 'interaction' else next(l for pp, jj, l in found_inter if jj == j and pp == p))
            locus_rows.append({'trait': t, 'kind': 'interaction' if kind == 'interaction' else 'main',
                               'lead_snp': snps[j], 'chr': chrom[j], 'pos_cM': cM[j],
                               'partner_snp': snps[p] if p is not None else '',
                               'markers': ';'.join(snps[names[k][1]] for k in grp), 'logp': entry,
                               'pct_phenotypic_var': 100 * parts[:, grp].sum(1).var(ddof=1) / vy, 'h2_genomic': h2})
        for k, (kind, j, p) in enumerate(names):
            part = A[:, 1 + k] * coef[1 + k]
            logp_entry = (next((l for c, l, _ in chosen if c == j), np.nan) if kind == 'main'
                          else next((l for pp, jj, l in found_inter if jj == j and pp == p), np.nan))
            peak_rows.append({'trait': t, 'kind': kind, 'snp': snps[j], 'chr': chrom[j], 'pos_cM': cM[j],
                              'partner_snp': snps[p] if p is not None else '', 'logp_at_entry': logp_entry,
                              'effect_per_dose': coef[1 + k] if t != 'rca_pct' else np.nan,
                              'effect_sd_units': coef[1 + k] / np.sqrt(vy),
                              'pct_phenotypic_var': 100 * part.var(ddof=1) / vy,
                              'pct_genetic_var_est': 100 * part.var(ddof=1) / vy / max(h2, 1e-3),
                              'h2_genomic': h2})
    print(f'{t}: h2 {h2:.2f} | {len(main_idx)} peaks, {len(found_inter)} interactions | {time.time() - t1:.0f}s',
          flush=True)

pd.concat(scan_rows).to_csv(GW / 'gwas_scan.csv', index=False)
P = pd.DataFrame(peak_rows)
P.to_csv(GW / 'gwas_peaks.csv', index=False)
L = pd.DataFrame(locus_rows)
L.to_csv(GW / 'gwas_loci.csv', index=False)
pd.set_option('display.width', 220)
print(f'threshold -log10 p = {THR:.2f}')
print(L.round(2).to_string(index=False))
