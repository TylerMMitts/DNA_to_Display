# How much genetic information a WiDiv model's SNP encoder keeps, and what kind
# it loses. Works on any WiDiv diffusion checkpoint, PCA (option A) or windows
# (option F), and needs no GPU and no training: every probe below is a ridge
# regression or a projection, solved in closed form on the CPU.
#
#   1. PCA retention. The eigen-decomposition of the SNP covariance matrix on the
#      lines the encoder was fitted on (through the lines x lines Gram matrix,
#      which has the same non-zero eigenvalues as the 90,790 x 90,790 covariance).
#      For every SNP: the share of its variance the kept components hold, on the
#      fitting lines and on test lines, and at other numbers of components. The
#      designed QTL are flagged, so it shows whether compression throws them away.
#   2. Decoding, as LiteVAE does for images: the encoder's representation of a
#      line (its PCA/window input, the MLP's 256-unit layer, and the 2,048
#      token values the UNet sees) is decoded back to all 90,790 dosages by ridge
#      regression fitted on training lines, then scored on test lines: R^2 per SNP,
#      genotype calls right, and how often minor-allele carriers decode as carriers.
#   3. Trait regression: ridge from each representation to every trait's true
#      genetic value (and the observed line means), scored on test lines, next to
#      ridge on all SNPs (GBLUP-like) and on the trait's QTL alone.
#   4. Bias summary: retention by minor-allele frequency, local LD and population
#      structure; QTL against background SNPs of the same frequency. For a PCA
#      encoder also the score shift: how much less spread a line the PCA was not
#      fitted on has on each component than the lines it was fitted on.
#   5. Alternative encodings fitted the same way (more PCs, LD-weighted PCA, local
#      PCA in genome windows, global + local, training scores matched to held-out
#      spread), scored by 2 and 3.
#
# The ground truth (QTL, genetic values, trait means) is used here only to score
# the encoders; no model ever trains on it.
#
#     python code/widiv/analyze_snp_encoding.py                     # the default checkpoints
#     python code/widiv/analyze_snp_encoding.py path/to/ckpt.pt ... # any others
#     python code/widiv/analyze_snp_encoding.py --skip-alternatives
#
# Writes results/widiv/snp_encoding_analysis/<checkpoint>/ and, for part 5,
# results/widiv/snp_encoding_analysis/alternatives/.

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('WIDIV_DATASET', 'widiv')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from widiv_paths import (WIDIV_GENETIC_VALUES, WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPE_MEANS,
                         WIDIV_GENOTYPES, WIDIV_QTL, WIDIV_RESULTS_DIR, WIDIV_SNP_MAP, WIDIV_SPLIT)
from widiv_data import load_split
from widiv_snp_encoding import DosagePCAProjector, heldout_score_scale, load_widiv_genotypes
from widiv_window_encoding import DosageWindowLayout, load_widiv_encoder

CHECKPOINT_DIR = Path(os.environ.get('WIDIV_CHECKPOINT_DIR', r'D:\Downloads\d2dimages'))
DEFAULT_CHECKPOINTS = [
    'diffusion_widiv_medium_random_vaev3_epoch_400.pt',
    'diffusion_widiv_medium_random_windows200_pc32_sp1e-06_vaev3_epoch_500.pt',
]
OUT_DIR = WIDIV_RESULTS_DIR / 'snp_encoding_analysis'
TRAITS = ['ccfn', 'stele_diam_um', 'mx_n', 'mx_diam_um', 'cort_cell_um', 'rca_pct', 'mcs_layers']
RETENTION_KS = [10, 50, 100, 200, 369, 400, 545, 642]
CHUNK = 4000                 # SNP columns per block; keeps memory under about 1 GB
LD_SPAN = 100                # local LD: SNPs within this many positions either side
MAF_BINS = [0.05, 0.075, 0.1, 0.15, 0.25, 0.35, 0.5]
LAMBDA_GRID = 10.0 ** np.arange(-3, 3.01, 0.5)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------- data

class Data:
    # Genotypes standardised by the fitting lines (training lines minus the
    # validation lines, as the encoders were fitted), plus everything used to score.

    def __init__(self, fit_lines, train_lines, test_lines):
        self.lines, self.snps, self.dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
        row = {g: i for i, g in enumerate(self.lines)}
        self.fit = np.array(sorted(row[g] for g in fit_lines))
        self.tr = np.array(sorted(row[g] for g in train_lines))      # every training-side line
        self.te = np.array(sorted(row[g] for g in test_lines))
        self.val = np.setdiff1d(self.tr, self.fit)                    # training side, not in the PCA fit
        snp_map = pd.read_csv(WIDIV_SNP_MAP).set_index('snp').loc[self.snps]
        self.chrom, self.pos = snp_map['chr'].to_numpy(), snp_map['pos'].to_numpy()
        self.p = len(self.snps)

        x = self.dosage[self.fit].astype(np.float32)
        self.mean = x.mean(0)
        sd = x.std(0, ddof=1)
        sd[sd == 0] = 1.0
        self.scale = sd
        del x
        self.Z = np.empty((len(self.lines), self.p), np.float32)
        for c in range(0, self.p, CHUNK):
            s = slice(c, c + CHUNK)
            self.Z[:, s] = (self.dosage[:, s] - self.mean[s]) / self.scale[s]
        alt = self.dosage[self.tr].astype(np.float32).mean(0) / 2
        self.alt_freq = alt
        self.maf = np.minimum(alt, 1 - alt)

        self.qtl = pd.read_csv(WIDIV_QTL) if WIDIV_QTL is not None else None
        col = {s: i for i, s in enumerate(self.snps)}
        self.is_qtl = np.zeros(self.p, bool)
        self.qtl_label = np.full(self.p, '', dtype=object)
        if self.qtl is not None:
            self.qtl['column'] = self.qtl.snp.map(col)
            for _, q in self.qtl.iterrows():
                self.is_qtl[q.column] = True
                self.qtl_label[q.column] = (self.qtl_label[q.column] + ';' if self.qtl_label[q.column] else '') \
                    + f'{q.trait}:{q.role}'

        self.targets = {}
        if WIDIV_GENETIC_VALUES is not None:
            g = pd.read_csv(WIDIV_GENETIC_VALUES).set_index('genotype').reindex(self.lines)
            for t in TRAITS:
                self.targets[f'g_{t}'] = g[f'g_{t}'].to_numpy(float)
        means = pd.read_csv(WIDIV_GENOTYPE_MEANS).set_index('genotype').reindex(self.lines)
        for t in TRAITS:
            if t in means:
                self.targets[f'mean_{t}'] = means[t].to_numpy(float)

        # Kernels on all SNPs, computed once: fitting lines x all lines.
        log(f'{len(self.lines)} lines ({len(self.fit)} fitting, {len(self.tr)} training side, '
            f'{len(self.te)} test) x {self.p:,} SNPs, {int(self.is_qtl.sum())} QTL SNPs')
        self.K_all = np.zeros((len(self.lines), len(self.lines)))
        for c in range(0, self.p, CHUNK):
            zc = self.Z[:, c:c + CHUNK].astype(np.float64)
            self.K_all += zc @ zc.T
        self.ld_score, self.ld_max = self.local_ld()

    # Sum and max of bias-corrected r^2 with the SNPs within LD_SPAN positions
    # either side on the same chromosome (SNPs are stored in genome order).
    def local_ld(self):
        n = len(self.fit)
        score, top = np.ones(self.p, np.float32), np.zeros(self.p, np.float32)
        for c in np.unique(self.chrom):
            cols = np.flatnonzero(self.chrom == c)
            zc = self.Z[np.ix_(self.fit, cols)]
            for d in range(1, LD_SPAN + 1):
                if d >= len(cols):
                    break
                r = (zc[:, :-d] * zc[:, d:]).sum(0) / (n - 1)
                r2 = r ** 2 - (1 - r ** 2) / (n - 2)
                score[cols[:-d]] += r2
                score[cols[d:]] += r2
                top[cols[:-d]] = np.maximum(top[cols[:-d]], r ** 2)
                top[cols[d:]] = np.maximum(top[cols[d:]], r ** 2)
        return score, top


# ----------------------------------------------------------------- ridge

def center_kernel(K_tr, K_te):
    m = K_tr.mean(0)
    Kc = K_tr - m[None, :] - m[:, None] + m.mean()
    Kte = K_te - K_te.mean(1, keepdims=True) - m[None, :] + m.mean()
    return Kc, Kte


class DualRidge:
    # Kernel ridge with an intercept, through one eigen-decomposition, so every
    # lambda and every target column cost a few matrix products. Leave-one-out
    # error picks lambda without touching the test lines.

    def __init__(self, K_tr, K_te):
        Kc, self.Kte = center_kernel(K_tr, K_te)
        s, u = np.linalg.eigh(Kc)
        self.S = np.clip(s, 0, None)
        self.U = u
        self.KteU = (self.Kte @ u)
        self.base = max(float(np.trace(Kc)) / len(Kc), 1e-12)

    def loo_error(self, Y, lam):
        Yc = Y - Y.mean(0)
        f = self.S / (self.S + lam)
        HY = self.U @ (f[:, None] * (self.U.T @ Yc))
        h = (self.U ** 2 * f).sum(1) + 1.0 / len(Y)
        R = (Yc - HY) / (1 - h)[:, None]
        return (R ** 2).mean(0) / np.maximum(Yc.var(0), 1e-12)

    def choose(self, Y):
        errors = [np.nanmean(self.loo_error(Y, self.base * g)) for g in LAMBDA_GRID]
        return self.base * LAMBDA_GRID[int(np.argmin(errors))]

    def predict(self, Y, lam):
        mu = Y.mean(0)
        return (self.KteU / (self.S + lam)) @ (self.U.T @ (Y - mu)) + mu


def feature_kernels(X, data):
    X = np.asarray(X, np.float64)
    X = X - X[data.tr].mean(0)
    K = X @ X.T
    return K[np.ix_(data.tr, data.tr)], K[np.ix_(data.te, data.tr)]


# ----------------------------------------------------------------- 1. PCA retention

# Full spectrum of the SNP covariance on the fitting lines and, per SNP, the share
# of its fitting-line variance in the top k components for each k.
def covariance_spectrum(data):
    K = data.K_all[np.ix_(data.fit, data.fit)]
    lam, U = np.linalg.eigh(K)
    order = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[order], 0, None), U[:, order]
    keep = lam > lam[0] * 1e-10
    lam, U = lam[keep], U[:, keep]
    ks = [k for k in RETENTION_KS if k <= len(lam)] + [len(lam)]
    ret = {k: np.zeros(data.p, np.float32) for k in ks}
    total = np.zeros(data.p, np.float32)
    for c in range(0, data.p, CHUNK):
        s = slice(c, c + CHUNK)
        z = data.Z[data.fit, s]
        proj2 = (U.T @ z) ** 2                         # (components, SNPs): s_k^2 * C_kj^2
        cum = np.cumsum(proj2, axis=0)
        total[s] = (z ** 2).sum(0)
        for k in ks:
            ret[k][s] = cum[k - 1]
    for k in ks:
        ret[k] /= np.maximum(total, 1e-12)
    return lam, U, ret


# A fitted PCA (the checkpoint's) applied to test lines: share of each SNP's
# test-line variance inside the kept subspace. Components are orthonormal rows.
def subspace_retention(components, data, rows):
    C = np.asarray(components, np.float32)
    S = np.zeros((len(rows), C.shape[0]), np.float32)
    for c in range(0, data.p, CHUNK):
        S += data.Z[rows, c:c + CHUNK] @ C[:, c:c + CHUNK].T
    out = np.zeros(data.p, np.float32)
    for c in range(0, data.p, CHUNK):
        z = data.Z[rows, c:c + CHUNK]
        z = z - z.mean(0)
        proj = (S - S.mean(0)) @ C[:, c:c + CHUNK]
        out[c:c + CHUNK] = (proj ** 2).sum(0) / np.maximum((z ** 2).sum(0), 1e-12)
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------- encoders

def load_checkpoint(path):
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    keep = {k: ck[k] for k in ('snp_projector', 'snp_encoder_config', 'snp_encoder_state_dict', 'split',
                               'val_genotypes', 'test_genotypes', 'epoch', 'encoding') if k in ck}
    del ck
    return keep


# The three stages of the encoder for every line: what goes into the MLP, the
# MLP's last hidden layer (256 units) and the flattened tokens the UNet reads.
@torch.no_grad()
def encoder_representations(ck, data):
    encoder, projector = load_widiv_encoder(ck)
    projector.check_columns(data.snps)
    windows = isinstance(projector, DosageWindowLayout)
    out = {'input': [], 'bottleneck': [], 'tokens': []}
    for i in range(0, len(data.lines), 64):
        d = data.dosage[i:i + 64]
        if windows:
            x = torch.as_tensor(d.astype(np.int8))
            z = encoder.standardize(x)
            gated = encoder.window_features(z) * encoder.gates()[None, :, None]
            inp = torch.cat([z @ encoder.pca_components.T, gated.flatten(1)], dim=1)
            tokens = encoder(x)
        else:
            # project_lines: as the model saw each line (matters for matched-score models).
            inp = torch.as_tensor(projector.project_lines(d, data.lines[i:i + 64]), dtype=torch.float32)
            tokens = encoder(inp)
        out['input'].append(inp.numpy())
        out['bottleneck'].append(encoder.net[:-1](inp).numpy())
        out['tokens'].append(tokens.flatten(1).numpy())
    reps = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    info = {'kind': 'windows' if windows else 'pca', 'input_dim': int(reps['input'].shape[1]),
            'matched': bool(getattr(projector, 'match_heldout', False))}
    if windows:
        gates = encoder.gates().numpy()
        info.update(n_windows=int(len(gates)), open_windows=int((gates > 0).sum()), n_pcs=projector.n_pcs)
        info['window_of_column'] = projector.window_of_column()
        info['gates'] = gates
        info['pca_components'] = projector.pca.components_
    else:
        info['pca_components'] = projector.components_
        info['n_pcs'] = projector.output_dim
    return reps, info


# Spread of PC scores on lines the PCA was not fitted on (validation, test) over
# the spread on the fitting lines, by block of components; last row: median
# length of the whole score vector, test over fitting. 1 means no shift.
def score_shift(X, data):
    v_fit = X[data.fit].var(0)
    rows = []
    edges = [e for e in (0, 10, 50, 150, 369, 581, 732) if e < X.shape[1]] + [X.shape[1]]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows.append({'components': f'{lo + 1}-{hi}',
                     'validation_over_fitting': float(np.median(X[data.val, lo:hi].var(0) / v_fit[lo:hi])),
                     'test_over_fitting': float(np.median(X[data.te, lo:hi].var(0) / v_fit[lo:hi]))})
    norm = lambda rows_: np.median(np.linalg.norm(X[rows_], axis=1))
    rows.append({'components': 'score vector length (median)',
                 'validation_over_fitting': float(norm(data.val) / norm(data.fit)),
                 'test_over_fitting': float(norm(data.te) / norm(data.fit))})
    return pd.DataFrame(rows)


def effective_rank(X, energy=0.99):
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), energy) + 1)


# ----------------------------------------------------------------- 2. decoding

def decode_snps(X, data, label):
    # Ridge from a representation back to every standardised dosage, fitted on
    # training-side lines, scored on test lines.
    K_tr, K_te = feature_kernels(X, data) if X is not None else \
        (data.K_all[np.ix_(data.tr, data.tr)], data.K_all[np.ix_(data.te, data.tr)])
    ridge = DualRidge(K_tr, K_te)
    rng = np.random.default_rng(0)
    probe = np.unique(np.concatenate([rng.choice(data.p, 3000, replace=False), np.flatnonzero(data.is_qtl)]))
    lam = ridge.choose(data.Z[np.ix_(data.tr, probe)].astype(np.float64))
    r2 = np.full(data.p, np.nan, np.float32)
    acc, base_acc = np.zeros(data.p, np.float32), np.zeros(data.p, np.float32)
    recall = np.full(data.p, np.nan, np.float32)
    for c in range(0, data.p, CHUNK):
        s = slice(c, c + CHUNK)
        y_tr = data.Z[data.tr, s].astype(np.float64)
        y_te = data.Z[data.te, s]
        pred = ridge.predict(y_tr, lam).astype(np.float32)
        sst = ((y_te - y_te.mean(0)) ** 2).sum(0)
        ok = sst > 0
        r2[s][ok] = 1 - ((y_te - pred) ** 2).sum(0)[ok] / sst[ok]
        true = data.dosage[data.te, s]
        # Called as homozygotes: 3% of WiDiv calls are heterozygous, and rounding a
        # shrunk prediction near 1 would call a heterozygote that is almost never there.
        called = np.where(pred * data.scale[s] + data.mean[s] >= 1, 2, 0).astype(np.int8)
        acc[s] = (called == true).mean(0)
        train_d = data.dosage[data.tr, s]
        mode = np.argmax(np.stack([(train_d == v).sum(0) for v in (0, 1, 2)]), axis=0).astype(np.int8)
        base_acc[s] = (true == mode[None, :]).mean(0)
        # Minor-allele carriers: dosage >= 1 of the minor allele (alt if alt is rarer).
        alt_minor = data.alt_freq[s] < 0.5
        carrier_true = np.where(alt_minor[None, :], true >= 1, true <= 1)
        carrier_call = np.where(alt_minor[None, :], called >= 1, called <= 1)
        n_car = carrier_true.sum(0)
        has = n_car > 0
        recall[s][has] = (carrier_true & carrier_call).sum(0)[has] / n_car[has]
    log(f'  decoded SNPs from {label}: median test R^2 {np.nanmedian(r2):.3f}, '
        f'calls right {acc.mean():.3f} (majority guess {base_acc.mean():.3f})')
    return {'r2': r2, 'acc': acc, 'base_acc': base_acc, 'minor_recall': recall, 'lambda': lam}


# Nearest training lines in the representation (standardised features): a
# decoder that is not limited to linear structure.
def knn_decode_r2(X, data, k=10):
    X = ((X - X[data.tr].mean(0)) / (X[data.tr].std(0) + 1e-6)).astype(np.float64)
    a, b = X[data.te], X[data.tr]
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    nn_idx = np.argsort(d2, axis=1)[:, :k]
    r2 = np.full(data.p, np.nan, np.float32)
    acc = np.zeros(data.p, np.float32)
    for c in range(0, data.p, CHUNK):
        s = slice(c, c + CHUNK)
        ztr = data.Z[data.tr, s]
        pred = ztr[nn_idx].mean(1)
        y = data.Z[data.te, s]
        sst = ((y - y.mean(0)) ** 2).sum(0)
        ok = sst > 0
        r2[s][ok] = 1 - ((y - pred) ** 2).sum(0)[ok] / sst[ok]
        called = np.where(pred * data.scale[s] + data.mean[s] >= 1, 2, 0)
        acc[s] = (called == data.dosage[data.te, s]).mean(0)
    return r2, acc


# ----------------------------------------------------------------- 3. traits

def trait_regression(X, data, label, targets=None):
    if X is None:
        K_tr, K_te = data.K_all[np.ix_(data.tr, data.tr)], data.K_all[np.ix_(data.te, data.tr)]
    else:
        K_tr, K_te = feature_kernels(X, data)
    full = DualRidge(K_tr, K_te)
    rows = []
    for name, y in (targets or data.targets).items():
        ok_tr, ok_te = np.isfinite(y[data.tr]), np.isfinite(y[data.te])
        # A couple of lines have no images, so no observed mean: fit without them.
        ridge = full if ok_tr.all() else DualRidge(K_tr[np.ix_(ok_tr, ok_tr)], K_te[:, ok_tr])
        ytr = y[data.tr][ok_tr][:, None]
        lam = ridge.choose(ytr)
        pred = ridge.predict(ytr, lam)[:, 0]
        yt = y[data.te]
        r = np.corrcoef(pred[ok_te], yt[ok_te])[0, 1]
        slope = np.polyfit(pred[ok_te], yt[ok_te], 1)[0]
        rows.append({'representation': label, 'target': name, 'r': r, 'slope_truth_on_pred': slope,
                     'sd_ratio_pred_truth': pred[ok_te].std() / yt[ok_te].std()})
    return rows


def qtl_only_regression(data):
    rows = []
    if data.qtl is None:
        return rows
    for t in TRAITS:
        cols = data.qtl.loc[data.qtl.trait == t, 'column'].to_numpy()
        if len(cols) == 0:
            continue
        for kind in ('g', 'mean'):
            name = f'{kind}_{t}'
            if name in data.targets:
                rows += trait_regression(data.Z[:, cols], data, 'qtl_only', {name: data.targets[name]})
    return rows


# ----------------------------------------------------------------- 4. biases

def bias_tables(df, value_cols):
    out = []
    df = df.copy()
    df['maf_bin'] = pd.cut(df.maf, MAF_BINS, include_lowest=True).astype(str)
    df['ld_bin'] = pd.qcut(df.ld_score, 5, labels=['LD q1 (lowest)', 'LD q2', 'LD q3', 'LD q4', 'LD q5 (highest)']).astype(str)
    df['structure_bin'] = pd.cut(df.retained_top10, [-0.01, 0.05, 0.1, 0.2, 0.4, 1.0]).astype(str)
    for by in ('maf_bin', 'ld_bin', 'structure_bin'):
        g = df.groupby(by)[value_cols].median()
        g.insert(0, 'n_snps', df.groupby(by).size())
        g.insert(0, 'grouping', by)
        out.append(g.reset_index().rename(columns={by: 'group'}))
    # QTL against background SNPs in the same MAF bin.
    if df.is_qtl.any():
        rows = []
        for _, q in df[df.is_qtl].iterrows():
            bg = df[(~df.is_qtl) & (df.maf_bin == q.maf_bin)]
            row = {'grouping': 'qtl_vs_same_maf_background', 'group': q.snp, 'n_snps': len(bg)}
            for v in value_cols:
                row[v] = q[v] - bg[v].median()
            rows.append(row)
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------- figures

def plot_genome(df, col, title, path):
    fig, ax = plt.subplots(figsize=(14, 4))
    offset, ticks = 0, []
    for c in sorted(df.chr.unique()):
        part = df[df.chr == c]
        x = part.pos.to_numpy() + offset
        ax.scatter(x, part[col], s=1, color='0.55' if c % 2 else '0.75', rasterized=True)
        q = part[part.is_qtl]
        ax.scatter(q.pos + offset, q[col], s=28, color='#d62728', zorder=3)
        ticks.append(offset + part.pos.max() / 2)
        offset += part.pos.max() + 1e7
    ax.set_xticks(ticks, [str(c) for c in sorted(df.chr.unique())])
    ax.set_xlabel('chromosome')
    ax.set_ylabel(col)
    ax.set_title(title + '  (red: designed QTL)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_spectrum(lam, k_kept, path):
    ratio = lam / lam.sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].semilogy(np.arange(1, len(lam) + 1), ratio)
    axes[0].axvline(k_kept, color='#d62728', ls='--', label=f'kept: {k_kept}')
    axes[0].set_xlabel('component')
    axes[0].set_ylabel('share of variance')
    axes[0].legend()
    axes[1].plot(np.arange(1, len(lam) + 1), np.cumsum(ratio))
    axes[1].axvline(k_kept, color='#d62728', ls='--')
    axes[1].set_xlabel('components kept')
    axes[1].set_ylabel('cumulative share of variance')
    fig.suptitle('SNP covariance eigenvalues on the fitting lines')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_maf(df, cols, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.cut(df.maf, MAF_BINS, include_lowest=True)
    for c in cols:
        med = df.groupby(bins, observed=True)[c].median()
        ax.plot([str(b) for b in med.index], med.to_numpy(), marker='o', label=c)
    q = df[df.is_qtl]
    for c in cols[:1]:
        ax.scatter([str(b) for b in pd.cut(q.maf, MAF_BINS, include_lowest=True)], q[c], color='#d62728',
                   s=18, zorder=3, label=f'QTL ({c})')
    ax.set_xlabel('minor-allele frequency')
    ax.set_ylabel('median')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- per checkpoint

def analyze_checkpoint(path, data, spectrum):
    tag = Path(path).stem
    out = OUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    log(f'== {tag}')
    ck = load_checkpoint(path)
    reps, info = encoder_representations(ck, data)
    lam, U, ret = spectrum
    k = int(info['n_pcs'])

    df = pd.DataFrame({'snp': data.snps, 'chr': data.chrom, 'pos': data.pos, 'maf': data.maf,
                       'ld_score': data.ld_score, 'ld_max_r2': data.ld_max, 'is_qtl': data.is_qtl,
                       'qtl': data.qtl_label})
    for kk, v in ret.items():
        df[f'retained_top{kk}' if kk != len(lam) else 'retained_all_fit_components'] = v
    df['retained_fit'] = ret[k] if k in ret else covariance_spectrum_at(data, U, k)
    df['retained_test'] = subspace_retention(info['pca_components'], data, data.te)
    if info['kind'] == 'windows':
        df['window'] = info['window_of_column']
        df['window_open'] = info['gates'][info['window_of_column']] > 0

    # 2. decoding from each stage of the encoder
    summary = {'checkpoint': str(path), 'kind': info['kind'], 'input_dim': info['input_dim'], 'pcs': k,
               'components_on_fitting_lines': int(len(lam)),
               'variance_kept_by_pcs': float(lam[:k].sum() / lam.sum())}
    for name, X in reps.items():
        summary[f'effective_rank_99pct_{name}'] = effective_rank(X[data.tr])
        res = decode_snps(X, data, name)
        df[f'decode_r2_{name}'] = res['r2']
        df[f'decode_acc_{name}'] = res['acc']
        df[f'minor_recall_{name}'] = res['minor_recall']
        if name == 'tokens':
            df['decode_acc_majority_guess'] = res['base_acc']
    knn_r2, knn_acc = knn_decode_r2(reps['tokens'], data)
    df['decode_r2_tokens_knn'] = knn_r2
    df['decode_acc_tokens_knn'] = knn_acc
    if info['kind'] == 'windows':
        summary.update(open_windows=info['open_windows'], n_windows=info['n_windows'])
    else:
        summary['matched_scores'] = info['matched']
        shift = score_shift(reps['input'], data)
        shift.to_csv(out / 'score_shift.csv', index=False, float_format='%.4g')
        summary['test_over_fitting_score_norm'] = float(shift.iloc[-1].test_over_fitting)
    df.to_csv(out / 'snp_retention.csv.gz', index=False, float_format='%.4g')

    value_cols = ['retained_fit', 'retained_test', 'decode_r2_input', 'decode_r2_bottleneck',
                  'decode_r2_tokens', 'decode_r2_tokens_knn', 'decode_acc_tokens', 'minor_recall_tokens']
    q = df[df.is_qtl][['snp', 'qtl', 'chr', 'pos', 'maf', 'ld_score', 'ld_max_r2'] + value_cols
                      + [c for c in ('window', 'window_open') if c in df]].copy()
    for c in ('retained_fit', 'decode_r2_tokens'):
        q[f'{c}_percentile'] = [float((df[c] < v).mean()) for v in q[c]]
    q.to_csv(out / 'qtl_retention.csv', index=False, float_format='%.4g')
    bias_tables(df, value_cols).to_csv(out / 'bias_summary.csv', index=False, float_format='%.4g')

    # 3. traits
    rows = []
    for name, X in reps.items():
        rows += trait_regression(X, data, name)
    pd.DataFrame(rows).to_csv(out / 'trait_regression.csv', index=False, float_format='%.4g')

    for c in value_cols:
        summary[f'median_{c}'] = float(np.nanmedian(df[c]))
        summary[f'qtl_median_{c}'] = float(np.nanmedian(df.loc[df.is_qtl, c]))
    summary['decode_acc_majority_guess'] = float(df.decode_acc_majority_guess.mean())
    summary['decode_acc_tokens_mean'] = float(df.decode_acc_tokens.mean())
    summary['share_snps_under_half_retained_fit'] = float((df.retained_fit < 0.5).mean())
    summary['share_qtl_under_half_retained_fit'] = float((df.loc[df.is_qtl, 'retained_fit'] < 0.5).mean())
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))

    plot_genome(df, 'retained_fit', f'{tag}: share of each SNP\'s variance kept by the {k} PCs', out / 'retention_along_genome.png')
    plot_genome(df, 'decode_r2_tokens', f'{tag}: SNP decoded from tokens, test-line R^2', out / 'token_decoding_along_genome.png')
    plot_by_maf(df, ['retained_fit', 'retained_test', 'decode_r2_input', 'decode_r2_tokens'], out / 'retention_by_maf.png')
    plot_spectrum(lam, k, out / 'covariance_spectrum.png')
    log(f'  wrote {out}')
    return summary, pd.DataFrame(rows)


def covariance_spectrum_at(data, U, k):
    out = np.zeros(data.p, np.float32)
    for c in range(0, data.p, CHUNK):
        z = data.Z[data.fit, c:c + CHUNK]
        out[c:c + CHUNK] = ((U[:, :k].T @ z) ** 2).sum(0) / np.maximum((z ** 2).sum(0), 1e-12)
    return out


# ----------------------------------------------------------------- 5. alternatives

def global_pca_scores(data, lam, U, k, weights=None):
    if weights is None:
        Kaf = data.K_all[:, data.fit]
        return (Kaf @ U[:, :k]) / np.sqrt(lam[:k])
    w = np.sqrt(weights).astype(np.float32)
    K = np.zeros((len(data.lines), len(data.fit)))
    for c in range(0, data.p, CHUNK):
        zc = data.Z[:, c:c + CHUNK] * w[c:c + CHUNK]
        K += zc.astype(np.float64) @ zc[data.fit].astype(np.float64).T
    lw, Uw = np.linalg.eigh(K[data.fit])
    order = np.argsort(lw)[::-1][:k]
    return (K @ Uw[:, order]) / np.sqrt(np.clip(lw[order], 1e-12, None))


# PCA inside each genome window (window_size SNPs in genome order, never across a
# chromosome), top m components per window, fitted on the fitting lines. With
# n_global > 0 the windows are fitted on what the global PCs leave over.
def local_pca_scores(data, window_size, m, lam=None, U=None, n_global=0):
    parts = []
    G = C = None
    if n_global:
        G = global_pca_scores(data, lam, U, n_global)                     # (lines, n_global)
        parts.append(G)
    for c in np.unique(data.chrom):
        cols = np.flatnonzero(data.chrom == c)
        for a in range(0, len(cols), window_size):
            w = cols[a:a + window_size]
            z = data.Z[:, w]
            if n_global:
                # Loadings of the global PCs on this window's SNPs, from the fitting lines.
                Cw = np.linalg.lstsq(G[data.fit], z[data.fit], rcond=None)[0]
                z = z - (G @ Cw).astype(np.float32)
            _, _, vt = np.linalg.svd(z[data.fit], full_matrices=False)
            parts.append(z @ vt[:m].T)
    return np.concatenate(parts, axis=1).astype(np.float32)


# Global PCA scores with the fitting lines' scores shrunk to held-out spread
# (DosagePCAProjector(match_heldout=True)).
def matched_pca_scores(data, lam, U, k):
    X = global_pca_scores(data, lam, U, k)
    X[data.fit] *= heldout_score_scale(data.K_all[np.ix_(data.fit, data.fit)], k)
    return X


def run_alternatives(data, spectrum, k_current):
    lam, U, _ = spectrum
    out = OUT_DIR / 'alternatives'
    out.mkdir(parents=True, exist_ok=True)
    k95 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
    candidates = {
        f'global PCA {k_current}, training scores matched to held-out spread':
            lambda: matched_pca_scores(data, lam, U, k_current),
        f'global PCA, all {len(lam)} components, training scores matched to held-out spread (recommended)':
            lambda: matched_pca_scores(data, lam, U, len(lam)),
        '32 global PCs + local PCA, 200-SNP windows x 16': lambda: local_pca_scores(data, 200, 16, lam, U, 32),
        f'global PCA {k_current} (current option A)': lambda: global_pca_scores(data, lam, U, k_current),
        f'global PCA {k95} (95% variance)': lambda: global_pca_scores(data, lam, U, k95),
        f'LD-weighted global PCA {k_current}': lambda: global_pca_scores(
            data, lam, U, k_current, weights=1.0 / np.maximum(data.ld_score, 1.0)),
        'local PCA, 250-SNP windows x 1': lambda: local_pca_scores(data, 250, 1),
        'local PCA, 500-SNP windows x 2': lambda: local_pca_scores(data, 500, 2),
        '32 global PCs + local PCA, 250-SNP windows x 1': lambda: local_pca_scores(data, 250, 1, lam, U, 32),
        '32 global PCs + local PCA, 200-SNP windows x 4 (option F layout)': lambda: local_pca_scores(
            data, 200, 4, lam, U, 32),
        f'all {len(lam)} fitting-line PCs (ceiling for any linear encoding fitted on these lines)':
            lambda: global_pca_scores(data, lam, U, len(lam)),
    }
    summary, per_snp, traits = [], {}, []
    for name, make in candidates.items():
        log(f'alternative: {name}')
        X = make()
        res = decode_snps(X, data, name)
        per_snp[name] = res['r2']
        tr = trait_regression(X, data, name)
        traits += tr
        tdf = pd.DataFrame(tr)
        row = {'encoding': name, 'dims': X.shape[1],
               'median_decode_r2': float(np.nanmedian(res['r2'])),
               'mean_decode_r2': float(np.nanmean(res['r2'])),
               'qtl_median_decode_r2': float(np.nanmedian(res['r2'][data.is_qtl])),
               'rare_maf_under_0.1_median_decode_r2': float(np.nanmedian(res['r2'][data.maf < 0.1])),
               'calls_right': float(res['acc'].mean()),
               'minor_allele_recall': float(np.nanmean(res['minor_recall'])),
               'qtl_minor_allele_recall': float(np.nanmean(res['minor_recall'][data.is_qtl]))}
        for kind in ('g', 'mean'):
            part = tdf[tdf.target.str.startswith(kind + '_')]
            row[f'trait_r_{kind}_mean'] = float(part.r.mean())
            row[f'trait_slope_{kind}_mean'] = float(part.slope_truth_on_pred.mean())
            row[f'trait_sd_ratio_{kind}_mean'] = float(part.sd_ratio_pred_truth.mean())
        row['test_over_fitting_score_norm'] = float(
            np.median(np.linalg.norm(X[data.te], axis=1)) / np.median(np.linalg.norm(X[data.fit], axis=1)))
        summary.append(row)
        pd.DataFrame(summary).to_csv(out / 'encoding_comparison.csv', index=False, float_format='%.4g')
    traits += trait_regression(None, data, 'all SNPs (GBLUP-like ridge)')
    traits += qtl_only_regression(data)
    pd.DataFrame(traits).to_csv(out / 'trait_regression.csv', index=False, float_format='%.4g')
    snp = pd.DataFrame({'snp': data.snps, 'maf': data.maf, 'is_qtl': data.is_qtl, **per_snp})
    snp.to_csv(out / 'snp_decode_r2.csv.gz', index=False, float_format='%.4g')

    plot_decode_by_maf(per_snp, data.maf, out / 'decode_r2_by_maf.png')
    log(f'wrote {out}')
    return pd.DataFrame(summary)


def plot_decode_by_maf(per_snp, maf, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = pd.cut(pd.Series(maf), MAF_BINS, include_lowest=True)
    colors = plt.get_cmap('tab20').colors
    for i, name in enumerate(per_snp):
        med = pd.Series(np.asarray(per_snp[name])).groupby(bins, observed=True).median()
        ax.plot([str(b) for b in med.index], med.to_numpy(), marker='o', label=name,
                lw=2.4 if 'recommended' in name else 1.2, color=colors[(2 * i) % 20 if i < 10 else (2 * i + 1) % 20])
    ax.set_xlabel('minor-allele frequency')
    ax.set_ylabel('median test-line decoding R^2')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='*')
    ap.add_argument('--skip-alternatives', action='store_true')
    ap.add_argument('--only-alternatives', action='store_true')
    args = ap.parse_args()
    paths = [Path(p) for p in args.checkpoints] or [CHECKPOINT_DIR / c for c in DEFAULT_CHECKPOINTS]

    first = load_checkpoint(paths[0])
    split = load_split(WIDIV_SPLIT)
    col = first['split'] if str(first.get('split', '')).startswith('split_') else f"split_{first.get('split', 'random')}"
    test = set(first['test_genotypes'])
    val = set(first['val_genotypes'])
    train = set(split.index[split[col] == 'train'])
    data = Data(train - val, train, test)
    log('covariance spectrum and per-SNP retention on the fitting lines')
    spectrum = covariance_spectrum(data)
    k_current = int(DosagePCAProjector.from_state_dict(first['snp_projector']).output_dim
                    if first['snp_projector'].get('kind') != DosageWindowLayout.kind else 369)
    del first

    summaries = []
    if not args.only_alternatives:
        for p in paths:
            ck = load_checkpoint(p)
            if set(ck['test_genotypes']) != test or set(ck['val_genotypes']) != val:
                raise SystemExit(f'{p.name} uses different test or validation lines; run it on its own')
            del ck
            s, _ = analyze_checkpoint(p, data, spectrum)
            summaries.append(s)
        pd.DataFrame(summaries).to_csv(OUT_DIR / 'checkpoint_summary.csv', index=False, float_format='%.4g')
    if not args.skip_alternatives:
        print(run_alternatives(data, spectrum, k_current).to_string(index=False))


if __name__ == '__main__':
    main()
