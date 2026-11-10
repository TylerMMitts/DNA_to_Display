# SNP encoding 'hybrid' for WiDiv-format datasets: genomic prediction of the
# image, plus the gated windows of option F.
#
# Why. On the RIL the windows model's tokens hold the genetic signal (r about
# 0.92 with the true genetic values), but its generated roots carry only 0.80,
# and 0.63 for aerenchyma. The loss is in the UNet: with one image per line the
# genotype tokens act as a name tag, and the UNet learns each training image's
# non-genetic noise along with its genetics. Generated aerenchyma on trained
# lines correlated 0.70-0.75 with the noise in their own image, about 0 on
# held-out lines (results/ril_v5/checkpoint_selection/README.md).
#
# What. Two sets of tokens go to the UNet:
#   1. summary tokens, from a GBLUP-style prediction of each line's image. Every
#      training image is described by a rotation-invariant tissue-layout vector
#      (tissue_layout_summary), reduced by PCA. A kernel ridge on standardised
#      dosages predicts those PCs from the genotype. Training lines get
#      out-of-fold predictions, from a ridge fitted without them, so their own
#      image's noise is never in their conditioning, and they look exactly like
#      unseen lines to the model.
#   2. the windows model's tokens (widiv_window_encoding.py), for what a
#      linear prediction cannot carry, such as the RIL's epistatic metaxylem
#      diameter. In training they are replaced by learned null tokens for a
#      share of samples (window_token_dropout), so the UNet takes everything
#      additive from the summary tokens instead of from the name tag.
#
# Only training images and genotypes are used; no trait, genetic value or QTL
# is read. Checked before training on the RIL (option-1 checks in
# results/ril_v5/checkpoint_selection/): the layout vector predicts the measured
# traits of its own image at r 0.73-1.00, and the out-of-fold prediction carries
# r 0.90-0.94 with the true genetic values for aerenchyma, sclerenchyma, cortex
# width and stele (the windows model draws 0.63-0.87), but 0.41 for the
# epistatic metaxylem diameter, which the window tokens carry.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from widiv_window_encoding import DosageWindowLayout, GatedDosageWindowEncoder

# Rendered tissue colours (RGB), nearest-colour classes: epidermis, exodermis,
# sclerenchyma, cortex, endodermis, pericycle, stele, phloem, protoxylem, white.
# Medians of 40 RIL v5 image + mask pairs (widiv_traits.build_palette); WiDiv's
# renderer gives the same colours to within a few levels.
TISSUE_PALETTE = [[179, 191, 154], [162, 180, 129], [93, 86, 80], [208, 216, 195], [96, 144, 170],
                  [168, 149, 175], [195, 184, 162], [165, 154, 135], [175, 100, 95], [255, 255, 255]]
WHITE, SCLERENCHYMA, STELE_CLASSES = 9, 2, (5, 6, 7, 8)
N_RINGS, N_DEPTH, N_FREQ = 48, 24, 24


class _PolarGrid:
    # Polar sampling grid around the image centre (roots are centred), cached per device and size.
    _cache = {}

    @classmethod
    def get(cls, size, device):
        key = (size, str(device))
        if key not in cls._cache:
            n_r = n_a = size
            half = (size - 1) / 2
            rad = torch.arange(n_r, dtype=torch.float32) * 0.5
            ang = torch.arange(n_a, dtype=torch.float32) * 2 * math.pi / n_a
            xs = (half + rad[:, None] * torch.cos(ang)[None]) / half - 1
            ys = (half - rad[:, None] * torch.sin(ang)[None]) / half - 1
            cls._cache[key] = torch.stack([xs, ys], -1)[None].to(device)
        return cls._cache[key]


# Rotation-invariant description of one root's tissue layout, from a uint8 RGB
# image (H, W, 3) as a tensor. Every pixel is given its nearest tissue colour,
# the class map is resampled along rays from the centre, and described as:
#   root and stele radius; the share of each class at 48 radii relative to the
#   root edge; the share of each class in 24 depth steps inward from the edge
#   (1 px each); the number and mean length of white gaps around each relative
#   ring (vessels, lacunae); and the angular spectrum of the white and
#   sclerenchyma rings (frequencies 1-24).
@torch.no_grad()
def tissue_layout_summary(rgb):
    device = rgb.device
    palette = torch.tensor(TISSUE_PALETTE, dtype=torch.float32, device=device)
    n_cls = len(palette)
    size = rgb.shape[0]
    cls = ((rgb.float()[..., None, :] - palette) ** 2).sum(-1).argmin(-1)
    onehot = F.one_hot(cls, n_cls).permute(2, 0, 1)[None].float()
    grid = _PolarGrid.get(size, device)
    polar = F.grid_sample(onehot, grid, mode='nearest', align_corners=True)[0].argmax(0)    # (radii, angles)
    n_r, n_a = polar.shape
    idx = torch.arange(n_r, device=device)[:, None].expand(n_r, n_a)
    edge = torch.where(polar != WHITE, idx, torch.zeros_like(idx)).amax(0).float()
    r_root = edge.median()
    stele = torch.isin(polar, torch.tensor(STELE_CLASSES, device=device))
    r_stele = torch.where(stele, idx, torch.zeros_like(idx)).amax(0).float().median()
    rel = (torch.arange(N_RINGS, device=device).float() + 0.5) / N_RINGS * 1.02
    ring = polar[(rel * r_root).round().long().clamp(0, n_r - 1)]
    share_rel = F.one_hot(ring, n_cls).float().mean(1)
    depth = torch.arange(N_DEPTH, device=device)[:, None] * 2
    band = polar.gather(0, (edge[None] - depth).clamp(0, n_r - 1).long())
    share_depth = F.one_hot(band, n_cls).float().mean(1)
    white = (ring == WHITE).float()
    starts = (white * (1 - white.roll(1, 1))).sum(1)
    gap_length = white.sum(1) / starts.clamp(min=1) * (rel * r_root * 0.5 * 2 * math.pi / n_a)
    spec_white = torch.log1p(torch.fft.rfft(white, dim=1).abs()[:, 1:N_FREQ + 1])
    spec_scl = torch.log1p(torch.fft.rfft((ring == SCLERENCHYMA).float(), dim=1).abs()[:, 1:N_FREQ + 1])
    return torch.cat([torch.stack([r_root, r_stele]) / 2, share_rel.flatten(), share_depth.flatten(),
                      starts, gap_length, spec_white.flatten(), spec_scl.flatten()]).cpu().numpy()


def _loo_lambdas(K, Y, grid):
    # Ridge penalty per target column by leave-one-out error, on a centred kernel.
    ev, V = np.linalg.eigh(K)
    ev = np.clip(ev, 0, None)
    base = ev.mean()
    Yc = Y - Y.mean(0)
    VtY = V.T @ Yc
    best, best_err = np.zeros(Y.shape[1]), np.full(Y.shape[1], np.inf)
    for g in grid:
        f = ev / (ev + g * base)
        H = V @ (f[:, None] * VtY)
        h = (V ** 2 * f).sum(1) + 1 / len(Y)
        err = (((Yc - H) / (1 - h)[:, None]) ** 2).mean(0)
        better = err < best_err
        best[better], best_err[better] = g * base, err[better]
    return best


def _centre(K_fit, K_new):
    m = K_fit.mean(0)
    return (K_fit - m[None] - m[:, None] + m.mean(),
            K_new - K_new.mean(1, keepdims=True) - m[None] + m.mean())


def _kernel_ridge(K_fit, K_new, Y, lambdas):
    Kc, Kn = _centre(K_fit, K_new)
    ev, V = np.linalg.eigh(Kc)
    ev = np.clip(ev, 0, None)
    mu = Y.mean(0)
    VtY = V.T @ (Y - mu)
    out = np.empty((len(K_new), Y.shape[1]))
    KnV = Kn @ V
    for lam in np.unique(lambdas):
        cols = lambdas == lam
        out[:, cols] = KnV @ (VtY[:, cols] / (ev + lam)[:, None]) + mu[cols]
    return out


class HybridLayout:
    # The fitted, non-learned part of the hybrid encoding; plays the projector's
    # role in train_widiv.py. Its input row for the encoder is the dosage row
    # followed by the standardised predicted summary PCs. Everything about the
    # windows is the wrapped DosageWindowLayout's.

    kind = 'widiv_dosage_hybrid'

    def __init__(self, window_size=200, n_pcs=32, summary_pcs=256, folds=10):
        self.layout = DosageWindowLayout(window_size, n_pcs)
        self.summary_pcs = int(summary_pcs)
        self.folds = int(folds)
        self.summary_mean = self.summary_scale = self.summary_components = None
        self.weights = self.intercept = None            # summary PCs = intercept + z @ weights
        self.pred_mean = self.pred_scale = None          # standardisation of the encoder input
        self.oof = {}                                    # fitting line -> out-of-fold prediction

    def __getattr__(self, name):
        # n_windows, n_pcs, pca, gather_index, window_bounds, ... come from the window layout.
        if name == 'layout':
            raise AttributeError(name)
        return getattr(self.layout, name)

    # dosage: every line's dosage row; fit_rows: rows of the training lines;
    # chromosomes, positions: per SNP; summaries: {line: summary vector} for the
    # training lines that have images (averaged over a line's images).
    # orthogonalize_columns: QTL columns to hide from both the windows and the
    # genomic prediction (DosageWindowLayout.fit).
    def fit(self, dosage, fit_rows, snp_names, chromosomes, positions, line_names, summaries,
            verbose=True, orthogonalize_columns=None):
        self.layout.fit(dosage[fit_rows], snp_names, chromosomes, positions, verbose=verbose,
                        orthogonalize_columns=orthogonalize_columns)
        lines = [line_names[r] for r in fit_rows if line_names[r] in summaries]
        rows = [r for r in fit_rows if line_names[r] in summaries]
        S = np.stack([summaries[g] for g in lines]).astype(np.float64)
        self.summary_mean = S.mean(0)
        self.summary_scale = S.std(0) + 1e-6
        Sz = (S - self.summary_mean) / self.summary_scale
        _, sv, vt = np.linalg.svd(Sz, full_matrices=False)
        k = min(self.summary_pcs, len(sv))
        self.summary_pcs = k
        self.summary_components = vt[:k].astype(np.float32)
        Y = Sz @ vt[:k].T
        variance = float((sv[:k] ** 2).sum() / (sv ** 2).sum())

        Z = self.layout.standardized(dosage[rows])
        p = Z.shape[1]
        K = (Z.astype(np.float64) @ Z.T.astype(np.float64)) / p
        Kc, _ = _centre(K, K)
        lambdas = _loo_lambdas(Kc, Y, 10.0 ** np.arange(-3, 3.01, 0.5))
        # Full fit, as primal weights so any genotype (edited ones included) can be predicted.
        ev, V = np.linalg.eigh(Kc)
        ev = np.clip(ev, 0, None)
        mu = Y.mean(0)
        alpha = np.empty_like(Y)
        VtY = V.T @ (Y - mu)
        for lam in np.unique(lambdas):
            cols = lambdas == lam
            alpha[:, cols] = V @ (VtY[:, cols] / (ev + lam)[:, None])
        # centred kernel = Zc Zc' / p with Zc = Z - its column means over these lines
        zbar = Z.mean(0)
        self.weights = ((Z - zbar).T.astype(np.float64) @ alpha / p).astype(np.float32)
        self.intercept = (mu - zbar.astype(np.float64) @ self.weights).astype(np.float32)
        # Out-of-fold predictions for the fitting lines.
        fold = np.random.default_rng(0).permutation(np.arange(len(rows)) % self.folds)
        oof = np.empty_like(Y)
        for f in range(self.folds):
            a, b = fold != f, fold == f
            oof[b] = _kernel_ridge(K[np.ix_(a, a)], K[np.ix_(b, a)], Y[a], lambdas)
        self.oof = {g: oof[i].astype(np.float32) for i, g in enumerate(lines)}
        self.pred_mean = oof.mean(0).astype(np.float32)
        self.pred_scale = (oof.std(0) + 1e-6).astype(np.float32)
        if verbose:
            r = [np.corrcoef(oof[:, j], Y[:, j])[0, 1] for j in range(min(8, k))]
            print(f"  hybrid: image summaries of {len(lines)} training lines, {k} summary PCs "
                  f"({variance:.0%} of their variance); out-of-fold genomic prediction r for "
                  f"PCs 1-{len(r)}: {np.round(r, 2).tolist()}")
        return self

    def predict_summary(self, dosage):
        return self.layout.standardized(dosage) @ self.weights + self.intercept

    def _row(self, dosage, summary):
        s = (summary - self.pred_mean) / self.pred_scale
        return np.concatenate([np.atleast_2d(dosage).astype(np.float16), s.astype(np.float16)], axis=1)

    # Encoder input for any genotypes, as for a line the ridge was not fitted on.
    def transform(self, dosage):
        return self._row(dosage, self.predict_summary(dosage))

    # Encoder input for named lines: the training lines' own out-of-fold
    # predictions, so a model never sees a prediction built from its own image.
    def project_lines(self, dosage, line_names):
        summary = self.predict_summary(dosage)
        for i, g in enumerate(line_names):
            if str(g) in self.oof:
                summary[i] = self.oof[str(g)]
        return self._row(dosage, summary)

    @property
    def output_dim(self):
        return self.layout.n_snps + self.summary_pcs

    def state_dict(self):
        return {'kind': self.kind, 'layout': self.layout.state_dict(), 'summary_pcs': self.summary_pcs,
                'folds': self.folds, 'summary_mean': self.summary_mean, 'summary_scale': self.summary_scale,
                'summary_components': self.summary_components, 'weights': self.weights.astype(np.float16),
                'intercept': self.intercept, 'pred_mean': self.pred_mean, 'pred_scale': self.pred_scale,
                'oof': self.oof, 'palette': TISSUE_PALETTE}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.layout = DosageWindowLayout.from_state_dict(state['layout'])
        for k in ('summary_pcs', 'folds', 'summary_mean', 'summary_scale', 'summary_components',
                  'intercept', 'pred_mean', 'pred_scale', 'oof'):
            setattr(obj, k, state[k])
        obj.weights = np.asarray(state['weights'], dtype=np.float32)
        return obj


class HybridDosageEncoder(GatedDosageWindowEncoder):
    # (dosage row, predicted summary) -> summary tokens followed by window tokens,
    # (B, summary_tokens + num_tokens, embedding_dim). The UNet's cross-attention
    # takes any number of tokens.

    def __init__(self, n_snps, n_pcs, n_windows, window_width, summary_dim, summary_tokens=8,
                 window_token_dropout=0.5, **kwargs):
        super().__init__(n_snps, n_pcs, n_windows, window_width, **kwargs)
        self.summary_dim, self.summary_tokens = int(summary_dim), int(summary_tokens)
        self.window_token_dropout = float(window_token_dropout)
        h, e = self.hidden_dim, self.embedding_dim
        self.summary_net = nn.Sequential(
            nn.Linear(self.summary_dim, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(self.dropout),
            nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.SiLU(), nn.Dropout(self.dropout),
            nn.Linear(h // 2, self.summary_tokens * e),
        )
        self.summary_positions = nn.Parameter(torch.randn(1, self.summary_tokens, e) * 0.02)
        # What the UNet sees in place of dropped window tokens.
        self.null_window_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, e))
        # True: summary tokens only, window tokens always null (for testing how
        # much the summary carries alone). Not saved.
        self.drop_windows = False

    @classmethod
    def for_layout(cls, layout, **kwargs):
        index = layout.gather_index()
        return cls(layout.n_snps, layout.n_pcs, layout.n_windows, index.shape[1],
                   summary_dim=layout.summary_pcs, **kwargs)

    def forward(self, x):
        dosage, summary = x[:, :self.n_snps], x[:, self.n_snps:].float()
        s_tok = self.summary_net(summary).view(-1, self.summary_tokens, self.embedding_dim) + self.summary_positions
        w_tok = super().forward(dosage)
        if self.drop_windows:
            keep = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        elif self.training and self.window_token_dropout > 0:
            keep = torch.rand(len(x), device=x.device) >= self.window_token_dropout
        else:
            keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
        w_tok = torch.where(keep[:, None, None], w_tok, self.null_window_tokens.expand_as(w_tok))
        return torch.cat([s_tok, w_tok], dim=1)

    def config(self):
        return {**super().config(), 'summary_dim': self.summary_dim, 'summary_tokens': self.summary_tokens,
                'window_token_dropout': self.window_token_dropout}
