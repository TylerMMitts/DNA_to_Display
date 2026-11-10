# SNP encoding for the WiDiv dataset: PCA on standardised 0/1/2 dosages.
#
# The MEMA encoders expand each locus into eight founder indicators before PCA.
# WiDiv has no founders: each SNP is a count of alt alleles (0, 1 or 2), which is
# already a number with a meaningful order, so there is nothing to one-hot.
# Each SNP is centred and scaled by its spread in the training lines (the usual
# genomic-relationship standardisation, so a rare SNP counts as much as a common
# one), then PCA reduces the 90,790 columns to a conditioning vector. The MLP
# that turns that vector into tokens is the same one the root and seed models use.
#
# Fitted on training lines only. The MEMA projector was fitted on all 200
# genotypes, held-out ones included; here the test lines never shape the basis,
# so a held-out score is held out of the encoder as well as the diffusion model.
#
# Profile of this data (results/widiv/genetics_profile/): 168 PCs hold 55% of
# the variance and 400 about 80%; ridge regression on 400 PCs predicts the true
# genetic values about as well as GBLUP under the random split (0.39-0.62), and
# 168 PCs fall 0.02-0.12 short. Rare designed QTL are poorly captured by any
# number of PCs (median R^2 0.06) - the reason for the window encoding planned next.

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from latent_diffusion.models.snp_encoding import OneHotSNPEncoder


# Reads genotypes_012.csv.gz into (line names, SNP names, int8 matrix), through a
# cache that is rebuilt whenever the CSV is newer than it.
def load_widiv_genotypes(csv_path, cache_path=None):
    csv_path = Path(csv_path)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
            c = np.load(cache_path, allow_pickle=False)
            return list(c['lines']), list(c['snps']), c['dosage']
    # One line at a time straight into int8: pandas' parser needs several GB for
    # 90,790 columns, which the 95 MB result does not justify.
    lines, rows = [], []
    with gzip.open(csv_path, 'rt') as f:
        snps = f.readline().strip().split(',')[1:]
        for text in f:
            name, rest = text.rstrip('\r\n').split(',', 1)
            fields = rest.split(',')
            if len(fields) != len(snps):
                raise SystemExit(f"{csv_path.name}: line {name} has {len(fields)} values, "
                                 f"expected {len(snps)}")
            # None are missing in this release; blanks or NA would be marked -1
            # so the projector imputes them with the training mean rather than
            # reading them as a dosage.
            values = np.array([v if v not in ('', 'NA', 'nan') else '-1' for v in fields])
            lines.append(name)
            rows.append(values.astype(np.float32).astype(np.int8))
    dosage = np.stack(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, lines=np.array(lines), snps=np.array(snps), dosage=dosage)
    return lines, snps, dosage


# Matrix columns of the designed QTL, and of every SNP within flank_kb of one on the
# same chromosome (flank_kb = 0: the QTL alone), for the leave-the-QTL-out runs.
def qtl_columns(snp_names, qtl_path, snp_map_path, flank_kb=0):
    qtl = pd.read_csv(qtl_path)
    column = {s: i for i, s in enumerate(snp_names)}
    qtl_cols = np.array(sorted({column[s] for s in qtl.snp}), dtype=np.int64)
    snp_map = pd.read_csv(snp_map_path).set_index('snp').loc[list(snp_names)]
    chrom, pos = snp_map['chr'].to_numpy(), snp_map['pos'].to_numpy()
    removed = np.zeros(len(snp_names), bool)
    for c in qtl_cols:
        removed |= (chrom == chrom[c]) & (np.abs(pos - pos[c]) <= flank_kb * 1000)
    return np.flatnonzero(removed), qtl_cols


# Held-out score shrinkage, per component, from the fitting lines alone.
#
# A line the PCA was fitted on scores far further out on the smaller components
# than a line it was not: on WiDiv, test lines have 16% of the fitting lines'
# score variance on PCs 151-369 and 30% on PCs 51-150, so a test line's score
# vector is about two-thirds as long. A model trained on fitting-line scores
# therefore meets test genotypes pulled towards the centre, which is part of why
# held-out generated traits vary too little (results/widiv/snp_encoding_analysis/).
# This estimates the ratio by refitting on 9/10 of the fitting lines and scoring
# the tenth, and returns sqrt(ratio) per component (capped at 1, smoothed over
# neighbouring components): multiplying the fitting lines' scores by it gives them
# the spread an unseen line has. Estimated without any test or validation line;
# it matched the test lines' actual ratio within 0.05 from PC 11 on.
def heldout_score_scale(gram, n_components, folds=10, seed=0, smooth=15):
    gram = np.asarray(gram, dtype=np.float64)
    n = len(gram)
    parts = np.array_split(np.random.default_rng(seed).permutation(n), folds)
    ratios = []
    for part in parts:
        inside = np.setdiff1d(np.arange(n), part)
        lam, u = np.linalg.eigh(gram[np.ix_(inside, inside)])
        order = np.argsort(lam)[::-1]
        lam, u = np.clip(lam[order], 1e-12, None), u[:, order]
        m = min(n_components, len(inside) - 1)
        held = gram[np.ix_(part, inside)] @ u[:, :m] / np.sqrt(lam[:m])
        own = u[:, :m] * np.sqrt(lam[:m])
        ratios.append(held.var(0) / np.maximum(own.var(0), 1e-12))
    m = min(len(r) for r in ratios)
    ratio = np.median(np.stack([r[:m] for r in ratios]), axis=0)
    half = smooth // 2
    ratio = np.array([np.median(ratio[max(0, i - half):i + half + 1]) for i in range(m)])
    ratio = np.concatenate([ratio, np.full(n_components - m, ratio[-1])])
    return np.sqrt(np.clip(ratio, 0, 1)).astype(np.float32)


class DosagePCAProjector:

    def __init__(self, n_components=None, target_variance=0.80, match_heldout=False):
        # n_components fixes the size; otherwise the smallest number reaching
        # target_variance of the training lines' total variance (1.0: every
        # component the fitting lines support, one fewer than their number).
        # match_heldout: shrink the fitting lines' own scores to the spread unseen
        # lines have (heldout_score_scale); see project_lines.
        self.n_components = n_components
        self.target_variance = target_variance
        self.match_heldout = bool(match_heldout)
        self.mean_ = None
        self.scale_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.snp_names = None
        self.fit_scale_ = None
        self.fitted_lines_ = None

    # Standardises dosages: missing (-1) becomes the training mean, so 0 after centring.
    def _standardize(self, dosage):
        x = np.asarray(dosage, dtype=np.float32)
        missing = x < 0
        z = (x - self.mean_) / self.scale_
        if missing.any():
            z[missing] = 0.0
        return z

    # exclude_columns: SNP columns the PCA never sees. Their loadings come out exactly
    # 0, so every later genotype matrix keeps its full width and changing one of these
    # SNPs changes nothing. orthogonalize_columns: SNPs whose linear signal is also
    # subtracted from the scores, by regression on the fitting lines; see transform.
    # line_names: the fitting lines, in the row order of dosage; needed with
    # match_heldout, so project_lines knows whose scores to shrink.
    def fit(self, dosage, snp_names=None, verbose=True, exclude_columns=None,
            orthogonalize_columns=None, line_names=None):
        if self.match_heldout and line_names is None:
            raise ValueError("match_heldout needs the fitting lines' names (line_names)")
        if self.match_heldout and orthogonalize_columns is not None:
            raise ValueError("match_heldout is not implemented together with orthogonalize_columns")
        x = np.asarray(dosage, dtype=np.float32)
        n, p = x.shape
        observed = x >= 0
        counts = np.maximum(observed.sum(0), 1)
        self.mean_ = (np.where(observed, x, 0).sum(0) / counts).astype(np.float32)
        var = (np.where(observed, x - self.mean_, 0) ** 2).sum(0) / np.maximum(counts - 1, 1)
        scale = np.sqrt(var).astype(np.float32)
        scale[scale == 0] = 1.0                      # monomorphic in training: contributes 0
        self.scale_ = scale
        self.snp_names = list(snp_names) if snp_names is not None else None

        # PCA from the lines x lines Gram matrix rather than an SVD of the
        # 90,790-column matrix, which runs out of memory on a laptop. If
        # Z Z' = U S^2 U', the loadings are U' Z / S.
        z = self._standardize(x)
        kept_out = z[:, orthogonalize_columns].copy() if orthogonalize_columns is not None else None
        if exclude_columns is not None:
            z[:, exclude_columns] = 0.0
        gram = z @ z.T
        lam, u = np.linalg.eigh(gram.astype(np.float64))
        order = np.argsort(lam)[::-1]
        lam, u = np.clip(lam[order], 0, None), u[:, order]
        usable = int((lam > lam[0] * 1e-10).sum())   # at most n - 1 after centring
        ratio = lam / lam.sum()
        k = self.n_components
        if k is None:
            k = int(np.searchsorted(np.cumsum(ratio), self.target_variance) + 1)
        k = min(k, usable)
        s = np.sqrt(lam[:k])
        # float32 throughout: float64 eigenvectors would promote the product to
        # a float64 lines x SNPs array, 0.5 GB more for no useful precision.
        self.components_ = (u[:, :k].T.astype(np.float32) @ z) / s[:k, None].astype(np.float32)
        self.excluded_columns_ = (np.asarray(exclude_columns, dtype=np.int64)
                                  if exclude_columns is not None else None)
        self.orthogonalize_columns_, self.orthogonalize_weights_ = None, None
        if kept_out is not None:
            self.orthogonalize_columns_ = np.asarray(orthogonalize_columns, dtype=np.int64)
            self.orthogonalize_weights_ = np.linalg.lstsq(
                kept_out, z @ self.components_.T, rcond=None)[0].astype(np.float32)
        del z
        self.explained_variance_ratio_ = ratio[:k]
        self.explained_variance_ = lam[:k] / max(n - 1, 1)
        self.fit_scale_, self.fitted_lines_ = None, None
        if self.match_heldout:
            self.fit_scale_ = heldout_score_scale(gram, k)
            self.fitted_lines_ = [str(g) for g in line_names]
        if verbose:
            print(f"  dosage PCA on {n} training lines x {p:,} SNPs: {k} components, "
                  f"{ratio[:k].sum():.1%} of variance")
            if self.match_heldout:
                print(f"  fitting lines' scores shrunk to held-out spread: factor "
                      f"{self.fit_scale_[:10].mean():.2f} on PCs 1-10, {self.fit_scale_[-50:].mean():.2f} on the last 50")
        return self

    # Scores for named lines. With match_heldout, the lines the PCA was fitted on
    # get their scores shrunk to the spread an unseen line has; every other line
    # (validation, test, edited genotypes) gets the plain projection. A model
    # trained this way must be conditioned through this method, not transform,
    # so its training lines look the way they did in training.
    def project_lines(self, dosage, line_names):
        scores = self.transform(dosage)
        if getattr(self, 'fit_scale_', None) is not None:
            fitted = set(self.fitted_lines_)
            rows = [i for i, g in enumerate(line_names) if str(g) in fitted]
            scores[rows] *= self.fit_scale_
        return scores

    def transform(self, dosage, chunk_size=128):
        if self.components_ is None:
            raise RuntimeError("DosagePCAProjector must be fitted before transform")
        dosage = np.atleast_2d(np.asarray(dosage))
        out = []
        for i in range(0, len(dosage), chunk_size):
            z = self._standardize(dosage[i:i + chunk_size])
            scores = z @ self.components_.T
            # Removing SNPs is not enough to hide them: relatives share them, so the
            # other SNPs' PCs still predict a designed QTL almost as well (R^2 0.37
            # with the 40 QTL gone, 0.32 with +/-5 Mb around each gone). This takes
            # out what the scores can linearly say about these SNPs.
            if getattr(self, 'orthogonalize_columns_', None) is not None:
                scores = scores - z[:, self.orthogonalize_columns_] @ self.orthogonalize_weights_
            out.append(scores)
        return np.concatenate(out, axis=0)

    @property
    def output_dim(self):
        return int(self.components_.shape[0])

    def state_dict(self):
        return {
            'kind': 'widiv_dosage_pca',
            'n_components': self.n_components,
            'target_variance': self.target_variance,
            'mean': self.mean_,
            'scale': self.scale_,
            # n_components x 90,790 float32: about 145 MB at 400 components.
            'components': self.components_,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'explained_variance': self.explained_variance_,
            # Saved so a checkpoint can never be applied to a matrix whose
            # columns are in a different order.
            'snp_names': self.snp_names,
            'excluded_columns': getattr(self, 'excluded_columns_', None),
            'orthogonalize_columns': getattr(self, 'orthogonalize_columns_', None),
            'orthogonalize_weights': getattr(self, 'orthogonalize_weights_', None),
            'match_heldout': self.match_heldout,
            'fit_scale': self.fit_scale_,
            'fitted_lines': self.fitted_lines_,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(n_components=state.get('n_components'),
                  target_variance=state.get('target_variance', 0.80),
                  match_heldout=state.get('match_heldout', False))
        obj.fit_scale_ = (np.asarray(state['fit_scale'], dtype=np.float32)
                          if state.get('fit_scale') is not None else None)
        obj.fitted_lines_ = state.get('fitted_lines')
        obj.mean_ = np.asarray(state['mean'], dtype=np.float32)
        obj.scale_ = np.asarray(state['scale'], dtype=np.float32)
        obj.components_ = np.asarray(state['components'], dtype=np.float32)
        obj.explained_variance_ratio_ = np.asarray(state['explained_variance_ratio'])
        obj.explained_variance_ = np.asarray(state['explained_variance'])
        obj.snp_names = state.get('snp_names')
        obj.excluded_columns_ = state.get('excluded_columns')
        obj.orthogonalize_columns_ = state.get('orthogonalize_columns')
        obj.orthogonalize_weights_ = state.get('orthogonalize_weights')
        return obj

    def check_columns(self, snp_names):
        if self.snp_names is not None and list(snp_names) != list(self.snp_names):
            raise SystemExit("this genotype matrix's SNP columns differ from the ones the "
                             "projector was fitted on - refusing to project it")


# The PCA scores go through the same MLP as the root and seed models: two hidden
# layers, then num_tokens tokens for the UNet's cross-attention. Subclassed only
# for the name; its weights and config are interchangeable with OneHotSNPEncoder.
class DosagePCAEncoder(OneHotSNPEncoder):
    pass


# The 'QTL only' encoding: the encoder sees nothing but the standardised
# dosages at the given QTL SNPs (no PCs, no other SNP), through the same MLP as
# the PCA encoding (DosagePCAEncoder with input_dim = number of QTL). The
# counterpart of orthogonalising the QTL out: how much of the genetic signal in
# generated roots those loci alone carry. Standardised on the training lines.
class QTLOnlyProjector:

    kind = 'widiv_dosage_qtl_only'

    def __init__(self):
        self.columns = self.mean_ = self.scale_ = self.snp_names = None

    def fit(self, dosage, snp_names, qtl_cols, verbose=True):
        self.snp_names = list(snp_names)
        self.columns = np.asarray(qtl_cols, dtype=np.int64)
        x = np.asarray(dosage, dtype=np.float32)[:, self.columns]
        observed = x >= 0
        counts = np.maximum(observed.sum(0), 1)
        self.mean_ = (np.where(observed, x, 0).sum(0) / counts).astype(np.float32)
        sd = np.sqrt((np.where(observed, x - self.mean_, 0) ** 2).sum(0) / np.maximum(counts - 1, 1))
        sd[sd == 0] = 1.0
        self.scale_ = sd.astype(np.float32)
        if verbose:
            print(f"  QTL only: the encoder sees the dosages of {len(self.columns)} QTL SNPs and nothing else")
        return self

    def transform(self, dosage):
        x = np.atleast_2d(np.asarray(dosage))[:, self.columns].astype(np.float32)
        z = (x - self.mean_) / self.scale_
        z[x < 0] = 0.0
        return z

    def project_lines(self, dosage, line_names):
        return self.transform(dosage)

    @property
    def output_dim(self):
        return len(self.columns)

    def check_columns(self, snp_names):
        if list(snp_names) != self.snp_names:
            raise SystemExit("this genotype matrix's SNP columns differ from the ones the "
                             "QTL-only projector was fitted on - refusing to encode it")

    def state_dict(self):
        return {'kind': self.kind, 'columns': self.columns, 'mean': self.mean_, 'scale': self.scale_,
                'snp_names': self.snp_names}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        obj.columns, obj.mean_, obj.scale_ = state['columns'], state['mean'], state['scale']
        obj.snp_names = list(state['snp_names'])
        return obj


def load_encoder_and_projector(checkpoint, device='cpu'):
    projector = DosagePCAProjector.from_state_dict(checkpoint['snp_projector'])
    encoder = DosagePCAEncoder(**checkpoint['snp_encoder_config'])
    encoder.load_state_dict(checkpoint['snp_encoder_state_dict'])
    encoder.to(device).eval()
    return encoder, projector
