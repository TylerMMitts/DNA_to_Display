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


class DosagePCAProjector:

    def __init__(self, n_components=None, target_variance=0.80):
        # n_components fixes the size; otherwise the smallest number reaching
        # target_variance of the training lines' total variance.
        self.n_components = n_components
        self.target_variance = target_variance
        self.mean_ = None
        self.scale_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.snp_names = None

    # Standardises dosages: missing (-1) becomes the training mean, so 0 after centring.
    def _standardize(self, dosage):
        x = np.asarray(dosage, dtype=np.float32)
        missing = x < 0
        z = (x - self.mean_) / self.scale_
        if missing.any():
            z[missing] = 0.0
        return z

    def fit(self, dosage, snp_names=None, verbose=True):
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
        del z
        self.explained_variance_ratio_ = ratio[:k]
        self.explained_variance_ = lam[:k] / max(n - 1, 1)
        if verbose:
            print(f"  dosage PCA on {n} training lines x {p:,} SNPs: {k} components, "
                  f"{ratio[:k].sum():.1%} of variance")
        return self

    def transform(self, dosage, chunk_size=128):
        if self.components_ is None:
            raise RuntimeError("DosagePCAProjector must be fitted before transform")
        dosage = np.atleast_2d(np.asarray(dosage))
        out = [self._standardize(dosage[i:i + chunk_size]) @ self.components_.T
               for i in range(0, len(dosage), chunk_size)]
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
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(n_components=state.get('n_components'),
                  target_variance=state.get('target_variance', 0.80))
        obj.mean_ = np.asarray(state['mean'], dtype=np.float32)
        obj.scale_ = np.asarray(state['scale'], dtype=np.float32)
        obj.components_ = np.asarray(state['components'], dtype=np.float32)
        obj.explained_variance_ratio_ = np.asarray(state['explained_variance_ratio'])
        obj.explained_variance_ = np.asarray(state['explained_variance'])
        obj.snp_names = state.get('snp_names')
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


def load_encoder_and_projector(checkpoint, device='cpu'):
    projector = DosagePCAProjector.from_state_dict(checkpoint['snp_projector'])
    encoder = DosagePCAEncoder(**checkpoint['snp_encoder_config'])
    encoder.load_state_dict(checkpoint['snp_encoder_state_dict'])
    encoder.to(device).eval()
    return encoder, projector
