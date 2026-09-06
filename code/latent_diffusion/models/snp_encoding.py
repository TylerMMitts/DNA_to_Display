# One-hot SNP encoding and the PCA projection built on it.
#
# Each locus becomes an 8-wide indicator vector, so every founder sits the
# same distance from every other one and no false ordering is implied. PCA
# then reduces that expansion to a manageable conditioning vector.

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

DEFAULT_FOUNDERS = (1, 2, 3, 4, 5, 6, 7, 8)

# Converts snp values into one-hot encoded representation
# This gives each locus a vector with size 8, where each position corresponds to a founder and is 1 if the locus matches that founder, 0 otherwise.
def one_hot_founders(snp_matrix, founders=DEFAULT_FOUNDERS, dtype=np.float32):
    snp_matrix = np.asarray(snp_matrix)
    founders = np.asarray(founders)
    N, L = snp_matrix.shape
    F = len(founders)

    codes = np.searchsorted(founders, snp_matrix)
    codes = np.clip(codes, 0, F - 1)
    valid = np.isin(snp_matrix, founders)

    out = np.zeros((N, L, F), dtype=dtype)
    np.put_along_axis(out, codes[:, :, None], 1.0, axis=2)
    out *= valid[:, :, None]
    return out.reshape(N, L * F)

# Same as one_hot_founders but implemented in PyTorch for a batch of SNPs.
def one_hot_founders_torch(snp_batch, founders=DEFAULT_FOUNDERS):
    founders_t = torch.as_tensor(founders, dtype=snp_batch.dtype,
                                 device=snp_batch.device)
    F = len(founders)
    B, L = snp_batch.shape

    match = (snp_batch[:, :, None] == founders_t[None, None, :]).to(snp_batch.dtype)
    return match.reshape(B, L * F)


class SNPProjector:

    def __init__(self, founders=DEFAULT_FOUNDERS, n_components=None,
                 target_variance=0.95, random_state=0):
        self.founders = tuple(int(f) for f in founders)
        self.n_components = n_components
        self.target_variance = target_variance
        self.random_state = random_state

        # Stored as plain arrays rather than a live sklearn PCA object.
        # A PCA transform is exactly (x - mean) @ components.T
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.n_loci = None

    def fit(self, snp_matrix, verbose=True):
        snp_matrix = np.asarray(snp_matrix)
        self.n_loci = snp_matrix.shape[1]

        if verbose:
            width = self.n_loci * len(self.founders)
            print(f"One-hot encoding {snp_matrix.shape[0]} genotypes x "
                  f"{self.n_loci} loci x {len(self.founders)} founders "
                  f"-> {width:,} columns "
                  f"({snp_matrix.shape[0] * width * 4 / 1e9:.2f} GB float32)")

        encoded = one_hot_founders(snp_matrix, self.founders)

        n_components = self.n_components
        if n_components is None:
            # PCA cannot extract more components than samples-1
            max_components = min(encoded.shape[0], encoded.shape[1]) - 1
            probe = PCA(n_components=max_components, random_state=self.random_state)
            probe.fit(encoded)
            cumulative = np.cumsum(probe.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumulative, self.target_variance) + 1)
            n_components = min(n_components, max_components)
            if verbose:
                print(f"  {n_components} components reach "
                      f"{cumulative[n_components - 1]:.2%} explained variance")

        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(encoded)
        self.mean_ = pca.mean_.astype(np.float64)
        self.components_ = pca.components_.astype(np.float64)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_.astype(np.float64)
        self.explained_variance_ = pca.explained_variance_.astype(np.float64)
        if verbose:
            print(f"  PCA fitted: {n_components} components, "
                  f"{self.explained_variance_ratio_.sum():.2%} variance")
        return self

    def transform(self, snp_matrix, chunk_size=64):

        if self.components_ is None:
            raise RuntimeError("SNPProjector must be fitted before transform")

        snp_matrix = np.atleast_2d(np.asarray(snp_matrix))
        chunks = []
        for start in range(0, len(snp_matrix), chunk_size):
            block = one_hot_founders(snp_matrix[start:start + chunk_size], self.founders)
            chunks.append((block - self.mean_) @ self.components_.T)
        return np.concatenate(chunks, axis=0)

    def inverse_transform(self, scores):

        if self.components_ is None:
            raise RuntimeError("SNPProjector must be fitted before inverse_transform")
        scores = np.atleast_2d(np.asarray(scores))
        return scores @ self.components_ + self.mean_

    @property
    def output_dim(self):
        return int(self.components_.shape[0])

    def state_dict(self):

        return {
            'founders': self.founders,
            'n_loci': self.n_loci,
            'random_state': self.random_state,
            'target_variance': self.target_variance,
            # float32 keeps the checkpoint small; the components matrix is
            # n_components x (n_loci * n_founders)
            'pca_mean': self.mean_.astype(np.float32),
            'pca_components': self.components_.astype(np.float32),
            'pca_explained_variance_ratio': self.explained_variance_ratio_,
            'pca_explained_variance': self.explained_variance_,
        }

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(founders=state['founders'], random_state=state.get('random_state', 0),
                  target_variance=state.get('target_variance', 0.95))
        obj.n_loci = state['n_loci']
        obj.mean_ = np.asarray(state['pca_mean'], dtype=np.float64)
        obj.components_ = np.asarray(state['pca_components'], dtype=np.float64)
        obj.explained_variance_ratio_ = np.asarray(
            state['pca_explained_variance_ratio'], dtype=np.float64)
        # Absent from checkpoints written before this field was added; left as
        # None so ensure_explained_variance() can recover it from the
        # population rather than silently substituting a wrong scale.
        ev = state.get('pca_explained_variance')
        obj.explained_variance_ = (None if ev is None
                                   else np.asarray(ev, dtype=np.float64))
        return obj

    def ensure_explained_variance(self, snp_matrix, verbose=True):

        if self.explained_variance_ is not None:
            return self
        scores = self.transform(snp_matrix)
        self.explained_variance_ = scores.var(axis=0, ddof=1)
        if verbose:
            print("  explained_variance_ not in checkpoint; recomputed from "
                  f"{scores.shape[0]} genotypes (population-derived)")
        return self

    def locus_contributions(self, component, snp_vector):

        if self.components_ is None:
            raise RuntimeError("SNPProjector must be fitted before "
                               "locus_contributions")

        L, F = self.n_loci, len(self.founders)
        loading_2d = self.components_[component].reshape(L, F)
        mean_2d = self.mean_.reshape(L, F)

        codes = np.asarray(snp_vector).ravel()
        onehot = one_hot_founders(codes[None, :], self.founders)[0].reshape(L, F)

        per_slot = loading_2d * (onehot - mean_2d)          # [L, F]
        per_locus = per_slot.sum(axis=1)                     # [L]

        founder_of_locus = np.full(L, -1, dtype=int)
        valid = np.isin(codes, self.founders)
        founder_of_locus[valid] = codes[valid].astype(int)

        return per_locus, founder_of_locus, per_slot

    def project_onehot_torch(self, encoded):

        mean = torch.as_tensor(self.mean_, dtype=encoded.dtype, device=encoded.device)
        components = torch.as_tensor(self.components_, dtype=encoded.dtype,
                                     device=encoded.device)
        return (encoded - mean) @ components.T

    def project_torch(self, snp_batch):
        return self.project_onehot_torch(
            one_hot_founders_torch(snp_batch, self.founders))


class OneHotSNPEncoder(nn.Module):

    def __init__(self, input_dim, embedding_dim=512, num_tokens=8, hidden_dim=1024,
                 dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tokens * embedding_dim),
        )
        self.token_positions = nn.Parameter(
            torch.randn(1, num_tokens, embedding_dim) * 0.02)

    def forward(self, projected):
        if projected.dtype != torch.float32:
            projected = projected.float()
        x = self.net(projected)
        x = x.view(-1, self.num_tokens, self.embedding_dim)
        return x + self.token_positions

    def config(self):
        return {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'num_tokens': self.num_tokens,
            'hidden_dim': self.hidden_dim,
        }


def load_encoder_and_projector(checkpoint, device='cpu'):
    projector = SNPProjector.from_state_dict(checkpoint['snp_projector'])
    encoder = OneHotSNPEncoder(**checkpoint['snp_encoder_config'])
    encoder.load_state_dict(checkpoint['snp_encoder_state_dict'])
    encoder.to(device).eval()
    return encoder, projector


class RawCodeOneHotEncoder(nn.Module):

    def __init__(self, projector, encoder):
        super().__init__()
        self.encoder = encoder
        self.founders = tuple(projector.founders)
        self.output_dim = projector.output_dim
        self.projector = projector
        self.pca = projector

        self.register_buffer('pca_mean',
                             torch.as_tensor(projector.mean_, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('pca_components',
                             torch.as_tensor(projector.components_, dtype=torch.float32),
                             persistent=False)

    def forward(self, snp_batch):
        if snp_batch.dtype != torch.float32:
            snp_batch = snp_batch.float()
        if snp_batch.dim() == 1:
            snp_batch = snp_batch[None, :]
        encoded = one_hot_founders_torch(snp_batch, self.founders)
        projected = (encoded - self.pca_mean) @ self.pca_components.T
        return self.encoder(projected)


    @property
    def net(self):
        return self.encoder.net

    @property
    def num_tokens(self):
        return self.encoder.num_tokens

    @property
    def embedding_dim(self):
        return self.encoder.embedding_dim

    @property
    def token_positions(self):
        return self.encoder.token_positions
