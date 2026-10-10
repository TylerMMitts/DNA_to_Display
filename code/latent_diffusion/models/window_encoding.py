# Founder-ancestry windows with learned gates: an alternative to the one-hot
# PCA encoding, for the seed trainer.
#
# The PCA encoding compresses the whole genome into directions of overall
# variation. Every line stays uniquely recognisable in that space, and a single
# region - the kernel-colour locus, about 456 of 33,527 genes - is about 1% of
# it, so the model learns to recall lines instead of learning what the region
# does. This encoding instead:
#   1. splits the loci, in table order, into windows of `window_size`
#      neighbouring genes (B73v5 gene IDs run along the chromosome, so a window
#      is a stretch of chromosome, give or take a chromosome boundary);
#   2. summarises each window as the share of its called loci from each founder,
#      8 numbers per window;
#   3. multiplies each window by a learned gate between 0 and 1. The gates are
#      hard-concrete (Louizos, Welling & Kingma 2018, "Learning sparse neural
#      networks through L0 regularization"): during training each is sampled and
#      is often exactly 0 or exactly 1, and the trainer adds a penalty per gate
#      expected to be open. A window stays open only if it lowers the denoising
#      error by more than that penalty. A handful of windows cannot single out
#      one of ~370 lines, so with few open the model has to learn what the open
#      windows do.
# The gated features then go through the same MLP as OneHotSNPEncoder and come
# out as the same tokens, so the UNet and every generation path are unchanged.

import math

import numpy as np
import torch
import torch.nn as nn

# Hard-concrete constants from the paper: temperature, and the stretch that
# lets a sampled gate land exactly on 0 or 1.
BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1


class WindowProjector:
    # Raw founder codes (uncalled = -1) -> founder shares per window. Holds no
    # fitted state beyond the window layout, so the same table always gives the
    # same features. Mirrors SNPProjector's state_dict/from_state_dict.

    kind = 'founder_windows'

    def __init__(self, founders, window_size=200, n_loci=None, snp_names=None):
        self.founders = tuple(int(f) for f in founders)
        self.window_size = int(window_size)
        self.n_loci = n_loci
        self.snp_names = list(snp_names) if snp_names is not None else None

    def fit(self, snp_matrix, snp_names=None):
        self.n_loci = int(np.asarray(snp_matrix).shape[1])
        if snp_names is not None:
            self.snp_names = list(snp_names)
        return self

    @property
    def n_windows(self):
        return math.ceil(self.n_loci / self.window_size)

    @property
    def output_dim(self):
        return self.n_windows * len(self.founders)

    # First and last gene of each window, for reporting which windows stay open.
    def window_bounds(self):
        out = []
        for w in range(self.n_windows):
            a, b = w * self.window_size, min((w + 1) * self.window_size, self.n_loci) - 1
            names = (self.snp_names[a], self.snp_names[b]) if self.snp_names else (str(a), str(b))
            out.append({'window': w, 'first_column': a, 'last_column': b,
                        'first_gene': names[0], 'last_gene': names[1]})
        return out

    def _padded(self, codes, lib):
        pad = self.n_windows * self.window_size - codes.shape[1]
        if pad:
            filler = lib.full((codes.shape[0], pad), -1, dtype=codes.dtype) if lib is np else \
                torch.full((codes.shape[0], pad), -1, dtype=codes.dtype, device=codes.device)
            codes = lib.concatenate([codes, filler], axis=1) if lib is np else torch.cat([codes, filler], dim=1)
        return codes.reshape(codes.shape[0], self.n_windows, self.window_size)

    def transform(self, snp_matrix):
        x = self._padded(np.asarray(snp_matrix, dtype=np.float32), np)
        called = np.maximum((x > 0).sum(axis=2), 1)
        shares = np.stack([(x == f).sum(axis=2) / called for f in self.founders], axis=2)
        return shares.reshape(shares.shape[0], -1).astype(np.float32)

    def transform_torch(self, snp_batch):
        if snp_batch.dim() == 1:
            snp_batch = snp_batch[None, :]
        x = self._padded(snp_batch.float(), torch)
        called = (x > 0).sum(dim=2).clamp(min=1).float()
        shares = torch.stack([(x == f).sum(dim=2).float() / called for f in self.founders], dim=2)
        return shares.reshape(shares.shape[0], -1)

    def state_dict(self):
        return {'kind': self.kind, 'founders': list(self.founders), 'window_size': self.window_size,
                'n_loci': self.n_loci, 'snp_names': self.snp_names}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['founders'], state['window_size'], state['n_loci'], state.get('snp_names'))


class GatedWindowEncoder(nn.Module):
    # Window features (B, n_windows * n_founders) -> tokens (B, num_tokens, dim).

    def __init__(self, n_windows, n_founders, embedding_dim=512, num_tokens=8, hidden_dim=1024,
                 dropout=0.1, init_open=0.9):
        super().__init__()
        self.n_windows, self.n_founders = int(n_windows), int(n_founders)
        self.embedding_dim, self.num_tokens, self.hidden_dim = embedding_dim, num_tokens, hidden_dim
        self.init_open = init_open
        # One gate per window. Starts mostly open, so the model sees every region
        # before the penalty starts closing the ones it does not need.
        self.log_alpha = nn.Parameter(torch.full((self.n_windows,), math.log(init_open / (1 - init_open))))
        input_dim = self.n_windows * self.n_founders
        # The same MLP as OneHotSNPEncoder.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tokens * embedding_dim),
        )
        self.token_positions = nn.Parameter(torch.randn(1, num_tokens, embedding_dim) * 0.02)

    # Sampled in training (one draw per batch); deterministic in eval, where a
    # gate whose open probability is low enough is exactly 0.
    def gates(self):
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    # Expected number of open gates: the L0 penalty, differentiable in log_alpha.
    def expected_open(self):
        return torch.sigmoid(self.log_alpha - BETA * math.log(-GAMMA / ZETA)).sum()

    def open_probability(self):
        return torch.sigmoid(self.log_alpha - BETA * math.log(-GAMMA / ZETA))

    def forward(self, features):
        if features.dtype != torch.float32:
            features = features.float()
        b = features.shape[0]
        x = features.view(b, self.n_windows, self.n_founders) * self.gates()[None, :, None]
        x = self.net(x.reshape(b, -1)).view(-1, self.num_tokens, self.embedding_dim)
        return x + self.token_positions

    def config(self):
        return {'n_windows': self.n_windows, 'n_founders': self.n_founders,
                'embedding_dim': self.embedding_dim, 'num_tokens': self.num_tokens,
                'hidden_dim': self.hidden_dim, 'init_open': self.init_open}


class RawCodeWindowEncoder(nn.Module):
    # Raw founder codes in, tokens out - the calling convention every analysis
    # script uses, as RawCodeOneHotEncoder gives the PCA models. There is no PCA,
    # so PCA-specific analyses see pca = None and skip or stop.

    def __init__(self, projector, encoder):
        super().__init__()
        self.projector, self.encoder = projector, encoder
        self.founders = projector.founders
        self.output_dim = projector.output_dim
        self.pca = None

    def forward(self, snp_batch):
        return self.encoder(self.projector.transform_torch(snp_batch))


def load_window_encoder(ckpt):
    projector = WindowProjector.from_state_dict(ckpt['snp_projector'])
    encoder = GatedWindowEncoder(**ckpt['snp_encoder_config'])
    encoder.load_state_dict(ckpt['snp_encoder_state_dict'])
    return RawCodeWindowEncoder(projector, encoder)
