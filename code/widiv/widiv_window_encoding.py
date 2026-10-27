# SNP encoding option F for WiDiv: a few PCs plus gated windows of SNPs.
#
# Option A (widiv_snp_encoding.py) reduces the 90,790 dosages to about 370 PCs.
# That predicts traits from genome-wide relatedness about as well as GBLUP, but
# keeps little of any single SNP (designed QTL R^2 from the PCs: median 0.32,
# rare QTL 0.06), so a designed gene moved its trait by about 0.4% of its
# designed size in generated roots (results/widiv/qtl_contribution/). This
# encoding keeps every SNP reachable:
#   1. a handful of PCs (n_pcs, 32 by default) carry the population structure,
#      the part of the genotype that says which lines are related;
#   2. the SNPs, in genome order, are cut into windows of window_size
#      neighbouring SNPs that never cross a chromosome boundary (about 460 at
#      200). Each window has its own learned linear map from its standardised
#      dosages to window_dim features, so one SNP can carry a weight of its own;
#   3. each window's features pass through a learned hard-concrete gate (the
#      same gates as latent_diffusion/models/window_encoding.py, Louizos et al.
#      2018). The trainer adds a penalty per gate expected to be open, so a
#      window stays open only if it lowers the denoising error by more than that.
# The PCs and gated window features go through the same MLP as the PCA encoder
# and come out as the same tokens, so the UNet and every sampling path are
# unchanged. The encoder takes raw 0/1/2 dosages (missing = -1) and does the
# standardisation itself, so its "projected" input is the dosage row.
#
# Fitted on training lines only, like option A: the standardisation and the PCs
# never see a test line.

import math

import numpy as np
import torch
import torch.nn as nn

from latent_diffusion.models.window_encoding import BETA, GAMMA, ZETA
from widiv_snp_encoding import DosagePCAEncoder, DosagePCAProjector


class DosageWindowLayout:
    # The fitted part of option F: dosage standardisation and PCs (a
    # DosagePCAProjector) and the window layout. Plays the projector's role in
    # train_widiv.py: transform() gives the encoder's input, which here is the
    # dosage itself; the encoder holds the numbers that turn it into features.

    kind = 'widiv_dosage_windows'

    def __init__(self, window_size=200, n_pcs=32):
        self.window_size = int(window_size)
        self.n_pcs = int(n_pcs)
        self.pca = None
        self.windows = None             # one list of matrix columns per window, genome order
        self.window_chromosome = None
        self.snp_names = None
        self.positions = None

    # chromosomes, positions: per SNP column of the dosage matrix (from the SNP map).
    def fit(self, dosage, snp_names, chromosomes, positions, verbose=True):
        self.snp_names = list(snp_names)
        self.pca = DosagePCAProjector(n_components=self.n_pcs).fit(dosage, snp_names, verbose=verbose)
        chromosomes, positions = np.asarray(chromosomes), np.asarray(positions)
        self.positions = positions.tolist()
        order = np.lexsort((positions, chromosomes))
        self.windows, self.window_chromosome = [], []
        for c in dict.fromkeys(chromosomes[order].tolist()):
            cols = order[chromosomes[order] == c]
            for start in range(0, len(cols), self.window_size):
                self.windows.append(cols[start:start + self.window_size].tolist())
                self.window_chromosome.append(c)
        if verbose:
            print(f"  {self.n_windows} windows of up to {self.window_size} SNPs on "
                  f"{len(set(self.window_chromosome))} chromosomes, plus {self.n_pcs} PCs")
        return self

    @property
    def n_windows(self):
        return len(self.windows)

    @property
    def n_snps(self):
        return len(self.snp_names)

    # The encoder's input: dosages as int8, columns checked by the caller.
    def transform(self, dosage):
        return np.atleast_2d(np.asarray(dosage)).astype(np.int8)

    @property
    def output_dim(self):
        return self.n_snps

    # (n_windows, longest window) column indices, padded with n_snps, which
    # points at an extra all-zero column the encoder appends before gathering.
    def gather_index(self):
        width = max(len(w) for w in self.windows)
        return np.array([w + [self.n_snps] * (width - len(w)) for w in self.windows], dtype=np.int64)

    # First and last SNP of each window, for the gate report (save_gate_report).
    def window_bounds(self):
        rows = []
        for i, (w, c) in enumerate(zip(self.windows, self.window_chromosome)):
            rows.append({'window': i, 'chr': c, 'first_snp': self.snp_names[w[0]],
                         'last_snp': self.snp_names[w[-1]], 'start_pos': self.positions[w[0]],
                         'end_pos': self.positions[w[-1]], 'n_snps': len(w)})
        return rows

    # Window index of every SNP column, for finding which window a QTL sits in.
    def window_of_column(self):
        out = np.full(self.n_snps, -1)
        for i, w in enumerate(self.windows):
            out[w] = i
        return out

    def check_columns(self, snp_names):
        if list(snp_names) != self.snp_names:
            raise SystemExit("this genotype matrix's SNP columns differ from the ones the "
                             "window layout was fitted on - refusing to encode it")

    def state_dict(self):
        return {'kind': self.kind, 'window_size': self.window_size, 'n_pcs': self.n_pcs,
                'pca': self.pca.state_dict(), 'windows': self.windows,
                'window_chromosome': self.window_chromosome, 'snp_names': self.snp_names,
                'positions': self.positions}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['window_size'], state['n_pcs'])
        obj.pca = DosagePCAProjector.from_state_dict(state['pca'])
        obj.windows = [list(map(int, w)) for w in state['windows']]
        obj.window_chromosome = list(state['window_chromosome'])
        obj.snp_names = list(state['snp_names'])
        obj.positions = list(state['positions'])
        return obj


class GatedDosageWindowEncoder(nn.Module):
    # Dosages (B, n_snps) -> tokens (B, num_tokens, embedding_dim).

    def __init__(self, n_snps, n_pcs, n_windows, window_width, window_dim=4, embedding_dim=256,
                 num_tokens=8, hidden_dim=512, dropout=0.1, init_open=0.9):
        super().__init__()
        self.n_snps, self.n_pcs, self.n_windows = int(n_snps), int(n_pcs), int(n_windows)
        self.window_width, self.window_dim = int(window_width), int(window_dim)
        self.embedding_dim, self.num_tokens, self.hidden_dim = embedding_dim, num_tokens, hidden_dim
        self.dropout, self.init_open = dropout, init_open
        # Fixed numbers from DosageWindowLayout, saved with the weights so a
        # checkpoint encodes dosages on its own (set_layout fills them once).
        self.register_buffer('snp_mean', torch.zeros(self.n_snps))
        self.register_buffer('snp_scale', torch.ones(self.n_snps))
        self.register_buffer('pca_components', torch.zeros(self.n_pcs, self.n_snps))
        self.register_buffer('window_index', torch.full((self.n_windows, self.window_width), self.n_snps,
                                                        dtype=torch.long))
        # Each window's own map from its SNPs to window_dim features; padding
        # slots read a zero column, so they never contribute. Scaled so a
        # window's features start at about unit spread.
        self.window_weight = nn.Parameter(torch.randn(self.n_windows, self.window_width, self.window_dim)
                                          / math.sqrt(self.window_width))
        self.window_bias = nn.Parameter(torch.zeros(self.n_windows, self.window_dim))
        # One gate per window, as GatedWindowEncoder: starts mostly open.
        self.log_alpha = nn.Parameter(torch.full((self.n_windows,), math.log(init_open / (1 - init_open))))
        # True holds every gate at exactly 1 (the trainer's penalty-free warm-up).
        # Not saved: a loaded checkpoint always uses its learned gates.
        self.force_open = False
        input_dim = self.n_pcs + self.n_windows * self.window_dim
        # The same MLP as DosagePCAEncoder / OneHotSNPEncoder.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tokens * embedding_dim),
        )
        self.token_positions = nn.Parameter(torch.randn(1, num_tokens, embedding_dim) * 0.02)

    @classmethod
    def for_layout(cls, layout, **kwargs):
        index = layout.gather_index()
        return cls(layout.n_snps, layout.n_pcs, layout.n_windows, index.shape[1], **kwargs)

    def set_layout(self, layout):
        pca = layout.pca
        with torch.no_grad():
            self.snp_mean.copy_(torch.as_tensor(pca.mean_))
            self.snp_scale.copy_(torch.as_tensor(pca.scale_))
            self.pca_components.copy_(torch.as_tensor(pca.components_))
            self.window_index.copy_(torch.as_tensor(layout.gather_index()))
        return self

    # Sampled in training (one draw per batch); deterministic in eval, where a
    # gate whose open probability is low enough is exactly 0.
    def gates(self):
        if self.force_open:
            return torch.ones_like(self.log_alpha)
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    # Expected number of open gates: the L0 penalty, differentiable in log_alpha.
    def expected_open(self):
        return self.open_probability().sum()

    def open_probability(self):
        return torch.sigmoid(self.log_alpha - BETA * math.log(-GAMMA / ZETA))

    # Standardised dosages, missing (-1) as 0, as DosagePCAProjector does.
    def standardize(self, dosage):
        x = dosage.float()
        z = (x - self.snp_mean) / self.snp_scale
        return torch.where(x < 0, torch.zeros_like(z), z)

    # Each window's features before its gate, (B, n_windows, window_dim).
    def window_features(self, z):
        padded = torch.cat([z, z.new_zeros(z.shape[0], 1)], dim=1)[:, self.window_index]
        return torch.einsum('bnw,nwd->bnd', padded, self.window_weight) + self.window_bias

    def forward(self, dosage):
        z = self.standardize(dosage)
        pcs = z @ self.pca_components.T
        w = self.window_features(z) * self.gates()[None, :, None]
        x = self.net(torch.cat([pcs, w.flatten(1)], dim=1))
        return x.view(-1, self.num_tokens, self.embedding_dim) + self.token_positions

    def config(self):
        return {'n_snps': self.n_snps, 'n_pcs': self.n_pcs, 'n_windows': self.n_windows,
                'window_width': self.window_width, 'window_dim': self.window_dim,
                'embedding_dim': self.embedding_dim, 'num_tokens': self.num_tokens,
                'hidden_dim': self.hidden_dim, 'dropout': self.dropout, 'init_open': self.init_open}


# Either encoding from a WiDiv diffusion checkpoint, in eval mode, with the
# projector that turns a dosage matrix into the encoder's input:
#     encoder, projector = load_widiv_encoder(ckpt)
#     tokens = encoder(torch.tensor(projector.transform(dosage), dtype=torch.float32))
def load_widiv_encoder(ckpt, device='cpu'):
    state = ckpt['snp_projector']
    if state.get('kind') == DosageWindowLayout.kind:
        projector = DosageWindowLayout.from_state_dict(state)
        encoder = GatedDosageWindowEncoder(**ckpt['snp_encoder_config'])
    else:
        projector = DosagePCAProjector.from_state_dict(state)
        encoder = DosagePCAEncoder(**ckpt['snp_encoder_config'])
    encoder.load_state_dict(ckpt['snp_encoder_state_dict'])
    return encoder.to(device).eval(), projector
