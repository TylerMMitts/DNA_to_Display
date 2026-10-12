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

    # columns, when given, is the layout: one list of matrix column indices per
    # window, in genome order. None is the seed layout - consecutive columns of
    # the table as loaded - which suits the seed table because its B73v5 gene IDs
    # already run along each chromosome.
    def __init__(self, founders, window_size=200, n_loci=None, snp_names=None, columns=None,
                 segment_of_window=None):
        self.founders = tuple(int(f) for f in founders)
        self.window_size = int(window_size)
        self.n_loci = n_loci
        self.snp_names = list(snp_names) if snp_names is not None else None
        self.columns = [list(map(int, c)) for c in columns] if columns is not None else None
        self.segment_of_window = list(segment_of_window) if segment_of_window is not None else None
        self._index = None

    # gene_order: gene names in genome order, for a table whose loaded column
    # order is not genomic (the root table, see file_gene_order). With
    # split_chromosomes, windows are also cut wherever neighbouring loci stop
    # sharing founders (chromosome_breaks), so no window spans two chromosomes.
    def fit(self, snp_matrix, snp_names=None, gene_order=None, split_chromosomes=False):
        snp_matrix = np.asarray(snp_matrix)
        self.n_loci = int(snp_matrix.shape[1])
        if snp_names is not None:
            self.snp_names = list(snp_names)
        if gene_order is None and not split_chromosomes:
            self.columns, self.segment_of_window = None, None
            return self
        if gene_order is not None:
            col = {g: i for i, g in enumerate(self.snp_names)}
            listed = set(gene_order)
            missing = [g for g in self.snp_names if g not in listed]
            order = [col[g] for g in gene_order if g in col] + [col[g] for g in missing]
        else:
            order = list(range(self.n_loci))
        breaks = set(chromosome_breaks(snp_matrix[:, order])) if split_chromosomes else set()
        self.columns, self.segment_of_window = [], []
        segment, start = 0, 0
        for i in range(len(order)):
            end_of_segment = i in breaks or i == len(order) - 1
            if i - start + 1 == self.window_size or end_of_segment:
                self.columns.append(order[start:i + 1])
                self.segment_of_window.append(segment)
                start = i + 1
                if i in breaks:
                    segment += 1
        return self

    @property
    def n_windows(self):
        return len(self.columns) if self.columns is not None else math.ceil(self.n_loci / self.window_size)

    @property
    def output_dim(self):
        return self.n_windows * len(self.founders)

    # First and last gene of each window, for reporting which windows stay open.
    def window_bounds(self):
        out = []
        for w in range(self.n_windows):
            if self.columns is not None:
                a, b = self.columns[w][0], self.columns[w][-1]
            else:
                a, b = w * self.window_size, min((w + 1) * self.window_size, self.n_loci) - 1
            names = (self.snp_names[a], self.snp_names[b]) if self.snp_names else (str(a), str(b))
            row = {'window': w, 'first_column': a, 'last_column': b,
                   'first_gene': names[0], 'last_gene': names[1]}
            if self.columns is not None:
                row.update({'n_genes': len(self.columns[w]), 'segment': self.segment_of_window[w]})
            out.append(row)
        return out

    # (n_windows, longest window) column indices, padded with n_loci, which
    # points at an extra uncalled (-1) column appended before gathering.
    def _gather_index(self):
        if self._index is None:
            width = max(len(c) for c in self.columns)
            self._index = np.array([c + [self.n_loci] * (width - len(c)) for c in self.columns])
        return self._index

    def _padded(self, codes, lib):
        pad = self.n_windows * self.window_size - codes.shape[1]
        if pad:
            filler = lib.full((codes.shape[0], pad), -1, dtype=codes.dtype) if lib is np else \
                torch.full((codes.shape[0], pad), -1, dtype=codes.dtype, device=codes.device)
            codes = lib.concatenate([codes, filler], axis=1) if lib is np else torch.cat([codes, filler], dim=1)
        return codes.reshape(codes.shape[0], self.n_windows, self.window_size)

    def transform(self, snp_matrix):
        snp_matrix = np.asarray(snp_matrix, dtype=np.float32)
        if self.columns is None:
            x = self._padded(snp_matrix, np)
        else:
            pad = np.full((snp_matrix.shape[0], 1), -1, dtype=np.float32)
            x = np.concatenate([snp_matrix, pad], axis=1)[:, self._gather_index()]
        called = np.maximum((x > 0).sum(axis=2), 1)
        shares = np.stack([(x == f).sum(axis=2) / called for f in self.founders], axis=2)
        return shares.reshape(shares.shape[0], -1).astype(np.float32)

    def transform_torch(self, snp_batch):
        if snp_batch.dim() == 1:
            snp_batch = snp_batch[None, :]
        snp_batch = snp_batch.float()
        if self.columns is None:
            x = self._padded(snp_batch, torch)
        else:
            pad = torch.full((snp_batch.shape[0], 1), -1.0, device=snp_batch.device)
            index = torch.as_tensor(self._gather_index(), device=snp_batch.device)
            x = torch.cat([snp_batch, pad], dim=1)[:, index]
        called = (x > 0).sum(dim=2).clamp(min=1).float()
        shares = torch.stack([(x == f).sum(dim=2).float() / called for f in self.founders], dim=2)
        return shares.reshape(shares.shape[0], -1)

    def state_dict(self):
        return {'kind': self.kind, 'founders': list(self.founders), 'window_size': self.window_size,
                'n_loci': self.n_loci, 'snp_names': self.snp_names, 'columns': self.columns,
                'segment_of_window': self.segment_of_window}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['founders'], state['window_size'], state['n_loci'], state.get('snp_names'),
                   state.get('columns'), state.get('segment_of_window'))


# Genes in the order the parquet stores them, first appearance per gene. The
# root table is stored chromosome by chromosome, but the loader pivots it to
# gene-ID order, which is not genomic. Measured on the root table: neighbouring
# genes share a founder in 99.7% of lines in file order (9 breaks, the 9
# chromosome boundaries) against 99.4% in ID order (23 breaks).
def file_gene_order(parquet_path, column='gene_model'):
    import pandas as pd
    return list(pd.unique(pd.read_parquet(parquet_path, columns=[column])[column]))


# Positions i (in the given column order) where locus i and i+1 share a founder
# in fewer than `threshold` of the lines that have both called: places where
# neighbouring loci are no longer linked, which in genome order are the
# chromosome boundaries (unlinked loci agree about 1 time in 8).
def chromosome_breaks(snp_matrix, threshold=0.2):
    x = np.asarray(snp_matrix)
    ok = (x[:, 1:] > 0) & (x[:, :-1] > 0)
    same = ((x[:, 1:] == x[:, :-1]) & ok).sum(0) / np.maximum(ok.sum(0), 1)
    return [int(i) for i in np.flatnonzero(same < threshold)]


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
        # True holds every gate at exactly 1, in training and eval, and leaves the
        # gates untouched. The trainer sets it for its penalty-free warm-up, so the
        # model learns to use the genotype before any window can close. Not saved:
        # a loaded checkpoint always uses its learned gates.
        self.force_open = False
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


# The sparsity penalty's weight in a given epoch (0-based): nothing during the
# warm-up, then rising linearly to full strength over the ramp. Shared by the
# seed and root trainers.
def penalty_weight(epoch, full, warmup, ramp):
    if epoch < warmup:
        return 0.0
    if ramp and epoch < warmup + ramp:
        return full * (epoch - warmup + 1) / ramp
    return full


# Gate report: one row per window, its genes and how open its gate is, most open
# first. Returns the number of open gates.
def save_gate_report(snp_encoder, projector, path):
    import pandas as pd
    with torch.no_grad():
        prob = snp_encoder.open_probability().cpu().numpy()
        was_training, forced = snp_encoder.training, snp_encoder.force_open
        snp_encoder.eval()
        snp_encoder.force_open = False
        gate = snp_encoder.gates().cpu().numpy()
        snp_encoder.train(was_training)
        snp_encoder.force_open = forced
    rows = projector.window_bounds()
    for r, p, g in zip(rows, prob, gate):
        r.update({'open_probability': float(p), 'gate_eval': float(g), 'open': bool(g > 0)})
    pd.DataFrame(rows).sort_values('open_probability', ascending=False).to_csv(path, index=False)
    return int((gate > 0).sum())


def load_window_encoder(ckpt):
    projector = WindowProjector.from_state_dict(ckpt['snp_projector'])
    encoder = GatedWindowEncoder(**ckpt['snp_encoder_config'])
    encoder.load_state_dict(ckpt['snp_encoder_state_dict'])
    return RawCodeWindowEncoder(projector, encoder)
