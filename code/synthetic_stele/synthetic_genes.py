# The added genes of the synthetic-stele experiment, shared by the block-size
# sweep, the data builder and the post-training test so all three agree on
# what the genes are.
#
# Every added gene carries the same founder code for a genotype: 1 for the
# eighth of genotypes with the smallest measured stele, up to 8 for the eighth
# with the largest. A block of identical copies is what the block size sets.
# Genotypes with no measured stele - genotyped lines that were never imaged -
# get a code drawn at random, so the block has a balanced spread of codes
# without making up a stele size for them; they are never trained on, since a
# genotype needs an image to be a training example.

import numpy as np
import pandas as pd

# Sorted ahead of every real gene ('Zm...') by the parquet pivot, so the block
# is the first columns of the matrix and one contiguous run of loci.
NAME_PREFIX = 'SYN_STELE_'


def gene_names(block_size):
    return [f'{NAME_PREFIX}{i:05d}' for i in range(1, block_size + 1)]


# Founder code per genotype, in the order of sample_names. Genotypes are split
# into n_founders groups of as near equal size as ranks allow, smallest stele
# first.
def stele_codes(ranking_csv, sample_names, n_founders=8, seed=0):
    ranking = pd.read_csv(ranking_csv)
    mean = dict(zip(ranking['genotype'], ranking['stele_diameter_mean']))
    measured = [g for g in sample_names if g in mean and np.isfinite(mean[g])]
    order = sorted(measured, key=lambda g: mean[g])
    group = {g: 1 + (i * n_founders) // len(order) for i, g in enumerate(order)}
    rng = np.random.default_rng(seed)
    codes = np.array([group[g] if g in group else rng.integers(1, n_founders + 1)
                      for g in sample_names], dtype=float)
    return codes, np.array([g in group for g in sample_names])


# The SNP matrix with the block in front: the same column order the parquet
# pivot gives the written copy, so a sweep measures exactly what training sees.
def with_block(snp_matrix, codes, block_size):
    if block_size == 0:
        return np.asarray(snp_matrix)
    block = np.repeat(np.asarray(codes, dtype=float)[:, None], block_size, axis=1)
    return np.hstack([block, np.asarray(snp_matrix)])
