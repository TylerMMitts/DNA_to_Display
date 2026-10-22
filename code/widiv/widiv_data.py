# Splits and image transforms shared by the WiDiv LiteVAE and diffusion trainers.

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

SPLITS = ('split_random', 'split_cluster')


# A cross-section has no up, down, left or right: the root is radially
# symmetric and centred on a white canvas, so quarter turns and flips give real
# new views without moving any of it out of frame. No colour jitter: the images
# are rendered with one colour per tissue, so colour identifies tissue rather
# than varying with lighting as it does in the photographed MEMA roots.
class RandomQuarterTurn:

    def __call__(self, x):
        return torch.rot90(x, int(torch.randint(4, (1,))), dims=(-2, -1))


def make_transforms(augment):
    base = [transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    if not augment:
        return transforms.Compose(base)
    return transforms.Compose(base + [transforms.RandomHorizontalFlip(p=0.5),
                                      RandomQuarterTurn()])


# The delivered split: line -> 'train' / 'test' under both splits, and its cluster.
def load_split(path):
    return pd.read_csv(path).set_index('genotype')


# Validation lines for checkpoint selection and the per-epoch genotype gain,
# taken from the training side so the delivered test lines stay untouched until
# evaluation. Chosen the same way as that split's test lines: random lines for
# split_random, whole genetic clusters for split_cluster (whose test set is
# whole clusters, so a random-line validation set would measure the easier case).
def choose_val_lines(split_table, split, fraction, seed):
    train = split_table[split_table[split] == 'train']
    rng = np.random.default_rng(seed)
    target = int(round(len(train) * fraction))
    if split == 'split_random':
        lines = train.index.to_numpy().copy()
        rng.shuffle(lines)
        return set(lines[:target])
    clusters = np.array(sorted(train.cluster.unique()))
    rng.shuffle(clusters)
    chosen = []
    for c in clusters:
        if sum(int((train.cluster == x).sum()) for x in chosen) >= target:
            break
        chosen.append(c)
    return set(train.index[train.cluster.isin(chosen)])
