# How many copies of the synthetic stele gene the model needs to be able to use
# it, measured on the encoder's input without training anything.
#
# The diffusion model never sees the genome itself: every genotype is one-hot
# encoded and reduced by PCA to the vector the SNP encoder reads. Whatever the
# added block does not put into that vector, no amount of training can recover.
# For each block size this fits that same one-hot + PCA step on the augmented
# table and measures:
#   - held-out accuracy: sort the genotypes the model never trains on into
#     their stele group by the nearest group centre of the training genotypes,
#     using only the PCA vector. Chance is 1 in 8. This is the closest thing to
#     "could the model learn the rule and apply it to a new line" that can be
#     measured without training.
#   - flip distance: how far switching the block from founder 1 to founder 8
#     moves a genotype's PCA vector, against the distance from a genotype to its
#     nearest real neighbour. Much smaller than that, and a flip looks to the
#     model like the same line.
#   - real-genome retention: how much of the real genes' variation the PCA still
#     keeps once the block takes its share. The cost of a large block.
#
# Writes block_size_sweep.csv, block_size_sweep.png and summary.json to
# results/synthetic_stele/block_size_sweep/.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    DIFFUSION_ONEHOT_MEDIUM_DIR, SNP_PARQUET, SYNTHETIC_STELE_RESULTS_DIR,
    apply_overrides, resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.models.snp_encoding import SNPProjector, one_hot_founders
from synthetic_stele.synthetic_genes import stele_codes, with_block


# Share of the variance of the chosen one-hot columns the fitted PCA keeps,
# measured by projecting onto the retained components and back. Done a slice of
# columns at a time: at the largest block the full centred and rebuilt matrices
# would each be several hundred megabytes.
def retained(projector, encoded, scores, cols, chunk=40000):
    total = lost = 0.0
    for start in range(0, len(cols), chunk):
        c = cols[start:start + chunk]
        centred = encoded[:, c] - projector.mean_[c]
        rebuilt = scores @ projector.components_[:, c]
        total += float((centred ** 2).sum())
        lost += float(((centred - rebuilt) ** 2).sum())
    return 1 - lost / total if total else float('nan')


def main(overrides=None):
    # Edit these values, then run:
    #     python code/synthetic_stele/block_size_sweep.py
    class cfg:
        snp_parquet = SNP_PARQUET
        ranking_csv = SYNTHETIC_STELE_RESULTS_DIR / 'real_root_traits' / 'genotype_stele_ranking.csv'
        # Any checkpoint trained on this genotype list and seed: only its record
        # of which genotypes were held out is read, which the synthetic run will
        # reproduce, since adding genes does not change the genotype list.
        split_checkpoint = DIFFUSION_ONEHOT_MEDIUM_DIR / 'diffusion_onehot_medium_epoch_300.pt'
        output_dir = SYNTHETIC_STELE_RESULTS_DIR / 'block_size_sweep'

        block_sizes = [0, 1, 10, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
        n_founders = 8
        target_variance = 0.95      # as train_onehot.py fits it
        seed = 0

        # The recommendation: the smallest block whose held-out accuracy is
        # within this many points of the best seen, and whose real-genome
        # retention has not fallen more than max_retention_drop below the run
        # with no block.
        accuracy_slack = 0.03
        max_retention_drop = 0.05

    apply_overrides(cfg, overrides)

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names, _, snp_matrix = load_snp_data_from_parquet(resolve_input(cfg.snp_parquet, 'SNP parquet'))
    names = list(names)
    snp_matrix = np.asarray(snp_matrix)
    founders = tuple(range(1, cfg.n_founders + 1))
    codes, measured = stele_codes(resolve_input(cfg.ranking_csv, 'stele ranking'), names,
                                  cfg.n_founders, cfg.seed)
    ckpt = torch.load(resolve_input(cfg.split_checkpoint, 'split checkpoint'),
                      map_location='cpu', weights_only=False)
    held = set(ckpt['val_genotypes'])
    del ckpt
    train = np.array([m and n not in held for n, m in zip(names, measured)])
    test = np.array([m and n in held for n, m in zip(names, measured)])
    print(f"Genotypes: {len(names)}  with a measured stele: {measured.sum()}  "
          f"train: {train.sum()}  held out: {test.sum()}")
    print(f"Codes, measured genotypes: {np.bincount(codes[measured].astype(int))[1:].tolist()}")

    rows = []
    for k in cfg.block_sizes:
        matrix = with_block(snp_matrix, codes, k)
        proj = SNPProjector(founders=founders, target_variance=cfg.target_variance,
                            random_state=cfg.seed).fit(matrix, verbose=False)
        encoded = one_hot_founders(matrix, founders)
        scores = proj.transform(matrix)

        # Held-out accuracy by nearest group centre of the training genotypes.
        centres = np.stack([scores[train & (codes == c)].mean(axis=0) for c in founders])
        d = ((scores[test][:, None, :] - centres[None]) ** 2).sum(-1)
        accuracy = float((np.argmin(d, axis=1) + 1 == codes[test]).mean())

        # Flip distance, block at founder 1 against founder 8, measured on the
        # genotypes that train, against each one's nearest real neighbour.
        if k:
            lo, hi = matrix[train].copy(), matrix[train].copy()
            lo[:, :k], hi[:, :k] = founders[0], founders[-1]
            flip = np.linalg.norm(proj.transform(hi) - proj.transform(lo), axis=1)
        else:
            flip = np.zeros(int(train.sum()))
        s = scores[train]
        pair = np.linalg.norm(s[:, None, :] - s[None], axis=-1)
        np.fill_diagonal(pair, np.inf)
        nearest = pair.min(axis=1)

        width = len(founders)
        real_cols = np.arange(k * width, encoded.shape[1])
        syn_cols = np.arange(0, k * width)
        rows.append({
            'block_size': k,
            'n_components': proj.output_dim,
            'block_share_of_variance': (
                float(encoded[:, syn_cols].var(axis=0).sum() / encoded.var(axis=0).sum())
                if k else 0.0),
            'block_retained': retained(proj, encoded, scores, syn_cols) if k else float('nan'),
            'real_genome_retained': retained(proj, encoded, scores, real_cols),
            'heldout_accuracy': accuracy,
            'flip_distance': float(np.median(flip)),
            'nearest_line_distance': float(np.median(nearest)),
            'flip_over_nearest': float(np.median(flip / nearest)),
        })
        r = rows[-1]
        print(f"  K={k:<5} comps {r['n_components']:>3}  block share {r['block_share_of_variance']:.3f}  "
              f"real kept {r['real_genome_retained']:.3f}  held-out acc {accuracy:.2f}  "
              f"flip/nearest {r['flip_over_nearest']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'block_size_sweep.csv', index=False)

    base = df.loc[df['block_size'] == 0, 'real_genome_retained'].iloc[0]
    with_block_rows = df[df['block_size'] > 0]
    best = with_block_rows['heldout_accuracy'].max()
    ok = with_block_rows[(with_block_rows['heldout_accuracy'] >= best - cfg.accuracy_slack)
                         & (with_block_rows['real_genome_retained'] >= base - cfg.max_retention_drop)]
    pick = int(ok['block_size'].min()) if len(ok) else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    x = with_block_rows['block_size']
    ax = axes[0]
    ax.plot(x, with_block_rows['heldout_accuracy'], 'o-', color='#1f77b4')
    ax.axhline(1 / cfg.n_founders, ls=':', color='0.5')
    ax.axhline(df.loc[df['block_size'] == 0, 'heldout_accuracy'].iloc[0], ls='--', color='0.5')
    ax.set_xscale('log'); ax.set_ylim(0, 1.02)
    ax.set_xlabel('block size (added genes)'); ax.set_ylabel('held-out group accuracy')
    ax.set_title('Can a new line be placed in its stele group?\n'
                 'dotted: chance, dashed: no block (real genome alone)')
    ax = axes[1]
    ax.plot(x, with_block_rows['flip_over_nearest'], 'o-', color='#ff7f0e')
    ax.axhline(1, ls='--', color='0.5')
    ax.set_xscale('log')
    ax.set_xlabel('block size'); ax.set_ylabel('flip distance / nearest-line distance')
    ax.set_title('How far a founder 1 -> 8 flip moves a line\n(1 = as far as its nearest real neighbour)')
    ax = axes[2]
    ax.plot(x, with_block_rows['real_genome_retained'], 'o-', color='#2ca02c', label='real genes kept')
    ax.plot(x, with_block_rows['block_share_of_variance'], 's-', color='#d62728', label='block share of variance')
    ax.axhline(base, ls='--', color='0.5')
    ax.set_xscale('log'); ax.set_ylim(0, 1.02)
    ax.set_xlabel('block size'); ax.legend(fontsize=9)
    ax.set_title('What the block costs the real genome\n(dashed: real genes kept with no block)')
    for a in axes:
        if pick:
            a.axvline(pick, color='#9467bd', lw=1.2, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out / 'block_size_sweep.png', dpi=140)
    plt.close(fig)

    summary = {'recommended_block_size': pick, 'rule': (
        f'smallest block with held-out accuracy within {cfg.accuracy_slack:.0%} of the best '
        f'({best:.2f}) and real-genome retention no more than {cfg.max_retention_drop:.0%} '
        f'below the no-block run ({base:.3f})'),
        'n_train': int(train.sum()), 'n_heldout': int(test.sum()), 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nRecommended block size: {pick}  ({summary['rule']})")
    print(f"Wrote block_size_sweep.csv, block_size_sweep.png and summary.json to {out}")


if __name__ == '__main__':
    main()
