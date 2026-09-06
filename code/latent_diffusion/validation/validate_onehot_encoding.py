# Checks that one-hot encoding preserves genetic similarity, and that it
# leaves the eight founders mutually equidistant as unordered labels should
# be. Run before committing to a retrain.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import RESULTS_DIR, SNP_PARQUET, resolve_input, resolve_output

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.models.snp_encoding import SNPProjector, one_hot_founders
from latent_diffusion.validation.validate_founder_encoding import (
    categorical_similarity, magnitude_similarity, upper_tri,
)


def pairwise_distances(vectors):
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def founder_geometry(projector_or_pca, snp_matrix, founders, one_hot=True):
    # Distances between the eight pure-founder vectors in PCA space.
    #
    # Under a correct categorical encoding these should be near-identical for
    # every founder pair. A spread of distances that tracks |i - j| is the
    # collinearity artifact.
    n_loci = snp_matrix.shape[1]
    pure = np.stack([np.full(n_loci, float(k), dtype=np.float32) for k in founders])

    if one_hot:
        scores = projector_or_pca.transform(pure)
    else:
        scores = projector_or_pca.transform(pure)

    dist = pairwise_distances(scores)
    iu = np.triu_indices(len(founders), k=1)
    gaps = np.array([abs(founders[i] - founders[j]) for i, j in zip(*iu)])
    return scores, dist, dist[iu], gaps


def save_comparison_figure(cat_flat, num_flat, oh_flat, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, flat, name in ((axes[0], num_flat, 'current: numeric codes -> PCA'),
                           (axes[1], oh_flat, 'proposed: one-hot -> PCA')):
        good = np.isfinite(cat_flat) & np.isfinite(flat)
        ax.scatter(cat_flat[good], flat[good], s=6, alpha=0.25,
                   edgecolor='none', color='steelblue')
        r = float(pd.Series(cat_flat[good]).corr(pd.Series(flat[good])))
        ax.set_title(f'{name}\nPearson r = {r:.3f}', fontsize=11)
        ax.set_xlabel('categorical similarity (correct)')
    axes[0].set_ylabel('PCA-space similarity')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_founder_geometry_figure(num_dists, num_gaps, oh_dists, oh_gaps, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, dists, gaps, name in (
            (axes[0], num_dists, num_gaps, 'current: numeric codes'),
            (axes[1], oh_dists, oh_gaps, 'proposed: one-hot')):
        ax.scatter(gaps, dists / dists.mean(), s=45, color='darkorange')
        ax.set_xlabel('numeric gap between founders  |i - j|')
        ax.set_ylabel('PCA distance / mean distance')
        spread = dists.std() / dists.mean()
        ax.set_title(f'{name}\nrelative spread = {spread:.3f}', fontsize=11)
        ax.axhline(1.0, color='crimson', ls='--', lw=1.2)
    fig.suptitle('Distance between pure-founder vectors\n'
                 'flat at 1.0 = all founder pairs equally distinct (correct); '
                 'sloped = collinear artifact', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/validation/validate_onehot_encoding.py
    class cfg:
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'onehot_encoding_preview'
        founders = None            # None -> detected from the data
        target_variance = 0.95
        random_state = 0

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    founders = cfg.founders
    if founders is None:
        founders = tuple(sorted(int(v) for v in np.unique(snp_matrix) if v > 0))
    print(f"Founders: {founders}   genotypes: {len(sample_names)}   "
          f"loci: {snp_matrix.shape[1]}")

    # the three similarity measures
    print("\nCategorical similarity (correct)...")
    cat_sim = categorical_similarity(snp_matrix, list(founders))

    print("Current representation: numeric codes -> PCA...")
    numeric_pca = PCA(n_components=min(snp_matrix.shape) - 1,
                      random_state=cfg.random_state).fit(snp_matrix)
    cumulative = np.cumsum(numeric_pca.explained_variance_ratio_)
    n_numeric = int(np.searchsorted(cumulative, cfg.target_variance) + 1)
    numeric_pca = PCA(n_components=n_numeric, random_state=cfg.random_state)
    numeric_scores = numeric_pca.fit_transform(snp_matrix)
    num_sim = np.corrcoef(numeric_scores)
    print(f"  {n_numeric} components")

    print("Proposed representation: one-hot -> PCA...")
    projector = SNPProjector(founders=founders, target_variance=cfg.target_variance,
                             random_state=cfg.random_state).fit(snp_matrix)
    onehot_scores = projector.transform(snp_matrix)
    oh_sim = np.corrcoef(onehot_scores)

    # compare
    cat_flat = upper_tri(cat_sim)
    num_flat = upper_tri(num_sim)
    oh_flat = upper_tri(oh_sim)

    def corr(a, b, method='pearson'):
        s = pd.DataFrame({'a': a, 'b': b}).dropna()
        return float(s['a'].corr(s['b'], method=method))

    results = {
        'numeric_pca_pearson': corr(cat_flat, num_flat),
        'numeric_pca_spearman': corr(cat_flat, num_flat, 'spearman'),
        'onehot_pca_pearson': corr(cat_flat, oh_flat),
        'onehot_pca_spearman': corr(cat_flat, oh_flat, 'spearman'),
        'numeric_components': int(n_numeric),
        'onehot_components': int(projector.output_dim),
    }

    save_comparison_figure(cat_flat, num_flat, oh_flat, out / 'similarity_comparison.png')

    # founder geometry
    print("\nChecking pure-founder geometry...")
    _, _, num_d, num_g = founder_geometry(numeric_pca, snp_matrix, founders, one_hot=False)
    _, _, oh_d, oh_g = founder_geometry(projector, snp_matrix, founders, one_hot=True)

    results['numeric_founder_distance_spread'] = float(num_d.std() / num_d.mean())
    results['onehot_founder_distance_spread'] = float(oh_d.std() / oh_d.mean())
    results['numeric_distance_vs_gap_correlation'] = float(
        pd.Series(num_d).corr(pd.Series(num_g)))
    results['onehot_distance_vs_gap_correlation'] = float(
        pd.Series(oh_d).corr(pd.Series(oh_g)))

    save_founder_geometry_figure(num_d, num_g, oh_d, oh_g,
                                 out / 'founder_geometry.png')

    pd.DataFrame([results]).to_csv(out / 'preview_results.csv', index=False)
    with open(out / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # report
    print("\n=== Genetic similarity vs. the correct categorical measure ===")
    print(f"{'representation':<32}{'Pearson':>10}{'Spearman':>11}{'components':>12}")
    print(f"{'current (numeric -> PCA)':<32}{results['numeric_pca_pearson']:>10.3f}"
          f"{results['numeric_pca_spearman']:>11.3f}{results['numeric_components']:>12}")
    print(f"{'proposed (one-hot -> PCA)':<32}{results['onehot_pca_pearson']:>10.3f}"
          f"{results['onehot_pca_spearman']:>11.3f}{results['onehot_components']:>12}")

    delta = results['onehot_pca_pearson'] - results['numeric_pca_pearson']
    print(f"\n  improvement: {delta:+.3f} Pearson")

    print("\n=== Pure-founder geometry ===")
    print(f"  current  spread of pairwise distances: "
          f"{results['numeric_founder_distance_spread']:.3f}"
          f"   corr with |i-j|: {results['numeric_distance_vs_gap_correlation']:+.3f}")
    print(f"  proposed spread of pairwise distances: "
          f"{results['onehot_founder_distance_spread']:.3f}"
          f"   corr with |i-j|: {results['onehot_distance_vs_gap_correlation']:+.3f}")
    print("  A spread near 0 with no correlation to |i-j| means all eight founders")
    print("  are mutually equidistant, which is the correct geometry for unordered")
    print("  categories and what should stop the archetypes collapsing into two")
    print("  clusters.")

    if delta > 0.2:
        print("\n  VERDICT: one-hot substantially improves the representation. "
              "The retrain is justified.")
    elif delta > 0.05:
        print("\n  VERDICT: a modest improvement. Worth doing, but expect the "
              "encoding to be one contributing factor rather than the whole story.")
    else:
        print("\n  VERDICT: little improvement - investigate before spending GPU "
              "time, since the encoding may not be the binding constraint.")

    print(f"\nWrote similarity_comparison.png, founder_geometry.png, "
          f"preview_results.csv and summary.json to {out}")


if __name__ == '__main__':
    main()
