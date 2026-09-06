# Does treating founder codes (1-8) as numbers distort genetic similarity?
#
# SNP values in this dataset name which of 8 MexiMAGIC founders a locus was
# inherited from - an unordered category. Everything downstream (PCA, the raw
# correlation fallback in analyze_snp_attention.py's load_similarity, and the
# SNP encoder's input) is computed on those codes as literal numbers, which
# imposes a false ordinal structure: founders 4 and 5 are treated as barely
# different, founders 1 and 8 as maximally different, for no biological reason.
# This was worked out analytically from the parent-archetype test
# (generate_parent_ archetypes.py) - the 8 pure-founder vectors are exactly
# collinear under PCA, which is only possible because of this numeric encoding.
#
# This script checks how much that actually distorts genetic similarity between
# REAL genotypes, not synthetic ones, by comparing two ways of measuring how
# alike two genotypes are:
#
# categorical similarity (the correct one)
#     Fraction of loci where two genotypes inherited from the SAME founder,
#     regardless of which one. "Different founder" counts as equally different
#     no matter which two founders are involved - founders 4 vs 5 count exactly
#     like 1 vs 8.
#
# magnitude similarity (what the pipeline currently computes)
#     Pearson correlation of the raw numeric codes, and the same thing again
#     after PCA projection (the encoder's actual input). Both let a large
#     numeric gap between founder codes inflate the apparent genetic distance.
#
# If these two measures agree closely across all genotype pairs, the numeric
# encoding is a theoretical concern but not a practical one - the specific
# founder numbering happened not to distort things much. If they diverge, the
# model has been learning from a genetic-similarity structure that does not
# match the real one, and that is worth fixing before trusting any result that
# depends on kinship, PCA components, or SNP-driven attention.
#
# The existing kinship_matrix.csv (used elsewhere in this project, e.g. to pick
# "similar genotype" groups in analyze_snp_attention.py) is scored the same
# way, which answers a second, higher-stakes question: was that file, and
# everything built on top of it this session, already using a distorted notion
# of similarity?
#
# Usage
#     python code/latent_diffusion/validation/validate_founder_encoding.py

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    KINSHIP_MATRIX, RESULTS_DIR, SNP_PARQUET, resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import (
    load_snp_data_from_parquet, build_snp_encoder_with_optimal_pca,
)


# Similarity matrices

def categorical_similarity(snp_matrix, parents):
    # Fraction of loci sharing the same founder - the correct treatment of an
    # unordered category.
    #
    # Vectorised as a sum over founders of indicator-matrix products rather than
    # a per-pair loop: for founder k, (X == k) is a 0/1 indicator matrix, and
    # indicator @ indicator.T counts, for every genotype pair, how many loci both
    # have founder k. Summing that over all 8 founders and dividing by locus
    # count gives the fraction of loci where the pair matches, in a handful of
    # matrix multiplications instead of a loop over ~20,000 genotype pairs.
    n_loci = snp_matrix.shape[1]
    sim = np.zeros((snp_matrix.shape[0], snp_matrix.shape[0]), dtype=np.float64)
    for k in parents:
        indicator = (snp_matrix == k).astype(np.float64)
        sim += indicator @ indicator.T
    return sim / n_loci


def magnitude_similarity(snp_matrix):
    # Pearson correlation of the raw numeric codes - the fallback already used
    # by load_similarity() in analyze_snp_attention.py when no kinship file is
    # available.
    return np.corrcoef(snp_matrix)


def pca_space_similarity(snp_matrix, pca):
    # Pearson correlation of PCA-projected genotype vectors - the actual
    # representation the SNP encoder's first layer receives.
    scores = pca.transform(snp_matrix)
    return np.corrcoef(scores)


def load_kinship_aligned(kinship_path, sample_names):
    df = pd.read_csv(kinship_path, index_col=0)
    df.index = df.index.astype(str).str.replace('_TC', '', regex=False)
    df.columns = df.columns.astype(str).str.replace('_TC', '', regex=False)

    shared = [n for n in sample_names if n in df.index and n in df.columns]
    if len(shared) < 2:
        return None, []
    return df.loc[shared, shared].to_numpy(dtype=float), shared


# Comparison

def upper_tri(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]


def compare_all(matrices):
    # Pearson and Spearman correlation between every pair of similarity
    # matrices, computed on their flattened upper triangles (one number per
    # genotype pair, diagonal and duplicate lower triangle excluded).
    flat = {name: upper_tri(m) for name, m in matrices.items()}
    names = list(flat)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = pd.DataFrame({'a': flat[a], 'b': flat[b]}).dropna()
            rows.append({
                'measure_a': a, 'measure_b': b, 'n_pairs': len(s),
                'pearson': float(s['a'].corr(s['b'])),
                'spearman': float(s['a'].corr(s['b'], method='spearman')),
            })
    return pd.DataFrame(rows)


def find_worst_disagreements(cat_sim, mag_sim, names, top_n=15):
    # Genotype pairs ranked very differently by the two measures.
    #
    # Rank (not raw value) is compared, since the two measures live on different
    # scales (fraction-matching vs. correlation) and only their ordering is
    # meaningful to compare directly.
    n = cat_sim.shape[0]
    iu = np.triu_indices(n, k=1)
    cat_rank = pd.Series(cat_sim[iu]).rank()
    mag_rank = pd.Series(mag_sim[iu]).rank()

    df = pd.DataFrame({
        'genotype_a': [names[i] for i in iu[0]],
        'genotype_b': [names[j] for j in iu[1]],
        'categorical_similarity': cat_sim[iu],
        'magnitude_similarity': mag_sim[iu],
        'categorical_rank': cat_rank.to_numpy(),
        'magnitude_rank': mag_rank.to_numpy(),
    })
    df['rank_gap'] = (df['categorical_rank'] - df['magnitude_rank']).abs()
    return df.sort_values('rank_gap', ascending=False).head(top_n)


def founder_pair_prevalence(snp_matrix, parents):
    # How often each unordered founder pair segregates in the real data,
    # estimated from marginal founder frequencies (freq(i) * freq(j) * 2), and
    # the numeric penalty the magnitude measure assigns to that pair (|i - j|)
    # versus the constant penalty a categorical measure assigns (1).
    #
    # The exact per-locus, per-genotype-pair computation is skipped in favour of
    # this estimate because it would mean checking every locus against every
    # genotype pair - about 870 million comparisons - for a number the marginal
    # frequencies already give directly, since founder assignment is
    # independent of position to first order in a MAGIC population.
    freq = {k: float((snp_matrix == k).mean()) for k in parents}
    rows = []
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            a, b = parents[i], parents[j]
            rows.append({
                'founder_a': a, 'founder_b': b,
                'numeric_gap': abs(a - b),
                'estimated_prevalence': 2 * freq[a] * freq[b],
            })
    return pd.DataFrame(rows)


# Figures

def save_scatter_grid(cat_sim, others, save_path):
    fig, axes = plt.subplots(1, len(others), figsize=(5.2 * len(others), 4.8),
                             squeeze=False)
    cat_flat = upper_tri(cat_sim)

    for ax, (name, mat) in zip(axes[0], others.items()):
        other_flat = upper_tri(mat)
        good = np.isfinite(cat_flat) & np.isfinite(other_flat)
        ax.scatter(cat_flat[good], other_flat[good], s=6, alpha=0.25,
                   edgecolor='none', color='steelblue')
        r = float(pd.Series(cat_flat[good]).corr(pd.Series(other_flat[good])))
        ax.set_xlabel('categorical similarity (correct)')
        ax.set_ylabel(f'{name}')
        ax.set_title(f'{name}\nPearson r = {r:.2f}', fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_prevalence_figure(prevalence_df, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(prevalence_df['numeric_gap'], prevalence_df['estimated_prevalence'] * 100,
               s=50, color='darkorange')
    for _, r in prevalence_df.iterrows():
        ax.annotate(f"{int(r['founder_a'])}-{int(r['founder_b'])}",
                    (r['numeric_gap'], r['estimated_prevalence'] * 100),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('numeric gap penalised by the magnitude measure  |i - j|')
    ax.set_ylabel('estimated share of loci where this founder pair segregates (%)')
    ax.set_title('Do heavily-penalised founder pairs actually occur less often?\n'
                 '(flat = no, the distortion is not naturally diluted by rarity)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_correlation_heatmap(compare_df, save_path):
    measures = sorted(set(compare_df['measure_a']) | set(compare_df['measure_b']))
    mat = pd.DataFrame(np.eye(len(measures)), index=measures, columns=measures)
    for _, r in compare_df.iterrows():
        mat.loc[r['measure_a'], r['measure_b']] = r['pearson']
        mat.loc[r['measure_b'], r['measure_a']] = r['pearson']

    fig, ax = plt.subplots(figsize=(1.4 * len(measures) + 2, 1.4 * len(measures) + 1))
    im = ax.imshow(mat.to_numpy(), cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(measures))); ax.set_xticklabels(measures, rotation=30, ha='right')
    ax.set_yticks(range(len(measures))); ax.set_yticklabels(measures)
    for i in range(len(measures)):
        for j in range(len(measures)):
            ax.text(j, i, f'{mat.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label='Pearson r between similarity measures')
    ax.set_title('Agreement between similarity measures')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/validation/validate_founder_encoding.py
    class cfg:
        snp_parquet = SNP_PARQUET
        kinship_path = KINSHIP_MATRIX   # None to skip
        output_dir = RESULTS_DIR / 'founder_encoding_validation'

        parents = None            # None -> detected from unique SNP values
        pca_target_variance = 0.95
        top_disagreements = 15
        seed = 0

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    parents = cfg.parents
    if parents is None:
        parents = sorted(int(v) for v in np.unique(snp_matrix) if v > 0)
    print(f"Founders: {parents}  ({len(sample_names)} genotypes, "
          f"{snp_matrix.shape[1]} loci)")

    # fit the same PCA the real training pipeline would fit
    print(f"\nFitting PCA to {cfg.pca_target_variance:.0%} explained variance "
          f"(matches build_snp_encoder_with_optimal_pca in training)...")
    _, pca, n_components = build_snp_encoder_with_optimal_pca(
        snp_matrix, target_variance=cfg.pca_target_variance)
    print(f"  {n_components} components")

    # similarity matrices
    print("\nComputing categorical similarity (correct treatment)...")
    cat_sim = categorical_similarity(snp_matrix, parents)
    print("Computing magnitude similarity (raw numeric correlation)...")
    mag_sim = magnitude_similarity(snp_matrix)
    print("Computing PCA-space similarity (the encoder's actual input)...")
    pca_sim = pca_space_similarity(snp_matrix, pca)

    matrices = {'categorical': cat_sim, 'magnitude_raw': mag_sim,
               'magnitude_pca': pca_sim}

    kinship_sim, kinship_names = None, []
    if cfg.kinship_path:
        try:
            kpath = resolve_input(cfg.kinship_path, 'kinship matrix')
            kinship_sim, kinship_names = load_kinship_aligned(kpath, sample_names)
        except FileNotFoundError as exc:
            print(f"\n{exc}\nSkipping kinship comparison.")

        if kinship_sim is not None:
            print(f"Kinship matrix matched: {len(kinship_names)}/{len(sample_names)} genotypes")
            idx = [sample_names.index(n) for n in kinship_names]
            matrices = {name: m[np.ix_(idx, idx)] for name, m in matrices.items()}
            matrices['kinship_matrix.csv'] = kinship_sim
            names_used = kinship_names
        else:
            names_used = sample_names
    else:
        names_used = sample_names

    # compare
    compare_df = compare_all(matrices)
    compare_df.to_csv(out / 'measure_agreement.csv', index=False)
    save_correlation_heatmap(compare_df, out / 'agreement_heatmap.png')

    others = {k: v for k, v in matrices.items() if k != 'categorical'}
    save_scatter_grid(matrices['categorical'], others, out / 'agreement_scatter.png')

    worst = find_worst_disagreements(matrices['categorical'], matrices['magnitude_raw'],
                                     names_used, cfg.top_disagreements)
    worst.to_csv(out / 'worst_disagreements.csv', index=False)

    prevalence_df = founder_pair_prevalence(snp_matrix, parents)
    prevalence_df.to_csv(out / 'founder_pair_prevalence.csv', index=False)
    save_prevalence_figure(prevalence_df, out / 'founder_pair_prevalence.png')
    prevalence_corr = float(prevalence_df['numeric_gap'].corr(
        prevalence_df['estimated_prevalence']))

    # report
    print(f"\n=== Agreement between similarity measures ({len(names_used)} genotypes) ===")
    cat_rows = compare_df[(compare_df.measure_a == 'categorical') |
                          (compare_df.measure_b == 'categorical')]
    for _, r in cat_rows.iterrows():
        other = r['measure_b'] if r['measure_a'] == 'categorical' else r['measure_a']
        print(f"  categorical vs {other:<20}  Pearson {r['pearson']:>+.3f}   "
              f"Spearman {r['spearman']:>+.3f}")

    print("\n  Pearson r near 1.0: the numeric encoding preserves real genetic")
    print("  relationships well in practice, despite the theoretical concern.")
    print("  Pearson r well below 1.0: the encoding is materially reshaping which")
    print("  genotypes look 'similar' to the model, including to PCA and to")
    print("  whatever downstream analysis used kinship_matrix.csv this session.")

    print(f"\n=== Founder-pair prevalence vs. numeric penalty ===")
    print(f"  correlation(|i-j| penalty, real prevalence) = {prevalence_corr:+.3f}")
    print("  Nowhere near -1 or +1 means heavily-penalised founder pairs (e.g. 1 vs 8)")
    print("  occur just as often in the real data as lightly-penalised ones (e.g. 4")
    print("  vs 5) - the distortion is spread uniformly across the genome, not")
    print("  concentrated in a few rare loci that would matter less.")

    print(f"\n=== Top {len(worst)} genotype pairs where the two measures disagree most ===")
    print(worst[['genotype_a', 'genotype_b', 'categorical_similarity',
                'magnitude_similarity', 'rank_gap']].to_string(index=False,
                float_format=lambda v: f'{v:.3f}'))

    summary = {
        'n_genotypes': len(names_used), 'n_loci': int(snp_matrix.shape[1]),
        'parents': parents, 'pca_components': int(n_components),
        'measure_agreement': compare_df.to_dict('records'),
        'founder_pair_prevalence_vs_penalty_correlation': prevalence_corr,
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nWrote measure_agreement.csv, worst_disagreements.csv, "
          f"founder_pair_prevalence.csv, agreement_heatmap.png, "
          f"agreement_scatter.png, founder_pair_prevalence.png and "
          f"summary.json to {out}")


if __name__ == '__main__':
    main()
