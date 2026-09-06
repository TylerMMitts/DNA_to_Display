# Do genotypes contribute more distinctly under the one-hot encoder?
#
# The conditioning-strength test asks whether the genotype moves the generated
# IMAGE. This asks the upstream question: whether the genotypes are distinct
# from each other in the signal the UNet is handed in the first place. A model
# whose genotypes all produce near-identical SNP embeddings cannot condition on
# them however well the UNet behaves, and that is exactly the failure mode the
# numeric encoding predicted - pure-founder vectors were provably collinear, so
# genotype differences were forced to line up along essentially one direction.
#
# Two levels are measured, both cheap (no diffusion sampling - this is all PCA
# transforms and encoder forward passes, so it runs over every genotype rather
# than a subsample):
#
# component level
#     each genotype's deviation from the population mean across PCA
#     components. This is the quantity analyze_genotype_contribution.py
#     decomposes per-genotype.
#
# embedding level
#     the encoder's actual output tokens - what cross-attention consumes.
#     Differentiation here is what genuinely limits conditioning.
#
# Metrics, chosen so they stay comparable even though the two encodings produce
# different numbers of components (150 vs 168), which rules out anything that
# needs component indices to line up:
#
# mean pairwise axis distance
#     1 - |cos| between unit-normalised profiles, so it measures whether
#     genotypes lie along different AXES rather than whether some simply have
#     larger magnitudes. Near 0 means every genotype is saying nearly the same
#     thing. The absolute value is essential rather than cosmetic - see
#     mean_pairwise_axis_distance for why plain cosine cannot detect the
#     collapse this is looking for.
#
# participation ratio
#     (sum L)^2 / sum(L^2) over the profile covariance eigenvalues - the
#     effective number of independent axes the genotypes vary along.
#     Collinearity drives this toward 1 regardless of how many components
#     exist.
#
# top-k overlap
#     mean Jaccard overlap between genotypes' top-k contributing components.
#     High overlap means the same components dominate for everyone, i.e. the
#     model is not using genotype-specific structure.
#
# Usage
#     python code/latent_diffusion/analysis/compare_genotype_contributions.py

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_NUMERIC_MODEL, DIFFUSION_ONEHOT_MODEL, RESULTS_DIR, SNP_PARQUET,
    resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.analysis.analyze_pca_sensitivity import embed_from_pca_vector


# Metrics

def unit_rows(M, eps=1e-12):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(norms, eps)


def mean_pairwise_axis_distance(profiles):
    # Mean 1 - |cos(a, b)| over all genotype pairs.
    #
    # ABSOLUTE cosine, deliberately. Profiles are centred, so a set of collinear
    # genotypes ends up spread along both directions of a single axis - some at
    # +d, some at -d. Plain cosine scores those two groups as maximally
    # dissimilar and returns ~1.0, indistinguishable from genuinely independent
    # profiles; measured directly, perfectly collinear synthetic data scored
    # 0.9989 against 1.0001 for independent data, so the metric was saturated and
    # could not detect the very collapse it exists to detect.
    #
    # Taking the absolute value asks whether genotypes lie along different AXES,
    # which is the right question here: the numeric encoding's defect was that
    # all founder differences were forced onto one axis, and which end of that
    # axis a genotype sat on was never the issue.
    #
    # Unit-normalising first keeps this about direction rather than magnitude -
    # otherwise one genotype with a large deviation would dominate the average.
    U = unit_rows(profiles)
    sim = np.abs(U @ U.T)
    iu = np.triu_indices(len(U), k=1)
    return float((1.0 - sim[iu]).mean()), (1.0 - sim[iu])


def participation_ratio(profiles):
    # Effective number of independent axes the genotypes vary along.
    #
    # Computed from the eigenvalues of the centred profile covariance. A set of
    # perfectly collinear profiles scores 1.0 no matter how many components the
    # space nominally has, which is precisely the numeric encoding's predicted
    # signature.
    X = profiles - profiles.mean(axis=0, keepdims=True)
    # Gram matrix over genotypes is smaller than the component covariance when
    # n_genotypes < n_components and has the same non-zero eigenvalues.
    C = X @ X.T
    eig = np.linalg.eigvalsh(C)
    eig = np.clip(eig, 0, None)
    total = eig.sum()
    if total <= 0:
        return float('nan')
    return float((total ** 2) / (eig ** 2).sum())


def mean_topk_overlap(profiles, k=10):
    # Mean Jaccard overlap of genotypes' top-k |contribution| components.
    tops = [set(np.argsort(np.abs(row))[::-1][:k]) for row in profiles]
    vals = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    return float(np.mean(vals)) if vals else float('nan')


def profile_metrics(profiles, top_k):
    cos_mean, cos_all = mean_pairwise_axis_distance(profiles)
    pr = participation_ratio(profiles)
    overlap = mean_topk_overlap(profiles, top_k)
    dims = int(profiles.shape[1])

    # Both raw numbers are bounded by the dimensionality, and the two encodings
    # do NOT have the same number of components (150 vs 168), so comparing them
    # raw partly just compares component counts. Normalising was not cosmetic
    # here: with a 50-component numeric fit the raw participation ratio looked
    # far worse than one-hot's while the NORMALISED values were nearly equal
    # (0.736 vs 0.719), and at the real 150 components the ranking is genuine
    # (0.411 vs 0.719). Report both so that trap is visible rather than latent.
    return {
        'mean_pairwise_axis_distance': cos_mean,
        'participation_ratio': pr,
        # Fraction of available axes the genotypes actually spread across.
        'participation_ratio_normalized': pr / dims if dims else float('nan'),
        'mean_topk_overlap': overlap,
        # Overlap relative to what independent profiles would give by chance
        # (roughly top_k / dims). ~1.0 means no component subset dominates;
        # >1 means the same handful leads for every genotype.
        'topk_overlap_vs_chance': overlap * dims / top_k if top_k else float('nan'),
        'n_dimensions': dims,
    }, cos_all


# Figures

def save_contribution_heatmaps(profiles_by_label, labels, genotypes, top_k,
                               save_path):
    # The genotype x component contribution heatmap, per model.
    #
    # Rows are genotypes, columns the top-variance components. A model using
    # genotype-specific structure shows a mottled, row-varying pattern; one that
    # has collapsed genotype differences onto a single axis shows near-identical
    # rows differing only in overall intensity.
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 5.6))
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        P = profiles_by_label[label]
        # Rank columns by how much they vary ACROSS genotypes - a component
        # every genotype scores identically on carries no genotype information
        # however large its absolute value.
        var_order = np.argsort(P.var(axis=0))[::-1][:top_k]
        sub = unit_rows(P)[:, var_order]

        lim = np.abs(sub).max()
        im = ax.imshow(sub, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                       interpolation='nearest')
        ax.set_title(f'{label}\ngenotype x component contribution '
                     f'(top {len(var_order)} by across-genotype variance)',
                     fontsize=10)
        ax.set_xlabel('component (re-ranked)')
        ax.set_ylabel('genotype')
        if len(genotypes) <= 40:
            ax.set_yticks(range(len(genotypes)))
            ax.set_yticklabels(genotypes, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Row-varying pattern = genotypes contributing distinctly.   '
                 'Near-identical rows = genotype differences collapsed onto one axis.',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_metric_figure(stats, labels, cos_by_label, level, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = ['#888888', '#2b7bba'][:len(labels)]
    x = np.arange(len(labels))

    ax = axes[0]
    for label, c in zip(labels, colors):
        ax.hist(cos_by_label[label], bins=40, alpha=0.6, label=label, color=c)
    ax.set_xlabel('1 - |cosine similarity| between two genotypes')
    ax.set_ylabel('genotype pairs')
    ax.set_title('How differently do genotypes contribute?\n'
                 'mass near 0 = all genotypes saying the same thing', fontsize=10)
    ax.legend(fontsize=8)

    for ax, key, title, note in (
            # Normalised versions plotted, not raw: the encodings have
            # different component counts, so raw bars would partly just be
            # comparing dimensionality.
            (axes[1], 'participation_ratio_normalized',
             'Participation ratio / dims',
             'fraction of axes genotypes spread across\n(higher = less collapsed)'),
            (axes[2], 'topk_overlap_vs_chance', 'Top-k overlap vs chance',
             'shared dominant components\n(1.0 = no subset dominates)')):
        vals = [stats[l][key] for l in labels]
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f'{title}\n{note}', fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=10)

    fig.suptitle(f'{level} level', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_founder_figure(founder_df, save_path):
    # Which founders drive contribution, for the one-hot model.
    #
    # Has no numeric-encoding counterpart by construction: with one column per
    # locus there was no per-founder quantity to report at all. This is new
    # information the encoding change makes available, not a comparison.
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    pivot = founder_df.groupby('founder')['abs_contribution'].sum()
    ax.bar([str(f) for f in pivot.index], pivot.to_numpy(), color='#2b7bba')
    ax.set_xlabel('founder')
    ax.set_ylabel('summed |contribution| over top loci')
    ax.set_title('Which founders drive the top contributing loci\n'
                 '(one-hot model only - numeric encoding could not express this)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/compare_genotype_contributions.py
    class cfg:
        checkpoints = {
            'numeric (old)': DIFFUSION_NUMERIC_MODEL,
            'one-hot (new)': DIFFUSION_ONEHOT_MODEL,
        }
        snp_parquet = SNP_PARQUET
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'    # legacy path only
        output_dir = RESULTS_DIR / 'genotype_contribution_comparison'

        # None -> every genotype. Affordable here because nothing in this
        # script runs the diffusion sampler; it is PCA transforms and encoder
        # forwards only.
        n_genotypes = None
        genotype_seed = 0

        top_k = 10               # for the top-k overlap metric
        heatmap_components = 40  # columns shown in the heatmap figure

        # Loci to pull per genotype for the founder-attribution summary.
        founder_top_loci = 50
        founder_components = 5   # components to draw those loci from

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    if cfg.n_genotypes is None:
        idx = np.arange(len(sample_names))
    else:
        rng = np.random.default_rng(cfg.genotype_seed)
        idx = np.sort(rng.choice(len(sample_names),
                                 size=min(cfg.n_genotypes, len(sample_names)),
                                 replace=False))
    genotypes = [sample_names[i] for i in idx]
    print(f"\nGenotypes: {len(genotypes)}")

    component_profiles, embedding_profiles = {}, {}
    founder_rows = []
    used_labels = []

    for label, ckpt_rel in cfg.checkpoints.items():
        print(f"\n\n{label}\n")
        try:
            ckpt_path = resolve_input(ckpt_rel, f"checkpoint for '{label}'")
        except FileNotFoundError as exc:
            print(f"SKIPPING: {exc}")
            continue

        snp_encoder, unet, unet_cfg = load_model(
            ckpt_path, snp_matrix, device,
            pca_cache=str(resolve_output(cfg.pca_cache)))
        if snp_encoder.pca is None:
            print("SKIPPING: this checkpoint's encoder has no PCA stage.")
            continue
        pca = snp_encoder.pca
        if hasattr(pca, 'ensure_explained_variance'):
            pca.ensure_explained_variance(snp_matrix)

        # component level
        scores = pca.transform(snp_matrix[idx])              # [G, K]
        comp_profile = scores - scores.mean(axis=0, keepdims=True)
        component_profiles[label] = comp_profile

        # embedding level
        with torch.no_grad():
            emb = embed_from_pca_vector(
                snp_encoder,
                torch.tensor(scores, dtype=torch.float32, device=device))
            emb = emb.flatten(1).cpu().numpy()               # [G, tokens*dim]
        embedding_profiles[label] = emb - emb.mean(axis=0, keepdims=True)

        print(f"  components: {comp_profile.shape[1]}   "
              f"embedding dims: {emb.shape[1]}")

        # founder attribution (one-hot only)
        if hasattr(pca, 'locus_contributions'):
            for gi, g in zip(idx, genotypes):
                for c in range(min(cfg.founder_components, comp_profile.shape[1])):
                    per_locus, founder_of, _ = pca.locus_contributions(
                        c, snp_matrix[gi])
                    order = np.argsort(np.abs(per_locus))[::-1][:cfg.founder_top_loci]
                    for i in order:
                        if founder_of[i] < 0:
                            continue         # missing call; no founder to attribute to
                        founder_rows.append({
                            'genotype': g, 'component': c,
                            'snp': snp_names[i], 'founder': int(founder_of[i]),
                            'contribution': float(per_locus[i]),
                            'abs_contribution': float(abs(per_locus[i])),
                        })
        used_labels.append(label)

    if not used_labels:
        raise SystemExit("No checkpoints could be loaded - check cfg.checkpoints.")

    # metrics
    results = {}
    for level, profiles in (('component', component_profiles),
                            ('embedding', embedding_profiles)):
        stats, cos_by_label = {}, {}
        for label in used_labels:
            m, cos_all = profile_metrics(profiles[label], cfg.top_k)
            stats[label] = m
            cos_by_label[label] = cos_all
        results[level] = stats

        save_metric_figure(stats, used_labels, cos_by_label, level,
                           out / f'{level}_level_metrics.png')

    save_contribution_heatmaps(component_profiles, used_labels, genotypes,
                               cfg.heatmap_components,
                               out / 'contribution_heatmaps.png')

    if founder_rows:
        founder_df = pd.DataFrame(founder_rows)
        founder_df.to_csv(out / 'founder_attribution.csv', index=False)
        save_founder_figure(founder_df, out / 'founder_attribution.png')

    rows = []
    for level, stats in results.items():
        for label, m in stats.items():
            rows.append({'level': level, 'checkpoint': label, **m})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / 'contribution_metrics.csv', index=False)
    with open(out / 'contribution_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # report
    for level in ('component', 'embedding'):
        print(f"\n\n{level.upper()} LEVEL\n")
        print(f"{'checkpoint':<20}{'axis dist':>10}{'PR':>9}{'PR/dims':>9}"
              f"{'topk':>8}{'topk/chance':>13}{'dims':>7}")
        for label in used_labels:
            m = results[level][label]
            print(f"{label:<20}{m['mean_pairwise_axis_distance']:>10.4f}"
                  f"{m['participation_ratio']:>9.2f}"
                  f"{m['participation_ratio_normalized']:>9.3f}"
                  f"{m['mean_topk_overlap']:>8.3f}"
                  f"{m['topk_overlap_vs_chance']:>13.2f}{m['n_dimensions']:>7}")

        if len(used_labels) == 2:
            a, b = used_labels
            ma, mb = results[level][a], results[level][b]
            d_cos = ma['mean_pairwise_axis_distance']
            if d_cos > 0:
                pct = (mb['mean_pairwise_axis_distance'] - d_cos) / d_cos * 100
                print(f"\n  {b} vs {a}: {pct:+.1f}% pairwise axis distance")
            # Normalised, because the two encodings have different component
            # counts and the raw ratio partly just reflects that.
            na, nb = (ma['participation_ratio_normalized'],
                      mb['participation_ratio_normalized'])
            if np.isfinite(na) and na > 0:
                print(f"  participation ratio / dims {na:.3f} -> {nb:.3f} "
                      f"({(nb - na) / na * 100:+.1f}%)   "
                      f"[raw {ma['participation_ratio']:.1f} -> "
                      f"{mb['participation_ratio']:.1f}]")

    print("\n  Reading these: higher axis distance and higher participation")
    print("  ratio both mean genotypes are contributing more distinctly. Lower")
    print("  top-k overlap means the same handful of components no longer")
    print("  dominates every genotype. The embedding level is the one that")
    print("  bounds how much conditioning can reach the UNet.")

    if founder_rows:
        print(f"\n  founder_attribution.csv: which founder drives each top locus -")
        print(f"  a quantity the numeric encoding had no way to represent.")

    print(f"\nWrote contribution_metrics.csv/.json, contribution_heatmaps.png,")
    print(f"component_level_metrics.png, embedding_level_metrics.png"
          f"{', founder_attribution.*' if founder_rows else ''} to {out}")


if __name__ == '__main__':
    main()
