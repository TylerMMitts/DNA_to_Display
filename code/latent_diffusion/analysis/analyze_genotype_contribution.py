# How much a whole genotype moves the output, broken down by PCA component.
#
# Answers whether the conditioning is doing anything at all, and which parts
# of a given genotype carry its effect.

import json
import pickle
import sys
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
    DIFFUSION_ONEHOT_MODEL, RESULTS_DIR, SNP_PARQUET, resolve_input,
    resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.utils import attention_analysis as aa

from latent_diffusion.analysis.analyze_snp_attention import (
    load_model, collect_attention, spearman,
)
from latent_diffusion.analysis.analyze_pca_sensitivity import (
    embed_from_pca_vector, population_sensitivity,
)


def snp_level_contribution(pca, snp_names, component, snp_vector, top_n=20):
    # Exact per-SNP decomposition of one genotype's deviation on one component.
    #
    # top_snps_for_component (in analyze_pca_sensitivity.py) ranks SNPs by
    # |loading| alone, which is a property of the component - the same list for
    # every genotype, since it says nothing about which genotype is being looked
    # at. It answers "what does this component generally measure," not "which
    # SNPs, for this genotype, actually drove its score on this component."
    #
    # PCA's transform is exactly linear (component_score = (x - mean) @ loading),
    # so the deviation decomposes exactly into a sum of per-SNP terms - this is
    # not an approximation the way the encoder-level sensitivity elsewhere in
    # this pipeline is. The identity
    #
    # sum(contribution over all SNPs) == raw_deviation_on_this_component
    #
    # holds to floating-point precision, which is checked in main() as a
    # correctness test on this function, not just a plausibility check.
    if hasattr(pca, 'locus_contributions'):
        # One-hot projector: components_ has n_founders columns per locus, so a
        # per-locus term is the sum over that locus's founder slots. See
        # SNPProjector.locus_contributions - the regrouping preserves exactness,
        # and additionally reports which founder the locus carries, which the
        # numeric encoding had no way to express.
        contribution, founder_of_locus, _ = pca.locus_contributions(
            component, snp_vector)
        order = np.argsort(np.abs(contribution))[::-1][:top_n]
        top = pd.DataFrame({
            'snp': [snp_names[i] for i in order],
            'founder': founder_of_locus[order],
            'contribution': contribution[order],
        })
        return top, float(contribution.sum())

    loading = pca.components_[component]                      # [num_snps]
    genotype_deviation = snp_vector - pca.mean_                # [num_snps]
    contribution = loading * genotype_deviation                # exact per-SNP terms

    order = np.argsort(np.abs(contribution))[::-1][:top_n]
    top = pd.DataFrame({
        'snp': [snp_names[i] for i in order],
        'loading': loading[order],
        'genotype_deviation': genotype_deviation[order],
        'contribution': contribution[order],
    })
    return top, float(contribution.sum())


# Population statistics

def population_pca_stats(pca, snp_matrix):
    scores = pca.transform(snp_matrix)                 # [N, K]
    mean = scores.mean(axis=0)
    std = scores.std(axis=0)

    expected_std = np.sqrt(np.clip(pca.explained_variance_, 1e-12, None))
    check = {
        'max_abs_population_mean': float(np.abs(mean).max()),
        'mean_std_ratio_to_explained_variance': float(np.mean(std / expected_std)),
    }
    return mean, std, check


@torch.no_grad()
def population_mean_embedding(encoder, pca, snp_matrix, genotype_indices, device):
    pca_vectors = pca.transform(snp_matrix[genotype_indices])
    batch = torch.tensor(pca_vectors, dtype=torch.float32, device=device)
    return embed_from_pca_vector(encoder, batch).mean(dim=0)   # [tokens, dim]


# Per-genotype contribution

def component_contributions(pca, snp_vector, pop_mean, pop_std, sensitivity):
    pca_vector = pca.transform(snp_vector.reshape(1, -1))[0]
    raw_dev = pca_vector - pop_mean
    z = raw_dev / np.where(pop_std > 0, pop_std, 1.0)
    expected = raw_dev * sensitivity                    # signed, embedding units

    return pd.DataFrame({
        'component': np.arange(len(pca_vector)),
        'genotype_score': pca_vector,
        'population_mean': pop_mean,
        'raw_deviation': raw_dev,
        'z_score': z,
        'sensitivity': sensitivity,
        'expected_contribution': expected,
    })


@torch.no_grad()
def ablation_attention_shift(encoder, unet, pca, snp_vector, components, pop_mean,
                             latent_shape, timestep, device, seed):
    # Resets each component in turn to the population mean and measures the
    # resulting attention-map shift, holding every other component at the
    # genotype's real value. Returns {layer: {component: shift}}.
    pca_vector = pca.transform(snp_vector.reshape(1, -1))[0]
    batch = [pca_vector]
    for c in components:
        ablated = pca_vector.copy()
        ablated[c] = pop_mean[c]
        batch.append(ablated)
    batch = np.stack(batch, axis=0)

    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(1, *latent_shape, generator=generator).to(device)
    B = batch.shape[0]
    z_t = noise.repeat(B, 1, 1, 1)
    t = torch.full((B,), timestep, device=device, dtype=torch.long)

    unet.set_store_attention(False)
    unet.clear_attention_history()

    embeddings = embed_from_pca_vector(
        encoder, torch.tensor(batch, dtype=torch.float32, device=device))
    unet(z_t, t, embeddings)

    shifts = {}
    for name, block in unet.iter_cross_attn_blocks():
        if block.attention_weights is None:
            continue
        attn = block.attention_weights.cpu().numpy()     # [1+len(components), N, M]
        base = attn[0]
        shifts[name] = {c: float(np.linalg.norm(attn[1 + i] - base))
                        for i, c in enumerate(components)}
    return shifts


@torch.no_grad()
def genotype_deviation_map(encoder, unet, snp_matrix, genotype_idx, background_indices,
                           latent_shape, timestep, layer, device, seed, chunk_size=16):
    # This genotype's attention deviation at one layer, against the same
    # background-sample baseline used elsewhere in this script. A self-
    # contained version of the measurement in analyze_snp_attention.py,
    # computed here for a single genotype so this script does not depend on
    # that one's group- selection machinery.
    batch_idx = [genotype_idx] + list(background_indices)
    snp_batch = torch.tensor(snp_matrix[batch_idx], dtype=torch.float32, device=device)
    attention = collect_attention(encoder, unet, snp_batch, timestep, latent_shape,
                                  device, seed, chunk_size)
    grids = aa.attention_to_grids(attention[layer])       # [B, H, W, M]
    baseline = grids[1:].mean(axis=0, keepdims=True)
    return grids[0] - baseline[0]                          # [H, W, M]


# Figure

def save_genotype_figure(genotype, contrib_top, layer_avg_shift, deviation_map,
                         layer_name, timestep, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={'width_ratios': [1.1, 1.1, 1.0]})

    order = contrib_top.sort_values('expected_contribution').index
    ax = axes[0]
    colors = ['crimson' if v < 0 else 'steelblue'
             for v in contrib_top.loc[order, 'expected_contribution']]
    ax.barh([str(c) for c in contrib_top.loc[order, 'component']],
            contrib_top.loc[order, 'expected_contribution'], color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('expected contribution\n(embedding units, signed)')
    ax.set_title('Ranked by embedding-level\nexpected contribution')
    ax.set_ylabel('component')

    ax = axes[1]
    comps = contrib_top['component'].tolist()
    shift_vals = [layer_avg_shift.get(c, 0.0) for c in comps]
    order2 = np.argsort(shift_vals)
    ax.barh([str(comps[i]) for i in order2], [shift_vals[i] for i in order2],
            color='darkorange')
    ax.set_xlabel('attention shift\n(L2, mean over layers)')
    ax.set_title('Same components, ranked by\nablation attention shift')

    ax = axes[2]
    heat = np.abs(deviation_map).mean(axis=-1)
    im = ax.imshow(heat, cmap='inferno', interpolation='nearest')
    ax.set_title(f'{genotype}\nattention deviation, {layer_name} @ t={timestep}')
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'Genotype {genotype}: which SNP-derived components '
                 f'drive its attention pattern', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/analyze_genotype_contribution.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'genotype_contribution_onehot'
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'
        sensitivity_cache = RESULTS_DIR / 'genotype_contribution_onehot' / 'population_sensitivity.csv'

        # Genotypes to analyse. None -> a random sample of n_genotypes.
        genotypes = None
        n_genotypes = 3

        # Genotypes used to estimate population statistics, sensitivity, and
        # the deviation-map baseline. Reused across all three for consistency.
        background_size = 32
        alpha = 1.0              # perturbation size (population std units) for sensitivity

        top_k = 10               # components to rank, visualise, and pull SNPs for
        top_snps_per_component = 15

        attention_timestep = 500
        deviation_layer = 'up_2'
        latent_size = 32
        chunk_size = 16
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out_root = resolve_output(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out_root}")

    parquet_path = resolve_input(cfg.snp_parquet, 'SNP parquet')
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(parquet_path)
    snp_matrix = np.asarray(snp_matrix)
    name_to_row = {n: i for i, n in enumerate(sample_names)}

    checkpoint_path = resolve_input(cfg.checkpoint, 'checkpoint')
    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    if snp_encoder.pca is None:
        raise SystemExit("This checkpoint's SNP encoder was not trained with PCA.")
    pca = snp_encoder.pca
    # One-hot projectors restored from a checkpoint written before
    # explained_variance_ was saved need it recovered from the population;
    # the sensitivity analysis scales its perturbations by component standard
    # deviation and would otherwise fail on a missing attribute.
    if hasattr(pca, 'ensure_explained_variance'):
        pca.ensure_explained_variance(snp_matrix)
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    rng = np.random.default_rng(cfg.seed)

    if cfg.genotypes:
        missing = [g for g in cfg.genotypes if g not in name_to_row]
        if missing:
            raise SystemExit(f"genotypes not found: {missing}")
        target_names = list(cfg.genotypes)
    else:
        target_names = list(rng.choice(sample_names, size=cfg.n_genotypes, replace=False))

    pool = [n for n in sample_names if n not in target_names]
    n_bg = min(cfg.background_size, len(pool))
    background_names = list(rng.choice(pool, size=n_bg, replace=False))
    background_indices = [name_to_row[n] for n in background_names]
    print(f"Targets: {target_names}\nBackground sample: {n_bg} genotypes")

    # population statistics
    pop_mean, pop_std, check = population_pca_stats(pca, snp_matrix)
    print(f"\nPopulation PCA stats sanity check: {check}")
    print("  (max_abs_population_mean should be near 0; std ratio should be near 1;")
    print("   large deviations mean this PCA does not match the population passed in)")

    pop_embedding = population_mean_embedding(snp_encoder, pca, snp_matrix,
                                              background_indices, device)

    # population-averaged sensitivity, cached across runs
    sens_cache_path = resolve_output(cfg.sensitivity_cache)
    sensitivity_df = None
    if sens_cache_path.exists():
        cached = pd.read_csv(sens_cache_path)
        if len(cached) == len(pop_mean):
            sensitivity_df = cached
            print(f"\nLoaded cached population sensitivity from {sens_cache_path}")
    if sensitivity_df is None:
        print("\nComputing population-averaged component sensitivity "
              f"(over {n_bg} background genotypes)...")
        sensitivity_df = population_sensitivity(
            snp_encoder, pca, snp_matrix, background_indices, device, alpha=cfg.alpha)
        sens_cache_path.parent.mkdir(parents=True, exist_ok=True)
        sensitivity_df.to_csv(sens_cache_path, index=False)

    sensitivity = sensitivity_df.set_index('component')['normalized_sensitivity_mean'] \
        .reindex(range(len(pop_mean))).to_numpy()

    # per genotype
    all_summaries = {}
    for genotype in target_names:
        print(f"\n=== {genotype} ===")
        idx = name_to_row[genotype]
        snp_vector = snp_matrix[idx]

        contrib = component_contributions(pca, snp_vector, pop_mean, pop_std, sensitivity)
        contrib = contrib.sort_values('expected_contribution', key=np.abs, ascending=False)
        contrib.to_csv(out_root / f'{genotype}_component_contributions.csv', index=False)

        top = contrib.head(cfg.top_k).copy()
        top_components = top['component'].tolist()

        print(f"Top {cfg.top_k} components by expected contribution:")
        for _, row in top.iterrows():
            print(f"  component {int(row['component']):>4}  z={row['z_score']:+.2f}  "
                  f"sensitivity={row['sensitivity']:.4f}  "
                  f"expected_contribution={row['expected_contribution']:+.4f}")

        # sanity check: sum of |per-component terms| vs. the genotype's real
        # total embedding deviation, computed directly (no linear approximation)
        pca_vector = pca.transform(snp_vector.reshape(1, -1))[0]
        real_embedding = embed_from_pca_vector(
            snp_encoder, torch.tensor(pca_vector, dtype=torch.float32,
                                      device=device).unsqueeze(0))[0]
        real_deviation_magnitude = float((real_embedding - pop_embedding).norm())
        sum_abs_contribution = float(contrib['expected_contribution'].abs().sum())
        linearity_ratio = (sum_abs_contribution / real_deviation_magnitude
                           if real_deviation_magnitude > 0 else float('nan'))
        print(f"  sanity check: sum|expected contributions| = {sum_abs_contribution:.3f}, "
              f"real embedding deviation = {real_deviation_magnitude:.3f}, "
              f"ratio = {linearity_ratio:.2f}")
        print("  (this is a rough plausibility check, not an exact decomposition -")
        print("   component interactions in the encoder's nonlinear layers are not")
        print("   captured by summing independent single-component terms)")

        shifts = ablation_attention_shift(
            snp_encoder, unet, pca, snp_vector, top_components, pop_mean,
            latent_shape, cfg.attention_timestep, device, cfg.seed)
        shift_rows = [{'genotype': genotype, 'component': c, 'layer': layer,
                       'attention_shift': shifts[layer][c]}
                      for layer in shifts for c in top_components]
        shift_df = pd.DataFrame(shift_rows)
        shift_df.to_csv(out_root / f'{genotype}_ablation_attention_shift.csv', index=False)

        layer_avg_shift = shift_df.groupby('component')['attention_shift'].mean().to_dict()
        rank_corr = spearman(top['expected_contribution'].abs(),
                             [layer_avg_shift.get(c, 0.0) for c in top_components])
        print(f"  Spearman(embedding-level rank, attention-level rank) = {rank_corr:.3f}")

        deviation_map = genotype_deviation_map(
            snp_encoder, unet, snp_matrix, idx, background_indices, latent_shape,
            cfg.attention_timestep, cfg.deviation_layer, device, cfg.seed, cfg.chunk_size)

        save_genotype_figure(genotype, top, layer_avg_shift, deviation_map,
                             cfg.deviation_layer, cfg.attention_timestep,
                             out_root / f'{genotype}_summary.png')

        loadings_dir = out_root / f'{genotype}_snp_loadings'
        loadings_dir.mkdir(parents=True, exist_ok=True)
        raw_dev_by_component = contrib.set_index('component')['raw_deviation'].to_dict()
        for c in top_components[:5]:
            snps, contribution_sum = snp_level_contribution(
                pca, snp_names, c, snp_vector, cfg.top_snps_per_component)
            snps.to_csv(loadings_dir / f'component_{c}.csv', index=False)

            # sum(contribution) over ALL snps should equal this component's raw
            # deviation exactly, since PCA's transform is exactly linear. This
            # is a correctness check on snp_level_contribution itself, not a
            # rough plausibility check like the embedding-level one above.
            expected = raw_dev_by_component[c]
            if not np.isclose(contribution_sum, expected, rtol=1e-3, atol=1e-6):
                print(f"  WARNING: component {c} SNP-level contributions sum to "
                      f"{contribution_sum:.6f}, expected {expected:.6f} - "
                      f"snp_level_contribution disagrees with the PCA transform")

        all_summaries[genotype] = {
            'top_components': top_components,
            'sum_abs_expected_contribution': sum_abs_contribution,
            'real_embedding_deviation': real_deviation_magnitude,
            'linearity_ratio': linearity_ratio,
            'embedding_vs_attention_rank_correlation': rank_corr,
        }

    with open(out_root / 'summary.json', 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'background_size': n_bg,
            'population_stats_check': check,
            'genotypes': all_summaries,
        }, f, indent=2)

    print(f"\nWrote per-genotype CSVs, figures, and SNP loading tables to {out_root}")


if __name__ == '__main__':
    main()
