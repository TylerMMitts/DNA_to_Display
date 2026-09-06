# Which PCA components of the SNP vector actually move the model's output.
#
# Perturbs one component at a time and measures how far the embedding and the
# attention maps shift.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_NUMERIC_MODEL, RESULTS_DIR, SNP_PARQUET, resolve_input,
    resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.utils import attention_analysis as aa

# Reuse the checkpoint-loading, path-resolution, and attention-collection
# machinery already built and tested in analyze_snp_attention.py rather than
# duplicating it.
from latent_diffusion.analysis.analyze_snp_attention import (
    load_model, collect_attention,
)


# Tier 1: encoder-level sensitivity

def embed_from_pca_vector(encoder, pca_batch):
    # Runs the SNPEncoder's network starting from PCA space directly.
    #
    # SNPEncoder.forward() expects raw SNP-space input and applies the PCA
    # transform internally. Perturbing one PCA component and re-encoding that way
    # would mean going raw -> PCA -> perturb -> inverse-PCA -> raw -> PCA again,
    # round-tripping through inverse_transform and losing precision for no
    # reason. This mirrors the back half of forward() (net -> reshape -> add
    # token_positions) starting from an already-PCA-space tensor instead.
    x = encoder.net(pca_batch)
    x = x.view(-1, encoder.num_tokens, encoder.embedding_dim)
    x = x + encoder.token_positions
    return x


@torch.no_grad()
def component_sensitivity(encoder, pca, snp_vector, device, alpha=1.0):
    # Perturbs each PCA component +-alpha standard deviations and measures the
    # resulting shift in the encoder's output embedding.
    #
    # Returns a DataFrame with one row per component: raw sensitivity (expected
    # impact of population-realistic variation) and normalized sensitivity
    # (impact per unit of variance, comparable across components).
    pca_vector = pca.transform(snp_vector.reshape(1, -1))[0]              # [K]
    K = pca_vector.shape[0]
    component_std = np.sqrt(np.clip(pca.explained_variance_, 1e-12, None))  # [K]
    deltas = alpha * component_std

    base = torch.tensor(pca_vector, dtype=torch.float32, device=device).unsqueeze(0)
    baseline_embedding = embed_from_pca_vector(encoder, base)[0]           # [tokens, dim]

    # One batched forward pass covering every component's +delta and -delta
    # perturbation, rather than 2*K separate calls.
    plus = np.tile(pca_vector, (K, 1))
    minus = np.tile(pca_vector, (K, 1))
    plus[np.arange(K), np.arange(K)] += deltas
    minus[np.arange(K), np.arange(K)] -= deltas

    perturbed = np.concatenate([plus, minus], axis=0)                      # [2K, K]
    perturbed_t = torch.tensor(perturbed, dtype=torch.float32, device=device)
    embeddings = embed_from_pca_vector(encoder, perturbed_t)               # [2K, tokens, dim]

    diffs = (embeddings - baseline_embedding.unsqueeze(0)).flatten(1)      # [2K, tokens*dim]
    magnitudes = diffs.norm(dim=1).cpu().numpy()                           # [2K]
    plus_mag, minus_mag = magnitudes[:K], magnitudes[K:]

    raw_sensitivity = 0.5 * (plus_mag + minus_mag)
    normalized_sensitivity = raw_sensitivity / deltas

    return pd.DataFrame({
        'component': np.arange(K),
        'explained_variance_ratio': pca.explained_variance_ratio_[:K],
        'component_std': component_std,
        'delta': deltas,
        'raw_sensitivity': raw_sensitivity,
        'normalized_sensitivity': normalized_sensitivity,
    })


def population_sensitivity(encoder, pca, snp_matrix, genotype_indices, device, alpha=1.0):
    # Averages component_sensitivity across a sample of genotypes.
    #
    # A single genotype's sensitivity ranking can be idiosyncratic (perturbing
    # around a specific point in a nonlinear function need not generalise).
    # Averaging over multiple genotypes gives a ranking of components the model
    # is sensitive to in general, rather than for one specific individual.
    per_genotype = []
    for idx in genotype_indices:
        df = component_sensitivity(encoder, pca, snp_matrix[idx], device, alpha=alpha)
        per_genotype.append(df.set_index('component'))

    stacked = pd.concat(per_genotype, keys=range(len(per_genotype)))
    summary = stacked.groupby('component').agg(
        explained_variance_ratio=('explained_variance_ratio', 'first'),
        component_std=('component_std', 'first'),
        raw_sensitivity_mean=('raw_sensitivity', 'mean'),
        raw_sensitivity_std=('raw_sensitivity', 'std'),
        normalized_sensitivity_mean=('normalized_sensitivity', 'mean'),
        normalized_sensitivity_std=('normalized_sensitivity', 'std'),
    ).reset_index()
    return summary


# Tier 2: does the sensitivity reach the actual attention maps?

@torch.no_grad()
def attention_shift_for_component(encoder, unet, pca, snp_vector, component,
                                  delta, latent_shape, timestep, device, seed):
    # Perturbs one component and measures the resulting change in cross-
    # attention maps, using the same shared-noise/fixed-timestep setup as
    # collect_attention() in analyze_snp_attention.py, so this is directly
    # comparable to the existing attention-heatmap analysis.
    pca_vector = pca.transform(snp_vector.reshape(1, -1))[0]
    plus, minus = pca_vector.copy(), pca_vector.copy()
    plus[component] += delta
    minus[component] -= delta

    batch = np.stack([pca_vector, plus, minus], axis=0)                   # [3, K]

    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(1, *latent_shape, generator=generator).to(device)
    z_t = noise.repeat(3, 1, 1, 1)
    t = torch.full((3,), timestep, device=device, dtype=torch.long)

    unet.set_store_attention(False)
    unet.clear_attention_history()

    embeddings = embed_from_pca_vector(
        encoder, torch.tensor(batch, dtype=torch.float32, device=device))
    unet(z_t, t, embeddings)

    shifts = {}
    for name, block in unet.iter_cross_attn_blocks():
        if block.attention_weights is None:
            continue
        attn = block.attention_weights.cpu().numpy()                      # [3, N, M]
        base, plus_attn, minus_attn = attn[0], attn[1], attn[2]
        shifts[name] = 0.5 * (np.linalg.norm(plus_attn - base) +
                              np.linalg.norm(minus_attn - base))
    return shifts


# SNP loadings for the components that matter

def top_snps_for_component(pca, snp_names, component, top_n=20):
    loadings = pca.components_[component]
    if getattr(pca, 'n_loci', None) is not None and \
            loadings.shape[0] == pca.n_loci * len(pca.founders):
        # One-hot: n_founders columns per locus. Indexing this directly by
        # locus would run off the end of snp_names and mislabel every row, so
        # the founder slots are collapsed to one number per locus first.
        #
        # Magnitude is summarised as the largest |loading| among the locus's
        # founder slots rather than their sum: the slots are mutually exclusive
        # (a genotype occupies exactly one), so the strongest single founder
        # signal is what the component can actually read off this locus, while
        # a sum would let opposite-signed slots cancel a locus that is in fact
        # highly discriminative.
        F = len(pca.founders)
        per_slot = loadings.reshape(pca.n_loci, F)
        strongest = np.abs(per_slot).max(axis=1)
        best_slot = np.abs(per_slot).argmax(axis=1)
        order = np.argsort(strongest)[::-1][:top_n]
        return pd.DataFrame({
            'snp': [snp_names[i] for i in order],
            'founder': [pca.founders[best_slot[i]] for i in order],
            'loading': [per_slot[i, best_slot[i]] for i in order],
        })

    order = np.argsort(np.abs(loadings))[::-1][:top_n]
    return pd.DataFrame({
        'snp': [snp_names[i] for i in order],
        'loading': loadings[order],
    })


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/analyze_pca_sensitivity.py
    class cfg:
        checkpoint = DIFFUSION_NUMERIC_MODEL
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'pca_sensitivity'
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'   # reuse the cached PCA if present

        # Genotypes to average sensitivity over. None -> a random sample of
        # sample_size genotypes from the full population.
        genotypes = None
        sample_size = 32
        seed = 0

        # Population-realistic perturbation size, in standard deviations of
        # each component. 1.0 = perturb by one typical unit of real variation.
        alpha = 1.0

        # Tier 2: how many top-ranked (by normalized sensitivity) components to
        # also test against the actual UNet attention maps.
        top_k_for_attention = 10
        attention_timestep = 500
        latent_size = 32

        # How many raw SNPs to report per top component.
        top_snps_per_component = 20

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out_root = resolve_output(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out_root}")

    # data
    parquet_path = resolve_input(cfg.snp_parquet, 'SNP parquet')
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(parquet_path)
    snp_matrix = np.asarray(snp_matrix)
    name_to_row = {n: i for i, n in enumerate(sample_names)}

    # model + PCA
    checkpoint_path = resolve_input(cfg.checkpoint, 'checkpoint')
    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))

    if snp_encoder.pca is None:
        raise SystemExit("This checkpoint's SNP encoder was not trained with PCA - "
                          "component sensitivity analysis has nothing to perturb.")
    pca = snp_encoder.pca
    # See the matching note in analyze_genotype_contribution.py: recovers
    # explained_variance_ for one-hot checkpoints predating that field.
    if hasattr(pca, 'ensure_explained_variance'):
        pca.ensure_explained_variance(snp_matrix)
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    # which genotypes to average sensitivity over
    if cfg.genotypes:
        missing = [g for g in cfg.genotypes if g not in name_to_row]
        if missing:
            raise SystemExit(f"genotypes not found in the SNP data: {missing}")
        genotype_names = list(cfg.genotypes)
    else:
        rng = np.random.default_rng(cfg.seed)
        n = min(cfg.sample_size, len(sample_names))
        genotype_names = list(rng.choice(sample_names, size=n, replace=False))
    genotype_indices = [name_to_row[g] for g in genotype_names]
    print(f"Averaging sensitivity over {len(genotype_names)} genotypes")

    # Tier 1: encoder-level sensitivity, every component
    print("\nTier 1: encoder sensitivity per PCA component")
    sensitivity = population_sensitivity(
        snp_encoder, pca, snp_matrix, genotype_indices, device, alpha=cfg.alpha)
    sensitivity = sensitivity.sort_values('normalized_sensitivity_mean', ascending=False)
    sensitivity.to_csv(out_root / 'component_sensitivity.csv', index=False)

    print(f"{'component':>9} {'expl.var%':>9} {'raw sens.':>10} {'norm. sens.':>11}")
    for _, row in sensitivity.head(15).iterrows():
        print(f"{int(row['component']):>9} {row['explained_variance_ratio']*100:>8.2f}% "
              f"{row['raw_sensitivity_mean']:>10.4f} {row['normalized_sensitivity_mean']:>11.4f}")

    # Sanity check on the PCA-variance confound described in the module
    # docstring: if raw and normalized rankings agree almost perfectly, the
    # "sensitivity" signal isn't adding anything beyond the PCA variance order.
    rank_corr = sensitivity[['raw_sensitivity_mean', 'normalized_sensitivity_mean']].corr(
        method='spearman').iloc[0, 1]
    print(f"\nSpearman(raw ranking, normalized ranking) = {rank_corr:.3f}  "
          f"(near 1.0 would mean normalization isn't changing the ranking)")

    # Tier 2: propagate top components to the actual attention maps
    top_components = sensitivity.head(cfg.top_k_for_attention)['component'].astype(int).tolist()
    print(f"\nTier 2: propagating top {len(top_components)} components to UNet attention "
          f"(t={cfg.attention_timestep})")

    attn_rows = []
    for genotype_idx, genotype_name in zip(genotype_indices, genotype_names):
        for component in top_components:
            delta = float(sensitivity.loc[sensitivity['component'] == component,
                                          'component_std'].iloc[0]) * cfg.alpha
            shifts = attention_shift_for_component(
                snp_encoder, unet, pca, snp_matrix[genotype_idx], component, delta,
                latent_shape, cfg.attention_timestep, device, cfg.seed)
            for layer, shift in shifts.items():
                attn_rows.append({'genotype': genotype_name, 'component': component,
                                  'layer': layer, 'attention_shift': shift})

    attn_df = pd.DataFrame(attn_rows)
    attn_df.to_csv(out_root / 'component_attention_shift.csv', index=False)

    attn_summary = attn_df.groupby('component')['attention_shift'].mean().sort_values(ascending=False)
    print("\nMean attention-map shift by component (averaged over genotypes and layers):")
    print(attn_summary.to_string())

    # SNP loadings for the top components
    print(f"\nTop {cfg.top_snps_per_component} SNPs for each top component "
          f"(by |loading|, from pca.components_):")
    loadings_dir = out_root / 'component_snp_loadings'
    loadings_dir.mkdir(parents=True, exist_ok=True)

    for component in top_components:
        snps = top_snps_for_component(pca, snp_names, component, cfg.top_snps_per_component)
        snps.to_csv(loadings_dir / f'component_{component}.csv', index=False)
        print(f"\n  component {component}:")
        for _, row in snps.head(5).iterrows():
            print(f"    {row['snp']:<20} loading={row['loading']:+.4f}")

    # summary
    summary = {
        'checkpoint': str(checkpoint_path),
        'n_genotypes_averaged': len(genotype_names),
        'alpha': cfg.alpha,
        'raw_vs_normalized_rank_correlation': float(rank_corr),
        'top_components_by_normalized_sensitivity': top_components,
        'top_components_by_attention_shift': attn_summary.head(cfg.top_k_for_attention).index.tolist(),
    }
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nFigures and CSVs written to {out_root}")


if __name__ == '__main__':
    main()
