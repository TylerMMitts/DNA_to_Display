# The most spatially DIFFERENT loci, not simply the strongest.
#
# Ranking by magnitude alone tends to return many loci that do much the same
# thing. This sweeps the genome, then picks a set whose attention maps are as
# unlike each other as possible.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_ONEHOT_MODEL, RESULTS_DIR, SNP_PARQUET, resolve_input,
    resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.analysis.analyze_pca_sensitivity import population_sensitivity
from latent_diffusion.analysis.analyze_snp_spatial_contribution import (
    locus_influence_scores, attention_grids_from_pca, population_contrast_map,
    save_snp_figure, founder_slot,
)
from latent_diffusion.analysis.rank_snp_contributions import (
    build_perturbed_batch, sweep_contributions,
)


# Stage 2: spatial signature for a candidate pool

@torch.no_grad()
def spatial_signature_sweep(encoder, unet, projector, snp_matrix, genotype_indices,
                            loci, target_slots, timestep, latent_shape, device,
                            seed, layer, batch_size):
    # Per-locus [H, W] deviation map, matching what save_snp_figure's "|mean|
    # over tokens" summary panel actually shows.
    #
    # Order of operations matters here and was gotten wrong in an earlier
    # version of this function. save_snp_figure computes
    # np.abs(m).mean(axis=-1) where m is ALREADY genotype-averaged - abs is
    # taken per token, per pixel, THEN averaged across tokens. Averaging the
    # signed delta across tokens first (mean over both the genotype and token
    # axes in one call) is not the same computation: different tokens often
    # move a given pixel in opposite directions, so a signed cross-token
    # average lets them cancel. Measured directly on real data, that
    # cancellation was not a minor rounding difference - the signed,
    # token-averaged quantity came out around 3.7e-9, smaller than the ~1.6e-7
    # floating-point noise floor that batching multiple loci together already
    # introduces (verified separately: the same locus computed alone vs as
    # part of a larger batch differs by that much, which is ordinary batched-
    # kernel float noise, not a bug). At 3.7e-9 the "signal" was actually
    # sitting below that noise floor - cosine similarity against the
    # known-correct single-locus reference came out at 0.55-0.78 instead of
    # the ~1.0 it should have been, which is what caught this.
    #
    # Taking abs() before the token-average, matching the figures, moved the
    # real signal to the ~1e-5 to 1e-4 range seen in the rendered heatmaps -
    # comfortably above that noise floor - and the same cosine check against
    # the reference then returns >=0.999997 (see the smoke test in this
    # module's development history).
    base_pca = projector.transform(snp_matrix[genotype_indices])
    base_grids = attention_grids_from_pca(
        encoder, unet, base_pca, timestep, latent_shape, device, seed, layer,
        chunk_size=batch_size)                                   # [n_g, H, W, M]

    batch = build_perturbed_batch(projector, snp_matrix, base_pca,
                                  genotype_indices, loci, target_slots)
    grids = attention_grids_from_pca(
        encoder, unet, batch, timestep, latent_shape, device, seed, layer,
        chunk_size=batch_size)                                   # [n_g*n_loci, H,W,M]

    n_g = len(genotype_indices)
    grids = grids.reshape(n_g, len(loci), *grids.shape[1:])
    delta = grids - base_grids[:, None]                          # [n_g, n_loci, H, W, M]
    per_locus = delta.mean(axis=0)                                # [n_loci, H, W, M] - signed, genotype-avg only
    return np.abs(per_locus).mean(axis=-1)                        # [n_loci, H, W] - abs BEFORE token-average


# Diversity selection

def greedy_diverse_selection(signatures, k, start_index=None):
    # Farthest-point greedy: repeatedly add whichever candidate is most
    # different from everything picked so far.
    #
    # Standard approximation for max-min diversity (k-center) selection - exact
    # is NP-hard, but greedy farthest-point is within a factor of 2 of optimal
    # and is what every practical implementation of this actually uses.
    #
    # Distance is 1 - cosine(unit-normalised, flattened maps). Returns indices
    # INTO `signatures`, in selection order (index 0 is the seed).
    n = len(signatures)
    k = min(k, n)
    flat = signatures.reshape(n, -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = flat / np.maximum(norms, 1e-12)

    if start_index is None:
        start_index = 0
    selected = [start_index]
    # min_dist[i] = distance from candidate i to its NEAREST already-selected
    # point. Maintained incrementally rather than recomputed, since
    # recomputing the full pairwise matrix every step would be O(k * n^2).
    sim_to_seed = unit @ unit[start_index]
    min_dist = 1.0 - sim_to_seed
    min_dist[start_index] = -np.inf     # never re-select

    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        sim_to_new = unit @ unit[nxt]
        min_dist = np.minimum(min_dist, 1.0 - sim_to_new)
        min_dist[nxt] = -np.inf

    return selected


# Figures

def save_distance_comparison(diverse_sigs, magnitude_top_sigs, save_path):
    # Pairwise distance matrices, diversity-selected vs plain top-N.
    #
    # This is the figure that justifies the whole exercise: if the diversity
    # selection did nothing, its matrix would look like the plain top-N
    # matrix - blocks of near-zero (dark) distance from duplicate loci in the
    # same haplotype block. If it worked, the off-diagonal should be
    # uniformly bright.
    def dist_matrix(sigs):
        n = len(sigs)
        flat = sigs.reshape(n, -1).astype(np.float64)
        unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        return 1.0 - (unit @ unit.T)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, sigs, title in ((axes[0], magnitude_top_sigs, 'plain top-N (by magnitude)'),
                            (axes[1], diverse_sigs, 'diversity-selected top-N')):
        D = dist_matrix(sigs)
        # vmax=1, not 2: signatures are non-negative (abs already taken per
        # token before this point), so cosine similarity is bounded in
        # [0, 1] and distance can never exceed 1.
        im = ax.imshow(D, cmap='viridis', vmin=0, vmax=1)
        off_diag = D[~np.eye(len(D), dtype=bool)]
        ax.set_title(f'{title}\nmean pairwise distance = {off_diag.mean():.3f}',
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Pairwise spatial-pattern distance between selected loci\n'
                 'dark blocks = near-duplicate maps (same haplotype block)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/select_diverse_snp_maps.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'snp_diverse_maps'
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'       # legacy path only
        sensitivity_cache = RESULTS_DIR / 'snp_diverse_maps' / 'population_sensitivity.csv'

        top_n = 50                # final diverse set
        # Candidate pool: the top n_pool loci by measured magnitude, before
        # diversifying. Must be >> top_n or there is nothing to diversify
        # among; must not be too close to n_loci or the magnitude floor stops
        # doing its job of keeping noise out.
        n_pool = 1000

        n_sweep_genotypes = 4     # for the full-genome magnitude sweep (stage 1)
        n_signature_genotypes = 8  # for the candidate pool's spatial maps (stage 2)
        n_map_genotypes = 24      # for the final rendered heatmaps
        sensitivity_sample = 24

        max_loci = None           # None -> every locus in stage 1
        locus_chunk = 256
        batch_size = 64
        max_per_group = 32
        include_population_contrast = True

        timestep = 500
        layer = 'up_2'
        latent_size = 32
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps_dir = out / 'diverse_snp_maps'
    maps_dir.mkdir(exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    snp_encoder, unet, unet_cfg = load_model(
        resolve_input(cfg.checkpoint, 'checkpoint'), snp_matrix, device,
        pca_cache=str(resolve_output(cfg.pca_cache)))
    projector = snp_encoder.pca
    if not hasattr(projector, 'locus_contributions'):
        raise SystemExit(
            "This checkpoint uses the legacy numeric encoding, which has one "
            "column per locus and so no per-founder alternatives to swap "
            "between. Use a train_onehot.py checkpoint.")
    projector.ensure_explained_variance(snp_matrix)

    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)
    rng = np.random.default_rng(cfg.seed)

    # component sensitivity, for choosing which founder to probe
    sens_path = resolve_output(cfg.sensitivity_cache)
    sensitivity = None
    if sens_path.exists():
        cached = pd.read_csv(sens_path)
        if len(cached) == projector.output_dim:
            sensitivity = cached['normalized_sensitivity_mean'].to_numpy()
            print(f"Loaded cached sensitivity from {sens_path}")
    if sensitivity is None:
        print(f"Computing component sensitivity over {cfg.sensitivity_sample} genotypes...")
        idx = rng.choice(len(sample_names), cfg.sensitivity_sample, replace=False)
        sens_df = population_sensitivity(snp_encoder, projector, snp_matrix, idx, device)
        sens_path.parent.mkdir(parents=True, exist_ok=True)
        sens_df.to_csv(sens_path, index=False)
        sensitivity = sens_df['normalized_sensitivity_mean'].to_numpy()

    print("\nScoring loci (linear proxy, used only to choose the probed founder)...")
    _, best_pair = locus_influence_scores(projector, sensitivity)

    n_loci = projector.n_loci if cfg.max_loci is None else min(cfg.max_loci, projector.n_loci)
    target_founders = np.zeros(n_loci, dtype=int)
    for L in range(n_loci):
        fa, fb = best_pair[L]
        n_a = int((snp_matrix[:, L] == fa).sum())
        n_b = int((snp_matrix[:, L] == fb).sum())
        target_founders[L] = fa if n_a <= n_b else fb
    target_slots = np.array([founder_slot(projector, f) for f in target_founders])

    # stage 1: magnitude sweep, every locus
    sweep_idx = rng.choice(len(sample_names), cfg.n_sweep_genotypes, replace=False)
    print(f"\nStage 1: magnitude sweep over {n_loci:,} loci "
         f"x {cfg.n_sweep_genotypes} genotypes...")
    magnitudes, _ = sweep_contributions(
        snp_encoder, unet, projector, snp_matrix, sweep_idx, target_slots,
        cfg.timestep, latent_shape, device, cfg.seed, cfg.layer,
        cfg.locus_chunk, cfg.batch_size)

    pool_order = np.argsort(magnitudes)[::-1][:cfg.n_pool]
    print(f"  candidate pool: top {len(pool_order)} loci by magnitude "
         f"(range {magnitudes[pool_order[-1]]:.3e} - {magnitudes[pool_order[0]]:.3e})")

    # stage 2: spatial signature for the candidate pool
    sig_idx = rng.choice(len(sample_names), cfg.n_signature_genotypes, replace=False)
    print(f"\nStage 2: spatial signatures for the {len(pool_order)}-locus pool "
         f"x {cfg.n_signature_genotypes} genotypes...")
    # Not preallocated from latent_shape: the attention grid's spatial size
    # depends on which UNet layer cfg.layer names, which is downsampled
    # relative to the full latent (up_2 is 16x16 against a 32x32 latent) -
    # assuming they matched caused a broadcast crash the first time this ran.
    # Built as a list and stacked once the actual grid shape is known instead.
    sig_chunks = []
    for start in range(0, len(pool_order), cfg.locus_chunk):
        stop = min(start + cfg.locus_chunk, len(pool_order))
        chunk_loci = pool_order[start:stop]
        sig_chunks.append(spatial_signature_sweep(
            snp_encoder, unet, projector, snp_matrix, sig_idx, chunk_loci,
            target_slots[chunk_loci], cfg.timestep, latent_shape, device,
            cfg.seed, cfg.layer, cfg.batch_size))
        print(f"  signatures {stop}/{len(pool_order)}")
    signatures = np.concatenate(sig_chunks, axis=0)

    # diversify
    # Seed the greedy selection with the pool's top-magnitude locus (index 0
    # of pool_order / signatures, since pool_order is already magnitude-sorted)
    # so the single strongest known effect is always included regardless of
    # how the diversity search proceeds from there.
    print(f"\nSelecting {cfg.top_n} diverse loci from the pool...")
    selected_local = greedy_diverse_selection(signatures, cfg.top_n, start_index=0)
    selected_loci = pool_order[selected_local]
    selected_sigs = signatures[selected_local]

    # comparison figure: diverse vs plain top-N
    plain_top_local = np.arange(cfg.top_n)          # pool is magnitude-sorted already
    save_distance_comparison(selected_sigs, signatures[plain_top_local],
                             out / 'diversity_comparison.png')

    # render heatmaps for the diverse set
    map_idx = rng.choice(len(sample_names), min(cfg.n_map_genotypes, len(sample_names)),
                         replace=False)
    print(f"\nRendering heatmaps for {len(selected_loci)} diverse loci...")

    rows = []
    for rank, locus in enumerate(selected_loci, start=1):
        locus = int(locus)
        founder = int(target_founders[locus])
        snp_name = snp_names[locus]

        from latent_diffusion.analysis.analyze_snp_spatial_contribution import counterfactual_map
        cf, n_eff = counterfactual_map(
            snp_encoder, unet, projector, snp_matrix, map_idx, locus, founder,
            cfg.timestep, latent_shape, device, cfg.seed, cfg.layer,
            cfg.batch_size, block_size=1)

        pop, n_car, n_oth = (None, 0, 0)
        if cfg.include_population_contrast:
            pop, n_car, n_oth = population_contrast_map(
                snp_encoder, unet, projector, snp_matrix, locus, founder,
                cfg.timestep, latent_shape, device, cfg.seed, cfg.layer,
                cfg.batch_size, cfg.max_per_group, rng)

        save_snp_figure(snp_name, locus, founder, cf, pop, n_car, n_oth,
                        cfg.layer, cfg.timestep, 1,
                        maps_dir / f"{rank:03d}_{snp_name}.png")

        rows.append({
            'diversity_rank': rank, 'snp': snp_name, 'locus': locus,
            'probed_founder': founder,
            'pool_magnitude_rank': int(np.flatnonzero(pool_order == locus)[0]) + 1,
            'magnitude': float(magnitudes[locus]),
            'min_distance_to_prior_selections': (
                float('nan') if rank == 1 else
                1.0 - float(np.max([
                    (selected_sigs[rank - 1].ravel() / np.linalg.norm(selected_sigs[rank - 1].ravel())) @
                    (selected_sigs[j].ravel() / np.linalg.norm(selected_sigs[j].ravel()))
                    for j in range(rank - 1)]))
            ),
        })
        print(f"  {rank:>3}. {snp_name:<20} "
             f"(pool rank {rows[-1]['pool_magnitude_rank']}, "
             f"magnitude {magnitudes[locus]:.3e})")

    detail = pd.DataFrame(rows)
    detail.to_csv(out / 'diverse_snp_details.csv', index=False)

    pool_df = pd.DataFrame({
        'snp': [snp_names[i] for i in pool_order],
        'locus': pool_order,
        'magnitude': magnitudes[pool_order],
        'selected': np.isin(pool_order, selected_loci),
    })
    pool_df.to_csv(out / 'candidate_pool.csv', index=False)

    with open(out / 'summary.json', 'w') as f:
        json.dump({
            'checkpoint': str(cfg.checkpoint), 'layer': cfg.layer,
            'timestep': cfg.timestep, 'n_loci_swept': int(n_loci),
            'n_pool': int(cfg.n_pool), 'top_n': int(cfg.top_n),
        }, f, indent=2)

    # report
    def mean_offdiag_dist(sigs):
        n = len(sigs)
        flat = sigs.reshape(n, -1)
        unit = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        D = 1.0 - (unit @ unit.T)
        return float(D[~np.eye(n, dtype=bool)].mean())

    diverse_mean_dist = mean_offdiag_dist(selected_sigs)
    plain_mean_dist = mean_offdiag_dist(signatures[plain_top_local])

    print(f"  plain top-{cfg.top_n} mean pairwise distance      : {plain_mean_dist:.3f}")
    print(f"  diversity-selected top-{cfg.top_n} mean pairwise distance : {diverse_mean_dist:.3f}")
    print(f"  improvement                              : "
         f"{(diverse_mean_dist - plain_mean_dist) / max(plain_mean_dist, 1e-9) * 100:+.1f}%")
    print("  Distance ranges 0 (same spatial regions) to 1 (disjoint regions) -")
    print("  signatures are non-negative intensity maps, so cosine distance")
    print("  cannot exceed 1 the way it could for a signed comparison.")
    print("  If the improvement is small, the top-N-by-magnitude set was")
    print("  already fairly diverse and this step is not adding much; if it")
    print("  is large, the plain ranking was dominated by a few near-")
    print("  duplicate haplotype blocks - check diversity_comparison.png.")

    print(f"\nWrote diverse_snp_details.csv, candidate_pool.csv, "
         f"diversity_comparison.png and {len(selected_loci)} heatmaps in {maps_dir}")


if __name__ == '__main__':
    main()
