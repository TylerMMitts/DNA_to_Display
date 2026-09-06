# Which SNPs contribute most to the image, measured across every locus.
#
# analyze_snp_spatial_contribution.py maps a handful of loci you name or that a
# cheap linear proxy shortlists. This sweeps ALL of them, ranks by what the
# model actually does rather than by the proxy, then renders heatmaps for the
# top few.
#
# Why measure rather than proxy-rank: locus_influence_scores() weights loading
# distance by component sensitivity, which is a linear approximation of a
# nonlinear encoder followed by a UNet. It is a good shortlist generator, but
# the ranking it produces is not the ranking of measured effect. Sweeping the
# genome properly is affordable once two wasteful things are removed:
#
# the unperturbed pass is computed once
#     attention for the untouched genotypes does not depend on which locus is
#     being perturbed, yet the per-SNP script recomputed it every time -
#     exactly half of its UNet work was redundant.
#
# loci are batched together
#     one forward pass covers many (locus, genotype) perturbations instead of
#     one locus at a time.
#
# That brings a full 43,788-locus sweep to roughly n_loci * n_genotypes forward
# items - about 175k at four genotypes, minutes on a GPU - versus the ~24 hours
# that rendering a figure per locus would cost. Rendering, not computing, was
# always the expensive half, so this computes everything and renders only the
# top ranks.
#
# Single-locus perturbations only. That is the correct unit for ranking
# individual SNPs, and unlike the block mode in the per-SNP script it needs no
# assumption about which columns are genomic neighbours.
#
# Usage
#     python code/latent_diffusion/analysis/rank_snp_contributions.py

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
from latent_diffusion.utils import attention_analysis as aa
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.analysis.analyze_pca_sensitivity import (
    embed_from_pca_vector, population_sensitivity,
)
from latent_diffusion.analysis.analyze_snp_spatial_contribution import (
    locus_influence_scores, attention_grids_from_pca, counterfactual_map,
    population_contrast_map, save_snp_figure, founder_slot,
)


# Batched genome sweep

def build_perturbed_batch(projector, snp_matrix, base_pca, genotype_indices,
                          loci, target_slots):
    # Perturbed PCA vectors for every (genotype, locus) pair in one array.
    #
    # Returns [n_genotypes * n_loci, K], ordered genotype-major so it reshapes
    # to (n_genotypes, n_loci, K) afterwards.
    #
    # Vectorised over loci: a single-locus swap is a difference of two loading
    # columns, so the whole chunk is two fancy-indexed column gathers rather
    # than a Python loop over loci.
    C = projector.components_                  # [K, L*F]
    F = len(projector.founders)
    founders = np.asarray(projector.founders)
    loci = np.asarray(loci)

    cols_new = loci * F + np.asarray(target_slots)
    col_new = C[:, cols_new]                   # [K, n_loci]

    out = np.empty((len(genotype_indices), len(loci), C.shape[0]))
    for gi_row, gi in enumerate(genotype_indices):
        codes = snp_matrix[gi, loci]
        valid = np.isin(codes, founders)
        # searchsorted is safe for invalid codes because the result is masked
        # out below; clipping only keeps the gather in bounds.
        slots_old = np.clip(np.searchsorted(founders, codes), 0, F - 1)
        col_old = C[:, loci * F + slots_old] * valid[None, :]
        delta = col_new - col_old              # [K, n_loci]
        out[gi_row] = (base_pca[gi_row][:, None] + delta).T

    return out.reshape(-1, C.shape[0])


@torch.no_grad()
def sweep_contributions(encoder, unet, projector, snp_matrix, genotype_indices,
                        target_slots, timestep, latent_shape, device, seed,
                        layer, locus_chunk, batch_size, progress_every=20):
    # Measured counterfactual magnitude for every locus.
    #
    # Only the scalar magnitude is kept, not the maps - storing all 43,788 maps
    # would be ~180 MB for data that is about to be discarded for every locus
    # outside the top ranks. The top ranks are re-mapped afterwards, which costs
    # a few dozen forward passes.
    # Length of target_slots, NOT projector.n_loci: cfg.max_loci can restrict
    # the sweep to a prefix of the genome, and taking the count from the
    # projector instead would index target_slots past its end.
    n_loci = len(target_slots)
    n_g = len(genotype_indices)

    base_pca = projector.transform(snp_matrix[genotype_indices])
    base_grids = attention_grids_from_pca(
        encoder, unet, base_pca, timestep, latent_shape, device, seed, layer,
        chunk_size=batch_size)                                  # [n_g, H, W, M]

    magnitudes = np.zeros(n_loci)
    n_chunks = int(np.ceil(n_loci / locus_chunk))

    for ci, start in enumerate(range(0, n_loci, locus_chunk)):
        stop = min(start + locus_chunk, n_loci)
        loci = np.arange(start, stop)

        batch = build_perturbed_batch(projector, snp_matrix, base_pca,
                                      genotype_indices, loci,
                                      target_slots[start:stop])
        grids = attention_grids_from_pca(
            encoder, unet, batch, timestep, latent_shape, device, seed, layer,
            chunk_size=batch_size)                              # [n_g*n_loci, H,W,M]

        grids = grids.reshape(n_g, len(loci), *grids.shape[1:])
        delta = grids - base_grids[:, None]                     # broadcast over loci
        magnitudes[start:stop] = np.abs(delta.mean(axis=0)).mean(axis=(1, 2, 3))

        if ci % progress_every == 0 or ci == n_chunks - 1:
            print(f"  swept {stop}/{n_loci} loci "
                  f"({100.0 * stop / n_loci:.1f}%)")

    return magnitudes, base_grids


# Ranking figure

def save_ranking_figure(ranking, top_n, proxy_scores, save_path):
    # Top-N contribution rates, plus where they sit in the full distribution.
    #
    # The distribution panel is the one that decides how to read the bar chart:
    # a top 50 sitting far out on a long tail is a genuinely distinct set of
    # loci, whereas a top 50 barely above the bulk means the ranking is slicing
    # a smooth continuum and the cut at 50 is arbitrary.
    top = ranking.head(top_n)
    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.42,
                          wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    ax.bar(np.arange(len(top)), top['contribution'].to_numpy(),
           color='#2b7bba')
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(top['snp'], rotation=90, fontsize=6)
    ax.set_ylabel('measured contribution\n(mean |attention change|)')
    ax.set_title(f'Top {len(top)} SNPs by measured contribution', fontsize=12)
    ax.margins(x=0.005)

    ax = fig.add_subplot(gs[1, 0])
    all_vals = ranking['contribution'].to_numpy()
    ax.hist(all_vals, bins=120, color='#bbbbbb', log=True)
    cut = float(top['contribution'].min())
    ax.axvline(cut, color='crimson', ls='--', lw=1.4,
               label=f'top-{len(top)} cutoff')
    ax.set_xlabel('measured contribution')
    ax.set_ylabel('loci (log scale)')
    ax.set_title(f'All {len(ranking):,} loci\n'
                 f'top {len(top)} is the {100.0 * len(top) / len(ranking):.2f}% tail',
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    order = np.argsort(all_vals)[::-1]
    ax.plot(np.arange(1, len(order) + 1), all_vals[order], lw=1.2,
            color='#2b7bba')
    ax.axvline(len(top), color='crimson', ls='--', lw=1.4)
    ax.set_xscale('log')
    ax.set_xlabel('rank (log scale)')
    ax.set_ylabel('measured contribution')
    ax.set_title('Contribution vs rank\nflat = no locus stands out', fontsize=10)

    # How well the cheap linear proxy predicted the measured ranking. Reported
    # rather than assumed: if it is high the proxy alone is a valid shortlist
    # for future runs, and if it is low the full sweep is doing real work.
    if proxy_scores is not None:
        rho = float(pd.Series(ranking['contribution'].to_numpy())
                    .corr(pd.Series(ranking['proxy_score'].to_numpy()),
                          method='spearman'))
        fig.suptitle(f'Spearman(linear proxy, measured contribution) = {rho:.3f}',
                     fontsize=10, y=0.995)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/rank_snp_contributions.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'snp_contribution_ranking'
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'      # legacy path only
        sensitivity_cache = RESULTS_DIR / 'snp_contribution_ranking' / 'population_sensitivity.csv'

        top_n = 50               # heatmaps rendered for this many
        # Genotypes perturbed per locus during the sweep. The sweep cost is
        # linear in this. Four averages out per-genotype idiosyncrasy without
        # making a genome-wide sweep expensive; raise it if the ranking looks
        # unstable between seeds.
        n_sweep_genotypes = 4
        # Genotypes used for the top-N heatmaps, where quality matters more
        # than speed.
        n_map_genotypes = 24
        sensitivity_sample = 24

        # None -> every locus. Set to an integer to sweep only the first N,
        # which is the fast way to sanity-check the pipeline before committing
        # to the full genome.
        max_loci = None

        locus_chunk = 256        # loci per batched perturbation build
        batch_size = 64          # UNet forward batch
        max_per_group = 32       # cap per side of the carrier contrast

        # Population contrast on the top-N figures. Linkage-confounded (see
        # analyze_snp_spatial_contribution.py) and costs extra UNet passes,
        # but it is the "images that do or don't have it" view.
        include_population_contrast = True

        timestep = 500
        layer = 'up_2'
        latent_size = 32
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps_dir = out / 'top_snp_maps'
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

    # component sensitivity, for the proxy shortlist
    sens_path = resolve_output(cfg.sensitivity_cache)
    sensitivity = None
    if sens_path.exists():
        cached = pd.read_csv(sens_path)
        if len(cached) == projector.output_dim:
            sensitivity = cached['normalized_sensitivity_mean'].to_numpy()
            print(f"Loaded cached sensitivity from {sens_path}")
    if sensitivity is None:
        print(f"Computing component sensitivity over {cfg.sensitivity_sample} "
              f"genotypes...")
        idx = rng.choice(len(sample_names), cfg.sensitivity_sample, replace=False)
        sens_df = population_sensitivity(snp_encoder, projector, snp_matrix,
                                         idx, device)
        sens_path.parent.mkdir(parents=True, exist_ok=True)
        sens_df.to_csv(sens_path, index=False)
        sensitivity = sens_df['normalized_sensitivity_mean'].to_numpy()

    # which founder to probe at each locus
    print("\nScoring loci (linear proxy, used to choose the probed founder)...")
    proxy_scores, best_pair = locus_influence_scores(projector, sensitivity)

    n_loci = projector.n_loci if cfg.max_loci is None else min(cfg.max_loci,
                                                               projector.n_loci)
    # Probe toward the rarer of the locus's two most-separated founders:
    # flipping toward a common founder largely reproduces the population
    # average and understates what the locus can do.
    target_founders = np.zeros(n_loci, dtype=int)
    for L in range(n_loci):
        fa, fb = best_pair[L]
        n_a = int((snp_matrix[:, L] == fa).sum())
        n_b = int((snp_matrix[:, L] == fb).sum())
        target_founders[L] = fa if n_a <= n_b else fb
    target_slots = np.array([founder_slot(projector, f) for f in target_founders])

    # sweep
    sweep_idx = rng.choice(len(sample_names), cfg.n_sweep_genotypes, replace=False)
    print(f"\nSweeping {n_loci:,} loci x {cfg.n_sweep_genotypes} genotypes "
          f"({n_loci * cfg.n_sweep_genotypes:,} forward items)...")

    magnitudes, _ = sweep_contributions(
        snp_encoder, unet, projector, snp_matrix, sweep_idx,
        target_slots, cfg.timestep, latent_shape, device, cfg.seed,
        cfg.layer, cfg.locus_chunk, cfg.batch_size)

    ranking = pd.DataFrame({
        'snp': snp_names[:n_loci],
        'locus': np.arange(n_loci),
        'probed_founder': target_founders,
        'contribution': magnitudes,
        'proxy_score': proxy_scores[:n_loci],
    }).sort_values('contribution', ascending=False).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    ranking.to_csv(out / 'snp_contribution_ranking.csv', index=False)
    print(f"\nWrote snp_contribution_ranking.csv ({len(ranking):,} loci)")

    save_ranking_figure(ranking, cfg.top_n, proxy_scores,
                        out / 'contribution_ranking.png')

    # heatmaps for the top N
    top = ranking.head(cfg.top_n)
    map_idx = rng.choice(len(sample_names),
                         min(cfg.n_map_genotypes, len(sample_names)),
                         replace=False)
    print(f"\nRendering heatmaps for the top {len(top)}...")

    rows = []
    for _, r in top.iterrows():
        locus = int(r['locus'])
        founder = int(r['probed_founder'])
        snp_name = r['snp']

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
                        maps_dir / f"{int(r['rank']):03d}_{snp_name}.png")

        rows.append({
            'rank': int(r['rank']), 'snp': snp_name, 'locus': locus,
            'probed_founder': founder,
            'sweep_contribution': float(r['contribution']),
            'map_contribution': float(np.abs(cf).mean()),
            'n_genotypes_changed': n_eff,
            'population_contrast': (float(np.abs(pop).mean())
                                    if pop is not None else float('nan')),
        })
        print(f"  {int(r['rank']):>3}. {snp_name:<20} "
              f"contribution {r['contribution']:.3e}")

    detail = pd.DataFrame(rows)
    detail.to_csv(out / 'top_snp_details.csv', index=False)

    with open(out / 'summary.json', 'w') as f:
        json.dump({
            'checkpoint': str(cfg.checkpoint),
            'layer': cfg.layer, 'timestep': cfg.timestep,
            'n_loci_swept': int(n_loci),
            'n_sweep_genotypes': int(cfg.n_sweep_genotypes),
            'top_n': int(cfg.top_n),
            'proxy_vs_measured_spearman': float(
                pd.Series(ranking['contribution']).corr(
                    pd.Series(ranking['proxy_score']), method='spearman')),
        }, f, indent=2)

    # report
    vals = ranking['contribution'].to_numpy()
    rho = float(pd.Series(vals).corr(pd.Series(ranking['proxy_score']),
                                     method='spearman'))
    cutoff = vals[cfg.top_n - 1]
    # How many loci sit within float-noise of the cutoff. Batched and
    # unbatched forward passes agree exactly when chunk shapes match, but
    # differ by ~1e-3 relative when they do not, so loci closer together than
    # that are not reliably ordered - and linkage makes near-ties common,
    # since adjacent loci in a haplotype block have nearly identical effects.
    near_tie = int((np.abs(vals - cutoff) <= 1e-3 * cutoff).sum())

    print(f"  loci swept                : {n_loci:,}")
    print(f"  top-{cfg.top_n} cutoff            : {cutoff:.3e}")
    print(f"  median locus              : {np.median(vals):.3e}")
    print(f"  top-1 / median ratio      : {vals[0] / np.median(vals):.1f}x")
    print(f"  loci within 0.1% of cutoff: {near_tie}")
    print(f"  proxy vs measured (rho)   : {rho:.3f}")
    print("  top-1 / median near 1 would mean no locus stands out and the")
    print("  ranking is slicing a flat continuum. A high proxy correlation")
    print("  means future runs could shortlist with the cheap score instead")
    print("  of sweeping; a low one means the sweep is doing real work.")
    if near_tie > 1:
        print(f"\n  {near_tie} loci sit within float-precision of the top-"
              f"{cfg.top_n} cutoff, so membership at the boundary is not")
        print("  meaningful - expect linked neighbours to trade places between")
        print("  runs. The magnitudes are stable; the tie-breaking is not.")

    print(f"\nWrote snp_contribution_ranking.csv, contribution_ranking.png,")
    print(f"top_snp_details.csv and {len(top)} heatmaps in {maps_dir}")


if __name__ == '__main__':
    main()
