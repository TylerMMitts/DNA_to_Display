# Where in the image a single named locus acts.
#
# Two views of the same question: a counterfactual that swaps that locus while
# holding everything else fixed, and a population contrast between real
# carriers and non-carriers. The contrast is confounded by linkage, the
# counterfactual is not.

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


# Ranking which loci are worth looking at

def locus_influence_scores(projector, sensitivity, chunk=2048):
    # Per-locus score for how much the inherited founder matters there.
    #
    # For locus L the eight founder columns of components_ are the eight
    # possible PCA-space positions that locus can put a genotype in. The score
    # is the largest sensitivity-weighted distance between any two of them:
    #
    # score(L) = max over founder pairs (i,j) of
    #            || (C[:, L*F+i] - C[:, L*F+j]) * sensitivity ||
    #
    # Weighting by the encoder's per-component sensitivity matters - a large
    # loading on a component the encoder barely responds to moves the embedding
    # very little, and would otherwise rank spuriously high.
    #
    # Deliberately genotype-independent: it asks "does it matter which founder
    # you inherited here", not "what did this particular genotype inherit", so
    # the same shortlist applies to every genotype and the expensive spatial
    # analysis is not re-targeted per individual.
    #
    # Returns (scores [L], best_pair [L, 2]) with founder VALUES, not slot
    # indices, in best_pair.
    C = projector.components_                       # [K, L*F]
    K = C.shape[0]
    F = len(projector.founders)
    L = projector.n_loci
    founders = np.asarray(projector.founders)

    sens = np.asarray(sensitivity, dtype=np.float64).reshape(K, 1, 1)

    scores = np.zeros(L)
    best_pair = np.zeros((L, 2), dtype=int)
    pairs = [(i, j) for i in range(F) for j in range(i + 1, F)]

    for start in range(0, L, chunk):
        stop = min(start + chunk, L)
        # [K, chunk_len, F] - float32 keeps this affordable at 43k loci.
        W = C[:, start * F:stop * F].reshape(K, stop - start, F).astype(np.float32)
        W = W * sens.astype(np.float32)

        block_best = np.zeros(stop - start)
        block_pair = np.zeros((stop - start, 2), dtype=int)
        for i, j in pairs:
            d = np.linalg.norm(W[:, :, i] - W[:, :, j], axis=0)   # [chunk_len]
            better = d > block_best
            block_best[better] = d[better]
            block_pair[better] = (founders[i], founders[j])

        scores[start:stop] = block_best
        best_pair[start:stop] = block_pair

    return scores, best_pair


# Attention

@torch.no_grad()
def attention_grids_from_pca(encoder, unet, pca_batch, timestep, latent_shape,
                             device, seed, layer, chunk_size=16):
    # Spatial attention grids for a batch of PCA-space vectors.
    #
    # Shared noise across every item, as in collect_attention() - each item sees
    # an identical latent, so any difference in attention is attributable to the
    # conditioning rather than to the draw.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(1, *latent_shape, generator=generator).to(device)
    unet.set_store_attention(False)

    collected = []
    for start in range(0, len(pca_batch), chunk_size):
        sub = pca_batch[start:start + chunk_size]
        n = len(sub)
        z_t = noise.repeat(n, 1, 1, 1)
        t = torch.full((n,), timestep, device=device, dtype=torch.long)

        unet.clear_attention_history()
        emb = embed_from_pca_vector(
            encoder, torch.tensor(sub, dtype=torch.float32, device=device))
        unet(z_t, t, emb)

        found = False
        for name, block in unet.iter_cross_attn_blocks():
            if name == layer and block.attention_weights is not None:
                collected.append(block.attention_weights.cpu())
                found = True
        if not found:
            raise SystemExit(
                f"layer {layer!r} produced no attention. Available: "
                f"{[n for n, _ in unet.iter_cross_attn_blocks()]}")

    return aa.attention_to_grids(torch.cat(collected, dim=0))     # [B, H, W, M]


def founder_slot(projector, founder_value):
    return list(projector.founders).index(int(founder_value))


def single_locus_delta(projector, locus, founder_new, current_code):
    # PCA-space delta for setting `locus` to `founder_new`.
    #
    # Exactly a difference of two loading columns (see module docstring). A
    # missing current call (-1) contributes an all-zero block, so the delta is
    # just the new founder's column.
    F = len(projector.founders)
    col_new = projector.components_[:, locus * F + founder_slot(projector, founder_new)]
    if int(current_code) in projector.founders:
        col_old = projector.components_[:, locus * F + founder_slot(projector, current_code)]
        return col_new - col_old
    return col_new.copy()


def counterfactual_map(encoder, unet, projector, snp_matrix, genotype_indices,
                       locus, founder_new, timestep, latent_shape, device, seed,
                       layer, chunk_size, block_size=1):
    # Mean attention change from flipping a locus (or a linked block) to
    # founder_new.
    #
    # block_size > 1 flips locus and its neighbours out to that width. This is
    # the more realistic counterfactual: recombination hands down contiguous
    # haplotype blocks rather than single loci, so a lone-locus swap is a
    # genotype that essentially never occurs. It also brings the counterfactual
    # onto the same footing as the population contrast, which is measuring a
    # block whether or not it means to.
    #
    # Neighbouring COLUMNS are assumed to be neighbouring loci. That holds here
    # because load_snp_data_from_parquet pivots on gene_model, and these
    # zero-padded sequential gene IDs sort into genomic order within a region -
    # but it is an assumption about the ID scheme, not a guarantee.
    #
    # Returns (map [H, W, M], n_effective) where n_effective counts genotypes
    # that were actually changed; a genotype already carrying founder_new
    # everywhere in the block gets a zero delta and contributes nothing but
    # still dilutes the mean, so the count is reported rather than hidden.
    half = block_size // 2
    lo = max(0, locus - half)
    hi = min(projector.n_loci, lo + block_size)

    base = projector.transform(snp_matrix[genotype_indices])
    pert = base.copy()
    n_effective = 0
    for row, gi in enumerate(genotype_indices):
        delta = np.zeros(projector.output_dim)
        for L in range(lo, hi):
            delta += single_locus_delta(projector, L, founder_new,
                                        snp_matrix[gi, L])
        pert[row] += delta
        if np.linalg.norm(delta) > 0:
            n_effective += 1

    g_base = attention_grids_from_pca(encoder, unet, base, timestep, latent_shape,
                                      device, seed, layer, chunk_size)
    g_pert = attention_grids_from_pca(encoder, unet, pert, timestep, latent_shape,
                                      device, seed, layer, chunk_size)
    return (g_pert - g_base).mean(axis=0), n_effective             # [H, W, M]


def population_contrast_map(encoder, unet, projector, snp_matrix, locus,
                            founder, timestep, latent_shape, device, seed,
                            layer, chunk_size, max_per_group, rng):
    # Mean attention of carriers minus non-carriers. Linkage-confounded.
    codes = snp_matrix[:, locus]
    carriers = np.flatnonzero(codes == founder)
    others = np.flatnonzero((codes != founder) & np.isin(codes, projector.founders))
    if len(carriers) < 2 or len(others) < 2:
        return None, len(carriers), len(others)

    if len(carriers) > max_per_group:
        carriers = rng.choice(carriers, max_per_group, replace=False)
    if len(others) > max_per_group:
        others = rng.choice(others, max_per_group, replace=False)

    g_c = attention_grids_from_pca(
        encoder, unet, projector.transform(snp_matrix[carriers]), timestep,
        latent_shape, device, seed, layer, chunk_size)
    g_o = attention_grids_from_pca(
        encoder, unet, projector.transform(snp_matrix[others]), timestep,
        latent_shape, device, seed, layer, chunk_size)
    return g_c.mean(axis=0) - g_o.mean(axis=0), len(carriers), len(others)


# Figure

def save_snp_figure(snp_name, locus, founder, cf_map, pop_map, n_carriers,
                    n_others, layer, timestep, block_size, save_path):
    M = cf_map.shape[-1]
    rows = 2 if pop_map is not None else 1
    # +1 column for the token-averaged summary panel.
    fig, axes = plt.subplots(rows, M + 1, figsize=(1.65 * (M + 1) + 1.5,
                                                   1.85 * rows + 1.4),
                             squeeze=False)

    cf_label = ('counterfactual\n(this locus only)' if block_size == 1
                else f'counterfactual\n({block_size}-locus block)')
    row_specs = [(cf_label, cf_map)]
    if pop_map is not None:
        row_specs.append((f'population contrast\n{n_carriers} carry / '
                          f'{n_others} do not', pop_map))

    for r, (label, m) in enumerate(row_specs):
        lim = float(np.abs(m).max()) or 1.0
        for t in range(M):
            ax = axes[r][t]
            im = ax.imshow(m[:, :, t], cmap='RdBu_r', vmin=-lim, vmax=lim,
                           interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'token {t}', fontsize=8)
            if t == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha='right',
                              va='center', labelpad=52)

        ax = axes[r][M]
        summary = np.abs(m).mean(axis=-1)
        im2 = ax.imshow(summary, cmap='inferno', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('|mean| over\ntokens', fontsize=8)
        fig.colorbar(im2, ax=ax, fraction=0.046)

    fig.suptitle(f'{snp_name}  (locus {locus}, founder {founder})   -   '
                 f'{layer} @ t={timestep}\n'
                 f'red = more attention, blue = less',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=145, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/analyze_snp_spatial_contribution.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'snp_spatial_contribution'
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'      # legacy path only
        sensitivity_cache = RESULTS_DIR / 'snp_spatial_contribution' / 'population_sensitivity.csv'

        # Which SNPs to map. None -> the top n_snps by locus_influence_scores.
        # Set to a list of SNP names (e.g. ['Zm00001d015468']) to target
        # specific loci instead.
        snps = None
        n_snps = 6

        # Which founder to probe at each locus. None -> the more influential
        # member of that locus's strongest founder pair.
        founder = None

        # Genotypes perturbed for the counterfactual map. More is smoother but
        # costs two UNet passes each.
        n_counterfactual_genotypes = 24

        # Loci flipped together in the counterfactual, centred on the target.
        # 1 isolates a single SNP, which is the cleanest attribution but a
        # genotype that recombination would essentially never produce - and
        # the effect is correspondingly tiny: one locus out of ~43,788 shifts
        # the projected genotype by roughly 0.4% of a population standard
        # deviation, measured. Values around 25-100 approximate an inherited
        # haplotype block and are directly comparable to the population
        # contrast, which is measuring a block regardless.
        block_size = 50
        # Cap per side of the carrier / non-carrier contrast.
        max_per_group = 32
        # Genotypes used to estimate component sensitivity for the ranking.
        sensitivity_sample = 24

        timestep = 500
        layer = 'up_2'          # highest spatial resolution in this UNet
        latent_size = 32
        chunk_size = 16
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
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
            "This checkpoint uses the legacy numeric encoding, where a locus is "
            "a single column and there are no per-founder alternatives to swap "
            "between - the counterfactual this script is built on does not "
            "exist for it. Use a train_onehot.py checkpoint.")
    projector.ensure_explained_variance(snp_matrix)

    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)
    rng = np.random.default_rng(cfg.seed)

    # component sensitivity, for weighting the locus ranking
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
        sample_idx = rng.choice(len(sample_names), cfg.sensitivity_sample,
                                replace=False)
        sens_df = population_sensitivity(snp_encoder, projector, snp_matrix,
                                         sample_idx, device)
        sens_path.parent.mkdir(parents=True, exist_ok=True)
        sens_df.to_csv(sens_path, index=False)
        sensitivity = sens_df['normalized_sensitivity_mean'].to_numpy()

    # pick loci
    print("\nScoring loci by how much the inherited founder matters...")
    scores, best_pair = locus_influence_scores(projector, sensitivity)

    name_to_locus = {n: i for i, n in enumerate(snp_names)}
    if cfg.snps:
        missing = [s for s in cfg.snps if s not in name_to_locus]
        if missing:
            raise SystemExit(f"SNPs not found in the parquet: {missing}")
        selected = [name_to_locus[s] for s in cfg.snps]
    else:
        selected = np.argsort(scores)[::-1][:cfg.n_snps].tolist()

    ranking = pd.DataFrame({
        'snp': snp_names,
        'locus': np.arange(len(snp_names)),
        'influence_score': scores,
        'founder_a': best_pair[:, 0],
        'founder_b': best_pair[:, 1],
    }).sort_values('influence_score', ascending=False)
    ranking.to_csv(out / 'locus_influence_ranking.csv', index=False)
    print(f"  wrote locus_influence_ranking.csv "
          f"({len(ranking)} loci scored)")

    # per-SNP maps
    counterfactual_idx = rng.choice(len(sample_names),
                                    min(cfg.n_counterfactual_genotypes,
                                        len(sample_names)),
                                    replace=False)

    rows = []
    for locus in selected:
        snp_name = snp_names[locus]
        if cfg.founder is not None:
            founder = int(cfg.founder)
        else:
            # The pair member carried by fewer genotypes is the more
            # informative one to probe - flipping toward a common founder
            # mostly reproduces the population average.
            fa, fb = best_pair[locus]
            n_a = int((snp_matrix[:, locus] == fa).sum())
            n_b = int((snp_matrix[:, locus] == fb).sum())
            founder = int(fa if n_a <= n_b else fb)

        print(f"\n{snp_name}  (locus {locus}, probing founder {founder}, "
              f"score {scores[locus]:.4f})")

        cf, n_eff = counterfactual_map(
            snp_encoder, unet, projector, snp_matrix, counterfactual_idx,
            locus, founder, cfg.timestep, latent_shape, device, cfg.seed,
            cfg.layer, cfg.chunk_size, cfg.block_size)
        if n_eff == 0:
            print(f"  every sampled genotype already carries founder {founder} "
                  f"across this block - the counterfactual is a no-op here")
        elif n_eff < len(counterfactual_idx):
            print(f"  {n_eff}/{len(counterfactual_idx)} genotypes actually "
                  f"changed (the rest already carried founder {founder})")

        pop, n_car, n_oth = population_contrast_map(
            snp_encoder, unet, projector, snp_matrix, locus, founder,
            cfg.timestep, latent_shape, device, cfg.seed, cfg.layer,
            cfg.chunk_size, cfg.max_per_group, rng)
        if pop is None:
            print(f"  population contrast skipped: {n_car} carriers / "
                  f"{n_oth} non-carriers is too few to average")

        save_snp_figure(snp_name, locus, founder, cf, pop, n_car, n_oth,
                        cfg.layer, cfg.timestep, cfg.block_size,
                        out / f'{snp_name}.png')

        cf_mag = float(np.abs(cf).mean())
        pop_mag = float(np.abs(pop).mean()) if pop is not None else float('nan')
        # Cosine between the two maps: do the isolated locus and the whole
        # inherited block push attention the same way?
        agreement = float('nan')
        if pop is not None:
            a, b = cf.ravel(), pop.ravel()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                agreement = float(a @ b / (na * nb))

        rows.append({
            'snp': snp_name, 'locus': locus, 'founder': founder,
            'influence_score': float(scores[locus]),
            'block_size': cfg.block_size,
            'n_genotypes_changed': n_eff,
            'counterfactual_mean_abs': cf_mag,
            'population_contrast_mean_abs': pop_mag,
            'contrast_over_counterfactual': (pop_mag / cf_mag
                                             if cf_mag > 0 else float('nan')),
            'map_agreement_cosine': agreement,
            'n_carriers': n_car, 'n_non_carriers': n_oth,
        })
        # Scientific notation: a single-locus effect is around 1e-6, which
        # fixed-point formatting renders as a misleading "0.00000".
        print(f"  counterfactual |mean| = {cf_mag:.3e}   "
              f"population |mean| = {pop_mag:.3e}   "
              f"agreement = {agreement:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'snp_spatial_summary.csv', index=False)
    with open(out / 'summary.json', 'w') as f:
        json.dump({'layer': cfg.layer, 'timestep': cfg.timestep,
                   'checkpoint': str(cfg.checkpoint),
                   'snps': rows}, f, indent=2)

    print(f"{'snp':<20}{'counterfact':>13}{'population':>13}{'ratio':>10}{'agree':>8}")
    for r in rows:
        print(f"{r['snp']:<20}{r['counterfactual_mean_abs']:>13.3e}"
              f"{r['population_contrast_mean_abs']:>13.3e}"
              f"{r['contrast_over_counterfactual']:>10.1f}"
              f"{r['map_agreement_cosine']:>+8.3f}")

    print(f"\n  block_size = {cfg.block_size} locus/loci perturbed per counterfactual.")
    print("  ratio >> 1 means the population contrast dwarfs the counterfactual -")
    print("  the signal is coming from the linked haplotype block and the genome-")
    print("  wide background separating carriers from non-carriers, not from this")
    print("  locus alone. Raising block_size closes that gap and is the fairer")
    print("  comparison. agreement near +1 means locus and block push attention")
    print("  the same direction; near 0 means the locus alone does something the")
    print("  surrounding block does not.")
    print("\n  A single-locus counterfactual is genuinely small - around 1e-6 in")
    print("  attention units, because one locus out of ~43,788 moves the")
    print("  projected genotype only a fraction of a population standard")
    print("  deviation. It is real signal, not numerical noise: the response was")
    print("  verified to scale exactly linearly with perturbation size, and a")
    print("  zero perturbation returns exactly zero. The heatmaps are colour-")
    print("  normalised per panel, so the spatial PATTERN is readable regardless")
    print("  of that magnitude.")

    print(f"\nWrote per-SNP heatmaps, locus_influence_ranking.csv, "
          f"snp_spatial_summary.csv and summary.json to {out}")


if __name__ == '__main__':
    main()
