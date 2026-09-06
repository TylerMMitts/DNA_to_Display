# Different ways of building a founder genotype to generate from, compared.
#
# A pure founder is far outside the training distribution, so this tries
# enriched and carrier-averaged alternatives and measures which separates the
# eight founders most.

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
    DIFFUSION_ONEHOT_DIR, LITEVAE_MODEL, RESULTS_DIR, SNP_PARQUET,
    find_latest_checkpoint, resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.analysis.analyze_pca_sensitivity import embed_from_pca_vector


# Generation from PCA coordinates

@torch.no_grad()
def generate_from_pca(encoder, unet, scheduler, decoder, pca_batch, seeds,
                      device, latent_shape, num_steps, guidance_scale=1.0):
    # DDIM sampling conditioned on PCA coordinates directly.
    #
    # Mirrors generate_batch() from generate_from_dataset.py, but enters at the
    # PCA stage via embed_from_pca_vector so that strategies which construct a
    # conditioning vector in PCA space (scaling, amplification, averaging real
    # genotypes) can be generated without an inverse transform back to founder
    # codes - which for several of them does not exist, since a scaled or
    # averaged PCA vector need not correspond to any valid one-hot genotype.
    #
    # guidance_scale > 1 applies classifier-free-style guidance:
    #
    # eps = eps_ref + w * (eps_cond - eps_ref)
    #
    # This is the amplification that CAN work. Scaling the conditioning vector
    # itself does nothing, because the encoder's leading LayerNorm renormalises
    # it away (see the module docstring - 20x input amplitude moved founder
    # embedding separation by 0.0003). Guidance amplifies the model's OUTPUT
    # instead: the difference between what the UNet predicts for this genotype
    # and what it predicts for the average one. Nothing renormalises that, so
    # unlike input scaling it is not architecturally neutered.
    #
    # The reference is the zero vector in PCA space, which is exactly the
    # population centroid - projector.transform subtracts the one-hot mean, and
    # the population's mean projected coordinate is ~1e-7 (verified by
    # population_pca_stats). So "the average genotype" is available as a
    # reference without needing a null-conditioning token the model was never
    # trained with.
    #
    # Caveat worth knowing: this model was NOT trained with conditioning
    # dropout, which is what true classifier-free guidance assumes. Using the
    # population centroid as a stand-in is a reasonable approximation, but high
    # w extrapolates along a direction the model never saw pushed that far, so
    # expect artefacts to appear at some point as w grows. Sweep it rather than
    # picking one value on faith.
    B = len(pca_batch)
    z_t = torch.empty(B, *latent_shape, device=device)
    for i, seed in enumerate(seeds):
        g = torch.Generator(device='cpu').manual_seed(int(seed))
        z_t[i] = torch.randn(*latent_shape, generator=g)

    cond = torch.tensor(np.asarray(pca_batch), dtype=torch.float32, device=device)
    emb = embed_from_pca_vector(encoder, cond)

    use_guidance = abs(guidance_scale - 1.0) > 1e-9
    if use_guidance:
        emb_ref = embed_from_pca_vector(encoder, torch.zeros_like(cond))

    timesteps = scheduler.get_timesteps(num_steps, device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
        t_prev = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)

        noise_pred = unet(z_t, t_batch, emb)
        if use_guidance:
            ref_pred = unet(z_t, t_batch, emb_ref)
            noise_pred = ref_pred + guidance_scale * (noise_pred - ref_pred)

        z_t = scheduler.denoise_step(z_t, noise_pred, t_batch, t_prev_batch)

    images = decoder(z_t, save_steps=False)
    images = torch.clamp((images + 1) / 2, 0, 1)
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


# Strategies - each returns [n_founders, K] PCA coordinates

def rescale_to_real_radius(pca_vecs, projector, snp_matrix):
    # Scale each row to the mean real-genotype distance from the centre.
    #
    # Needed by every strategy that AVERAGES several genotypes' PCA vectors.
    # Averaging vectors that point in different directions produces a result
    # shorter than any of them - measured here, averaging 8 enriched
    # backgrounds collapsed the radius from ~117 (pure founders) to ~72 and
    # founder separation from 175 to 44, i.e. the averaging alone made the
    # founders LESS distinguishable than the baseline it was meant to improve
    # on. That shrinkage is an artefact of the mean, not a statement that the
    # founders are similar: the direction each average points is still a good,
    # low-variance estimate of that founder's identity. Rescaling keeps the
    # estimated direction and discards only the artefactual amplitude loss.
    real_radius = np.linalg.norm(projector.transform(snp_matrix), axis=1).mean()
    norms = np.linalg.norm(pca_vecs, axis=1, keepdims=True)
    return pca_vecs * (real_radius / np.maximum(norms, 1e-12))


def strat_pure(projector, snp_matrix, founders, rng, **kw):
    L = snp_matrix.shape[1]
    return projector.transform(
        np.stack([np.full(L, float(k), dtype=np.float32) for k in founders]))


def strat_scaled(projector, snp_matrix, founders, rng, scale=None,
                 match_radius=False, **kw):
    # Pure-founder direction, rescaled.
    #
    # match_radius targets the mean real-genotype radius, correcting the
    # measured 62% radius deficit without rotating the vector - the founder's
    # identity (direction) is preserved and only its amplitude changes.
    pure = strat_pure(projector, snp_matrix, founders, rng)
    if match_radius:
        real_radius = np.linalg.norm(projector.transform(snp_matrix), axis=1).mean()
        norms = np.linalg.norm(pure, axis=1, keepdims=True)
        return pure * (real_radius / np.maximum(norms, 1e-12))
    return pure * float(scale)


def strat_enriched(projector, snp_matrix, founders, rng, purity=0.25,
                   n_backgrounds=8, **kw):
    # Real genotype backgrounds with a fraction of loci switched to the
    # founder.
    #
    # The same background genotypes are reused for every founder, so a
    # difference between two founders' images cannot be an artefact of having
    # drawn different backgrounds. Averaging the resulting PCA vectors across
    # backgrounds (rather than generating each separately) keeps one image per
    # founder while damping whatever is idiosyncratic to any single background.
    #
    # Substitution is categorical - loci are reassigned to the founder's code -
    # never a numeric blend, since the 1-8 values are labels and an averaged
    # code like 4.5 names no founder.
    L = snp_matrix.shape[1]
    bg_idx = rng.choice(len(snp_matrix), min(n_backgrounds, len(snp_matrix)),
                        replace=False)
    k_loci = int(round(purity * L))

    out = []
    for k in founders:
        vecs = []
        for gi in bg_idx:
            v = snp_matrix[gi].astype(np.float32).copy()
            if k_loci > 0:
                # Same locus subset for every founder at a given background,
                # so founders are compared on identical positions.
                sub = np.random.default_rng(int(gi)).choice(L, k_loci, replace=False)
                v[sub] = float(k)
            vecs.append(v)
        out.append(projector.transform(np.stack(vecs)).mean(axis=0))
    return rescale_to_real_radius(np.stack(out), projector, snp_matrix)


def strat_top_carriers(projector, snp_matrix, founders, rng, n_top=10, **kw):
    # Mean PCA coordinate of the real genotypes richest in each founder.
    #
    # No synthetic genotype is involved, so this cannot leave the data manifold
    # - it is the empirical centroid of the genotypes that already look most
    # like each founder. Averaging is done in PCA space, which is continuous and
    # where a mean is meaningful, rather than over raw founder codes, where it
    # would not be.
    out = []
    for k in founders:
        share = (snp_matrix == k).mean(axis=1)
        top = np.argsort(share)[::-1][:n_top]
        out.append(projector.transform(snp_matrix[top]).mean(axis=0))
    return rescale_to_real_radius(np.stack(out), projector, snp_matrix)


# Figures

def save_strategy_grid(images_by_strategy, strategy_names, founders, save_path):
    n_rows, n_cols = len(strategy_names), len(founders)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.85 * n_cols, 2.0 * n_rows), squeeze=False)
    for r, name in enumerate(strategy_names):
        for c, k in enumerate(founders):
            ax = axes[r][c]
            ax.imshow(images_by_strategy[name][c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'founder {k}', fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=8.5, rotation=0,
                              ha='right', va='center', labelpad=58)
    fig.suptitle('Founder archetypes under different conditioning strategies\n'
                 'all founders in a row share one noise seed, so differences '
                 'across a row are the founder', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_separation_figure(summary, real_reference, save_path):
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    names = summary['strategy'].tolist()
    vals = summary['mean_pairwise_rmse'].to_numpy()
    colors = ['#c9622a' if n == 'pure' else '#2b7bba' for n in names]
    ax.bar(np.arange(len(names)), vals, color=colors)
    ax.axhline(real_reference, color='crimson', ls='--', lw=1.5,
               label=f'8 random real genotypes ({real_reference:.1f})')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('mean pairwise image RMSE\nbetween the 8 founders')
    ax.set_title('How distinct are the eight founders from each other?\n'
                 'the dashed line is normal genotype-driven differentiation '
                 'for this model', fontsize=11)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/generation/founder_archetype_strategies.py
    class cfg:
        checkpoint_dir = DIFFUSION_ONEHOT_DIR
        litevae_checkpoint = LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'   # legacy path only
        output_dir = RESULTS_DIR / 'founder_strategies'

        founders = None          # None -> detected from the data
        seed = 0                 # shared across all founders within a strategy

        # Scaling CONTROLS, not candidates. LayerNorm at the front of the
        # encoder makes the output nearly scale-invariant (see the module
        # docstring - 20x amplitude changed founder embedding separation by
        # 0.0003), so these are expected to reproduce `pure` almost exactly.
        # One factor is enough to demonstrate that; sweeping several just
        # burns generation time re-confirming the same no-op.
        amplify_factors = [5.0]
        # Enrichment fractions. ~29% is the largest real single-founder share
        # in this population, so 0.25 is near-realistic and 0.5 is already
        # beyond anything observed.
        enrich_purities = [0.25, 0.5]
        enrich_backgrounds = 8
        top_carriers_n = 10

        # Classifier-free-style guidance sweep, applied to guidance_strategy.
        # This is the amplification that is NOT defeated by the encoder's
        # LayerNorm, because it scales the UNet's output rather than its
        # input (see generate_from_pca). 1.0 is ordinary generation and is
        # included so the sweep contains its own baseline.
        #
        # Costs one extra UNet forward per step per scale, so keep the list
        # short. Expect artefacts at large w: the model was not trained with
        # conditioning dropout, so high guidance extrapolates further than it
        # ever saw. Judge these by eye, not by RMSE alone - RMSE rises with
        # artefacts too, so a high score here is not automatically a win.
        guidance_scales = [1.0, 2.0, 3.0, 5.0]
        guidance_strategy = 'top_carriers'

        sampling_steps = 50
        imgsz = 256
        latent_size = 32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    founders = cfg.founders
    if founders is None:
        founders = sorted(int(v) for v in np.unique(snp_matrix) if v > 0)
    print(f"Founders: {founders}")

    ckpt_path = find_latest_checkpoint(
        resolve_input(cfg.checkpoint_dir, 'checkpoint directory'))
    snp_encoder, unet, unet_cfg = load_model(
        ckpt_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    projector = snp_encoder.pca
    if not hasattr(projector, 'locus_contributions'):
        raise SystemExit(
            "This checkpoint uses the legacy numeric encoding. The founder "
            "geometry this script reasons about is specific to the one-hot "
            "projector; use a train_onehot.py checkpoint.")

    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    from latent_diffusion.diffusion.scheduler import DiffusionScheduler
    from litevae.models import LiteVAEDecoder

    scheduler = DiffusionScheduler()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    vae_ckpt = torch.load(resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                          map_location=device, weights_only=False)
    decoder = LiteVAEDecoder(latent_channels=unet_cfg['latent_channels'],
                             output_channels=3, base_channels=512, num_res_blocks=2)
    decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
    decoder.to(device).eval()

    rng = np.random.default_rng(cfg.seed)

    # assemble the strategies
    strategies = {'pure': dict(fn=strat_pure, kw={})}
    strategies['sigma_matched'] = dict(fn=strat_scaled, kw=dict(match_radius=True))
    for a in cfg.amplify_factors:
        strategies[f'amplified_{a:g}x'] = dict(fn=strat_scaled, kw=dict(scale=a))
    for p in cfg.enrich_purities:
        strategies[f'enriched_{p:.0%}'] = dict(
            fn=strat_enriched, kw=dict(purity=p, n_backgrounds=cfg.enrich_backgrounds))
    strategies['top_carriers'] = dict(
        fn=strat_top_carriers, kw=dict(n_top=cfg.top_carriers_n))

    # generate
    images_by_strategy, rows = {}, []
    real_pca_all = projector.transform(snp_matrix)
    real_radius = np.linalg.norm(real_pca_all, axis=1).mean()

    for name, spec in strategies.items():
        pca_vecs = spec['fn'](projector, snp_matrix, founders, rng, **spec['kw'])
        radius = float(np.linalg.norm(pca_vecs, axis=1).mean())
        sep_pca = float(np.mean([
            np.linalg.norm(pca_vecs[i] - pca_vecs[j])
            for i, j in combinations(range(len(founders)), 2)]))

        print(f"\n{name}: PCA radius {radius:.1f} "
              f"(real {real_radius:.1f}), founder separation {sep_pca:.1f}")

        imgs = generate_from_pca(snp_encoder, unet, scheduler, decoder, pca_vecs,
                                 [cfg.seed] * len(founders), device, latent_shape,
                                 cfg.sampling_steps)
        images_by_strategy[name] = imgs
        for k, img in zip(founders, imgs):
            from PIL import Image
            Image.fromarray(img).save(out / 'images' / f'{name}_founder{k}.png')

        rmses = [float(np.sqrt(((imgs[i].astype(np.float64) -
                                 imgs[j].astype(np.float64)) ** 2).mean()))
                 for i, j in combinations(range(len(founders)), 2)]
        rows.append({
            'strategy': name,
            'pca_radius': radius,
            'pca_founder_separation': sep_pca,
            'mean_pairwise_rmse': float(np.mean(rmses)),
            'min_pairwise_rmse': float(np.min(rmses)),
            'max_pairwise_rmse': float(np.max(rmses)),
        })
        print(f"  image separation: mean pairwise RMSE {np.mean(rmses):.2f}")

    # guidance sweep
    # Applied on top of the best DIRECTION-based strategy rather than on
    # `pure`: guidance amplifies whatever founder-specific signal is already
    # there, so starting from the strategy with the most of it compounds,
    # while starting from pure would amplify a weaker signal.
    if cfg.guidance_scales:
        g_spec = strategies[cfg.guidance_strategy]
        g_vecs = g_spec['fn'](projector, snp_matrix, founders, rng, **g_spec['kw'])
        print(f"\nGuidance sweep on '{cfg.guidance_strategy}':")
        for w in cfg.guidance_scales:
            name = f'{cfg.guidance_strategy}_cfg{w:g}'
            imgs = generate_from_pca(
                snp_encoder, unet, scheduler, decoder, g_vecs,
                [cfg.seed] * len(founders), device, latent_shape,
                cfg.sampling_steps, guidance_scale=w)
            images_by_strategy[name] = imgs
            for k, img in zip(founders, imgs):
                from PIL import Image
                Image.fromarray(img).save(out / 'images' / f'{name}_founder{k}.png')

            rmses = [float(np.sqrt(((imgs[i].astype(np.float64) -
                                     imgs[j].astype(np.float64)) ** 2).mean()))
                     for i, j in combinations(range(len(founders)), 2)]
            rows.append({
                'strategy': name,
                'pca_radius': float(np.linalg.norm(g_vecs, axis=1).mean()),
                'pca_founder_separation': float(np.mean([
                    np.linalg.norm(g_vecs[i] - g_vecs[j])
                    for i, j in combinations(range(len(founders)), 2)])),
                'mean_pairwise_rmse': float(np.mean(rmses)),
                'min_pairwise_rmse': float(np.min(rmses)),
                'max_pairwise_rmse': float(np.max(rmses)),
            })
            print(f"  w={w:g}: mean pairwise RMSE {np.mean(rmses):.2f}")

    # reference: 8 random REAL genotypes
    ref_idx = rng.choice(len(sample_names), len(founders), replace=False)
    ref_imgs = generate_from_pca(
        snp_encoder, unet, scheduler, decoder, real_pca_all[ref_idx],
        [cfg.seed] * len(founders), device, latent_shape, cfg.sampling_steps)
    ref_rmse = float(np.mean([
        np.sqrt(((ref_imgs[i].astype(np.float64) - ref_imgs[j].astype(np.float64)) ** 2).mean())
        for i, j in combinations(range(len(founders)), 2)]))
    print(f"\nReference - 8 random real genotypes: mean pairwise RMSE {ref_rmse:.2f}")

    summary = pd.DataFrame(rows).sort_values('mean_pairwise_rmse', ascending=False)
    summary['fraction_of_real_reference'] = summary['mean_pairwise_rmse'] / ref_rmse
    summary.to_csv(out / 'strategy_comparison.csv', index=False)

    save_strategy_grid(images_by_strategy, list(strategies), founders,
                       out / 'strategy_grid.png')
    save_separation_figure(summary, ref_rmse, out / 'strategy_separation.png')

    with open(out / 'summary.json', 'w') as f:
        json.dump({'checkpoint': str(ckpt_path),
                   'real_reference_rmse': ref_rmse,
                   'real_pca_radius': float(real_radius),
                   'strategies': rows}, f, indent=2)

    # report
    print(f"{'strategy':<20}{'PCA radius':>12}{'PCA sep':>10}"
          f"{'image RMSE':>12}{'vs real':>10}")
    for _, r in summary.iterrows():
        print(f"{r['strategy']:<20}{r['pca_radius']:>12.1f}"
              f"{r['pca_founder_separation']:>10.1f}"
              f"{r['mean_pairwise_rmse']:>12.2f}"
              f"{r['fraction_of_real_reference']:>9.0%}")
    print(f"{'(8 real genotypes)':<20}{real_radius:>12.1f}{'-':>10}"
          f"{ref_rmse:>12.2f}{'100%':>10}")

    best = summary.iloc[0]
    pure_rmse = float(summary.loc[summary.strategy == 'pure',
                                  'mean_pairwise_rmse'].iloc[0])
    print(f"\n  best: {best['strategy']} at {best['mean_pairwise_rmse']:.2f} RMSE, "
          f"{best['mean_pairwise_rmse'] / max(pure_rmse, 1e-9):.1f}x the pure baseline")
    print("\n  sigma_matched and amplified_* are CONTROLS: they change only the")
    print("  conditioning vector's length, and the encoder's leading LayerNorm")
    print("  discards length, so they should land on top of `pure`. Their landing")
    print("  there confirms amplitude is not the lever and is not a failure.")
    print("  The strategies that can actually help are the ones that change")
    print("  DIRECTION - top_carriers and enriched_*, which build the founder")
    print("  vector out of real genotypes instead of a synthetic uniform one.")
    print("\n  Prefer top_carriers for anything you intend to publish: it is the")
    print("  empirical centroid of the real genotypes richest in each founder,")
    print("  so unlike a pure-founder vector it corresponds to genotypes that")
    print("  actually exist, and the model was trained in that region.")
    print("\n  The *_cfg* rows amplify the UNet's output rather than its input,")
    print("  which is the one form of amplification LayerNorm cannot cancel.")
    print("  Note the real-genotype reference is itself the ceiling here: it is")
    print("  what genotype conditioning is worth in this model at all. Guidance")
    print("  above w=1 can exceed it, but exceeding it means the founders are")
    print("  now MORE separated than real genotypes are, which is an")
    print("  extrapolation rather than a measurement - inspect those images for")
    print("  artefacts before trusting them.")

    print(f"\nWrote strategy_grid.png, strategy_separation.png, "
          f"strategy_comparison.csv and per-image PNGs to {out}")


if __name__ == '__main__':
    main()
