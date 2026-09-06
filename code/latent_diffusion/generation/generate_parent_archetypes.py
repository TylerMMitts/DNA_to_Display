# What each of the eight founder parents is predicted to look like.
#
# No real plant is a pure founder, so these are extrapolations; the script
# reports how far outside the real population each one sits, and optionally
# measures traits on every generated image with the segmentation model.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_ONEHOT_DIR, LITEVAE_MODEL, RESULTS_DIR, SEGMENTATION_MODEL,
    SNP_PARQUET, find_latest_checkpoint, resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.generation.generate_from_dataset import generate_batch


# Synthetic genotypes

def pure_parent_vector(n_snps, parent, dtype=np.float32):
    # Every locus assigned to one founder.
    return np.full(n_snps, float(parent), dtype=dtype)


def enriched_vector(base_vector, parent, purity, rng):
    # Replaces a random `purity` fraction of loci with the founder's value.
    #
    # Categorical substitution rather than numeric blending: the 1-8 codes are
    # founder labels, so an averaged value like 4.5 would not correspond to any
    # founder and would be a meaningless input dressed up as a valid one.
    out = np.array(base_vector, dtype=np.float32, copy=True)
    n = len(out)
    k = int(round(purity * n))
    if k <= 0:
        return out
    idx = rng.choice(n, size=min(k, n), replace=False)
    out[idx] = float(parent)
    return out


def pca_reconstruction_error(pca, vectors):
    # Share of a vector's deviation that the retained components fail to keep.
    #
    # This has to be checked before the sigma distances below can be believed. The
    # encoder only ever sees the PCA projection, so if a synthetic vector's
    # distinctiveness lived mostly in the discarded components, it would project
    # close to the population centre and score a small sigma while actually being
    # unlike anything real - the number would say "ordinary" for the wrong reason.
    # Comparing this against the same figure for real genotypes settles it: a
    # similar error means the components represent both equally well.
    vectors = np.atleast_2d(vectors)

    if hasattr(pca, 'locus_contributions'):
        # One-hot projector: PCA was fit on the L*F one-hot expansion, not the
        # raw L-length founder codes, so its inverse_transform lands back in
        # that same expanded space. Comparing that against raw codes 1-8
        # directly would be comparing incommensurable things - one-hot-encode
        # `vectors` first so both sides of the residual live in the same
        # space PCA actually operates in.
        from latent_diffusion.models.snp_encoding import one_hot_founders
        target = one_hot_founders(vectors, pca.founders)
    else:
        target = vectors

    reconstructed = pca.inverse_transform(pca.transform(vectors))
    residual = np.linalg.norm(target - reconstructed, axis=1)
    total = np.linalg.norm(target - pca.mean_, axis=1)
    return residual / np.where(total > 0, total, 1.0)


def ood_distance(pca, population_scores, vector):
    # How far a synthetic vector sits from the real population, in PCA space.
    #
    # Reported in units of each component's own population standard deviation and
    # summarised as an RMS z-score, so it is comparable across components with
    # very different scales. Real genotypes are scored the same way for reference,
    # because the number only means something next to what a genuine genotype
    # scores.
    score = pca.transform(vector.reshape(1, -1))[0]
    mean = population_scores.mean(axis=0)
    std = population_scores.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (score - mean) / std
    return float(np.sqrt((z ** 2).mean())), float(np.abs(z).max())


# Figures

def save_side_by_side(images, labels, save_path, title, sublabels=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(2.15 * n, 2.9), squeeze=False)
    for i, (img, label) in enumerate(zip(images, labels)):
        ax = axes[0][i]
        ax.imshow(img)
        ax.set_title(label, fontsize=11)
        if sublabels is not None:
            ax.set_xlabel(sublabels[i], fontsize=8.5, color='0.35')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_multi_seed_grid(images_by_seed, parents, seeds, save_path,
                         overlays_by_seed=None, traits_by_seed=None):
    # Founder x seed grid, each cell optionally followed by its segmented
    # overlay directly underneath, with vessel/root/stele stats under that.
    #
    # overlays_by_seed, when given, is {seed: [overlay per parent]} in the same
    # parent order as images_by_seed - each seed then occupies TWO rows (image,
    # then its segmentation) instead of one, so the overlay sits immediately
    # under the image it was measured on rather than in a separate figure where
    # that correspondence has to be inferred. None reproduces the original
    # image-only layout, so a caller that could not run segmentation (missing
    # weights, cfg.segment off) still gets a usable figure instead of a crash.
    #
    # traits_by_seed, when given, is {seed: trait_df subset indexed by parent}
    # - a DataFrame per seed so a lookup is by PARENT LABEL rather than by
    #   position, the same reasoning as the sublabels in parents_segmented.png:
    #   a positional zip would silently mismatch a label to the wrong image if
    #   either list were ever reordered independently. Only used when
    #   overlays_by_seed is also given, since a trait sublabel with no
    #   segmented image above it to belong to would be a label floating under
    #   a photo instead of the mask it was measured on.
    n_seeds, n_cols = len(seeds), len(parents)
    rows_per_seed = 2 if overlays_by_seed is not None else 1
    n_rows = n_seeds * rows_per_seed
    # Extra vertical room per segmented row for the 3-line trait sublabel -
    # without it, tight_layout has to compress the grid to fit the text and
    # the images end up noticeably smaller than the plain-overlay version.
    extra_height = 0.55 * n_seeds if traits_by_seed is not None else 0.0

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.05 * n_cols, 2.15 * n_rows + extra_height),
                             squeeze=False)
    for si, seed in enumerate(seeds):
        img_row = si * rows_per_seed
        for c, parent in enumerate(parents):
            ax = axes[img_row][c]
            ax.imshow(images_by_seed[seed][c])
            ax.set_xticks([]); ax.set_yticks([])
            if img_row == 0:
                ax.set_title(f'parent {parent}', fontsize=10.5)
            if c == 0:
                ax.set_ylabel(f'seed {seed}', fontsize=9.5, rotation=0,
                              ha='right', va='center', labelpad=26)

        if overlays_by_seed is not None:
            seg_row = img_row + 1
            for c, parent in enumerate(parents):
                ax = axes[seg_row][c]
                ax.imshow(overlays_by_seed[seed][c])
                ax.set_xticks([]); ax.set_yticks([])
                if traits_by_seed is not None:
                    ax.set_xlabel(
                        format_trait_sublabel(traits_by_seed[seed].loc[parent]),
                        fontsize=7, color='0.35')
                if c == 0:
                    ax.set_ylabel('segmented', fontsize=8, rotation=0,
                                  ha='right', va='center', labelpad=26)

    title = ('Founder archetypes across noise seeds\n'
            'differences down a column are the founder; '
            'differences across a row are the noise')
    if overlays_by_seed is not None:
        title += '\nred = root, green = stele, blue = vessel'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if traits_by_seed is not None:
        # tight_layout leaves the same gap between every pair of rows, which is
        # not enough for the 3-line sublabel under a segmented row: on every
        # row but the last, the 'stele' line lands under the next seed's images
        # and is clipped away. Widening the row gap afterwards gives each label
        # its own space. Applied only when there are labels to make room for,
        # so the unlabelled layouts keep their original tighter spacing.
        fig.subplots_adjust(hspace=0.42)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_purity_sweep(images_by_parent, parents, purities, save_path):
    n_rows, n_cols = len(parents), len(purities)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.95 * n_cols, 2.05 * n_rows),
                             squeeze=False)
    for r, parent in enumerate(parents):
        for c, p in enumerate(purities):
            ax = axes[r][c]
            ax.imshow(images_by_parent[parent][c])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'{p:.0%}', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'parent {parent}', fontsize=9.5, rotation=0,
                              ha='right', va='center', labelpad=30)
    fig.suptitle('Founder enrichment sweep\n'
                 'fraction of loci assigned to that founder '
                 '(~12.5% is population-typical, 100% is far out of distribution)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def format_trait_sublabel(row):
    # Vessel count, root diameter, and stele diameter as a compact 3-line
    # label.
    #
    # NaN is shown as 'n/a' rather than formatted as a number - a founder vector
    # can be far enough out of distribution that the segmenter finds no root or
    # stele at all, which is itself worth seeing rather than hiding behind a
    # misleadingly precise-looking 'nan px'.
    def fmt(v):
        return f'{v:.0f}px' if np.isfinite(v) else 'n/a'
    return (f"{int(row['vessel_count_cc'])} vessels\n"
           f"root: {fmt(row['root_diameter_px'])}\n"
           f"stele: {fmt(row['stele_diameter_px'])}")


def save_trait_comparison_figure(trait_df, parents, save_path):
    # Vessel count, root diameter, and stele diameter across founders.
    #
    # Built from every seed, not just the one shown in parents_segmented.png, so
    # the error bars show how much a founder's measured traits vary with noise
    # alone - a founder with a wide bar is one whose single-image trait numbers
    # should not be read too literally.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = [('vessel_count_cc', 'Vessel count', axes[0]),
            ('root_diameter_px', 'Root diameter (px)', axes[1]),
            ('stele_diameter_px', 'Stele diameter (px)', axes[2])]

    for col, title, ax in specs:
        means = [trait_df.loc[trait_df.parent == k, col].mean() for k in parents]
        stds = [trait_df.loc[trait_df.parent == k, col].std() for k in parents]
        ax.bar([str(k) for k in parents], means,
              yerr=[s if np.isfinite(s) else 0 for s in stds],
              capsize=3, color='steelblue')
        ax.set_xlabel('founder')
        ax.set_title(title, fontsize=11)

        # bar() autoscales the x-axis from the bar heights, and with every
        # height NaN (a founder's vector can be so far out of distribution
        # that no root/stele is detected in any seed) that collapses to a
        # sliver around position 0, hiding every category tick beyond the
        # first even though they are still technically registered. Pinning
        # xlim explicitly makes all `len(parents)` categories visible
        # regardless of which values are missing. Verified by reproducing
        # the all-NaN case directly: get_xticklabels() lists all 8 ticks even
        # though only the first was actually visible before this fix.
        ax.set_xlim(-0.5, len(parents) - 0.5)
        if all(not np.isfinite(m) for m in means):
            ax.text(0.5, 0.5, 'not detected\nfor any founder', ha='center',
                    va='center', transform=ax.transAxes, color='0.5', fontsize=9)

    fig.suptitle(f'Measured traits by founder (mean +/- std across '
                f'{trait_df.seed.nunique()} seeds)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_ood_figure(ood_df, real_rms, save_path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(ood_df['parent'].astype(str), ood_df['rms_z'], color='steelblue',
           label='pure founder vector')
    ax.axhline(real_rms, color='crimson', ls='--', lw=1.6,
               label=f'mean real genotype ({real_rms:.2f})')
    ax.set_xlabel('founder')
    ax.set_ylabel('distance from population (RMS z across PCA components)')
    ax.set_title('How far outside the training distribution each founder sits')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/generation/generate_parent_archetypes.py
    class cfg:
        # find_latest_checkpoint() picks the highest numbered checkpoint in this
        # folder automatically - point it at whichever weights you want
        # archetypes from, numeric or one-hot, rather than naming an epoch here.
        checkpoint_dir = DIFFUSION_ONEHOT_DIR
        litevae_checkpoint = LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'   # legacy path only
        seg_weights = SEGMENTATION_MODEL

        # Separate from the numeric-model run's output so both sets of
        # archetypes stay available side by side for comparison.
        output_dir = RESULTS_DIR / 'parent_archetypes_onehot'

        # Founder labels present in the data. None -> detected automatically
        # from the unique SNP values.
        parents = None

        # Seeds for the multi-seed grid. The first is used for the main
        # side-by-side figure. Every founder uses the same seeds.
        seeds = [0, 1, 2, 3, 4]

        # Enrichment sweep. None disables it.
        purities = [0.125, 0.25, 0.5, 0.75, 1.0]

        segment = True           # measure traits on each founder image

        sampling_steps = 50
        imgsz = 256
        latent_size = 32
        batch_size = 8
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    # data
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    n_snps = snp_matrix.shape[1]

    parents = cfg.parents
    if parents is None:
        parents = sorted(int(v) for v in np.unique(snp_matrix) if v > 0)
    print(f"Founders detected: {parents}")

    composition = {k: float((snp_matrix == k).mean()) for k in parents}
    print("Mean share of loci per founder in real genotypes: " +
          ", ".join(f"{k}:{v:.1%}" for k, v in composition.items()))

    # model
    checkpoint_path = find_latest_checkpoint(
        resolve_input(cfg.checkpoint_dir, 'checkpoint directory'))
    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
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

    # how far out of distribution is each founder?
    ood_rows, real_rms = [], float('nan')
    if snp_encoder.pca is not None:
        pca = snp_encoder.pca
        population_scores = pca.transform(snp_matrix)
        real_z = []
        mean, std = population_scores.mean(axis=0), population_scores.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        for s in population_scores:
            real_z.append(np.sqrt((((s - mean) / std) ** 2).mean()))
        real_rms = float(np.mean(real_z))

        real_recon = pca_reconstruction_error(pca, snp_matrix)
        for k in parents:
            vec = pure_parent_vector(n_snps, k)
            rms, mx = ood_distance(pca, population_scores, vec)
            ood_rows.append({
                'parent': k, 'rms_z': rms, 'max_abs_z': mx,
                'population_share': composition[k],
                'pca_reconstruction_error': float(pca_reconstruction_error(pca, vec)[0]),
            })
        ood_df = pd.DataFrame(ood_rows)
        ood_df.to_csv(out / 'out_of_distribution.csv', index=False)
        save_ood_figure(ood_df, real_rms, out / 'out_of_distribution.png')
    else:
        ood_df = pd.DataFrame()
        print("Encoder has no PCA; skipping the out-of-distribution check.")

    # generate the pure founders
    pure = np.stack([pure_parent_vector(n_snps, k) for k in parents])
    pure_t = torch.tensor(pure, dtype=torch.float32, device=device)

    images_by_seed = {}
    for seed in cfg.seeds:
        print(f"Generating founders at seed {seed}...")
        imgs = []
        for start in range(0, len(parents), cfg.batch_size):
            chunk = pure_t[start:start + cfg.batch_size]
            # Same seed for every founder in the batch, so the only thing that
            # differs between panels is the conditioning.
            imgs.extend(generate_batch(
                snp_encoder, unet, scheduler, decoder, chunk,
                [seed] * chunk.shape[0], device, latent_shape, cfg.sampling_steps))
        images_by_seed[seed] = imgs
        for k, img in zip(parents, imgs):
            Image.fromarray(img).save(out / 'images' / f'parent_{k}_seed{seed}.png')

    main_seed = cfg.seeds[0]
    sublabels = None
    if len(ood_df):
        sublabels = [f"{ood_df.loc[ood_df.parent == k, 'rms_z'].iloc[0]:.1f} sigma from population"
                     for k in parents]
    save_side_by_side(images_by_seed[main_seed], [f'parent {k}' for k in parents],
                      out / 'parents_side_by_side.png',
                      f'MexiMAGIC founder archetypes (all loci set to one founder, '
                      f'seed {main_seed})', sublabels)
    # traits
    # Segmentation runs before the multi-seed grid (rather than after, as in
    # an earlier version of this script) because that grid now wants an
    # overlay for every seed, not just main_seed - overlays_by_seed collects
    # one per (seed, parent) instead of only keeping main_seed's.
    trait_rows = []
    overlays_by_seed = None
    traits_by_seed = None
    if cfg.segment:
        try:
            from ultralytics import YOLO
            from feature_segmentation.evaluation.reconstruction_fidelity_test import measure, overlay
            seg = YOLO(str(resolve_input(cfg.seg_weights, 'segmentation weights')))
        except (SystemExit, FileNotFoundError, ImportError) as exc:
            print(f"Skipping segmentation: {exc}")
            seg = None

        if seg is not None:
            overlays_by_seed = {seed: [] for seed in cfg.seeds}
            for seed in cfg.seeds:
                for k, img in zip(parents, images_by_seed[seed]):
                    result = seg.predict(img[:, :, ::-1], conf=0.25, imgsz=cfg.imgsz,
                                         device=cfg.device, verbose=False)[0]
                    traits, masks = measure(result, cfg.imgsz)
                    trait_rows.append({'parent': k, 'seed': seed, **traits})
                    overlays_by_seed[seed].append(overlay(img, masks))

            trait_df = pd.DataFrame(trait_rows)
            trait_df.to_csv(out / 'traits.csv', index=False)

            # Indexed by parent so sublabels line up with the overlays, which
            # were appended in `parents` order - a plain positional zip would
            # silently mismatch labels to images if either list were ever
            # reordered independently. Same reasoning applies per-seed for
            # the multi-seed grid's trait sublabels below.
            traits_by_seed = {seed: trait_df[trait_df.seed == seed].set_index('parent')
                              for seed in cfg.seeds}
            main_rows = traits_by_seed[main_seed]
            sub = [format_trait_sublabel(main_rows.loc[k]) for k in parents]
            save_side_by_side(overlays_by_seed[main_seed], [f'parent {k}' for k in parents],
                              out / 'parents_segmented.png',
                              f'Founder archetypes, segmented (seed {main_seed})\n'
                              'red = root, green = stele, blue = vessel', sub)

            save_trait_comparison_figure(trait_df, parents,
                                         out / 'parents_trait_comparison.png')

            print("\n=== Traits per founder (mean +/- std across seeds) ===")
            agg = trait_df.groupby('parent').agg(
                root_diam_mean=('root_diameter_px', 'mean'),
                root_diam_std=('root_diameter_px', 'std'),
                stele_diam_mean=('stele_diameter_px', 'mean'),
                stele_diam_std=('stele_diameter_px', 'std'),
                vessels_mean=('vessel_count_cc', 'mean'),
                vessels_std=('vessel_count_cc', 'std'),
                vessel_area_mean=('vessel_total_area_px', 'mean'))
            print(agg.to_string(float_format=lambda v: f'{v:.1f}'))

    save_multi_seed_grid(images_by_seed, parents, cfg.seeds,
                         out / 'parents_multi_seed.png',
                         overlays_by_seed=overlays_by_seed,
                         traits_by_seed=traits_by_seed)

    # purity sweep
    if cfg.purities:
        print("\nGenerating enrichment sweep...")
        rng = np.random.default_rng(main_seed)
        # Start from the population-typical genotype so purity 0 is a real
        # mixture rather than an arbitrary founder.
        base = snp_matrix[rng.integers(len(snp_matrix))]

        sweep_images = {}
        for k in parents:
            vectors = np.stack([enriched_vector(base, k, p, rng) for p in cfg.purities])
            vt = torch.tensor(vectors, dtype=torch.float32, device=device)
            imgs = []
            for start in range(0, len(cfg.purities), cfg.batch_size):
                chunk = vt[start:start + cfg.batch_size]
                imgs.extend(generate_batch(
                    snp_encoder, unet, scheduler, decoder, chunk,
                    [main_seed] * chunk.shape[0], device, latent_shape,
                    cfg.sampling_steps))
            sweep_images[k] = imgs
        save_purity_sweep(sweep_images, parents, cfg.purities,
                          out / 'purity_sweep.png')

    # summary
    summary = {
        'checkpoint': str(checkpoint_path),
        'parents': parents,
        'seeds': cfg.seeds,
        'population_share_per_parent': composition,
        'mean_real_genotype_rms_z': real_rms,
        'pure_parent_rms_z': (ood_df.set_index('parent')['rms_z'].to_dict()
                              if len(ood_df) else {}),
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    if len(ood_df):
        print(f"\n=== Out-of-distribution check ===")
        print(f"  mean real genotype:  {real_rms:.2f} sigma from the population centre")
        for _, r in ood_df.iterrows():
            print(f"  pure parent {int(r['parent'])}:     {r['rms_z']:.2f} sigma"
                  f"   (max single component {r['max_abs_z']:.1f},"
                  f" PCA residual {r['pca_reconstruction_error']:.2f})")
        print(f"\n  PCA residual for real genotypes: mean {real_recon.mean():.2f}, "
              f"range {real_recon.min():.2f}-{real_recon.max():.2f}")
        print("  If the founders' residuals sit in that range, the components represent")
        print("  them as faithfully as real genotypes and the sigma values can be")
        print("  trusted. A much higher residual would mean a founder looks ordinary")
        print("  only because what makes it unusual was discarded before the encoder.")
        print("\n  Either way a pure founder is a genotype that does not exist here -")
        print("  the largest real single-founder share is about 29% - so treat these")
        print("  images as extrapolations.")

    print(f"\nWrote parents_side_by_side.png, parents_multi_seed.png, "
          f"parents_segmented.png, parents_trait_comparison.png, purity_sweep.png, "
          f"out_of_distribution.png, traits.csv and per-image PNGs to {out}")


if __name__ == '__main__':
    main()
