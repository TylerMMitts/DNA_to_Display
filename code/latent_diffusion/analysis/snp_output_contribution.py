# Where in the FINAL 256x256 image does one SNP change what the model draws?
#
# analyze_snp_spatial_contribution.py answers a nearby question in attention
# space, where the grid is 16x16 at best. Stretching that to 256x256 does not
# create detail: 256 real numbers get spread over 65,536 pixels, and attention
# is a proxy for influence rather than influence itself.
#
# This measures the thing directly instead. Generate a genotype's image, flip
# one locus to a different founder, generate again from the SAME noise seed,
# and subtract. Every pixel that moved, moved because of that locus - so the
# difference map is a true 256x256 answer with no interpolation anywhere.
#
# The upsampled attention map is still drawn beside it, clearly labelled, so
# the two views can be compared.
#
# Usage
#     python code/latent_diffusion/analysis/snp_output_contribution.py

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
    DIFFUSION_ONEHOT_MODEL, RESULTS_DIR, SEGMENTATION_MODEL, SNP_PARQUET,
    resolve_input, resolve_output,
    apply_overrides,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.utils import attention_analysis as aa
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.analysis.analyze_pca_sensitivity import population_sensitivity
from latent_diffusion.analysis.analyze_snp_spatial_contribution import (
    locus_influence_scores, counterfactual_map,
)

CLASS_NAMES = {0: 'root', 1: 'stele', 2: 'vessel'}


# Generation, kept in float
#
# generate_from_dataset.generate_batch rounds to uint8 on the way out, which is
# fatal here: a single-locus change moves pixels by far less than one 0-255
# step, so quantising would floor most of the signal to zero. This is the same
# sampling loop returning the decoder's float output untouched.
@torch.no_grad()
def generate_float(snp_encoder, unet, scheduler, decoder, snp_batch, seeds,
                   device, latent_shape, num_steps):
    B = snp_batch.shape[0]
    z_t = torch.empty(B, *latent_shape, device=device)
    for i, seed in enumerate(seeds):
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        z_t[i] = torch.randn(*latent_shape, generator=generator)

    snp_embedding = snp_encoder(snp_batch)
    timesteps = scheduler.get_timesteps(num_steps, device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
        t_prev = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)

        noise_pred = unet(z_t, t_batch, snp_embedding)
        z_t = scheduler.denoise_step(z_t, noise_pred, t_batch, t_prev_batch)

    images = decoder(z_t, save_steps=False)
    images = torch.clamp((images + 1) / 2, 0, 1)
    return images.permute(0, 2, 3, 1).cpu().numpy()          # [B, H, W, 3] float


# Builds the flipped genotype in raw founder-code space.
#
# The perturbation is done on the raw 1-8 codes rather than in PCA space, which
# is both simpler and exactly equivalent: one-hot encoding is elementwise and
# the PCA projection is affine, so encoding a modified code vector gives the
# same point as adding that locus's loading-column difference to the original
# projection. Raw codes are also what the encoder's forward() already expects.
def flipped_codes(snp_vector, locus, founder_new, n_loci, block_size=1):
    half = block_size // 2
    lo = max(0, locus - half)
    hi = min(n_loci, lo + block_size)
    out = np.array(snp_vector, dtype=np.float32, copy=True)
    out[lo:hi] = float(founder_new)
    return out, int((np.asarray(snp_vector)[lo:hi] != founder_new).sum())


# Mean absolute and signed pixel change caused by the flip, over several
# genotypes.
#
# Averaging across genotypes matters: a single genotype's difference carries
# that genotype's own idiosyncrasy, and what is wanted is the part attributable
# to the locus itself. Both the baseline and the flipped run use the same seed
# per genotype, so the noise draw cancels exactly.
def output_difference_map(snp_encoder, unet, scheduler, decoder, snp_matrix,
                          genotype_indices, locus, founder_new, device,
                          latent_shape, num_steps, seed, block_size,
                          batch_size=8):
    n_loci = snp_matrix.shape[1]
    base_rows, pert_rows, seeds, n_changed = [], [], [], 0

    for k, gi in enumerate(genotype_indices):
        base = np.asarray(snp_matrix[gi], dtype=np.float32)
        pert, changed = flipped_codes(base, locus, founder_new, n_loci, block_size)
        base_rows.append(base)
        pert_rows.append(pert)
        seeds.append(seed + k)
        n_changed += 1 if changed else 0

    baseline, perturbed = [], []
    for start in range(0, len(base_rows), batch_size):
        sl = slice(start, start + batch_size)
        s = seeds[sl]
        b = torch.tensor(np.stack(base_rows[sl]), dtype=torch.float32, device=device)
        p = torch.tensor(np.stack(pert_rows[sl]), dtype=torch.float32, device=device)
        baseline.append(generate_float(snp_encoder, unet, scheduler, decoder,
                                       b, s, device, latent_shape, num_steps))
        perturbed.append(generate_float(snp_encoder, unet, scheduler, decoder,
                                        p, s, device, latent_shape, num_steps))

    baseline = np.concatenate(baseline, axis=0)              # [G, H, W, 3]
    perturbed = np.concatenate(perturbed, axis=0)
    delta = perturbed - baseline                             # [G, H, W, 3]

    # Collapse colour before averaging genotypes: the question is where the
    # image moved, not which channel moved.
    per_genotype = np.abs(delta).mean(axis=-1)               # [G, H, W]
    abs_map = per_genotype.mean(axis=0)                      # [H, W]
    signed_map = delta.mean(axis=-1).mean(axis=0)            # [H, W]
    return baseline, perturbed, abs_map, signed_map, per_genotype, n_changed


# Averages the per-region enrichment over every genotype, each scored against
# its OWN anatomy.
#
# Scoring the genotype-averaged difference map against a single plant's masks
# would be comparing two different things: the map is an average over genotypes
# whose roots differ in size and stele position, so no one plant's segmentation
# describes it. Worse, it makes the whole breakdown hostage to that one
# segmentation - a reference image where the segmenter misses the stele reports
# no stele contribution at all, which is a fact about that image and not about
# the locus.
#
# Genotypes whose segmentation fails for a class simply do not contribute to
# that class's mean, and the count is returned so a thin average is visible.
def regions_over_genotypes(seg_model, baseline, per_genotype, imgsz, conf):
    acc, counts = {}, {}
    for img_f, m in zip(baseline, per_genotype):
        img = (img_f * 255).round().astype(np.uint8)
        stats = contribution_by_region(seg_model, img, m, imgsz, conf)
        for k, v in stats.items():
            if not np.isfinite(v):
                continue
            acc[k] = acc.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1

    out = {k: acc[k] / counts[k] for k in acc}
    for name in CLASS_NAMES.values():
        out[f'{name}_n_genotypes_measured'] = counts.get(f'{name}_enrichment', 0)
    return out


# Splits one genotype's difference map by anatomy, which is what turns a
# heatmap into a sentence. Returns mean |change| inside each class mask, the
# share of the total falling there, and that share divided by the mask's area
# share.
#
# Two things about these classes, both measured rather than assumed:
#
# The masks are very nearly disjoint rather than nested - the 'root' mask
# overlaps the stele by about 3% of the stele's area, so in practice 'root'
# means the cortex OUTSIDE the stele, not the whole cross-section. Read
# root_share as cortex.
#
# Vessels do overlap the stele (~27% of vessel area sits inside it), so the
# shares are not a strict partition and can sum slightly over what they cover.
# The outside_share closes the rest of the gap: it is everything falling on no
# mask at all, which is mostly background around the root.
#
# Masks are not hole-filled here, unlike the trait measurements: region shares
# want the cortex and stele as separate regions, which is what the unfilled
# masks already are.
def contribution_by_region(seg_model, image_rgb, abs_map, imgsz, conf):
    from feature_segmentation.evaluation.reconstruction_fidelity_test import segmenter_input
    pixels, seg_imgsz = segmenter_input(seg_model, image_rgb)
    result = seg_model.predict(pixels[:, :, ::-1], conf=conf, imgsz=seg_imgsz,
                               verbose=False)[0]
    out = {}
    if result.masks is None or len(result.masks.data) == 0:
        return out

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    total = float(abs_map.sum()) or 1.0
    covered = np.zeros(abs_map.shape, dtype=bool)

    for cls_id, name in CLASS_NAMES.items():
        sel = [m for m, c in zip(masks, classes) if int(c) == cls_id]
        if not sel:
            continue
        union = np.zeros(masks.shape[1:], dtype=bool)
        for m in sel:
            union |= m > 0.5
        if union.shape != abs_map.shape:
            union = aa.resize_map(union.astype(np.float32), abs_map.shape[0]) > 0.5
        if not union.any():
            continue
        covered |= union
        area_frac = union.sum() / abs_map.size
        share = float(abs_map[union].sum() / total)
        out[f'{name}_mean_abs'] = float(abs_map[union].mean())
        out[f'{name}_share'] = share
        out[f'{name}_area_px'] = int(union.sum())
        # Share on its own mostly measures how big the tissue is - the cortex
        # covers about half the frame, so it collects about half the change no
        # matter what the locus does. Enrichment divides that out: 1.0 means the
        # change lands there exactly as often as area alone predicts, above 1.0
        # means the locus really is acting on that tissue preferentially.
        out[f'{name}_enrichment'] = float(share / area_frac) if area_frac else float('nan')

    outside = ~covered
    out['outside_share'] = float(abs_map[outside].sum() / total) if outside.any() else 0.0
    out['outside_area_px'] = int(outside.sum())
    return out


# Generates one genotype twice from the same seed and reports how far the two
# runs differ.
#
# This is what makes the difference maps interpretable. A single locus out of
# ~43,788 moves the image very little, so before believing any of it you need
# to know what "unchanged" costs. Sampling is deterministic for a fixed seed, so
# this is expected to come back at exactly zero - and if it ever does not, every
# reported magnitude has to be read against it rather than against zero.
def reproducibility_floor(snp_encoder, unet, scheduler, decoder, snp_matrix,
                          genotype_index, device, latent_shape, num_steps, seed):
    v = torch.tensor(np.asarray(snp_matrix[genotype_index], dtype=np.float32)[None],
                     dtype=torch.float32, device=device)
    a = generate_float(snp_encoder, unet, scheduler, decoder, v, [seed], device,
                       latent_shape, num_steps)
    b = generate_float(snp_encoder, unet, scheduler, decoder, v, [seed], device,
                       latent_shape, num_steps)
    return float(np.abs(a - b).mean()), float(np.abs(a - b).max())


# Draws a heat map over an image so the image stays visible underneath.
#
# A constant-alpha overlay tints every pixel equally, including the ones where
# nothing happened, so the root vanishes behind a uniform wash and the question
# the figure exists to answer - WHERE on the root - stops being readable. Here
# alpha tracks the value instead: transparent where the locus changed nothing,
# opaque only where it did.
#
# gamma > 1 bends that curve so mid values stay faint and only genuine peaks
# turn solid, which matters because these maps have a long tail of small
# nonzero values covering the whole root.
def overlay_heat(ax, base_img, heat, hi, cmap='inferno', gamma=1.8,
                 max_alpha=0.85, contour_at=None):
    ax.imshow(base_img)
    norm = np.clip(heat / (hi or 1.0), 0, 1)

    rgba = matplotlib.colormaps[cmap](norm)
    rgba[..., 3] = (norm ** gamma) * max_alpha
    ax.imshow(rgba)

    # An RGBA image carries no scalar mapping, so a colourbar cannot be built
    # from it directly. This stands in for one, describing the same 0-to-hi
    # range the alpha ramp was built from.
    mappable = plt.cm.ScalarMappable(cmap=cmap,
                                     norm=plt.Normalize(vmin=0.0, vmax=1.0))
    mappable.set_array([])

    # A crisp line around the hottest region, since a soft alpha ramp alone can
    # leave the eye guessing where the peak actually stops.
    if contour_at:
        levels = [float(np.percentile(heat, p)) for p in contour_at]
        levels = sorted(set(l for l in levels if np.isfinite(l) and l > 0))
        if levels:
            ax.contour(heat, levels=levels, colors='white', linewidths=0.7,
                       alpha=0.75)
    return mappable


def save_contribution_figure(snp_name, locus, founder, baseline_img, perturbed_img,
                             abs_map, signed_map, smooth_map, attention_map,
                             attention_native, region_stats, n_genotypes,
                             block_size, layer, smooth_sigma, backdrop,
                             save_path):
    has_attn = attention_map is not None
    has_smooth = smooth_map is not None
    n_cols = 4 + int(has_smooth) + int(has_attn)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.1 * n_cols, 3.7), squeeze=False)
    ax = list(axes[0])

    ax[0].imshow(baseline_img)
    ax[0].set_title(f'generated as-is\n{backdrop}', fontsize=10)

    ax[1].imshow(perturbed_img)
    label = ('locus flipped' if block_size == 1
             else f'{block_size}-locus block flipped')
    ax[1].set_title(f'{label}\nto founder {founder}', fontsize=10)

    # Scaled to the 99th percentile rather than the max. A handful of extreme
    # pixels would otherwise set the colour range and flatten everything else to
    # near-white, hiding the pattern that is actually being looked for.
    lim = float(np.percentile(np.abs(signed_map), 99)) or 1.0
    im2 = ax[2].imshow(signed_map, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax[2].set_title('signed change\nred = brighter', fontsize=10)
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    hi = float(np.percentile(abs_map, 99)) or 1.0
    im3 = overlay_heat(ax[3], baseline_img, abs_map, hi)
    ax[3].set_title('|change| over the image\n(true 256x256)', fontsize=10)
    fig.colorbar(im3, ax=ax[3], fraction=0.046)

    col = 4
    if has_smooth:
        # A single locus moves texture all over the root rather than shifting
        # one structure, so the raw map is speckle. Blurring it does not add
        # information - it pools nearby pixels so the regions where that
        # speckle is densest become visible, which is the "where most" question.
        #
        # The contour rings the top decile, which is the panel that is actually
        # meant to answer "where most" and benefits from a hard edge.
        im4 = overlay_heat(ax[col], baseline_img, smooth_map,
                           float(smooth_map.max()), contour_at=(90, 97))
        ax[col].set_title(f'|change| pooled\n(blurred, sigma={smooth_sigma})',
                          fontsize=10)
        fig.colorbar(im4, ax=ax[col], fraction=0.046)
        col += 1

    if has_attn:
        overlay_heat(ax[col], baseline_img, attention_map,
                     float(attention_map.max()))
        ax[col].set_title(f'attention, {layer}\n'
                          f'(upsampled from {attention_native}x'
                          f'{attention_native})', fontsize=10)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    subtitle = (f'{snp_name}   locus {locus}, flipped to founder {founder}, '
                f'averaged over {n_genotypes} genotypes')
    if region_stats:
        # Enrichment rather than raw share. The cortex covers about half the
        # frame, so it collects about half of any change whatever the locus
        # does; dividing by area is what separates "acts on this tissue" from
        # "this tissue is big".
        label_for = {'root': 'cortex', 'stele': 'stele', 'vessel': 'vessel'}
        parts = [f"{label_for[n]} {region_stats[f'{n}_enrichment']:.2f}x"
                 for n in CLASS_NAMES.values() if f'{n}_enrichment' in region_stats]
        if parts:
            subtitle += ('\nchange per pixel vs. what tissue area alone predicts'
                         ' -  ' + '   '.join(parts))
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# Every locus's pooled map on one row, over a shared reference image.
#
# The per-locus figures are each normalised to their own peak, which is right
# for reading one of them but hides how they compare: two loci with very
# different magnitudes can look identical side by side. This panel normalises
# them all against the strongest map in the set, so a weak locus reads as weak.
# Each title still carries that locus's own mean, since the shared scale makes
# the faint ones hard to judge by eye alone.
def save_locus_comparison(entries, reference_img, save_path, smooth_sigma):
    n = len(entries)
    if n == 0:
        return
    peak = max(float(e['map'].max()) for e in entries) or 1.0

    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.6), squeeze=False)
    for ax, e in zip(axes[0], entries):
        im = overlay_heat(ax, reference_img, e['map'], peak, contour_at=(90, 97))
        ax.set_title(f"{e['name']}\nlocus {e['locus']}, founder {e['founder']}\n"
                     f"mean {e['mean']:.2e}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02)

    fig.suptitle('Where each SNP changes the generated image\n'
                 f'pooled with sigma={smooth_sigma}, all panels on one shared '
                 'scale so magnitudes are comparable', fontsize=12)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/snp_output_contribution.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        litevae_checkpoint = None        # None -> paths.LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        output_dir = RESULTS_DIR / 'snp_output_contribution'

        # Which loci to map, in order of precedence.
        #
        # diverse_csv is usually what you want. select_diverse_snp_maps.py picks
        # loci whose spatial signatures are as UNLIKE each other as possible,
        # where ranking by influence score alone returns near-duplicates: scores
        # vary smoothly along a linked block, so the top of that list is one
        # region seen several times over and every map comes out looking the
        # same. Reading its output here reuses that work, including the founder
        # it probed at each locus.
        #
        # None -> fall back to the influence-score ranking below.
        diverse_csv = RESULTS_DIR / 'snp_diverse_maps' / 'diverse_snp_details.csv'

        # A list of SNP names (e.g. ['Zm00001d027230']) overrides both.
        snp_names = None
        top_n = 6

        # Minimum gap, in loci, between two selected positions.
        #
        # Without this the top-N comes back as a run of neighbours: influence
        # scores vary smoothly along a linked haplotype block, so the highest
        # scores cluster in one region and you get six views of what is
        # effectively the same locus. Spacing them out gives six genuinely
        # different SNPs. Set to 0 to take the raw ranking.
        min_locus_spacing = 200

        # Genotypes averaged per locus. Cost is 2 x this many DDIM samplings per
        # locus, so this is the main thing to turn down for a quick look.
        n_genotypes = 8

        # What the heat maps are drawn on top of.
        #
        # None averages all n_genotypes baselines together, which is the honest
        # backdrop: the heat map is itself an average over those genotypes, so
        # laying it over a single plant would put a population-level measurement
        # on top of one individual's anatomy - the stele in the picture would be
        # that plant's stele, not the one the numbers describe.
        #
        # An integer picks that one genotype instead, which gives a sharper
        # image to look at. Useful for seeing detail, as long as you read the
        # heat as belonging to the population rather than to that root.
        #
        # Either way it is the same backdrop for every locus, and that is not a
        # bug: the baseline is the genotype generated WITHOUT any flip, so it
        # does not depend on which locus is being tested. Only the heat changes.
        reference_genotype = None
        sensitivity_sample = 24

        # 1 flips a single locus. A single SNP moves the 43,788-long genotype
        # vector very little, so the change in the image is real but faint;
        # raising this flips a contiguous haplotype block, which is both a more
        # realistic counterfactual and a much stronger signal. Start at 1 to see
        # the honest single-SNP effect, raise it if the map is too faint to read.
        block_size = 1

        # None -> the rarer of the two most-separating founders at that locus.
        founder = None

        # Gaussian blur width for the pooled panel, in pixels. None drops that
        # panel. Display only - every number reported comes from the raw map.
        smooth_sigma = 6

        # Anatomical breakdown of where the change lands.
        segment = True
        seg_weights = SEGMENTATION_MODEL
        seg_conf = 0.25

        # Attention panel for comparison. None skips it.
        attention_layer = 'up_2'
        attention_timestep = 500

        sampling_steps = 50
        imgsz = 256
        latent_size = 32
        batch_size = 8
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    maps_dir = out / 'maps'
    # Cleared rather than merged into. Which loci get selected depends on the
    # config, so leaving old files behind mixes several different selections in
    # one folder with nothing to say which run each came from.
    if maps_dir.exists():
        stale = list(maps_dir.glob('*.png')) + list(maps_dir.glob('*.npy'))
        for f in stale:
            f.unlink()
        if stale:
            print(f"Cleared {len(stale)} files from a previous run in {maps_dir}")
    maps_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    # data
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    # model
    snp_encoder, unet, unet_cfg = load_model(
        resolve_input(cfg.checkpoint, 'checkpoint'), snp_matrix, device)
    projector = getattr(snp_encoder, 'projector', None)
    if projector is None:
        raise SystemExit(
            "This script needs a one-hot checkpoint: the per-locus flip is "
            "defined on founder codes, which the numeric encoder does not keep "
            "separable. Point cfg.checkpoint at models/diffusion_onehot/.")

    # Checkpoints written before explained_variance_ was stored leave it unset,
    # and the sensitivity weighting needs it. This recovers it from the
    # population rather than substituting a wrong scale.
    projector.ensure_explained_variance(snp_matrix)

    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    from latent_diffusion.diffusion.scheduler import DiffusionScheduler
    from litevae.models import LiteVAEDecoder
    from paths import LITEVAE_MODEL

    scheduler = DiffusionScheduler()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    vae_path = resolve_input(cfg.litevae_checkpoint or LITEVAE_MODEL,
                             'LiteVAE checkpoint')
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    decoder = LiteVAEDecoder(latent_channels=unet_cfg['latent_channels'],
                             output_channels=3, base_channels=512,
                             num_res_blocks=2)
    decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
    decoder.to(device).eval()

    seg_model = None
    if cfg.segment:
        try:
            from ultralytics import YOLO
            seg_model = YOLO(str(resolve_input(cfg.seg_weights,
                                               'segmentation weights')))
        except Exception as exc:
            print(f"Skipping the anatomical breakdown: {exc}")

    # choose loci
    rng = np.random.default_rng(cfg.seed)
    # Founder to probe per locus, when the source names one. Empty means fall
    # back to the rarer-of-the-best-pair rule below.
    founder_for = {}
    if cfg.snp_names:
        name_to_locus = {n: i for i, n in enumerate(snp_names)}
        missing = [n for n in cfg.snp_names if n not in name_to_locus]
        if missing:
            raise SystemExit(f"SNP name(s) not in the matrix: {missing}")
        selected = [name_to_locus[n] for n in cfg.snp_names]
        best_pair = None
    elif cfg.diverse_csv and Path(cfg.diverse_csv).exists():
        d = pd.read_csv(cfg.diverse_csv).sort_values('diversity_rank')
        d = d.head(cfg.top_n)
        selected = [int(x) for x in d['locus']]
        founder_for = {int(r.locus): int(r.probed_founder) for r in d.itertuples()}
        best_pair = None
        print(f"\nUsing {len(selected)} loci from {Path(cfg.diverse_csv).name} "
              "(chosen for spatially DIFFERENT effects, not just the strongest)")
        print(f"  {', '.join(d['snp'].astype(str))}")
    else:
        if cfg.diverse_csv:
            print(f"\n{cfg.diverse_csv} not found - run "
                  "select_diverse_snp_maps.py to get spatially diverse loci. "
                  "Falling back to the influence-score ranking, whose top "
                  "entries tend to be neighbours in one linked block.")
        print(f"\nScoring loci over {cfg.sensitivity_sample} genotypes...")
        sens_idx = rng.choice(len(sample_names), cfg.sensitivity_sample,
                              replace=False)
        sens_df = population_sensitivity(snp_encoder, projector, snp_matrix,
                                         sens_idx, device)
        sensitivity = sens_df['normalized_sensitivity_mean'].to_numpy()
        scores, best_pair = locus_influence_scores(projector, sensitivity)

        order = np.argsort(scores)[::-1]
        if cfg.min_locus_spacing:
            selected = []
            for L in order:
                if all(abs(int(L) - s) >= cfg.min_locus_spacing for s in selected):
                    selected.append(int(L))
                if len(selected) == cfg.top_n:
                    break
            print(f"  {len(selected)} loci selected by influence score, kept at "
                  f"least {cfg.min_locus_spacing} loci apart")
        else:
            selected = [int(L) for L in order[:cfg.top_n]]
            print(f"  top {len(selected)} loci selected by influence score")

    geno_idx = rng.choice(len(sample_names), cfg.n_genotypes, replace=False)

    floor_mean, floor_max = reproducibility_floor(
        snp_encoder, unet, scheduler, decoder, snp_matrix, geno_idx[0], device,
        latent_shape, cfg.sampling_steps, cfg.seed)
    print(f"\nReproducibility floor (same genotype twice, same seed): "
          f"mean {floor_mean:.3e}, max {floor_max:.3e}")
    if floor_mean == 0.0:
        print("  exactly zero - every nonzero pixel below is caused by the flip")
    else:
        print("  NOT zero, so treat this as the noise floor: a locus whose mean")
        print("  change is near it has not been shown to do anything.")

    rows = []
    attention_native = None
    comparison, reference_img = [], None
    for locus in selected:
        name = snp_names[locus]
        if cfg.founder is not None:
            founder = int(cfg.founder)
        elif locus in founder_for:
            # Whatever the diverse selection probed, so the map here measures
            # the same perturbation its signature was chosen for.
            founder = founder_for[locus]
        elif best_pair is not None:
            # The rarer of the two most-separating founders: flipping toward a
            # common founder mostly reproduces the population average.
            fa, fb = best_pair[locus]
            n_a = int((snp_matrix[:, locus] == fa).sum())
            n_b = int((snp_matrix[:, locus] == fb).sum())
            founder = int(fa if n_a <= n_b else fb)
        else:
            present, counts = np.unique(snp_matrix[:, locus], return_counts=True)
            founder = int(present[np.argmin(counts)])

        print(f"\n{name}  (locus {locus}, flipping to founder {founder})")

        baseline, perturbed, abs_map, signed_map, per_genotype, n_changed = output_difference_map(
            snp_encoder, unet, scheduler, decoder, snp_matrix, geno_idx, locus,
            founder, device, latent_shape, cfg.sampling_steps, cfg.seed,
            cfg.block_size, cfg.batch_size)

        # Backdrop for the figures. Averaging the baselines matches the heat
        # map, which is averaged over the same genotypes.
        if cfg.reference_genotype is None:
            base_img, pert_img = baseline.mean(axis=0), perturbed.mean(axis=0)
            backdrop = f'mean of {len(baseline)} genotypes'
        else:
            ref = min(cfg.reference_genotype, len(baseline) - 1)
            base_img, pert_img = baseline[ref], perturbed[ref]
            backdrop = f'genotype {int(geno_idx[ref])}'

        if n_changed == 0:
            print(f"  every sampled genotype already carries founder {founder} "
                  f"here - nothing was flipped, so the map is all zero")

        print(f"  mean |change| {abs_map.mean():.3e}   "
              f"peak {abs_map.max():.3e}   "
              f"({n_changed}/{len(geno_idx)} genotypes actually changed)")

        region = {}
        if seg_model is not None:
            region = regions_over_genotypes(seg_model, baseline, per_genotype,
                                            cfg.imgsz, cfg.seg_conf)
            if region:
                shown = {'root': 'cortex', 'stele': 'stele', 'vessel': 'vessel'}
                bits = [f"{shown[n]} {region[f'{n}_enrichment']:.2f}x"
                        f"[{region.get(f'{n}_n_genotypes_measured', 0)}]"
                        for n in CLASS_NAMES.values() if f'{n}_enrichment' in region]
                print('  enrichment vs tissue area [genotypes measured]: '
                      + '  '.join(bits))
                thin = [shown[n] for n in CLASS_NAMES.values()
                        if region.get(f'{n}_n_genotypes_measured', 0)
                        < len(geno_idx) // 2]
                if thin:
                    print(f"    NOTE: the segmenter found {', '.join(thin)} in "
                          "under half the genotypes, so that figure rests on "
                          "few images")

        attn = None
        if cfg.attention_layer:
            cf, _ = counterfactual_map(
                snp_encoder, unet, projector, snp_matrix, geno_idx, locus,
                founder, cfg.attention_timestep, latent_shape, device, cfg.seed,
                cfg.attention_layer, chunk_size=16, block_size=cfg.block_size)
            attention_native = cf.shape[0]
            attn = aa.resize_map(np.abs(cf).mean(axis=-1), cfg.imgsz)

        smooth = None
        if cfg.smooth_sigma:
            from scipy.ndimage import gaussian_filter
            smooth = gaussian_filter(abs_map, sigma=cfg.smooth_sigma)

        save_contribution_figure(
            name, locus, founder, base_img, pert_img, abs_map, signed_map,
            smooth, attn, attention_native, region, len(geno_idx), cfg.block_size,
            cfg.attention_layer, cfg.smooth_sigma, backdrop,
            out / 'maps' / f'{name}_locus{locus}.png')

        np.save(out / 'maps' / f'{name}_locus{locus}_abs.npy', abs_map)

        if reference_img is None:
            reference_img = base_img
        comparison.append({
            'name': name, 'locus': int(locus), 'founder': founder,
            'mean': float(abs_map.mean()),
            'map': smooth if smooth is not None else abs_map,
        })

        rows.append({
            'snp_name': name, 'locus': int(locus), 'founder': founder,
            'n_genotypes': len(geno_idx), 'n_genotypes_changed': n_changed,
            'block_size': cfg.block_size,
            'mean_abs_change': float(abs_map.mean()),
            'peak_abs_change': float(abs_map.max()),
            'signed_mean': float(signed_map.mean()),
            **region,
        })

    if comparison and reference_img is not None:
        save_locus_comparison(comparison, reference_img,
                              out / 'locus_comparison.png', cfg.smooth_sigma)

    df = pd.DataFrame(rows)
    df.to_csv(out / 'output_contribution.csv', index=False)

    with open(out / 'summary.json', 'w') as f:
        json.dump({
            'checkpoint': str(cfg.checkpoint),
            'n_loci': len(rows), 'n_genotypes': int(cfg.n_genotypes),
            'block_size': cfg.block_size,
            'sampling_steps': cfg.sampling_steps,
            'image_size': cfg.imgsz,
            'attention_layer': cfg.attention_layer,
            'attention_native_grid': attention_native,
            'reproducibility_floor_mean': floor_mean,
            'reproducibility_floor_max': floor_max,
        }, f, indent=2, default=float)

    print(f"\nWrote {len(rows)} per-locus maps, locus_comparison.png, "
          f"output_contribution.csv and summary.json to {out}")
    print("\n  The |change| panel is measured at the full 256x256 output: every")
    print("  pixel there is a real difference between two generated images, not")
    print("  an interpolation.")
    if attention_native:
        print(f"  The attention panel beside it IS interpolated, from a "
              f"{attention_native}x{attention_native} grid, and is included only "
              "for comparison.")
    print("\n  A single locus is one of ~43,788, so its effect on the image is")
    print("  faint in absolute terms even when the spatial pattern is clear.")
    print("  Each map is normalised to its own peak, so the panels show WHERE")
    print("  the locus acts; mean_abs_change in the CSV is HOW MUCH, and that")
    print("  is the column to compare between loci.")


if __name__ == '__main__':
    main()
