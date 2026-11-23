# Collects every image the conference talk needs into one folder.
#
# Three sources: architecture diagrams and real-activation figures rendered from
# make_slide_assets.py, figures that analysis scripts already produced and are
# copied in, and two montages built here.
#
# Run after make_hook_assets.py, which covers the opening seven slides:
#     python code/make_hook_assets.py
#     python code/make_talk_assets.py

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    pick_device, CROPPED_IMAGES_DIR, DIFFUSION_ONEHOT_MEDIUM_DIR, MODEL_ANALYSIS_DIR,
    RESULTS_DIR, SEED_SCALED_DIR, checkpoint_path, resolve_input, resolve_output,
)

import make_slide_assets as msa
from analysis_pipeline import model_label
from latent_diffusion.analysis import analyze_snp_attention


# A grid of real images on the deck's dark background, so montages sit on the
# slide without a panel around them.
def montage(paths, cols, cell, out_path, gap=10, bg=(11, 16, 32)):
    rows = (len(paths) + cols - 1) // cols
    W = cols * cell + (cols - 1) * gap
    H = rows * cell + (rows - 1) * gap
    sheet = Image.new('RGB', (W, H), bg)
    for i, p in enumerate(paths):
        img = Image.open(p).convert('RGB').resize((cell, cell), Image.LANCZOS)
        sheet.paste(img, ((i % cols) * (cell + gap), (i // cols) * (cell + gap)))
    sheet.save(out_path)
    return out_path


def evenly(items, n):
    return [items[i] for i in np.linspace(0, len(items) - 1, n).astype(int)]


# Shares of pixel variance across a genotype x seed grid of generated images: the
# part explained by seed, by genotype, and what is left. A model that draws its
# genotype would put a large share on genotype; one that ignores it puts nearly
# all of it on the seed.
def variance_shares(images):
    x = images.astype(np.float32) / 255.0
    grand = x.mean(axis=(0, 1))
    seed = float(((x.mean(axis=0) - grand) ** 2).mean())
    genotype = float(((x.mean(axis=1) - grand) ** 2).mean())
    rest = float(((x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + grand) ** 2).mean())
    total = seed + genotype + rest
    return {'seed': seed / total, 'genotype': genotype / total, 'residual': rest / total}


# One latent as the diffusion model actually leaves it, for the decoder slide.
#
# The decoder figure could be run on a latent obtained by encoding a real
# photograph, which is what the reconstruction figures do. It would be
# misleading here: an encoded latent decodes back to something far sharper than
# the pipeline ever produces, so the slide would promise an output quality the
# model does not have. This runs the real sampler instead.
def generated_latent(checkpoint, device, num_steps=50, seed=0, genotype=0):
    from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
    from latent_diffusion.analysis.analyze_snp_attention import load_model
    from latent_diffusion.diffusion.scheduler import DiffusionScheduler
    from paths import SNP_PARQUET

    _, _, snp_matrix = load_snp_data_from_parquet(
        resolve_input(SNP_PARQUET, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    encoder, unet, ucfg = load_model(resolve_input(checkpoint, 'checkpoint'),
                                     snp_matrix, device)

    sch = DiffusionScheduler()
    for attr in ('betas', 'alphas', 'alpha_bars'):
        setattr(sch, attr, getattr(sch, attr).to(device))

    gen = torch.Generator(device='cpu').manual_seed(seed)
    z_t = torch.randn(1, ucfg['latent_channels'], 32, 32, generator=gen).to(device)
    snp = torch.tensor(snp_matrix[genotype][None], dtype=torch.float32,
                       device=device)
    with torch.no_grad():
        emb = encoder(snp)
        timesteps = sch.get_timesteps(num_steps, device)
        for i, t in enumerate(timesteps):
            tb = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
            tp = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            tpb = torch.full((1,), tp, device=device, dtype=torch.long)
            z_t = sch.denoise_step(z_t, unet(z_t, tb, emb), tb, tpb)
    return z_t


# Several loci as population-contrast attention, one row each.
#
# The counterfactual row that analyze_snp_spatial_contribution.py puts above this
# one is the cleaner measurement - flip the locus, change nothing else - but at
# this checkpoint a single locus barely moves the attention and the row is
# speckle. The population contrast, carriers minus non-carriers, is structured
# enough to read from the back of a room. It is also confounded: carriers differ
# from non-carriers at thousands of other loci, so the slide and these labels say
# correlation rather than effect.
#
# Loci are chosen by farthest-point search on the maps themselves, so the rows
# are as unlike each other as the candidate pool allows rather than being the
# strongest few, which tend to repeat one pattern.
def attention_variety(checkpoint, out, cfg, device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd

    from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
    from latent_diffusion.analysis.analyze_snp_attention import load_model
    from latent_diffusion.analysis.analyze_snp_spatial_contribution import (
        population_contrast_map, founder_slot,
    )
    from latent_diffusion.analysis.select_diverse_snp_maps import (
        greedy_diverse_selection,
    )
    from paths import SNP_PARQUET

    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(SNP_PARQUET, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    encoder, unet, unet_cfg = load_model(resolve_input(checkpoint, 'checkpoint'),
                                         snp_matrix, device)
    projector = encoder.pca
    projector.ensure_explained_variance(snp_matrix)
    latent_shape = (unet_cfg['latent_channels'], 32, 32)

    # The pool the model's own analysis already ranked, so this reuses that work
    # rather than sweeping the genome again.
    model_dir = resolve_output(MODEL_ANALYSIS_DIR / cfg.attention_model_dir)
    pool_csv = model_dir / 'snp_diverse_maps' / 'candidate_pool.csv'
    if not pool_csv.exists():
        print(f'  skipped: no {pool_csv}')
        return None
    pool = pd.read_csv(pool_csv).sort_values('magnitude', ascending=False)
    pool = pool.head(cfg.attention_pool)

    rng = np.random.default_rng(0)
    maps, kept = [], []
    for r in pool.itertuples():
        locus = int(r.locus)
        # The rarer founder present at this locus, so the contrast is between
        # two groups that actually exist in the population.
        codes = snp_matrix[:, locus]
        present, counts = np.unique(codes[np.isin(codes, projector.founders)],
                                    return_counts=True)
        if len(present) < 2:
            continue
        founder = int(present[np.argmin(counts)])
        pop, n_c, n_o = population_contrast_map(
            encoder, unet, projector, snp_matrix, locus, founder,
            cfg.attention_timestep, latent_shape, device, 0, cfg.attention_layer,
            16, cfg.attention_per_group, rng)
        if pop is None:
            continue
        maps.append(pop)
        kept.append({'snp': str(r.snp), 'locus': locus, 'founder': founder,
                     'n_carriers': n_c, 'n_others': n_o})
        print(f'  {len(kept):>3}/{len(pool)} {r.snp}  '
              f'{n_c} carry / {n_o} do not')

    if len(kept) < 2:
        print('  skipped: not enough loci with two founder groups')
        return None

    stack = np.stack(maps)
    summaries = np.abs(stack).mean(axis=-1)          # [n, H, W]
    pick = greedy_diverse_selection(summaries, min(cfg.attention_rows, len(kept)),
                                    start_index=0)

    M = stack.shape[-1]
    n = len(pick)
    fig, axes = plt.subplots(n, M + 1, figsize=(1.55 * (M + 1) + 1.6,
                                                1.72 * n + 1.5), squeeze=False)
    for row, idx in enumerate(pick):
        m, info = stack[idx], kept[idx]
        # Scaled per locus, since what the row is for is the shape of the
        # pattern; magnitudes between loci are compared in the CSV instead.
        lim = float(np.abs(m).max()) or 1.0
        for t in range(M):
            ax = axes[row][t]
            ax.imshow(m[:, :, t], cmap='RdBu_r', vmin=-lim, vmax=lim,
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f'token {t}', fontsize=8)
            if t == 0:
                ax.set_ylabel(f"{info['snp']}\n{info['n_carriers']} carry / "
                              f"{info['n_others']} do not", fontsize=8,
                              rotation=0, ha='right', va='center', labelpad=62)
        ax = axes[row][M]
        ax.imshow(np.abs(m).mean(axis=-1), cmap='inferno', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title('|mean| over\ntokens', fontsize=8)

    fig.suptitle('Attention of the genotype tokens, lines carrying each founder '
                 'allele minus lines that do not\n'
                 f'{cfg.attention_layer} @ t={cfg.attention_timestep}, a '
                 f'{stack.shape[1]}x{stack.shape[2]} view   -   '
                 'red = more attention, blue = less\n'
                 f'{n} loci chosen from {len(kept)} for how much their patterns '
                 'differ - this is a correlation, not an intervention',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / 'attention_tokens.png'
    fig.savefig(path, dpi=145, bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame([kept[i] for i in pick]).to_csv(
        out / 'attention_tokens_loci.csv', index=False)
    print(f'  attention_tokens.png  ({n} loci)')
    return path


# One real root and one generated root, both measured by the trait segmenter,
# side by side with the numbers it read off each.
#
# The point of the slide is that a generated image is measurable in the same way
# a photograph is - the same segmenter, the same traits - so the comparison is run
# exactly as genetic_fidelity_test.py runs it.
def segmentation_demo(checkpoint, out, cfg):
    from ultralytics import YOLO

    from feature_segmentation.evaluation.genetic_fidelity_test import save_pair_figure
    from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
    from model_toolkit import load_session

    from paths import SEGMENTATION_MODEL

    session = load_session(checkpoint, cfg.device)
    # The pair the audience has already seen in the hook, unless overridden: a
    # different photograph of the same line can be a much younger root, which
    # would make the comparison look worse than the model is.
    info = json.loads((RESULTS_DIR / 'hook_slides' / 'assets_epoch300' / 'hook.json').read_text())
    genotype = cfg.demo_genotype or info['genotype']
    seed = info['seeds'][0] if cfg.demo_seed is None else cfg.demo_seed
    segmenter = YOLO(str(resolve_input(SEGMENTATION_MODEL, 'segmentation weights')))

    paths = session.image_paths(genotype)
    which = next((i for i, p in enumerate(paths) if p.name == info.get('real_image')), 0)
    real = session.real_images(genotype)[which]
    fake = session.generate(genotype, seeds=[seed])[0, 0]
    real_traits, real_masks = segment(segmenter, real, 0.25, cfg.device)
    gen_traits, gen_masks = segment(segmenter, fake, 0.25, cfg.device)
    save_pair_figure(f'{genotype}, seed {seed}', genotype, real, real_masks,
                     real_traits, fake, gen_masks, gen_traits, out / 'fig_segmentation_demo.png')
    print(f"  fig_segmentation_demo.png ({genotype}): "
          f"root {real_traits['root_diameter_px']:.0f} vs {gen_traits['root_diameter_px']:.0f} px, "
          f"vessels {real_traits['vessel_count_cc']} vs {gen_traits['vessel_count_cc']}")


def seed_vs_genotype(checkpoint, out, cfg):
    from model_toolkit import load_session, save_grid
    session = load_session(checkpoint, cfg.device)
    held = [g for g in session.genotypes_in('held out') if session.image_paths(g)]
    trained = session.genotypes_in('trained')

    shown = evenly(held, cfg.grid_genotypes_per_group) + evenly(trained, cfg.grid_genotypes_per_group)
    grid = session.generate(shown, seeds=cfg.grid_seeds)
    # Plain panels: the slide labels rows and columns itself, at a size that reads.
    for r, genotype in enumerate(shown):
        for c, seed in enumerate(cfg.grid_seeds):
            Image.fromarray(grid[r, c]).save(out / f'seedgrid_r{r}_c{c}.png')

    stats = {}
    for group, pool in (('held out', held), ('trained', trained)):
        picked = evenly(pool, cfg.stats_genotypes_per_group)
        stats[group] = {'genotypes': picked,
                        **variance_shares(session.generate(picked, seeds=cfg.stats_seeds))}
    both = evenly(held, cfg.stats_genotypes_per_group) + evenly(trained, cfg.stats_genotypes_per_group)
    stats['all'] = {'genotypes': both, **variance_shares(session.generate(both, seeds=cfg.stats_seeds))}
    (out / 'seed_vs_genotype.json').write_text(json.dumps({
        'shown_genotypes': shown, 'shown_groups': [session.group(g) for g in shown],
        'shown_seeds': cfg.grid_seeds, 'stats_seeds': cfg.stats_seeds, 'shares': stats,
    }, indent=2))
    for group, s in stats.items():
        print(f"  {group:9s} seed {s['seed']:.1%}  genotype {s['genotype']:.1%}  residual {s['residual']:.1%}")


def main():
    # Edit these values, then run:
    #     python code/make_talk_assets.py
    class cfg:
        # The model the talk presents. Epoch 300 of the resized medium model: it
        # memorises least of the checkpoints tested, and epochs 500 and 800 show
        # memorisation growing again with longer training.
        checkpoint = checkpoint_path(DIFFUSION_ONEHOT_MEDIUM_DIR, 300)
        # A folder of its own, so the assets behind the earlier deck stay intact.
        output_dir = RESULTS_DIR / 'talk_slides' / 'assets_epoch300'
        # The sampling trajectory is one full DDIM run and the only slow part.
        include_trajectory = True
        # The decoder slide, run on a latent the diffusion model produced
        # rather than one obtained by encoding a photograph. Costs one more
        # DDIM run.
        include_decoder_walkthrough = True
        decoder_seed = 0
        decoder_genotype = 0
        # Attention is measured again for this checkpoint rather than copied from
        # the first model's analysis, since the slide shows this model's maps.
        include_attention = True

        # The per-token attention figure: how many loci get a row, how many
        # candidates they are chosen from, and how many genotypes go into each
        # side of the carrier/non-carrier contrast.
        include_attention_variety = True
        attention_rows = 4
        attention_pool = 24
        attention_per_group = 32
        attention_layer = 'up_2'
        attention_timestep = 500
        # Which model's candidate pool to read. Follows cfg.checkpoint.
        attention_model_dir = None

        # Generated roots for several genotypes from the same noise seeds, and how
        # much of the pixel variation follows the seed rather than the genotype.
        # Genotypes are taken evenly through each group, not picked by look.
        # A real root and a generated one, each measured by the trait segmenter,
        # for the slide about reading traits back off a generated image.
        include_segmentation_demo = True
        demo_genotype = None      # None -> the hook's genotype
        demo_seed = None          # None -> the first seed the hook shows

        include_seed_grid = True
        grid_genotypes_per_group = 2
        grid_seeds = [0, 4, 7, 10, 3]
        # The variance split uses more genotypes and seeds than the figure shows.
        stats_genotypes_per_group = 6
        stats_seeds = [0, 1, 2, 3, 4, 5]
        device = pick_device()

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    checkpoint = resolve_input(cfg.checkpoint, 'diffusion checkpoint')
    print(f"Device: {device}\nCheckpoint: {checkpoint.name}\nOutput: {out}\n")

    # Diagrams are drawn from the checkpoint, so they describe the model the
    # talk is actually about rather than values typed by hand.
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    unet_cfg = dict(ckpt['unet_config'])
    unet_cfg.setdefault('channel_mult', (1, 2, 4, 8))
    enc_cfg = dict(ckpt['snp_encoder_config'])
    n_loci = int(ckpt['snp_projector']['n_loci'])
    founders = len(ckpt['founders'])

    print('Diagrams...')
    # The two the talk leads on: what goes into each piece and what comes out,
    # and the same chain with the dataset-specific ends swapped.
    msa.diagram_three_pieces(out, unet_cfg, enc_cfg, n_loci)
    msa.diagram_generic_recipe(out)
    msa.diagram_pipeline(out, unet_cfg, enc_cfg)
    msa.diagram_litevae(out, unet_cfg)
    msa.diagram_litevae_encoder(out, latent_ch=unet_cfg['latent_channels'])
    msa.diagram_unet(out, unet_cfg, enc_cfg)
    # The same UNet with the sampling loop and decoder around it.
    msa.diagram_diffusion_model(out, unet_cfg, enc_cfg)
    msa.diagram_cross_attention(out, unet_cfg, enc_cfg)
    msa.diagram_onehot(out, enc_cfg, n_loci, founders)

    print('Real activations...')
    enc, dec, x, ref = msa.load_reference(device)
    msa.fig_wavelet_levels(out, x)
    msa.fig_encoder_stages(out, enc, x)
    msa.diagram_encoder_walkthrough(out, enc, x,
                                    latent_ch=unet_cfg['latent_channels'])
    if cfg.include_decoder_walkthrough:
        print('  decoder walkthrough (on a generated latent)...')
        z = generated_latent(checkpoint, device, seed=cfg.decoder_seed,
                             genotype=cfg.decoder_genotype)
        msa.diagram_decoder_walkthrough(out, enc, dec, x, z=z,
                                        latent_ch=unet_cfg['latent_channels'])
    msa.fig_latent_channels(out, enc, x)
    if cfg.include_trajectory:
        print('  sampling trajectory (one full DDIM run)...')
        msa.fig_sampling_trajectory(out, device, checkpoint=checkpoint)
        # The same run as a clean strip for the title slide.
        msa.fig_denoise_strip(out, device, checkpoint=checkpoint)
        # And again as an animation, every step decoded, for the generator slide.
        msa.gif_denoise(out, device, checkpoint=checkpoint)
        # The three pieces as shaded sections, real tensors throughout, on the
        # same run as the strip and the GIF.
        msa.diagram_pipeline_overview(out, enc, dec, x, device, checkpoint=checkpoint)

    if cfg.include_segmentation_demo:
        print('\nSegmenter on a real and a generated root...')
        segmentation_demo(checkpoint, out, cfg)

    # Figures other scripts already produced. Reported rather than skipped, so a
    # missing one is visible instead of quietly leaving a hole in the deck.
    print('\nCopying analysis figures...')
    # Only figures that do not depend on the diffusion checkpoint belong here.
    # The genetic fidelity figures used to be in this list and were copied from
    # the first model's folder, which quietly replaced a figure measured on this
    # checkpoint with one measured on a different one. They are taken from the
    # model's own folder below instead.
    wanted = {
        'reconstruction_fidelity/agreement_summary.png': 'recon_fidelity.png',
        'onehot_encoding_preview/similarity_comparison.png': 'onehot_similarity.png',
        'parent_archetypes_onehot/parents_multi_seed.png': 'founder_archetypes.png',
    }
    missing = []
    for src, dst in wanted.items():
        p = RESULTS_DIR / src
        if p.exists():
            shutil.copy2(p, out / dst)
            print(f'  {dst}')
        else:
            missing.append(src)

    # Figures from this checkpoint's own analysis folder rather than the first
    # model's, since the slides describe this model. evaluate_diffusion_model.py
    # writes them; run its snp_diverse_maps and snp_output_contribution steps if
    # the locus maps are missing.
    model_dir = resolve_output(MODEL_ANALYSIS_DIR / model_label(checkpoint, ckpt.get('epoch')))
    from_model = {
        'snp_output_contribution/locus_comparison.png': 'locus_comparison.png',
        'genetic_fidelity/summary.png': 'genetic_fidelity_summary.png',
        'genetic_fidelity/genotype_scatter.png': 'genotype_scatter.png',
    }
    for src, dst in from_model.items():
        p = model_dir / src
        if p.exists():
            shutil.copy2(p, out / dst)
            print(f'  {dst}  (from {model_dir.name})')
        elif (out / dst).exists():
            # Left alone rather than overwritten or deleted: a figure put here
            # by hand, from a run on the cluster, is the only copy there is.
            print(f'  {dst}  KEPT - {p.relative_to(RESULTS_DIR)} not found, '
                  'leaving the existing file in place')
        else:
            missing.append(str(p.relative_to(RESULTS_DIR)))

    if cfg.include_attention_variety:
        print('\nPer-token attention, several loci...')
        if cfg.attention_model_dir is None:
            cfg.attention_model_dir = model_dir.name
        if attention_variety(checkpoint, out, cfg, device) is None:
            missing.append('attention_tokens.png')

    # The highest-resolution attention grid the UNet produces, as a deviation
    # from the population mean - raw attention is dominated by a shared bias.
    # One group at the one timestep the slide shows, without the untrained null,
    # which the slide does not use.
    attention_dir = out / 'attention_run'
    if cfg.include_attention and not sorted(attention_dir.rglob('up_2_t800_deviation.png')):
        print('  measuring attention for this checkpoint...')
        analyze_snp_attention.main({
            'checkpoint': checkpoint, 'output_dir': attention_dir,
            'pca_cache': out / 'attention_run' / 'pca.pkl',
            'timesteps': [800], 'n_groups': 1, 'untrained_null': False,
            'device': cfg.device,
        })
    hits = sorted(attention_dir.rglob('up_2_t800_deviation.png'))
    if hits:
        shutil.copy2(hits[0], out / 'attention_up2.png')
        print('  attention_up2.png')
    else:
        missing.append('attention_run/**/up_2_t800_deviation.png')

    if cfg.include_seed_grid:
        print('\nSeed versus genotype...')
        seed_vs_genotype(checkpoint, out, cfg)

    print('\nMontages...')
    roots = sorted(CROPPED_IMAGES_DIR.glob('*.JPG'))
    if roots:
        step = max(1, len(roots) // 24)
        montage([roots[i * step] for i in range(24)], 8, 190,
                out / 'real_root_montage.png')
        print(f'  real_root_montage.png (24 of {len(roots)} crops)')

    seeds = sorted(SEED_SCALED_DIR.glob('*.png')) if SEED_SCALED_DIR.exists() else []
    if seeds:
        step = max(1, len(seeds) // 24)
        montage([seeds[i * step] for i in range(24)], 8, 190,
                out / 'seed_kernel_montage.png', bg=(255, 255, 255))
        print(f'  seed_kernel_montage.png (24 of {len(seeds)} kernels)')
    else:
        missing.append('results/seeds/scaled_images')

    if missing:
        print(f'\n{len(missing)} source(s) not found:')
        for m in missing:
            print(f'  MISSING  {m}')
        print('  Run the script that produces each, then rerun this.')

    print(f'\n{len(list(out.glob("*.png")))} images in {out}')


if __name__ == '__main__':
    main()
