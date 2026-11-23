# Builds the images for the opening of the conference talk.
#
# Everything here is real: the SNP strips are actual founder calls, the generated
# roots come from the presented checkpoint conditioned on one genotype, and the
# photograph is one of that genotype's real cross-sections.
#
# The hook's genotype is one the checkpoint never trained on, and it is chosen,
# not drawn at random: every held-out genotype is generated from several noise
# seeds, generated and real roots are compared on what the eye picks up first -
# colour, stele size relative to the root, vessel count and how crowded the
# vessels are - and the closest genotype, seeds and photograph are used. It is the
# best-matching example, not a typical one, and the speaker notes say so; the rest
# of the talk is about how far from typical it is.
#
# Writes the strips, generated and real roots, founder legend, population strips
# for the context slide, hook_candidates.csv, hook_candidates.png and hook.json,
# which build_talk_deck.js reads so the slides name the right genotype.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    DIFFUSION_ONEHOT_MEDIUM_DIR, RESULTS_DIR, SEGMENTATION_MODEL, checkpoint_path,
    pick_device, resolve_input, resolve_output,
)

from ultralytics import YOLO

from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
from model_toolkit import load_session, save_grid

# One colour per founder, readable against a dark slide and distinguishable
# from each other without relying on hue alone.
FOUNDER_COLORS = ['#4C9BE8', '#F2A65A', '#5FD3A6', '#E8607A',
                  '#B48BE8', '#F2D65A', '#5FC8D3', '#E88A4C']

FEATURES = ['L', 'a', 'b', 'stele_root_ratio', 'vessel_count', 'vessel_crowding']


# The genotype as a colour bar, one column per marker.
def snp_strip(codes, founders, width_px, height_px, path):
    lut = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                    for c in FOUNDER_COLORS[:len(founders)]], dtype=np.uint8)
    index = np.searchsorted(np.asarray(founders), codes).clip(0, len(founders) - 1)
    Image.fromarray(lut[index][None, :, :]).resize((width_px, height_px), Image.NEAREST).save(path)


# The whole genome as a block rather than a bar: the markers wrapped into rows,
# so 43,788 of them fit beside a root image instead of stretching across the
# slide as a thin strip nobody can see the structure in.
#
# The last few markers of the final row are dropped so the grid is rectangular -
# at most rows-1 of 43,788, which is not visible at this scale.
def snp_block(codes, founders, rows, width_px, height_px, path):
    lut = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                    for c in FOUNDER_COLORS[:len(founders)]], dtype=np.uint8)
    index = np.searchsorted(np.asarray(founders), codes).clip(0, len(founders) - 1)
    cols = len(index) // rows
    grid = lut[index[:rows * cols].reshape(rows, cols)]
    Image.fromarray(grid).resize((width_px, height_px), Image.NEAREST).save(path)


# What a viewer compares between two root images at a glance. Colour is the mean
# CIE Lab of the root itself, so the dark background does not dilute it; vessel
# crowding is vessel area over stele area.
def appearance(segmenter, image, device):
    traits, masks = segment(segmenter, image, 0.25, device)
    root = max(masks[0], key=lambda m: m.sum()) if masks[0] else np.ones(image.shape[:2], bool)
    stele_area = traits['stele_area_px']
    lab = rgb2lab(image / 255.0)[root]
    return {'L': lab[:, 0].mean(), 'a': lab[:, 1].mean(), 'b': lab[:, 2].mean(),
            'stele_root_ratio': traits['stele_root_diameter_ratio'],
            'vessel_count': traits['vessel_count_cc'],
            'vessel_crowding': traits['vessel_total_area_px'] / stele_area if stele_area > 0 else np.nan}


def main():
    # Edit these values, then run:
    #     uv run code/make_hook_assets.py
    class cfg:
        # The checkpoint the talk presents; see make_talk_assets.py.
        checkpoint = checkpoint_path(DIFFUSION_ONEHOT_MEDIUM_DIR, 300)
        output_dir = RESULTS_DIR / 'hook_slides' / 'assets_epoch300'

        # None searches every held-out genotype. Set a name to force one; its
        # seeds and photograph are still chosen by the same comparison.
        genotype = None
        # Seeds generated per genotype, and how many of the closest are shown.
        candidate_seeds = list(range(12))
        n_shown = 4

        # Genotypes whose strips illustrate the population on the context slide,
        # picked evenly through the list so the choice is not a judgement call.
        n_population_strips = 4

        sampling_steps = 50
        # How many markers the zoomed strips show, so individual founder blocks
        # are visible rather than averaged into texture.
        zoom_markers = 260
        # Rows the whole genome is wrapped into for genome_block.png.
        block_rows = 18
        device = pick_device()

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = load_session(resolve_input(cfg.checkpoint, 'diffusion checkpoint'), cfg.device)
    segmenter = YOLO(str(resolve_input(SEGMENTATION_MODEL, 'segmentation weights')))
    print(f"Checkpoint: {session.checkpoint.name}\nOutput: {out}")

    held_out = [g for g in session.genotypes_in('held out') if session.image_paths(g)]
    candidates = [cfg.genotype] if cfg.genotype else held_out
    if cfg.genotype and session.group(cfg.genotype) != 'held out':
        raise SystemExit(f"{cfg.genotype} is {session.group(cfg.genotype)}, not held out - "
                         "the hook claims the model never saw it")

    print(f"\nGenerating {len(candidates)} held-out genotypes x {len(cfg.candidate_seeds)} seeds...")
    generated = session.generate(candidates, seeds=cfg.candidate_seeds, steps=cfg.sampling_steps)

    rows = []
    for gi, genotype in enumerate(candidates):
        for path, image in zip(session.image_paths(genotype), session.real_images(genotype)):
            rows.append({'genotype': genotype, 'kind': 'real', 'item': path.name,
                         **appearance(segmenter, image, cfg.device)})
        for si, seed in enumerate(cfg.candidate_seeds):
            rows.append({'genotype': genotype, 'kind': 'generated', 'item': seed,
                         **appearance(segmenter, generated[gi, si], cfg.device)})
    features = pd.DataFrame(rows)

    # Each feature in units of its spread across the real roots, so colour and
    # vessel count weigh alike.
    real = features[features.kind == 'real']
    z = (features[FEATURES] - real[FEATURES].mean()) / real[FEATURES].std()
    features[[f'z_{f}' for f in FEATURES]] = z.to_numpy()

    choices = []
    for genotype in candidates:
        mine = features[features.genotype == genotype]
        gen = mine[mine.kind == 'generated'].dropna(subset=[f'z_{f}' for f in FEATURES])
        for _, photo in mine[mine.kind == 'real'].dropna(subset=[f'z_{f}' for f in FEATURES]).iterrows():
            dist = np.linalg.norm(gen[[f'z_{f}' for f in FEATURES]].to_numpy()
                                  - photo[[f'z_{f}' for f in FEATURES]].to_numpy(dtype=float), axis=1)
            order = np.argsort(dist)[:cfg.n_shown]
            if len(order) < cfg.n_shown:
                continue
            choices.append({'genotype': genotype, 'real_image': photo['item'],
                            'seeds': [int(s) for s in gen['item'].to_numpy()[order]],
                            'mean_distance': float(dist[order].mean())})
    ranking = (pd.DataFrame(choices).sort_values('mean_distance')
               .drop_duplicates('genotype').reset_index(drop=True))
    ranking.to_csv(out / 'hook_candidates.csv', index=False)
    print("\nClosest held-out genotypes (distance in SDs of the real roots, lower is closer):")
    print(ranking.head(8).to_string(index=False))

    # The top candidates side by side, so the pick can be checked by eye.
    sheet, labels = [], []
    for _, c in ranking.head(6).iterrows():
        gi = candidates.index(c.genotype)
        photo = next(i for p, i in zip(session.image_paths(c.genotype), session.real_images(c.genotype))
                     if p.name == c.real_image)
        sheet.append([photo] + [generated[gi, cfg.candidate_seeds.index(s)] for s in c.seeds])
        labels.append(f"{c.genotype}  d={c.mean_distance:.2f}")
    save_grid(sheet, out / 'hook_candidates.png', row_labels=labels)

    pick = ranking.iloc[0]
    genotype, gi = pick.genotype, candidates.index(pick.genotype)
    for k, seed in enumerate(pick.seeds):
        Image.fromarray(generated[gi, cfg.candidate_seeds.index(seed)]).save(out / f'generated_{k}.png')
    photo_index = [p.name for p in session.image_paths(genotype)].index(pick.real_image)
    Image.fromarray(session.real_images(genotype)[photo_index]).save(out / 'real_root.png')

    founders = list(session.founders)
    codes = session.codes(genotype)[0]
    snp_strip(codes, founders, 3600, 150, out / 'snp_strip_full.png')
    snp_strip(codes[:cfg.zoom_markers], founders, 3600, 150, out / 'snp_strip_zoom.png')
    snp_block(codes, founders, cfg.block_rows, 1800, 600, out / 'genome_block.png')
    runs = np.diff(np.flatnonzero(np.diff(codes) != 0))

    others = [g for g in session.genotypes if g != genotype]
    population = [others[i] for i in np.linspace(0, len(others) - 1, cfg.n_population_strips).astype(int)]
    for k, g in enumerate(population):
        snp_strip(session.codes(g)[0][:cfg.zoom_markers], founders, 3600, 150,
                  out / f'population_strip_{k}.png')

    fig, ax = plt.subplots(figsize=(9.0, 0.55))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    for i, f in enumerate(founders):
        ax.add_patch(plt.Rectangle((i * 1.25, 0.15), 0.42, 0.55, facecolor=FOUNDER_COLORS[i], edgecolor='none'))
        ax.text(i * 1.25 + 0.55, 0.42, f'founder {f}', va='center', ha='left', fontsize=11, color='#CADCFC')
    ax.set_xlim(0, len(founders) * 1.25)
    ax.set_ylim(0, 0.85)
    fig.savefig(out / 'founder_legend.png', dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)

    trained = session.genotypes_in('trained')
    info = {
        'checkpoint': session.checkpoint.name, 'epoch': session.epoch,
        'genotype': genotype, 'seeds': pick.seeds, 'real_image': pick.real_image,
        'match_rank_among_held_out': 1, 'n_held_out_candidates': len(candidates),
        'mean_distance': pick.mean_distance,
        'n_markers': int(len(codes)), 'zoom_markers': cfg.zoom_markers,
        'median_founder_run': int(np.median(runs)),
        'n_trained_genotypes': len(trained),
        'n_trained_images': int(sum(len(session.image_paths(g)) for g in trained)),
        'n_founders': len(founders), 'population_genotypes': population,
    }
    (out / 'hook.json').write_text(json.dumps(info, indent=2))
    print(f"\nHook: {genotype}, seeds {pick.seeds}, photograph {pick.real_image}")
    print(f"Wrote strips, generated_0-{cfg.n_shown - 1}.png, real_root.png, population strips, "
          f"hook_candidates.csv/.png and hook.json to {out}")


if __name__ == '__main__':
    main()
