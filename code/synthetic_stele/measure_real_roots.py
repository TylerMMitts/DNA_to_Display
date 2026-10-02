# Segments every real root image, ranks the genotypes by stele size, and checks
# whether the segmentations can be trusted.
#
# The first step of the synthetic-stele experiment: build_synthetic_snps.py
# turns this ranking into the founder codes of the added genes, so a ranking
# built on bad segmentations would build a bad experiment. Three checks:
#   - flags: no stele found, a stele as wide as the root, a stele/root ratio far
#     outside the rest, or an image far from the other images of its genotype
#   - repeatability: how well images of one genotype agree, and so how much a
#     genotype's mean can be trusted to place it in the ranking
#   - contact sheets of the segmentation drawn over the image, for the genotypes
#     at both ends of the ranking, the flagged images and a random sample, to be
#     looked at rather than taken on trust
#
# Measurements made earlier on this machine with the same segmenter are reused
# rather than repeated, after re-measuring a sample of them to confirm they come
# out the same.
#
# Writes per_image_traits.csv, genotype_stele_ranking.csv, ranking.png, the
# contact sheets and summary.json to results/synthetic_stele/real_root_traits/.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    CROPPED_IMAGES_DIR, IMAGE_METADATA, MODEL_ANALYSIS_DIR, SEGMENTATION_MODEL,
    SNP_PARQUET, SYNTHETIC_STELE_RESULTS_DIR, apply_overrides, pick_device,
    resolve_input, resolve_output,
)

from feature_segmentation.evaluation.reconstruction_fidelity_test import overlay, segment
from feature_segmentation.evaluation.genetic_fidelity_test import repeatability
from latent_diffusion.generation.generate_from_dataset import load_original

TRAITS = ['root_area_px', 'root_diameter_px', 'stele_area_px', 'stele_diameter_px',
          'vessel_total_area_px', 'vessel_count_cc', 'vessel_count_instances',
          'stele_root_diameter_ratio']


# Out-of-fold error of the 512 px segmenter against the hand annotations, from
# cross_validate_segmentation.py. Used to put the spread between genotypes in
# proportion: an error that is small next to that spread cannot reorder it much.
CV_STELE_ERROR_PX = 1.9


# Rows of a previous run's real-image measurements, keyed by filename, renamed
# from real_<trait> to <trait>. None if the file is missing or was written before
# the segmenter last changed, since then it measured with different weights.
def load_reusable(path, seg_weights):
    path = Path(path)
    if not path.exists():
        return None, f'{path} not found'
    if path.stat().st_mtime < Path(seg_weights).stat().st_mtime:
        return None, f'{path.name} predates the current segmenter'
    summary = path.parent / 'summary.json'
    if summary.exists():
        meta = json.loads(summary.read_text())
        used = meta.get('seg_weights') or meta.get('segmenter')
        if used and Path(used).name != Path(seg_weights).name:
            return None, f'{path.name} was measured with {Path(used).name}'
    df = pd.read_csv(path)
    # genetic_fidelity_test.py prefixes real-image traits with real_; this
    # script's own per_image_traits.csv does not.
    prefix = 'real_' if f'real_{TRAITS[0]}' in df.columns else ''
    keep = {f'{prefix}{t}': t for t in TRAITS if f'{prefix}{t}' in df.columns}
    if len(keep) != len(TRAITS):
        return None, f'{path.name} lacks some traits'
    df = df[['filename', *keep]].rename(columns=keep)
    return {r.filename: {t: getattr(r, t) for t in TRAITS} for r in df.itertuples()}, None


# One labelled overlay per image, tiled into a sheet.
def contact_sheet(items, path, cols=6, cell=230):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    label_h = 34
    sheet = Image.new('RGB', (cols * cell, rows * (cell + label_h)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + label_h)
        sheet.paste(Image.fromarray(image).resize((cell - 6, cell - 6)), (x + 3, y + 3))
        draw.text((x + 5, y + cell - 2), label, fill='black')
    sheet.save(path)


def main(overrides=None):
    # Edit these values, then run:
    #     python code/synthetic_stele/measure_real_roots.py
    class cfg:
        image_dir = CROPPED_IMAGES_DIR
        metadata_path = IMAGE_METADATA
        snp_parquet = SNP_PARQUET
        seg_weights = SEGMENTATION_MODEL
        output_dir = SYNTHETIC_STELE_RESULTS_DIR / 'real_root_traits'

        # Earlier real-image measurements to reuse instead of segmenting again:
        # this script's own last run, then the epoch 300 fidelity run.
        reuse = [SYNTHETIC_STELE_RESULTS_DIR / 'real_root_traits' / 'per_image_traits.csv',
                 MODEL_ANALYSIS_DIR / 'diffusion_onehot_medium_epoch_300'
                 / 'genetic_fidelity_local' / 'per_image_measurements.csv']
        # Reused images re-measured to confirm they match. 0 trusts them unchecked.
        verify_reused = 20
        verify_tolerance_px = 0.05

        # Exactly as genetic_fidelity_test.py measures real images, so reused
        # and new measurements are the same measurement.
        imgsz = 256
        conf = 0.25
        min_vessel_px = 4
        connectivity = 2

        # Flag an image this many robust standard deviations from its own
        # genotype's median stele, or from the population's stele/root ratio.
        flag_z = 3.5
        # Flags that drop an image from the ranking. Looking at the flagged
        # sheet, these three are segmentation failures: no stele found, box-
        # shaped stele masks, and partial root masks that wreck the ratio. A
        # genotype outlier was almost always a correct segmentation of a root
        # that differs from its siblings - real variation - so it is reported
        # but kept, rather than pulling every genotype towards its median.
        exclude_flags = ['flag_no_stele', 'flag_stele_not_inside', 'flag_ratio_outlier']
        # Genotypes shown from each end of the ranking, and random images shown.
        n_extreme_genotypes = 6
        n_random = 18
        seed = 0
        device = pick_device()

    apply_overrides(cfg, overrides)

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg_weights = resolve_input(cfg.seg_weights, 'segmentation weights')
    print(f"Segmenter: {seg_weights}\nOutput: {out}")

    from ultralytics import YOLO
    seg = YOLO(str(seg_weights))

    meta = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))
    image_dir = Path(resolve_input(cfg.image_dir, 'cropped image directory'))
    meta = meta[[(image_dir / f).exists() for f in meta['new_filename']]]
    snp_ids = set(pd.read_parquet(resolve_input(cfg.snp_parquet, 'SNP parquet'),
                                  columns=['ID'])['ID'].str.replace('_TC', '').unique())

    def measure_one(filename):
        image = load_original(image_dir / filename, cfg.imgsz)
        traits, masks = segment(seg, image, cfg.conf, cfg.device,
                                cfg.min_vessel_px, cfg.connectivity)
        return image, traits, masks

    # Earlier measurements, checked before any of them is used.
    reused = {}
    for path in cfg.reuse:
        rows, why = load_reusable(path, seg_weights)
        if rows is None:
            print(f"  not reusing: {why}")
            continue
        reused.update(rows)
    reused = {f: t for f, t in reused.items() if f in set(meta['new_filename'])}
    if reused and cfg.verify_reused:
        rng = np.random.default_rng(cfg.seed)
        check = rng.choice(sorted(reused), min(cfg.verify_reused, len(reused)), replace=False)
        worst = 0.0
        for f in check:
            _, t, _ = measure_one(f)
            for key in ('root_diameter_px', 'stele_diameter_px'):
                a, b = reused[f][key], t[key]
                if np.isfinite(a) and np.isfinite(b):
                    worst = max(worst, abs(a - b))
                elif np.isfinite(a) != np.isfinite(b):
                    worst = float('inf')
        if worst > cfg.verify_tolerance_px:
            print(f"  reused measurements differ from a fresh run by up to {worst:.3f} px, "
                  "so everything is measured again")
            reused = {}
        else:
            print(f"  {len(reused)} earlier measurements reused; {len(check)} re-measured "
                  f"agree to within {worst:.4f} px")

    rows = []
    todo = [f for f in meta['new_filename'] if f not in reused]
    print(f"\nImages: {len(meta)}   reused: {len(meta) - len(todo)}   to segment: {len(todo)}")
    genotype_of = dict(zip(meta['new_filename'], meta['genotype']))
    for i, f in enumerate(todo, 1):
        _, t, _ = measure_one(f)
        rows.append({'filename': f, 'source': 'measured', **t})
        if i % 50 == 0 or i == len(todo):
            print(f"  segmented {i}/{len(todo)}")
    rows += [{'filename': f, 'source': 'reused', **t} for f, t in reused.items()]
    df = pd.DataFrame(rows)
    df.insert(1, 'genotype', df['filename'].map(genotype_of))
    df.insert(2, 'has_snp', df['genotype'].isin(snp_ids))

    # Flags. Robust z uses the median absolute deviation, so the outliers being
    # looked for do not inflate the scale they are measured against.
    def robust_z(x):
        mad = np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826
        return (x - np.nanmedian(x)) / (mad or np.nan)

    df['flag_no_stele'] = ~np.isfinite(df['stele_diameter_px'])
    df['flag_stele_not_inside'] = df['stele_diameter_px'] >= df['root_diameter_px']
    df['flag_ratio_outlier'] = np.abs(robust_z(df['stele_root_diameter_ratio'].to_numpy())) > cfg.flag_z
    within = df.groupby('genotype')['stele_diameter_px'].transform(
        lambda s: robust_z(s.to_numpy()) if s.notna().sum() >= 4 else 0.0)
    df['flag_genotype_outlier'] = np.abs(within) > cfg.flag_z
    flag_cols = ['flag_no_stele', 'flag_stele_not_inside', 'flag_ratio_outlier',
                 'flag_genotype_outlier']
    df['flagged'] = df[flag_cols].any(axis=1)
    df['excluded'] = df[list(cfg.exclude_flags)].any(axis=1)
    df = df.sort_values(['genotype', 'filename'])
    df.to_csv(out / 'per_image_traits.csv', index=False)

    # Genotype ranking on the images that were not excluded, smallest stele first.
    ok = df[~df['excluded']]
    g = ok.groupby('genotype').agg(
        n_images=('stele_diameter_px', 'count'),
        stele_diameter_mean=('stele_diameter_px', 'mean'),
        stele_diameter_sd=('stele_diameter_px', 'std'),
        root_diameter_mean=('root_diameter_px', 'mean'),
        stele_root_ratio_mean=('stele_root_diameter_ratio', 'mean'),
        has_snp=('has_snp', 'first'),
    ).reset_index()
    g['stele_diameter_sem'] = g['stele_diameter_sd'] / np.sqrt(g['n_images'])
    g = g.sort_values('stele_diameter_mean').reset_index(drop=True)
    g.insert(0, 'rank', np.arange(1, len(g) + 1))
    g.to_csv(out / 'genotype_stele_ranking.csv', index=False)

    # How much the ranking can be trusted. The ICC is how much of the image-to-
    # image variation is between genotypes; the reliability of a genotype mean
    # over n images follows from it by Spearman-Brown.
    icc, var_b, var_w = repeatability(ok['stele_diameter_px'], ok['genotype'])
    n_bar = float(g['n_images'].mean())
    reliability = n_bar * icc / (1 + (n_bar - 1) * icc) if np.isfinite(icc) else float('nan')
    spread = float(g['stele_diameter_mean'].std())

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [2.4, 1]})
    ax = axes[0]
    colors = ['#1f77b4' if s else '#bbbbbb' for s in g['has_snp']]
    ax.errorbar(g['rank'], g['stele_diameter_mean'], yerr=g['stele_diameter_sem'],
                fmt='none', ecolor='#999999', lw=0.8, zorder=1)
    ax.scatter(g['rank'], g['stele_diameter_mean'], c=colors, s=14, zorder=2)
    ax.set_xlabel('genotype rank, smallest stele first')
    ax.set_ylabel('mean stele diameter (px, 256 px image)')
    ax.set_title('Genotypes ranked by stele diameter (bars: standard error; grey: no SNP data)')
    ax = axes[1]
    ax.hist(ok['stele_diameter_px'].dropna(), bins=40, color='#1f77b4', alpha=0.8)
    ax.set_xlabel('stele diameter per image (px)')
    ax.set_title(f'All images (n = {ok["stele_diameter_px"].notna().sum()})')
    fig.tight_layout()
    fig.savefig(out / 'ranking.png', dpi=140)
    plt.close(fig)

    # Contact sheets: re-segmented so the masks can be drawn.
    def drawn(filenames, label):
        items = []
        for f in filenames:
            image, t, masks = measure_one(f)
            items.append((overlay(image, masks), label(f, t)))
        return items

    images_of = ok.groupby('genotype')['filename'].apply(list)
    small = g['genotype'].head(cfg.n_extreme_genotypes)
    large = g['genotype'].tail(cfg.n_extreme_genotypes)[::-1]
    for name, chosen in (('smallest', small), ('largest', large)):
        files = [f for gen in chosen for f in images_of[gen]]
        contact_sheet(drawn(files, lambda f, t: f"{genotype_of[f]}  stele {t['stele_diameter_px']:.0f}"),
                      out / f'sheet_{name}_stele_genotypes.png')
    flagged = df[df['flagged']]
    contact_sheet(drawn(flagged['filename'],
                        lambda f, t: f"{genotype_of[f]}  " + ','.join(
                            c[5:] for c in flag_cols if flagged.loc[flagged['filename'] == f, c].iloc[0])),
                  out / 'sheet_flagged.png')
    rng = np.random.default_rng(cfg.seed + 1)
    sample = rng.choice(ok['filename'].to_numpy(), min(cfg.n_random, len(ok)), replace=False)
    contact_sheet(drawn(sample, lambda f, t: f"{genotype_of[f]}  stele {t['stele_diameter_px']:.0f}"),
                  out / 'sheet_random.png')

    summary = {
        'segmenter': str(seg_weights),
        'n_images': int(len(df)),
        'n_reused': int((df['source'] == 'reused').sum()),
        'n_segmented': int((df['source'] == 'measured').sum()),
        'stele_detection_rate': float(1 - df['flag_no_stele'].mean()),
        'n_flagged': int(df['flagged'].sum()),
        'n_excluded_from_ranking': int(df['excluded'].sum()),
        'exclude_flags': list(cfg.exclude_flags),
        'flags': {c: int(df[c].sum()) for c in flag_cols},
        'n_genotypes_ranked': int(len(g)),
        'n_genotypes_ranked_with_snp': int(g['has_snp'].sum()),
        'stele_icc_between_genotypes': float(icc),
        'mean_images_per_genotype': n_bar,
        'reliability_of_genotype_mean': float(reliability),
        'genotype_mean_stele_range_px': [float(g['stele_diameter_mean'].min()),
                                         float(g['stele_diameter_mean'].max())],
        'genotype_mean_stele_sd_px': spread,
        'segmenter_cv_stele_error_px': CV_STELE_ERROR_PX,
        'cv_error_over_genotype_spread': CV_STELE_ERROR_PX / spread if spread else float('nan'),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nStele found in {summary['stele_detection_rate']:.1%} of images; "
          f"{summary['n_flagged']} flagged {summary['flags']}")
    print(f"Ranked {len(g)} genotypes ({summary['n_genotypes_ranked_with_snp']} with SNP data), "
          f"mean stele {summary['genotype_mean_stele_range_px'][0]:.1f}-"
          f"{summary['genotype_mean_stele_range_px'][1]:.1f} px")
    print(f"ICC {icc:.2f} -> reliability of a genotype mean over {n_bar:.1f} images {reliability:.2f}")
    print(f"Wrote per_image_traits.csv, genotype_stele_ranking.csv, ranking.png, "
          f"contact sheets and summary.json to {out}")


if __name__ == '__main__':
    main()
