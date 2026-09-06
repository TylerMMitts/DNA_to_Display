# The founder images score a large pixel difference but look the same. Why?
#
# At guidance w=5 the eight founders reach a mean pairwise RMSE of 78 against a
# real-genotype reference of 32 - nominally 2.4x more different from each other
# than real genotypes are - and still look interchangeable. A number that large
# paired with no visible difference means RMSE is measuring something the eye
# does not read as "a different root".
#
# This separates the possibilities instead of guessing between them. Pixel
# difference between two images is decomposed into three parts, each removed in
# turn:
#
# brightness   difference in per-image mean. A uniform lift or dim of the
#              whole frame. Large RMSE, near-zero perceptual effect on
#              "which root is this".
# contrast     difference in per-image standard deviation, after brightness
#              is matched. Same story - the image looks washed out or harsh
#              rather than anatomically different.
# structure    what remains once both images have the same mean and standard
#              deviation. This is the part that can actually move a vessel,
#              resize a stele, or change a root's outline.
#
# If most of the RMSE is brightness plus contrast, the model is modulating
# appearance and not anatomy, and no amount of further amplification will help
# because it is amplifying the wrong axis - guidance would simply push the tint
# further.
#
# The second, independent check does not rely on pixels at all: the
# segmentation model measures vessel count and root/stele diameter on each
# founder image. Those are the traits the project actually cares about. If they
# are identical across founders while RMSE is large, that settles it - the
# differences are cosmetic regardless of what any pixel metric says.
#
# Reads the PNGs that founder_archetype_strategies.py already wrote, so it
# costs no diffusion sampling and can be run on existing output.
#
# Usage
#     python code/latent_diffusion/analysis/diagnose_founder_similarity.py

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import RESULTS_DIR, SEGMENTATION_MODEL, resolve_input, resolve_output


# Decomposition

def decompose_pair(a, b):
    # Split the RMSE between two images into brightness/contrast/structure.
    #
    # Removing the components in this order matters and is not arbitrary:
    # brightness first, because a mean offset inflates every other statistic;
    # then contrast, because a scale difference inflates the structural
    # residual; what survives both is difference in PATTERN rather than in
    # level or amplitude.
    #
    # Returned components are RMSE values in 0-255 units, defined so that the
    # structural term is exactly the RMSE that remains after both images have
    # been matched in mean and standard deviation. They are NOT constrained to
    # sum to the raw RMSE - they are three separate distances, not a partition -
    # so the useful reading is the ratio structure/raw, not the arithmetic.
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    raw = float(np.sqrt(((a - b) ** 2).mean()))

    # Brightness matched: remove each image's own mean.
    ac, bc = a - a.mean(), b - b.mean()
    after_brightness = float(np.sqrt(((ac - bc) ** 2).mean()))

    # Contrast matched too: scale each to unit standard deviation, then back
    # to a common scale so the number stays in comparable 0-255 units.
    sa, sb = ac.std(), bc.std()
    common = 0.5 * (sa + sb)
    an = ac / max(sa, 1e-9) * common
    bn = bc / max(sb, 1e-9) * common
    after_contrast = float(np.sqrt(((an - bn) ** 2).mean()))

    return {
        'raw_rmse': raw,
        'brightness_delta': float(abs(a.mean() - b.mean())),
        'contrast_delta': float(abs(sa - sb)),
        'rmse_after_brightness_match': after_brightness,
        'rmse_after_contrast_match': after_contrast,
        'structural_fraction': after_contrast / raw if raw > 0 else float('nan'),
    }


def load_strategy_images(images_dir, founders):
    # {strategy: [image per founder]} from the saved PNGs.
    images_dir = Path(images_dir)
    pattern = re.compile(r'^(?P<strategy>.+)_founder(?P<k>\d+)\.png$')
    found = {}
    for p in sorted(images_dir.glob('*_founder*.png')):
        m = pattern.match(p.name)
        if not m:
            continue
        found.setdefault(m.group('strategy'), {})[int(m.group('k'))] = p

    out = {}
    for strategy, by_k in found.items():
        if not all(k in by_k for k in founders):
            continue          # partial run; skip rather than compare unevenly
        out[strategy] = [np.array(Image.open(by_k[k]).convert('RGB'))
                         for k in founders]
    return out


# Figures

def save_decomposition_figure(summary, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    names = summary['strategy'].tolist()
    x = np.arange(len(names))

    ax = axes[0]
    width = 0.38
    ax.bar(x - width / 2, summary['raw_rmse'], width, label='raw RMSE',
           color='#bbbbbb')
    ax.bar(x + width / 2, summary['rmse_after_contrast_match'], width,
           label='after matching brightness + contrast', color='#2b7bba')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('mean pairwise RMSE')
    ax.set_title('How much of the difference survives\n'
                 'once brightness and contrast are equalised?', fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    frac = summary['structural_fraction'].to_numpy()
    bars = ax.bar(x, frac, color=['#c9622a' if f < 0.5 else '#2b7bba' for f in frac])
    ax.axhline(0.5, color='crimson', ls='--', lw=1.2,
               label='half the difference is structural')
    ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('structural fraction of RMSE')
    ax.set_title('Low bar = the founders differ in tint/exposure,\n'
                 'not in anatomy', fontsize=11)
    for b, f in zip(bars, frac):
        ax.text(b.get_x() + b.get_width() / 2, f, f'{f:.2f}',
                ha='center', va='bottom', fontsize=8)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_difference_maps(images, founders, strategy, save_path, amplify=6.0):
    # Each founder minus the mean founder, amplified, to show WHERE they
    # differ.
    #
    # Amplification is for display only and is stated in the title - the point
    # is to reveal whether the residual structure is anatomical (edges around
    # vessels, the stele boundary) or diffuse (a global wash), which is hard to
    # judge at the true amplitude.
    stack = np.stack(images).astype(np.float64)
    mean_img = stack.mean(axis=0)

    n = len(founders)
    fig, axes = plt.subplots(2, n, figsize=(1.85 * n, 4.2), squeeze=False)
    for c, k in enumerate(founders):
        axes[0][c].imshow(images[c])
        axes[0][c].set_title(f'founder {k}', fontsize=9)
        axes[0][c].set_xticks([]); axes[0][c].set_yticks([])

        diff = (stack[c] - mean_img).mean(axis=-1)      # grey, signed
        lim = max(np.abs(diff).max() / max(amplify, 1e-9), 1e-9)
        axes[1][c].imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim)
        axes[1][c].set_xticks([]); axes[1][c].set_yticks([])

    axes[0][0].set_ylabel('image', fontsize=9)
    axes[1][0].set_ylabel(f'deviation from\nmean founder\n({amplify:g}x)', fontsize=8)
    fig.suptitle(f'{strategy}: where does each founder actually differ?\n'
                 'diffuse wash = global tint; sharp edges at vessels/stele = anatomy',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=145, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/diagnose_founder_similarity.py
    class cfg:
        # Reads what founder_archetype_strategies.py already generated.
        images_dir = RESULTS_DIR / 'founder_strategies/images'
        output_dir = RESULTS_DIR / 'founder_similarity_diagnosis'

        founders = [1, 2, 3, 4, 5, 6, 7, 8]

        # Strategies to render difference maps for. None -> the highest raw
        # RMSE one plus the plain baseline, which is the most informative
        # contrast to look at.
        difference_map_strategies = None
        difference_amplify = 6.0

        # Anatomical check with the segmentation model. This is the part that
        # does not depend on any pixel metric.
        segment = True
        seg_weights = SEGMENTATION_MODEL
        seg_conf = 0.25
        imgsz = 256
        device = 'cpu'

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = resolve_input(cfg.images_dir, 'founder images directory')
    print(f"Reading images from: {images_dir}\nOutput: {out}")

    by_strategy = load_strategy_images(images_dir, cfg.founders)
    if not by_strategy:
        raise SystemExit(
            f"no complete founder image sets found under {images_dir} - run "
            f"founder_archetype_strategies.py first.")
    print(f"Strategies found: {len(by_strategy)}")

    # decomposition
    rows, per_pair = [], []
    for strategy, imgs in by_strategy.items():
        parts = [decompose_pair(imgs[i], imgs[j])
                 for i, j in combinations(range(len(cfg.founders)), 2)]
        for (i, j), p in zip(combinations(cfg.founders, 2), parts):
            per_pair.append({'strategy': strategy, 'founder_a': i,
                             'founder_b': j, **p})
        agg = {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}
        rows.append({'strategy': strategy, **agg})

    summary = pd.DataFrame(rows).sort_values('raw_rmse', ascending=False)
    summary.to_csv(out / 'rmse_decomposition.csv', index=False)
    pd.DataFrame(per_pair).to_csv(out / 'per_pair_decomposition.csv', index=False)
    save_decomposition_figure(summary, out / 'rmse_decomposition.png')

    print(f"{'strategy':<24}{'raw':>9}{'-bright':>10}{'-contrast':>11}"
          f"{'struct frac':>13}{'dBright':>10}")
    for _, r in summary.iterrows():
        print(f"{r['strategy']:<24}{r['raw_rmse']:>9.2f}"
              f"{r['rmse_after_brightness_match']:>10.2f}"
              f"{r['rmse_after_contrast_match']:>11.2f}"
              f"{r['structural_fraction']:>13.2f}"
              f"{r['brightness_delta']:>10.2f}")

    # difference maps
    targets = cfg.difference_map_strategies
    if targets is None:
        targets = [summary.iloc[0]['strategy']]
        if 'pure' in by_strategy and 'pure' not in targets:
            targets.append('pure')
    for strategy in targets:
        if strategy not in by_strategy:
            print(f"  (skipping difference map for missing strategy {strategy!r})")
            continue
        save_difference_maps(by_strategy[strategy], cfg.founders, strategy,
                             out / f'difference_map_{strategy}.png',
                             cfg.difference_amplify)

    # anatomical check
    trait_rows = []
    if cfg.segment:
        try:
            from ultralytics import YOLO
            from feature_segmentation.evaluation.reconstruction_fidelity_test import measure
            seg = YOLO(str(resolve_input(cfg.seg_weights, 'segmentation weights')))
        except (SystemExit, FileNotFoundError, ImportError) as exc:
            print(f"\nSkipping segmentation: {exc}")
            seg = None

        if seg is not None:
            print("\nMeasuring anatomy on each founder image...")
            for strategy, imgs in by_strategy.items():
                for k, img in zip(cfg.founders, imgs):
                    res = seg.predict(img[:, :, ::-1], conf=cfg.seg_conf,
                                      imgsz=cfg.imgsz, device=cfg.device,
                                      verbose=False)[0]
                    traits, _ = measure(res, cfg.imgsz)
                    trait_rows.append({
                        'strategy': strategy, 'founder': k,
                        'vessel_count': traits.get('vessel_count_cc', np.nan),
                        'root_diameter_px': traits.get('root_diameter_px', np.nan),
                        'stele_diameter_px': traits.get('stele_diameter_px', np.nan),
                    })

    if trait_rows:
        traits_df = pd.DataFrame(trait_rows)
        traits_df.to_csv(out / 'founder_traits.csv', index=False)

        # Spread of each trait ACROSS founders, per strategy. Near-zero spread
        # is the decisive result: identical anatomy regardless of pixel RMSE.
        spread = traits_df.groupby('strategy').agg(
            vessel_std=('vessel_count', 'std'),
            vessel_mean=('vessel_count', 'mean'),
            root_std=('root_diameter_px', 'std'),
            root_mean=('root_diameter_px', 'mean'),
            stele_std=('stele_diameter_px', 'std'),
            stele_mean=('stele_diameter_px', 'mean'),
        ).reset_index()
        spread.to_csv(out / 'trait_spread_by_strategy.csv', index=False)

        print("ANATOMICAL SPREAD ACROSS FOUNDERS (std over the 8 founders)")
        print(f"{'strategy':<24}{'vessels':>18}{'root px':>18}{'stele px':>18}")
        for _, r in spread.iterrows():
            def fmt(m, s):
                if not np.isfinite(m):
                    return '    n/a'
                return f"{m:.1f} +/- {0.0 if not np.isfinite(s) else s:.1f}"
            print(f"{r['strategy']:<24}{fmt(r['vessel_mean'], r['vessel_std']):>18}"
                  f"{fmt(r['root_mean'], r['root_std']):>18}"
                  f"{fmt(r['stele_mean'], r['stele_std']):>18}")
        print("  A std near zero means every founder produced the same anatomy.")
        print("  That is the decisive check: it does not care what the pixel")
        print("  metric said, and a large RMSE alongside a near-zero trait")
        print("  spread means the model is repainting the same root rather")
        print("  than growing a different one.")

    with open(out / 'summary.json', 'w') as f:
        json.dump({'strategies': rows,
                   'n_founders': len(cfg.founders)}, f, indent=2)

    best = summary.iloc[0]
    print(f"\n  Highest raw RMSE: {best['strategy']} at {best['raw_rmse']:.1f}, "
          f"of which {best['structural_fraction']:.0%} is structural")
    if best['structural_fraction'] < 0.5:
        print("  -> most of that difference is brightness and contrast, which is")
        print("     why it does not read as a different root. Pushing guidance")
        print("     higher will mostly push the tint further, not the anatomy.")
    else:
        print("  -> the difference is mostly structural, so if it still does not")
        print("     read as a different root, look at difference_map_*.png to see")
        print("     whether the structure that changed is anatomically meaningful")
        print("     or diffuse texture.")

    print(f"\nWrote rmse_decomposition.csv/.png, per_pair_decomposition.csv, "
          f"difference maps"
          f"{', founder_traits.csv and trait_spread_by_strategy.csv' if trait_rows else ''} "
          f"to {out}")


if __name__ == '__main__':
    main()
