# Renders the grids that test_conditioning_strength.py measured.
#
# Reads that script's output folder, so run it first.

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    CROPPED_IMAGES_DIR, IMAGE_METADATA, KINSHIP_MATRIX, RESULTS_DIR,
    SEGMENTATION_MODEL, resolve_input, resolve_output,
)

from latent_diffusion.generation.generate_from_dataset import load_original


def rmse(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((d ** 2).mean()))


def precompute_pairwise(grid):
    # Every pairwise RMSE this module needs, computed exactly once.
    #
    # genotype_distance_matrix, decompose and bootstrap_ratio all reduce to
    # slicing and averaging these two small arrays rather than recomputing RMSE
    # on 256x256x3 images. That matters most for the bootstrap: resampling from
    # the raw grid directly would recompute full-resolution RMSE on the order of
    # a million times over 2000 iterations (roughly 10 minutes, measured) for
    # numbers this function makes available in milliseconds - the bootstrap
    # only ever needs to know which of the already-computed distances a given
    # resample selects, never a new pixel comparison.
    n_geno, n_seeds = grid.shape[:2]

    geno_dist = np.zeros((n_geno, n_geno, n_seeds))
    for g1, g2 in combinations(range(n_geno), 2):
        # All seeds for this genotype pair in one vectorised call rather than
        # one rmse() call per seed - at 200 genotypes there are 19,900 pairs,
        # so cutting the per-pair Python call count by n_seeds matters.
        a = grid[g1].astype(np.float64)   # [n_seeds, H, W, 3]
        b = grid[g2].astype(np.float64)
        d = np.sqrt(((a - b) ** 2).mean(axis=(1, 2, 3)))   # [n_seeds]
        geno_dist[g1, g2] = d
        geno_dist[g2, g1] = d

    noise_dist = np.zeros((n_geno, n_seeds, n_seeds))
    for g in range(n_geno):
        for s1, s2 in combinations(range(n_seeds), 2):
            d = rmse(grid[g, s1], grid[g, s2])
            noise_dist[g, s1, s2] = d
            noise_dist[g, s2, s1] = d

    return geno_dist, noise_dist


def genotype_distance_matrix(geno_dist):
    # Mean pairwise image distance between genotypes, averaged over seeds.
    #
    # Averaging over seeds is what makes this a statement about the genotypes
    # rather than about any one noise draw.
    return geno_dist.mean(axis=2)


def decompose(geno_dist, noise_dist):
    n_geno = geno_dist.shape[0]
    n_seeds = noise_dist.shape[1]
    genotype_vals = [geno_dist[g1, g2, s]
                     for s in range(n_seeds)
                     for g1, g2 in combinations(range(n_geno), 2)]
    noise_vals = [noise_dist[g, s1, s2]
                  for g in range(n_geno)
                  for s1, s2 in combinations(range(n_seeds), 2)]
    return np.array(genotype_vals), np.array(noise_vals)


def bootstrap_ratio(geno_dist, noise_dist, n_boot=2000, seed=0):
    # Confidence interval for the conditioning ratio, resampling GENOTYPES.
    #
    # Resampling the pairwise distances directly would badly understate the
    # uncertainty: with few genotypes the pairs number in the dozens, but they
    # are built from far fewer independent objects, so the pairs are heavily
    # dependent. Resampling whole genotypes respects that and gives an interval
    # that reflects how few genomes this actually rests on.
    #
    # Vectorised over genotype pairs rather than looped, because the number of
    # pairs is quadratic in n_geno: at 8 genotypes there are 28 pairs and a
    # per-pair Python loop is instant, but at ~200 (the full dataset) there are
    # 19,900, and 2000 bootstrap iterations of that loop measured at ~117s - not
    # a hang, but slow enough to be worth the fix given the loop body is just
    # array indexing that numpy can do directly.
    rng = np.random.default_rng(seed)
    n_geno = geno_dist.shape[0]
    n_seeds = noise_dist.shape[1]
    gi, gj = np.triu_indices(n_geno, k=1)     # fixed positions; only the pick changes
    si, sj = np.triu_indices(n_seeds, k=1)

    ratios = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_geno, size=n_geno)

        g_sub = geno_dist[np.ix_(pick, pick)]          # [n_geno, n_geno, n_seeds]
        g_vals = g_sub[gi, gj, :]                       # [n_pairs, n_seeds]
        valid_pairs = pick[gi] != pick[gj]              # same genotype resampled twice -> exclude
        g_vals = g_vals[valid_pairs]

        n_sub = noise_dist[pick]                        # [n_geno, n_seeds, n_seeds]
        n_vals = n_sub[:, si, sj]                        # [n_geno, n_seed_pairs]

        if g_vals.size == 0:
            continue
        n_mean = n_vals.mean()
        if n_mean > 0:
            ratios.append(g_vals.mean() / n_mean)
    ratios = np.array(ratios)
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def load_kinship(path, genotypes):
    # Kinship submatrix for these genotypes, or None if unavailable.
    try:
        p = resolve_input(path, 'kinship matrix')
    except FileNotFoundError:
        return None
    df = pd.read_csv(p, index_col=0)
    df.index = df.index.astype(str).str.replace('_TC', '', regex=False)
    df.columns = df.columns.astype(str).str.replace('_TC', '', regex=False)
    if not all(g in df.index and g in df.columns for g in genotypes):
        return None
    return df.loc[genotypes, genotypes].to_numpy(dtype=float)


def upper_tri(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


def find_original_image(genotype, metadata, image_dir):
    # First cropped photo on file for this genotype, or None.
    #
    # A genotype can have several rows (rootnode/replication/rootnumber
    # variants); the first match is used rather than trying to pick the "best"
    # one, matching the setdefault-first-match convention already used for
    # preview genotypes in train_onehot.py.
    rows = metadata[metadata['genotype'] == genotype]
    for _, row in rows.iterrows():
        path = Path(image_dir) / row['new_filename']
        if path.exists():
            return path
    return None


def format_trait_sublabel(traits):
    # 3-line label: vessel count, root diameter, stele diameter.
    #
    # Matches the sublabel format already established in
    # generate_parent_archetypes.py, so a reader who has seen one recognises
    # the other. NaN (segmentation did not find that class) prints as 'n/a'
    # rather than silently becoming a blank line.
    def fmt(key, template):
        v = traits.get(key, float('nan'))
        return 'n/a' if not np.isfinite(v) else template.format(v)

    return (f"{fmt('vessel_count_cc', '{:.0f}')} vessels\n"
           f"root: {fmt('root_diameter_px', '{:.0f}')}px\n"
           f"stele: {fmt('stele_diameter_px', '{:.0f}')}px")


# Figures

def save_side_by_side(grids, labels, genotypes, seeds, show_seeds, save_path):
    # Same genotype, same noise, one model beside the other.
    n_geno = len(genotypes)
    n_show = len(show_seeds)
    n_models = len(labels)
    ncols = n_show * n_models

    fig, axes = plt.subplots(n_geno, ncols,
                             figsize=(1.5 * ncols, 1.62 * n_geno))
    axes = np.atleast_2d(axes)

    for i in range(n_geno):
        for k, s_i in enumerate(show_seeds):
            for m, label in enumerate(labels):
                ax = axes[i, k * n_models + m]
                ax.imshow(grids[label][i, s_i])
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f'seed {seeds[s_i]}\n{label}', fontsize=7)
                if k == 0 and m == 0:
                    ax.set_ylabel(genotypes[i], fontsize=7, rotation=0,
                                  ha='right', va='center', labelpad=30)
                # Thin coloured frame so the two models stay distinguishable
                # once the eye is deep in the grid.
                for spine in ax.spines.values():
                    spine.set_edgecolor('#888888' if m == 0 else '#2b7bba')
                    spine.set_linewidth(1.8)

    fig.suptitle('Same DNA, same noise - one model beside the other\n'
                 'grey frame = ' + labels[0] + '    blue frame = ' + labels[1],
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_structure_figure(dmats, labels, genotypes, kinship, stats, save_path):
    # Genotype-distance heatmaps, kinship, and the ratio with its CI.
    n_panels = len(labels) + (1 if kinship is not None else 0) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.5))
    axes = np.atleast_1d(axes)

    # Shared colour scale, or the two heatmaps cannot be compared by eye.
    vmax = max(d.max() for d in dmats.values())
    panel = 0
    for label in labels:
        ax = axes[panel]; panel += 1
        im = ax.imshow(dmats[label], cmap='viridis', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(genotypes)))
        ax.set_yticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=90, fontsize=7)
        ax.set_yticklabels(genotypes, fontsize=7)
        spread = upper_tri(dmats[label]).std() / upper_tri(dmats[label]).mean()
        ax.set_title(f'{label}\nimage distance between genotypes\n'
                     f'relative spread = {spread:.3f}', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    if kinship is not None:
        ax = axes[panel]; panel += 1
        im = ax.imshow(kinship, cmap='magma')
        ax.set_xticks(range(len(genotypes)))
        ax.set_yticks(range(len(genotypes)))
        ax.set_xticklabels(genotypes, rotation=90, fontsize=7)
        ax.set_yticklabels(genotypes, fontsize=7)
        ax.set_title('genetic relatedness (kinship)\n'
                     'higher = more closely related', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[panel]
    x = np.arange(len(labels))
    ratios = [stats[l]['conditioning_ratio'] for l in labels]
    los = [stats[l]['ratio_ci_low'] for l in labels]
    his = [stats[l]['ratio_ci_high'] for l in labels]
    err = np.array([[r - lo for r, lo in zip(ratios, los)],
                    [hi - r for r, hi in zip(ratios, his)]])
    ax.bar(x, ratios, yerr=err, capsize=6, color=['#888888', '#2b7bba'][:len(labels)])
    ax.axhline(1.0, color='crimson', ls='--', lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('genotype effect / noise effect')
    ax.set_title('Conditioning strength\n95% CI, resampled over genotypes', fontsize=9)
    for xi, r in zip(x, ratios):
        ax.text(xi, r + 0.02, f'{r:.3f}', ha='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_genotype_panel(genotype, original_img, generated_imgs, seed_labels,
                        seg_model, imgsz, conf, save_path):
    # One genotype: real photo + N generations, plain on top, segmented and
    # measured underneath.
    #
    # Top row lets you look at the images directly. Bottom row answers the
    # question a photo alone cannot: does the segmentation model agree the
    # generated roots have plausible, and plausibly DIFFERENT, structure -
    # not just plausible texture. original_img may be None if no photo was
    # found for this genotype; that column is then left blank rather than
    # dropped, so column position (image N) still lines up between panels.
    from feature_segmentation.evaluation.reconstruction_fidelity_test import measure, overlay

    images = [original_img] + list(generated_imgs)
    titles = ['original'] + [f'seed {s}' for s in seed_labels]
    n = len(images)

    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 4.7))

    rows_out = []
    for j, (img, title) in enumerate(zip(images, titles)):
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        axes[0, j].set_title(title, fontsize=9)

        if img is None:
            axes[0, j].text(0.5, 0.5, 'no photo\non file', ha='center',
                            va='center', fontsize=8, transform=axes[0, j].transAxes)
            axes[1, j].axis('off')
            rows_out.append({'image': title, **{k: np.nan for k in
                             ('root_diameter_px', 'stele_diameter_px',
                              'vessel_count_cc')}})
            continue

        axes[0, j].imshow(img)

        result = seg_model.predict(img[:, :, ::-1], conf=conf, imgsz=imgsz,
                                   verbose=False)[0]
        traits, masks = measure(result, imgsz=imgsz)
        axes[1, j].imshow(overlay(img, masks))
        axes[1, j].set_xlabel(format_trait_sublabel(traits), fontsize=7.5)

        rows_out.append({'image': title, **{k: traits.get(k, np.nan) for k in
                         ('root_diameter_px', 'stele_diameter_px',
                          'vessel_count_cc')}})

    axes[0, 0].set_ylabel('image', fontsize=9)
    axes[1, 0].set_ylabel('segmented +\nmeasured', fontsize=9)

    fig.suptitle(f'{genotype}   (one-hot model)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=145, bbox_inches='tight')
    plt.close(fig)
    return rows_out


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/compare_conditioning_grids.py
    class cfg:
        results_dir = RESULTS_DIR / 'conditioning_strength'
        # Must match the labels used in test_conditioning_strength.py, since
        # the .npy filenames were derived from them.
        labels = ['numeric (old)', 'one-hot (new)']

        kinship_path = KINSHIP_MATRIX
        # Fallback only. test_conditioning_strength.py now writes genotypes.json
        # into results_dir with the exact genotype order and seeds it used -
        # that is read automatically below and takes priority over these two
        # lists whenever the file is present, which it will be for any run
        # produced after that change. These stay here only so a results_dir
        # from an older run (no genotypes.json) still works.
        genotypes = ['MEMA018', 'MEMA025', 'MEMA041', 'MEMA098',
                     'MEMA114', 'MEMA180', 'MEMA233', 'MEMA298']
        seeds = [0, 1, 2, 3, 4, 5]

        show_seeds = [0, 1, 2]      # which seed columns appear in side_by_side.png
        # side_by_side.png and genotype_structure.png stop being readable well
        # before 200 genotypes - one gets 200 image rows, the other a 200x200
        # heatmap with illegible tick labels. Above this count, those two
        # figures are built from a fixed-seed random subsample instead; every
        # statistic (ratio, CI, kinship correlation, distance matrices) still
        # uses every genotype regardless, since those don't depend on being
        # human-legible.
        max_genotypes_in_figures = 24
        figure_subsample_seed = 0

        n_boot = 2000

        # per-genotype panels: original + generations + segmentation
        # For each genotype, one figure with the real photo plus several
        # one-hot-model generations on top, and the same images segmented and
        # measured (vessel count, root/stele diameter) underneath.
        build_genotype_panels = True
        panel_label = 'one-hot (new)'   # must be a key in cfg.labels
        n_show_seeds = 5                # how many generations to display
        seg_weights = SEGMENTATION_MODEL
        metadata_path = IMAGE_METADATA
        # Cropped roots, not dataset/images - the segmentation model was
        # trained on crops and the earlier feature-segmentation validation
        # found it detects the stele in ~91% of crops vs ~2% of raw images.
        image_dir = CROPPED_IMAGES_DIR
        seg_conf = 0.25
        imgsz = 256

    out = resolve_output(cfg.results_dir)
    print(f"Reading grids from: {out}")

    grids = {}
    for label in cfg.labels:
        fname = f'grid_{label.replace(" ", "_").replace("/", "-")}.npy'
        path = Path(out) / fname
        if not path.exists():
            raise SystemExit(
                f"missing {path}\nRun test_conditioning_strength.py first, and "
                f"make sure cfg.labels here matches the labels it used.")
        grids[label] = np.load(path)
        print(f"  {label}: {grids[label].shape}")

    genotypes_json = Path(out) / 'genotypes.json'
    if genotypes_json.exists():
        with open(genotypes_json) as f:
            saved = json.load(f)
        genotypes = saved['genotypes']
        run_seeds = saved['seeds']
        print(f"  genotypes.json: {len(genotypes)} genotypes, "
              f"seeds {run_seeds} (overrides cfg.genotypes/cfg.seeds)")
    else:
        genotypes = cfg.genotypes
        run_seeds = cfg.seeds
        print("  no genotypes.json in results_dir - falling back to "
              "cfg.genotypes/cfg.seeds (only correct for runs from before "
              "test_conditioning_strength.py started saving it)")

    n_geno = next(iter(grids.values())).shape[0]
    if len(genotypes) != n_geno:
        raise SystemExit(
            f"{'genotypes.json lists' if genotypes_json.exists() else 'cfg.genotypes lists'} "
            f"{len(genotypes)} names but the grids hold {n_geno} genotypes - "
            f"results_dir may hold grids from a different run than genotypes.json.")

    kinship = load_kinship(cfg.kinship_path, genotypes)
    if kinship is None:
        print("  kinship matrix unavailable or missing these genotypes - "
              "skipping the relatedness comparison")

    # statistics
    stats, dmats = {}, {}
    for label in cfg.labels:
        grid = grids[label]
        geno_dist, noise_dist = precompute_pairwise(grid)
        g_vals, n_vals = decompose(geno_dist, noise_dist)
        lo, hi = bootstrap_ratio(geno_dist, noise_dist, n_boot=cfg.n_boot)
        D = genotype_distance_matrix(geno_dist)
        dmats[label] = D

        entry = {
            'genotype_effect_mean': float(g_vals.mean()),
            'genotype_effect_std': float(g_vals.std()),
            'noise_effect_mean': float(n_vals.mean()),
            'noise_effect_std': float(n_vals.std()),
            'conditioning_ratio': float(g_vals.mean() / n_vals.mean()),
            'ratio_ci_low': lo,
            'ratio_ci_high': hi,
            # How unevenly the model separates genotype pairs. A model that
            # treats every pair alike scores near 0 here regardless of how
            # large its mean genotype effect is.
            'genotype_distance_spread': float(upper_tri(D).std() / upper_tri(D).mean()),
        }
        if kinship is not None:
            r = float(pd.Series(upper_tri(D)).corr(pd.Series(upper_tri(kinship))))
            entry['kinship_correlation'] = r
        stats[label] = entry

    # figures
    # side_by_side.png (one row per genotype) and the genotype_structure.png
    # heatmaps (n_geno x n_geno with a tick label per genotype) stop being
    # readable well before the full dataset's ~200 genotypes. Above the cap, a
    # fixed-seed random subset stands in for these two figures specifically;
    # every number in comparison_stats.csv/json and the full distance-matrix
    # CSVs below still comes from all of genotypes, since only human legibility
    # is the constraint here, not statistical validity.
    if len(genotypes) > cfg.max_genotypes_in_figures:
        fig_rng = np.random.default_rng(cfg.figure_subsample_seed)
        fig_idx = np.sort(fig_rng.choice(len(genotypes), cfg.max_genotypes_in_figures,
                                        replace=False))
        print(f"\n  {len(genotypes)} genotypes exceeds max_genotypes_in_figures="
              f"{cfg.max_genotypes_in_figures}; side_by_side.png and "
              f"genotype_structure.png use a random {cfg.max_genotypes_in_figures}-"
              f"genotype subset. Statistics and distance-matrix CSVs still use "
              f"all {len(genotypes)}.")
    else:
        fig_idx = np.arange(len(genotypes))

    fig_genotypes = [genotypes[i] for i in fig_idx]
    fig_grids = {label: grids[label][fig_idx] for label in cfg.labels}
    fig_dmats = {label: dmats[label][np.ix_(fig_idx, fig_idx)] for label in cfg.labels}
    # kinship (if available) is [n_geno, n_geno] aligned with the full
    # genotypes list - it needs the same subsetting or its shape would no
    # longer match fig_genotypes and the heatmap would show the wrong pairs.
    fig_kinship = kinship[np.ix_(fig_idx, fig_idx)] if kinship is not None else None

    # cfg.show_seeds names seed COLUMN INDICES, which only makes sense against
    # however many seeds this particular run actually has - clamped here
    # rather than left to fail inside matplotlib with an IndexError, since a
    # smaller n_seeds (e.g. cut down for a full-dataset run, see
    # test_conditioning_strength.py's cfg.n_seeds comment) is an expected,
    # not exceptional, way to end up with fewer seed columns than 3.
    show_seeds = [s for s in cfg.show_seeds if s < len(run_seeds)] or [0]
    save_side_by_side(fig_grids, cfg.labels, fig_genotypes, run_seeds, show_seeds,
                      Path(out) / 'side_by_side.png')
    save_structure_figure(fig_dmats, cfg.labels, fig_genotypes, fig_kinship, stats,
                          Path(out) / 'genotype_structure.png')

    for label in cfg.labels:
        np.savetxt(Path(out) / f'distance_matrix_{label.replace(" ", "_")}.csv',
                   dmats[label], delimiter=',',
                   header=','.join(genotypes), comments='')

    pd.DataFrame(stats).T.to_csv(Path(out) / 'comparison_stats.csv')
    with open(Path(out) / 'comparison_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # report
    print(f"{'':<24}{'ratio':>8}{'95% CI':>18}{'pair spread':>14}{'kinship r':>12}")
    for label in cfg.labels:
        s = stats[label]
        ci = f"[{s['ratio_ci_low']:.3f}, {s['ratio_ci_high']:.3f}]"
        kin = f"{s['kinship_correlation']:+.3f}" if 'kinship_correlation' in s else '   n/a'
        print(f"{label:<24}{s['conditioning_ratio']:>8.3f}{ci:>18}"
              f"{s['genotype_distance_spread']:>14.3f}{kin:>12}")

    a, b = cfg.labels[0], cfg.labels[1]
    overlap = (stats[a]['ratio_ci_low'] <= stats[b]['ratio_ci_high'] and
               stats[b]['ratio_ci_low'] <= stats[a]['ratio_ci_high'])
    print(f"\n  Confidence intervals "
          f"{'OVERLAP - the ratio difference is not statistically resolved' if overlap else 'are separated'}.")

    print(f"\n  pair spread: how unevenly a model separates different genotype "
          f"pairs.\n  Higher means it distinguishes between them rather than "
          f"treating all\n  genomes as roughly equally different.")
    if kinship is not None:
        print(f"\n  kinship r: correlation between image distance and relatedness.\n"
              f"  Should be NEGATIVE - more related genotypes ought to generate\n"
              f"  more similar images. Near zero means the differences the model\n"
              f"  produces are not tracking real genetics.")

    print(f"\nWrote side_by_side.png, genotype_structure.png, distance matrices,\n"
          f"comparison_stats.csv and comparison_stats.json to {out}")

    # per-genotype panels
    if cfg.build_genotype_panels:
        if cfg.panel_label not in grids:
            raise SystemExit(
                f"cfg.panel_label={cfg.panel_label!r} is not in cfg.labels "
                f"{cfg.labels} - fix one or the other.")

        print(f"\n\nPer-genotype panels ({cfg.panel_label})\n")
        from ultralytics import YOLO

        seg_model = YOLO(str(resolve_input(cfg.seg_weights, 'segmentation weights')))
        metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))
        image_dir = resolve_input(cfg.image_dir, 'image directory')

        panel_grid = grids[cfg.panel_label]
        n_show = min(cfg.n_show_seeds, panel_grid.shape[1])
        seed_labels = run_seeds[:n_show]
        panel_dir = Path(out) / 'genotype_panels'
        panel_dir.mkdir(exist_ok=True)

        all_trait_rows = []
        n_missing_photo = 0
        for g_i, genotype in enumerate(genotypes):
            photo_path = find_original_image(genotype, metadata, image_dir)
            if photo_path is None:
                n_missing_photo += 1
                original_img = None
            else:
                original_img = load_original(photo_path, cfg.imgsz)

            generated_imgs = [panel_grid[g_i, s_i] for s_i in range(n_show)]

            rows = save_genotype_panel(
                genotype, original_img, generated_imgs, seed_labels,
                seg_model, cfg.imgsz, cfg.seg_conf,
                panel_dir / f'{genotype}.png')
            for r in rows:
                r['genotype'] = genotype
            all_trait_rows.extend(rows)
            print(f"  {genotype}: wrote panel"
                 f"{' (no original photo found)' if photo_path is None else ''}")

        trait_df = pd.DataFrame(all_trait_rows)[
            ['genotype', 'image', 'root_diameter_px', 'stele_diameter_px',
             'vessel_count_cc']]
        trait_df.to_csv(Path(out) / 'genotype_panel_traits.csv', index=False)

        if n_missing_photo:
            print(f"\n  {n_missing_photo}/{len(genotypes)} genotypes had no "
                 f"original photo under {image_dir} - those panels show the "
                 f"generations only.")
        print(f"\nWrote {len(genotypes)} panels to {panel_dir} and "
             f"genotype_panel_traits.csv to {out}")


if __name__ == '__main__':
    main()
