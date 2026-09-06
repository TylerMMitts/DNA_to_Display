# Does the model do better on genotypes it trained on than ones it never saw?
#
# genetic_fidelity_test.py measures genotype -> trait accuracy over every
# genotype pooled together. That answers "is it accurate", but not the question
# that decides what to fix next: whether the accuracy that exists comes from
# having memorised specific training genotypes. Splitting the same measurements
# by train/val separates two very different situations that look identical when
# pooled:
#
# train >> val      the conditioning pathway CAN learn genotype -> anatomy,
#                   but it memorised individuals instead of learning
#                   something transferable. The fix is regularisation, more
#                   genotypes, or fewer conditioning dimensions.
#
# train ~ val ~ 0   it never learned the mapping at all, not even on data it
#                   saw hundreds of times. No amount of regularisation helps
#                   something that was never fit; the fix is the objective or
#                   the architecture (auxiliary trait supervision, far fewer
#                   PCA components).
#
# Costs no generation and no segmentation: it reads the per-image real and
# generated trait measurements genetic_fidelity_test.py already wrote to
# per_image_measurements.csv, and only adds the split dimension. Run that
# script first.
#
# Two things this handles that a naive split-and-compare would get wrong:
#
# the permutation null is recomputed PER SPLIT
#     With ~110 train genotypes and ~27 val genotypes, chance correlation is
#     much higher on the smaller split - roughly 0.16 at 137 genotypes but
#     appreciably larger at 27. Comparing raw r between splits without that
#     would make val look worse than train even when both are pure noise.
#
# the split is taken from the checkpoint, not reconstructed, when possible
#     train_onehot.py records val_genotypes in every checkpoint it writes, so
#     the exact partition the model was trained under is recoverable. Only
#     the legacy train.py checkpoints need reconstructing (seed 42, shuffle,
#     80/20 by genotype), and that reconstruction is only valid while the SNP
#     parquet still yields sample_names in the same order.
#
# Usage
#     python code/feature_segmentation/evaluation/train_vs_test_accuracy.py

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
from feature_segmentation.evaluation.genetic_fidelity_test import (
    TRAITS, repeatability, permutation_null, safe_corr,
)


# Recovering the split the checkpoint was trained under

def split_from_checkpoint(checkpoint_path, snp_parquet):
    # (val_genotypes, description). Prefers the checkpoint's own record.
    #
    # train_onehot.py saves 'val_genotypes' precisely so this never has to be
    # guessed - a reconstructed split that disagreed with the real one by even
    # a few genotypes would quietly contaminate the "unseen" set with training
    # data and flatter the val numbers.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'val_genotypes' in ckpt:
        return set(ckpt['val_genotypes']), (
            f"read from checkpoint ({len(ckpt['val_genotypes'])} val genotypes)")

    # Legacy train.py: reproduce its create_dataloaders() split exactly -
    # np.random.seed(42), shuffle the parquet's sample_names, first 80% train.
    sample_names, _, _ = load_snp_data_from_parquet(snp_parquet)
    names = list(sample_names)
    np.random.seed(42)
    np.random.shuffle(names)
    split_idx = int(len(names) * 0.8)
    return set(names[split_idx:]), (
        f"reconstructed from train.py's seed-42 80/20 split "
        f"({len(names) - split_idx} val genotypes of {len(names)}) - "
        f"valid only if the parquet still orders genotypes the same way")


# Per-split statistics

def split_stats(df, trait, n_perm, seed):
    # Genotype-level accuracy for one trait on one split of the data.
    geno = df.groupby('genotype').agg(
        real=(f'real_{trait}', 'mean'),
        gen=(f'gen_{trait}', 'mean'),
        n_images=('filename', 'size')).reset_index()

    null_mean, null_p95 = permutation_null(geno['real'], geno['gen'],
                                           n_perm=n_perm, seed=seed)
    icc, var_b, var_w = repeatability(df[f'real_{trait}'], df['genotype'])
    r = safe_corr(geno['real'], geno['gen'])

    # Same four error statistics as litevae_ceiling_stats, same names, same
    # formulas - computed here on the genotype-level means (geno['real'] /
    # geno['gen']) rather than per image, consistent with how pearson/
    # spearman above are already genotype-level for this function. Matching
    # names lets the two CSVs be read side by side as the same kind of table
    # even though one is genotype-level and the other per-image.
    err = geno['gen'] - geno['real']
    denom = geno['real'].abs().replace(0, np.nan)

    return {
        'n_genotypes': int(len(geno)),
        'n_images': int(len(df)),
        'pearson': r,
        'spearman': safe_corr(geno['real'], geno['gen'], method='spearman'),
        'mean_abs_error': float(err.abs().mean()),
        'signed_bias': float(err.mean()),
        'mean_abs_pct_error': float((err.abs() / denom).mean() * 100),
        'variance_ratio': (float(geno['gen'].var() / geno['real'].var())
                           if geno['real'].var() > 0 else float('nan')),
        'permutation_null_mean': null_mean,
        'permutation_null_p95': null_p95,
        # The honest headline: a correlation that does not clear its own
        # split's chance threshold is not evidence of anything, and the
        # threshold differs between splits because their sizes do.
        'beats_chance': bool(np.isfinite(r) and np.isfinite(null_p95)
                             and r > null_p95),
        'margin_over_chance': (float(r - null_p95)
                               if np.isfinite(r) and np.isfinite(null_p95)
                               else float('nan')),
        'icc_real': icc,
        'real_mean': float(geno['real'].mean()),
        'gen_mean': float(geno['gen'].mean()),
    }


# LiteVAE: the autoencoder ceiling

def litevae_ceiling_stats(df, trait):
    # How well a real image's trait survives encode -> decode.
    #
    # LiteVAE is unconditioned, so "genotype -> trait accuracy" is not defined
    # for it. What IS defined, and matters more, is whether the trait survives
    # the round trip at all: the diffusion model produces latents that LiteVAE
    # decodes, so any trait the autoencoder cannot preserve on a REAL image is
    # a trait no conditioning could make it produce correctly. This is a
    # ceiling on the whole pipeline, measured independently of any genotype.
    #
    # Paired per image (original vs its own reconstruction), not aggregated by
    # genotype - the question is whether this specific image comes back
    # carrying the same measurement, which is a property of the autoencoder
    # rather than of any genetic grouping.
    o, r = f'orig_{trait}', f'recon_{trait}'
    if o not in df.columns or r not in df.columns:
        return None

    sub = df[[o, r]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return None

    err = sub[r] - sub[o]
    denom = sub[o].abs().replace(0, np.nan)
    return {
        'n_images': int(len(sub)),
        'pearson': safe_corr(sub[o], sub[r]),
        'spearman': safe_corr(sub[o], sub[r], method='spearman'),
        'mean_abs_error': float(err.abs().mean()),
        'signed_bias': float(err.mean()),
        'mean_abs_pct_error': float((err.abs() / denom).mean() * 100),
        'orig_mean': float(sub[o].mean()),
        'recon_mean': float(sub[r].mean()),
        # Does the reconstruction preserve BETWEEN-image spread, or regress
        # everything toward the population mean? A ratio well under 1 means
        # the autoencoder is flattening the trait, which erases exactly the
        # variation a genotype signal would have to live in.
        'variance_ratio': (float(sub[r].var() / sub[o].var())
                           if sub[o].var() > 0 else float('nan')),
    }


def save_ceiling_figure(litevae_rows, diffusion_rows, save_path):
    # Autoencoder trait preservation next to the diffusion model's accuracy.
    traits = [t for t, _, _ in TRAITS]
    lv = {r['trait']: r for r in litevae_rows}
    dv = {r['trait']: r for r in diffusion_rows if r['split'] == 'val'}

    x = np.arange(len(traits))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))

    ax = axes[0]
    vals = [lv[t]['pearson'] if t in lv else np.nan for t in traits]
    bars = ax.bar(x, vals, color='#4c9a6a')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(traits, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('orig vs reconstruction Pearson r')
    ax.set_title('LiteVAE: does the trait survive encode -> decode?\n'
                 'low bar = the autoencoder itself destroys this trait',
                 fontsize=10.5)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    keep = [lv[t]['variance_ratio'] if t in lv else np.nan for t in traits]
    ax.bar(x, keep, color='#4c9a6a')
    ax.axhline(1.0, color='crimson', ls='--', lw=1.3,
               label='spread fully preserved')
    ax.set_xticks(x); ax.set_xticklabels(traits, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('reconstruction variance / original variance')
    ax.set_title('Does it preserve BETWEEN-image spread?\n'
                 'well under 1 = traits flattened toward the mean',
                 fontsize=10.5)
    ax.legend(fontsize=8)

    fig.suptitle('Autoencoder ceiling: the diffusion model cannot express a trait '
                 'LiteVAE cannot preserve', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_split_figure(rows, save_path):
    # Train vs val correlation per trait, each against its own chance line.
    traits = [t for t, _, _ in TRAITS]
    fig, axes = plt.subplots(1, len(traits), figsize=(3.5 * len(traits), 4.6),
                             squeeze=False)

    for ax, trait in zip(axes[0], traits):
        sub = {r['split']: r for r in rows if r['trait'] == trait}
        splits = [s for s in ('train', 'val') if s in sub]
        x = np.arange(len(splits))
        vals = [sub[s]['pearson'] for s in splits]
        nulls = [sub[s]['permutation_null_p95'] for s in splits]

        bars = ax.bar(x, vals, color=['#2b7bba' if sub[s]['beats_chance']
                                      else '#c9622a' for s in splits])
        # One chance line per bar, not one for the panel - the threshold is
        # split-size dependent, so a single shared line would be wrong.
        for xi, nv in zip(x, nulls):
            ax.plot([xi - 0.4, xi + 0.4], [nv, nv], color='crimson',
                    ls='--', lw=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(n={sub[s]['n_genotypes']})" for s in splits],
                           fontsize=9)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(trait, fontsize=9.5)
        ax.set_ylabel('genotype-level Pearson r')
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                        ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)

    fig.suptitle('Genotype -> trait accuracy, seen vs unseen genotypes\n'
                 'dashed red = that split\'s own chance threshold (95th pct of '
                 'shuffled labels); blue clears it, orange does not',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/train_vs_test_accuracy.py
    class cfg:
        # Written by genetic_fidelity_test.py - run that first. Must come from
        # the SAME checkpoint named below, or the split will be applied to
        # measurements the model never produced.
        measurements = RESULTS_DIR / 'genetic_fidelity' / 'per_image_measurements.csv'
        # Used only to recover the train/val split, not to generate anything.
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET

        # Written by reconstruction_fidelity_test.py - real image vs its own
        # LiteVAE reconstruction, trait by trait. None skips the autoencoder
        # section entirely.
        #
        # No train/val split is reported for LiteVAE because none is
        # recoverable: VAE_training.py calls torch.utils.data.random_split
        # with no generator and never calls manual_seed, and its checkpoints
        # save weights and losses but no split record - so which images were
        # held out during that run is simply not knowable after the fact.
        # CustomImageDataset also enumerates files with os.walk, whose order
        # is filesystem-dependent, so even adding a seed would not make an
        # OLD split reproducible. The autoencoder numbers below are therefore
        # pooled over every measured image, and are a ceiling rather than a
        # generalisation test.
        litevae_measurements = RESULTS_DIR / 'reconstruction_fidelity' / 'measurements.csv'

        output_dir = RESULTS_DIR / 'train_vs_test_accuracy'

        n_permutations = 1000
        seed = 0

    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    df = pd.read_csv(resolve_input(cfg.measurements, 'per-image measurements'))
    print(f"Measurements: {len(df)} images, {df['genotype'].nunique()} genotypes")

    ckpt_path = resolve_input(cfg.checkpoint, 'checkpoint')
    val_genotypes, how = split_from_checkpoint(ckpt_path,
                                               resolve_input(cfg.snp_parquet, 'SNP parquet'))
    print(f"Split: {how}")

    df['split'] = np.where(df['genotype'].isin(val_genotypes), 'val', 'train')
    present = df.groupby('split')['genotype'].nunique().to_dict()
    print(f"  measured genotypes by split: {present}")

    if present.get('val', 0) < 3 or present.get('train', 0) < 3:
        raise SystemExit(
            "one side of the split has fewer than 3 measured genotypes - the "
            "correlations and permutation nulls would be meaningless. Check "
            "that cfg.checkpoint matches the checkpoint the measurements came "
            "from.")

    # per split, per trait
    rows = []
    for trait, label, _ in TRAITS:
        for split in ('train', 'val'):
            sub = df[df['split'] == split]
            if sub.empty:
                continue
            rows.append({'trait': trait, 'label': label, 'split': split,
                         **split_stats(sub, trait, cfg.n_permutations, cfg.seed)})

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out / 'train_vs_test_stats.csv', index=False)
    save_split_figure(rows, out / 'train_vs_test_accuracy.png')

    with open(out / 'summary.json', 'w') as f:
        json.dump({'checkpoint': str(ckpt_path), 'split_source': how,
                   'measurements': str(cfg.measurements), 'rows': rows},
                  f, indent=2)

    # report
    # Same four columns (r, var ratio, bias, MAPE%) and formatting as the
    # LiteVAE ceiling table below, plus the columns specific to what this
    # table is actually for - a split comparison against chance, which the
    # LiteVAE table has no equivalent of since it has no train/val split.
    print(f"{'trait':<22}{'split':>6}{'n':>5}{'r':>8}{'var ratio':>10}"
          f"{'bias':>9}{'MAPE %':>8}{'chance':>8}{'beats?':>7}{'ICC':>6}")
    for trait, label, _ in TRAITS:
        for split in ('train', 'val'):
            m = [r for r in rows if r['trait'] == trait and r['split'] == split]
            if not m:
                continue
            r = m[0]
            print(f"{label if split == 'train' else '':<22}{split:>6}"
                  f"{r['n_genotypes']:>5}{r['pearson']:>8.3f}"
                  f"{r['variance_ratio']:>10.3f}{r['signed_bias']:>9.2f}"
                  f"{r['mean_abs_pct_error']:>8.1f}"
                  f"{r['permutation_null_p95']:>8.3f}"
                  f"{('YES' if r['beats_chance'] else 'no'):>7}"
                  f"{r['icc_real']:>6.3f}")

    n_train_beat = sum(1 for r in rows if r['split'] == 'train' and r['beats_chance'])
    n_val_beat = sum(1 for r in rows if r['split'] == 'val' and r['beats_chance'])
    n_traits = len(TRAITS)
    print(f"\n  traits beating chance:  train {n_train_beat}/{n_traits}   "
          f"val {n_val_beat}/{n_traits}")

    print("\n  How to read this:")
    if n_train_beat == 0 and n_val_beat == 0:
        print("    Neither split clears chance. The model has not learned the")
        print("    genotype -> trait mapping even on genotypes it saw repeatedly")
        print("    during training, so this is not an overfitting problem and")
        print("    regularisation will not help. The lever is the objective or")
        print("    the conditioning representation - auxiliary trait supervision,")
        print("    or far fewer PCA components than the current 168 from 200")
        print("    genotypes.")
    elif n_train_beat > n_val_beat:
        print("    Train clears chance where val does not: the pathway CAN fit")
        print("    genotype -> trait, but what it fit does not transfer to unseen")
        print("    genotypes. That is a generalisation gap, and the levers are")
        print("    more genotypes, stronger regularisation, or a lower-dimensional")
        print("    conditioning vector.")
    else:
        print("    Val clears chance too, so the mapping is genuinely")
        print("    transferable rather than memorised. Compare the margins")
        print("    rather than the raw correlations - the chance threshold is")
        print("    higher on the smaller split.")

    n_tests = len(TRAITS) * 2
    expected_fp = 0.05 * n_tests
    print(f"\n  Multiple comparisons: {n_tests} tests are run ({len(TRAITS)} traits x")
    print(f"  2 splits) against a 95th-percentile threshold, so about "
          f"{expected_fp:.1f} of them")
    print("  are expected to clear it by chance alone. Verified on synthetic data")
    print("  with no signal at all, where 1 of 10 still crossed. A SINGLE trait")
    print("  clearing the line is therefore weak evidence; several clearing it on")
    print("  the same split, or one clearing it by a wide margin, is not.")
    print("  margin_over_chance in the CSV is the number to compare.")

    print("\n  ICC is the ceiling from the REAL images on that split, not a")
    print("  model score: it is the share of trait variance that lies between")
    print("  genotypes at all. A trait cannot be predicted from genotype more")
    print("  accurately than its own ICC allows, however good the model is.")

    # LiteVAE: the autoencoder ceiling
    litevae_rows = []
    if cfg.litevae_measurements:
        try:
            lv_path = resolve_input(cfg.litevae_measurements,
                                    'LiteVAE reconstruction measurements')
        except FileNotFoundError as exc:
            print(f"\nSkipping LiteVAE section: {exc}")
            print("  (run reconstruction_fidelity_test.py to produce it)")
            lv_path = None

        if lv_path is not None:
            lv_df = pd.read_csv(lv_path)
            print(f"LITEVAE AUTOENCODER CEILING  ({len(lv_df)} images)")
            print("  Pooled over all images - LiteVAE's own train/val split is not")
            print("  recoverable (no seed, no split saved in the checkpoint), so")
            print("  this is a ceiling measurement, not a generalisation test.")
            print()
            print(f"{'trait':<26}{'r':>9}{'var ratio':>11}{'bias':>11}{'MAPE %':>9}")

            for trait, label, _ in TRAITS:
                s = litevae_ceiling_stats(lv_df, trait)
                if s is None:
                    print(f"{label:<26}{'n/a':>9}   (not present in measurements)")
                    continue
                litevae_rows.append({'trait': trait, 'label': label, **s})
                print(f"{label:<26}{s['pearson']:>9.3f}{s['variance_ratio']:>11.3f}"
                      f"{s['signed_bias']:>11.2f}{s['mean_abs_pct_error']:>9.1f}")

            if litevae_rows:
                pd.DataFrame(litevae_rows).to_csv(
                    out / 'litevae_ceiling_stats.csv', index=False)
                save_ceiling_figure(litevae_rows, rows,
                                    out / 'litevae_ceiling.png')

                # Which traits does the autoencoder itself already lose? Those
                # cannot be rescued by anything done to the conditioning.
                weak = [r for r in litevae_rows
                        if np.isfinite(r['pearson']) and r['pearson'] < 0.5]
                flat = [r for r in litevae_rows
                        if np.isfinite(r['variance_ratio']) and r['variance_ratio'] < 0.5]

                print("\n  Reading this: r is how faithfully a real image's trait")
                print("  survives encode -> decode. variance ratio is whether spread")
                print("  BETWEEN images survives - a trait flattened toward the mean")
                print("  carries no genotype signal even if its correlation looks ok.")
                if weak:
                    print(f"\n  Poorly preserved (r < 0.5): "
                          f"{', '.join(r['label'] for r in weak)}")
                    print("    The diffusion model cannot express these accurately no")
                    print("    matter how good its conditioning becomes - the")
                    print("    autoencoder loses them before conditioning is even")
                    print("    involved. Fixing the SNP encoder cannot recover them.")
                if flat:
                    print(f"\n  Spread collapsed (variance ratio < 0.5): "
                          f"{', '.join(r['label'] for r in flat)}")
                    print("    Reconstructions regress toward the population mean for")
                    print("    these, which is the same signature the founder")
                    print("    archetypes showed. Worth checking whether that")
                    print("    flattening originates here rather than in the UNet.")
                if not weak and not flat:
                    print("\n  The autoencoder preserves every trait well, so it is NOT")
                    print("  the binding constraint - the genotype-level failures above")
                    print("  belong to the conditioning pathway, not to LiteVAE.")

        with open(out / 'summary.json', 'w') as f:
            json.dump({'checkpoint': str(ckpt_path), 'split_source': how,
                       'measurements': str(cfg.measurements),
                       'diffusion_rows': rows,
                       'litevae_ceiling': litevae_rows}, f, indent=2)

    print(f"\nWrote train_vs_test_stats.csv, train_vs_test_accuracy.png"
          f"{', litevae_ceiling_stats.csv, litevae_ceiling.png' if litevae_rows else ''}"
          f" and summary.json to {out}")


if __name__ == '__main__':
    main()
