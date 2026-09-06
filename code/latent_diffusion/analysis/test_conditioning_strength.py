# Does changing the genotype change the image more than changing the seed?
#
# If seed variation dominates, the model is producing plausible roots while
# largely ignoring the genetics it was conditioned on. Runs both the numeric
# and one-hot checkpoints so the two can be compared.

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
    DIFFUSION_NUMERIC_MODEL, DIFFUSION_ONEHOT_MODEL, LITEVAE_MODEL,
    RESULTS_DIR, SNP_PARQUET, resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from litevae.models import LiteVAEDecoder
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.generation.generate_from_dataset import generate_batch


# Distances

def rmse(a, b):
    # Root mean squared difference between two uint8 RGB images, in 0-255.
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((d ** 2).mean()))


def decompose(grid):
    # Splits image variation into genotype-driven and noise-driven parts.
    #
    # grid is [n_genotypes, n_seeds, H, W, 3].
    #
    # Returns the mean pairwise RMSE across genotypes at fixed noise, the mean
    # pairwise RMSE across seeds at fixed genotype, and the per-slice values so
    # the spread can be reported alongside the mean. A single mean with no spread
    # would make a 5% gap between checkpoints look as solid as a 200% one.
    n_geno, n_seeds = grid.shape[:2]

    genotype_vals = []
    for s in range(n_seeds):
        for g1, g2 in combinations(range(n_geno), 2):
            genotype_vals.append(rmse(grid[g1, s], grid[g2, s]))

    noise_vals = []
    for g in range(n_geno):
        for s1, s2 in combinations(range(n_seeds), 2):
            noise_vals.append(rmse(grid[g, s1], grid[g, s2]))

    return np.array(genotype_vals), np.array(noise_vals)


# Figures

def save_grid_figure(grid, genotypes, seeds, title, save_path, max_rows=24,
                     subsample_seed=0):
    # The genotype x seed grid itself.
    #
    # Reading it: down a COLUMN the noise is fixed, so any variation is the
    # genotype talking. Across a ROW the genotype is fixed, so variation is the
    # noise. A grid whose columns are internally identical is a model ignoring
    # its conditioning, and that is visible here without any statistics.
    #
    # Capped at max_rows: at the full dataset's ~200 genotypes this would be a
    # 200-row, tens-of-megapixel image that is unusable as a "look at it"
    # figure - only grid_*.npy needs every genotype, since that is what
    # compare_conditioning_grids.py's statistics are computed from.
    if grid.shape[0] > max_rows:
        n_total = grid.shape[0]
        rng = np.random.default_rng(subsample_seed)
        idx = np.sort(rng.choice(n_total, max_rows, replace=False))
        grid = grid[idx]
        genotypes = [genotypes[i] for i in idx]
        title = title + f'\n(random {max_rows}/{n_total} genotype subset)'

    n_geno, n_seeds = grid.shape[:2]
    fig, axes = plt.subplots(n_geno, n_seeds,
                             figsize=(1.55 * n_seeds, 1.62 * n_geno))
    axes = np.atleast_2d(axes)
    if n_seeds == 1:
        axes = axes.reshape(n_geno, 1)

    for i in range(n_geno):
        for j in range(n_seeds):
            ax = axes[i, j]
            ax.imshow(grid[i, j])
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'seed {seeds[j]}', fontsize=8)
            if j == 0:
                ax.set_ylabel(genotypes[i], fontsize=7, rotation=0,
                              ha='right', va='center', labelpad=28)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def save_comparison_figure(results, save_path):
    names = list(results.keys())
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Left: the two effects side by side, with spread.
    ax = axes[0]
    width = 0.38
    g_means = [results[n]['genotype_effect_mean'] for n in names]
    g_errs = [results[n]['genotype_effect_std'] for n in names]
    n_means = [results[n]['noise_effect_mean'] for n in names]
    n_errs = [results[n]['noise_effect_std'] for n in names]
    ax.bar(x - width / 2, g_means, width, yerr=g_errs, capsize=4,
           label='genotype effect (fixed noise)', color='#2b7bba')
    ax.bar(x + width / 2, n_means, width, yerr=n_errs, capsize=4,
           label='noise effect (fixed genotype)', color='#bbbbbb')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('mean pairwise RMSE (0-255)')
    ax.set_title('What moves the image?', fontsize=11)
    ax.legend(fontsize=8)

    # Right: the normalised headline number.
    ax = axes[1]
    ratios = [results[n]['conditioning_ratio'] for n in names]
    bars = ax.bar(x, ratios, color=['#888888', '#2b7bba'][:len(names)]
                  if len(names) <= 2 else None)
    ax.axhline(1.0, color='crimson', ls='--', lw=1.2,
               label='genotype matters as much as noise')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('genotype effect / noise effect')
    ax.set_title('Conditioning strength\n(higher = DNA drives the image more)',
                 fontsize=11)
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v + max(ratios) * 0.02,
                f'{v:.3f}', ha='center', fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/test_conditioning_strength.py
    class cfg:
        # Checkpoints to compare, as {label: path}. Both encodings are handled
        # automatically - load_model reads each checkpoint's own encoding and
        # rebuilds the matching SNP pathway.
        checkpoints = {
            'numeric (old)': DIFFUSION_NUMERIC_MODEL,
            'one-hot (new)': DIFFUSION_ONEHOT_MODEL,
        }

        litevae_checkpoint = LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'   # legacy path only

        output_dir = RESULTS_DIR / 'conditioning_strength'

        # Genotypes are drawn at random rather than chosen for maximum genetic
        # distance. Hand-picking the most dissimilar genotypes would give the
        # conditioning its best possible showing, which is the wrong default
        # for a test whose job is to report honestly whether it works.
        #
        # None -> every genotype in the SNP data (no subsampling). Total
        # generations per checkpoint is n_genotypes * n_seeds, so going from 8
        # to "all ~200" multiplies the DDIM sampling cost by ~25x - at
        # sampling_steps=50 that is the dominant cost of this script by far.
        # Lowering n_seeds is the cheapest way to bring that back down: the
        # noise-effect estimate gets its power from many genotypes x few
        # seeds just as well as from few genotypes x many seeds, since
        # decompose() pools every genotype's seed-pairs together anyway.
        n_genotypes = None
        n_seeds = 3
        genotype_seed = 0        # which genotypes get drawn when n_genotypes is a number

        sampling_steps = 50
        batch_size = 8
        imgsz = 256
        latent_size = 32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    # data
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    if cfg.n_genotypes is None:
        idx = np.arange(len(sample_names))
    else:
        rng = np.random.default_rng(cfg.genotype_seed)
        idx = rng.choice(len(sample_names),
                         size=min(cfg.n_genotypes, len(sample_names)), replace=False)
        idx = np.sort(idx)
    genotypes = [sample_names[i] for i in idx]
    seeds = list(range(cfg.n_seeds))
    print(f"\nGenotypes ({len(genotypes)}): "
          f"{genotypes if len(genotypes) <= 20 else genotypes[:20] + ['...']}")
    print(f"Seeds: {seeds}")

    n_jobs_per_ckpt = len(genotypes) * len(seeds)
    print(f"\n{n_jobs_per_ckpt} generations per checkpoint "
          f"({len(cfg.checkpoints)} checkpoints, {cfg.sampling_steps} DDIM steps each) "
          f"= {n_jobs_per_ckpt * len(cfg.checkpoints)} generations total.")

    # Saved so compare_conditioning_grids.py can read the exact genotype order
    # back out instead of needing it typed in by hand - a mismatch there would
    # silently mislabel every panel and heatmap row.
    with open(out / 'genotypes.json', 'w') as f:
        json.dump({'genotypes': genotypes, 'seeds': seeds}, f, indent=2)

    # shared decoder
    vae_ckpt = torch.load(resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                          map_location=device, weights_only=False)

    scheduler = DiffusionScheduler()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    # run each checkpoint
    results = {}
    for label, ckpt_rel in cfg.checkpoints.items():
        print(f"\n\n{label}\n")
        try:
            ckpt_path = resolve_input(ckpt_rel, f"checkpoint for '{label}'")
        except FileNotFoundError as exc:
            print(f"SKIPPING: {exc}")
            continue

        snp_encoder, unet, unet_cfg = load_model(
            ckpt_path, snp_matrix, device,
            pca_cache=str(resolve_output(cfg.pca_cache)))

        decoder = LiteVAEDecoder(latent_channels=unet_cfg['latent_channels'],
                                 output_channels=3, base_channels=512,
                                 num_res_blocks=2)
        decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
        decoder.to(device).eval()
        latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

        # Flatten the grid so every (genotype, seed) cell is one batch item.
        # generate_batch draws item i's noise from seeds[i], so the same seed
        # value gives byte-identical starting noise across genotypes - which is
        # the property the whole decomposition rests on.
        jobs = [(g_i, s_i) for g_i in range(len(genotypes)) for s_i in range(len(seeds))]
        grid = np.zeros((len(genotypes), len(seeds), cfg.imgsz, cfg.imgsz, 3),
                        dtype=np.uint8)

        for start in range(0, len(jobs), cfg.batch_size):
            chunk = jobs[start:start + cfg.batch_size]
            snp_batch = torch.tensor(
                np.stack([snp_matrix[idx[g_i]] for g_i, _ in chunk]),
                dtype=torch.float32, device=device)
            batch_seeds = [seeds[s_i] for _, s_i in chunk]

            images = generate_batch(snp_encoder, unet, scheduler, decoder,
                                    snp_batch, batch_seeds, device, latent_shape,
                                    cfg.sampling_steps)
            for (g_i, s_i), img in zip(chunk, images):
                grid[g_i, s_i] = img
            print(f"  generated {min(start + len(chunk), len(jobs))}/{len(jobs)}")

        genotype_vals, noise_vals = decompose(grid)
        ratio = float(genotype_vals.mean() / noise_vals.mean()) if noise_vals.mean() else float('nan')

        results[label] = {
            'checkpoint': str(ckpt_path),
            'genotype_effect_mean': float(genotype_vals.mean()),
            'genotype_effect_std': float(genotype_vals.std()),
            'noise_effect_mean': float(noise_vals.mean()),
            'noise_effect_std': float(noise_vals.std()),
            'conditioning_ratio': ratio,
            'n_genotype_pairs': int(len(genotype_vals)),
            'n_noise_pairs': int(len(noise_vals)),
        }

        np.save(out / f'grid_{label.replace(" ", "_").replace("/", "-")}.npy', grid)
        save_grid_figure(
            grid, genotypes, seeds,
            f'{label}   -   down a column: same noise, different DNA',
            out / f'grid_{label.replace(" ", "_").replace("/", "-")}.png')

        print(f"\n  genotype effect (fixed noise):   "
              f"{genotype_vals.mean():7.3f} +/- {genotype_vals.std():.3f}")
        print(f"  noise effect (fixed genotype):   "
              f"{noise_vals.mean():7.3f} +/- {noise_vals.std():.3f}")
        print(f"  conditioning ratio:              {ratio:7.3f}")

    if not results:
        raise SystemExit(
            "No checkpoints could be loaded - check the paths in cfg.checkpoints.")

    # report
    df = pd.DataFrame(results).T
    df.index.name = 'checkpoint_label'
    df.to_csv(out / 'conditioning_strength.csv')
    with open(out / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    if len(results) >= 1:
        save_comparison_figure(results, out / 'conditioning_comparison.png')

    print(f"\n\nSUMMARY\n")
    print(f"{'checkpoint':<22}{'genotype':>11}{'noise':>10}{'ratio':>9}")
    for label, r in results.items():
        print(f"{label:<22}{r['genotype_effect_mean']:>11.3f}"
              f"{r['noise_effect_mean']:>10.3f}{r['conditioning_ratio']:>9.3f}")

    if len(results) == 2:
        labels = list(results)
        a, b = results[labels[0]]['conditioning_ratio'], results[labels[1]]['conditioning_ratio']
        if a and np.isfinite(a) and np.isfinite(b):
            change = (b - a) / a * 100
            print(f"\n  {labels[1]} vs {labels[0]}: {change:+.1f}% conditioning strength")
            if change > 20:
                print("  The genotype drives the image substantially more than before.")
            elif change > 5:
                print("  A modest increase. Worth confirming once epoch counts match,")
                print("  since a partly-trained model can still be gaining ground.")
            elif change > -5:
                print("  Essentially unchanged. The encoding fix corrected the geometry")
                print("  of the SNP space, but that has not yet translated into the")
                print("  conditioning having more influence on the output.")
            else:
                print("  Conditioning is weaker than the old checkpoint. If the new run")
                print("  has trained for fewer epochs, re-check once they match before")
                print("  reading this as a regression.")

    print(f"\nWrote conditioning_strength.csv, summary.json, "
          f"conditioning_comparison.png and per-checkpoint grids to {out}")


if __name__ == '__main__':
    main()
