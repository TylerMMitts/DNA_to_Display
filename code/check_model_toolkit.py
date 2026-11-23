# Runs every part of model_toolkit.py once, on real checkpoints, and times it.
#
# The check to run before relying on the toolkit live: if this finishes, loading,
# the SNP encoder, genotype editing, generation, trait measurement and denoising
# scores all work on this machine, and the timings say how long each kind of
# request will take. It doubles as a worked example of the toolkit.
#
# Writes model_comparison.png, encoder_checks.csv, traits.csv, denoise_loss.csv
# and timings.csv to results/playground/toolkit_check/.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import apply_overrides

from model_toolkit import load_session, measure, output_dir, save_grid


def main(overrides=None):
    # Edit these values, then run:
    #     uv run code/check_model_toolkit.py
    class cfg:
        # Epochs of the medium model to load side by side.
        epochs = [300, 500]
        # None -> the first held-out genotype that has images.
        genotype = None
        seeds = [0, 1, 2]
        sampling_steps = 50
        # Other held-out genotypes scored against the genotype's own images.
        n_wrong_genotypes = 3
        output_name = 'toolkit_check'

    apply_overrides(cfg, overrides)
    out = output_dir(cfg.output_name)
    timings = []

    def timed(label, fn):
        start = time.time()
        value = fn()
        timings.append({'step': label, 'seconds': round(time.time() - start, 2)})
        print(f"  {label}: {timings[-1]['seconds']:.1f}s")
        return value

    print("Loading")
    sessions = [timed(f'load epoch {e}', lambda e=e: load_session(e)) for e in cfg.epochs]
    first = sessions[0]
    held_out = [g for g in first.genotypes_in('held out') if first.image_paths(g)]
    genotype = cfg.genotype or held_out[0]
    wrong = [g for g in held_out if g != genotype][:cfg.n_wrong_genotypes]
    print(f"\nGenotype: {genotype} ({first.group(genotype)}, "
          f"{len(first.image_paths(genotype))} images)")

    print("\nSNP encoder")
    tokens = timed('embed all 200 genotypes', lambda: first.embed(first.matrix))
    print(f"  tokens {tokens.shape}, finite: {np.isfinite(tokens).all()}")
    neighbours = timed('nearest genotypes', lambda: first.nearest(genotype, k=5))
    print(neighbours.to_string(index=False))

    print("\nEditing genotypes")
    founder_3 = first.founder_line(3)
    swapped = timed('swap 5000 loci to founder 3', lambda: first.swap(genotype, slice(0, 5000), 3))
    cross = timed('recombinant of two held-out genotypes',
                  lambda: first.recombinant([genotype, wrong[0]], n_blocks=20))
    rows = []
    for label, codes in (('founder 3 line', founder_3), (f'{genotype} with 5000 loci -> founder 3', swapped),
                         (f'{genotype} x {wrong[0]}', cross)):
        rows.append({'edit': label, **first.distance(genotype, codes)})
    edits = pd.DataFrame(rows)
    print(edits.to_string(index=False))
    edits.to_csv(out / 'encoder_checks.csv', index=False)

    print("\nGenerating")
    real = first.real_images(genotype, n=len(cfg.seeds))
    grid, labels = [list(real)], [f'{genotype} real replicates']
    generated = {}
    for s in sessions:
        images = timed(f'generate {len(cfg.seeds)} images, {s.name}',
                       lambda s=s: s.generate(genotype, seeds=cfg.seeds, steps=cfg.sampling_steps))
        generated[s.name] = images[0]
        grid.append(list(images[0]))
        labels.append(f"{s.name}, seeds {', '.join(map(str, cfg.seeds))}")
    save_grid(grid, out / 'model_comparison.png', row_labels=labels)

    print("\nMeasuring traits")
    frames = [timed('measure real images', lambda: measure(real)).assign(source='real')]
    for name, images in generated.items():
        frames.append(timed(f'measure {name}', lambda images=images: measure(images)).assign(source=name))
    traits = pd.concat(frames, ignore_index=True)
    traits.to_csv(out / 'traits.csv', index=False)
    print(traits.groupby('source').mean().round(2).to_string())

    print("\nDenoising loss: own genotype vs other held-out genotypes, on its real images")
    losses = []
    for s in sessions:
        table = timed(f'denoise loss, {s.name}',
                      lambda s=s: s.denoise_loss(s.real_images(genotype), [genotype] + wrong))
        mean = table.mean()
        print(f"  {s.name}: own {mean[genotype]:.4f}   others {mean[wrong].mean():.4f}   "
              f"right-genotype gain {mean[wrong].mean() / mean[genotype] - 1:+.2%}")
        losses.append(table.assign(model=s.name))
    pd.concat(losses).to_csv(out / 'denoise_loss.csv')

    pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
    print(f"\nWrote model_comparison.png, encoder_checks.csv, traits.csv, denoise_loss.csv "
          f"and timings.csv to {out}")


if __name__ == '__main__':
    main()
