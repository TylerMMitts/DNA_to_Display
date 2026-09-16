# Generates several images for every root whose genotype the model never trained on.
#
# Each real held-out image is saved beside one generated image per noise seed, so
# what the seeds share shows what the model predicts for that genotype and what
# differs between them is only noise. Held out means held out of this checkpoint's
# own training, read from the split it saved.
#
# Writes one <image>.png per real image, the raw generated images in generated/,
# held_out_images.csv and summary.json.

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    CROPPED_IMAGES_DIR, DIFFUSION_ONEHOT_MODEL, IMAGE_METADATA, LITEVAE_MODEL,
    RESULTS_DIR, SNP_PARQUET, apply_overrides, pick_device, resolve_input,
    resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import (
    generate_batch, load_original, save_multi_comparison,
)
from latent_diffusion.training.train_onehot import load_litevae


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/generation/generate_held_out_seeds.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        litevae_checkpoint = LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        metadata_path = IMAGE_METADATA
        image_dir = CROPPED_IMAGES_DIR
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'
        output_dir = RESULTS_DIR / 'held_out_seeds'

        # The same seeds for every genotype, so a column of images across
        # genotypes starts from identical noise and differs only by genotype.
        seeds = [0, 1, 2, 3, 4, 5, 6]

        # None -> every held-out genotype. A small number is a quick check that
        # runs the same code on fewer genotypes.
        max_genotypes = None

        # Resume an interrupted run: a genotype whose images all exist is skipped.
        skip_existing = True

        sampling_steps = 50
        imgsz = 256
        latent_size = 32
        device = pick_device()

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    generated_dir = out / 'generated'
    generated_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_input(cfg.checkpoint, 'diffusion checkpoint')
    print(f"Device: {device}\nCheckpoint: {checkpoint_path}\nOutput: {out}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'val_genotypes' not in ckpt:
        raise SystemExit(f"{checkpoint_path.name} records no validation split, so "
                         "there is no way to tell which genotypes it never trained on")
    held_out = sorted(ckpt['val_genotypes'])
    epoch = ckpt.get('epoch')
    del ckpt

    names, _, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    row_of = {n: i for i, n in enumerate(names)}

    image_dir = resolve_input(cfg.image_dir, 'image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))
    images_of = {}
    for r in metadata.itertuples():
        if r.genotype in held_out and r.genotype in row_of \
                and (image_dir / r.new_filename).exists():
            images_of.setdefault(r.genotype, []).append(image_dir / r.new_filename)

    genotypes = [g for g in held_out if g in images_of]
    if cfg.max_genotypes is not None:
        genotypes = genotypes[:cfg.max_genotypes]
    if not genotypes:
        raise SystemExit("no held-out genotype has both SNP data and an image file")
    n_images = sum(len(images_of[g]) for g in genotypes)
    print(f"\nHeld-out genotypes in the checkpoint: {len(held_out)}")
    print(f"  with SNP data and images: {len(genotypes)}  ({n_images} images)")
    print(f"  seeds per genotype: {len(cfg.seeds)}")

    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    unet.set_store_attention(False)
    _, decoder = load_litevae(resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                              device, unet_cfg['latent_channels'])
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    scheduler = DiffusionScheduler()
    # Plain tensors rather than module buffers, so they are moved by hand.
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    rows, n_skipped = [], 0
    start = time.time()
    for k, genotype in enumerate(genotypes, start=1):
        comparisons = [out / f'{p.stem}.png' for p in images_of[genotype]]
        raw_paths = [generated_dir / f'{genotype}_seed{s}.png' for s in cfg.seeds]

        # Generation depends only on genotype and seed, so each genotype is
        # sampled once and the same images sit beside each of its real roots.
        if cfg.skip_existing and all(p.exists() for p in comparisons + raw_paths):
            n_skipped += 1
            generated = None
        else:
            snp_batch = torch.tensor(np.repeat(snp_matrix[row_of[genotype]][None],
                                               len(cfg.seeds), axis=0),
                                     dtype=torch.float32, device=device)
            generated = generate_batch(snp_encoder, unet, scheduler, decoder, snp_batch,
                                       cfg.seeds, device, latent_shape, cfg.sampling_steps)
            for image, path in zip(generated, raw_paths):
                Image.fromarray(image).save(path)
            for real_path, comparison in zip(images_of[genotype], comparisons):
                save_multi_comparison(load_original(real_path, cfg.imgsz), list(generated),
                                      genotype, cfg.seeds, comparison)

        for real_path, comparison in zip(images_of[genotype], comparisons):
            rows.append({'genotype': genotype, 'real_image': real_path.name,
                         'comparison': comparison.name})

        elapsed = time.time() - start
        print(f"  {k}/{len(genotypes)} {genotype}: {len(comparisons)} images"
              f"{'  (already done)' if generated is None else ''}  "
              f"elapsed {elapsed / 60:.1f} min")

    pd.DataFrame(rows).to_csv(out / 'held_out_images.csv', index=False)
    summary = {
        'checkpoint': str(checkpoint_path),
        'epoch': epoch,
        'n_held_out_genotypes_in_checkpoint': len(held_out),
        'n_genotypes': len(genotypes),
        'n_images': n_images,
        'n_genotypes_already_done': n_skipped,
        'seeds': list(cfg.seeds),
        'sampling_steps': cfg.sampling_steps,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {n_images} comparisons, {len(genotypes) * len(cfg.seeds)} generated "
          f"images, held_out_images.csv and summary.json to {out}")


if __name__ == '__main__':
    main()
