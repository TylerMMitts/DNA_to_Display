# Generates one image per real image and saves the two side by side.
#
# The main way to see what the model produces for genotypes it was trained
# on. Safe to interrupt and rerun - it skips images that already exist.

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_NUMERIC_DIR, IMAGES_DIR, IMAGE_METADATA, LITEVAE_MODEL,
    RESULTS_DIR, SNP_PARQUET, find_latest_checkpoint, resolve_input,
    resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model


# Generation.

@torch.no_grad()
def generate_batch(snp_encoder, unet, scheduler, decoder, snp_batch, seeds,
                   device, latent_shape, num_steps):
    # Full DDIM sampling with one independent noise draw per item.
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
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


def load_original(path, imgsz):
    return np.array(Image.open(path).convert('RGB').resize((imgsz, imgsz), Image.LANCZOS))


def save_comparison(original_rgb, generated_rgb, label, save_path, divider=4):
    # Original on top, generated on bottom, with a thin divider and a small
    # corner label naming the genotype - so files stay identifiable once there
    # are a thousand of them sitting in one folder.
    h, w = original_rgb.shape[:2]
    canvas = np.zeros((h * 2 + divider, w, 3), dtype=np.uint8)
    canvas[:h] = original_rgb
    canvas[h:h + divider] = 40                    # dark divider line
    canvas[h + divider:] = generated_rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for y, text in ((4, f'{label}  (original)'), (h + divider + 4, f'{label}  (generated)')):
        draw.rectangle([2, y - 2, 6 + 7 * len(text), y + 12], fill=(0, 0, 0))
        draw.text((4, y), text, fill=(255, 255, 255))
    img.save(save_path)


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/generation/generate_from_dataset.py
    class cfg:
        checkpoint_dir = DIFFUSION_NUMERIC_DIR   # newest checkpoint in here is used
        litevae_checkpoint = LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        metadata_path = IMAGE_METADATA
        image_dir = IMAGES_DIR
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'

        output_dir = RESULTS_DIR / 'diffusion_results'

        # None -> every image in the metadata with a matching genotype and file.
        # Set to a small number for a first sanity-check run before committing
        # to the full dataset.
        max_images = None

        skip_existing = True     # resume a partially-completed run without redoing work

        batch_size = 8            # images per DDIM sampling batch
        sampling_steps = 50
        seed = 0                  # base seed; each image gets seed + its row index
        imgsz = 256
        latent_size = 32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(cfg.device)
    out_root = resolve_output(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out_root}")
    if device.type == 'cpu':
        print("WARNING: running on CPU. Full-dataset DDIM sampling here is slow - "
              "consider setting max_images for a first pass, or using a GPU.")

    # data
    image_dir = resolve_input(cfg.image_dir, 'image directory')
    metadata_path = resolve_input(cfg.metadata_path, 'image metadata')
    metadata = pd.read_csv(metadata_path)

    parquet_path = resolve_input(cfg.snp_parquet, 'SNP parquet')
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(parquet_path)
    snp_matrix = np.asarray(snp_matrix)
    name_to_row = {n: i for i, n in enumerate(sample_names)}

    # model
    checkpoint_dir = resolve_input(cfg.checkpoint_dir, 'checkpoint directory')
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    from latent_diffusion.diffusion.scheduler import DiffusionScheduler
    from litevae.models import LiteVAEDecoder

    scheduler = DiffusionScheduler()
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    vae_path = resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint')
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    decoder = LiteVAEDecoder(latent_channels=unet_cfg['latent_channels'],
                             output_channels=3, base_channels=512, num_res_blocks=2)
    decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
    decoder.to(device).eval()

    # build the work list
    rows = []
    n_no_snp, n_no_file = 0, 0
    for idx, row in metadata.iterrows():
        genotype = row['genotype']
        image_path = image_dir / row['new_filename']

        if genotype not in name_to_row:
            n_no_snp += 1
            continue
        if not image_path.exists():
            n_no_file += 1
            continue

        rows.append({'row_index': idx, 'genotype': genotype,
                    'filename': row['new_filename'], 'image_path': image_path})

    print(f"\nMetadata rows: {len(metadata)}")
    print(f"  skipped (genotype has no SNP data): {n_no_snp}")
    print(f"  skipped (image file not found):     {n_no_file}")
    print(f"  eligible: {len(rows)}")

    if cfg.max_images is not None:
        rows = rows[:cfg.max_images]
        print(f"  limited to max_images: {len(rows)}")

    # resume support
    to_process = []
    n_skipped_existing = 0
    for r in rows:
        out_path = out_root / f"{Path(r['filename']).stem}.png"
        if cfg.skip_existing and out_path.exists():
            n_skipped_existing += 1
            continue
        r['out_path'] = out_path
        to_process.append(r)

    if n_skipped_existing:
        print(f"  already generated, skipping: {n_skipped_existing}")
    print(f"  will generate: {len(to_process)}")

    if not to_process:
        print("\nNothing to do.")
        return

    # generate
    log_rows = []
    start_time = time.time()

    for batch_start in range(0, len(to_process), cfg.batch_size):
        batch = to_process[batch_start:batch_start + cfg.batch_size]

        snp_batch = torch.tensor(
            np.stack([snp_matrix[name_to_row[r['genotype']]] for r in batch]),
            dtype=torch.float32, device=device)
        seeds = [cfg.seed + r['row_index'] for r in batch]

        generated = generate_batch(snp_encoder, unet, scheduler, decoder,
                                   snp_batch, seeds, device, latent_shape,
                                   cfg.sampling_steps)

        for r, gen_img in zip(batch, generated):
            try:
                original = load_original(r['image_path'], cfg.imgsz)
            except Exception as exc:
                log_rows.append({'filename': r['filename'], 'genotype': r['genotype'],
                                 'status': f'failed to load original: {exc}'})
                continue

            save_comparison(original, gen_img, r['genotype'], r['out_path'])
            log_rows.append({'filename': r['filename'], 'genotype': r['genotype'],
                             'status': 'generated', 'output': str(r['out_path'])})

        done = batch_start + len(batch)
        elapsed = time.time() - start_time
        rate = elapsed / done
        remaining = rate * (len(to_process) - done)
        print(f"  {done}/{len(to_process)}  "
              f"elapsed {elapsed/60:.1f}m  est. remaining {remaining/60:.1f}m")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(out_root / 'generation_log.csv', index=False)

    summary = {
        'checkpoint': str(checkpoint_path),
        'total_metadata_rows': len(metadata),
        'skipped_no_snp_data': n_no_snp,
        'skipped_no_image_file': n_no_file,
        'skipped_already_generated': n_skipped_existing,
        'generated': int((log_df['status'] == 'generated').sum()) if len(log_df) else 0,
        'failed': int((log_df['status'] != 'generated').sum()) if len(log_df) else 0,
        'sampling_steps': cfg.sampling_steps,
    }
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {summary['generated']} generated, {summary['failed']} failed.")
    print(f"Wrote generation_log.csv and summary.json to {out_root}")


if __name__ == '__main__':
    main()
