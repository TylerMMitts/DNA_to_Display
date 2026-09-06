# Trains the genotype-conditioned diffusion model on one-hot SNP encoding.
#
# This is the current trainer. Weights go to models/diffusion_onehot/,
# previews and loss history to results/training/diffusion_onehot/. LiteVAE is
# loaded frozen, so a trained autoencoder has to exist first.

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    CROPPED_IMAGES_DIR, DIFFUSION_NUMERIC_MODEL, DIFFUSION_ONEHOT_DIR,
    IMAGE_METADATA, LITEVAE_MODEL, SNP_PARQUET, TRAINING_RESULTS_DIR,
    best_checkpoint_path, checkpoint_path, find_latest_checkpoint,
    resolve_input, resolve_output,
)

from latent_diffusion.models.ldm import LatentDiffusionModel
from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.models.snp_encoding import SNPProjector, OneHotSNPEncoder
from latent_diffusion.models.unet import DenoisingUNet
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from litevae.models import LiteVAEEncoder, LiteVAEDecoder
from latent_diffusion.generation.generate_from_dataset import (
    generate_batch, load_original, save_comparison,
)

# Every checkpoint this script writes is prefixed with this, so a stray .pt file
# still says which model produced it once it has been copied elsewhere.
MODEL_NAME = 'diffusion_onehot'


# Loads images and their genotype's precomputed projected coordinates.
class ProjectedRootDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'snp': torch.tensor(sample['projected'], dtype=torch.float32),
            'genotype': sample['genotype'],
        }

# Ensures that training sample has both image and snp data
def build_samples(metadata, image_dir, projected_by_genotype):
    samples, skipped_no_snp, skipped_no_file = [], 0, 0
    for _, row in metadata.iterrows():
        genotype = row['genotype']
        if genotype not in projected_by_genotype:
            skipped_no_snp += 1
            continue
        path = Path(image_dir) / row['new_filename']
        if not path.exists():
            skipped_no_file += 1
            continue
        samples.append({'image_path': path, 'genotype': genotype,
                        'projected': projected_by_genotype[genotype]})
    return samples, skipped_no_snp, skipped_no_file

# Data augmentation and preprocessing transforms
def make_transforms(image_size, augment):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    if not augment:
        return transforms.Compose(base)
    # Flips and mild colour jitter only. Rotation and translation would move a
    # root that already fills the frame partly out of it, which is the same
    # constraint the feature-segmentation training runs under.
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ] + base)

# Loads frozen LiteVAE encoder and decoder
def load_litevae(checkpoint_path, device, latent_channels=4):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = LiteVAEEncoder(in_channels=3, latent_channels=latent_channels,
                             feature_channels=64, num_blocks=3)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.to(device).eval()

    decoder = LiteVAEDecoder(latent_channels=latent_channels, output_channels=3,
                             base_channels=512, num_res_blocks=2)
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    decoder.to(device).eval()

    print(f"LiteVAE loaded (epoch {ckpt.get('epoch', '?')}), frozen")
    return encoder, decoder

# Warm-starts a UNet from a previous checkpoint, starting this one with fixed snp encoder
def warm_start_unet(unet, checkpoint_path, device):

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'unet_state_dict' not in ckpt:
        raise SystemExit(f"{checkpoint_path} has no 'unet_state_dict'")

    saved = ckpt['unet_state_dict']
    current = unet.state_dict()
    mismatched = [k for k in current
                  if k in saved and saved[k].shape != current[k].shape]
    missing = [k for k in current if k not in saved]

    if mismatched or missing:
        raise SystemExit(
            "UNet architecture does not match the warm-start checkpoint.\n"
            f"  shape mismatches: {mismatched[:5]}{'...' if len(mismatched) > 5 else ''}\n"
            f"  missing keys:     {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            "Set the UNet config to match the checkpoint, or set "
            "warm_start_checkpoint = None to train from scratch.")

    unet.load_state_dict(saved)
    print(f"UNet warm-started from {Path(checkpoint_path).name} "
          f"(epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', float('nan')):.4f})")
    return unet


# Finds the newest checkpoint to resume from, or None if the folder holds none.
# paths.find_latest_checkpoint raises when there is nothing to load, which is a
# normal first run here rather than an error.
def find_resumable_checkpoint(save_dir):
    try:
        return find_latest_checkpoint(save_dir, MODEL_NAME)
    except FileNotFoundError:
        return None


def main():

    class cfg:
        snp_parquet = SNP_PARQUET
        metadata_path = IMAGE_METADATA
        image_dir = CROPPED_IMAGES_DIR
        litevae_checkpoint = LITEVAE_MODEL

        # Weights go to models/diffusion_onehot/, named after this model.
        save_dir = DIFFUSION_ONEHOT_DIR
        # Previews, loss curves and the run summary are results, not weights.
        results_dir = TRAINING_RESULTS_DIR / MODEL_NAME
        save_every = 10
        val_fraction = 0.2

        # Saves sample images for preview during training
        save_previews = True
        n_preview_genotypes = 3
        preview_sampling_steps = 20
        preview_latent_size = 32

        # Resumes from the latest checkpoint in save_dir if one exists.
        resume = True
        warm_start_checkpoint = DIFFUSION_NUMERIC_MODEL

        # Auto finds the number of founders in snp data
        founders = None
        # Sets the target variance for PCA on the SNP data
        pca_target_variance = 0.95
        pca_random_state = 0

        # Must match the warm-start checkpoint's UNet, or loading will fail.
        latent_channels = 4
        base_channels = 128
        snp_embed_dim = 512
        d_attention = 512
        num_tokens = 8
        num_res_blocks = 2
        attention_resolutions = [1, 2, 4]
        encoder_hidden_dim = 1024

        # Diffusion / optimisation
        num_steps = 1000
        beta_start = 1e-4
        beta_end = 0.02
        # Lower than train.py's 1e-4: the UNet is already trained and only needs
        # to adapt to a new conditioning signal, so a large step risks
        # destroying what the warm start was meant to preserve.
        learning_rate = 3e-5
        # The freshly initialised SNP encoder does need to move faster than the
        # warm-started UNet, so it gets its own higher rate
        snp_encoder_learning_rate = 1e-4
        num_epochs = 150
        batch_size = 16
        num_workers = 4
        image_size = 256
        seed = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    save_dir = resolve_output(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = resolve_output(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nCheckpoints: {save_dir}\nResults: {results_dir}")

    resume_path = find_resumable_checkpoint(save_dir) if cfg.resume else None
    resume_ckpt = None
    start_epoch = 0
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = resume_ckpt['epoch']
        print(f"\nFound {resume_path.name} - resuming "
              f"({start_epoch} epochs already completed)")
    elif cfg.resume:
        print(f"\nresume=True but no {MODEL_NAME}_epoch_N.pt found in save_dir - "
              "starting fresh")
    else:
        print("\nresume=False - starting fresh")

    # SNP data and projection
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    if resume_ckpt is not None:
        # Reuses the exact basis already in use rather than refitting
        founders = tuple(resume_ckpt['founders'])
        projector = SNPProjector.from_state_dict(resume_ckpt['snp_projector'])
        print(f"Founders: {founders}")
        print(f"Restored SNP projector from checkpoint: "
              f"{projector.output_dim} dimensions")
    else:
        founders = cfg.founders
        if founders is None:
            founders = tuple(sorted(int(v) for v in np.unique(snp_matrix) if v > 0))
        print(f"Founders: {founders}")

        print("\nFitting one-hot PCA projection...")
        t0 = time.time()
        projector = SNPProjector(founders=founders,
                                 target_variance=cfg.pca_target_variance,
                                 random_state=cfg.pca_random_state).fit(snp_matrix)
        print(f"  projected to {projector.output_dim} dimensions "
              f"in {time.time() - t0:.1f}s")

    projected = projector.transform(snp_matrix)
    projected_by_genotype = {name: projected[i] for i, name in enumerate(sample_names)}

    # Image data and metadata
    image_dir = resolve_input(cfg.image_dir, 'image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))
    samples, no_snp, no_file = build_samples(metadata, image_dir, projected_by_genotype)
    print(f"\nImages: {len(samples)} usable "
          f"({no_snp} skipped for missing SNP data, {no_file} for missing files)")
    if not samples:
        raise SystemExit("no usable samples - check image_dir and the SNP parquet")

    # Split by GENOTYPE, not by image. Splitting by image would put replicate
    # photos of the same genotype on both sides, so the validation loss would
    # be reporting on genotypes the model had already been conditioned on.
    genotypes = sorted({s['genotype'] for s in samples})
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(genotypes)
    n_val = max(1, int(round(len(genotypes) * cfg.val_fraction)))
    val_genotypes = set(genotypes[:n_val])

    if resume_ckpt is not None:
        saved_val_genotypes = set(resume_ckpt['val_genotypes'])
        if saved_val_genotypes != val_genotypes:
            raise SystemExit(
                "Resuming, but the train/val genotype split does not match the "
                "checkpoint's saved split - image_dir or the metadata must have "
                "changed since that checkpoint was written. Fix the data "
                "mismatch, or set resume = False to start over.")

    train_samples = [s for s in samples if s['genotype'] not in val_genotypes]
    val_samples = [s for s in samples if s['genotype'] in val_genotypes]
    print(f"  train: {len(train_samples)} images / "
          f"{len(genotypes) - n_val} genotypes")
    print(f"  val:   {len(val_samples)} images / {n_val} genotypes")

    preview_pool = val_samples if val_samples else train_samples
    preview_by_genotype = {}
    for s in preview_pool:
        preview_by_genotype.setdefault(s['genotype'], s)
    preview_candidates = sorted(preview_by_genotype.keys())
    preview_rng = np.random.default_rng(cfg.seed + 1)
    preview_rng.shuffle(preview_candidates)
    preview_genotypes = preview_candidates[:cfg.n_preview_genotypes]
    preview_samples = [preview_by_genotype[g] for g in preview_genotypes]
    if cfg.save_previews and preview_samples:
        print(f"  preview genotypes ({len(preview_samples)}, from "
              f"{'val' if val_samples else 'train'}): {preview_genotypes}")
    elif cfg.save_previews:
        print("  save_previews=True but no genotypes available for previews - "
              "disabling")

    train_loader = DataLoader(
        ProjectedRootDataset(train_samples, make_transforms(cfg.image_size, True)),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        ProjectedRootDataset(val_samples, make_transforms(cfg.image_size, False)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True)

    if resume_ckpt is not None:
        encoder_config = resume_ckpt['snp_encoder_config']
        unet_config = resume_ckpt['unet_config']
    else:
        encoder_config = {
            'input_dim': projector.output_dim,
            'embedding_dim': cfg.snp_embed_dim,
            'num_tokens': cfg.num_tokens,
            'hidden_dim': cfg.encoder_hidden_dim,
        }
        unet_config = {
            'latent_channels': cfg.latent_channels, 'base_channels': cfg.base_channels,
            'snp_embed_dim': cfg.snp_embed_dim, 'd_attention': cfg.d_attention,
            'num_res_blocks': cfg.num_res_blocks,
            'attention_resolutions': cfg.attention_resolutions,
        }

    snp_encoder = OneHotSNPEncoder(**encoder_config).to(device)
    unet = DenoisingUNet(**unet_config).to(device)

    if resume_ckpt is not None:
        snp_encoder.load_state_dict(resume_ckpt['snp_encoder_state_dict'])
        unet.load_state_dict(resume_ckpt['unet_state_dict'])
        print("Restored SNP encoder and UNet weights from the resume checkpoint")
    elif cfg.warm_start_checkpoint:
        warm_start_unet(unet, resolve_input(cfg.warm_start_checkpoint,
                                            'warm-start checkpoint'), device)
    else:
        print("UNet initialised from scratch (no warm start)")

    # Keeps attention_history from growing unboundedly during training, each
    # cross-attention block appends to it on every forward pass otherwise.
    unet.set_store_attention(False)

    litevae_encoder, litevae_decoder = load_litevae(
        resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'), device,
        cfg.latent_channels)

    scheduler = DiffusionScheduler(num_steps=cfg.num_steps, beta_start=cfg.beta_start, beta_end=cfg.beta_end)
    # The scheduler holds plain tensors, not module buffers, so .to(device) on
    # the model does not move them; indexing them with a device-side timestep
    # would fail otherwise.
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    ldm = LatentDiffusionModel(
        litevae_encoder=litevae_encoder, litevae_decoder=litevae_decoder,
        snp_encoder=snp_encoder, unet=unet, scheduler=scheduler, device=device)

    trainable = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable:,}")

    optimizer = optim.Adam([
        {'params': ldm.unet.parameters(), 'lr': cfg.learning_rate},
        {'params': ldm.snp_encoder.parameters(), 'lr': cfg.snp_encoder_learning_rate},
    ])

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        print("Restored optimizer state")

    # T_max is the epochs remaining, not the run's original total
    # This ensures the cosine annealing schedule continues smoothly from the current epoch
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.num_epochs - start_epoch, 1), eta_min=1e-7)

    if start_epoch >= cfg.num_epochs:
        raise SystemExit(
            f"Checkpoint already has {start_epoch} epochs completed, which is "
            f">= num_epochs={cfg.num_epochs}. Raise num_epochs to continue "
            f"training, or lower it if this run is meant to stop here.")

    best_val = float('inf')
    best_path = best_checkpoint_path(save_dir, MODEL_NAME)
    if best_path.exists():
        try:
            best_val = torch.load(best_path, map_location='cpu',
                                  weights_only=False)['val_loss']
        except Exception:
            pass
    elif resume_ckpt is not None:
        best_val = resume_ckpt.get('val_loss', float('inf'))

    history = []
    history_path = results_dir / 'training_history.csv'
    if resume_ckpt is not None and history_path.exists():
        history = pd.read_csv(history_path).to_dict('records')

    if resume_ckpt is not None:
        print(f"Resuming at epoch {start_epoch + 1}/{cfg.num_epochs}, "
              f"best validation loss so far {best_val:.4f}")

    # Trainable parts in train mode, LiteVAE always in eval.
    # ldm.train() would otherwise flip the frozen LiteVAE into training mode too.
    # Its parameters have requires_grad=False so they cannot be updated, but
    # train mode still changes dropout and normalisation behaviour, which would
    # make the encoder produce slightly different latents than it does at
    # inference - shifting the very target the diffusion model is trying to learn.
    def set_train_mode():
        ldm.train()
        if ldm.litevae_encoder is not None:
            ldm.litevae_encoder.eval()
        if ldm.litevae_decoder is not None:
            ldm.litevae_decoder.eval()

    preview_dir = results_dir / 'previews'
    preview_latent_shape = (cfg.latent_channels, cfg.preview_latent_size,
                            cfg.preview_latent_size)
    # One fixed seed per genotype, reused at every preview call rather than drawn fresh each epoch
    preview_seeds = [cfg.seed + i for i in range(len(preview_samples))]

    def save_previews(epoch_num):
        ldm.eval()
        preview_dir.mkdir(parents=True, exist_ok=True)
        snp_batch = torch.tensor(np.stack([s['projected'] for s in preview_samples]),
                                 dtype=torch.float32, device=device)
        generated = generate_batch(snp_encoder, unet, scheduler, litevae_decoder,
                                   snp_batch, preview_seeds, device,
                                   preview_latent_shape, cfg.preview_sampling_steps)
        for s, gen_img in zip(preview_samples, generated):
            original = load_original(s['image_path'], cfg.image_size)
            save_comparison(original, gen_img, s['genotype'],
                            preview_dir / f"{s['genotype']}_epoch{epoch_num:04d}.png")
        set_train_mode()
        print(f"  wrote {len(preview_samples)} preview images to {preview_dir}")

    for epoch in range(start_epoch, cfg.num_epochs):
        set_train_mode()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}"):
            images = batch['image'].to(device, non_blocking=True)
            snp = batch['snp'].to(device, non_blocking=True)

            loss = ldm(images, snp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ldm.unet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(ldm.snp_encoder.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        train_loss = total / max(len(train_loader), 1)

        ldm.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                snp = batch['snp'].to(device, non_blocking=True)
                val_total += ldm(images, snp).item()
        val_loss = val_total / max(len(val_loader), 1)

        lr_scheduler.step()
        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_loss': val_loss,
                        'lr': lr_scheduler.get_last_lr()[0]})
        print(f"  train {train_loss:.4f}   val {val_loss:.4f}   "
              f"lr {lr_scheduler.get_last_lr()[0]:.2e}")

        def checkpoint(path, note):
            torch.save({
                'epoch': epoch + 1,
                'note': note,
                'unet_state_dict': unet.state_dict(),
                'snp_encoder_state_dict': snp_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                # Saved so analysis never has to refit and guess the basis.
                'snp_projector': projector.state_dict(),
                'snp_encoder_config': encoder_config,
                'unet_config': unet_config,
                'encoding': 'one_hot_founders',
                'founders': list(founders),
                'val_genotypes': sorted(val_genotypes),
            }, path)

        if (epoch + 1) % cfg.save_every == 0:
            checkpoint(checkpoint_path(save_dir, epoch + 1, MODEL_NAME), 'periodic')
            if cfg.save_previews and preview_samples:
                save_previews(epoch + 1)
        if val_loss < best_val:
            best_val = val_loss
            checkpoint(best_path, 'best validation loss')

        pd.DataFrame(history).to_csv(history_path, index=False)

    with open(results_dir / 'run_summary.json', 'w') as f:
        json.dump({
            'encoding': 'one_hot_founders', 'founders': list(founders),
            'projected_dim': int(projector.output_dim),
            'warm_start': str(cfg.warm_start_checkpoint),
            'resumed_from': str(resume_path) if resume_path else None,
            'started_at_epoch': start_epoch,
            'image_dir': str(image_dir),
            'epochs': cfg.num_epochs,
            'best_val_loss': best_val,
            'final': history[-1] if history else None,
            'preview_genotypes': preview_genotypes if cfg.save_previews else None,
        }, f, indent=2, default=float)

    print(f"\nTraining complete. Best validation loss {best_val:.4f}")
    print(f"Checkpoints in {save_dir}")
    print(f"training_history.csv and run_summary.json in {results_dir}")
    if cfg.save_previews and preview_samples:
        print(f"Preview images ({', '.join(preview_genotypes)}) in {preview_dir}")


if __name__ == '__main__':
    main()
