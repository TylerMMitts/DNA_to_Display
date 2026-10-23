# Trains the LiteVAE autoencoder on the WiDiv images, at 512 px.
#
# Forked from litevae/train_litevae_seeds.py rather than sharing it, like every
# other dataset's trainer. What differs:
#   - 512 px in, 4 x 64 x 64 latent out, through ScaleFreeLiteVAEEncoder; the
#     root encoder is fixed at 256 px, too coarse for cortical cells
#   - trains only on lines that are training lines under BOTH delivered splits
#     (637 lines, 2,536 images), so no test line of either split is ever
#     reconstructed during training and one autoencoder serves both diffusion runs
#   - quarter turns and both flips, no colour jitter (see widiv_data.py)
#   - a linear output layer instead of the decoder's tanh, background pixels
#     weighted down in the loss, and a short learning-rate warm-up. The first
#     run without these collapsed: 77% of each image is white background, the
#     decoder saturated its tanh chasing it, and from epoch 2 it drew every
#     image all white with a latent that carried nothing (see widiv_litevae.py)
#
# Run prepare_widiv_images.py first. Weights go to models/litevae_widiv_v2/,
# reconstructions and loss curves to results/training/litevae_widiv_v2/. Resumes
# from the newest checkpoint if one exists.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

# Puts code/ and code/widiv/ on the import path so this runs directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (pick_device, TRAINING_RESULTS_DIR, best_checkpoint_path, checkpoint_path,
                   find_latest_checkpoint, resolve_input, resolve_output)
from widiv_paths import (WIDIV_IMAGE_SIZE, WIDIV_LITEVAE_DIR, WIDIV_SCALED_DIR,
                         WIDIV_SCALED_METADATA, WIDIV_SPLIT)
from widiv_data import SPLITS, load_split, make_transforms
from widiv_litevae import LinearOutputLiteVAEDecoder, ScaleFreeLiteVAEEncoder

# Every checkpoint this script writes is prefixed with this, so a stray .pt file
# still says which model produced it once it has been copied elsewhere.
MODEL_NAME = WIDIV_LITEVAE_DIR.name          # litevae_widiv_v2


class WidivImageDataset(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# The whole root disc: every pixel no farther from the centre than the outermost
# tissue pixel, plus a small margin so the edge always counts in full.
# prepare_widiv_images.py centres every root, so a disc covers it, including the
# white vessel lumens and aerenchyma gaps inside it that a "not white" test would miss.
def foreground_mask(images, margin=3):
    B, _, H, W = images.shape
    ys = torch.arange(H, device=images.device, dtype=torch.float32) - (H - 1) / 2
    xs = torch.arange(W, device=images.device, dtype=torch.float32) - (W - 1) / 2
    dist = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)               # H, W
    tissue = images.min(dim=1).values < 0.96                             # white is +1
    radius = torch.where(tissue, dist, torch.zeros_like(dist)).flatten(1).amax(1)
    return (dist[None] <= radius[:, None, None] + margin).float()[:, None]


class LiteVAELoss(nn.Module):

    def __init__(self, recon_weight=1.0, kl_weight=0.001, background_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.background_weight = background_weight

    # Squared error with background pixels counted at background_weight of a
    # root pixel. Unweighted, three quarters of the loss is white paper, which
    # an all-white output gets right; weighted, the tissue decides it.
    def forward(self, recon, original, z_mean, z_logvar):
        fg = foreground_mask(original)
        weight = fg + self.background_weight * (1 - fg)
        err = (recon - original) ** 2
        recon_loss = (err * weight).sum() / (weight.sum() * original.shape[1])
        # KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var)), averaged over
        # elements so the weighting does not depend on batch size.
        kl_loss = -0.5 * torch.sum(
            1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / z_mean.numel()
        # Plain MSE as well, comparable with the first run and other autoencoders.
        return (self.recon_weight * recon_loss + self.kl_weight * kl_loss,
                recon_loss, kl_loss, err.mean())


# Finds the newest checkpoint to resume from, or None if the folder holds none.
def find_resumable_checkpoint(save_dir):
    try:
        return find_latest_checkpoint(save_dir, MODEL_NAME)
    except FileNotFoundError:
        return None


def main():
    # Edit these values, then run:
    #     python code/widiv/train_litevae_widiv.py
    # On Hellbender: sbatch code/widiv/train_litevae_widiv_hellbender.sbatch
    class cfg:
        image_dir = WIDIV_SCALED_DIR
        metadata_path = WIDIV_SCALED_METADATA
        split_path = WIDIV_SPLIT
        save_dir = WIDIV_LITEVAE_DIR
        results_dir = TRAINING_RESULTS_DIR / MODEL_NAME

        # Architecture, the same as the root and seed autoencoders. At 512 px in
        # these give a 4 x 64 x 64 latent, which train_widiv.py assumes.
        latent_channels = 4
        feature_channels = 64
        base_channels = 512
        num_blocks = 3
        num_res_blocks = 2

        num_epochs = 250
        # Half the seed autoencoder's 16: each image is four times the pixels.
        batch_size = 8
        learning_rate = 1e-4
        recon_weight = 1.0
        kl_weight = 0.001
        # Weight of a background pixel in the reconstruction loss, relative to
        # a root pixel. Background is about 77% of each image.
        background_weight = 0.1
        # Optimiser steps over which the learning rate rises from 0 to
        # learning_rate (about 2 epochs), so the first steps cannot throw the
        # outputs to an extreme before the network has learned anything.
        warmup_steps = 500
        val_split = 0.1
        num_workers = 8
        image_size = WIDIV_IMAGE_SIZE
        seed = 0

        save_interval = 5
        resume = True
        device = pick_device()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    image_dir = resolve_input(cfg.image_dir, 'scaled WiDiv image directory')
    save_dir = resolve_output(cfg.save_dir)
    results_dir = resolve_output(cfg.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'reconstructions').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nImages: {image_dir}")
    print(f"Checkpoints: {save_dir}\nResults: {results_dir}")

    # Only lines that are training lines under both splits.
    split = load_split(resolve_input(cfg.split_path, 'WiDiv train/test split'))
    train_both = set(split.index[np.logical_and.reduce([split[s] == 'train' for s in SPLITS])])
    meta = pd.read_csv(resolve_input(cfg.metadata_path, 'scaled WiDiv metadata '
                                     '(run prepare_widiv_images.py)'))
    meta = meta[meta.genotype.isin(train_both)]
    paths = sorted(image_dir / f for f in meta.scaled_filename if (image_dir / f).exists())
    if not paths:
        raise SystemExit(f"no images found in {image_dir} - run "
                         "code/widiv/prepare_widiv_images.py first")
    with Image.open(paths[0]) as im:
        if im.size != (cfg.image_size, cfg.image_size):
            raise SystemExit(f"{paths[0].name} is {im.size}, expected {cfg.image_size} px - "
                             "the scaled images were written for a different size")

    # Split by line, so a line's four blocks never sit on both sides.
    lines = sorted({p.stem.split('_')[0] for p in paths})
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(lines)
    val_lines = set(lines[:max(1, int(round(cfg.val_split * len(lines))))])
    val_paths = [p for p in paths if p.stem.split('_')[0] in val_lines]
    train_paths = [p for p in paths if p.stem.split('_')[0] not in val_lines]

    train_loader = DataLoader(
        WidivImageDataset(train_paths, make_transforms(True)),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(
        WidivImageDataset(val_paths, make_transforms(False)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())
    print(f"Images from {len(lines)} lines training under both splits: {len(paths)} total, "
          f"{len(train_paths)} train, {len(val_paths)} val, {len(train_loader)} batches per epoch")

    model_config = {
        'latent_channels': cfg.latent_channels,
        'feature_channels': cfg.feature_channels,
        'base_channels': cfg.base_channels,
        'num_blocks': cfg.num_blocks,
        'num_res_blocks': cfg.num_res_blocks,
        'learning_rate': cfg.learning_rate,
        # Read by load_widiv_litevae to build the matching decoder.
        'output': 'linear',
        'background_weight': cfg.background_weight,
    }

    resume_path = find_resumable_checkpoint(save_dir) if cfg.resume else None
    resume_ckpt = None
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        saved = {k: v for k, v in resume_ckpt['config'].items() if k != 'learning_rate'}
        wanted = {k: v for k, v in model_config.items() if k != 'learning_rate'}
        if saved != wanted or resume_ckpt.get('image_size') != cfg.image_size:
            raise SystemExit(
                f"{resume_path.name} was trained with a different architecture or size.\n"
                f"  checkpoint: {saved}, {resume_ckpt.get('image_size')} px\n"
                f"  config:     {wanted}, {cfg.image_size} px\n"
                "Restore the settings, or move that folder's checkpoints aside.")

    encoder = ScaleFreeLiteVAEEncoder(in_channels=3, latent_channels=cfg.latent_channels,
                                      feature_channels=cfg.feature_channels,
                                      num_blocks=cfg.num_blocks).to(device)
    decoder = LinearOutputLiteVAEDecoder(latent_channels=cfg.latent_channels, output_channels=3,
                                         base_channels=cfg.base_channels,
                                         num_res_blocks=cfg.num_res_blocks).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)
    criterion = LiteVAELoss(recon_weight=cfg.recon_weight, kl_weight=cfg.kl_weight,
                            background_weight=cfg.background_weight)

    # The loss of an output that is white everywhere, on the validation images:
    # a run whose plain MSE stays at this has collapsed the way the first did.
    white_total, white_n = 0.0, 0
    for images in val_loader:
        white_total += float(((images - 1) ** 2).sum())
        white_n += images.numel()
    white_mse = white_total / white_n
    print(f"All-white output would score plain MSE {white_mse:.4f} on validation")

    start_epoch = 1
    best_val = float('inf')
    history = []
    step = 0

    if resume_ckpt is not None:
        encoder.load_state_dict(resume_ckpt['encoder_state_dict'])
        decoder.load_state_dict(resume_ckpt['decoder_state_dict'])
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        start_epoch = resume_ckpt['epoch'] + 1
        step = resume_ckpt.get('step', 0)
        best_val = resume_ckpt.get('val_loss', float('inf'))
        history = [r for r in resume_ckpt.get('history', [])
                   if r['epoch'] < start_epoch]
        print(f"\nResuming from {resume_path.name} at epoch {start_epoch}, "
              f"best validation loss {best_val:.4f}")
    else:
        n_enc = sum(p.numel() for p in encoder.parameters())
        n_dec = sum(p.numel() for p in decoder.parameters())
        print(f"\nParameters: {n_enc + n_dec:,} (encoder {n_enc:,}, decoder {n_dec:,})")

    if start_epoch > cfg.num_epochs:
        raise SystemExit(
            f"Checkpoint already has {start_epoch - 1} epochs completed, which is "
            f">= num_epochs={cfg.num_epochs}. Raise num_epochs to continue.")

    best_path = best_checkpoint_path(save_dir, MODEL_NAME)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        encoder.train()
        decoder.train()
        totals = np.zeros(4)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for images in progress:
            images = images.to(device, non_blocking=True)
            # Linear warm-up; afterwards the plateau scheduler alone sets the rate.
            if step < cfg.warmup_steps:
                for group in optimizer.param_groups:
                    group['lr'] = cfg.learning_rate * (step + 1) / cfg.warmup_steps
            z, z_mean, z_logvar = encoder(images)
            recon = decoder(z, save_steps=False)
            loss, recon_loss, kl_loss, plain_mse = criterion(recon, images, z_mean, z_logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            step += 1

            totals += [loss.item(), recon_loss.item(), kl_loss.item(), plain_mse.item()]
            progress.set_postfix({'loss': f'{loss.item():.4f}',
                                  'recon': f'{recon_loss.item():.4f}'})

        train_loss, train_recon, train_kl, train_mse = totals / max(len(train_loader), 1)

        encoder.eval()
        decoder.eval()
        val_total, val_mse = 0.0, 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device, non_blocking=True)
                z, z_mean, z_logvar = encoder(images)
                recon = decoder(z, save_steps=False)
                out = criterion(recon, images, z_mean, z_logvar)
                val_total += out[0].item()
                val_mse += out[3].item()
        val_loss = val_total / max(len(val_loader), 1)
        val_mse /= max(len(val_loader), 1)

        if step >= cfg.warmup_steps:
            scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'recon_loss': train_recon, 'kl_loss': train_kl,
                        'train_plain_mse': train_mse,
                        'val_loss': val_loss, 'val_plain_mse': val_mse,
                        'all_white_mse': white_mse,
                        'lr': optimizer.param_groups[0]['lr']})
        print(f"  train {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.6f})"
              f"   val {val_loss:.4f}   val plain MSE {val_mse:.4f} (all white {white_mse:.4f})"
              f"   lr {optimizer.param_groups[0]['lr']:.2e}")
        if epoch >= start_epoch + 2 and val_mse > 0.95 * white_mse:
            print("  WARNING: validation MSE is still within 5% of an all-white output - "
                  "the decoder may have collapsed to drawing the background. Check the "
                  "reconstructions before training further.")

        def save_checkpoint(path):
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,
                'config': model_config,
                'step': step,
                'image_size': cfg.image_size,
                'dataset': 'widiv',
                'trained_on_lines': sorted(lines),
            }, path)

        # Every save_interval epochs rather than every epoch: with optimizer
        # state these are around 200 MB each. The last epoch is always kept.
        periodic = epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs
        if periodic:
            save_checkpoint(checkpoint_path(save_dir, epoch, MODEL_NAME))
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path)
            print(f"  new best validation loss {best_val:.4f}")

        if periodic or epoch == start_epoch:
            with torch.no_grad():
                images = next(iter(val_loader))[:6].to(device)
                recon = decoder(encoder(images)[0], save_steps=False)
            grid = torch.cat([images, recon.clamp(-1, 1)])
            save_image((grid + 1) / 2,
                       results_dir / 'reconstructions' / f'epoch_{epoch:03d}.png',
                       nrow=len(images))

            frame = np.array([[r['epoch'], r['train_loss'], r['val_loss']]
                              for r in history])
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(frame[:, 0], frame[:, 1], label='train')
            ax.plot(frame[:, 0], frame[:, 2], label='validation')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_title(f'{MODEL_NAME} training')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(results_dir / 'loss_curve.png', dpi=140)
            plt.close(fig)

        with open(results_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=float)

    print(f"\nTraining complete. Best validation loss {best_val:.4f}")
    print(f"Checkpoints in {save_dir}")
    print(f"Reconstructions and loss curve in {results_dir}")
    print(f"\ntrain_widiv.py reads {best_path.name} by default.")


if __name__ == '__main__':
    main()
