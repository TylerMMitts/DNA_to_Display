# Trains the LiteVAE autoencoder on the scaled seed kernel images.
#
# A separate file from train_litevae.py rather than a shared one with a switch,
# so running this trains on seeds and running that trains on roots, with nothing
# to configure first.
#
# The diffusion model trains inside whatever latent space this produces, so the
# seed diffusion trainer needs this to exist first. The root LiteVAE is not a
# substitute: it was fitted to photographed cross-sections, and kernels are
# flat-shaded shapes on white, which is not what it learned to compress.
#
# Weights go to models/litevae_seeds/, reconstructions and loss curves to
# results/training/litevae_seeds/. Resumes from the newest checkpoint if one
# exists.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    pick_device, SEED_LITEVAE_DIR, SEED_SCALED_DIR, TRAINING_RESULTS_DIR,
    best_checkpoint_path, checkpoint_path, find_latest_checkpoint,
    resolve_input, resolve_output,
)

from litevae.models import LiteVAEEncoder, LiteVAEDecoder

# Every checkpoint this script writes is prefixed with this, so a stray .pt file
# still says which model produced it once it has been copied elsewhere.
MODEL_NAME = 'litevae_seeds'


class SeedImageDataset(Dataset):

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


# Data augmentation and preprocessing transforms
def make_transforms(image_size, augment):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    if not augment:
        return transforms.Compose(base)
    # Horizontal flips only, unlike the root autoencoder, which also rotates,
    # flips vertically, jitters colour and blurs.
    #
    # A kernel has a real top and bottom and every plate is drawn the same way
    # up, so vertical flips and rotations invent orientations that never occur.
    # Colour jitter is worse here: the plates encode kernel colour as horizontal
    # bands, so it is a trait rather than a nuisance, and an autoencoder taught
    # to ignore it will spend less capacity reproducing it. Blur would simulate
    # a focus that rendered plates do not have.
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ] + base)


class LiteVAELoss(nn.Module):

    def __init__(self, recon_weight=1.0, kl_weight=0.001):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.mse = nn.MSELoss()

    def forward(self, recon, original, z_mean, z_logvar):
        recon_loss = self.mse(recon, original)
        # KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var)), averaged over
        # elements so the weighting does not depend on batch size.
        kl_loss = -0.5 * torch.sum(
            1 + z_logvar - z_mean.pow(2) - z_logvar.exp()) / z_mean.numel()
        return (self.recon_weight * recon_loss + self.kl_weight * kl_loss,
                recon_loss, kl_loss)


# Finds the newest checkpoint to resume from, or None if the folder holds none.
# paths.find_latest_checkpoint raises when there is nothing to load, which is a
# normal first run here rather than an error.
def find_resumable_checkpoint(save_dir):
    try:
        return find_latest_checkpoint(save_dir, MODEL_NAME)
    except FileNotFoundError:
        return None


def main():
    # Edit these values, then run:
    #     python code/litevae/train_litevae_seeds.py
    class cfg:
        # The rescaled images, not the cropped ones - the diffusion model trains
        # on these, so the autoencoder has to be fitted to the same thing.
        image_dir = SEED_SCALED_DIR
        save_dir = SEED_LITEVAE_DIR
        results_dir = TRAINING_RESULTS_DIR / MODEL_NAME

        # Architecture. These four decide the latent shape, and the diffusion
        # trainer assumes 4 x 32 x 32, so changing latent_channels here means
        # changing it there too.
        latent_channels = 4
        feature_channels = 64
        base_channels = 512
        num_blocks = 3
        num_res_blocks = 2

        num_epochs = 250
        batch_size = 16
        learning_rate = 1e-4
        recon_weight = 1.0
        kl_weight = 0.001
        val_split = 0.1
        num_workers = 4
        image_size = 256
        seed = 0

        # How often to save a checkpoint, reconstructions and the loss curve,
        # in epochs. The best checkpoint is written whenever validation improves
        # regardless of this.
        save_interval = 5
        resume = True
        device = pick_device()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    image_dir = resolve_input(cfg.image_dir, 'scaled seed image directory')
    save_dir = resolve_output(cfg.save_dir)
    results_dir = resolve_output(cfg.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'reconstructions').mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}\nImages: {image_dir}")
    print(f"Checkpoints: {save_dir}\nResults: {results_dir}")

    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not paths:
        raise SystemExit(f"no images found in {image_dir} - run "
                         "code/rescale_seed_crops.py first")

    # Split by image, with each split getting its own dataset object. Sharing one
    # dataset between splits and then setting the transform afterwards would set
    # it for both, silently turning augmentation off for training.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(paths))
    n_val = max(1, int(round(cfg.val_split * len(paths))))
    val_paths = [paths[i] for i in order[:n_val]]
    train_paths = [paths[i] for i in order[n_val:]]

    train_loader = DataLoader(
        SeedImageDataset(train_paths, make_transforms(cfg.image_size, True)),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(
        SeedImageDataset(val_paths, make_transforms(cfg.image_size, False)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())

    print(f"Images: {len(paths)} total, {len(train_paths)} train, "
          f"{len(val_paths)} val, {len(train_loader)} batches per epoch")

    model_config = {
        'latent_channels': cfg.latent_channels,
        'feature_channels': cfg.feature_channels,
        'base_channels': cfg.base_channels,
        'num_blocks': cfg.num_blocks,
        'num_res_blocks': cfg.num_res_blocks,
        'learning_rate': cfg.learning_rate,
    }

    resume_path = find_resumable_checkpoint(save_dir) if cfg.resume else None
    resume_ckpt = None
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        saved = {k: v for k, v in resume_ckpt['config'].items() if k != 'learning_rate'}
        wanted = {k: v for k, v in model_config.items() if k != 'learning_rate'}
        if saved != wanted:
            raise SystemExit(
                f"{resume_path.name} was trained with a different architecture.\n"
                f"  checkpoint: {saved}\n  config:     {wanted}\n"
                "Restore the settings, or move that folder's checkpoints aside.")

    encoder = LiteVAEEncoder(in_channels=3, latent_channels=cfg.latent_channels,
                             feature_channels=cfg.feature_channels,
                             num_blocks=cfg.num_blocks).to(device)
    decoder = LiteVAEDecoder(latent_channels=cfg.latent_channels, output_channels=3,
                             base_channels=cfg.base_channels,
                             num_res_blocks=cfg.num_res_blocks).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)
    criterion = LiteVAELoss(recon_weight=cfg.recon_weight, kl_weight=cfg.kl_weight)

    start_epoch = 1
    best_val = float('inf')
    history = []

    if resume_ckpt is not None:
        encoder.load_state_dict(resume_ckpt['encoder_state_dict'])
        decoder.load_state_dict(resume_ckpt['decoder_state_dict'])
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        start_epoch = resume_ckpt['epoch'] + 1
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
        totals = np.zeros(3)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for images in progress:
            images = images.to(device, non_blocking=True)
            z, z_mean, z_logvar = encoder(images, save_steps=False)
            recon = decoder(z, save_steps=False)
            loss, recon_loss, kl_loss = criterion(recon, images, z_mean, z_logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            totals += [loss.item(), recon_loss.item(), kl_loss.item()]
            progress.set_postfix({'loss': f'{loss.item():.4f}',
                                  'recon': f'{recon_loss.item():.4f}'})

        train_loss, train_recon, train_kl = totals / max(len(train_loader), 1)

        encoder.eval()
        decoder.eval()
        val_total = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device, non_blocking=True)
                z, z_mean, z_logvar = encoder(images, save_steps=False)
                recon = decoder(z, save_steps=False)
                val_total += criterion(recon, images, z_mean, z_logvar)[0].item()
        val_loss = val_total / max(len(val_loader), 1)

        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'recon_loss': train_recon, 'kl_loss': train_kl,
                        'val_loss': val_loss,
                        'lr': optimizer.param_groups[0]['lr']})
        print(f"  train {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f})"
              f"   val {val_loss:.4f}   lr {optimizer.param_groups[0]['lr']:.2e}")

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
                'dataset': 'seeds',
            }, path)

        # Every save_interval epochs rather than every epoch. These carry the
        # optimizer and scheduler state as well as the weights, so they are
        # around 200 MB each - one per epoch over a 250 epoch run is about 50 GB
        # and will exhaust a cluster quota. The last epoch is always kept, so
        # the end of a run is never left unsaved when num_epochs is not a
        # multiple of the interval.
        periodic = epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs
        if periodic:
            save_checkpoint(checkpoint_path(save_dir, epoch, MODEL_NAME))
        # The best one is kept whenever it improves, whatever epoch that lands
        # on, so the most useful checkpoint is never the one that got skipped.
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path)
            print(f"  new best validation loss {best_val:.4f}")

        if periodic or epoch == start_epoch:
            with torch.no_grad():
                images = next(iter(val_loader))[:8].to(device)
                recon = decoder(encoder(images, save_steps=False)[0], save_steps=False)
            grid = torch.cat([images, recon])
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
    print(f"\nPoint cfg.litevae_checkpoint in train_seeds.py at "
          f"{best_path.name} before training the diffusion model.")


if __name__ == '__main__':
    main()
