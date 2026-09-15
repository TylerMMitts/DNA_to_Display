# Trains the genotype-conditioned diffusion model on one-hot SNP encoding.
#
# This is the current trainer. The UNet and SNP encoder train together from
# scratch. Weights go to models/diffusion_onehot_<size>/, previews and loss
# history to results/training/diffusion_onehot_<size>/. LiteVAE is loaded
# frozen, so a trained autoencoder has to exist first.

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    pick_device,
    CROPPED_IMAGES_DIR, IMAGE_METADATA, LITEVAE_MODEL, MODELS_DIR, SNP_PARQUET,
    TRAINING_RESULTS_DIR, best_checkpoint_path, checkpoint_path, find_latest_checkpoint,
    resolve_input, resolve_output,
)

from latent_diffusion.models.ldm import LatentDiffusionModel
from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.models.snp_encoding import SNPProjector, OneHotSNPEncoder
from latent_diffusion.models.unet import DenoisingUNet
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from litevae.models import LiteVAEEncoder, LiteVAEDecoder
from latent_diffusion.generation.generate_from_dataset import (
    generate_batch, load_original, save_multi_comparison,
)

# Every checkpoint this script writes is prefixed with this, so a stray .pt file
# still says which model produced it once it has been copied elsewhere.
MODEL_NAME = 'diffusion_onehot'

# Model sizes, from smallest to largest. Each run trains one of them into its
# own folder, models/diffusion_onehot_<size>/, so sizes never resume from each
# other's checkpoints or from the original 209M model in models/diffusion_onehot/.
#
# Sized for ~700 training images from ~110 genotypes. The original UNet
# (base 128, levels 1-2-4-8) had 209M parameters, 79% of them in the 4x4 and
# 2x2 levels where the latent has only 16 and 4 positions, and it memorised:
# training images scored 22% better with their own genotype than with another
# training genotype, while held-out genotypes scored the same either way.
#
# All sizes keep four levels (32 -> 16 -> 8 -> 4 -> 2) and cross-attention at
# levels 1 and 2, so the analysis scripts' 'up_2' layer is still the 16x16 one.
# Levels 1-2-2-2 follow the DDPM CIFAR-10 layout, the closest well-tested
# reference for a 32x32 input; 'large' is about that model's size, which was
# tuned for 50k images, so it is the upper end rather than the target.
SIZE_PRESETS = {
    #            UNet    SNP encoder
    'small':  {'base_channels': 32, 'channel_mult': (1, 2, 2, 2),       # 2.7M   0.2M
               'snp_embed_dim': 128, 'd_attention': 128, 'encoder_hidden_dim': 256},
    'medium': {'base_channels': 64, 'channel_mult': (1, 2, 2, 2),       # 8.8M   0.75M
               'snp_embed_dim': 256, 'd_attention': 256, 'encoder_hidden_dim': 512},
    'large':  {'base_channels': 128, 'channel_mult': (1, 2, 2, 2),      # 29.3M  0.75M
               'snp_embed_dim': 256, 'd_attention': 256, 'encoder_hidden_dim': 512},
}


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
            # Another genotype's coordinates, used only by evaluation to measure
            # how much the right genotype actually helps.
            'wrong_snp': torch.tensor(sample.get('wrong_projected', sample['projected']),
                                      dtype=torch.float32),
            'genotype': sample['genotype'],
        }


# Gives every sample a fixed stand-in genotype from its own split, never its own.
# Fixed so the with/without comparison uses the same pairs every epoch.
def assign_wrong_genotypes(samples, projected_by_genotype, seed):
    rng = np.random.default_rng(seed)
    pool = sorted({s['genotype'] for s in samples})
    if len(pool) < 2:
        return
    partner = {}
    for g in pool:
        others = [o for o in pool if o != g]
        partner[g] = others[rng.integers(len(others))]
    for s in samples:
        s['wrong_projected'] = projected_by_genotype[partner[s['genotype']]]


# Saves and restores every random generator evaluation touches, so evaluation
# can reseed for identical draws each epoch without disturbing training.
def get_rng_state(device):
    state = {'cpu': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    elif device.type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    return state


def set_rng_state(device, state):
    torch.set_rng_state(state['cpu'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'], device)
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])

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
def find_resumable_checkpoint(save_dir, run_name):
    try:
        return find_latest_checkpoint(save_dir, run_name)
    except FileNotFoundError:
        return None


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/training/train_onehot.py
    # On Hellbender, train_onehot_hellbender.sbatch runs one size per array task.
    class cfg:
        # One of SIZE_PRESETS. The DIFFUSION_SIZE environment variable overrides
        # it, which is how the Hellbender array job trains several at once.
        model_size = 'medium'

        snp_parquet = SNP_PARQUET
        metadata_path = IMAGE_METADATA
        image_dir = CROPPED_IMAGES_DIR
        litevae_checkpoint = LITEVAE_MODEL

        save_every = 25
        val_fraction = 0.2

        # Saves sample images for preview during training
        save_previews = True
        n_preview_genotypes = 3
        # Noise seeds per genotype. One sample cannot show whether a change
        # between epochs is the model or that particular noise draw, so each
        # preview generates several from the same genotype.
        n_preview_seeds = 3
        preview_sampling_steps = 20
        preview_latent_size = 32

        # Resumes from the latest checkpoint in this size's folder if one exists.
        resume = True
        # None trains the UNet and SNP encoder together from the first step. A
        # warm start only loads into a UNet of exactly the same size, and it
        # means the encoder joins a UNet already trained without it.
        warm_start_checkpoint = None

        # Auto finds the number of founders in snp data
        founders = None
        # Sets the target variance for PCA on the SNP data
        pca_target_variance = 0.95
        pca_random_state = 0

        latent_channels = 4
        num_tokens = 8
        num_res_blocks = 2
        attention_resolutions = [1, 2]

        # Diffusion / optimisation
        num_steps = 1000
        beta_start = 1e-4
        beta_end = 0.02
        # Both start from scratch, so both use the same rate. They are separate
        # settings so the encoder can be slowed or sped up on its own.
        learning_rate = 1e-4
        snp_encoder_learning_rate = 1e-4
        # ~43 optimiser steps per epoch at this batch size, so 1000 epochs is
        # about 43k steps. The best checkpoint is kept by validation loss, so
        # running long does not force using an overfitted final epoch.
        num_epochs = 1000
        batch_size = 16
        num_workers = 4
        image_size = 256
        seed = 0
        device = pick_device()

        # Evaluation draws the same timesteps and noise every epoch, so a change
        # in validation loss is the model changing, not a different random draw.
        # Each image is scored this many times at different timesteps.
        eval_repeats = 4
        # Training images scored the same way, to compare against validation.
        n_train_eval_images = 160

    size = os.environ.get('DIFFUSION_SIZE', cfg.model_size)
    if size not in SIZE_PRESETS:
        raise SystemExit(f"unknown model size {size!r}; choose one of {list(SIZE_PRESETS)}")
    preset = SIZE_PRESETS[size]
    run_name = f'{MODEL_NAME}_{size}'

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    # Weights to models/diffusion_onehot_<size>/, everything else to results/.
    save_dir = resolve_output(MODELS_DIR / run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = resolve_output(TRAINING_RESULTS_DIR / run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model size: {size}\nDevice: {device}\nCheckpoints: {save_dir}\nResults: {results_dir}")

    resume_path = find_resumable_checkpoint(save_dir, run_name) if cfg.resume else None
    resume_ckpt = None
    start_epoch = 0
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = resume_ckpt['epoch']
        print(f"\nFound {resume_path.name} - resuming "
              f"({start_epoch} epochs already completed)")
    elif cfg.resume:
        print(f"\nresume=True but no {run_name}_epoch_N.pt found in save_dir - "
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

    assign_wrong_genotypes(train_samples, projected_by_genotype, cfg.seed)
    assign_wrong_genotypes(val_samples, projected_by_genotype, cfg.seed)
    # A fixed spread of training images, scored exactly like validation.
    pick = np.linspace(0, len(train_samples) - 1,
                       min(cfg.n_train_eval_images, len(train_samples))).astype(int)
    train_eval_samples = [train_samples[i] for i in pick]

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
        pin_memory=torch.cuda.is_available())
    train_eval_loader = DataLoader(
        ProjectedRootDataset(train_eval_samples, make_transforms(cfg.image_size, False)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available())

    encoder_config = {
        'input_dim': projector.output_dim,
        'embedding_dim': preset['snp_embed_dim'],
        'num_tokens': cfg.num_tokens,
        'hidden_dim': preset['encoder_hidden_dim'],
    }
    unet_config = {
        'latent_channels': cfg.latent_channels, 'base_channels': preset['base_channels'],
        'snp_embed_dim': preset['snp_embed_dim'], 'd_attention': preset['d_attention'],
        'num_res_blocks': cfg.num_res_blocks,
        'attention_resolutions': cfg.attention_resolutions,
        'channel_mult': tuple(preset['channel_mult']),
    }
    if resume_ckpt is not None:
        saved_unet = dict(resume_ckpt['unet_config'])
        saved_unet['channel_mult'] = tuple(saved_unet.get('channel_mult', (1, 2, 4, 8)))
        if saved_unet != unet_config or resume_ckpt['snp_encoder_config'] != encoder_config:
            raise SystemExit(
                f"{resume_path.name} was trained with a different architecture than "
                f"the '{size}' preset now describes.\n"
                f"  checkpoint: {saved_unet}\n  preset:     {unet_config}\n"
                "Restore the preset, or move that folder's checkpoints aside to start over.")

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

    n_unet = sum(p.numel() for p in unet.parameters())
    n_encoder = sum(p.numel() for p in snp_encoder.parameters())
    print(f"\nTrainable parameters: {n_unet + n_encoder:,} "
          f"(UNet {n_unet:,}, SNP encoder {n_encoder:,})")

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
    best_path = best_checkpoint_path(save_dir, run_name)
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
        # History is written every epoch but checkpoints only every save_every,
        # so a job killed in between leaves rows the resumed run will redo.
        #
        # A run killed partway through writing this file - or one that ran out of
        # disk while writing it - leaves a file that exists but has no header.
        # The history is only the loss curve, so losing it is not worth losing
        # the run for; say so and carry on with an empty one.
        try:
            history = [r for r in pd.read_csv(history_path).to_dict('records')
                       if r['epoch'] <= start_epoch]
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            print(f"  {history_path.name} is unreadable ({type(exc).__name__}) - "
                  "starting a fresh loss curve, training is unaffected")
            history = []

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
    # A fixed set of seeds per genotype, reused at every preview call rather than
    # drawn fresh each epoch, so what changes between previews is the model and
    # not the noise it started from.
    preview_seeds = [[cfg.seed + 100 * i + j for j in range(cfg.n_preview_seeds)]
                     for i in range(len(preview_samples))]

    def save_previews(epoch_num):
        ldm.eval()
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Each genotype repeated once per seed, so one sampling run covers every
        # panel of every preview rather than one run per genotype.
        snp_batch = torch.tensor(
            np.stack([s['projected'] for s in preview_samples
                      for _ in range(cfg.n_preview_seeds)]),
            dtype=torch.float32, device=device)
        flat_seeds = [s for seeds in preview_seeds for s in seeds]
        generated = generate_batch(snp_encoder, unet, scheduler, litevae_decoder,
                                   snp_batch, flat_seeds, device,
                                   preview_latent_shape, cfg.preview_sampling_steps)
        for i, s in enumerate(preview_samples):
            original = load_original(s['image_path'], cfg.image_size)
            start = i * cfg.n_preview_seeds
            save_multi_comparison(
                original, generated[start:start + cfg.n_preview_seeds],
                s['genotype'], preview_seeds[i],
                preview_dir / f"{s['genotype']}_epoch{epoch_num:04d}.png")
        set_train_mode()
        print(f"  wrote {len(preview_samples)} preview images "
              f"({cfg.n_preview_seeds} seeds each) to {preview_dir}")

    # Denoising loss with each image's own genotype and with another genotype
    # from the same split, on identical timesteps and noise every call.
    #
    # The gap between the two is how much the genotype actually helps. On
    # held-out genotypes that gap is the thing worth maximising: it only exists
    # if the model learned something about genotypes that carries over. On
    # training genotypes a gap far larger than the held-out one means the
    # genotype is being used as a key to recall memorised images.
    eval_cache = {}

    def evaluate(name, loader):
        ldm.eval()
        saved = get_rng_state(device)
        true_total, wrong_total, n = 0.0, 0.0, 0
        with torch.no_grad():
            # LiteVAE is frozen, so each image is encoded once on the first call
            # and its latent reused, rather than re-encoded every epoch.
            if name not in eval_cache:
                torch.manual_seed(cfg.seed + 999)
                eval_cache[name] = [
                    (ldm.encode_image(b['image'].to(device)), b['snp'].to(device),
                     b['wrong_snp'].to(device)) for b in loader]
            # Reseeded after encoding, so the first call draws the same
            # timesteps and noise as every later one.
            torch.manual_seed(cfg.seed + 1000)
            for _ in range(cfg.eval_repeats):
                for z, snp, wrong_snp in eval_cache[name]:
                    t = torch.randint(0, scheduler.num_steps, (z.shape[0],), device=device)
                    noise = torch.randn_like(z)
                    z_t = scheduler.add_noise(z, noise, t)
                    true_total += F.mse_loss(unet(z_t, t, snp_encoder(snp)), noise,
                                             reduction='sum').item()
                    wrong_total += F.mse_loss(unet(z_t, t, snp_encoder(wrong_snp)), noise,
                                              reduction='sum').item()
                    n += noise.numel()
        set_rng_state(device, saved)
        true_loss, wrong_loss = true_total / n, wrong_total / n
        return true_loss, wrong_loss, wrong_loss / true_loss - 1.0

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

        val_loss, val_wrong, val_gain = evaluate('val', val_loader)
        train_eval, train_wrong, train_gain = evaluate('train', train_eval_loader)

        lr_scheduler.step()
        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_loss': val_loss, 'val_loss_wrong_genotype': val_wrong,
                        'train_eval_loss': train_eval,
                        'train_eval_loss_wrong_genotype': train_wrong,
                        'genotype_gain_heldout': val_gain,
                        'genotype_gain_trained': train_gain,
                        'lr': lr_scheduler.get_last_lr()[0]})
        print(f"  train {train_loss:.4f}   val {val_loss:.4f}   "
              f"genotype gain: held-out {val_gain:+.1%}, trained {train_gain:+.1%}   "
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
                'model_size': size,
                'genotype_gain_heldout': val_gain,
                'genotype_gain_trained': train_gain,
            }, path)

        if (epoch + 1) % cfg.save_every == 0:
            checkpoint(checkpoint_path(save_dir, epoch + 1, run_name), 'periodic')
            if cfg.save_previews and preview_samples:
                save_previews(epoch + 1)
        if val_loss < best_val:
            best_val = val_loss
            checkpoint(best_path, 'best validation loss')

        # Written to a temporary file and moved into place, so an interrupted
        # or out-of-disk write cannot leave a half-written file behind. The
        # move is atomic on POSIX.
        tmp_path = history_path.with_suffix('.csv.tmp')
        pd.DataFrame(history).to_csv(tmp_path, index=False)
        os.replace(tmp_path, history_path)

    with open(results_dir / 'run_summary.json', 'w') as f:
        json.dump({
            'model_size': size, 'unet_parameters': n_unet, 'snp_encoder_parameters': n_encoder,
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
