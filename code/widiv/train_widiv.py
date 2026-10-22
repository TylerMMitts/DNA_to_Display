# Trains the genotype-conditioned diffusion model on the WiDiv synthetic roots.
#
# Forked from latent_diffusion/training/train_seeds.py, one file per dataset as
# for roots and seeds. What differs:
#   - genotypes are 0/1/2 SNP dosages, not founder codes, so they go through
#     DosagePCAProjector (widiv_snp_encoding.py) instead of the one-hot PCA
#   - the train/test split is the one delivered with the data, split_random or
#     split_cluster; test lines are never trained on, never shape the PCA, and
#     never pick a checkpoint. Validation lines come from the training side,
#     chosen the way that split chose its test lines (widiv_data.py)
#   - 512 px images at one common scale (prepare_widiv_images.py), so a
#     4 x 64 x 64 latent from the WiDiv LiteVAE (train_litevae_widiv.py)
#   - quarter turns and both flips, no colour jitter
#
# The UNet presets are the root and seed ones. It is fully convolutional, so
# the parameter counts are unchanged at 64x64; each level is twice the size
# (64 -> 32 -> 16 -> 8 -> 4), so the analysis scripts' 'up_2' layer is 32x32
# here rather than 16x16, and each step costs about four times as much.
#
# Weights go to models/diffusion_widiv_<size>_<split>/, previews and loss
# history to results/training/diffusion_widiv_<size>_<split>/.

import json
import os
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
from tqdm import tqdm

# Puts code/ and code/widiv/ on the import path so this runs directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    pick_device, MODELS_DIR, TRAINING_RESULTS_DIR,
    apply_overrides, best_checkpoint_path, checkpoint_path, find_latest_checkpoint,
    resolve_input, resolve_output,
)
from widiv_paths import (
    WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_IMAGE_SIZE, WIDIV_LITEVAE_MODEL,
    WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA, WIDIV_SPLIT,
)
from widiv_data import SPLITS, choose_val_lines, load_split, make_transforms
from widiv_litevae import load_widiv_litevae
from widiv_snp_encoding import DosagePCAEncoder, DosagePCAProjector, load_widiv_genotypes

from latent_diffusion.models.ldm import LatentDiffusionModel
from latent_diffusion.models.unet import DenoisingUNet
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import (
    generate_batch, load_original, save_multi_comparison,
)

# Every checkpoint this script writes is prefixed with this, so a stray .pt file
# still says which model produced it once it has been copied elsewhere.
MODEL_NAME = 'diffusion_widiv'

# Same presets as the root and seed trainers. WiDiv has about 3,000 training
# images from about 730 lines once validation lines are set aside: 1.6 times
# the MEMA crops and five times the lines, so medium (about 2.7k parameters per
# training image, against 4.4k for roots and 16k for seeds) is the safe default
# and large (about 9k) a reasonable second run.
SIZE_PRESETS = {
    #            UNet    SNP encoder (input 400 PCs)
    'small':  {'base_channels': 32, 'channel_mult': (1, 2, 2, 2),       # 2.7M   0.2M
               'snp_embed_dim': 128, 'd_attention': 128, 'encoder_hidden_dim': 256},
    'medium': {'base_channels': 64, 'channel_mult': (1, 2, 2, 2),       # 8.8M   0.85M
               'snp_embed_dim': 256, 'd_attention': 256, 'encoder_hidden_dim': 512},
    'large':  {'base_channels': 128, 'channel_mult': (1, 2, 2, 2),      # 29.3M  0.85M
               'snp_embed_dim': 256, 'd_attention': 256, 'encoder_hidden_dim': 512},
}


# Loads images and their line's precomputed PCA scores.
class ProjectedWidivDataset(Dataset):

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
            # Another line's scores, used only by evaluation to measure how much
            # the right genotype actually helps.
            'wrong_snp': torch.tensor(sample.get('wrong_projected', sample['projected']),
                                      dtype=torch.float32),
            'genotype': sample['genotype'],
        }


# Ensures that each training sample has both an image and genotype scores.
def build_samples(metadata, image_dir, projected_by_genotype, lines):
    samples, skipped_no_snp, skipped_no_file = [], 0, 0
    for _, row in metadata[metadata.genotype.isin(sorted(lines))].iterrows():
        genotype = row['genotype']
        if genotype not in projected_by_genotype:
            skipped_no_snp += 1
            continue
        path = Path(image_dir) / row['scaled_filename']
        if not path.exists():
            skipped_no_file += 1
            continue
        samples.append({'image_path': path, 'genotype': genotype,
                        'projected': projected_by_genotype[genotype]})
    return samples, skipped_no_snp, skipped_no_file


# Gives every sample a fixed stand-in line from its own split, never its own.
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


# Finds the newest checkpoint to resume from, or None if the folder holds none.
def find_resumable_checkpoint(save_dir, run_name):
    try:
        return find_latest_checkpoint(save_dir, run_name)
    except FileNotFoundError:
        return None


# 'random' and 'cluster' are accepted for the two delivered split columns.
def split_column(name):
    name = name if name.startswith('split_') else f'split_{name}'
    if name not in SPLITS:
        raise SystemExit(f"split must be one of {SPLITS}, not {name!r}")
    return name


def main(overrides=None):
    # Edit these values, then run:
    #     python code/widiv/train_widiv.py
    # On Hellbender, train_widiv_hellbender.sbatch runs one split per array task.
    class cfg:
        # One of SIZE_PRESETS. DIFFUSION_SIZE overrides it.
        model_size = os.environ.get('DIFFUSION_SIZE', 'medium')
        # Which delivered split to train on: 'random' (test lines have relatives
        # in training, like real use) or 'cluster' (whole genetic clusters held
        # out). The README asks for both; the drop between them is how much of
        # the score comes from relatives. WIDIV_SPLIT overrides it.
        split = os.environ.get('WIDIV_SPLIT', 'random')

        # None -> diffusion_widiv_<size>_<split>.
        run_name = None

        genotypes_path = WIDIV_GENOTYPES
        genotype_cache = WIDIV_GENOTYPE_CACHE
        split_path = WIDIV_SPLIT
        metadata_path = WIDIV_SCALED_METADATA
        # The common-scale images, not the delivered ones (see prepare_widiv_images.py).
        image_dir = WIDIV_SCALED_DIR
        # The WiDiv autoencoder. Train it first with code/widiv/train_litevae_widiv.py.
        litevae_checkpoint = WIDIV_LITEVAE_MODEL

        # Share of training lines held back for checkpoint selection and the
        # per-epoch genotype gain. The delivered test lines are separate again.
        val_fraction = 0.1
        save_every = 25

        # PCA size. 0.80 of the training lines' variance is 369 components for
        # split_random and 340 for split_cluster: on this data ridge regression
        # on about 400 PCs matched GBLUP and 168 fell short
        # (results/widiv/genetics_profile/). pca_n_components fixes the number instead.
        pca_target_variance = 0.80
        pca_n_components = None

        # Saves sample images for preview during training
        save_previews = True
        n_preview_genotypes = 3
        n_preview_seeds = 3
        preview_sampling_steps = 20
        preview_latent_size = WIDIV_IMAGE_SIZE // 8

        resume = True

        latent_channels = 4
        num_tokens = 8
        num_res_blocks = 2
        attention_resolutions = [1, 2]

        # Diffusion / optimisation
        num_steps = 1000
        beta_start = 1e-4
        beta_end = 0.02
        learning_rate = 1e-4
        snp_encoder_learning_rate = 1e-4
        # About 180 optimiser steps per epoch at this batch size, against about
        # 23 for seeds, so 400 epochs is about three times the seed run in steps.
        num_epochs = 400
        batch_size = 16
        num_workers = 8
        image_size = WIDIV_IMAGE_SIZE
        seed = 0
        device = pick_device()

        # Evaluation draws the same timesteps and noise every epoch, so a change
        # in validation loss is the model changing, not a different random draw.
        eval_repeats = 4
        n_train_eval_images = 160

    apply_overrides(cfg, overrides)
    size = cfg.model_size
    if size not in SIZE_PRESETS:
        raise SystemExit(f"unknown model size {size!r}; choose one of {list(SIZE_PRESETS)}")
    split = split_column(cfg.split)
    preset = SIZE_PRESETS[size]
    run_name = cfg.run_name or f"{MODEL_NAME}_{size}_{split.replace('split_', '')}"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    save_dir = resolve_output(MODELS_DIR / run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = resolve_output(TRAINING_RESULTS_DIR / run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model size: {size}\nSplit: {split}\nDevice: {device}\n"
          f"Checkpoints: {save_dir}\nResults: {results_dir}")

    resume_path = find_resumable_checkpoint(save_dir, run_name) if cfg.resume else None
    resume_ckpt = None
    start_epoch = 0
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_epoch = resume_ckpt['epoch']
        print(f"\nFound {resume_path.name} - resuming "
              f"({start_epoch} epochs already completed)")
        if resume_ckpt.get('split') != split:
            raise SystemExit(f"{resume_path.name} was trained on {resume_ckpt.get('split')}, "
                             f"not {split}. Give this run its own run_name.")
    elif cfg.resume:
        print(f"\nresume=True but no {run_name}_epoch_N.pt found in save_dir - starting fresh")
    else:
        print("\nresume=False - starting fresh")

    # Lines: training lines of this split minus validation lines fit the model;
    # test lines are left out entirely.
    split_table = load_split(resolve_input(cfg.split_path, 'WiDiv train/test split'))
    val_lines = choose_val_lines(split_table, split, cfg.val_fraction, cfg.seed)
    train_lines = set(split_table.index[split_table[split] == 'train']) - val_lines
    test_lines = set(split_table.index[split_table[split] == 'test'])
    if resume_ckpt is not None and set(resume_ckpt['val_genotypes']) != val_lines:
        raise SystemExit(
            "Resuming, but the validation lines do not match the checkpoint's - the "
            "split file or val_fraction must have changed. Fix the mismatch, or set "
            "resume = False to start over.")

    t0 = time.time()
    line_names, snp_names, dosage = load_widiv_genotypes(
        resolve_input(cfg.genotypes_path, 'WiDiv genotype table'), resolve_output(cfg.genotype_cache))
    print(f"\nGenotypes: {len(line_names)} lines x {len(snp_names):,} SNPs "
          f"({time.time() - t0:.1f}s)")
    row_of = {n: i for i, n in enumerate(line_names)}

    if resume_ckpt is not None:
        # Reuses the exact basis already in use rather than refitting.
        projector = DosagePCAProjector.from_state_dict(resume_ckpt['snp_projector'])
        projector.check_columns(snp_names)
        print(f"Restored SNP projector from checkpoint: {projector.output_dim} components")
    else:
        fit_rows = [row_of[g] for g in sorted(train_lines) if g in row_of]
        projector = DosagePCAProjector(n_components=cfg.pca_n_components,
                                       target_variance=cfg.pca_target_variance)
        projector.fit(dosage[fit_rows], snp_names)

    projected = projector.transform(dosage)
    projected_by_genotype = {name: projected[i] for i, name in enumerate(line_names)}

    # Image data and metadata
    image_dir = resolve_input(cfg.image_dir, 'scaled WiDiv image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'scaled WiDiv metadata '
                                         '(run prepare_widiv_images.py)'))
    train_samples, no_snp, no_file = build_samples(metadata, image_dir, projected_by_genotype,
                                                   train_lines)
    val_samples, v_no_snp, v_no_file = build_samples(metadata, image_dir, projected_by_genotype,
                                                     val_lines)
    print(f"\nImages: {len(train_samples)} train from {len({s['genotype'] for s in train_samples})} "
          f"lines, {len(val_samples)} val from {len({s['genotype'] for s in val_samples})} lines "
          f"({no_snp + v_no_snp} skipped for missing genotypes, {no_file + v_no_file} for "
          f"missing files); {len(test_lines)} test lines left out")
    if not train_samples or not val_samples:
        raise SystemExit("no usable samples - run prepare_widiv_images.py and check image_dir")

    assign_wrong_genotypes(train_samples, projected_by_genotype, cfg.seed)
    assign_wrong_genotypes(val_samples, projected_by_genotype, cfg.seed)
    pick = np.linspace(0, len(train_samples) - 1,
                       min(cfg.n_train_eval_images, len(train_samples))).astype(int)
    train_eval_samples = [train_samples[i] for i in pick]

    preview_by_genotype = {}
    for s in val_samples:
        preview_by_genotype.setdefault(s['genotype'], s)
    preview_candidates = sorted(preview_by_genotype.keys())
    np.random.default_rng(cfg.seed + 1).shuffle(preview_candidates)
    preview_genotypes = preview_candidates[:cfg.n_preview_genotypes]
    preview_samples = [preview_by_genotype[g] for g in preview_genotypes]
    if cfg.save_previews:
        print(f"  preview lines (validation): {preview_genotypes}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ProjectedWidivDataset(train_samples, make_transforms(True)),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=pin, drop_last=True)
    val_loader = DataLoader(
        ProjectedWidivDataset(val_samples, make_transforms(False)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    train_eval_loader = DataLoader(
        ProjectedWidivDataset(train_eval_samples, make_transforms(False)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

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
        saved_unet['channel_mult'] = tuple(saved_unet['channel_mult'])
        if saved_unet != unet_config or resume_ckpt['snp_encoder_config'] != encoder_config:
            raise SystemExit(
                f"{resume_path.name} was trained with a different architecture than "
                f"the '{size}' preset now describes.\n"
                f"  checkpoint: {saved_unet}\n  preset:     {unet_config}\n"
                "Restore the preset, or move that folder's checkpoints aside to start over.")

    snp_encoder = DosagePCAEncoder(**encoder_config).to(device)
    unet = DenoisingUNet(**unet_config).to(device)
    if resume_ckpt is not None:
        snp_encoder.load_state_dict(resume_ckpt['snp_encoder_state_dict'])
        unet.load_state_dict(resume_ckpt['unet_state_dict'])
        print("Restored SNP encoder and UNet weights from the resume checkpoint")
    else:
        print("UNet initialised from scratch")
    # Keeps attention_history from growing unboundedly during training, each
    # cross-attention block appends to it on every forward pass otherwise.
    unet.set_store_attention(False)

    litevae_encoder, litevae_decoder = load_widiv_litevae(
        resolve_input(cfg.litevae_checkpoint, 'WiDiv LiteVAE checkpoint '
                      '(train it with train_litevae_widiv.py)'), device)

    scheduler = DiffusionScheduler(num_steps=cfg.num_steps, beta_start=cfg.beta_start,
                                   beta_end=cfg.beta_end)
    # The scheduler holds plain tensors, not module buffers, so .to(device) on
    # the model does not move them.
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    ldm = LatentDiffusionModel(
        litevae_encoder=litevae_encoder, litevae_decoder=litevae_decoder,
        snp_encoder=snp_encoder, unet=unet, scheduler=scheduler, device=device)

    # The latent the UNet will see must be the one previews and analysis assume.
    with torch.no_grad():
        probe = ldm.encode_image(next(iter(val_loader))['image'][:1].to(device))
    if tuple(probe.shape[1:]) != (cfg.latent_channels, cfg.preview_latent_size,
                                  cfg.preview_latent_size):
        raise SystemExit(f"LiteVAE gives a {tuple(probe.shape[1:])} latent, expected "
                         f"{cfg.latent_channels} x {cfg.preview_latent_size} x "
                         f"{cfg.preview_latent_size}")

    n_unet = sum(p.numel() for p in unet.parameters())
    n_encoder = sum(p.numel() for p in snp_encoder.parameters())
    print(f"\nTrainable parameters: {n_unet + n_encoder:,} "
          f"(UNet {n_unet:,}, SNP encoder {n_encoder:,}); "
          f"{(n_unet + n_encoder) / len(train_samples):,.0f} per training image")

    optimizer = optim.Adam([{'params': ldm.unet.parameters(), 'lr': cfg.learning_rate},
                            {'params': ldm.snp_encoder.parameters(),
                             'lr': cfg.snp_encoder_learning_rate}])
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        print("Restored optimizer state")

    # T_max is the epochs remaining, so the cosine schedule continues smoothly
    # from the resumed epoch.
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.num_epochs - start_epoch, 1), eta_min=1e-7)

    if start_epoch >= cfg.num_epochs:
        raise SystemExit(
            f"Checkpoint already has {start_epoch} epochs completed, which is "
            f">= num_epochs={cfg.num_epochs}. Raise num_epochs to continue training.")

    best_val = float('inf')
    best_path = best_checkpoint_path(save_dir, run_name)
    if best_path.exists():
        try:
            best_val = torch.load(best_path, map_location='cpu', weights_only=False)['val_loss']
        except Exception:
            pass
    elif resume_ckpt is not None:
        best_val = resume_ckpt.get('val_loss', float('inf'))

    history = []
    history_path = results_dir / 'training_history.csv'
    if resume_ckpt is not None and history_path.exists():
        # Checkpoints are written every save_every epochs, history every epoch,
        # so rows past the checkpoint are redone. An unreadable file only costs
        # the loss curve, not the run.
        try:
            history = [r for r in pd.read_csv(history_path).to_dict('records')
                       if r['epoch'] <= start_epoch]
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            print(f"  {history_path.name} is unreadable ({type(exc).__name__}) - "
                  "starting a fresh loss curve, training is unaffected")
            history = []

    # Trainable parts in train mode, the frozen LiteVAE always in eval, so its
    # latents never shift under the diffusion model.
    def set_train_mode():
        ldm.train()
        ldm.litevae_encoder.eval()
        ldm.litevae_decoder.eval()

    preview_dir = results_dir / 'previews'
    preview_latent_shape = (cfg.latent_channels, cfg.preview_latent_size, cfg.preview_latent_size)
    # Fixed seeds per line, reused at every preview, so what changes between
    # previews is the model and not the noise it started from.
    preview_seeds = [[cfg.seed + 100 * i + j for j in range(cfg.n_preview_seeds)]
                     for i in range(len(preview_samples))]

    def save_previews(epoch_num):
        ldm.eval()
        preview_dir.mkdir(parents=True, exist_ok=True)
        snp_batch = torch.tensor(
            np.stack([s['projected'] for s in preview_samples for _ in range(cfg.n_preview_seeds)]),
            dtype=torch.float32, device=device)
        flat_seeds = [s for seeds in preview_seeds for s in seeds]
        with torch.no_grad():
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
        print(f"  wrote {len(preview_samples)} preview images to {preview_dir}")

    # Denoising loss with each image's own line and with another line from the
    # same set, on identical timesteps and noise every call. The held-out gap is
    # what carries over to new lines; a trained gap far larger than it means the
    # genotype is being used as a key to recall memorised images.
    eval_cache = {}

    def evaluate(name, loader):
        ldm.eval()
        saved = get_rng_state(device)
        true_total, wrong_total, n = 0.0, 0.0, 0
        with torch.no_grad():
            if name not in eval_cache:
                torch.manual_seed(cfg.seed + 999)
                eval_cache[name] = [
                    (ldm.encode_image(b['image'].to(device)), b['snp'].to(device),
                     b['wrong_snp'].to(device)) for b in loader]
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

    if resume_ckpt is not None:
        print(f"Resuming at epoch {start_epoch + 1}/{cfg.num_epochs}, "
              f"best validation loss so far {best_val:.4f}")

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
                'encoding': 'widiv_dosage_pca',
                'split': split,
                'val_genotypes': sorted(val_lines),
                'test_genotypes': sorted(test_lines),
                'model_size': size,
                'image_size': cfg.image_size,
                'latent_size': cfg.preview_latent_size,
                # Says which dataset this came from, so a WiDiv checkpoint copied
                # next to a root one still identifies itself.
                'dataset': 'widiv',
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
        # or out-of-disk write cannot leave a half-written file behind.
        tmp_path = history_path.with_suffix('.csv.tmp')
        pd.DataFrame(history).to_csv(tmp_path, index=False)
        os.replace(tmp_path, history_path)

    with open(results_dir / 'run_summary.json', 'w') as f:
        json.dump({
            'dataset': 'widiv', 'split': split,
            'model_size': size, 'unet_parameters': n_unet,
            'snp_encoder_parameters': n_encoder,
            'encoding': 'widiv_dosage_pca',
            'projected_dim': int(projector.output_dim),
            'pca_variance': float(projector.explained_variance_ratio_.sum()),
            'train_lines': len({s['genotype'] for s in train_samples}),
            'val_lines': len(val_lines), 'test_lines': len(test_lines),
            'train_images': len(train_samples),
            'resumed_from': str(resume_path) if resume_path else None,
            'started_at_epoch': start_epoch,
            'image_dir': str(image_dir),
            'image_size': cfg.image_size,
            'epochs': cfg.num_epochs,
            'best_val_loss': best_val,
            'final': history[-1] if history else None,
            'preview_genotypes': preview_genotypes if cfg.save_previews else None,
        }, f, indent=2, default=float)

    print(f"\nTraining complete. Best validation loss {best_val:.4f}")
    print(f"Checkpoints in {save_dir}")
    print(f"training_history.csv and run_summary.json in {results_dir}")


if __name__ == '__main__':
    main()
