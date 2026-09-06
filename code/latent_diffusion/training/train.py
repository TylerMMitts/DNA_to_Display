# The original trainer, which fed SNP founder codes in as plain numbers.
#
# Superseded by train_onehot.py and kept only to reproduce the older
# checkpoints. Weights go to models/diffusion_numeric/.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_NUMERIC_DIR, IMAGES_DIR, IMAGE_METADATA, LITEVAE_MODEL,
    SNP_PARQUET, TRAINING_RESULTS_DIR, resolve_output,
)

from latent_diffusion.models.ldm import LatentDiffusionModel
from latent_diffusion.models.snp_encoder import (
    SNPEncoder, build_snp_encoder_with_optimal_pca, load_snp_data_from_parquet,
)
from latent_diffusion.models.unet import DenoisingUNet
from latent_diffusion.diffusion.scheduler import DiffusionScheduler

from litevae.models import LiteVAEEncoder, LiteVAEDecoder

# This is the original numeric-encoding trainer, kept for reproducing the older
# checkpoints. train_onehot.py is the one-hot successor and the current default.
MODEL_NAME = 'diffusion_numeric'


class Config:
    # Data
    image_dir = IMAGES_DIR
    metadata_path = IMAGE_METADATA
    snp_parquet_path = SNP_PARQUET
    batch_size = 16
    num_workers = 4

    # Model
    latent_channels = 4
    base_channels = 128
    snp_embed_dim = 512
    d_attention = 512
    num_tokens = 8
    num_res_blocks = 2
    attention_resolutions = [1, 2, 4]

    # Diffusion
    num_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    learning_rate = 1e-4
    num_epochs = 200
    # Weights go to models/diffusion_numeric/, loss logs to results/training/.
    save_dir = DIFFUSION_NUMERIC_DIR
    results_dir = TRAINING_RESULTS_DIR / MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LiteVAE
    litevae_checkpoint_path = LITEVAE_MODEL


def load_image_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    genotype_to_images = {}
    for _, row in df.iterrows():
        genotype = row['genotype']
        image_file = row['new_filename']
        if genotype not in genotype_to_images:
            genotype_to_images[genotype] = []
        genotype_to_images[genotype].append(image_file)
    return genotype_to_images, df

def load_litevae(checkpoint_path, device='cuda'):

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create LiteVAE models
    encoder = LiteVAEEncoder(
        in_channels=3,
        latent_channels=4,
        feature_channels=64,
        num_blocks=3
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()
    
    decoder = LiteVAEDecoder(
        latent_channels=4,
        output_channels=3,
        base_channels=512,
        num_res_blocks=2
    )
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.to(device)
    decoder.eval()
    
    print("LiteVAE loaded successfully")
    return encoder, decoder


class MaizeRootDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, metadata_path, snp_parquet_path, 
                 transform=None, sample_names=None, snp_matrix=None):
        
        self.image_dir = image_dir
        self.transform = transform
        
        # Load SNP data
        if snp_matrix is None:
            self.sample_names, self.snp_names, self.snp_matrix = load_snp_data_from_parquet(snp_parquet_path)
        else:
            self.sample_names = sample_names
            self.snp_matrix = snp_matrix
            self.snp_names = None
        
        self.genotype_to_images, _ = load_image_metadata(metadata_path)
        self.sample_to_idx = {name: i for i, name in enumerate(self.sample_names)}
        
        self.samples = []
        for genotype, images in self.genotype_to_images.items():
            if genotype not in self.sample_to_idx:
                continue
            idx = self.sample_to_idx[genotype]
            snp_vector = self.snp_matrix[idx]
            for image_file in images:
                image_path = os.path.join(image_dir, image_file)
                if os.path.exists(image_path):
                    self.samples.append({
                        'image_path': image_path,
                        'genotype': genotype,
                        'snp_vector': snp_vector,
                    })
        
        print(f"Dataset: {len(self.samples)} images, {len(self.sample_names)} genotypes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        snp = torch.tensor(sample['snp_vector'], dtype=torch.float32)
        return {
            'image': image,
            'snp': snp,
            'genotype': sample['genotype'],
        }


def create_dataloaders(config):
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(config.snp_parquet_path)
    
    np.random.seed(42)
    np.random.shuffle(sample_names)
    split_idx = int(len(sample_names) * 0.8)
    train_samples = sample_names[:split_idx]
    val_samples = sample_names[split_idx:]
    
    print(f"Train: {len(train_samples)} genotypes, Val: {len(val_samples)} genotypes")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset = MaizeRootDataset(
        image_dir=config.image_dir,
        metadata_path=config.metadata_path,
        snp_parquet_path=config.snp_parquet_path,
        transform=train_transform,
        sample_names=train_samples,
        snp_matrix=snp_matrix,
    )
    
    val_dataset = MaizeRootDataset(
        image_dir=config.image_dir,
        metadata_path=config.metadata_path,
        snp_parquet_path=config.snp_parquet_path,
        transform=val_transform,
        sample_names=val_samples,
        snp_matrix=snp_matrix,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, snp_matrix, sample_names

def main():
    config = Config()
    print(f"Device: {config.device}")
    
    # Create dataloaders
    train_loader, val_loader, snp_matrix, sample_names = create_dataloaders(config)
    
    # Create components
    snp_encoder, pca, n_components = build_snp_encoder_with_optimal_pca(
        snp_matrix=snp_matrix,
        target_variance=0.95,  # Standard for genetic data
        embedding_dim=config.snp_embed_dim,
        num_tokens=config.num_tokens,
        hidden_dim=1024,
    )
    snp_encoder.to(config.device)
    
    unet = DenoisingUNet(
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
        snp_embed_dim=config.snp_embed_dim,
        d_attention=config.d_attention,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
    )
    unet.to(config.device)
    
    scheduler = DiffusionScheduler(
        num_steps=config.num_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    
    litevae_encoder, litevae_decoder = load_litevae(
        checkpoint_path=config.litevae_checkpoint_path, device=config.device
    )

    # Create LDM 
    ldm = LatentDiffusionModel(
        litevae_encoder=litevae_encoder,
        litevae_decoder=litevae_decoder,
        snp_encoder=snp_encoder,
        unet=unet,
        scheduler=scheduler,
        device=config.device,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in ldm.parameters())
    trainable_params = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer (only trainable components)
    optimizer = optim.Adam(
        list(ldm.snp_encoder.parameters()) + list(ldm.unet.parameters()),
        lr=config.learning_rate,
    )
    
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    
    # Training loop
    for epoch in range(config.num_epochs):
        ldm.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(config.device)
            snp = batch['snp'].to(config.device)
            
            loss = ldm(images, snp)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ldm.unet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(ldm.snp_encoder.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler_lr.step()
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler_lr.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_dir = resolve_output(config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': unet.state_dict(),
                'snp_encoder_state_dict': snp_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_dir / f'{MODEL_NAME}_epoch_{epoch + 1}.pt')

    print(f"Training complete. Checkpoints in {resolve_output(config.save_dir)}")


if __name__ == "__main__":
    main()