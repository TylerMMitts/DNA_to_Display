import sys
from pathlib import Path

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from litevae.models import LiteVAEEncoder, LiteVAEDecoder


def get_root_augmentation_transforms(image_size=256, augment=True):

    # Base transforms that are always applied
    base_transforms = [
        # Resizes to 256x256 for LiteVAE input
        transforms.Resize((image_size, image_size)),
        # Converts PIL image to tensor and normalizes to [-1, 1]
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    if augment:
        augmentation = [
            # 50% chance to flip horizontally, simulating different orientations of roots
            transforms.RandomHorizontalFlip(p=0.5),

            # 20% chance to flip vertically, simulating roots that may be upside down
            transforms.RandomVerticalFlip(p=0.5),

            # Adds small random rotations to simulate slight misalignments in imaging
            transforms.RandomRotation(degrees=15),
            
            # Color jittering to simulate variations in lighting and camera settings
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1, 
                hue=0.05
            ),

            # 20% chance to apply Gaussian blur, simulating slight out-of-focus or motion blur
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.3))],
                p=0.2
            ),
            
        ]
        transform = transforms.Compose(augmentation + base_transforms)
    else:
        # No augmentation for validation/testing, only resizing and normalization
        transform = transforms.Compose(base_transforms)
    
    return transform


def create_dataloaders(data_path, batch_size=16, image_size=256, 
                       val_split=0.1, num_workers=4, augment=True):

    # batch_size - number of images processed in one forward/backward pass
    # num_workers - how many processes are used to load data in parallel
    # both increase memory usage

    # Get transforms
    train_transform = get_root_augmentation_transforms(image_size, augment=True)
    val_transform = get_root_augmentation_transforms(image_size, augment=False)
    
    # Load dataset
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path {data_path} is not a valid directory")
    
    # Load dataset from flat directory (all images directly in the folder)
    full_dataset = CustomImageDataset(data_path, transform=train_transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply validation transform
    if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'transform'):
        val_dataset.dataset.transform = val_transform
    else:
        pass
    
    # Create dataloaders
    # DataLoader handles batching, shuffling, and parallel loading of data
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        'total_images': dataset_size,
        'train_size': train_size,
        'val_size': val_size,
    }
    
    return train_loader, val_loader, dataset_info


class CustomImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Find all image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for root, _, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        if len(self.image_paths) == 0:
            print(f"No images found in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class LiteVAELoss(nn.Module):
    
    def __init__(self, recon_weight=1.0, kl_weight=0.001):

        # recon_weight - how much to weight the reconstruction loss (MSE)
        # kl_weight - how much to weight the KL divergence loss

        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

        # Creates a mean squared error loss function for reconstruction
        # MSE = (1/N) * sum((x - x_hat)^2), where x is original and x_hat is reconstructed and n is the number of elements
        self.mse = nn.MSELoss()
    
    def forward(self, recon, original, z_mean, z_logvar):

        # Reconstruction loss
        recon_loss = self.mse(recon, original)
        
        # KL divergence measures how different two probability distributions are
        # In VAE, we want the latent distribution to be close to a standard normal distribution
        # This regularizes the latent space and prevents overfitting
        # KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(
            1 + z_logvar - z_mean.pow(2) - z_logvar.exp()
        ) / z_mean.numel()
        
        # Total loss
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


def get_latest_checkpoint(checkpoint_dir):

    if not os.path.exists(checkpoint_dir):
        return None, None
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if not checkpoint_files:
        return None, None
    
    # Extract epoch numbers and find the latest
    epoch_numbers = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_')[-1].split('.')[0])
            epoch_numbers.append((epoch, f))
        except:
            continue
    
    if not epoch_numbers:
        return None, None
    
    # Sort by epoch number and get the latest
    epoch_numbers.sort(key=lambda x: x[0])
    latest_epoch, latest_file = epoch_numbers[-1]
    
    return os.path.join(checkpoint_dir, latest_file), latest_epoch


def train_litevae(
    # Data parameters
    data_path="data/train",
    batch_size=16,
    image_size=256,
    val_split=0.1,
    num_workers=4,
    
    # Model parameters
    latent_channels=4,
    feature_channels=64,
    base_channels=512,
    num_blocks=3,
    num_res_blocks=2,
    
    # Training parameters
    num_epochs=100,
    learning_rate=1e-4,
    recon_weight=1.0,
    kl_weight=0.001,
    save_interval=5,
    
    # Output parameters
    save_dir="litevae_training",
    resume=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Latent channels: {latent_channels}")
    print(f"Data path: {data_path}")
    print(f"Save directory: {save_dir}")
    
    # Create save directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/reconstructions", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    
    # Check if data exists - exit if not found
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found")
        return None, None, None, None
    
    # Check if data directory is empty
    if not os.listdir(data_path):
        print(f"Data path {data_path} is empty")
        return None, None, None, None
    
    train_loader, val_loader, dataset_info = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        image_size=image_size,
        val_split=val_split,
        num_workers=num_workers,
        augment=True
    )
    
    print(f"Total images: {dataset_info['total_images']}")
    print(f"Train samples: {dataset_info['train_size']}")
    print(f"Val samples: {dataset_info['val_size']}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    start_epoch = 1
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []

    if resume:
        checkpoint_dir = f"{save_dir}/checkpoints"
        checkpoint_path, latest_epoch = get_latest_checkpoint(checkpoint_dir)

        if checkpoint_path is not None:

            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint['config']
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            recon_losses = checkpoint.get('recon_losses', [])
            kl_losses = checkpoint.get('kl_losses', [])

            print(f"Resuming from epoch {start_epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Train losses: {len(train_losses)} epochs")
            print(f"Val losses: {len(val_losses)} epochs")
            
            # Initialize models with saved config
            encoder = LiteVAEEncoder(
                in_channels=3,
                latent_channels=config['latent_channels'],
                feature_channels=config['feature_channels'],
                num_blocks=config['num_blocks']
            )
            
            decoder = LiteVAEDecoder(
                latent_channels=config['latent_channels'],
                output_channels=3,
                base_channels=config['base_channels'],
                num_res_blocks=config['num_res_blocks']
            )
            
            # Load weights
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            
            # Recreate optimizer
            optimizer = optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=config.get('learning_rate', learning_rate),
                betas=(0.9, 0.999),
                weight_decay=1e-5
            )
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Recreate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Put models in training mode
            encoder.train()
            decoder.train()
            
            print("\nSuccessfully resumed training")
            
        else:
            print("\nNo checkpoints found. Starting fresh training.")
            resume = False

    if not resume:
        encoder = LiteVAEEncoder(
            in_channels=3,
            latent_channels=latent_channels,
            feature_channels=feature_channels,
            num_blocks=num_blocks
        )
        
        decoder = LiteVAEDecoder(
            latent_channels=latent_channels,
            output_channels=3,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks
        )
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        # Count parameters
        enc_params = sum(p.numel() for p in encoder.parameters())
        dec_params = sum(p.numel() for p in decoder.parameters())
        total_params = enc_params + dec_params
        
        print(f"Encoder parameters: {enc_params:,}")
        print(f"Decoder parameters: {dec_params:,}")
        print(f"Total parameters: {total_params:,}")
        
        # Combine parameters for optimization
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    
    # Loss function
    criterion = LiteVAELoss(
        recon_weight=recon_weight,
        kl_weight=kl_weight
    )
    
    for epoch in range(start_epoch, num_epochs + 1):

        encoder.train()
        decoder.train()
        
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        # Progress bar for training loop in console
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(device)
            
            # Forward pass through encoder
            z, z_mean, z_logvar = encoder(images, save_steps=False)
            
            # Forward pass through decoder
            recon = decoder(z, save_steps=False)
            
            # Compute loss
            loss, recon_loss, kl_loss = criterion(recon, images, z_mean, z_logvar)
            
            # Backward pass
            # Zero gradients from previous step
            optimizer.zero_grad()
            # Compute gradients for current step
            loss.backward()
            
            # Gradient clipping
            # Scales down the gradients if their norm exceeds 1.0 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            
            # Update model parameters based on computed gradients
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        # Average training losses
        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon = train_recon / len(train_loader)
        avg_train_kl = train_kl / len(train_loader)
        
        train_losses.append(avg_train_loss)
        recon_losses.append(avg_train_recon)
        kl_losses.append(avg_train_kl)
        
        # Start validation loop
        encoder.eval()
        decoder.eval()
        
        val_loss = 0
        
        # Disables gradient computation for validation
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                
                z, z_mean, z_logvar = encoder(images, save_steps=False)

                recon = decoder(z, save_steps=False)
                
                # Computes loss
                # Doesn't reparameterize during validation so results are deterministic
                loss, _, _ = criterion(recon, images, z_mean, z_logvar)

                # Track validation loss
                val_loss += loss.item()
        
        # Average validation loss
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'recon_losses': recon_losses,
            'kl_losses': kl_losses,
            'config': {
                'latent_channels': latent_channels,
                'feature_channels': feature_channels,
                'base_channels': base_channels,
                'num_blocks': num_blocks,
                'num_res_blocks': num_res_blocks,
                'learning_rate': learning_rate,
            }
        }, f"{save_dir}/checkpoints/checkpoint_epoch_{epoch:03d}.pt")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'val_loss': avg_val_loss,
                'config': {
                    'latent_channels': latent_channels,
                    'feature_channels': feature_channels,
                    'base_channels': base_channels,
                    'num_blocks': num_blocks,
                    'num_res_blocks': num_res_blocks,
                }
            }, f"{save_dir}/checkpoints/best_model.pt")
            print(f"New best model saved (Val Loss: {avg_val_loss:.4f})")
        
        # Save reconstruction visualizations
        if epoch % save_interval == 0 or epoch == 1:
            with torch.no_grad():
                # Get a batch of validation images
                val_images = next(iter(val_loader))[:8].to(device)
                z_val, z_mean_val, z_logvar_val = encoder(val_images, save_steps=False)
                recon_val = decoder(z_val, save_steps=False)
                
                # Denormalize from [-1, 1] to [0, 1] for saving
                def denorm(img):
                    return (img + 1) / 2
                
                # Save comparison
                comparison = torch.cat([denorm(val_images), denorm(recon_val)])
                save_image(comparison, 
                          f"{save_dir}/reconstructions/epoch_{epoch:03d}_reconstructions.png",
                          nrow=8, normalize=False)
                
                print(f"Saved reconstruction visualization")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, label='Reconstruction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/logs/training_curves.png", dpi=150)
    plt.close()
    
    # Save loss history
    np.savez(f"{save_dir}/logs/loss_history.npz",
             train_losses=train_losses,
             val_losses=val_losses,
             recon_losses=recon_losses,
             kl_losses=kl_losses)
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}/checkpoints/")
    print(f"Visualizations saved to: {save_dir}/reconstructions/")
    print(f"Training logs saved to: {save_dir}/logs/")
    
    return encoder, decoder, train_losses, val_losses, best_val_loss


# Loads model for reconstruction or inference
def load_litevae_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize models
    encoder = LiteVAEEncoder(
        in_channels=3,
        latent_channels=config['latent_channels'],
        feature_channels=config['feature_channels'],
        num_blocks=config['num_blocks']
    )
    
    decoder = LiteVAEDecoder(
        latent_channels=config['latent_channels'],
        output_channels=3,
        base_channels=config['base_channels'],
        num_res_blocks=config['num_res_blocks']
    )
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder.eval()
    decoder.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return encoder, decoder, config


def reconstruct_image(image_path, encoder, decoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        z, z_mean, z_logvar = encoder(image_tensor, save_steps=False)
        recon = decoder(z, save_steps=False)
    
    # Denormalize
    def denorm(img):
        return (img + 1) / 2
    
    original = denorm(image_tensor).cpu().squeeze(0).permute(1, 2, 0).numpy()
    reconstructed = denorm(recon).cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Clip to valid range
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    return original, reconstructed


if __name__ == "__main__":
    
    # Data path - put your images here
    DATA_PATH = "results/cropped_images"
    
    # Training parameters
    BATCH_SIZE = 16
    IMAGE_SIZE = 256
    LATENT_CHANNELS = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # Model parameters - matches checkpoint_epoch_055 configuration
    FEATURE_CHANNELS = 64
    BASE_CHANNELS = 512
    NUM_BLOCKS = 3
    NUM_RES_BLOCKS = 2
    
    # Loss weights
    RECON_WEIGHT = 1.0
    KL_WEIGHT = 0.001
    
    # Train LiteVAE model
    encoder, decoder, train_losses, val_losses, best_val_loss = train_litevae(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        latent_channels=LATENT_CHANNELS,
        feature_channels=FEATURE_CHANNELS,
        base_channels=BASE_CHANNELS,
        num_blocks=NUM_BLOCKS,
        num_res_blocks=NUM_RES_BLOCKS,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        recon_weight=RECON_WEIGHT,
        kl_weight=KL_WEIGHT,
        save_dir="litevae_training",
        resume=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nTraining complete")
    print(f"Best validation loss: {best_val_loss:.4f}")