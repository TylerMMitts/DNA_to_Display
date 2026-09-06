# Reconstruction quality on a folder of test images, plus a latent-averaging
# experiment that blends each image with its nearest neighbours in latent
# space.

import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from paths import DATASET_DIR, LITEVAE_MODEL, RESULTS_DIR

# Import directly from the model files (not through the package)
from litevae.models.encoder import LiteVAEEncoder
from litevae.models.decoder import LiteVAEDecoder

# Edit these before running.
CHECKPOINT_PATH = LITEVAE_MODEL
TEST_IMAGES_PATH = DATASET_DIR / 'test_reconstruction_images'
OUTPUT_DIR = RESULTS_DIR

# Device to use ('cuda' or 'cpu')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Target image size (must match training size)
IMAGE_SIZE = 256

# Latent averaging experiment parameters
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity (0-1) for latent averaging
TOP_K_LATENTS = 5          # Number of similar latents to consider for averaging


def load_model(checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # Load a trained LiteVAE model from checkpoint.
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
    print(f"Config: {config}")
    
    return encoder, decoder, config


def load_images_from_folder(folder_path: str, image_size: int = 256) -> list:
    # Load all images from a folder.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    images = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(folder_path, file)
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                images.append((img_path, image_tensor))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    # Denormalize from [-1, 1] to [0, 1].
    return (tensor + 1) / 2


def save_reconstruction_comparison(original: torch.Tensor, reconstructed: torch.Tensor, 
                                   save_path: str, filename: str):
    # Save a comparison image with original and reconstructed side by side.
    # Denormalize
    original_denorm = denormalize(original)
    reconstructed_denorm = denormalize(reconstructed)
    
    # Create comparison grid (1 row, 2 columns)
    comparison = torch.cat([original_denorm, reconstructed_denorm], dim=0)
    grid = make_grid(comparison, nrow=2, normalize=False)
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    save_path_full = os.path.join(save_path, f"{filename}_comparison.png")
    save_image(grid, save_path_full, normalize=False)
    return save_path_full


def save_latent_average_experiment(original1: torch.Tensor, original2: torch.Tensor,
                                   averaged_recon: torch.Tensor, save_path: str, 
                                   filename: str, similarity: float = None):
    # Save latent average experiment results.
    # Denormalize
    orig1_denorm = denormalize(original1)
    orig2_denorm = denormalize(original2)
    avg_recon_denorm = denormalize(averaged_recon)
    
    # Create grid (1 row, 3 columns)
    grid_images = torch.cat([orig1_denorm, orig2_denorm, avg_recon_denorm], dim=0)
    grid = make_grid(grid_images, nrow=3, normalize=False)
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    if similarity is not None:
        save_path_full = os.path.join(save_path, f"{filename}_sim{similarity:.3f}.png")
    else:
        save_path_full = os.path.join(save_path, f"{filename}.png")
    save_image(grid, save_path_full, normalize=False)
    return save_path_full


def find_similar_latents(encoded_images: list, 
                         target_idx: int, top_k: int = 5) -> list:
    # Find images with most similar latent codes to a target image.
    if len(encoded_images) <= 1:
        return []
    
    # A LiteVAE latent is [1, 4, 32, 32], so it is flattened to a single vector
    # before comparing. Dropping only the batch dimension would leave a [4, 32,
    # 32] tensor, and a cosine similarity across that returns one value per
    # spatial position rather than the single number this function is for.
    target_z = encoded_images[target_idx][2].reshape(-1)   # z_mean

    similarities = []

    for i, (_, _, z_mean) in enumerate(encoded_images):
        if i == target_idx:
            continue

        z_mean = z_mean.reshape(-1)
        sim = torch.dot(target_z, z_mean) / (
            torch.norm(target_z) * torch.norm(z_mean) + 1e-8)

        similarities.append((i, sim.item()))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k with similarity scores
    return similarities[:top_k]


def test_reconstructions():
    # Main function to test reconstructions and latent averaging experiments.
    # Uses config variables defined at the top of the file.
    print("LiteVAE Reconstruction Testing")
    
    # Every path comes from paths.py already anchored to the project root, so
    # there is nothing to join here.
    checkpoint_path = CHECKPOINT_PATH
    test_images_path = TEST_IMAGES_PATH
    output_dir = OUTPUT_DIR

    # Create output directories
    recon_output_dir = os.path.join(output_dir, "reconstruction_tests")
    latent_output_dir = os.path.join(output_dir, "latent_average_experiments")
    os.makedirs(recon_output_dir, exist_ok=True)
    os.makedirs(latent_output_dir, exist_ok=True)
    
    # Print paths for debugging
    print(f"\nCheckpoint path: {checkpoint_path}")
    print(f"Test images path: {test_images_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first or update CHECKPOINT_PATH")
        return
    
    # Load model
    print("\nLoading model...")
    encoder, decoder, config = load_model(checkpoint_path, DEVICE)
    
    # Load test images
    print(f"\nLoading test images from: {test_images_path}")
    if not os.path.exists(test_images_path):
        print(f"ERROR: Test images path {test_images_path} does not exist!")
        print("Please create the folder and add images, or update TEST_IMAGES_PATH")
        return
    
    test_images = load_images_from_folder(test_images_path, IMAGE_SIZE)
    if not test_images:
        print(f"ERROR: No images found in {test_images_path}")
        print(f"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Process each image for reconstruction
    print("Running Reconstruction Tests")
    
    encoded_data = []
    reconstruction_results = []
    
    for idx, (img_path, img_tensor) in enumerate(tqdm(test_images, desc="Processing images")):
        # Move to device
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Encode
        with torch.no_grad():
            z, z_mean, z_logvar = encoder(img_tensor, save_steps=False)
        
        # Store encoded data for later
        filename = Path(img_path).stem
        
        # Store z and z_mean (detach and move to CPU for later use)
        encoded_data.append((filename, z.detach().cpu(), z_mean.detach().cpu()))
        reconstruction_results.append((filename, img_tensor.detach().cpu()))
    
    # Now decode all images for reconstruction
    print("\nGenerating reconstructions...")
    for idx, (filename, img_tensor) in enumerate(tqdm(reconstruction_results, desc="Decoding")):
        # Move to device for decoding
        img_tensor_device = img_tensor.to(DEVICE)
        z, z_mean, z_logvar = encoder(img_tensor_device, save_steps=False)
        recon = decoder(z, save_steps=False)
        
        # Save reconstruction comparison
        save_reconstruction_comparison(
            img_tensor,
            recon.cpu(),
            recon_output_dir,
            filename
        )
        
        # Update reconstruction_results with the decoded image
        reconstruction_results[idx] = (filename, img_tensor, recon.cpu())
    
    print(f"\nReconstruction comparisons saved to: {recon_output_dir}")
    
    # Process latent averaging experiments
    print("Running Latent Averaging Experiments")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Top K latents: {TOP_K_LATENTS}")
    
    # For each image, find similar latents and average them
    latent_experiments_count = 0
    all_similarities = []
    
    for idx in tqdm(range(len(encoded_data)), desc="Latent averaging experiments"):
        # Find similar latents
        similar_latents = find_similar_latents(encoded_data, idx, top_k=TOP_K_LATENTS)
        
        if not similar_latents:
            continue
        
        # Get the target image data
        target_filename, target_z, target_z_mean = encoded_data[idx]
        target_tensor = reconstruction_results[idx][1]  # Original image
        
        # For each similar image, perform latent averaging
        for similar_idx, similarity in similar_latents:
            # Skip if similarity is below threshold
            if similarity < SIMILARITY_THRESHOLD:
                continue
            
            similar_filename, similar_z, similar_z_mean = encoded_data[similar_idx]
            similar_tensor = reconstruction_results[similar_idx][1]  # Original image
            
            # Average the latent codes (z, not z_mean)
            avg_z = (target_z + similar_z) / 2
            
            # Move to device for decoding
            avg_z = avg_z.to(DEVICE)
            
            # Decode the averaged latent
            with torch.no_grad():
                avg_recon = decoder(avg_z, save_steps=False)
            
            # Save the latent average experiment
            experiment_name = f"{target_filename}_avg_{similar_filename}"
            save_latent_average_experiment(
                target_tensor,
                similar_tensor,
                avg_recon.cpu(),
                latent_output_dir,
                experiment_name,
                similarity
            )
            
            all_similarities.append(similarity)
            latent_experiments_count += 1
    
    print(f"\nLatent averaging experiments saved to: {latent_output_dir}")
    print(f"Generated {latent_experiments_count} latent averaging experiments")
    
    if all_similarities:
        print(f"Average similarity: {np.mean(all_similarities):.3f}")
        print(f"Min similarity: {np.min(all_similarities):.3f}")
        print(f"Max similarity: {np.max(all_similarities):.3f}")
    
    # Generate summary report
    generate_summary_report(reconstruction_results, encoded_data, 
                           recon_output_dir, latent_output_dir,
                           latent_experiments_count, all_similarities,
                           checkpoint_path, test_images_path)
    
    print("Testing Complete!")
    print(f"Total images tested: {len(test_images)}")
    print(f"Reconstruction comparisons: {len(test_images)}")
    print(f"Latent averaging experiments: {latent_experiments_count}")
    print(f"Results saved to: {output_dir}")


def generate_summary_report(reconstruction_results, encoded_data, 
                           recon_dir, latent_dir,
                           num_experiments, similarities,
                           checkpoint_path, test_images_path):
    # Generate a summary report with statistics and visualizations.
    report_path = os.path.join(Path(recon_dir).parent, "test_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("LiteVAE Test Summary Report\n\n")

        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Test images path: {test_images_path}\n")
        f.write(f"Number of images tested: {len(reconstruction_results)}\n")
        f.write(f"Reconstruction directory: {recon_dir}\n")
        f.write(f"Latent experiment directory: {latent_dir}\n")
        f.write(f"Number of latent experiments: {num_experiments}\n")
        f.write(f"Similarity threshold: {SIMILARITY_THRESHOLD}\n")
        f.write(f"Top K latents: {TOP_K_LATENTS}\n")
        
        if similarities:
            f.write(f"\nLatent Averaging Statistics:\n")
            f.write(f"  Average similarity: {np.mean(similarities):.3f}\n")
            f.write(f"  Min similarity: {np.min(similarities):.3f}\n")
            f.write(f"  Max similarity: {np.max(similarities):.3f}\n")
            f.write(f"  Std similarity: {np.std(similarities):.3f}\n")
        
        f.write("\nTested Images:\n")
        for idx, (filename, _, _) in enumerate(reconstruction_results, 1):
            f.write(f"{idx}. {filename}\n")

        f.write("\nEnd of Report\n")
    
    print(f"Summary report saved to: {report_path}")


# Main execution

if __name__ == "__main__":
    test_reconstructions()