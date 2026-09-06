# Reconstruction metrics on a folder of test images: PSNR, SSIM and LPIPS.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS

import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import DATASET_DIR, LITEVAE_MODEL, RESULTS_DIR

# Import your LiteVAE models
from litevae.models import LiteVAEEncoder, LiteVAEDecoder


class LiteVAETester:
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create models
        self.encoder = LiteVAEEncoder(
            in_channels=3,
            latent_channels=4,
            feature_channels=64,
            num_blocks=3
        )
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.to(device)
        self.encoder.eval()
        
        self.decoder = LiteVAEDecoder(
            latent_channels=4,
            output_channels=3,
            base_channels=512,
            num_res_blocks=2
        )
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.decoder.to(device)
        self.decoder.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(f"  Decoder parameters: {sum(p.numel() for p in self.decoder.parameters()):,}")
        
        # LPIPS for perceptual similarity
        self.lpips = LPIPS(net='alex').to(device)
        self.lpips.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        self.transform_no_norm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def load_images_from_folder(self, folder_path, extensions=['.png', '.jpg', '.jpeg']):
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
        
        # Remove duplicates
        image_paths = list(set(image_paths))
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        return sorted(image_paths)
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        img_original = self.transform_no_norm(img).unsqueeze(0).to(self.device)
        return img_tensor, img_original
    
    def reconstruct_image(self, image_tensor):
        with torch.no_grad():
            # Encoder returns (z, z_mean, z_logvar)
            # We only need z for reconstruction
            z, z_mean, z_logvar = self.encoder(image_tensor, save_steps=False)
            reconstructed = self.decoder(z, save_steps=False)
        return reconstructed, z
    
    def compute_metrics(self, original, reconstructed):
        # Convert to numpy for skimage metrics
        orig_np = original[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        recon_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
        
        # Clip to [0, 1] for metrics
        orig_np = np.clip(orig_np, 0, 1)
        recon_np = np.clip(recon_np, 0, 1)
        
        # PSNR
        psnr = peak_signal_noise_ratio(orig_np, recon_np, data_range=1.0)
        
        # SSIM
        ssim = structural_similarity(orig_np, recon_np, channel_axis=2, data_range=1.0)
        
        # LPIPS (perceptual similarity)
        with torch.no_grad():
            # LPIPS expects inputs in [-1, 1] range
            orig_lpips = original * 2 - 1  # [0,1] → [-1,1]
            recon_lpips = reconstructed * 2 - 1
            lpips_score = self.lpips(orig_lpips, recon_lpips).item()
        
        # MSE
        mse = torch.nn.functional.mse_loss(reconstructed, original).item()
        
        # L1
        l1 = torch.nn.functional.l1_loss(reconstructed, original).item()
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_score,
            'mse': mse,
            'l1': l1,
        }
    
    def test_folder(self, folder_path, output_dir='test_results'):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reconstructions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
        
        # Load images
        image_paths = self.load_images_from_folder(folder_path)
        
        if len(image_paths) == 0:
            print("No images found!")
            return {}
        
        results = {}
        all_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'mse': [], 'l1': []}
        
        print(f"\nTesting {len(image_paths)} images...")
        
        for i, img_path in enumerate(tqdm(image_paths)):
            # Get image name
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Load and reconstruct
            img_tensor, img_original = self.preprocess_image(img_path)
            recon_tensor, latent = self.reconstruct_image(img_tensor)
            
            # Denormalize for saving
            img_display = (img_tensor + 1) / 2
            recon_display = (recon_tensor + 1) / 2
            
            # Compute metrics
            metrics = self.compute_metrics(img_display, recon_display)
            results[img_name] = metrics
            
            # Accumulate metrics
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # Save reconstructed image
            save_path = os.path.join(output_dir, 'reconstructions', f'{img_name}_recon.png')
            save_image(recon_display, save_path)
            
            # Save comparison (original, reconstruction, difference)
            self.save_comparison(img_display, recon_display, img_name, output_dir)
            
            # Save latent code visualization (optional)
            self.save_latent_visualization(latent, img_name, output_dir)
        
        # Compute average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        std_metrics = {key: np.std(values) for key, values in all_metrics.items()}
        
        # Save results
        self.save_results(results, avg_metrics, std_metrics, output_dir)
        
        # Print summary
        print("TEST RESULTS SUMMARY")
        print(f"Images tested: {len(image_paths)}")
        print(f"\nAverage Metrics:")
        print(f"  PSNR:  {avg_metrics['psnr']:.2f} dB (±{std_metrics['psnr']:.2f})")
        print(f"  SSIM:  {avg_metrics['ssim']:.4f} (±{std_metrics['ssim']:.4f})")
        print(f"  LPIPS: {avg_metrics['lpips']:.4f} (±{std_metrics['lpips']:.4f})")
        print(f"  MSE:   {avg_metrics['mse']:.6f} (±{std_metrics['mse']:.6f})")
        print(f"  L1:    {avg_metrics['l1']:.6f} (±{std_metrics['l1']:.6f})")
        print(f"\nResults saved to: {output_dir}/")
        
        return results, avg_metrics
    
    def save_comparison(self, original, reconstructed, img_name, output_dir):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        orig = original[0].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(np.clip(orig, 0, 1))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Reconstruction
        recon = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
        axes[1].imshow(np.clip(recon, 0, 1))
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')
        
        # Difference (amplified)
        diff = np.abs(orig - recon)
        axes[2].imshow(diff * 5)  # Amplify for visibility
        axes[2].set_title('Difference (5x)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparisons', f'{img_name}_comparison.png'), dpi=150)
        plt.close()
    
    def save_latent_visualization(self, latent, img_name, output_dir):
        latent_dir = os.path.join(output_dir, 'latent_vis')
        os.makedirs(latent_dir, exist_ok=True)
        
        z = latent[0].detach().cpu()  # [4, 32, 32]
        n_channels = z.shape[0]
        
        fig, axes = plt.subplots(1, n_channels, figsize=(n_channels * 2, 2))
        if n_channels == 1:
            axes = [axes]
        
        for i in range(n_channels):
            channel = z[i].numpy()
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            axes[i].imshow(channel, cmap='viridis')
            axes[i].set_title(f'Ch {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(latent_dir, f'{img_name}_latent.png'), dpi=150)
        plt.close()
    
    def save_results(self, results, avg_metrics, std_metrics, output_dir):
        import csv
        
        # Save per-image metrics
        csv_path = os.path.join(output_dir, 'metrics_per_image.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'psnr', 'ssim', 'lpips', 'mse', 'l1'])
            for img_name, metrics in results.items():
                writer.writerow([
                    img_name,
                    metrics['psnr'],
                    metrics['ssim'],
                    metrics['lpips'],
                    metrics['mse'],
                    metrics['l1'],
                ])
        
        # Save summary
        summary_path = os.path.join(output_dir, 'metrics_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("LiteVAE Test Results\n")
            f.write("="*60 + "\n")
            f.write(f"Images tested: {len(results)}\n\n")
            f.write("Average Metrics:\n")
            for key in avg_metrics:
                f.write(f"  {key}: {avg_metrics[key]:.4f} (±{std_metrics[key]:.4f})\n")


def main():
    # Edit these values, then run:
    #     python code/litevae/evaluation/test_litevae.py
    CHECKPOINT_PATH = LITEVAE_MODEL
    TEST_FOLDER = DATASET_DIR / 'test_reconstruction_images'
    OUTPUT_DIR = RESULTS_DIR / 'litevae_test_results'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    tester = LiteVAETester(CHECKPOINT_PATH, device=DEVICE)

    results, avg_metrics = tester.test_folder(
        folder_path=TEST_FOLDER,
        output_dir=OUTPUT_DIR
    )

    print("\nTesting complete")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()