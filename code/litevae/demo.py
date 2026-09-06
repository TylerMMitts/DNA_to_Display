# Runs one image through the encoder and decoder, saving every intermediate
# stage.
#
# Uses freshly initialised weights, so the reconstruction is meaningless - the
# point is to see the shape of each stage, not the output quality.

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import PROJECT_ROOT, RESULTS_DIR

from litevae.models import LiteVAEEncoder, LiteVAEDecoder
from litevae.utils.visualization import save_comparison

# Untrained weights, so this only shows the shape of each stage, not a good
# reconstruction. Its outputs go under results/ like every other script's.
DEMO_OUTPUT_DIR = RESULTS_DIR / 'litevae_output'

# Prepares the image for the model: resize, convert to tensor, normalize
def load_and_preprocess_image(image_path, target_size=256):
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def run_litevae_demo(image_path, save_dir=DEMO_OUTPUT_DIR):
    x = load_and_preprocess_image(image_path)
    
    # Initialize encoder and decoder
    encoder = LiteVAEEncoder(
        in_channels=3,
        latent_channels=4,
        feature_channels=64,
        num_blocks=3
    )
    
    decoder = LiteVAEDecoder(
        latent_channels=4,
        output_channels=3,
        base_channels=512,
        num_res_blocks=2
    )
    
    # Put in evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Run encoder. It returns three tensors, and only the sampled latent goes
    # on to the decoder - passing the whole tuple through reaches conv2d as a
    # tuple and fails there.
    with torch.no_grad():
        z, z_mean, z_logvar = encoder(x, save_steps=True, save_dir=save_dir)

    # Run decoder
    with torch.no_grad():
        recon = decoder(z, save_steps=True, save_dir=save_dir)
    
    # Save comparison
    save_comparison(x, recon, save_dir)
    
    return z, recon


if __name__ == "__main__":
    image_path = PROJECT_ROOT / 'image.JPG'
    z, recon = run_litevae_demo(image_path, DEMO_OUTPUT_DIR)

    print(f"Processing complete. Output in {DEMO_OUTPUT_DIR}")