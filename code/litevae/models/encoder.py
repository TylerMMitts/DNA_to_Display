# The LiteVAE encoder: wavelet decomposition, per-level feature extraction,
# then aggregation into a 4x32x32 latent.

import torch
import torch.nn as nn
import torch.nn.functional as F

from paths import RESULTS_DIR

from .blocks import ResidualBlock, LiteVAEUNetBlock
from ..utils.wavelet import dwt_2d
from ..utils.visualization import (
    save_wavelet_bands, save_feature_maps, save_latent_code,
)

# Where save_steps=True writes its intermediate stages when the caller does not
# name a folder. Absolute, so it never lands next to whatever launched the run.
DEFAULT_STEP_DIR = RESULTS_DIR / 'litevae_output'


class LiteVAEEncoder(nn.Module):
    
    def __init__(self, 
                 in_channels=3,          # RGB input
                 latent_channels=4,      # n_z (4 or 12) 
                 feature_channels=64,    # C (number of channels in feature maps)
                 num_blocks=3):          # Number of residual blocks per level
        
        super().__init__()
        self.latent_channels = latent_channels
        self.feature_channels = feature_channels
        self.in_channels = in_channels
        
        # Creates feature maps for each level of wavelet decomposition
        # Input is 4 sub-bands (LL, LH, HL, HH) * 3 channels = 12 channels
        self.f1 = LiteVAEUNetBlock(in_channels * 4, feature_channels, num_res_blocks=num_blocks)
        self.f2 = LiteVAEUNetBlock(in_channels * 4, feature_channels, num_res_blocks=num_blocks)
        self.f3 = LiteVAEUNetBlock(in_channels * 4, feature_channels, num_res_blocks=num_blocks)

        # Scale for latent code, learns which channels are more important for reconstruction. Initialized to 1 for all channels.
        self.latent_scale = nn.Parameter(torch.ones(latent_channels, 1, 1))
        
        # Combines the feature maps from all levels into the latent code through aggregation
        # Uses a UNet-based block instead of the simple sequential convolution stack
        self.agg = LiteVAEUNetBlock(feature_channels * 3, latent_channels * 2, num_res_blocks=2)
        
    def normalize_wavelet_coefficients(self, LL, LH, HL, HH, level):
        # Normalizes wavelet coefficients to keep values in [-1, 1] range
        scale = 2 ** level
        return LL / scale, LH / scale, HL / scale, HH / scale
    
    def reparameterize(self, z_mean, z_logvar):
        # Reparameterization is what makes VAE a variational autoencoder and not just a regular autoencoder
        # It enables gradient flow through random sampling (adjusting parameters of distibution to minimize error)
        # Gets the standard deviation from the log variance
        std = torch.exp(0.5 * z_logvar)
        # Generates random noise from a standard normal distribution with the same shape as std
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def forward(self, x, save_steps=True, save_dir=DEFAULT_STEP_DIR):
        
        # batch_size, channels, height, width = x.shape
        B, C, H, W = x.shape
        
        # If input image is not 256×256, resize it to 256×256 for the encoder
        if H != 256 or W != 256:
            # Uses bilinear interpolation to resize the image to 256×256 while preserving the aspect ratio and minimizing distortion
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            B, C, H, W = x.shape
        
        # Ensure the input has the expected number of channels
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # Generates wavelet sub-bands for the input image
        LL1, LH1, HL1, HH1 = dwt_2d(x)      # 256×256 → 128×128
        LL2, LH2, HL2, HH2 = dwt_2d(LL1)     # 128×128 → 64×64
        LL3, LH3, HL3, HH3 = dwt_2d(LL2)     # 64×64 → 32×32
        
        # Normalizes wavelet coefficients to keep values in [-1, 1] range
        # This is a critical step in the official LiteVAE implementation
        LL1, LH1, HL1, HH1 = self.normalize_wavelet_coefficients(LL1, LH1, HL1, HH1, level=1)
        LL2, LH2, HL2, HH2 = self.normalize_wavelet_coefficients(LL2, LH2, HL2, HH2, level=2)
        LL3, LH3, HL3, HH3 = self.normalize_wavelet_coefficients(LL3, LH3, HL3, HH3, level=3)
        
        # Save wavelet sub-bands
        if save_steps:
            save_wavelet_bands(x, LL1, LH1, HL1, HH1, LL2, LH2, HL2, HH2, LL3, LH3, HL3, HH3, save_dir)
        
        # Generates feature maps for each level of wavelet decomposition using UNet-based blocks
        input1 = torch.cat([LL1, LH1, HL1, HH1], dim=1)
        F1 = self.f1(input1)  # [B, 64, 128, 128]
        
        input2 = torch.cat([LL2, LH2, HL2, HH2], dim=1)
        F2 = self.f2(input2)  # [B, 64, 64, 64]
        
        input3 = torch.cat([LL3, LH3, HL3, HH3], dim=1)
        F3 = self.f3(input3)  # [B, 64, 32, 32]
        
        # Save feature maps
        if save_steps:
            save_feature_maps(F1, F2, F3, save_dir)
        
        # Compresses the feature maps from all levels to 32×32 spatial dimensions for aggregation
        F1_aligned = F.interpolate(F1, size=(32, 32), mode='bilinear', align_corners=False)
        F2_aligned = F.interpolate(F2, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension
        F_combined = torch.cat([F1_aligned, F2_aligned, F3], dim=1)
        
        # Combines the feature maps from all levels into the latent code through aggregation
        # Uses a UNet-based block instead of the simple sequential convolution stack
        z_params = self.agg(F_combined)

        # Splits the latent code into mean and log variance for reparameterization
        z_mean, z_logvar = torch.chunk(z_params, 2, dim=1)

        # Applies the learned latent scale to the latent code
        z_mean = z_mean * self.latent_scale.view(1, -1, 1, 1)

        z = self.reparameterize(z_mean, z_logvar)
        
        # Save latent code
        if save_steps:
            save_latent_code(z, save_dir)
        
        return z, z_mean, z_logvar