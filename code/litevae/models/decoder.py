import torch
import torch.nn as nn
import os

from .blocks import DecoderBlock, ResidualBlock, LiteVAEUNetBlock, SMCResidualBlock
from ..utils.visualization import save_decoder_step, save_decoder_channels


class LiteVAEDecoder(nn.Module):

    def __init__(self, 
                 latent_channels=4,      # Input latent channels (n_z)
                 output_channels=3,      # RGB output
                 base_channels=512,      # Starting channel count
                 num_res_blocks=2):      # Residual blocks per stage
        
        super().__init__()
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        
        # Processes the latent code to produce initial feature maps for the decoder
        # The parameters learn how to expand the latent representation into a richer feature space of [base_channels] channels
        # Keeps the same spatial dimensions (32×32) while increasing the channel depth
        # Uses a UNet-based block with GroupNorm for consistent feature processing with the encoder
        self.init_conv = LiteVAEUNetBlock(latent_channels, base_channels, num_res_blocks=2)
        
        # Decoder stages: doubles spatial size, halves channel count, and applies residual blocks
        # Stage 1: 32×32 → 64×64
        self.stage1 = DecoderBlock(base_channels, base_channels // 2, num_res_blocks)
        
        # Stage 2: 64×64 → 128×128
        self.stage2 = DecoderBlock(base_channels // 2, base_channels // 4, num_res_blocks)
        
        # Stage 3: 128×128 → 256×256
        self.stage3 = DecoderBlock(base_channels // 4, base_channels // 8, num_res_blocks)
        
        # Final output layer
        # Converts the final feature maps to RGB output with a Tanh activation to ensure pixel values are in the range [-1, 1]
        # Normalizing the output to [-1, 1] helps the model learn better and ensures consistency with the input image normalization
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels // 8, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, save_steps=True, save_dir="litevae_output"):
        
        # Create decoder output directory
        if save_steps:
            os.makedirs(f"{save_dir}/decoder", exist_ok=True)
            os.makedirs(f"{save_dir}/decoder/channels", exist_ok=True)
        
        # [B, 4, 32, 32] → [B, 512, 32, 32]
        x = self.init_conv(z)
        
        if save_steps:
            save_decoder_step(x, "01_initial_features_32x32", save_dir)
            save_decoder_channels(x, "01_initial_features", save_dir)
        
        # [B, 512, 32, 32] → [B, 256, 64, 64]
        x = self.stage1(x)
        
        if save_steps:
            save_decoder_step(x, "02_stage1_64x64", save_dir)
            save_decoder_channels(x, "02_stage1_features", save_dir)
        
        # [B, 256, 64, 64] → [B, 128, 128, 128]
        x = self.stage2(x)
        
        if save_steps:
            save_decoder_step(x, "03_stage2_128x128", save_dir)
            save_decoder_channels(x, "03_stage2_features", save_dir)
        
        # [B, 128, 128, 128] → [B, 64, 256, 256]
        x = self.stage3(x)
        
        if save_steps:
            save_decoder_step(x, "04_stage3_256x256", save_dir)
            save_decoder_channels(x, "04_stage3_features", save_dir)
        
        # [B, 64, 256, 256] → [B, 3, 256, 256]
        output = self.final_conv(x)
        
        if save_steps:
            save_decoder_step(output, "05_final_output_RGB", save_dir, is_rgb=True)
        
        return output