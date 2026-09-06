# Ties the pieces together for training: LiteVAE encodes the image to a
# latent, the scheduler noises it, and the UNet predicts that noise
# conditioned on the genotype. LiteVAE stays frozen throughout.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DenoisingUNet
from .snp_encoder import SNPEncoder


class LatentDiffusionModel(nn.Module):
    
    def __init__(self,
                 litevae_encoder=None,
                 litevae_decoder=None,
                 snp_encoder=None,
                 unet=None,
                 scheduler=None,
                 device='cuda'):
        
        super().__init__()
        
        # LiteVAE components (pretrained, frozen)
        self.litevae_encoder = litevae_encoder
        self.litevae_decoder = litevae_decoder
        
        # Trainable components
        self.snp_encoder = snp_encoder
        self.unet = unet
        
        # Scheduler
        self.scheduler = scheduler
        
        self.device = device
        
        # Freeze LiteVAE
        if self.litevae_encoder is not None:
            for param in self.litevae_encoder.parameters():
                param.requires_grad = False
        if self.litevae_decoder is not None:
            for param in self.litevae_decoder.parameters():
                param.requires_grad = False
        
        # Move to device
        self.to(device)
        
    def encode_image(self, image):
        # Encode image to latent using LiteVAE
        if self.litevae_encoder is None:
            raise ValueError("LiteVAE encoder not provided")
        with torch.no_grad():
            result = self.litevae_encoder(image, save_steps=False)
            if isinstance(result, tuple):
                return result[0]
            return result
    
    def decode_latent(self, z):
        # Decode latent to image using LiteVAE
        if self.litevae_decoder is None:
            raise ValueError("LiteVAE decoder not provided")
        with torch.no_grad():
            return self.litevae_decoder(z, save_steps=False)
    
    def forward(self, image, snp):

        # Encode image to latent
        z = self.encode_image(image)
        
        # Sample random timestep
        batch_size = z.shape[0]
        t = torch.randint(0, self.scheduler.num_steps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(z)
        z_t = self.scheduler.add_noise(z, noise, t)
        
        # Encode snp
        snp_embedding = self.snp_encoder(snp)
        
        # Predict noise
        noise_pred = self.unet(z_t, t, snp_embedding)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def generate(self, snp, num_steps=50, guidance_scale=7.5):

        # Set model to evaluation mode
        self.eval()
        
        batch_size = snp.shape[0]
        
        # Start from pure noise
        z_t = torch.randn(batch_size, 4, 32, 32, device=self.device)
        
        # Get timesteps
        timesteps = self.scheduler.get_timesteps(num_steps, self.device)
        
        # Encode SNP
        with torch.no_grad():
            snp_embedding = self.snp_encoder(snp)
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            t_prev_val = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            t_prev_batch = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred = self.unet(z_t, t_batch, snp_embedding)
                z_t = self.scheduler.denoise_step(z_t, noise_pred, t_batch, t_prev_batch)
        
        # Decode to image
        images = self.decode_latent(z_t)
        # Denormalize images from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        return images
    
    def get_attention_weights(self, snp, num_steps=1, clear_history=True):

        self.eval()
        
        # Enable attention storage in UNet
        self.unet.store_attention = True
        
        # Clear previous history if requested
        if clear_history:
            self.unet.clear_attention_history()
        
        with torch.no_grad():
            # Encode snp
            snp_embedding = self.snp_encoder(snp)
            
            # Start from pure noise
            z_t = torch.randn(snp.shape[0], 4, 32, 32, device=self.device)
            
            # Get timesteps for sampling
            timesteps = self.scheduler.get_timesteps(num_steps, self.device)
            
            # Store attention at each step
            attention_by_step = []
            
            for step_idx, t in enumerate(timesteps):
                t_batch = torch.full((snp.shape[0],), t, device=self.device, dtype=torch.long)

                t_prev_val = int(timesteps[step_idx + 1].item()) if step_idx + 1 < len(timesteps) else -1
                t_prev_batch = torch.full((snp.shape[0],), t_prev_val, device=self.device, dtype=torch.long)

                noise_pred = self.unet(z_t, t_batch, snp_embedding)
                step_attention = self.unet.get_all_attention_weights()

                for layer_name, attn in step_attention.items():
                    attention_by_step.append({...})

                z_t = self.scheduler.denoise_step(z_t, noise_pred, t_batch, t_prev_batch)
            
            # Disable attention storage
            self.unet.store_attention = False
            
            return attention_by_step
    
    def get_attention_for_single_forward(self, snp):

        self.eval()
        
        # Enable attention storage
        self.unet.store_attention = True
        self.unet.clear_attention_history()
        
        with torch.no_grad():
            # Encode SNP
            snp_embedding = self.snp_encoder(snp)
            
            # Start from pure noise (or use a specific noisy latent)
            z_t = torch.randn(snp.shape[0], 4, 32, 32, device=self.device)
            
            # Use a fixed timestep (500)
            t = torch.full((snp.shape[0],), 500, device=self.device, dtype=torch.long)
            
            # Forward pass
            noise_pred = self.unet(z_t, t, snp_embedding)
            
            # Collect attention
            attention_weights = self.unet.get_all_attention_weights()
            
            # Move to CPU
            for key in attention_weights:
                attention_weights[key] = attention_weights[key].cpu()
            
            # Disable attention storage
            self.unet.store_attention = False
            
            return attention_weights