# DDIM sampling - turns a noise latent into an image latent in a chosen
# number of steps, far fewer than the 1000 used during training.

import torch
from tqdm import tqdm


class DDIMSampler:

    def __init__(self, model, scheduler, device='cuda'):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
    def sample(self, kinship, num_steps=50, guidance_scale=7.5, eta=0.0):
        
        # Finds how many rows within the input kinship matrix, indicating the number of samples to generate
        batch_size = kinship.shape[0]
        
        # Start from pure noise
        z_t = torch.randn(batch_size, 4, 32, 32, device=self.device)
        
        # Get timesteps
        timesteps = self.scheduler.get_timesteps(num_steps, self.device)
        
        # Encode kinship
        with torch.no_grad():
            snp_embedding = self.model.kinship_encoder(kinship)

        # Creates an embedding of all zeros, which is used for unconditional guidance
        # This is used to generate a sample without any conditioning information
        # then the difference between the unconditional and conditional predictions
        # are used to guide how much the model should follow the conditioning information (kinship) during sampling
        null_embedding = torch.zeros_like(snp_embedding)
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            t_prev_val = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            t_prev_batch = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred_cond = self.model.unet(z_t, t_batch, snp_embedding)
                noise_pred_uncond = self.model.unet(z_t, t_batch, null_embedding)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                z_t = self.scheduler.denoise_step(z_t, noise_pred, t_batch, t_prev_batch, eta)
        
        return z_t