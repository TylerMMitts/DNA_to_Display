# The noise schedule: how much noise is added at each timestep, and the
# coefficients used to step back out again during sampling.

import torch


class DiffusionScheduler:

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        
        # Creates a linear schedule of betas from beta_start to beta_end (from not noisy to very noisy)
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        
        # Precompute alpha values (the amount of the original image is preserved at each step) and their cumulative product (alpha_bar)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, z, noise, t):
        # Gets alpha_bar value and reshapes it to match the dimensions of z for broadcasting
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        # Adds noise to the latent z according to the diffusion process formula
        z_t = torch.sqrt(alpha_bar) * z + torch.sqrt(1 - alpha_bar) * noise
        return z_t
    
    # This function implements the denoising step of the DDIM algorithm, which predicts the previous latent z_{t-1} from the current latent z_t and the predicted noise
    # This is used after the model is already trained and we want to generate new images from noise
    def denoise_step(self, z_t, noise_pred, t, t_prev=None, eta=0.0):

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        # Only valid when sampling every single timestep (num_steps == self.num_steps).
        # Any strided/accelerated schedule must pass t_prev explicitly.
        if t_prev is None:
            t_prev = t - 1

        # alpha_bar is defined as 1.0 "before" step 0, i.e. the fully denoised sample.
        t_prev_idx = torch.clamp(t_prev, min=0)
        alpha_bar_prev = torch.where(
            t_prev.view(-1, 1, 1, 1) >= 0,
            self.alpha_bars[t_prev_idx].view(-1, 1, 1, 1),
            torch.ones_like(alpha_bar_t),
        )

        z_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        z_0_pred = torch.clamp(z_0_pred, -3, 3)

        if eta > 0:
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            sigma = eta * torch.sqrt(torch.clamp(variance, min=0))
        else:
            sigma = torch.zeros_like(alpha_bar_t)

        direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0)) * noise_pred
        z_prev = torch.sqrt(alpha_bar_prev) * z_0_pred + direction

        if eta > 0:
            mask = (t_prev.view(-1, 1, 1, 1) >= 0).float()
            z_prev = z_prev + mask * sigma * torch.randn_like(z_t)

        return z_prev
    
    def get_timesteps(self, num_steps, device):
        # Returns a tensor of timesteps from num_steps-1 to 0, which is used in the sampling process
        return torch.linspace(self.num_steps-1, 0, num_steps, dtype=torch.long, device=device)