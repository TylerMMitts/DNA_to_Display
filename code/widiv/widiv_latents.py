# Latent scaling for the WiDiv diffusion model.
#
# The diffusion noise schedule (linear betas 1e-4 to 0.02 over 1000 steps)
# assumes its data has about unit spread, which is why latent diffusion models
# usually rescale their autoencoder's latents before training. Sampled WiDiv
# latents happen to be at unit spread (about 0.99), because the background is
# the encoder's N(0, 1) prior; mean latents are not (0.60 overall: 1.0 in the
# root, 0.28 in the background, 77% of the image). LatentScaler standardises
# each channel to mean 0 and spread 1 over the training latents; the UNet
# learns and generates in those units, and they are undone just before decoding.
#
# The shift and scale are fitted once at the start of a run and saved in every
# checkpoint, so generation and analysis decode with exactly the same numbers.

import torch
import torch.nn as nn

from latent_diffusion.diffusion.scheduler import DiffusionScheduler


class LatentScaler:

    def __init__(self, shift, scale):
        self.shift = torch.as_tensor(shift, dtype=torch.float32).view(1, -1, 1, 1)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).view(1, -1, 1, 1)

    @classmethod
    def identity(cls, channels=4):
        return cls(torch.zeros(channels), torch.ones(channels))

    # Per-channel mean and spread over every pixel of the given latents (N, C, H, W).
    @classmethod
    def fit(cls, latents):
        x = latents.float().transpose(0, 1).flatten(1)
        return cls(x.mean(1).cpu(), x.std(1).cpu())

    def encode(self, z):
        return (z - self.shift.to(z.device)) / self.scale.to(z.device)

    def decode(self, z):
        return z * self.scale.to(z.device) + self.shift.to(z.device)

    def state_dict(self):
        return {'shift': self.shift.flatten().tolist(), 'scale': self.scale.flatten().tolist()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['shift'], state['scale'])

    # The scaler a WiDiv diffusion checkpoint was trained with; identity for runs
    # from before scaling existed, whose latents were never rescaled.
    @classmethod
    def from_checkpoint(cls, ckpt, channels=4):
        state = ckpt.get('latent_scaler')
        return cls.from_state_dict(state) if state else cls.identity(channels)

    def __repr__(self):
        return (f"LatentScaler(shift={[round(v, 3) for v in self.shift.flatten().tolist()]}, "
                f"scale={[round(v, 3) for v in self.scale.flatten().tolist()]})")


# The LiteVAE decoder fed with scaled latents: undoes the scaling first, so
# anything that decodes a generated latent (generate_batch, previews, analysis)
# can be handed this in place of the decoder.
class ScaledDecoder(nn.Module):

    def __init__(self, decoder, scaler):
        super().__init__()
        self.decoder = decoder
        self.scaler = scaler

    def forward(self, z, save_steps=False):
        return self.decoder(self.scaler.decode(z), save_steps=save_steps)


# DiffusionScheduler with the clamp on the predicted clean latent moved into
# scaled units. The parent clamps to +/-3, a bound chosen for the LiteVAE's own
# latents; after scaling, the same bound in those units is scaler.encode(+/-3),
# per channel. Without this a 1/0.6 rescale would clip root values from 1.8
# original units, about 7% of the root, instead of 0.4%. Same sampling otherwise.
class ScaledClampScheduler(DiffusionScheduler):

    def set_scaler(self, scaler, bound=3.0):
        channels = scaler.shift.shape[1]
        self.clamp_lo = scaler.encode(torch.full((1, channels, 1, 1), -bound))
        self.clamp_hi = scaler.encode(torch.full((1, channels, 1, 1), bound))
        return self

    def denoise_step(self, z_t, noise_pred, t, t_prev=None, eta=0.0):
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        if t_prev is None:
            t_prev = t - 1
        t_prev_idx = torch.clamp(t_prev, min=0)
        alpha_bar_prev = torch.where(
            t_prev.view(-1, 1, 1, 1) >= 0,
            self.alpha_bars[t_prev_idx].view(-1, 1, 1, 1),
            torch.ones_like(alpha_bar_t),
        )
        z_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        lo = getattr(self, 'clamp_lo', None)
        if lo is None:
            z_0_pred = torch.clamp(z_0_pred, -3, 3)
        else:
            z_0_pred = torch.maximum(torch.minimum(z_0_pred, self.clamp_hi.to(z_t.device)), lo.to(z_t.device))
        if eta > 0:
            variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            sigma = eta * torch.sqrt(torch.clamp(variance, min=0))
        else:
            sigma = torch.zeros_like(alpha_bar_t)
        direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0)) * noise_pred
        z_prev = torch.sqrt(alpha_bar_prev) * z_0_pred + direction
        if eta > 0:
            mask = (t_prev.view(-1, 1, 1, 1) >= 0).float()
            z_prev = z_prev + mask * sigma * torch.randn_like(z_t)
        return z_prev
