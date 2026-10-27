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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_diffusion.diffusion.scheduler import DiffusionScheduler


# The layout channel: where the root is, at the latent's resolution.
#
# Root and background latents differ mostly in spread, not in average (per
# channel, root mean 0.0/+0.50/0.0/-0.16 against background 0.0/-0.23/0.0/+0.05,
# spreads about 1.0 against 0.25). Early in sampling the UNet sees little but
# averages, so where the root ends is barely visible to it and is settled late,
# by the starting noise - the seed-dependent broken roots. This channel is +1
# inside the root and -1 outside, a contrast in average that survives heavy
# noise, so the root's outline and size are settled in the first, noisiest
# steps and tissue fills in within it. The UNet generates it as a fifth channel;
# the decoder never sees it (ScaledDecoder drops it).
#
# The root is a centred disc reaching the outermost tissue pixel, as in the
# LiteVAE's foreground weighting: prepare_widiv_images.py centres every root,
# and a disc includes the white vessel lumens and aerenchyma gaps inside it.
# Averaged over each 8x8 block, so the rim is fractional rather than a hard step.
def root_layout(images, factor=8):
    B, _, H, W = images.shape
    ys = torch.arange(H, device=images.device, dtype=torch.float32) - (H - 1) / 2
    xs = torch.arange(W, device=images.device, dtype=torch.float32) - (W - 1) / 2
    dist = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    tissue = images.min(dim=1).values < 0.96                      # white is +1
    radius = torch.where(tissue, dist, torch.zeros_like(dist)).flatten(1).amax(1)
    disc = (dist[None] <= radius[:, None, None]).float()[:, None]
    return F.avg_pool2d(disc, factor) * 2 - 1


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


# The LiteVAE decoder fed with what the UNet generates: keeps the LiteVAE's
# channels (dropping a layout channel if there is one) and undoes the scaling,
# so anything that decodes a generated latent (generate_batch, previews,
# analysis) can be handed this in place of the decoder.
class ScaledDecoder(nn.Module):

    def __init__(self, decoder, scaler):
        super().__init__()
        self.decoder = decoder
        self.scaler = scaler

    def forward(self, z, save_steps=False):
        c = self.scaler.shift.shape[1]
        return self.decoder(self.scaler.decode(z[:, :c]), save_steps=save_steps)


# The generated root layout as a 0-1 mask at image resolution, for a latent the
# UNet generated with a layout channel (its last channel); None without one.
def generated_layout(z, scaler, size=None):
    c = scaler.shift.shape[1]
    if z.shape[1] <= c:
        return None
    m = ((z[:, c:c + 1].clamp(-1, 1) + 1) / 2)
    return F.interpolate(m, size=size, mode='bilinear', align_corners=False) if size else m


# DiffusionScheduler with the clamp on the predicted clean latent moved into
# scaled units. The parent clamps to +/-3, a bound chosen for the LiteVAE's own
# latents; after scaling, the same bound in those units is scaler.encode(+/-3),
# per channel. Without this a 1/0.6 rescale would clip root values from 1.8
# original units, about 7% of the root, instead of 0.4%. Same sampling otherwise.
class ScaledClampScheduler(DiffusionScheduler):

    # extra_channels: channels after the LiteVAE's (the layout channel), which
    # are not scaled and keep the plain +/-bound.
    def set_scaler(self, scaler, bound=3.0, extra_channels=0):
        channels = scaler.shift.shape[1]
        lo = scaler.encode(torch.full((1, channels, 1, 1), -bound))
        hi = scaler.encode(torch.full((1, channels, 1, 1), bound))
        if extra_channels:
            lo = torch.cat([lo, torch.full((1, extra_channels, 1, 1), -bound)], 1)
            hi = torch.cat([hi, torch.full((1, extra_channels, 1, 1), bound)], 1)
        self.clamp_lo, self.clamp_hi = lo, hi
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


# Starting noise whose per-channel average is the one training inputs had.
#
# The linear schedule keeps sqrt(alpha_bar) = 0.006 of the clean latent even at
# the noisiest step, so in training each channel's spatial average carries a
# faint copy of that latent's overall level, and the UNet learns to read it -
# amplified about 158x in its prediction of the clean latent. Pure noise has a
# random average instead (sd 1/64 per channel), so each seed fixed a random
# level for the whole root. With the v2 LiteVAE, whose latents average about 0,
# that cost a share of roots; with v3, whose background sits at -2, it drew the
# background as tissue for most seeds (12% valid roots). Replacing each
# channel's spatial average with sqrt(alpha_bar) x the mean training latent
# took v3 to 79% and v2 to 71% (results/widiv/seed_sweep/, *_dc). The noise
# keeps its spatial pattern, so a seed still means the same thing everywhere.
#
# latent_mean: per-channel mean of the training latents in the UNet's units
# (scaled, layout channel included), as saved in each checkpoint.
def matched_start_noise(noise, latent_mean, alpha_bar_start):
    mean = torch.as_tensor(latent_mean, dtype=noise.dtype, device=noise.device).view(1, -1, 1, 1)
    return noise - noise.mean((2, 3), keepdim=True) + torch.sqrt(alpha_bar_start) * mean


# Per-channel mean of a stack of latents (N, C, H, W) as a plain list, the form
# checkpoints keep it in.
def channel_mean(latents):
    return latents.float().mean((0, 2, 3)).cpu().tolist()


# DDIM sampling for WiDiv, as generate_from_dataset.generate_batch but from
# matched starting noise when latent_mean is given (None = plain noise, as every
# run before it), and with optional stochastic steps (eta). One noise draw per
# seed, so a seed gives the same start for every genotype. Returns uint8 images
# (B, H, W, 3) through the given decoder (a ScaledDecoder for scaled or layout runs).
def generate_widiv_batch(snp_encoder, unet, scheduler, decoder, snp_batch, seeds, device,
                         latent_shape, num_steps, latent_mean=None, eta=0.0, decode_batch=4):
    z_t = torch.stack([torch.randn(*latent_shape, generator=torch.Generator().manual_seed(int(s)))
                       for s in seeds]).to(device)
    timesteps = scheduler.get_timesteps(num_steps, device)
    if latent_mean is not None:
        z_t = matched_start_noise(z_t, latent_mean, scheduler.alpha_bars[int(timesteps[0])])
    snp_embedding = snp_encoder(snp_batch)
    B = z_t.shape[0]
    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
        t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
        z_t = scheduler.denoise_step(z_t, unet(z_t, t_batch, snp_embedding), t_batch,
                                     t_prev_batch, eta=eta)
    # Decoded a few at a time: the 512 px decoder is the memory peak on a 4 GB GPU.
    images = torch.cat([decoder(z_t[i:i + decode_batch], save_steps=False)
                        for i in range(0, B, decode_batch)])
    images = torch.clamp((images + 1) / 2, 0, 1)
    return (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


# Per-channel mean latent in the UNet's units from a batch of training images
# (N, 3, H, W), for checkpoints that predate their saved latent_channel_mean.
# Uses the encoder's mean latent, which has the same average as its samples.
def encoded_latent_mean(litevae_encoder, images, scaler, layout=False):
    with torch.no_grad():
        _, mean, _ = litevae_encoder(images)
        z = scaler.encode(mean)
        if layout:
            z = torch.cat([z, root_layout(images)], 1)
    return channel_mean(z)
