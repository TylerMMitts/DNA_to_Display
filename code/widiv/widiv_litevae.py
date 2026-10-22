# LiteVAE for WiDiv: the same encoder and decoder, at any size divisible by 8.
#
# The root LiteVAEEncoder resizes every input to 256 px and aligns its three
# wavelet levels on a fixed 32x32 grid, so a 512 px image would be squashed to
# 256 before encoding - exactly the detail (cortical cells, about 4 px at 512)
# the higher resolution is for. This subclass keeps the input size and aligns
# on H/8 x W/8 instead: 512 px in, a 4 x 64 x 64 latent out. Its layers and
# weights are identical to the parent's, so at 256 px it gives the same result.
#
# The decoder is fully convolutional (three 2x upsampling stages), so the
# unchanged LiteVAEDecoder already turns a 64x64 latent into a 512 px image.

import torch
import torch.nn.functional as F

from litevae.models import LiteVAEDecoder, LiteVAEEncoder
from litevae.utils.wavelet import dwt_2d


class ScaleFreeLiteVAEEncoder(LiteVAEEncoder):

    def forward(self, x, save_steps=False, save_dir=None):
        if save_steps:
            raise NotImplementedError("step images are only drawn by the 256 px LiteVAEEncoder")
        B, C, H, W = x.shape
        if H % 8 or W % 8:
            raise ValueError(f"image size must be divisible by 8, got {H}x{W}")
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        LL1, LH1, HL1, HH1 = dwt_2d(x)          # H   -> H/2
        LL2, LH2, HL2, HH2 = dwt_2d(LL1)        # H/2 -> H/4
        LL3, LH3, HL3, HH3 = dwt_2d(LL2)        # H/4 -> H/8
        LL1, LH1, HL1, HH1 = self.normalize_wavelet_coefficients(LL1, LH1, HL1, HH1, level=1)
        LL2, LH2, HL2, HH2 = self.normalize_wavelet_coefficients(LL2, LH2, HL2, HH2, level=2)
        LL3, LH3, HL3, HH3 = self.normalize_wavelet_coefficients(LL3, LH3, HL3, HH3, level=3)

        F1 = self.f1(torch.cat([LL1, LH1, HL1, HH1], dim=1))
        F2 = self.f2(torch.cat([LL2, LH2, HL2, HH2], dim=1))
        F3 = self.f3(torch.cat([LL3, LH3, HL3, HH3], dim=1))

        # The parent aligns on a fixed 32x32; here the latent grid is H/8 x W/8.
        size = F3.shape[-2:]
        F1_aligned = F.interpolate(F1, size=size, mode='bilinear', align_corners=False)
        F2_aligned = F.interpolate(F2, size=size, mode='bilinear', align_corners=False)
        z_params = self.agg(torch.cat([F1_aligned, F2_aligned, F3], dim=1))

        z_mean, z_logvar = torch.chunk(z_params, 2, dim=1)
        z_mean = z_mean * self.latent_scale.view(1, -1, 1, 1)
        return self.reparameterize(z_mean, z_logvar), z_mean, z_logvar


# Loads a WiDiv LiteVAE checkpoint (train_litevae_widiv.py), frozen, in eval mode.
def load_widiv_litevae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    encoder = ScaleFreeLiteVAEEncoder(in_channels=3, latent_channels=cfg['latent_channels'],
                                      feature_channels=cfg['feature_channels'],
                                      num_blocks=cfg['num_blocks'])
    decoder = LiteVAEDecoder(latent_channels=cfg['latent_channels'], output_channels=3,
                             base_channels=cfg['base_channels'],
                             num_res_blocks=cfg['num_res_blocks'])
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    encoder.to(device).eval()
    decoder.to(device).eval()
    print(f"WiDiv LiteVAE loaded (epoch {ckpt.get('epoch', '?')}, "
          f"{ckpt.get('image_size', '?')} px), frozen")
    return encoder, decoder
