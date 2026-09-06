# Saves each intermediate stage of an encode/decode pass as an image.
#
# Used by the demo to make the architecture visible one step at a time.

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Normalizes an image tensor to the range [0, 1] for visualization purposes
def _normalize_image(img):
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min + 1e-8)
    return np.zeros_like(img)


def save_wavelet_bands(x, LL1, LH1, HL1, HH1, LL2, LH2, HL2, HH2, LL3, LH3, HL3, HH3, save_dir):
    os.makedirs(f"{save_dir}/encoder/wavelet", exist_ok=True)
    
    def save_tensor(t, name):
        img = t[0].mean(dim=0).detach().cpu().numpy()
        img = _normalize_image(img)
        plt.imsave(f"{save_dir}/encoder/wavelet/{name}.png", img, cmap='gray')
    
    # Save original image
    orig = x[0].mean(dim=0).detach().cpu().numpy()
    orig = _normalize_image(orig)
    plt.imsave(f"{save_dir}/encoder/00_original_input.png", orig, cmap='gray')
    
    # Level 1
    save_tensor(LL1, "L1_LL")
    save_tensor(LH1, "L1_LH")
    save_tensor(HL1, "L1_HL")
    save_tensor(HH1, "L1_HH")
    
    # Level 2
    save_tensor(LL2, "L2_LL")
    save_tensor(LH2, "L2_LH")
    save_tensor(HL2, "L2_HL")
    save_tensor(HH2, "L2_HH")
    
    # Level 3
    save_tensor(LL3, "L3_LL")
    save_tensor(LH3, "L3_LH")
    save_tensor(HL3, "L3_HL")
    save_tensor(HH3, "L3_HH")
    
    # Combined grid
    bands = [
        LL1[0].mean(dim=0), LH1[0].mean(dim=0), HL1[0].mean(dim=0), HH1[0].mean(dim=0),
        LL2[0].mean(dim=0), LH2[0].mean(dim=0), HL2[0].mean(dim=0), HH2[0].mean(dim=0),
        LL3[0].mean(dim=0), LH3[0].mean(dim=0), HL3[0].mean(dim=0), HH3[0].mean(dim=0)
    ]
    bands = [b.detach().cpu().numpy() for b in bands]
    bands = [_normalize_image(b) for b in bands]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    titles = ['LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2', 'LL3', 'LH3', 'HL3', 'HH3']
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(bands[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/encoder/wavelet/combined_wavelet_bands.png", dpi=150)
    plt.close()


def save_feature_maps(F1, F2, F3, save_dir):
    os.makedirs(f"{save_dir}/encoder/feature_maps", exist_ok=True)
    
    def save_feature(t, name):
        img = t[0].mean(dim=0).detach().cpu().numpy()
        img = _normalize_image(img)
        plt.imsave(f"{save_dir}/encoder/feature_maps/{name}.png", img, cmap='viridis')
    
    save_feature(F1, "F1_features_128x128")
    save_feature(F2, "F2_features_64x64")
    save_feature(F3, "F3_features_32x32")
    
    # Save channel grids
    _save_feature_grid(F1, "F1", save_dir)
    _save_feature_grid(F2, "F2", save_dir)
    _save_feature_grid(F3, "F3", save_dir)


def _save_feature_grid(feats, name, save_dir):
    f = feats[0][:16].detach().cpu()
    f_min, f_max = f.min(), f.max()
    if f_max > f_min:
        f = (f - f_min) / (f_max - f_min + 1e-8)
    else:
        f = torch.zeros_like(f)
    
    n_ch = min(f.shape[0], 16)
    rows = (n_ch + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(8, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows):
        for j in range(4):
            idx = i * 4 + j
            if idx < n_ch:
                axes[i, j].imshow(f[idx], cmap='viridis')
                axes[i, j].set_title(f"Ch{idx}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/encoder/feature_maps/{name}_channels_grid.png", dpi=100)
    plt.close()


def save_latent_code(z, save_dir):
    os.makedirs(f"{save_dir}/encoder/latent", exist_ok=True)
    
    z_img = z[0].detach().cpu()
    n_ch = min(z_img.shape[0], 8)
    
    for i in range(n_ch):
        channel = z_img[i].numpy()
        channel = _normalize_image(channel)
        plt.imsave(f"{save_dir}/encoder/latent/z_channel_{i:02d}.png", channel, cmap='viridis')
    
    rows = (n_ch + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(8, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows):
        for j in range(4):
            idx = i * 4 + j
            if idx < n_ch:
                img = z_img[idx].numpy()
                img = _normalize_image(img)
                axes[i, j].imshow(img, cmap='viridis')
                axes[i, j].set_title(f"Ch{idx}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/encoder/latent/latent_grid.png", dpi=150)
    plt.close()


def save_decoder_step(tensor, name, save_dir, is_rgb=False):
    img = tensor[0].detach().cpu()
    
    if is_rgb:
        img = img.permute(1, 2, 0).numpy()
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        plt.imsave(f"{save_dir}/decoder/{name}.png", img)
    else:
        img = img.mean(dim=0).numpy()
        img = _normalize_image(img)
        plt.imsave(f"{save_dir}/decoder/{name}.png", img, cmap='viridis')


def save_decoder_channels(tensor, name, save_dir):
    img = tensor[0].detach().cpu()
    n_ch = min(img.shape[0], 16)
    f = img[:n_ch]
    
    for i in range(n_ch):
        channel = f[i].numpy()
        channel = _normalize_image(channel)
        plt.imsave(f"{save_dir}/decoder/channels/{name}_ch{i:02d}.png", channel, cmap='viridis')
    
    rows = (n_ch + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(8, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows):
        for j in range(4):
            idx = i * 4 + j
            if idx < n_ch:
                img = f[idx].numpy()
                img = _normalize_image(img)
                axes[i, j].imshow(img, cmap='viridis')
                axes[i, j].set_title(f"Ch{idx}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/decoder/channels/{name}_grid.png", dpi=100)
    plt.close()


def save_comparison(original, reconstructed, save_dir):
    # Original: [B, C, H, W] → [H, W, C]
    orig = original[0].detach().cpu()
    orig = orig.permute(1, 2, 0).numpy()
    orig = (orig + 1) / 2
    orig = np.clip(orig, 0, 1)
    
    # Reconstructed
    recon = reconstructed[0].detach().cpu()
    recon = recon.permute(1, 2, 0).numpy()
    recon = (recon + 1) / 2
    recon = np.clip(recon, 0, 1)
    
    # Save individual images
    plt.imsave(f"{save_dir}/00_original_input.png", orig)
    plt.imsave(f"{save_dir}/06_reconstructed_output.png", recon)
    
    # Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(recon)
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')
    
    diff = np.abs(orig - recon)
    axes[2].imshow(diff * 5)
    axes[2].set_title("Difference (amplified 5x)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison.png", dpi=150)
    plt.close()