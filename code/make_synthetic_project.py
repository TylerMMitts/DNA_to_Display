# Mints a complete stand-in project - dataset, metadata and random-weight
# checkpoints - so every script in the repo runs before the real weights and
# scans arrive.
#
# Two things this is NOT. It is not a trained model: the weights are freshly
# initialised, so anything it generates is noise and any analysis run against
# it is measuring an untrained network. And it is not the real dataset: the
# images are drawn, not photographed.
#
# What it IS is a genuine genotype -> phenotype map. Vessel count, stele size
# and root radius are deterministic functions of designated causal loci, so
# the synthetic set has real signal in it - which means training on it should
# converge, and genetic_fidelity_test.py has a right answer to find. That
# makes it a working end-to-end rehearsal, not just a plumbing check.
#
# Everything written here is stamped. Each checkpoint carries synthetic=True,
# and models/.synthetic makes every script in the repo print a banner on
# import. Delete that marker only when the real weights replace these.

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    CROPPED_IMAGES_DIR, DIFFUSION_NUMERIC_DIR, DIFFUSION_ONEHOT_DIR,
    IMAGES_DIR, IMAGE_METADATA, KINSHIP_MATRIX, LITEVAE_DIR, METADATA_DIR,
    MODELS_DIR, RESULTS_DIR, SEGMENTATION_DIR, SNP_PARQUET,
    SYNTHETIC_MARKER, best_checkpoint_path, checkpoint_path,
)

from latent_diffusion.models.snp_encoding import SNPProjector, OneHotSNPEncoder
from latent_diffusion.models.snp_encoder import SNPEncoder
from latent_diffusion.models.unet import DenoisingUNet
from litevae.models import LiteVAEEncoder, LiteVAEDecoder

FOUNDERS = (1, 2, 3, 4, 5, 6, 7, 8)


# Genotypes as founder mosaics, the way a real MAGIC population looks: each
# genotype is a run of chromosome blocks, each block inherited whole from one
# founder. Independent per-locus draws would leave no population structure for
# the kinship matrix to describe.
# O(n_geno * n_loci)
def make_genotypes(rng, n_geno, n_loci, block_len):
    n_blocks = int(np.ceil(n_loci / block_len))
    blocks = rng.integers(1, len(FOUNDERS) + 1, size=(n_geno, n_blocks))
    return np.repeat(blocks, block_len, axis=1)[:, :n_loci].astype(np.int16)


# Traits from designated causal loci. Three disjoint windows of the genome
# each drive one trait, so the map is real but simple enough to verify.
# O(n_loci)
def phenotype_of(codes, n_loci):
    a, b, c = (slice(0, n_loci // 8), slice(n_loci // 8, n_loci // 4),
               slice(n_loci // 4, 3 * n_loci // 8))
    # Fraction of each window carrying an odd-numbered founder allele.
    fa, fb, fc = (float((codes[s] % 2 == 1).mean()) for s in (a, b, c))
    return {
        'n_vessels': int(round(6 + 12 * fa)),          # 6 - 18
        'stele_ratio': 0.34 + 0.22 * fb,               # 0.34 - 0.56
        'root_ratio': 0.72 + 0.16 * fc,                # 0.72 - 0.88
    }


# Draws one root cross-section: outer root disc, inner stele, and a ring of
# xylem vessels inside the stele. Jitter is per-replicate, not per-genotype,
# so the trait stays the signal and the jitter stays the noise.
# O(size^2)
def render_root(traits, rng, size):
    img = Image.new('RGB', (size, size), (14, 12, 10))
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2
    r_root = size / 2 * traits['root_ratio'] * rng.uniform(0.96, 1.04)
    r_stele = r_root * traits['stele_ratio'] * rng.uniform(0.96, 1.04)
    warm = rng.integers(-12, 13)

    draw.ellipse([cx - r_root, cy - r_root, cx + r_root, cy + r_root],
                 fill=(168 + warm, 141 + warm, 104 + warm))
    draw.ellipse([cx - r_stele, cy - r_stele, cx + r_stele, cy + r_stele],
                 fill=(96 + warm, 108 + warm, 84 + warm))

    n = traits['n_vessels']
    phase = rng.uniform(0, 2 * np.pi)
    r_ring, r_vessel = r_stele * 0.62, max(2.0, r_stele * 0.13)
    for k in range(n):
        th = phase + 2 * np.pi * k / n
        vx, vy = cx + r_ring * np.cos(th), cy + r_ring * np.sin(th)
        draw.ellipse([vx - r_vessel, vy - r_vessel, vx + r_vessel, vy + r_vessel],
                     fill=(226 + warm // 2, 220 + warm // 2, 198 + warm // 2))

    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0, 6.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# The dataset folder plus its four metadata files.
# O(n_images * size^2)
def write_dataset(rng, codes, genotypes, cfg):
    for d in (IMAGES_DIR, METADATA_DIR, CROPPED_IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for gi, g in enumerate(genotypes):
        traits = phenotype_of(codes[gi], cfg.n_loci)
        for rep in range(1, cfg.replications + 1):
            for num in range(1, cfg.roots_per_plant + 1):
                name = f'{g}_primary_{rep}_{num}.JPG'
                img = render_root(traits, rng, cfg.image_size)
                img.save(IMAGES_DIR / name, quality=95)
                # The crops are what everything downstream trains on, and
                # crop_root_model.py keeps the source filename.
                img.save(CROPPED_IMAGES_DIR / name, quality=95)
                rows.append({
                    'image_id': len(rows) + 1,
                    'new_filename': name,
                    'original_filename': f'raw_{g}_r{rep}_{num}.tif',
                    'genotype': g,
                    'rootnode': 'primary',
                    'replication': rep,
                    'rootnumber': num,
                    'pixel_size_um': 2.5,
                    'image_width_px': cfg.image_size,
                    'image_height_px': cfg.image_size,
                    'quality_notes': f"SYNTHETIC - vessels={traits['n_vessels']}",
                })

    pd.DataFrame(rows).to_csv(IMAGE_METADATA, index=False)
    pd.DataFrame({'genotype_id': genotypes}).to_csv(
        METADATA_DIR / 'genotype_list.csv', index=False)

    # Long format, matching load_snp_data_from_parquet's pivot. The real file
    # keeps a _TC suffix on the ID, which that loader strips.
    loci = [f'SNP_{i:05d}' for i in range(cfg.n_loci)]
    pd.DataFrame({
        'ID': np.repeat([f'{g}_TC' for g in genotypes], cfg.n_loci),
        'gene_model': np.tile(loci, len(genotypes)),
        'value': codes.reshape(-1).astype(np.float32),
    }).to_parquet(SNP_PARQUET, index=False)

    # Relatedness as allele sharing between founder mosaics.
    share = np.array([[float((codes[i] == codes[j]).mean())
                       for j in range(len(genotypes))]
                      for i in range(len(genotypes))])
    labels = [f'{g}_TC' for g in genotypes]
    pd.DataFrame(share, index=labels, columns=labels).to_csv(KINSHIP_MATRIX)
    return len(rows)


# Reproduces train_onehot.py's split exactly, so a later run with resume=True
# recognises this checkpoint's split instead of refusing to continue.
# O(n_geno log n_geno)
def val_split(genotypes, seed, fraction):
    shuffled = sorted(genotypes)
    np.random.default_rng(seed).shuffle(shuffled)
    return sorted(shuffled[:max(1, int(round(len(shuffled) * fraction)))])


# O(1) in the dataset - cost is initialising the nets
def mint_litevae(cfg):
    LITEVAE_DIR.mkdir(parents=True, exist_ok=True)
    enc = LiteVAEEncoder(3, cfg.latent_channels, 64, 3)
    dec = LiteVAEDecoder(cfg.latent_channels, 3, 512, 2)
    blob = {
        'epoch': 250,
        'encoder_state_dict': enc.state_dict(),
        'decoder_state_dict': dec.state_dict(),
        'train_loss': float('nan'), 'val_loss': float('nan'),
        'config': {'latent_channels': cfg.latent_channels, 'feature_channels': 64,
                   'base_channels': 512, 'num_blocks': 3, 'num_res_blocks': 2},
        'synthetic': True,
        'note': 'RANDOM WEIGHTS - never trained. Reconstructions are noise.',
    }
    torch.save(blob, checkpoint_path(LITEVAE_DIR, 250, 'litevae'))
    torch.save(blob, best_checkpoint_path(LITEVAE_DIR, 'litevae'))
    return sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())


# O(n_geno * n_loci * n_founders) for the PCA fit
def mint_diffusion_onehot(codes, genotypes, cfg):
    DIFFUSION_ONEHOT_DIR.mkdir(parents=True, exist_ok=True)
    projector = SNPProjector(founders=FOUNDERS,
                             target_variance=cfg.pca_target_variance).fit(codes, verbose=False)

    enc_cfg = {'input_dim': projector.output_dim, 'embedding_dim': 512,
               'num_tokens': 8, 'hidden_dim': 1024}
    unet_cfg = {'latent_channels': cfg.latent_channels, 'base_channels': 128,
                'snp_embed_dim': 512, 'd_attention': 512, 'num_res_blocks': 2,
                'attention_resolutions': [1, 2, 4]}
    snp_encoder, unet = OneHotSNPEncoder(**enc_cfg), DenoisingUNet(**unet_cfg)

    # Real optimiser and schedule objects, so resume=True finds the state it
    # expects rather than a KeyError.
    opt = torch.optim.Adam([{'params': unet.parameters(), 'lr': 3e-5},
                            {'params': snp_encoder.parameters(), 'lr': 1e-4}])
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-7)

    torch.save({
        'epoch': 100, 'note': 'RANDOM WEIGHTS - never trained. Output is noise.',
        'unet_state_dict': unet.state_dict(),
        'snp_encoder_state_dict': snp_encoder.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'lr_scheduler_state_dict': lrs.state_dict(),
        'loss': float('nan'), 'val_loss': float('nan'),
        'snp_projector': projector.state_dict(),
        'snp_encoder_config': enc_cfg, 'unet_config': unet_cfg,
        'encoding': 'one_hot_founders', 'founders': list(FOUNDERS),
        'val_genotypes': val_split(genotypes, cfg.seed, cfg.val_fraction),
        'synthetic': True,
    }, checkpoint_path(DIFFUSION_ONEHOT_DIR, 100, 'diffusion_onehot'))
    return projector.output_dim, sum(p.numel() for p in unet.parameters())


# The legacy numeric checkpoint, which train_onehot.py warm-starts its UNet
# from. No snp_projector and no encoding key, so load_model reads it as legacy.
# O(1)
def mint_diffusion_numeric(n_loci, cfg):
    DIFFUSION_NUMERIC_DIR.mkdir(parents=True, exist_ok=True)
    unet_cfg = {'latent_channels': cfg.latent_channels, 'base_channels': 128,
                'snp_embed_dim': 512, 'd_attention': 512, 'num_res_blocks': 2,
                'attention_resolutions': [1, 2, 4]}
    torch.save({
        'epoch': 500, 'note': 'RANDOM WEIGHTS - never trained.',
        'unet_state_dict': DenoisingUNet(**unet_cfg).state_dict(),
        'snp_encoder_state_dict': SNPEncoder(
            num_snps=n_loci, embedding_dim=512, num_tokens=8,
            pca_components=cfg.numeric_pca, hidden_dim=1024).state_dict(),
        'loss': float('nan'), 'unet_config': unet_cfg, 'synthetic': True,
    }, checkpoint_path(DIFFUSION_NUMERIC_DIR, 500, 'diffusion_numeric'))


# An untrained three-class YOLOv8 segmenter, built from the architecture yaml
# with nc dropped from 80 to root/stele/vessel.
# O(1)
def mint_segmentation():
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(ASSETS).parent / 'cfg' / 'models' / 'v8' / 'yolov8-seg.yaml'
    if not src.exists():
        return None
    scratch = RESULTS_DIR / 'synthetic'
    scratch.mkdir(parents=True, exist_ok=True)
    dst = scratch / 'yolov8n-seg-3class.yaml'
    dst.write_text(src.read_text().replace('nc: 80', 'nc: 3'))

    model = YOLO(str(dst))
    model.model.names = {0: 'root', 1: 'stele', 2: 'vessel'}
    out = best_checkpoint_path(SEGMENTATION_DIR, 'feature_segmentation')
    model.save(str(out))
    return out


def main():
    # Edit these values, then run:
    #     python code/make_synthetic_project.py
    class cfg:
        n_genotypes = 24
        n_loci = 400
        block_len = 20            # founder mosaic block, in loci
        replications = 2
        roots_per_plant = 1
        image_size = 256
        latent_channels = 4
        pca_target_variance = 0.95
        numeric_pca = 20          # components for the legacy numeric encoder
        val_fraction = 0.2        # must match train_onehot.py's cfg
        seed = 0                  # must match train_onehot.py's cfg
        overwrite = False         # refuse to clobber a real dataset or weights

    real = [p for p in (IMAGES_DIR, SNP_PARQUET) if p.exists()]
    if real and not cfg.overwrite and not SYNTHETIC_MARKER.exists():
        raise SystemExit(
            "Refusing to overwrite what looks like real data:\n  "
            + '\n  '.join(str(p) for p in real)
            + "\nSet overwrite = True if you are sure.")

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    genotypes = [f'MEMA{i + 1:03d}' for i in range(cfg.n_genotypes)]
    codes = make_genotypes(rng, cfg.n_genotypes, cfg.n_loci, cfg.block_len)

    print(f"Synthetic project -> {MODELS_DIR.parent}")
    n_images = write_dataset(rng, codes, genotypes, cfg)
    print(f"  dataset      {n_images} images, {cfg.n_genotypes} genotypes, "
          f"{cfg.n_loci} loci")

    vae_params = mint_litevae(cfg)
    print(f"  litevae      random weights, {vae_params / 1e6:.1f}M params")

    dim, unet_params = mint_diffusion_onehot(codes, genotypes, cfg)
    print(f"  diffusion    random weights, {unet_params / 1e6:.1f}M params, "
          f"SNP PCA -> {dim} components")

    mint_diffusion_numeric(cfg.n_loci, cfg)
    print("  numeric      random weights (legacy warm-start source)")

    try:
        seg = mint_segmentation()
        print(f"  segmenter    {'random weights, 3 classes' if seg else 'SKIPPED (yaml not found)'}")
    except Exception as exc:
        print(f"  segmenter    SKIPPED ({type(exc).__name__}: {exc})")

    SYNTHETIC_MARKER.write_text(
        "Every checkpoint beside this file has random, untrained weights, and\n"
        "dataset/ holds drawn images rather than scans. Written by\n"
        "code/make_synthetic_project.py. Delete this file when real weights\n"
        "replace them - paths.py prints a banner while it exists.\n")

    print(f"\nStamped {SYNTHETIC_MARKER.relative_to(MODELS_DIR.parent)}. "
          "Every script now prints a synthetic-data banner.")
    print("Try:  python code/latent_diffusion/generation/generate_from_dataset.py")


if __name__ == '__main__':
    main()
