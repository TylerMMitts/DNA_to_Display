# What comes out when two root latents are averaged.
#
# Also the shared LiteVAE loader and image helpers that the other evaluation
# scripts import.

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import CROPPED_IMAGES_DIR, LITEVAE_MODEL, RESULTS_DIR, SEGMENTATION_MODEL

from litevae.models import LiteVAEEncoder, LiteVAEDecoder
from ultralytics import YOLO

CLASS_NAMES = {0: 'root', 1: 'stele', 2: 'vessel'}


# LiteVAE

def load_litevae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = LiteVAEEncoder(in_channels=3, latent_channels=4,
                             feature_channels=64, num_blocks=3)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.to(device).eval()

    decoder = LiteVAEDecoder(latent_channels=4, output_channels=3,
                             base_channels=512, num_res_blocks=2)
    decoder.load_state_dict(ckpt['decoder_state_dict'])
    decoder.to(device).eval()

    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad = False

    print(f"LiteVAE loaded (epoch {ckpt.get('epoch', '?')})")
    return encoder, decoder


def load_image(path, imgsz=256):
    # Matches LiteVAE's training transform: squash to 256x256, scale to [-1,
    # 1].
    im = Image.open(path).convert('RGB').resize((imgsz, imgsz), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def to_uint8(tensor):
    # Decoder output [-1, 1] -> uint8 HWC.
    arr = ((tensor.clamp(-1, 1) + 1) / 2 * 255).round().byte()
    return arr.permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def encode(encoder, batch):
    # Returns z_mean, the deterministic centre of the latent distribution.
    #
    # The encoder's first return value is a reparameterised sample, z_mean +
    # eps * std, so it carries fresh noise on every call. Averaging those would
    # mix latent interpolation with sampling noise and make the run
    # irreproducible; z_mean is what actually represents the image.
    _, z_mean, _ = encoder(batch, save_steps=False)
    return z_mean


# Trait extraction from segmentation masks

def measure(result, imgsz=256, min_vessel_px=4):
    # Pulls the four target measurements out of one YOLO segmentation result.
    traits = {
        'root_area_px': np.nan, 'root_diameter_px': np.nan,
        'stele_area_px': np.nan, 'stele_diameter_px': np.nan,
        'vessel_count': 0, 'vessel_total_area_px': 0.0,
        'stele_root_area_ratio': np.nan, 'vessel_stele_area_ratio': np.nan,
        'root_conf': np.nan, 'stele_conf': np.nan,
        'n_root_detections': 0, 'n_stele_detections': 0,
    }
    if result.masks is None or len(result.masks.data) == 0:
        return traits

    masks = result.masks.data.cpu().numpy()          # [N, h, w], may be padded
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    # Mask resolution can differ from the input; rescale areas to image pixels.
    mh, mw = masks.shape[1], masks.shape[2]
    scale = (imgsz * imgsz) / float(mh * mw)

    def area_of(m):
        return float((m > 0.5).sum()) * scale

    for cls_id, key in ((0, 'root'), (1, 'stele')):
        idx = np.where(classes == cls_id)[0]
        traits[f'n_{key}_detections'] = int(len(idx))
        if len(idx):
            # Exactly one root and one stele exist per image; if the model
            # returns several, the most confident is the intended one.
            best = idx[np.argmax(confs[idx])]
            area = area_of(masks[best])
            traits[f'{key}_area_px'] = area
            # Equivalent-circle diameter: the diameter a circle of this area
            # would have. More robust than a bounding-box side for non-circular
            # cross-sections.
            traits[f'{key}_diameter_px'] = 2.0 * np.sqrt(area / np.pi)
            traits[f'{key}_conf'] = float(confs[best])

    vessel_idx = np.where(classes == 2)[0]
    vessel_areas = [area_of(masks[i]) for i in vessel_idx]
    vessel_areas = [a for a in vessel_areas if a >= min_vessel_px]
    traits['vessel_count'] = int(len(vessel_areas))
    traits['vessel_total_area_px'] = float(sum(vessel_areas))

    if traits['root_area_px'] > 0:
        traits['stele_root_area_ratio'] = traits['stele_area_px'] / traits['root_area_px']
    if traits['stele_area_px'] > 0:
        traits['vessel_stele_area_ratio'] = traits['vessel_total_area_px'] / traits['stele_area_px']
    return traits


def overlay(image_rgb, result, alpha=0.45):
    # Draws masks over an image for visual inspection.
    colors = {0: (255, 60, 60), 1: (60, 255, 60), 2: (60, 160, 255)}
    canvas = image_rgb.astype(np.float32).copy()
    if result.masks is None:
        return canvas.astype(np.uint8)

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    H, W = image_rgb.shape[:2]
    for m, c in zip(masks, classes):
        mt = torch.from_numpy(m)[None, None]
        mt = F.interpolate(mt, size=(H, W), mode='nearest')[0, 0].numpy() > 0.5
        canvas[mt] = (1 - alpha) * canvas[mt] + alpha * np.array(colors[int(c)], np.float32)
    return canvas.astype(np.uint8)


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/evaluation/latent_average_test.py
    class cfg:
        # inputs
        # LiteVAE's own training set. Switch to
        # 'dataset/root_features_256/train/images' to run on the annotated W3
        # images instead (in domain for the segmenter, out of domain for LiteVAE).
        source_images = CROPPED_IMAGES_DIR
        litevae_checkpoint = LITEVAE_MODEL
        seg_weights = SEGMENTATION_MODEL

        output_dir = RESULTS_DIR / 'segmentation_results'

        # experiment
        n_sources = 12          # source images to draw
        n_pairs = 20            # latent-average pairs to generate
        # Blend weights. 0.5 is the average; the off-centre values trace the
        # path between two roots so a trait can be checked for moving smoothly
        # rather than jumping.
        alphas = [0.25, 0.5, 0.75]
        seed = 0

        imgsz = 256
        conf = 0.25             # segmentation confidence threshold
        min_vessel_px = 4       # ignore specks below this area

        save_images = True
        save_overlays = True
        device = 'cpu'          # 'cuda' if a dedicated GPU is free

    device = torch.device(cfg.device)
    out = Path(cfg.output_dir)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    for path, label in ((cfg.litevae_checkpoint, 'LiteVAE checkpoint'),
                        (cfg.seg_weights, 'segmentation weights'),
                        (cfg.source_images, 'source images')):
        if not Path(path).exists():
            raise SystemExit(f"missing {label}: {path}")

    encoder, decoder = load_litevae(cfg.litevae_checkpoint, device)
    seg = YOLO(str(cfg.seg_weights))

    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = sorted(p for p in Path(cfg.source_images).iterdir() if p.suffix in exts)
    if len(all_images) < 2:
        raise SystemExit(f"need at least 2 source images, found {len(all_images)}")

    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.n_sources, len(all_images))
    sources = [all_images[i] for i in rng.choice(len(all_images), size=n, replace=False)]
    print(f"Using {len(sources)} source images from {cfg.source_images.name}")

    batch = torch.stack([load_image(p, cfg.imgsz) for p in sources]).to(device)
    latents = encode(encoder, batch)
    print(f"Latents: {tuple(latents.shape)}")

    # build every image to be measured
    # kind: original (no LiteVAE at all), reconstruction (round trip),
    # average (latent blend). Comparing the three separates compression loss
    # from blending behaviour.
    records, to_segment = [], []

    for i, path in enumerate(sources):
        to_segment.append(('original', f'orig_{i:02d}', to_uint8(batch[i]),
                           {'source_a': path.name, 'source_b': '', 'alpha': np.nan}))

    with torch.no_grad():
        recon = decoder(latents, save_steps=False)
    for i, path in enumerate(sources):
        to_segment.append(('reconstruction', f'recon_{i:02d}', to_uint8(recon[i]),
                           {'source_a': path.name, 'source_b': '', 'alpha': np.nan}))

    pairs = list(combinations(range(len(sources)), 2))
    rng.shuffle(pairs)
    pairs = pairs[:cfg.n_pairs]

    blend_latents, blend_meta = [], []
    for a, b in pairs:
        for alpha in cfg.alphas:
            blend_latents.append((1 - alpha) * latents[a] + alpha * latents[b])
            blend_meta.append((a, b, alpha))

    if blend_latents:
        with torch.no_grad():
            blended = decoder(torch.stack(blend_latents), save_steps=False)
        for k, (a, b, alpha) in enumerate(blend_meta):
            to_segment.append((
                'average', f'avg_{a:02d}x{b:02d}_a{int(alpha * 100):03d}',
                to_uint8(blended[k]),
                {'source_a': sources[a].name, 'source_b': sources[b].name, 'alpha': alpha}))

    print(f"Generated {len(to_segment)} images "
          f"({len(sources)} original, {len(sources)} reconstruction, {len(blend_meta)} average)")

    # segment and measure
    print("Segmenting...")
    for kind, name, image, meta in to_segment:
        result = seg.predict(image[:, :, ::-1], conf=cfg.conf, imgsz=cfg.imgsz,
                             device=cfg.device, verbose=False)[0]
        traits = measure(result, cfg.imgsz, cfg.min_vessel_px)
        records.append({'kind': kind, 'name': name, **meta, **traits})

        if cfg.save_images:
            Image.fromarray(image).save(out / 'images' / f'{name}.png')
        if cfg.save_overlays:
            Image.fromarray(overlay(image, result)).save(out / 'overlays' / f'{name}.png')

    df = pd.DataFrame(records)
    df.to_csv(out / 'measurements.csv', index=False)

    # how well did each parent's traits predict the blend?
    # For a smooth latent space, a trait at blend weight alpha should sit near
    # the alpha-weighted mix of the two parents' reconstructions. Comparing
    # against reconstructions rather than originals keeps LiteVAE's compression
    # loss out of the comparison, so this measures the blending alone.
    recon_by_idx = {int(r['name'].split('_')[1]): r for _, r in
                    df[df.kind == 'reconstruction'].iterrows()}
    interp_rows = []
    for _, row in df[df.kind == 'average'].iterrows():
        a, b = row['name'].split('_')[1].split('x')
        ra, rb = recon_by_idx.get(int(a)), recon_by_idx.get(int(b))
        if ra is None or rb is None:
            continue
        alpha = row['alpha']
        for trait in ('root_area_px', 'stele_area_px', 'vessel_total_area_px',
                      'vessel_count', 'stele_root_area_ratio'):
            expected = (1 - alpha) * ra[trait] + alpha * rb[trait]
            interp_rows.append({
                'name': row['name'], 'alpha': alpha, 'trait': trait,
                'observed': row[trait], 'expected_linear': expected,
                'abs_error': abs(row[trait] - expected),
                'rel_error': (abs(row[trait] - expected) / expected
                              if expected and np.isfinite(expected) and expected != 0 else np.nan),
            })
    interp = pd.DataFrame(interp_rows)
    if not interp.empty:
        interp.to_csv(out / 'interpolation_linearity.csv', index=False)

    # summary
    traits = ['root_area_px', 'root_diameter_px', 'stele_area_px', 'stele_diameter_px',
              'vessel_count', 'vessel_total_area_px', 'stele_root_area_ratio']
    summary = df.groupby('kind')[traits].agg(['mean', 'std'])
    summary.to_csv(out / 'summary_by_kind.csv')

    detect = df.groupby('kind')[['n_root_detections', 'n_stele_detections',
                                 'root_conf', 'stele_conf']].mean()

    print("\n=== Trait means by image kind ===")
    print(summary.xs('mean', axis=1, level=1).to_string(float_format=lambda v: f'{v:.2f}'))
    print("\n=== Detection reliability (1.0 root/stele per image is correct) ===")
    print(detect.to_string(float_format=lambda v: f'{v:.3f}'))

    if not interp.empty:
        print("\n=== Latent-average linearity (vs. blend of the two reconstructions) ===")
        print(interp.groupby('trait')['rel_error'].agg(['mean', 'median'])
              .to_string(float_format=lambda v: f'{v:.3f}'))

    json_summary = {
        'n_sources': len(sources),
        'n_generated': len(to_segment),
        'alphas': cfg.alphas,
        'source_images': str(cfg.source_images),
        'trait_means_by_kind': summary.xs('mean', axis=1, level=1).to_dict(),
        'mean_relative_interpolation_error': (
            interp.groupby('trait')['rel_error'].mean().to_dict() if not interp.empty else {}),
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(json_summary, f, indent=2, default=float)

    print(f"\nWrote measurements.csv, summary_by_kind.csv, summary.json, "
          f"and {len(to_segment)} images/overlays to {out}")


if __name__ == '__main__':
    main()
