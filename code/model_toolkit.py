# Loads a trained root diffusion model once and exposes what a quick live test needs.
#
# Built for trying an idea on the spot - in a meeting, say - rather than for a full
# analysis run. Each piece wraps a function the analysis scripts already use
# (load_model, generate_batch, the segmenter's measure), so a number seen here is
# computed exactly as those scripts compute it.
#
#     from model_toolkit import load_session, save_grid, output_dir
#     s = load_session(300)                      # medium model, epoch 300
#     tokens = s.embed(['MEMA004', 'MEMA006'])   # SNP encoder only
#     images = s.generate('MEMA004', seeds=[0, 1, 2])
#     save_grid(images, output_dir('first_look') / 'mema004.png')
#
# Anything that takes a genotype accepts a name, a list of names, or founder codes
# (one row of 43,788 values from 1 to 8), so edited genotypes go anywhere real
# ones do. Heavy shared pieces - the SNP table, LiteVAE, the segmenter - load on
# first use and are shared by every session, so two checkpoints can be compared
# side by side without loading anything twice.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Puts code/ on the import path so this file can be imported from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    CROPPED_IMAGES_DIR, DIFFUSION_ONEHOT_MEDIUM_DIR, DIFFUSION_ONEHOT_MODEL,
    IMAGE_METADATA, LITEVAE_MODEL, PLAYGROUND_DIR, SEGMENTATION_MODEL,
    SNP_MATRIX_CACHE, SNP_PARQUET, best_checkpoint_path, checkpoint_path,
    pick_device, resolve_input, resolve_output,
)

from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import generate_batch, load_original
from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet

MEDIUM_NAME = 'diffusion_onehot_medium'

# The traits the segmenter reports, in the order genetic_fidelity_test.py uses.
TRAITS = ['root_diameter_px', 'stele_diameter_px', 'vessel_total_area_px',
          'vessel_count_cc', 'stele_root_diameter_ratio']

_shared = {}


def _log(message):
    print(f"[toolkit] {message}", flush=True)


# Which checkpoint a short name means. An int is an epoch of the medium model,
# since that is the model being compared across epochs; 'best' is its best
# validation checkpoint and 'original' the 209M model it replaced. Anything else
# is taken as a path.
def resolve_checkpoint(which):
    if isinstance(which, (int, np.integer)):
        return resolve_input(checkpoint_path(DIFFUSION_ONEHOT_MEDIUM_DIR, int(which), MEDIUM_NAME),
                             f'medium checkpoint for epoch {which}')
    if which == 'best':
        return resolve_input(best_checkpoint_path(DIFFUSION_ONEHOT_MEDIUM_DIR, MEDIUM_NAME),
                             'best medium checkpoint')
    if which == 'original':
        return resolve_input(DIFFUSION_ONEHOT_MODEL, 'original one-hot checkpoint')
    return resolve_input(which, 'diffusion checkpoint')


# Every checkpoint on disk the short names can reach, newest epoch last.
def available_checkpoints():
    folder = resolve_output(DIFFUSION_ONEHOT_MEDIUM_DIR)
    return sorted(p.name for p in folder.glob('*.pt')) if folder.exists() else []


# The SNP table as (genotype names, locus names, genotype x locus codes), from the
# cache when it matches the parquet on disk and rebuilt from the parquet when not.
def snp_table():
    if 'snp' in _shared:
        return _shared['snp']
    parquet = resolve_input(SNP_PARQUET, 'SNP parquet')
    stamp = f'{parquet.stat().st_size}:{parquet.stat().st_mtime_ns}'
    cache = resolve_output(SNP_MATRIX_CACHE)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if str(z['stamp']) == stamp:
                _shared['snp'] = (z['names'].tolist(), z['loci'].tolist(), z['matrix'])
                return _shared['snp']
    _log("building the SNP matrix cache from the parquet (once)...")
    names, loci, matrix = load_snp_data_from_parquet(parquet)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), loci=np.array(loci),
             matrix=np.asarray(matrix, dtype=np.float32), stamp=np.array(stamp))
    _shared['snp'] = (names, loci, np.asarray(matrix, dtype=np.float32))
    return _shared['snp']


def _metadata():
    if 'metadata' not in _shared:
        _shared['metadata'] = pd.read_csv(resolve_input(IMAGE_METADATA, 'image metadata'))
    return _shared['metadata']


def _litevae(device):
    key = ('litevae', str(device))
    if key not in _shared:
        from latent_diffusion.training.train_onehot import load_litevae, make_transforms
        encoder, decoder = load_litevae(resolve_input(LITEVAE_MODEL, 'LiteVAE checkpoint'), device)
        _shared[key] = (encoder, decoder, make_transforms(256, False))
    return _shared[key]


def _segmenter():
    if 'segmenter' not in _shared:
        from ultralytics import YOLO
        _shared['segmenter'] = YOLO(str(resolve_input(SEGMENTATION_MODEL, 'segmentation weights')))
    return _shared['segmenter']


def _scheduler(device):
    key = ('scheduler', str(device))
    if key not in _shared:
        scheduler = DiffusionScheduler()
        # Plain tensors rather than module buffers, so they are moved by hand.
        scheduler.betas = scheduler.betas.to(device)
        scheduler.alphas = scheduler.alphas.to(device)
        scheduler.alpha_bars = scheduler.alpha_bars.to(device)
        _shared[key] = scheduler
    return _shared[key]


class Session:

    def __init__(self, which, device=None):
        start = time.time()
        self.device = torch.device(device or pick_device())
        self.checkpoint = resolve_checkpoint(which)
        ckpt = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        self.epoch = ckpt.get('epoch')
        self.model_size = ckpt.get('model_size')
        self.held_out = set(ckpt.get('val_genotypes', []))
        del ckpt

        self.genotypes, self.loci, self.matrix = snp_table()
        self._row = {g: i for i, g in enumerate(self.genotypes)}
        self._locus = {name: i for i, name in enumerate(self.loci)}
        self.encoder, self.unet, self.unet_cfg = load_model(self.checkpoint, self.matrix, self.device)
        self.unet.set_store_attention(False)
        self.projector = self.encoder.projector
        self.founders = tuple(self.projector.founders)

        imaged = set(_metadata()['genotype'])
        self._group = {g: 'held out' if g in self.held_out
                       else 'trained' if g in imaged else 'never imaged'
                       for g in self.genotypes}
        self.name = f'epoch {self.epoch}' if self.epoch is not None else self.checkpoint.stem
        _log(f"{self.checkpoint.name} ready on {self.device} in {time.time() - start:.1f}s "
             f"({sum(v == 'held out' for v in self._group.values())} held-out genotypes)")

    def __repr__(self):
        return f"<Session {self.checkpoint.name} epoch={self.epoch} device={self.device}>"

    # Genotypes and their data

    # 'trained', 'held out' (imaged, but kept out of this model's training) or
    # 'never imaged' (SNP data only, so no model has ever seen an image of it).
    def group(self, genotype):
        return self._group[genotype]

    def genotypes_in(self, group):
        return [g for g in self.genotypes if self._group[g] == group]

    # Founder codes as a genotype x locus float32 array, from a name, one row of
    # codes, several rows, or a list mixing names and edited genotypes - the
    # natural way to put a real genotype beside an edit of it.
    def codes(self, x):
        if isinstance(x, str):
            return self.matrix[[self._row[x]]]
        if isinstance(x, (list, tuple)):
            return np.concatenate([self.codes(item) for item in x])
        return np.atleast_2d(np.asarray(x, dtype=np.float32))

    # Human-readable labels for whatever codes() accepted; edited genotypes are
    # numbered in the order they appear.
    def labels(self, x):
        if isinstance(x, str):
            return [x]
        items = x if isinstance(x, (list, tuple)) else list(self.codes(x))
        labels, n_custom = [], 0
        for item in items:
            if isinstance(item, str):
                labels.append(item)
            else:
                for _ in range(len(self.codes(item))):
                    labels.append(f'custom {n_custom}')
                    n_custom += 1
        return labels

    def image_paths(self, genotype):
        meta = _metadata()
        image_dir = resolve_input(CROPPED_IMAGES_DIR, 'cropped images')
        return [image_dir / f for f in meta.loc[meta.genotype == genotype, 'new_filename']
                if (image_dir / f).exists()]

    # The genotype's real cropped images at model resolution, as uint8 RGB.
    def real_images(self, genotype, n=None):
        paths = self.image_paths(genotype)[:n]
        return np.stack([load_original(p, 256) for p in paths]) if paths else np.empty((0, 256, 256, 3), np.uint8)

    # Share of loci inherited from each founder; missing calls are left out.
    def founder_share(self, x):
        codes = self.codes(x)
        return pd.DataFrame({f: (codes == f).mean(axis=1) for f in self.founders},
                            index=self.labels(x))

    # SNP encoder

    # The 168 PCA scores the encoder's MLP receives.
    def project(self, x):
        return self.projector.transform(self.codes(x))

    # The 8 x 256 conditioning tokens the UNet attends to.
    def embed(self, x, batch_size=64):
        codes = self.codes(x)
        out = []
        with torch.no_grad():
            for s in range(0, len(codes), batch_size):
                out.append(self.encoder(torch.tensor(codes[s:s + batch_size], device=self.device)).cpu().numpy())
        return np.concatenate(out)

    # How far apart two genotypes are at each stage of the encoder, and how many
    # loci actually differ between them.
    def distance(self, a, b):
        ca, cb = self.codes(a)[0], self.codes(b)[0]
        pa, pb = self.project(ca)[0], self.project(cb)[0]
        ta, tb = self.embed(ca)[0].ravel(), self.embed(cb)[0].ravel()
        return {'loci_differing': int((ca != cb).sum()),
                'pca_distance': float(np.linalg.norm(pa - pb)),
                'token_distance': float(np.linalg.norm(ta - tb)),
                'token_cosine': float(ta @ tb / (np.linalg.norm(ta) * np.linalg.norm(tb)))}

    # The real genotypes closest to x in 'pca' or 'tokens' space, with each
    # neighbour's group and share of identical loci.
    def nearest(self, x, k=5, space='tokens'):
        query = (self.project(x) if space == 'pca' else self.embed(x).reshape(1, -1))[0]
        cache_key = f'_all_{space}'
        if not hasattr(self, cache_key):
            setattr(self, cache_key, self.project(self.matrix) if space == 'pca'
                    else self.embed(self.matrix).reshape(len(self.matrix), -1))
        everyone = getattr(self, cache_key)
        dist = np.linalg.norm(everyone - query, axis=1)
        order = np.argsort(dist)[:k]
        codes = self.codes(x)[0]
        return pd.DataFrame({'genotype': [self.genotypes[i] for i in order],
                             'group': [self._group[self.genotypes[i]] for i in order],
                             f'{space}_distance': dist[order],
                             'shared_loci': [(self.matrix[i] == codes).mean() for i in order]})

    # Editing genotypes

    # Every locus set to one founder - that founder's line, as the model sees it.
    def founder_line(self, founder):
        return np.full((1, len(self.loci)), float(founder), dtype=np.float32)

    # Locus indices from an index, a slice, a list of indices or a list of gene
    # names like 'Zm00001d027231'.
    def locus_indices(self, loci):
        if isinstance(loci, slice):
            return np.arange(len(self.loci))[loci]
        loci = [loci] if isinstance(loci, (str, int, np.integer)) else list(loci)
        return np.array([self._locus[l] if isinstance(l, str) else int(l) for l in loci])

    # x with the given loci replaced, either by one founder everywhere (an int)
    # or by the same loci of another genotype. Loci are in the table's column
    # order, sorted by gene ID, which is not strictly genomic position.
    def swap(self, x, loci, source):
        codes = self.codes(x)[0].copy()
        idx = self.locus_indices(loci)
        codes[idx] = (float(source) if isinstance(source, (int, float, np.integer))
                      else self.codes(source)[0][idx])
        return codes[None, :]

    # A new line stitched from contiguous blocks of real parents, the realistic
    # shape of a new cross. Each parent gets an equal share of blocks.
    def recombinant(self, parents, n_blocks=20, seed=0):
        rng = np.random.default_rng(seed)
        parent_codes = self.codes(parents)
        blocks = np.array_split(np.arange(len(self.loci)), n_blocks)
        owner = np.resize(np.arange(len(parent_codes)), n_blocks)
        rng.shuffle(owner)
        return np.concatenate([parent_codes[o, b] for o, b in zip(owner, blocks)])[None, :]

    # Generating and measuring

    # Images for every genotype in x and every seed, shaped
    # (genotypes, seeds, 256, 256, 3). The same seed starts from the same noise
    # for every genotype, so a column differs only by genotype.
    def generate(self, x, seeds=(0,), steps=50, batch_size=8):
        start = time.time()
        codes = self.codes(x)
        seeds = list(seeds)
        _, decoder, _ = _litevae(self.device)
        jobs = [(g, s) for g in range(len(codes)) for s in seeds]
        out = np.empty((len(codes), len(seeds), 256, 256, 3), dtype=np.uint8)
        for b in range(0, len(jobs), batch_size):
            chunk = jobs[b:b + batch_size]
            images = generate_batch(
                self.encoder, self.unet, _scheduler(self.device), decoder,
                torch.tensor(codes[[g for g, _ in chunk]], device=self.device),
                [s for _, s in chunk], self.device, (self.unet_cfg['latent_channels'], 32, 32), steps)
            for (g, s), image in zip(chunk, images):
                out[g, seeds.index(s)] = image
        _log(f"generated {len(jobs)} images at {steps} steps in {time.time() - start:.1f}s")
        return out

    # Denoising error for each image under each conditioning genotype, with the
    # same noise for every condition, so the only difference between columns is
    # the genotype. Lower is better; a genotype that describes the image should
    # score lower than one that does not. This is the measurement behind
    # analyze_genotype_memorization.py, applied to any images and genotypes.
    def denoise_loss(self, images, conditions, timesteps=(100, 250, 500, 750), seed=0):
        encoder, _, transform = _litevae(self.device)
        images = np.asarray(images).reshape(-1, 256, 256, 3)
        batch = torch.stack([transform(Image.fromarray(im)) for im in images]).to(self.device)
        scheduler = _scheduler(self.device)
        cond_codes = self.codes(conditions)
        labels = self.labels(conditions)
        rows = []
        with torch.no_grad():
            torch.manual_seed(seed)
            # In chunks, so a long list of images fits in a small GPU's memory.
            z = torch.cat([encoder(batch[s:s + 16], save_steps=False)[0]
                           for s in range(0, len(batch), 16)])
            noise = torch.randn(z.shape, generator=torch.Generator().manual_seed(seed + 1)).to(self.device)
            tokens = self.encoder(torch.tensor(cond_codes, device=self.device))
            for t in timesteps:
                t_batch = torch.full((len(z),), t, device=self.device, dtype=torch.long)
                z_t = scheduler.add_noise(z, noise, t_batch)
                for c, label in enumerate(labels):
                    emb = tokens[c:c + 1].expand(len(z), -1, -1)
                    loss = F.mse_loss(self.unet(z_t, t_batch, emb), noise, reduction='none')
                    for i, v in enumerate(loss.mean(dim=(1, 2, 3)).cpu().numpy()):
                        rows.append({'image': i, 'condition': label, 'timestep': t, 'loss': float(v)})
        frame = pd.DataFrame(rows)
        return frame.pivot_table(index='image', columns='condition', values='loss', aggfunc='mean')[labels]


# A session for one checkpoint: an epoch number of the medium model, 'best',
# 'original' or a path. Loading the same one twice returns the first.
def load_session(which=300, device=None):
    key = ('session', str(which), str(device))
    if key not in _shared:
        _shared[key] = Session(which, device)
    return _shared[key]


# Trait measurements from the segmenter for any images, one row per image. The
# segmenter is still being improved, so read these as approximate.
def measure(images):
    from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
    segmenter = _segmenter()
    device = pick_device()
    images = np.asarray(images).reshape(-1, 256, 256, 3)
    rows = []
    for image in images:
        traits, _ = segment(segmenter, image, 0.25, device)
        rows.append({t: traits[t] for t in TRAITS})
    return pd.DataFrame(rows)


# A folder for one test's output, under results/playground/.
def output_dir(name):
    folder = resolve_output(PLAYGROUND_DIR / name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# Saves a grid of images - anything shaped (rows, cols, H, W, 3), or a list of
# rows - with optional row and column labels drawn onto the panels.
def save_grid(images, path, row_labels=None, col_labels=None, divider=4):
    rows = [list(r) for r in (images if isinstance(images, list) else np.asarray(images))]
    if np.asarray(rows[0][0]).ndim == 2:
        rows = [rows]
    h, w = np.asarray(rows[0][0]).shape[:2]
    n_cols = max(len(r) for r in rows)
    canvas = Image.new('RGB', (n_cols * (w + divider) - divider, len(rows) * (h + divider) - divider),
                       (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            x, y = c * (w + divider), r * (h + divider)
            canvas.paste(Image.fromarray(np.asarray(panel, dtype=np.uint8)), (x, y))
            text = ' | '.join(t for t in (row_labels[r] if row_labels and c == 0 else None,
                                          col_labels[c] if col_labels and r == 0 else None) if t)
            if text:
                draw.rectangle([x + 2, y + 2, x + 6 + 6 * len(text), y + 16], fill=(0, 0, 0))
                draw.text((x + 4, y + 4), text, fill=(255, 255, 255))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    _log(f"saved {path}")
    return path
