# Generates images for chosen lines from a trained d2d diffusion checkpoint.
# Inference only. Uses the project's own sampling code (code/widiv/widiv_latents.py)
# through the launcher's in-memory path redirection, so nothing outside the d2d
# workspace is read for data or written.
#
# Same seeds for every line, matched starting noise, 50 DDIM steps: differences
# between lines come from the genotype, not the noise. Resumable: images already
# on disk are kept.
from pathlib import Path

import numpy as np
import pandas as pd

from .launch_trainer import prepare
from .project import ROOT


def find_checkpoint(ds, preferred=''):
    models = ROOT / 'models'
    cands = sorted(models.glob(f'diffusion_{ds.name}*/*.pt')) if models.exists() else []
    if preferred:
        run, _, epoch = preferred.partition(' epoch ')
        for c in cands:
            if run in str(c.parent.name) and (not epoch or c.stem.endswith(f'_epoch_{int(epoch)}')):
                return c
    best = [c for c in cands if c.stem.endswith('_best')]
    return best[0] if best else (cands[-1] if cands else None)


def generate(ds, checkpoint, lines_role, n_seeds, out_dir, steps=50, batch=8, progress=None, lines=None,
             seed_start=0, genotypes=None):
    """Images <line>_seed<s>.png for the checkpoint's validation or test lines, or for
    `lines` when given (any genotyped line), seeds seed_start .. seed_start + n_seeds - 1.
    genotypes: {label: dosage row} for edited genotypes (a gene switched on, say); their
    labels are used as the line names and must not be real line names."""
    import torch
    prepare(ds)
    from latent_diffusion.models.unet import DenoisingUNet
    from widiv_latents import LatentScaler, ScaledClampScheduler, ScaledDecoder, generate_widiv_batch
    from widiv_litevae import load_widiv_litevae
    from widiv_paths import WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_LITEVAE_MODEL
    from widiv_snp_encoding import load_widiv_genotypes
    from widiv_window_encoding import load_widiv_encoder
    from PIL import Image

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if genotypes is not None:
        lines = list(genotypes)
    elif lines is None:
        lines = list(ck['val_genotypes'] if lines_role == 'validation' else ck['test_genotypes'])
    lines = [str(l) for l in lines]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(seed_start, seed_start + n_seeds)
    jobs = [(g, s) for g in lines for s in seeds if not (out_dir / f'{g}_seed{s}.png').exists()]
    from . import resources
    batch = resources.gpu_batch(batch)       # smaller when retrying after running out of memory
    if jobs:
        _, dec_vae = load_widiv_litevae(WIDIV_LITEVAE_MODEL, dev)
        sched = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
        sched.betas, sched.alphas, sched.alpha_bars = (sched.betas.to(dev), sched.alphas.to(dev),
                                                       sched.alpha_bars.to(dev))
        enc, proj = load_widiv_encoder(ck, dev)
        ucfg = dict(ck['unet_config'])
        ucfg['channel_mult'] = tuple(ucfg['channel_mult'])
        unet = DenoisingUNet(**ucfg)
        unet.load_state_dict(ck['unet_state_dict'])
        unet.set_store_attention(False)
        unet = unet.to(dev).eval()
        scaler = LatentScaler.from_checkpoint(ck)
        sched.set_scaler(scaler)
        decoder = ScaledDecoder(dec_vae, scaler)
        names, snps, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
        # The projector indexes markers by position: a workspace matrix whose columns
        # differ from the ones the model was trained on would silently feed it the
        # wrong markers.
        trained = getattr(proj, 'snp_names', None)
        if trained is not None and list(trained) != list(snps):
            raise SystemExit(f'{checkpoint.name} was trained on {len(trained)} markers that differ from the '
                             f"workspace's {len(snps)}; regenerate with the genotype matrix it was trained on")
        row = {g: i for i, g in enumerate(names)}
        need = sorted({g for g, _ in jobs})
        rows = np.stack([np.asarray(genotypes[g], dtype=dosage.dtype) if genotypes is not None else dosage[row[g]]
                         for g in need])
        cond = torch.tensor(proj.project_lines(rows, need), dtype=torch.float32, device=dev)
        local = {g: i for i, g in enumerate(need)}
        latent = (ucfg['latent_channels'], ck.get('latent_size', 64), ck.get('latent_size', 64))
        with torch.no_grad():
            for b in range(0, len(jobs), batch):
                part = jobs[b:b + batch]
                imgs = generate_widiv_batch(enc, unet, sched, decoder, cond[[local[g] for g, _ in part]],
                                            [s for _, s in part], dev, latent, steps,
                                            latent_mean=ck.get('latent_channel_mean'))
                for (g, s), im in zip(part, imgs):
                    Image.fromarray(np.asarray(im, dtype=np.uint8)).save(out_dir / f'{g}_seed{s}.png')
                if progress:
                    progress(min(b + batch, len(jobs)), len(jobs))
    files = sorted(out_dir.glob('*_seed*.png'))
    return pd.DataFrame({'image': [f.name for f in files],
                         'line': [f.stem.rsplit('_seed', 1)[0] for f in files],
                         'seed': [int(f.stem.rsplit('_seed', 1)[1]) for f in files]})
