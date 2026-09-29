# The reverse diffusion process for a few genotypes: the latent at chosen steps
# on top, and that latent decoded underneath.
#
# Works on a root or a seed checkpoint; which one is read from the checkpoint,
# along with the SNP table and the autoencoder that go with it. Genotypes are
# taken evenly from the ones the checkpoint trained on and the ones it held
# out, so the two can be compared, unless cfg.genotypes names them.
#
# The same noise seed is used for every genotype, so the columns of two figures
# start from identical noise and differ only by genotype.
#
# Writes trajectory_<genotype>.png per genotype and summary.json.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    SEED_DIFFUSION_MODEL, SEED_RESULTS_DIR, apply_overrides, pick_device,
    resolve_input, resolve_output,
)

from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.utils.dataset_inputs import (
    checkpoint_dataset, load_decoder, load_snp_table, metadata_for,
)


# Lag-1 spatial autocorrelation of one [H, W] map. A channel carrying layout is
# smooth and scores high; one carrying fine detail sits near zero.
def autocorrelation(a):
    a = a - a.mean()
    denom = float((a * a).sum()) or 1.0
    return float(((a[1:, :] * a[:-1, :]).sum() + (a[:, 1:] * a[:, :-1]).sum()) / (2 * denom))


# (noise, [(step, t, latent, image), ...], final latent) for one genotype.
@torch.no_grad()
def run_trajectory(encoder, unet, scheduler, decoder, snp, seed, latent_shape,
                   num_steps, keep, device):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    z_t = torch.randn(1, *latent_shape, generator=gen).to(device)
    noise = z_t.clone()
    emb = encoder(torch.tensor(snp[None], dtype=torch.float32, device=device))
    timesteps = scheduler.get_timesteps(num_steps, device)
    snapshots = []
    for i, t in enumerate(timesteps):
        tb = torch.full((1,), int(t.item()), device=device, dtype=torch.long)
        tp = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
        z_t = scheduler.denoise_step(z_t, unet(z_t, tb, emb), tb,
                                     torch.full((1,), tp, device=device, dtype=torch.long))
        if i in keep:
            image = torch.clamp(decoder(z_t, save_steps=False), -1, 1)
            snapshots.append((i, int(t.item()), z_t.clone(), image))
    return noise, snapshots, z_t


def to_rgb(image):
    a = image[0].permute(1, 2, 0).cpu().numpy()
    return np.clip((a + 1) / 2, 0, 1)


def save_trajectory(name, split, dataset, epoch, seed, noise, noise_image,
                    snapshots, channel, num_steps, save_path):
    cols = [('noise', None, noise, noise_image)] + [
        (f'step {i + 1}/{num_steps}', t, z, img) for i, t, z, img in snapshots]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.1 * len(cols), 4.7), squeeze=False)
    for c, (title, t, z, img) in enumerate(cols):
        axes[0][c].imshow(z[0, channel].cpu().numpy(), cmap='RdBu_r',
                          interpolation='nearest')
        axes[0][c].set_title(title if t is None else f'{title}\nt = {t}', fontsize=9.5)
        axes[1][c].imshow(to_rgb(img))
        for ax in (axes[0][c], axes[1][c]):
            ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_ylabel(f'latent\nchannel {channel}', fontsize=9.5, rotation=0,
                          ha='right', va='center', labelpad=32)
    axes[1][0].set_ylabel('decoded', fontsize=9.5, rotation=0, ha='right',
                          va='center', labelpad=32)
    fig.suptitle(f'Reverse process: {name} ({split}), {dataset} model epoch {epoch}, '
                 f'noise seed {seed}', fontsize=12)
    fig.tight_layout(rect=[0.03, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/sampling_trajectory.py
    class cfg:
        checkpoint = SEED_DIFFUSION_MODEL
        output_dir = SEED_RESULTS_DIR / 'sampling_trajectory'
        pca_cache = SEED_RESULTS_DIR / 'pca.pkl'
        # None -> the autoencoder and SNP table of the checkpoint's own dataset.
        litevae_checkpoint = None
        snp_parquet = None

        # Genotypes to draw, by name. None -> n_trained taken evenly from the
        # genotypes the checkpoint trained on and n_held_out from those it did not.
        genotypes = None
        n_trained = 2
        n_held_out = 2

        seed = 0
        sampling_steps = 50
        # Steps decoded and shown, spread across the run including the last. The
        # starting noise is always shown as well, as the first column.
        n_show = 6
        latent_size = 32
        device = pick_device()

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_input(cfg.checkpoint, 'diffusion checkpoint')
    dataset = checkpoint_dataset(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    held_out = set(ckpt.get('val_genotypes') or [])
    epoch = ckpt.get('epoch')
    del ckpt
    print(f"Device: {device}\nCheckpoint: {checkpoint_path}\nDataset: {dataset}\nOutput: {out}")

    names, _, snp_matrix = load_snp_table(dataset, cfg.snp_parquet)
    snp_matrix = np.asarray(snp_matrix)
    row_of = {n: i for i, n in enumerate(names)}

    if cfg.genotypes:
        missing = [g for g in cfg.genotypes if g not in row_of]
        if missing:
            raise SystemExit(f"genotype(s) not in the SNP table: {missing}")
        chosen = list(cfg.genotypes)
    else:
        # Trained means imaged and not held out: a genotype with SNP data but no
        # image was never seen in training either, so it is not counted as trained.
        meta_path, _ = metadata_for(dataset)
        imaged = set(pd.read_csv(resolve_input(meta_path, 'image metadata'))['genotype'])
        trained = sorted(g for g in imaged if g in row_of and g not in held_out)
        unseen = sorted(g for g in held_out if g in row_of)

        def evenly(items, n):
            if not items or n <= 0:
                return []
            idx = np.linspace(0, len(items) - 1, min(n, len(items))).round().astype(int)
            return [items[i] for i in sorted(set(idx))]
        chosen = evenly(trained, cfg.n_trained) + evenly(unseen, cfg.n_held_out)
        if not held_out:
            print("  the checkpoint records no held-out genotypes, so all are drawn "
                  "from the trained set")
    if not chosen:
        raise SystemExit("no genotypes to draw")

    encoder, unet, unet_cfg = load_model(checkpoint_path, snp_matrix, device,
                                         pca_cache=str(resolve_output(cfg.pca_cache)))
    unet.set_store_attention(False)
    decoder = load_decoder(dataset, device, unet_cfg['latent_channels'],
                           cfg.litevae_checkpoint)
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    scheduler = DiffusionScheduler()
    # Plain tensors rather than module buffers, so they are moved by hand.
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    keep = set(np.linspace(0, cfg.sampling_steps - 1, cfg.n_show).round().astype(int).tolist())

    rows = []
    for genotype in chosen:
        split = 'held out' if genotype in held_out else 'trained'
        noise, snapshots, final = run_trajectory(
            encoder, unet, scheduler, decoder, snp_matrix[row_of[genotype]],
            cfg.seed, latent_shape, cfg.sampling_steps, keep, device)
        with torch.no_grad():
            noise_image = torch.clamp(decoder(noise, save_steps=False), -1, 1)

        # One channel is drawn, the one carrying the most spatial layout in the
        # finished latent, and it is the same channel in every column. Which
        # channel that is depends on the autoencoder, so it is chosen rather than
        # fixed at 0, and named on the figure.
        scores = [autocorrelation(final[0, c].cpu().numpy())
                  for c in range(final.shape[1])]
        channel = int(np.argmax(scores))

        path = out / f'trajectory_{genotype}.png'
        save_trajectory(genotype, split, dataset, epoch, cfg.seed, noise, noise_image,
                        snapshots, channel, cfg.sampling_steps, path)
        rows.append({'genotype': genotype, 'split': split, 'channel_shown': channel,
                     'channel_autocorrelation': [round(v, 3) for v in scores],
                     'figure': path.name})
        print(f"  {genotype:<14} {split:<9} channel {channel}  -> {path.name}")

    (out / 'summary.json').write_text(json.dumps({
        'checkpoint': str(checkpoint_path), 'dataset': dataset, 'epoch': epoch,
        'seed': cfg.seed, 'sampling_steps': cfg.sampling_steps,
        'steps_shown': sorted(int(k) + 1 for k in keep), 'genotypes': rows,
    }, indent=2))
    print(f"\nWrote {len(rows)} trajectory figures and summary.json to {out}")


if __name__ == '__main__':
    main()
