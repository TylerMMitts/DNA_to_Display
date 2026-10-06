# Tests whether a model uses the genotype to describe a plant or to recall one.
#
# Each image is noised the same way twice and denoised once with its own
# genotype and once with another genotype from the same split. If the genotype
# carries real information about anatomy, the right one should help on
# genotypes the model never trained on, not only on ones it did.
#
# The first one-hot model failed this cleanly: the right genotype improved
# denoising by 22% on training genotypes and by 0.1% on held-out ones, and a
# training image given the wrong genotype scored exactly like an unseen one. The
# genotype was acting as a key to memorised images.
#
# Writes per_image.csv, per_timestep.csv, summary.json and memorization.png.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    CROPPED_IMAGES_DIR, DIFFUSION_ONEHOT_MODEL, IMAGE_METADATA, KINSHIP_MATRIX,
    LITEVAE_MODEL, RESULTS_DIR, SNP_PARQUET, apply_overrides, pick_device,
    resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.analysis.analyze_snp_attention import load_model, load_similarity
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
# The loader and transforms are taken from the root trainer rather than written
# again here, so images are prepared exactly as they were during training.
from latent_diffusion.training.train_onehot import load_litevae, make_transforms


# A partner genotype for every image, drawn from its own split and never its own.
#
# The least related genotype in that split, by kinship, rather than a random one.
# The pools here are small - 27 held-out lines - and every line in this population
# descends from the same eight founders, so a random partner often shares long
# stretches of genome with the right one. That makes the wrong condition easier
# than it should be and shrinks the measured gain. Taking the least related line
# makes the contrast as stark as this population allows.
#
# Ties and genotypes missing from the kinship matrix fall back to a seeded random
# pick, so the choice is always defined and always reproducible. Returns the
# partners and how related each pair is, which the summary reports: a "wrong"
# genotype that is still a close relative is worth knowing about.
def wrong_partners(genotypes, splits, seed, similarity=None, names=None,
                   mode='least related'):
    rng = np.random.default_rng(seed)
    index_of = {n: i for i, n in enumerate(names or [])}
    partner = np.empty(len(genotypes), dtype=object)
    relatedness = np.full(len(genotypes), np.nan)

    for split in np.unique(splits):
        pool = sorted(set(genotypes[splits == split]))
        if len(pool) < 2:
            raise SystemExit(f"the {split} split has fewer than two genotypes, "
                             "so there is no other genotype to swap in")
        for i in np.flatnonzero(splits == split):
            others = [g for g in pool if g != genotypes[i]]
            scored = []
            if mode == 'least related' and similarity is not None and genotypes[i] in index_of:
                # Sorted pool, so equal kinship always resolves the same way.
                scored = [(similarity[index_of[genotypes[i]], index_of[g]], g)
                          for g in others if g in index_of]
            if scored:
                relatedness[i], partner[i] = min(scored)
                continue
            partner[i] = rng.choice(others)
            if similarity is not None and genotypes[i] in index_of and partner[i] in index_of:
                relatedness[i] = similarity[index_of[genotypes[i]], index_of[partner[i]]]
    return partner, relatedness


# Mean kinship between every pair of different genotypes within a split, as the
# reference the chosen partners are judged against.
def pool_relatedness(genotypes, splits, split, similarity, names):
    index_of = {n: i for i, n in enumerate(names or [])}
    pool = [g for g in sorted(set(genotypes[splits == split])) if g in index_of]
    values = [similarity[index_of[a], index_of[b]]
              for k, a in enumerate(pool) for b in pool[k + 1:]]
    return float(np.mean(values)) if values else np.nan


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/analyze_genotype_memorization.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        litevae_checkpoint = LITEVAE_MODEL
        snp_parquet = SNP_PARQUET
        metadata_path = IMAGE_METADATA
        image_dir = CROPPED_IMAGES_DIR
        output_dir = RESULTS_DIR / 'genotype_memorization'

        # Which genotype stands in as the wrong one. 'least related' picks the
        # furthest line in the same split by kinship, so the two conditions are
        # as different as the population allows; 'random' is a seeded draw from
        # the split, which is easier and was what this test used before.
        partner = 'least related'
        # None falls back to correlation between raw SNP vectors.
        kinship = KINSHIP_MATRIX

        # Spread across the schedule. Memorisation shows most in the middle:
        # at the lowest noise every model does well and at the highest none can.
        timesteps = [20, 100, 250, 500, 750, 950]

        # None -> every image with a genotype and a file. A small number is a
        # quick check that runs the same code on fewer images.
        max_images = None
        batch_size = 32
        image_size = 256
        seed = 0
        device = pick_device()

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_input(cfg.checkpoint, 'diffusion checkpoint')
    print(f"Device: {device}\nCheckpoint: {checkpoint_path}\nOutput: {out}")

    names, _, snp_matrix = load_snp_data_from_parquet(
        resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)
    row_of = {n: i for i, n in enumerate(names)}

    # The split is read from the checkpoint, so held-out means held out of this
    # model's own training. A model warm-started from another one may still have
    # seen these genotypes through that model's weights.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'val_genotypes' not in ckpt:
        raise SystemExit(f"{checkpoint_path.name} records no validation split, so "
                         "there is no way to tell trained genotypes from held-out ones")
    held_out = set(ckpt['val_genotypes'])
    # A run trained on a subset of lines (n_train_genotypes) records which ones;
    # the lines it left out are neither trained nor held out, so they are not
    # scored. Older checkpoints trained on every line outside the split.
    trained_lines = set(ckpt['train_genotypes']) if ckpt.get('train_genotypes') else None
    condition_noise = float(ckpt.get('condition_noise', 0.0))
    epoch = ckpt.get('epoch')
    del ckpt

    snp_encoder, unet, _ = load_model(checkpoint_path, snp_matrix, device)
    unet.set_store_attention(False)
    litevae_encoder, _ = load_litevae(
        resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint'), device)

    scheduler = DiffusionScheduler()
    # Plain tensors rather than module buffers, so they are moved by hand.
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    image_dir = resolve_input(cfg.image_dir, 'image directory')
    metadata = pd.read_csv(resolve_input(cfg.metadata_path, 'image metadata'))
    items = [(image_dir / r.new_filename, r.genotype) for r in metadata.itertuples()
             if r.genotype in row_of and (image_dir / r.new_filename).exists()
             and (trained_lines is None or r.genotype in trained_lines
                  or r.genotype in held_out)]
    if cfg.max_images is not None:
        items = items[:cfg.max_images]
    if not items:
        raise SystemExit("no images have both a file and a genotype")

    genotypes = np.array([g for _, g in items])
    splits = np.array(['held out' if g in held_out else 'trained' for g in genotypes])
    for split in ('trained', 'held out'):
        sel = splits == split
        print(f"  {split:9s} {int(sel.sum()):4d} images from "
              f"{len(set(genotypes[sel])):3d} genotypes")
    similarity, sim_names, sim_source = load_similarity(cfg.kinship, names, snp_matrix)
    partners, relatedness = wrong_partners(genotypes, splits, cfg.seed, similarity,
                                           sim_names, cfg.partner)
    print(f"  wrong genotype: {cfg.partner} in the same split, by {sim_source}")
    for split in ('trained', 'held out'):
        sel = splits == split
        print(f"    {split:9s} mean relatedness to its partner "
              f"{np.nanmean(relatedness[sel]):.3f}  "
              f"(average pair in that split {pool_relatedness(genotypes, splits, split, similarity, sim_names):.3f})")

    # Latents are encoded once, seeded, so every timestep sees the same ones.
    transform = make_transforms(cfg.image_size, False)
    torch.manual_seed(cfg.seed)
    latents = []
    with torch.no_grad():
        for start in range(0, len(items), cfg.batch_size):
            batch = torch.stack([transform(Image.open(p).convert('RGB'))
                                 for p, _ in items[start:start + cfg.batch_size]])
            latents.append(litevae_encoder(batch.to(device), save_steps=False)[0].cpu())
    latents = torch.cat(latents)

    with torch.no_grad():
        embeddings = snp_encoder(torch.tensor(snp_matrix, dtype=torch.float32, device=device))
    embedding_of = {n: embeddings[i] for i, n in enumerate(names)}

    # One noise draw per image, shared by every timestep and both conditions,
    # so the only thing that differs between the two losses is the genotype.
    noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(cfg.seed + 1))

    rows = []
    with torch.no_grad():
        for t in cfg.timesteps:
            for start in range(0, len(items), cfg.batch_size):
                end = start + cfg.batch_size
                z = latents[start:end].to(device)
                eps = noise[start:end].to(device)
                t_batch = torch.full((len(z),), t, device=device, dtype=torch.long)
                z_t = scheduler.add_noise(z, eps, t_batch)
                for condition, keys in (('own', genotypes[start:end]),
                                        ('wrong', partners[start:end])):
                    emb = torch.stack([embedding_of[g] for g in keys])
                    loss = F.mse_loss(unet(z_t, t_batch, emb), eps, reduction='none')
                    for k, value in enumerate(loss.mean(dim=(1, 2, 3)).cpu().numpy()):
                        rows.append({'image': items[start + k][0].name,
                                     'genotype': genotypes[start + k],
                                     'wrong_genotype': partners[start + k],
                                     'relatedness': float(relatedness[start + k]),
                                     'split': splits[start + k], 'timestep': t,
                                     'condition': condition, 'loss': float(value)})
            print(f"  t={t} done")

    per_image = pd.DataFrame(rows)
    per_image.to_csv(out / 'per_image.csv', index=False)
    per_timestep = (per_image.groupby(['timestep', 'split', 'condition'])['loss']
                    .mean().unstack(['split', 'condition']))
    per_timestep.to_csv(out / 'per_timestep.csv')

    means = per_image.groupby(['split', 'condition'])['loss'].mean()

    # How much worse each split gets when handed the wrong genotype. The
    # held-out figure is the one that matters: it is only above zero if the
    # model learned something about genotypes that carries over.
    def gain(split):
        return float(means[(split, 'wrong')] / means[(split, 'own')] - 1.0)

    summary = {
        'checkpoint': str(checkpoint_path),
        'epoch': epoch,
        'n_images': len(items),
        'n_trained_genotypes': len(set(genotypes[splits == 'trained'])),
        'n_held_out_genotypes': len(set(genotypes[splits == 'held out'])),
        'condition_noise': condition_noise,
        'timesteps': list(cfg.timesteps),
        'partner': cfg.partner,
        'similarity_source': sim_source,
        'partner_relatedness': {s: float(np.nanmean(relatedness[splits == s]))
                                for s in ('trained', 'held out')},
        'pool_relatedness': {s: pool_relatedness(genotypes, splits, s, similarity, sim_names)
                             for s in ('trained', 'held out')},
        'loss': {f'{s} / {c}': float(v) for (s, c), v in means.items()},
        'genotype_gain_trained': gain('trained'),
        'genotype_gain_held_out': gain('held out'),
        'held_out_vs_trained_gap': float(means[('held out', 'own')]
                                         / means[('trained', 'own')] - 1.0),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))

    print(f"\nRight genotype vs wrong one:")
    print(f"  trained genotypes   {summary['genotype_gain_trained']:+.1%}")
    print(f"  held-out genotypes  {summary['genotype_gain_held_out']:+.1%}")
    print("A large trained gain with a held-out gain near zero means the genotype is "
          "being used\nto recall training images rather than to describe the plant.")

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    splits_order = ['trained', 'held out']
    x = np.arange(len(splits_order))
    for offset, condition, color in ((-0.2, 'own', '#F2A65A'), (0.2, 'wrong', '#5FC8D3')):
        values = [means[(s, condition)] for s in splits_order]
        bars = ax_bar.bar(x + offset, values, width=0.4, color=color,
                          label=f'{"its own" if condition == "own" else "another"} genotype')
        for bar, v in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, v, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9)
    ax_bar.set_xticks(x, [f'{s}\n({"+" if gain(s) >= 0 else ""}{gain(s):.1%} gain)'
                          for s in splits_order])
    ax_bar.set_ylabel('denoising error (lower is better)')
    ax_bar.set_title('Mean over timesteps', fontsize=11)
    ax_bar.legend(fontsize=9)

    for split, style in (('trained', '-'), ('held out', '--')):
        for condition, color in (('own', '#F2A65A'), ('wrong', '#5FC8D3')):
            ax_line.plot(per_timestep.index, per_timestep[(split, condition)],
                         linestyle=style, color=color, marker='o', markersize=4,
                         label=f'{split}, {"own" if condition == "own" else "another"}')
    ax_line.set_xlabel('timestep')
    ax_line.set_ylabel('denoising error')
    ax_line.set_yscale('log')
    ax_line.set_title('By timestep', fontsize=11)
    ax_line.legend(fontsize=8)

    for ax in (ax_bar, ax_line):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle(f'Does the genotype help on plants the model never saw?  '
                 f'({checkpoint_path.name})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / 'memorization.png', dpi=150)
    plt.close(fig)
    print(f"\nWrote per_image.csv, per_timestep.csv, summary.json and memorization.png to {out}")


if __name__ == '__main__':
    main()
