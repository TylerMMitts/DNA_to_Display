# Generates roots for every WiDiv line from one diffusion checkpoint and measures
# their traits, for sharing and for GWAS. Inference only.
#
# 4 images per line by default, first from the same seeds for every line, so
# differences between lines come from the genotype and not the noise. Sampling is
# 50 DDIM steps from matched starting noise (widiv_latents.matched_start_noise).
# A broken root (see is_valid) is regenerated from that line's next unused seed
# until it passes, so every line ends with that many good images.
#
# Writes to results/widiv/generated_all_lines/<run>_e<epoch>/:
#   images/<line>_seed<k>.png   one 512 px image per line and seed
#   traits.csv                  one row per image: traits, set, role
#   README.txt                  what the columns mean
# and moves every broken root, with its traits, to <run>_e<epoch>_rejected/, beside it.
#
# Resumable: images already listed in traits.csv are kept, so a run stopped by a
# sleep or a crash continues where it left off.
#
#     python code/widiv/generate_all_lines.py <checkpoint.pt>

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import pick_device
from widiv_data import load_split, make_transforms
from widiv_latents import (LatentScaler, ScaledClampScheduler, ScaledDecoder, encoded_latent_mean,
                           generate_widiv_batch)
from widiv_litevae import load_widiv_litevae
from widiv_paths import (WIDIV_GENOTYPE_CACHE, WIDIV_GENOTYPES, WIDIV_PLOT_PHENOTYPES, WIDIV_RESULTS_DIR,
                         WIDIV_SCALED_DIR, WIDIV_SCALED_METADATA, WIDIV_SPLIT)
from widiv_snp_encoding import load_widiv_genotypes
from widiv_window_encoding import load_widiv_encoder
import widiv_traits as WT

from latent_diffusion.models.unet import DenoisingUNet

# A root counts as broken when one of these traits falls outside the range real
# plots span (0.5-99.5 percentile). Metaxylem diameter is left out: the model draws
# vessels about 13 um thinner than real ones on average, so for lines with small
# vessels a whole, well-formed root fell under the real minimum, and rejecting those
# would remove genetic signal rather than broken images. Sclerenchyma is left out
# because many real plots have none, so its range rejects nothing useful.
VALID_TRAITS = ['root_diam_um', 'stele_diam_um', 'mx_n', 'rca_pct']
MAX_ATTEMPTS = 40

README = """Generated WiDiv roots: {run}, epoch {epoch}, LiteVAE {litevae}
{n_lines} lines x {n_images} images. 50 DDIM steps, matched starting noise.

images/<line>_seed<k>.png   512 x 512 px, 5.371 um per pixel.

Every image is a whole, valid root: {valid_traits} all inside the range real plots span
(0.5-99.5 percentile). Each line's images start from seeds 0-{last_seed}, the same seeds
for every line. A root that came out broken was regenerated from that line's next unused
seed until it passed, so seeds {n_images} and up are replacements ({n_replaced} images here).
Broken roots are kept in {rejected} for reference.

Metaxylem diameter is not part of the check: the model draws vessels thinner than real
ones, so whole roots of small-vessel lines fell below the real minimum, and rejecting
them would have removed genuine differences between lines.

Regenerating until valid favours draws that come out well formed. Broken roots were
somewhat more common for some lines than others, so a replaced image can differ slightly
from what that line's first seed would have given.

traits.csv, one row per image:
  image          file name in images/
  genotype       line
  seed           noise seed the image was generated from
  set            training or test: the delivered {split} column of train_test_split.csv
  role           trained: the model trained on this line's images
                 validation: on the training side, held back from fitting and used to pick checkpoints
                 test: never seen by any part of the model
  root_diam_um, stele_diam_um, mx_n, mx_diam_um, rca_pct, mcs_layers
                 measured from the image's tissue colours. Cortical cell size and cortical
                 cell files (ccfn) cannot be measured reliably on generated roots, so they are not included.

For GWAS use test lines to judge prediction: training lines' generated roots can carry
plot-level noise memorised from their real images.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--litevae', type=Path, default=None,
                        help="LiteVAE checkpoint; default: the file the diffusion checkpoint names, beside it")
    parser.add_argument('--seeds', type=int, default=4, help='images per line')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(pick_device())
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    stem = args.checkpoint.stem.replace('diffusion_widiv_medium_', '')
    run, epoch = stem.rsplit('_epoch_', 1)
    out = WIDIV_RESULTS_DIR / 'generated_all_lines' / f'{run}_e{epoch}'
    rejected = out.parent / f'{out.name}_rejected'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (rejected / 'images').mkdir(parents=True, exist_ok=True)
    csv_path, rejected_csv = out / 'traits.csv', rejected / 'traits.csv'

    lines, _, dosage = load_widiv_genotypes(WIDIV_GENOTYPES, WIDIV_GENOTYPE_CACHE)
    row_of = {g: i for i, g in enumerate(lines)}
    split = load_split(WIDIV_SPLIT)[ck['split']]
    val, test = set(ck['val_genotypes']), set(ck['test_genotypes'])
    real = pd.read_csv(WIDIV_PLOT_PHENOTYPES)
    real = real[real.has_image.astype(str).str.upper() == 'TRUE']
    low, high = real[VALID_TRAITS].quantile(0.005), real[VALID_TRAITS].quantile(0.995)

    def is_valid(traits):
        return all(low[t] <= traits[t] <= high[t] for t in VALID_TRAITS)    # NaN fails

    litevae_path = args.litevae or args.checkpoint.parent / Path(ck['litevae']).name
    litevae_encoder, litevae_decoder = load_widiv_litevae(litevae_path, device)
    scaler = LatentScaler.from_checkpoint(ck)
    latent_mean = ck.get('latent_channel_mean')
    if latent_mean is None:
        # Checkpoints from before it was saved: the same number, from real images.
        metadata = pd.read_csv(WIDIV_SCALED_METADATA)
        transform = make_transforms(False)
        images = torch.stack([transform(Image.open(WIDIV_SCALED_DIR / f).convert('RGB'))
                              for f in metadata.scaled_filename.iloc[::100]]).to(device)
        latent_mean = np.mean([encoded_latent_mean(litevae_encoder, images[i:i + 6], scaler)
                               for i in range(0, len(images), 6)], axis=0)
        del images
    del litevae_encoder

    encoder, projector = load_widiv_encoder(ck, device)
    unet_config = dict(ck['unet_config'])
    unet_config['channel_mult'] = tuple(unet_config['channel_mult'])
    unet = DenoisingUNet(**unet_config)
    unet.load_state_dict(ck['unet_state_dict'])
    unet.set_store_attention(False)
    unet = unet.to(device).eval()
    scheduler = ScaledClampScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)
    scheduler.set_scaler(scaler)
    decoder = ScaledDecoder(litevae_decoder, scaler)

    def make(jobs):
        # (genotype, seed) pairs -> rows of measured traits, images saved to images/.
        snp = torch.tensor(projector.transform(dosage[[row_of[g] for g, _ in jobs]]), dtype=torch.float32,
                           device=device)
        with torch.no_grad():
            images = generate_widiv_batch(encoder, unet, scheduler, decoder, snp, [s for _, s in jobs], device,
                                          (unet_config['latent_channels'], 64, 64), args.steps,
                                          latent_mean=latent_mean)
        rows = []
        for (g, s), image in zip(jobs, images):
            name = f'{g}_seed{s}.png'
            Image.fromarray(image).save(out / 'images' / name)
            rows.append({'image': name, 'genotype': g, 'seed': s,
                         'set': 'test' if split[g] == 'test' else 'training',
                         'role': 'test' if g in test else 'validation' if g in val else 'trained',
                         # Cortical cell size is left out: the texture measure finds no
                         # clean cortex patches in about 90% of generated roots.
                         **{k: v for k, v in WT.measure(image).items() if k != 'cort_cell_um'}})
        return rows

    def reject(rows):
        # Broken roots go to the rejected folder with their traits, never deleted.
        for r in rows:
            shutil.move(str(out / 'images' / r['image']), str(rejected / 'images' / r['image']))
        pd.DataFrame(rows).to_csv(rejected_csv, mode='a', header=not rejected_csv.exists(), index=False)

    # 1. The first images: seeds 0 .. seeds-1 for every line, skipping ones already made.
    table = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    done = set(table.image) if len(table) else set()
    jobs = [(g, s) for g in lines for s in range(args.seeds) if f'{g}_seed{s}.png' not in done]
    print(f'{run} epoch {epoch}: {len(lines)} lines x {args.seeds} images, {len(done)} made, {len(jobs)} to go -> {out}',
          flush=True)
    for b in range(0, len(jobs), args.batch):
        rows = make(jobs[b:b + args.batch])
        pd.DataFrame(rows).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        if b % (args.batch * 25) == 0:
            print(f'  {b + len(rows)}/{len(jobs)}', flush=True)

    # 2. Broken roots out, replacements in, until every line has only valid images.
    table = pd.read_csv(csv_path).drop(columns=['valid'], errors='ignore')
    ok = table.apply(is_valid, axis=1)
    broken = table[~ok]
    reject(broken.to_dict('records'))
    table = table[ok]
    used = pd.concat([table[['genotype', 'seed']]] +
                     ([pd.read_csv(rejected_csv)[['genotype', 'seed']]] if rejected_csv.exists() else []))
    next_seed = used.groupby('genotype').seed.max().add(1).clip(lower=args.seeds).to_dict()
    # One job per missing image, so a run stopped between rounds still refills every line.
    short = args.seeds - table.groupby('genotype').size().reindex(lines, fill_value=0)
    pending = [g for g in lines for _ in range(max(int(short[g]), 0))]
    print(f'{len(broken)} broken roots moved to {rejected.name}; {len(pending)} images to regenerate', flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        jobs = []
        for g in pending:
            jobs.append((g, next_seed[g]))
            next_seed[g] += 1
        good, bad = [], []
        for b in range(0, len(jobs), args.batch):
            for r in make(jobs[b:b + args.batch]):
                (good if is_valid(r) else bad).append(r)
        table = pd.concat([table, pd.DataFrame(good)], ignore_index=True)
        reject(bad)
        pending = [r['genotype'] for r in bad]
        print(f'  attempt {attempt}: {len(good)} replaced, {len(pending)} still broken', flush=True)
        # Saved after every round, so an interruption keeps the replacements made so far.
        table.sort_values(['genotype', 'seed']).to_csv(csv_path, index=False)
    if pending:
        print(f'WARNING: {len(pending)} images still broken after {MAX_ATTEMPTS} attempts: {sorted(set(pending))}')

    table = table.sort_values(['genotype', 'seed'])
    table.to_csv(csv_path, index=False)
    (out / 'README.txt').write_text(README.format(
        run=run, epoch=epoch, litevae=Path(ck['litevae']).name, n_lines=len(lines), n_images=args.seeds,
        last_seed=args.seeds - 1, n_replaced=int((table.seed >= args.seeds).sum()), rejected=rejected.name,
        split=ck['split'], valid_traits=', '.join(VALID_TRAITS)), encoding='utf-8')
    counts = table.groupby('genotype').size()
    print(f'{len(table)} images, {counts.min()}-{counts.max()} per line, '
          f'{int((table.seed >= args.seeds).sum())} replacements; rejected {len(pd.read_csv(rejected_csv)) if rejected_csv.exists() else 0}')


if __name__ == '__main__':
    main()
