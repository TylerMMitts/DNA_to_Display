# Which SNP tokens the UNet attends to, and where in the image.
#
# Also the shared loader: load_model() rebuilds a trained model from any
# checkpoint, detecting whether it was trained with numeric or one-hot
# encoding, so most other scripts import it from here.

import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import (
    DIFFUSION_ONEHOT_MODEL, KINSHIP_MATRIX, LITEVAE_MODEL, RESULTS_DIR,
    SNP_PARQUET, resolve_input, resolve_output,
)

from latent_diffusion.models.snp_encoder import SNPEncoder, load_snp_data_from_parquet
from latent_diffusion.models.snp_encoding import (
    OneHotSNPEncoder, RawCodeOneHotEncoder, SNPProjector,
)
from latent_diffusion.models.unet import DenoisingUNet
from latent_diffusion.utils import attention_analysis as aa


# Rebuilding the trained model from a checkpoint.

def infer_snp_encoder_config(state_dict):
    # The checkpoint stores weights only, so shapes are read back out of the
    # state dict rather than assuming the values in train.py's Config still
    # match the run that produced this checkpoint.
    first = state_dict['net.0.weight']          # [hidden_dim, input_dim]
    tokens = state_dict['token_positions']      # [1, num_tokens, embed_dim]

    return {
        'hidden_dim': int(first.shape[0]),
        'input_dim': int(first.shape[1]),
        'num_tokens': int(tokens.shape[1]),
        'embedding_dim': int(tokens.shape[2]),
    }


def infer_unet_config(state_dict):
    init_conv = state_dict['init_conv.weight']  # [base_channels, latent_channels, 3, 3]
    base_channels = int(init_conv.shape[0])
    latent_channels = int(init_conv.shape[1])

    num_down = len({k.split('.')[1] for k in state_dict if k.startswith('down_blocks.')})
    num_res_blocks = len({
        k.split('.')[3] for k in state_dict if k.startswith('down_blocks.0.res_blocks.')
    })

    # A down block carries cross-attention exactly when its index is listed in
    # attention_resolutions, so the list can be read back off the weight names.
    attention_resolutions = [
        i for i in range(num_down)
        if f'down_blocks.{i}.cross_attn.W_Q.weight' in state_dict
    ]

    if attention_resolutions:
        i = attention_resolutions[0]
        d_attention = int(state_dict[f'down_blocks.{i}.cross_attn.W_Q.weight'].shape[0])
        snp_embed_dim = int(state_dict[f'down_blocks.{i}.cross_attn.W_K.weight'].shape[1])
    else:
        wq_key = next(k for k in state_dict if 'bottleneck' in k and 'W_Q' in k)
        d_attention = int(state_dict[wq_key].shape[0])
        snp_embed_dim = int(state_dict[wq_key.replace('W_Q', 'W_K')].shape[1])

    return {
        'latent_channels': latent_channels,
        'base_channels': base_channels,
        'snp_embed_dim': snp_embed_dim,
        'd_attention': d_attention,
        'num_res_blocks': num_res_blocks,
        'attention_resolutions': attention_resolutions,
    }


def fit_pca(snp_matrix, n_components, cache_path=None, random_state=0):
    # The checkpoint does not store the PCA that was fitted at training time, so
    # it has to be refitted here from the same SNP matrix.
    #
    # sklearn selects the randomized SVD solver at this shape and train.py does
    # not pin random_state, so the refitted basis is close to, but not identical
    # with, the one used during training. Results are reproducible across runs of
    # this script (fixed random_state plus the cache below), but token identities
    # are only approximately those the model was trained with. Saving the PCA
    # alongside future checkpoints would remove this caveat entirely.
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            pca = pickle.load(f)
        if pca.n_components_ == n_components and pca.n_features_in_ == snp_matrix.shape[1]:
            print(f"Loaded cached PCA from {cache_path}")
            return pca
        print("Cached PCA does not match this checkpoint, refitting")

    print(f"Fitting PCA with {n_components} components (read from checkpoint shape)")
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(snp_matrix)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Cached PCA to {cache_path}")

    return pca


def load_model(checkpoint_path, snp_matrix, device, pca_cache=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Checkpoint epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', float('nan')):.4f}")

    enc_cfg = infer_snp_encoder_config(ckpt['snp_encoder_state_dict'])
    unet_cfg = infer_unet_config(ckpt['unet_state_dict'])
    print(f"SNP encoder: {enc_cfg}")
    print(f"UNet: {unet_cfg}")

    # Which encoding this checkpoint was trained under decides how raw founder
    # codes must be turned into encoder input. Getting it wrong is silent, not
    # loud: SNPEncoder and OneHotSNPEncoder have identical state-dict keys and
    # shapes, so a one-hot checkpoint loads into the numeric path with
    # strict=True and no error, and then gets fed numeric-PCA coordinates -
    # reintroducing, at analysis time, the exact magnitude-encoding bug the
    # one-hot retrain was done to remove. Both markers are checked because
    # either one alone identifies a train_onehot.py checkpoint.
    is_one_hot = (ckpt.get('encoding') == 'one_hot_founders'
                  or 'snp_projector' in ckpt)

    if is_one_hot:
        if 'snp_projector' not in ckpt:
            raise SystemExit(
                f"{Path(checkpoint_path).name} declares encoding="
                f"{ckpt.get('encoding')!r} but carries no 'snp_projector'. "
                "Refitting the one-hot PCA here would not reproduce the basis "
                "the model was trained against, so there is no safe fallback.")

        projector = SNPProjector.from_state_dict(ckpt['snp_projector'])
        encoder_config = ckpt.get('snp_encoder_config', {
            'input_dim': enc_cfg['input_dim'],
            'embedding_dim': enc_cfg['embedding_dim'],
            'num_tokens': enc_cfg['num_tokens'],
            'hidden_dim': enc_cfg['hidden_dim'],
        })

        inner = OneHotSNPEncoder(**encoder_config)
        inner.load_state_dict(ckpt['snp_encoder_state_dict'])

        # Wrapped so forward() still accepts raw founder codes, keeping the
        # calling convention every existing analysis script relies on.
        snp_encoder = RawCodeOneHotEncoder(projector, inner)
        snp_encoder.to(device).eval()

        print(f"Encoding: one-hot founders {projector.founders} -> PCA "
              f"({projector.output_dim} components, restored from checkpoint)")
    else:
        print("Encoding: numeric SNP codes -> PCA (legacy checkpoint)")
        uses_pca = enc_cfg['input_dim'] != snp_matrix.shape[1]
        pca = fit_pca(snp_matrix, enc_cfg['input_dim'], pca_cache) if uses_pca else None

        snp_encoder = SNPEncoder(
            num_snps=snp_matrix.shape[1],
            embedding_dim=enc_cfg['embedding_dim'],
            num_tokens=enc_cfg['num_tokens'],
            pca_components=enc_cfg['input_dim'] if uses_pca else None,
            hidden_dim=enc_cfg['hidden_dim'],
        )
        if pca is not None:
            snp_encoder.set_pca(pca)
        snp_encoder.load_state_dict(ckpt['snp_encoder_state_dict'])
        snp_encoder.to(device).eval()

    unet = DenoisingUNet(
        latent_channels=unet_cfg['latent_channels'],
        base_channels=unet_cfg['base_channels'],
        snp_embed_dim=unet_cfg['snp_embed_dim'],
        d_attention=unet_cfg['d_attention'],
        num_res_blocks=unet_cfg['num_res_blocks'],
        attention_resolutions=unet_cfg['attention_resolutions'],
    )
    unet.load_state_dict(ckpt['unet_state_dict'])
    unet.to(device).eval()

    return snp_encoder, unet, unet_cfg


# Choosing which genotypes to compare

def load_similarity(kinship_path, sample_names, snp_matrix):
    # Prefers the SNP-based kinship matrix; falls back to correlation between
    # raw SNP vectors when it is unavailable.
    if kinship_path is not None:
        try:
            path = resolve_input(kinship_path, 'kinship matrix')
        except FileNotFoundError as exc:
            print(f"{exc}\nFalling back to SNP correlation.")
        else:
            df = pd.read_csv(path, index_col=0)
            # The kinship file keeps the _TC suffix that load_snp_data_from_parquet strips.
            df.index = df.index.astype(str).str.replace('_TC', '', regex=False)
            df.columns = df.columns.astype(str).str.replace('_TC', '', regex=False)

            shared = [g for g in sample_names if g in df.index and g in df.columns]
            if len(shared) >= 2:
                print(f"Kinship matrix: {len(shared)}/{len(sample_names)} genotypes matched")
                return df.loc[shared, shared].to_numpy(dtype=float), shared, 'kinship'
            print("Kinship matrix shares no genotype IDs with the SNP data, "
                  "falling back to SNP correlation.")

    print("Using correlation between raw SNP vectors as the similarity measure")
    sim = np.corrcoef(np.asarray(snp_matrix, dtype=np.float64))
    return np.nan_to_num(sim), list(sample_names), 'snp_correlation'


def select_groups(sim, names, group_size, n_similar_groups):
    # Similar groups: the genotypes whose nearest neighbours are closest.
    # Control group: the mutually least-related genotypes. The control is what
    # makes the similar groups interpretable - without it there is no baseline
    # for how alike two attention maps look by default.
    sim = np.array(sim, dtype=float)
    np.fill_diagonal(sim, -np.inf)

    k = group_size - 1
    neighbour_scores = np.sort(sim, axis=1)[:, -k:].mean(axis=1)

    groups, used = [], set()
    for anchor in np.argsort(neighbour_scores)[::-1]:
        if len(groups) >= n_similar_groups:
            break
        if anchor in used:
            continue
        neighbours = [i for i in np.argsort(sim[anchor])[::-1] if i not in used][:k]
        if len(neighbours) < k:
            continue
        members = [int(anchor)] + [int(i) for i in neighbours]
        used.update(members)
        groups.append({
            'name': f'similar_{names[anchor]}',
            'kind': 'similar',
            'genotypes': [names[i] for i in members],
        })

    # Greedy pick of mutually dissimilar genotypes for the control.
    remaining = [i for i in range(len(names)) if i not in used]
    pool = remaining if len(remaining) >= group_size else list(range(len(names)))
    control = [int(pool[int(np.argmin(neighbour_scores[pool]))])]
    while len(control) < group_size:
        best = min((i for i in pool if i not in control),
                   key=lambda i: max(sim[i][j] for j in control))
        control.append(int(best))

    groups.append({
        'name': 'control_dissimilar',
        'kind': 'control',
        'genotypes': [names[i] for i in control],
    })
    return groups


# Attention extraction

@torch.no_grad()
def collect_attention(snp_encoder, unet, snp_batch, timestep, latent_shape,
                      device, seed, chunk_size=16):
    # Runs forward passes with shared noise and returns {layer: [B, N, M]}.
    # One noise tensor, reused for every genotype and every chunk. Each genotype
    # therefore sees an identical latent, so differences in attention are
    # attributable to the SNP conditioning alone.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(1, *latent_shape, generator=generator).to(device)

    unet.set_store_attention(False)   # keeps attention_history from growing
    unet.clear_attention_history()

    collected = {}
    for start in range(0, snp_batch.shape[0], chunk_size):
        sub = snp_batch[start:start + chunk_size]
        n = sub.shape[0]

        z_t = noise.repeat(n, 1, 1, 1)
        t = torch.full((n,), timestep, device=device, dtype=torch.long)

        snp_embedding = snp_encoder(sub)
        unet(z_t, t, snp_embedding)

        # attention_weights is written on every forward pass regardless of the
        # store_attention flag, which only gates the unbounded history list.
        for name, block in unet.iter_cross_attn_blocks():
            if block.attention_weights is not None:
                collected.setdefault(name, []).append(block.attention_weights.cpu())

    return {name: torch.cat(parts, dim=0) for name, parts in collected.items()}


@torch.no_grad()
def decode_images(snp_encoder, unet, scheduler, decoder, snp_batch, device,
                  latent_shape, num_steps, seed):
    # Full DDIM sampling with shared starting noise, for heatmap overlays.
    B = snp_batch.shape[0]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    z_t = torch.randn(1, *latent_shape, generator=generator).to(device).repeat(B, 1, 1, 1)

    snp_embedding = snp_encoder(snp_batch)
    timesteps = scheduler.get_timesteps(num_steps, device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
        t_prev = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)

        noise_pred = unet(z_t, t_batch, snp_embedding)
        z_t = scheduler.denoise_step(z_t, noise_pred, t_batch, t_prev_batch)

    images = decoder(z_t, save_steps=False)
    images = torch.clamp((images + 1) / 2, 0, 1)
    return images.permute(0, 2, 3, 1).cpu().numpy()


# Per-group figures and metrics

def analyse_group(group, layer, timestep, grids, deviations, entropies,
                  batch_index, sim, name_to_row, out_root, images=None,
                  overlay_layer=None):
    genotypes = group['genotypes']
    rows_local = [batch_index[g] for g in genotypes]

    g_raw = grids[rows_local]
    g_dev = deviations[rows_local]
    side = g_raw.shape[1]

    group_dir = out_root / 'groups' / group['name']
    group_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{layer}_t{timestep}'
    sublabels = [f'{side}x{side}'] * len(genotypes)

    aa.save_genotype_token_grid(
        g_raw, genotypes, group_dir / f'{tag}_raw.png',
        title=f'{group["name"]} - {layer} @ t={timestep} - raw attention',
        sublabels=sublabels)

    aa.save_genotype_token_grid(
        g_dev, genotypes, group_dir / f'{tag}_deviation.png',
        title=f'{group["name"]} - {layer} @ t={timestep} - genotype-specific deviation',
        deviation=True, sublabels=sublabels)

    aa.save_dominant_token_figure(
        g_raw, genotypes, group_dir / f'{tag}_dominant_token.png',
        title=f'{group["name"]} - {layer} @ t={timestep} - dominant SNP token')

    if images is not None and layer == overlay_layer:
        for i, genotype in enumerate(genotypes):
            aa.save_overlay_figure(
                images[i], np.abs(g_dev[i]).mean(axis=-1),
                group_dir / f'{tag}_overlay_{genotype}.png',
                title=f'{genotype} - {layer} @ t={timestep}')

    stat_rows = [{
        'group': group['name'], 'group_kind': group['kind'], 'genotype': genotype,
        'layer': layer, 'timestep': timestep, 'grid': f'{side}x{side}',
        'token_entropy': float(entropies[batch_index[genotype]]),
        'max_attention': float(g_raw[i].max()),
        'deviation_l2': float(np.linalg.norm(g_dev[i])),
    } for i, genotype in enumerate(genotypes)]

    pair_rows = []
    for i, j in combinations(range(len(genotypes)), 2):
        raw = aa.map_similarity(g_raw[i], g_raw[j])
        dev = aa.map_similarity(g_dev[i], g_dev[j])
        pair_rows.append({
            'group': group['name'], 'group_kind': group['kind'],
            'genotype_a': genotypes[i], 'genotype_b': genotypes[j],
            'layer': layer, 'timestep': timestep,
            'genetic_similarity': float(sim[name_to_row[genotypes[i]],
                                            name_to_row[genotypes[j]]]),
            'attention_cosine_raw': raw['cosine'],
            'attention_cosine_deviation': dev['cosine'],
            'attention_pearson_deviation': dev['pearson'],
        })

    return stat_rows, pair_rows


def pairwise_similarity_only(groups, grids_by_key, batch_index, sim, name_to_row):
    # Pair metrics without writing figures, used for the untrained null.
    rows = []
    for (layer, timestep), (grids, deviations) in grids_by_key.items():
        for group in groups:
            genotypes = group['genotypes']
            local = [batch_index[g] for g in genotypes]
            for i, j in combinations(range(len(genotypes)), 2):
                rows.append({
                    'group_kind': group['kind'], 'layer': layer, 'timestep': timestep,
                    'genetic_similarity': float(sim[name_to_row[genotypes[i]],
                                                    name_to_row[genotypes[j]]]),
                    'attention_cosine_deviation': aa.map_similarity(
                        deviations[local[i]], deviations[local[j]])['cosine'],
                })
    return pd.DataFrame(rows)


def run_untrained_null(unet_cfg, snp_encoder, snp_batch, groups, batch_index,
                       sim, name_to_row, cfg, latent_shape, device,
                       background_slice):
    # Same measurement on freshly initialised weights.
    #
    # Any smooth network maps similar inputs to similar outputs, so genetically
    # similar genotypes produce correlated attention maps even with random
    # weights. The trained model is only informative to the extent it beats this
    # null - a positive correlation on its own proves nothing.
    import copy

    null_unet = DenoisingUNet(
        latent_channels=unet_cfg['latent_channels'],
        base_channels=unet_cfg['base_channels'],
        snp_embed_dim=unet_cfg['snp_embed_dim'],
        d_attention=unet_cfg['d_attention'],
        num_res_blocks=unet_cfg['num_res_blocks'],
        attention_resolutions=unet_cfg['attention_resolutions'],
    ).to(device).eval()

    # Re-initialise the encoder too, but keep the fitted PCA so the input
    # representation is identical and only the learned weights differ.
    null_encoder = copy.deepcopy(snp_encoder)
    for module in null_encoder.net:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    with torch.no_grad():
        null_encoder.token_positions.normal_(0, 0.02)
    null_encoder.to(device).eval()

    grids_by_key = {}
    for timestep in cfg.timesteps:
        attention = collect_attention(null_encoder, null_unet, snp_batch, timestep,
                                      latent_shape, device, cfg.seed, cfg.chunk_size)
        for layer, attn in sorted(attention.items()):
            grids = aa.attention_to_grids(attn)
            baseline = grids[background_slice].mean(axis=0, keepdims=True)
            grids_by_key[(layer, timestep)] = (grids, grids - baseline)

    return pairwise_similarity_only(groups, grids_by_key, batch_index, sim, name_to_row)


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    # Edit these values, then run:
    #     python code/latent_diffusion/analysis/analyze_snp_attention.py
    #
    # Relative input paths are searched against the working directory, then
    # code/, then the project root, so they work from either location.
    class cfg:
        # inputs
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET
        kinship = KINSHIP_MATRIX   # None -> use SNP correlation

        # outputs (relative to the working directory)
        output_dir = RESULTS_DIR / 'attention_analysis_onehot'
        pca_cache = RESULTS_DIR / 'attention_analysis' / 'pca.pkl'

        # which genotypes to compare
        # Set genotypes to a list, e.g. ['MEMA004', 'MEMA006', 'MEMA017'], to
        # compare exactly those. Leave it None to pick groups automatically
        # from the kinship matrix.
        genotypes = None
        group_size = 4          # genotypes per group
        n_groups = 2            # similar groups; a dissimilar control is always added

        # Genotypes used to estimate the population baseline that deviation
        # maps are measured against. Setting this to 0 falls back to a
        # within-group baseline, where deviations sum to zero and force the
        # mean pairwise cosine to -1/(group_size - 1) - an artifact of group
        # size rather than anything about the model. Keep it well above zero.
        background_size = 32

        # measurement
        timesteps = [800, 500, 200]   # high t shows layout, low t shows detail
        seed = 0
        latent_size = 32
        chunk_size = 16               # lower this if you run out of GPU memory
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Repeat the whole measurement on randomly initialised weights. Slower,
        # but without it a positive correlation is not interpretable.
        untrained_null = True

        # optional image decoding
        decode = False          # sample images so heatmaps can be overlaid
        litevae_checkpoint = LITEVAE_MODEL
        sampling_steps = 50
        overlay_layer = 'up_2'  # layer whose deviation map is overlaid

    device = torch.device(cfg.device)
    out_root = resolve_output(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out_root}")

    # data
    parquet_path = resolve_input(cfg.snp_parquet, 'SNP parquet')
    sample_names, snp_names, snp_matrix = load_snp_data_from_parquet(parquet_path)

    kinship_arg = None if str(cfg.kinship).lower() == 'none' else cfg.kinship
    sim, names, sim_source = load_similarity(kinship_arg, sample_names, snp_matrix)

    # load_similarity may drop genotypes missing from the kinship matrix, so the
    # SNP rows are re-indexed to stay aligned with the similarity matrix.
    original_row = {n: i for i, n in enumerate(sample_names)}
    snp_matrix = np.asarray(snp_matrix)[[original_row[n] for n in names]]
    name_to_row = {n: i for i, n in enumerate(names)}

    # model
    checkpoint_path = resolve_input(cfg.checkpoint, 'checkpoint')
    snp_encoder, unet, unet_cfg = load_model(
        checkpoint_path, snp_matrix, device, pca_cache=str(resolve_output(cfg.pca_cache)))
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)

    # groups
    if cfg.genotypes:
        missing = [g for g in cfg.genotypes if g not in name_to_row]
        if missing:
            raise SystemExit(f"genotypes not found in the SNP data: {missing}")
        groups = [{'name': 'user_selected', 'kind': 'user',
                   'genotypes': list(cfg.genotypes)}]
    else:
        groups = select_groups(sim, names, cfg.group_size, cfg.n_groups)

    print(f"\nSimilarity source: {sim_source}")
    for g in groups:
        pairs = [sim[name_to_row[a], name_to_row[b]]
                 for a, b in combinations(g['genotypes'], 2)]
        print(f"  {g['name']:<28} {g['genotypes']}  mean similarity {np.mean(pairs):.4f}")

    # batch: compared genotypes first, then the background sample
    compared = list(dict.fromkeys(g_ for g in groups for g_ in g['genotypes']))
    rng = np.random.default_rng(cfg.seed)
    pool = [n for n in names if n not in compared]
    n_background = min(cfg.background_size, len(pool))
    background = list(rng.choice(pool, size=n_background, replace=False)) if n_background else []

    batch_names = compared + background
    batch_index = {n: i for i, n in enumerate(batch_names)}
    snp_batch = torch.tensor(snp_matrix[[name_to_row[n] for n in batch_names]],
                             dtype=torch.float32, device=device)
    print(f"\nForward batch: {len(compared)} compared + {len(background)} background genotypes")

    # optional image decoding
    images_by_group = {}
    if cfg.decode:
        from latent_diffusion.diffusion.scheduler import DiffusionScheduler
        from litevae.models import LiteVAEDecoder

        scheduler = DiffusionScheduler()
        # The scheduler keeps its buffers on the CPU but is indexed with a
        # device-side timestep tensor, so move them alongside the model.
        scheduler.betas = scheduler.betas.to(device)
        scheduler.alphas = scheduler.alphas.to(device)
        scheduler.alpha_bars = scheduler.alpha_bars.to(device)

        vae_path = resolve_input(cfg.litevae_checkpoint, 'LiteVAE checkpoint')
        vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
        decoder = LiteVAEDecoder(latent_channels=unet_cfg['latent_channels'],
                                 output_channels=3, base_channels=512, num_res_blocks=2)
        decoder.load_state_dict(vae_ckpt['decoder_state_dict'])
        decoder.to(device).eval()

        for g in groups:
            print(f"Sampling images for {g['name']} ({cfg.sampling_steps} steps)")
            batch = torch.tensor(snp_matrix[[name_to_row[x] for x in g['genotypes']]],
                                 dtype=torch.float32, device=device)
            images_by_group[g['name']] = decode_images(
                snp_encoder, unet, scheduler, decoder, batch, device,
                latent_shape, cfg.sampling_steps, cfg.seed)

    # run
    background_slice = slice(len(compared), len(batch_names)) if background else slice(None)
    all_stats, all_pairs = [], []

    for timestep in cfg.timesteps:
        print(f"\nTimestep {timestep}")
        attention = collect_attention(snp_encoder, unet, snp_batch, timestep,
                                      latent_shape, device, cfg.seed, cfg.chunk_size)

        for layer, attn in sorted(attention.items()):
            grids = aa.attention_to_grids(attn)              # [B, H, W, M]
            baseline = grids[background_slice].mean(axis=0, keepdims=True)
            deviations = grids - baseline
            entropies = aa.token_entropy(attn)
            print(f"  {layer:<14} {grids.shape[1]}x{grids.shape[1]} grid, "
                  f"{grids.shape[-1]} tokens, mean entropy {entropies.mean():.3f}")

            for group in groups:
                stats, pairs = analyse_group(
                    group, layer, timestep, grids, deviations, entropies,
                    batch_index, sim, name_to_row, out_root,
                    images=images_by_group.get(group['name']),
                    overlay_layer=cfg.overlay_layer)
                all_stats.extend(stats)
                all_pairs.extend(pairs)

    # summary
    analysis_dir = out_root / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    stats_df = pd.DataFrame(all_stats)
    pairs_df = pd.DataFrame(all_pairs)
    stats_df.to_csv(analysis_dir / 'attention_stats.csv', index=False)
    pairs_df.to_csv(analysis_dir / 'pairwise_similarity.csv', index=False)

    summary = {
        'checkpoint': str(checkpoint_path),
        'similarity_source': sim_source,
        'timesteps': cfg.timesteps,
        'seed': cfg.seed,
        'background_genotypes': len(background),
        'groups': {g['name']: g['genotypes'] for g in groups},
        'spearman_by_layer': {},
    }

    for layer, sub in pairs_df.groupby('layer'):
        rho = spearman(sub['genetic_similarity'], sub['attention_cosine_deviation'])
        summary['spearman_by_layer'][layer] = rho
        aa.save_similarity_scatter(
            sub['genetic_similarity'], sub['attention_cosine_deviation'],
            analysis_dir / f'similarity_{layer}.png',
            title=f'{layer}: genetic vs attention similarity (rho={rho:.3f})',
            labels=[f'{a}/{b}' for a, b in zip(sub['genotype_a'], sub['genotype_b'])],
            xlabel=f'{sim_source} similarity')

    summary['spearman_overall'] = spearman(pairs_df['genetic_similarity'],
                                           pairs_df['attention_cosine_deviation'])
    group_means = pairs_df.groupby('group_kind')['attention_cosine_deviation'].mean()
    summary['mean_attention_similarity_by_group_kind'] = group_means.to_dict()

    null_df = None
    if cfg.untrained_null:
        print("\nRunning untrained null")
        null_df = run_untrained_null(unet_cfg, snp_encoder, snp_batch, groups,
                                     batch_index, sim, name_to_row, cfg,
                                     latent_shape, device, background_slice)
        null_df.to_csv(analysis_dir / 'pairwise_similarity_untrained.csv', index=False)
        summary['untrained_null'] = {
            'spearman_overall': spearman(null_df['genetic_similarity'],
                                         null_df['attention_cosine_deviation']),
            'spearman_by_layer': {
                layer: spearman(sub['genetic_similarity'],
                                sub['attention_cosine_deviation'])
                for layer, sub in null_df.groupby('layer')},
        }

    with open(analysis_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    print(f"Spearman(genetic similarity, attention similarity): "
          f"{summary['spearman_overall']:.3f}")
    if null_df is not None:
        null_rho = summary['untrained_null']['spearman_overall']
        print(f"  untrained null:                                   {null_rho:.3f}")
        print(f"  the trained model is only informative to the extent it beats this")
    print("\nMean attention similarity (deviation maps) by group kind:")
    print(group_means.to_string())
    print("\nPer layer:")
    for layer, rho in summary['spearman_by_layer'].items():
        print(f"  {layer:<14} rho = {rho:.3f}")
    print(f"\nFigures and CSVs written to {out_root}")


if __name__ == '__main__':
    main()
