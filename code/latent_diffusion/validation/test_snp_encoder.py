# Runs the raw SNP data through the SNP encoder alone, with no UNet or VAE, to
# test the encoder itself.
#
# The encoder is three stages: founder codes -> one-hot (8 slots per locus) ->
# PCA to 168 numbers -> an MLP that emits 8 tokens of 512. Each check below
# targets one of those stages, and every genotype is labelled by how much of
# the encoder has seen it: the PCA was fitted on all 200, but the MLP only
# trained on genotypes that have images and are not in the validation split.

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
    DIFFUSION_ONEHOT_MODEL, IMAGE_METADATA, KINSHIP_MATRIX, RESULTS_DIR, SNP_PARQUET,
    pick_device, resolve_input, resolve_output,
    apply_overrides,
)

from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.models.snp_encoding import (
    RawCodeOneHotEncoder, load_encoder_and_projector, one_hot_founders,
)
from latent_diffusion.validation.validate_founder_encoding import (
    categorical_similarity, upper_tri,
)

GROUP_COLORS = {'trained': '#1971C2', 'held out': '#E8590C', 'never imaged': '#868E96'}


def cosine_matrix(a):
    u = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return u @ u.T


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method='spearman'))


# Shape and value range at every layer, for one genotype.
def stage_walk(encoder, projector, codes, device):
    rows = []

    def add(name, t):
        t = np.asarray(t, dtype=np.float64)
        rows.append({'stage': name, 'shape': 'x'.join(map(str, t.shape)),
                     'min': t.min(), 'max': t.max(), 'mean': t.mean(), 'std': t.std()})

    add('raw founder codes', codes)
    onehot = one_hot_founders(codes[None, :], projector.founders)[0]
    add('one-hot', onehot)
    scores = (onehot - projector.mean_) @ projector.components_.T
    add('PCA scores', scores)

    x = torch.tensor(scores[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, layer in enumerate(encoder.net):
            x = layer(x)
            if not isinstance(layer, torch.nn.Dropout):
                add(f'net[{i}] {type(layer).__name__}', x.cpu().numpy()[0])
        tokens = x.view(-1, encoder.num_tokens, encoder.embedding_dim) + encoder.token_positions
    add('tokens (+ positions)', tokens.cpu().numpy()[0])
    return pd.DataFrame(rows), onehot


# How much of each genotype survives the 168-number PCA bottleneck.
#
# Scores are projected back to one-hot space and each locus is read as its
# largest slot. The fraction of loci that come back as the right founder is
# the most direct measure of what the conditioning can still tell apart.
def locus_recovery(projector, snp_matrix, chunk=8):
    L, F = snp_matrix.shape[1], len(projector.founders)
    founders = np.asarray(projector.founders)
    out = np.empty(len(snp_matrix))
    for s in range(0, len(snp_matrix), chunk):
        block = snp_matrix[s:s + chunk]
        recon = projector.inverse_transform(projector.transform(block))
        guess = founders[recon.reshape(len(block), L, F).argmax(axis=2)]
        out[s:s + chunk] = (guess == block).mean(axis=1)
    return out


# The same readout on genotypes the PCA was never fitted on.
#
# The 200 real genotypes are exactly the ones the PCA was fitted on, so their
# recovery is in-sample and near-perfect by construction. A new line is a
# recombinant of existing ones, so each test genotype is stitched together from
# contiguous blocks of real parents - the realistic case for a new cross.
def recombinant_recovery(projector, snp_matrix, n_parents, n_blocks, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    blocks = np.array_split(np.arange(snp_matrix.shape[1]), n_blocks)
    made = []
    for _ in range(n_samples):
        # Every parent gets an equal share of blocks, so none is silently dropped
        # and the result is never just a copy of one real genotype.
        parents = rng.choice(len(snp_matrix), size=n_parents, replace=False)
        owner = np.resize(parents, n_blocks)
        rng.shuffle(owner)
        made.append(np.concatenate([snp_matrix[p, b] for p, b in zip(owner, blocks)]))
    return locus_recovery(projector, np.stack(made))


# Embedding change as more loci are changed, done two ways.
#
# 'random' switches scattered loci to a random other founder. 'block' copies a
# contiguous run of loci from another real genotype, which is what an actual
# genetic difference looks like. They behave very differently: the PCA was fit
# on real haplotype structure, so it mostly discards scattered noise but tracks
# block changes. Both are reported against the number of loci that actually
# differ, as a fraction of the median distance between two real genotypes, so
# 1.0 means the edit moved the embedding as far as swapping in another genotype.
def locus_sensitivity(encoder_raw, snp_matrix, base_idx, ks, n_repeats, device, seed=0):
    rng = np.random.default_rng(seed)
    founders = np.asarray(encoder_raw.founders)
    n_loci = snp_matrix.shape[1]
    with torch.no_grad():
        base = encoder_raw(torch.tensor(snp_matrix[base_idx], device=device)).flatten(1).cpu().numpy()

    def embed(codes):
        with torch.no_grad():
            return encoder_raw(torch.tensor(codes, device=device)).flatten(1).cpu().numpy()[0]

    rows = []
    for gi, g in enumerate(base_idx):
        for k in ks:
            for _ in range(n_repeats):
                donor = rng.choice(np.delete(np.arange(len(snp_matrix)), g))
                start = rng.integers(0, n_loci - k + 1)
                block = snp_matrix[g].copy()
                block[start:start + k] = snp_matrix[donor, start:start + k]
                n_changed = int((block != snp_matrix[g]).sum())

                # The same number of changed loci, scattered instead of contiguous.
                scattered = snp_matrix[g].copy()
                loci = rng.choice(n_loci, size=n_changed, replace=False)
                pos = np.searchsorted(founders, scattered[loci])
                scattered[loci] = founders[(pos + rng.integers(1, len(founders), n_changed))
                                           % len(founders)]

                for mode, codes in (('block', block), ('random', scattered)):
                    rows.append({'genotype_index': int(g), 'mode': mode, 'k': k,
                                 'loci_changed': n_changed,
                                 'distance': float(np.linalg.norm(embed(codes) - base[gi]))})
    return pd.DataFrame(rows)


def main(overrides=None):
    # Edit these values, then run:
    #     python code/latent_diffusion/validation/test_snp_encoder.py
    class cfg:
        checkpoint = DIFFUSION_ONEHOT_MODEL
        snp_parquet = SNP_PARQUET
        kinship = KINSHIP_MATRIX
        image_metadata = IMAGE_METADATA
        output_dir = RESULTS_DIR / 'snp_encoder_test'
        walk_genotype = None       # None -> the first genotype
        recombinants = [(2, 2), (4, 20)]   # (parents, blocks) for new test genotypes
        recombinant_samples = 8
        flip_counts = [10, 100, 1000, 5000, 10000, 21894]
        flip_genotypes = 6         # genotypes used for the sensitivity sweep
        flip_repeats = 3
        device = pick_device()

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    out = resolve_output(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\nOutput: {out}")

    names, _, snp_matrix = load_snp_data_from_parquet(resolve_input(cfg.snp_parquet, 'SNP parquet'))
    snp_matrix = np.asarray(snp_matrix)

    ckpt = torch.load(resolve_input(cfg.checkpoint, 'diffusion_onehot checkpoint'),
                      map_location=device, weights_only=False)
    if ckpt.get('encoding') != 'one_hot_founders':
        raise SystemExit(f"{Path(cfg.checkpoint).name} is not a one-hot checkpoint; "
                         "this test covers the one-hot encoder only.")
    encoder, projector = load_encoder_and_projector(ckpt, device)
    encoder_raw = RawCodeOneHotEncoder(projector, encoder).to(device).eval()
    print(f"Encoder: {ckpt['snp_encoder_config']}  "
          f"({sum(p.numel() for p in encoder.parameters()):,} parameters)")
    print(f"PCA: {projector.n_loci} loci x {len(projector.founders)} founders -> "
          f"{projector.output_dim} components, "
          f"{projector.explained_variance_ratio_.sum():.2%} variance")

    imaged = set(pd.read_csv(resolve_input(cfg.image_metadata, 'image metadata'))['genotype'])
    held_out = set(ckpt['val_genotypes'])
    groups = np.array(['held out' if n in held_out else 'trained' if n in imaged
                       else 'never imaged' for n in names])
    for g in GROUP_COLORS:
        print(f"  {g:13s} {int((groups == g).sum())} genotypes")

    report = {}

    # 1. every stage, one genotype
    walk_idx = 0 if cfg.walk_genotype is None else names.index(cfg.walk_genotype)
    walk, onehot = stage_walk(encoder, projector, snp_matrix[walk_idx], device)
    per_locus = onehot.reshape(-1, len(projector.founders)).sum(axis=1)
    print(f"\n1. Stage walk for {names[walk_idx]}")
    print(walk.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"   one-hot: every locus has exactly one slot set: {bool((per_locus == 1).all())}")
    walk.to_csv(out / 'stage_walk.csv', index=False)

    # 2. all genotypes, two independent paths
    with torch.no_grad():
        tokens = torch.cat([encoder_raw(torch.tensor(snp_matrix[s:s + 16], device=device))
                            for s in range(0, len(snp_matrix), 16)]).cpu().numpy()
        again = encoder_raw(torch.tensor(snp_matrix[:16], device=device)).cpu().numpy()
        scores = projector.transform(snp_matrix)
        via_numpy = encoder(torch.tensor(scores, dtype=torch.float32, device=device)).cpu().numpy()

    path_gap = float(np.abs(tokens - via_numpy).max())
    repeat_gap = float(np.abs(tokens[:16] - again).max())
    print(f"\n2. Consistency over all {len(names)} genotypes -> tokens {tokens.shape}")
    print(f"   finite: {bool(np.isfinite(tokens).all())}")
    print(f"   torch one-hot path vs numpy PCA path, max |diff|: {path_gap:.2e}")
    print(f"   same input twice, max |diff|: {repeat_gap:.2e}  (0 = deterministic)")
    report['consistency'] = {'finite': bool(np.isfinite(tokens).all()),
                             'torch_vs_numpy_max_abs_diff': path_gap,
                             'repeat_max_abs_diff': repeat_gap}

    # 3. PCA bottleneck
    print("\n3. Locus recovery through the PCA bottleneck...")
    recovery = locus_recovery(projector, snp_matrix)
    print(f"   the 200 real genotypes (in-sample for the PCA): mean {recovery.mean():.1%}  "
          f"(min {recovery.min():.1%}; chance 12.5%)")
    report['locus_recovery'] = {g: float(recovery[groups == g].mean()) for g in GROUP_COLORS}
    report['recombinant_recovery'] = {}
    for n_parents, n_blocks in cfg.recombinants:
        rec = recombinant_recovery(projector, snp_matrix, n_parents, n_blocks, cfg.recombinant_samples)
        print(f"   new recombinant, {n_parents} parents in {n_blocks} blocks: mean {rec.mean():.1%}  "
              f"(min {rec.min():.1%})")
        report['recombinant_recovery'][f'{n_parents}_parents_{n_blocks}_blocks'] = float(rec.mean())

    # 4. does similarity survive each stage
    flat = tokens.reshape(len(names), -1)
    specific = flat - flat.mean(axis=0)
    dist = np.linalg.norm(flat[:, None] - flat[None], axis=2)
    off = ~np.eye(len(names), dtype=bool)

    sims = {'raw (fraction of loci shared)': categorical_similarity(snp_matrix, projector.founders),
            'PCA scores (cosine)': cosine_matrix(scores),
            'tokens (cosine)': cosine_matrix(flat),
            'tokens minus shared mean (cosine)': cosine_matrix(specific)}
    kin = pd.read_csv(resolve_input(cfg.kinship, 'kinship matrix'), index_col=0)
    kin.index = kin.index.str.replace('_TC', '')
    kin.columns = kin.columns.str.replace('_TC', '')
    if set(names) <= set(kin.index):
        sims['kinship matrix'] = kin.loc[names, names].values

    raw = upper_tri(sims['raw (fraction of loci shared)'])
    raw_nn = np.where(off, sims['raw (fraction of loci shared)'], -np.inf).argmax(axis=1)
    print("\n4. Genetic similarity preserved, Spearman vs raw loci shared "
          "(and nearest-neighbour agreement)")
    report['similarity'] = {}
    for label, m in sims.items():
        if label.startswith('raw'):
            continue
        rho = spearman(upper_tri(m), raw)
        nn = float((np.where(off, m, -np.inf).argmax(axis=1) == raw_nn).mean())
        print(f"   {label:36s} rho {rho:+.3f}   same nearest neighbour {nn:.0%}")
        report['similarity'][label] = {'spearman_vs_raw': rho, 'nearest_neighbour_agreement': nn}

    frac_specific = float(specific.std() / flat.std())
    print(f"   min distance between two genotypes: {dist[off].min():.3f}  "
          f"(median {np.median(dist[off]):.3f}); {frac_specific:.0%} of token variation "
          "is genotype-specific")
    report['distinctness'] = {'min_pair_distance': float(dist[off].min()),
                              'median_pair_distance': float(np.median(dist[off])),
                              'fraction_genotype_specific': frac_specific}

    # 5. per-group behaviour of the MLP
    norms = np.linalg.norm(tokens, axis=2).mean(axis=1)
    print("\n5. By group (does the MLP treat unseen genotypes differently?)")
    report['groups'] = {}
    for g in GROUP_COLORS:
        sel = groups == g
        if not sel.any():
            continue
        # Each genotype in the group against every other genotype, self excluded.
        pairs = off & sel[:, None]
        rho = spearman(sims['tokens (cosine)'][pairs],
                       sims['raw (fraction of loci shared)'][pairs])
        print(f"   {g:13s} n={int(sel.sum()):3d}  mean token norm {norms[sel].mean():6.2f}  "
              f"locus recovery {recovery[sel].mean():.1%}  rho vs raw {rho:+.3f}")
        report['groups'][g] = {'n': int(sel.sum()), 'mean_token_norm': float(norms[sel].mean()),
                               'locus_recovery': float(recovery[sel].mean()),
                               'spearman_vs_raw': rho}

    # 6. sensitivity to individual loci
    print("\n6. Locus sensitivity (embedding change / median gap between two real genotypes)")
    base_idx = np.linspace(0, len(names) - 1, cfg.flip_genotypes).astype(int)
    sens = locus_sensitivity(encoder_raw, snp_matrix, base_idx, cfg.flip_counts,
                             cfg.flip_repeats, device)
    median_pair = float(np.median(dist[off]))
    sens['relative_to_median_pair'] = sens['distance'] / median_pair
    summary = sens.groupby(['k', 'mode']).agg(
        loci_changed=('loci_changed', 'mean'), mean=('relative_to_median_pair', 'mean'),
        lo=('relative_to_median_pair', 'min'), hi=('relative_to_median_pair', 'max')).reset_index()
    for k, part in summary.groupby('k'):
        r = part.set_index('mode')
        print(f"   ~{int(r.loc['block', 'loci_changed']):6d} loci differ   "
              f"copied block {r.loc['block', 'mean']:.3f}   "
              f"scattered random {r.loc['random', 'mean']:.4f}")
    sens.to_csv(out / 'locus_sensitivity.csv', index=False)
    report['locus_sensitivity'] = {
        mode: {int(r['loci_changed']): float(r['mean']) for _, r in part.iterrows()}
        for mode, part in summary.groupby('mode')}

    np.savez_compressed(out / 'snp_embeddings.npz', names=np.array(names), groups=groups,
                        tokens=tokens, pca_scores=scores)
    pd.DataFrame({'genotype': names, 'group': groups, 'mean_token_norm': norms,
                  'locus_recovery': recovery}).to_csv(out / 'per_genotype.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    cols = [GROUP_COLORS[g] for g in groups]
    iu = np.triu_indices(len(names), k=1)
    pair_col = np.where((groups[iu[0]] == 'trained') & (groups[iu[1]] == 'trained'),
                        GROUP_COLORS['trained'], GROUP_COLORS['held out'])

    for ax, label in ((axes[0, 0], 'tokens (cosine)'),
                      (axes[0, 1], 'kinship matrix' if 'kinship matrix' in sims else 'PCA scores (cosine)')):
        ax.scatter(raw, upper_tri(sims[label]), s=3, alpha=0.35, c=pair_col, linewidths=0)
        ax.set_xlabel('raw: fraction of loci sharing a founder')
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs raw   rho {report['similarity'][label]['spearman_vs_raw']:+.3f}",
                     fontsize=10.5)
    axes[0, 0].scatter([], [], c=GROUP_COLORS['trained'], label='both genotypes trained')
    axes[0, 0].scatter([], [], c=GROUP_COLORS['held out'], label='at least one unseen by the MLP')
    axes[0, 0].legend(fontsize=8, loc='lower right')

    ax = axes[1, 0]
    for mode, color, label in (('block', '#1971C2', 'block copied from another genotype'),
                               ('random', '#E8590C', 'same number scattered at random')):
        s = summary[summary['mode'] == mode]
        ax.errorbar(s['loci_changed'], s['mean'], yerr=[s['mean'] - s['lo'], s['hi'] - s['mean']],
                    marker='o', color=color, capsize=3, label=label)
    ax.axhline(1.0, color='#868E96', linestyle='--', linewidth=0.9)
    ax.text(summary['loci_changed'].min(), 1.03, 'distance between two typical real genotypes',
            fontsize=8, color='#868E96')
    ax.set_xscale('log')
    ax.set_xlabel('loci that differ from the original genotype')
    ax.set_ylabel('embedding change / median genotype gap')
    ax.set_title('Locus sensitivity', fontsize=10.5)
    ax.legend(fontsize=8, loc='center left')

    ax = axes[1, 1]
    for g, c in GROUP_COLORS.items():
        sel = groups == g
        if sel.any():
            ax.scatter(recovery[sel] * 100, norms[sel], s=14, c=c, label=f'{g} ({int(sel.sum())})')
    ax.set_xlabel('loci recovered through PCA (%)')
    ax.set_ylabel('mean token norm')
    ax.set_title('Per genotype', fontsize=10.5)
    ax.legend(fontsize=8)

    for a in axes.ravel():
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.suptitle('SNP encoder, tested on its own', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / 'snp_encoder_test.png', dpi=150)
    plt.close(fig)

    print(f"\nWrote stage_walk.csv, per_genotype.csv, locus_sensitivity.csv, "
          f"snp_embeddings.npz, summary.json and snp_encoder_test.png to {out}")


if __name__ == '__main__':
    main()
