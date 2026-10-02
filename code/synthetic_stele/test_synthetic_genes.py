# Tests whether a model trained on the synthetic table learned the stele genes.
#
# Run after train_synthetic_stele.py, on Hellbender or anywhere with the
# checkpoint (it only generates, it does not train). Four checks:
#
#   1. Dose response. Only the added block changes: every gene in it is set to
#      founder 1, then 2, ... 8, with the rest of the genome and the starting
#      noise held fixed, and the stele of each generated root is measured. A
#      model that learned the rule draws a wider stele as the code rises; one
#      that did not draws the same root eight times.
#   2. Held-out lines. Each line the model never trained on is generated with
#      its own genome, block included, and its generated stele is compared with
#      its real one. This is the rule applied to new lines.
#   3. Contribution maps. snp_output_contribution.py flips the whole block to
#      founder 1 and to founder 8 and maps where the image changes, with its
#      tissue breakdown: a learned stele gene should change the stele most.
#   4. Control. The same founder 1 -> 8 flip applied to runs of real genes of
#      the same length, so the synthetic effect is read against what an equal
#      amount of real genome does.
#
# Writes dose_response.csv/.png, held_out.csv/.png, control.csv, summary.json
# and the contribution maps to results/synthetic_stele/test_<checkpoint name>/.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    SEGMENTATION_MODEL, SYNTHETIC_STELE_DIR, SYNTHETIC_STELE_MODEL_DIR,
    SYNTHETIC_STELE_RESULTS_DIR, SYNTHETIC_STELE_RUN, apply_overrides,
    best_checkpoint_path, pick_device, resolve_input, resolve_output,
)

from feature_segmentation.evaluation.reconstruction_fidelity_test import segment
from latent_diffusion.analysis.analyze_snp_attention import load_model
from latent_diffusion.diffusion.scheduler import DiffusionScheduler
from latent_diffusion.generation.generate_from_dataset import generate_batch
from latent_diffusion.models.snp_encoder import load_snp_data_from_parquet
from latent_diffusion.utils.dataset_inputs import load_decoder
from synthetic_stele.synthetic_genes import NAME_PREFIX


def main(overrides=None):
    # Edit these values, then run:
    #     python code/synthetic_stele/test_synthetic_genes.py
    class cfg:
        # None -> the best checkpoint of the synthetic run.
        checkpoint = None
        block_size = 4000
        snp_dir = SYNTHETIC_STELE_DIR
        seg_weights = SEGMENTATION_MODEL
        output_dir = None          # None -> results/synthetic_stele/test_<checkpoint>/

        # Dose response: this many trained and held-out lines, each generated at
        # every code from these seeds. Kept small enough to run in minutes.
        n_trained = 6
        n_held_out = 6
        seeds = [0, 1, 2]
        # Held-out check: every held-out line with a measured stele, these seeds.
        held_out_seeds = [0, 1, 2]
        # Runs of real genes flipped as the control.
        n_control_blocks = 3
        run_contribution_maps = True
        contribution_genotypes = 8

        sampling_steps = 50
        batch_size = 8
        latent_size = 32
        imgsz = 256
        seed = 0
        device = pick_device()

    apply_overrides(cfg, overrides)

    device = torch.device(cfg.device)
    k = int(cfg.block_size)
    checkpoint = Path(resolve_input(cfg.checkpoint or best_checkpoint_path(
        SYNTHETIC_STELE_MODEL_DIR, SYNTHETIC_STELE_RUN), 'synthetic checkpoint'))
    parquet = resolve_input(Path(cfg.snp_dir) / f'MEMA_gene_matrix_synthetic_stele_k{k}.parquet',
                            'synthetic SNP table')
    codes_csv = resolve_input(Path(cfg.snp_dir) / f'synthetic_codes_k{k}.csv', 'synthetic codes')
    out = resolve_output(cfg.output_dir or SYNTHETIC_STELE_RESULTS_DIR / f'test_{checkpoint.stem}')
    out.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {checkpoint}\nTable: {parquet}\nOutput: {out}")

    names, snp_names, matrix = load_snp_data_from_parquet(parquet)
    names, snp_names, matrix = list(names), list(snp_names), np.asarray(matrix)
    if not all(s.startswith(NAME_PREFIX) for s in snp_names[:k]):
        raise SystemExit("the first block_size genes of the table are not the synthetic block")
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    n_loci = int(ckpt['snp_projector']['n_loci'])
    if n_loci != matrix.shape[1]:
        raise SystemExit(f"the checkpoint was trained on {n_loci} genes and this table has "
                         f"{matrix.shape[1]}: it is not the table this model was trained on")
    held = set(ckpt['val_genotypes'])
    founders = list(ckpt['founders'])
    del ckpt
    row = {g: i for i, g in enumerate(names)}

    codes = pd.read_csv(codes_csv)
    codes = codes[codes['measured'] & codes['genotype'].isin(row)]
    trained = codes[~codes['genotype'].isin(held)]
    unseen = codes[codes['genotype'].isin(held)]

    encoder, unet, unet_cfg = load_model(checkpoint, matrix, device)
    unet.set_store_attention(False)
    decoder = load_decoder('roots', device, unet_cfg['latent_channels'])
    scheduler = DiffusionScheduler()
    for attr in ('betas', 'alphas', 'alpha_bars'):
        setattr(scheduler, attr, getattr(scheduler, attr).to(device))
    latent_shape = (unet_cfg['latent_channels'], cfg.latent_size, cfg.latent_size)
    from ultralytics import YOLO
    seg = YOLO(str(resolve_input(cfg.seg_weights, 'segmentation weights')))

    # Generates one root per (genome row, seed) and returns its stele diameter.
    def stele_of(rows, seeds):
        out_vals = []
        jobs = [(r, s) for r in rows for s in seeds]
        for b in range(0, len(jobs), cfg.batch_size):
            chunk = jobs[b:b + cfg.batch_size]
            snp = torch.tensor(np.stack([r for r, _ in chunk]), dtype=torch.float32, device=device)
            with torch.no_grad():
                images = generate_batch(encoder, unet, scheduler, decoder, snp,
                                        [s for _, s in chunk], device, latent_shape,
                                        cfg.sampling_steps)
            for image in images:
                traits, _ = segment(seg, image, 0.25, cfg.device, 4, 2)
                out_vals.append(traits['stele_diameter_px'])
        return np.array(out_vals).reshape(len(rows), len(seeds))

    def evenly(df, n):
        if n <= 0 or df.empty:
            return df.iloc[:0]
        idx = np.linspace(0, len(df) - 1, min(n, len(df))).round().astype(int)
        return df.sort_values('stele_diameter_mean').iloc[sorted(set(idx))]

    # 1. Dose response.
    print("\n1. Dose response: the block set to each founder, everything else fixed")
    chosen = pd.concat([evenly(trained, cfg.n_trained).assign(split='trained'),
                        evenly(unseen, cfg.n_held_out).assign(split='held out')])
    dose = []
    for r in chosen.itertuples():
        base = matrix[row[r.genotype]]
        variants = []
        for c in founders:
            v = base.copy(); v[:k] = c
            variants.append(v)
        stele = stele_of(variants, cfg.seeds)
        for c, values in zip(founders, stele):
            for s, val in zip(cfg.seeds, values):
                dose.append({'genotype': r.genotype, 'split': r.split, 'own_code': r.code,
                             'code': c, 'seed': s, 'stele_diameter_px': val})
        print(f"  {r.genotype} ({r.split}, own code {r.code}): "
              + ' '.join(f'{np.nanmean(v):.0f}' for v in stele))
    dose = pd.DataFrame(dose)
    dose.to_csv(out / 'dose_response.csv', index=False)
    d_ok = dose.dropna(subset=['stele_diameter_px'])
    rho, p_rho = spearmanr(d_ok['code'], d_ok['stele_diameter_px'])
    slope = float(np.polyfit(d_ok['code'], d_ok['stele_diameter_px'], 1)[0])

    # 2. Held-out lines with their own genome.
    print("\n2. Held-out lines generated from their own genome")
    ho = stele_of([matrix[row[g]] for g in unseen['genotype']], cfg.held_out_seeds)
    held_df = unseen[['genotype', 'code', 'stele_diameter_mean']].copy()
    held_df['generated_stele_mean'] = np.nanmean(ho, axis=1)
    held_df.to_csv(out / 'held_out.csv', index=False)
    ok = held_df.dropna()
    r_ho, p_ho = pearsonr(ok['stele_diameter_mean'], ok['generated_stele_mean'])
    rho_code, p_code = spearmanr(ok['code'], ok['generated_stele_mean'])
    print(f"  {len(ok)} lines: real vs generated stele r = {r_ho:.2f} (p = {p_ho:.3f}); "
          f"code vs generated stele rho = {rho_code:.2f} (p = {p_code:.3f})")

    # 4. Control: runs of real genes of the same length, flipped 1 -> 8.
    print("\n4. Control: runs of real genes the same length, flipped founder 1 -> 8")
    rng = np.random.default_rng(cfg.seed)
    starts = sorted(rng.choice(np.arange(k, matrix.shape[1] - k), cfg.n_control_blocks, replace=False))
    control = []

    def flip_effect(lo_hi_slice):
        effects = []
        for r in chosen.itertuples():
            lo, hi = matrix[row[r.genotype]].copy(), matrix[row[r.genotype]].copy()
            lo[lo_hi_slice], hi[lo_hi_slice] = founders[0], founders[-1]
            s = stele_of([lo, hi], cfg.seeds)
            effects.append(np.nanmean(s[1] - s[0]))
        return float(np.nanmean(effects))

    synthetic_effect = flip_effect(slice(0, k))
    control.append({'block': 'synthetic', 'start': 0, 'stele_change_1_to_8_px': synthetic_effect})
    for s in starts:
        control.append({'block': 'real genes', 'start': int(s),
                        'stele_change_1_to_8_px': flip_effect(slice(int(s), int(s) + k))})
        print(f"  real genes {s}-{s + k - 1}: {control[-1]['stele_change_1_to_8_px']:+.1f} px")
    print(f"  synthetic block:      {synthetic_effect:+.1f} px")
    pd.DataFrame(control).to_csv(out / 'control.csv', index=False)

    # Figures.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for split, color in (('trained', '#1f77b4'), ('held out', '#ff7f0e')):
        m = d_ok[d_ok['split'] == split].groupby('code')['stele_diameter_px']
        ax.errorbar(m.mean().index, m.mean(), yerr=m.sem(), fmt='o-', color=color,
                    capsize=3, label=split)
    real = codes.groupby('code')['stele_diameter_mean'].mean()
    ax.plot(real.index, real.values, 's--', color='0.4', label='real lines with that code')
    ax.set_xlabel('founder code of the added block'); ax.set_ylabel('stele diameter (px)')
    ax.set_title(f'Dose response: Spearman {rho:.2f}, {slope:+.1f} px per code step')
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.scatter(ok['stele_diameter_mean'], ok['generated_stele_mean'], c=ok['code'], cmap='viridis')
    lims = [min(ok.min(numeric_only=True)[['stele_diameter_mean', 'generated_stele_mean']]) - 3,
            max(ok.max(numeric_only=True)[['stele_diameter_mean', 'generated_stele_mean']]) + 3]
    ax.plot(lims, lims, '--', color='crimson', lw=1)
    ax.set_xlabel('real mean stele (px)'); ax.set_ylabel('generated mean stele (px)')
    ax.set_title(f'Held-out lines, own genome: r = {r_ho:.2f} (n = {len(ok)})')
    fig.tight_layout()
    fig.savefig(out / 'dose_response.png', dpi=140)
    plt.close(fig)

    # 3. Contribution maps, flipping the whole block each way.
    if cfg.run_contribution_maps:
        from latent_diffusion.analysis import snp_output_contribution
        for target in (founders[0], founders[-1]):
            print(f"\n3. Contribution map: block flipped to founder {target}")
            snp_output_contribution.main({
                'checkpoint': checkpoint, 'snp_parquet': parquet,
                'output_dir': out / f'contribution_to_founder_{target}',
                'snp_names': [snp_names[0]], 'block_size': k, 'founder': target,
                'n_genotypes': cfg.contribution_genotypes, 'top_n': 1,
                'device': cfg.device,
            })

    summary = {
        'checkpoint': str(checkpoint), 'block_size': k,
        'dose_response': {'spearman': float(rho), 'p': float(p_rho),
                          'px_per_code_step': slope,
                          'mean_stele_by_code': d_ok.groupby('code')['stele_diameter_px'].mean().round(2).to_dict()},
        'held_out': {'n': int(len(ok)), 'real_vs_generated_r': float(r_ho), 'p': float(p_ho),
                     'code_vs_generated_spearman': float(rho_code), 'code_p': float(p_code)},
        'control': {'synthetic_1_to_8_px': synthetic_effect,
                    'real_blocks_1_to_8_px': [c['stele_change_1_to_8_px'] for c in control[1:]]},
        'real_lines_mean_stele_by_code': real.round(2).to_dict(),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nDose response Spearman {rho:.2f} ({slope:+.1f} px/step); held-out r {r_ho:.2f}; "
          f"synthetic 1->8 {synthetic_effect:+.1f} px vs real blocks "
          f"{[round(c['stele_change_1_to_8_px'], 1) for c in control[1:]]}")
    print(f"Wrote results to {out}")


if __name__ == '__main__':
    main()
