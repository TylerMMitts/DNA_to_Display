# Trains the root diffusion model on the synthetic copy of the SNP table.
#
# train_onehot.py with three things changed and nothing else: the SNP table is
# the synthetic copy, the run has a name of its own so its checkpoints never mix
# with the real model's, and it refuses to start outside a Slurm job. Every
# other setting - size, epochs, split, seed - is the real model's, so the only
# difference between the two models is the added genes.
#
# Runs on Hellbender through train_synthetic_stele_hellbender.sbatch. Weights
# go to models/diffusion_onehot_synthetic_stele/, previews and loss curves to
# results/training/diffusion_onehot_synthetic_stele/.

import os
import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import SYNTHETIC_STELE_DIR, SYNTHETIC_STELE_RUN, apply_overrides, resolve_input


def main(overrides=None):
    # Edit these values, then run on Hellbender:
    #     sbatch code/synthetic_stele/train_synthetic_stele_hellbender.sbatch
    class cfg:
        # The copy build_synthetic_snps.py wrote for this block size.
        block_size = 4000
        snp_dir = SYNTHETIC_STELE_DIR
        # The real model this is compared against was trained at medium.
        model_size = 'medium'
        run_name = SYNTHETIC_STELE_RUN
        # Training belongs on the cluster, not on a laptop GPU. True lets it run
        # outside a Slurm job anyway.
        allow_outside_slurm = False

        # 'pca' as the real model, or 'windows' for the founder-window encoder
        # (train_onehot.py). A window run gets its own folder, named with its
        # settings, so it never mixes with the PCA planted-gene run. The
        # DIFFUSION_* environment variables override these, as in train_onehot.py.
        encoding = os.environ.get('DIFFUSION_ENCODING', 'pca')
        window_size = int(os.environ.get('DIFFUSION_WINDOW_SIZE', 200))
        window_sparsity = float(os.environ.get('DIFFUSION_WINDOW_SPARSITY', 1e-4))
        window_warmup_epochs = int(os.environ.get('DIFFUSION_WINDOW_WARMUP', 100))
        window_ramp_epochs = int(os.environ.get('DIFFUSION_WINDOW_RAMP', 100))

    apply_overrides(cfg, overrides)

    if not cfg.allow_outside_slurm and 'SLURM_JOB_ID' not in os.environ:
        raise SystemExit("this trains a diffusion model and runs on Hellbender: submit "
                         "code/synthetic_stele/train_synthetic_stele_hellbender.sbatch")
    if os.environ.get('DIFFUSION_SIZE', cfg.model_size) != cfg.model_size:
        raise SystemExit("DIFFUSION_SIZE is set to a different size; unset it for this run")

    parquet = resolve_input(Path(cfg.snp_dir) / f'MEMA_gene_matrix_synthetic_stele_k{cfg.block_size}.parquet',
                            'synthetic SNP table (run build_synthetic_snps.py first)')

    from latent_diffusion.training import train_onehot
    window = {'encoding': cfg.encoding, 'window_size': cfg.window_size,
              'window_sparsity': cfg.window_sparsity,
              'window_warmup_epochs': cfg.window_warmup_epochs,
              'window_ramp_epochs': cfg.window_ramp_epochs}
    run_name = cfg.run_name
    if cfg.encoding == 'windows':
        # The real model's naming, with this run's name in place of its own.
        full = train_onehot.default_run_name(cfg.model_size, encoding='windows', window_size=cfg.window_size,
                                             window_sparsity=cfg.window_sparsity,
                                             warmup=cfg.window_warmup_epochs, ramp=cfg.window_ramp_epochs)
        run_name = cfg.run_name + full[len(f'{train_onehot.MODEL_NAME}_{cfg.model_size}'):]
    print(f"Synthetic SNP table: {parquet}\nRun: {run_name}\n")

    train_onehot.main({'snp_parquet': parquet, 'run_name': run_name,
                       'model_size': cfg.model_size, **window})


if __name__ == '__main__':
    main()
