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

    apply_overrides(cfg, overrides)

    if not cfg.allow_outside_slurm and 'SLURM_JOB_ID' not in os.environ:
        raise SystemExit("this trains a diffusion model and runs on Hellbender: submit "
                         "code/synthetic_stele/train_synthetic_stele_hellbender.sbatch")
    if os.environ.get('DIFFUSION_SIZE', cfg.model_size) != cfg.model_size:
        raise SystemExit("DIFFUSION_SIZE is set to a different size; unset it for this run")

    parquet = resolve_input(Path(cfg.snp_dir) / f'MEMA_gene_matrix_synthetic_stele_k{cfg.block_size}.parquet',
                            'synthetic SNP table (run build_synthetic_snps.py first)')
    print(f"Synthetic SNP table: {parquet}\nRun: {cfg.run_name}\n")

    from latent_diffusion.training import train_onehot
    train_onehot.main({'snp_parquet': parquet, 'run_name': cfg.run_name,
                       'model_size': cfg.model_size})


if __name__ == '__main__':
    main()
