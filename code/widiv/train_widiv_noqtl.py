# Retrains the WiDiv diffusion model on split_random with the designed QTL left out
# of the genotype, to see how much the model leans on them.
#
# train_widiv.py with the QTL settings changed and nothing else. Every other setting
# matches diffusion_widiv_medium_random_vaev3: the v3 LiteVAE and its latent cache,
# the validation and test lines, and 500 epochs, which also sets the cosine learning
# rate. Compare at epoch 400 as well as 500. Two tests:
#
#   literal  the 40 QTL SNPs removed. A weak test: relatives share the QTL, so the
#            remaining PCs still predict them (R^2 0.37 vs 0.375).
#   strict   +/-1 Mb around each QTL removed (3,399 SNPs), and the scores orthogonalized
#            to the QTL. QTL R^2 on test lines falls to 0.01. It also removes relatedness
#            that overlaps the QTL; a linear genomic predictor on the same scores falls
#            too (stele 0.62 to 0.43, results/widiv/genetics_profile/noqtl_option_check.csv).
#
# Each test has its own folder:
#   models/diffusion_widiv_medium_random_noqtl0kb_vaev3/
#   models/diffusion_widiv_medium_random_noqtl1000kb_orth_vaev3/
#
# From the project root on Hellbender, in a GPU job:
#     python code/widiv/train_widiv_noqtl.py strict
#     python code/widiv/train_widiv_noqtl.py literal
# or both as a job array: sbatch code/widiv/train_widiv_noqtl_hellbender.sbatch

import argparse
import os
import sys
from pathlib import Path

# Puts code/ and code/widiv/ on the import path so this runs directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import apply_overrides

# test -> (kb removed around each QTL, orthogonalize the scores to the QTL)
TESTS = {'literal': (0, False), 'strict': (1000, True)}


def main(overrides=None):
    class cfg:
        # 'literal' or 'strict', above. The command line's first argument overrides it.
        test = 'strict'
        split = 'random'
        model_size = 'medium'
        num_epochs = 500
        # Training belongs on the cluster, not on a laptop GPU. True lets it run
        # outside a Slurm job anyway.
        allow_outside_slurm = False

    apply_overrides(cfg, overrides)
    if cfg.test not in TESTS:
        raise SystemExit(f"test must be one of {list(TESTS)}, not {cfg.test!r}")
    if not cfg.allow_outside_slurm and 'SLURM_JOB_ID' not in os.environ:
        raise SystemExit("this trains a diffusion model and runs on Hellbender: run it inside a "
                         "GPU job (srun or sbatch), or set allow_outside_slurm")
    kb, orthogonalize = TESTS[cfg.test]
    print(f"No-QTL test: {cfg.test} (+/-{kb} kb removed, orthogonalized: {orthogonalize})\n")

    import train_widiv
    train_widiv.main({'split': cfg.split, 'model_size': cfg.model_size, 'num_epochs': cfg.num_epochs,
                      'exclude_qtl_kb': kb, 'orthogonalize_qtl': orthogonalize})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test', nargs='?', choices=list(TESTS))
    args = parser.parse_args()
    main({'test': args.test} if args.test else None)
