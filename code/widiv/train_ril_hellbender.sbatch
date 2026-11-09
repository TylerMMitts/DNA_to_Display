#!/bin/bash
# Trains the diffusion model on the RIL (ril_v5) roots on Hellbender, with the
# settings of the working WiDiv v3 models (train_widiv.py with WIDIV_DATASET=ril_v5:
# medium UNet, sampled latents, no latent scaling, no layout channel, matched
# start noise in previews), one SNP encoding per array task:
#
#   0  PCA       dosage PCA to 95% of the training lines' variance (124 PCs);
#                train_widiv.py DATASET_DEFAULTS says why 95% rather than WiDiv's 80%
#   1  windows   option F: 32 PCs plus gated windows of 200 markers (50 windows),
#                sparsity 1e-6 per open window
#   2  hybrid    the windows tokens plus tokens from an out-of-fold genomic
#                prediction of each line's image (256 summary PCs), window
#                tokens dropped for half the training samples
#                (widiv_hybrid_encoding.py). Not in the default array: run it
#                with --array=2. Its folder is
#                ..._windows200_pc32_sp1e-06_hybrid256_drop0.5_vaev3/.
#
# Both train on split_random only (the RIL has no other split), with the same
# 198 validation lines held out of the 1,979 training lines; the 495 test lines
# are never trained on, never shape the PCA and never pick a checkpoint.
# 800 epochs, the same number of optimiser steps as the WiDiv v3 runs' 500 (one
# image per line instead of about four); the window run's warm-up and ramp are
# 160 + 160 epochs for the same reason.
#
# Needs the RIL LiteVAE first (train_litevae_ril_hellbender.sbatch), plus, copied
# up from the laptop:
#     dataset/ril_v5/genetic/    (genotypes_012.csv.gz, snp_map.csv)
#     dataset/ril_v5/metadata/   (train_test_split.csv)
#     results/ril_v5/scaled_512/ and results/ril_v5/scaled_512_metadata.csv
#
# Same environment as the WiDiv jobs. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/widiv/train_ril_hellbender.sbatch             # both encodings
#     sbatch --array=0 code/widiv/train_ril_hellbender.sbatch   # PCA only
#     sbatch --array=2 code/widiv/train_ril_hellbender.sbatch   # hybrid
#
# Folders: models/diffusion_ril_v5_medium_random_vaev3/ and
# models/diffusion_ril_v5_medium_random_windows200_pc32_sp1e-06_vaev3/. A
# resubmitted task resumes from its newest checkpoint. The first task to start
# encodes every image into the latent cache (results/cache/ril_v5_latents/);
# if both start together, both build it and the second one's copy wins.
#SBATCH --job-name=diffusion_ril_v5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

export WIDIV_DATASET=ril_v5
export WIDIV_SPLIT=random
export DIFFUSION_SIZE=${DIFFUSION_SIZE:-medium}
ENCODINGS=(pca windows hybrid)
export WIDIV_SNP_ENCODING=${ENCODINGS[${SLURM_ARRAY_TASK_ID:-0}]}
if [ "$WIDIV_SNP_ENCODING" != pca ]; then
  export WIDIV_WINDOW_SIZE=200
  export WIDIV_WINDOW_PCS=32
  export WIDIV_WINDOW_SPARSITY=${WIDIV_WINDOW_SPARSITY:-1e-6}
fi

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "dataset: $WIDIV_DATASET   encoding: $WIDIV_SNP_ENCODING   size: $DIFFUSION_SIZE   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L

python - <<'PYCHECK'
import sys
assert sys.version_info >= (3, 9), \
    f"needs Python 3.9 or newer, this is {sys.version.split()[0]}"
import torch
assert int(torch.__version__.split('.')[0]) >= 2, \
    f"needs torch 2.0 or newer, this is {torch.__version__}"
assert torch.cuda.is_available(), "torch sees no GPU on this node"
print(f"python {sys.version.split()[0]}   torch {torch.__version__}   "
      f"{torch.cuda.get_device_name(0)}")
PYCHECK

python -u code/widiv/train_widiv.py
