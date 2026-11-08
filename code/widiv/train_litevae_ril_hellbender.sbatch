#!/bin/bash
# Trains the RIL (ril_v5) LiteVAE autoencoder (512 px) on Hellbender, with the
# WiDiv v3 recipe: linear output, background-weighted loss, background prior
# (train_litevae_widiv.py, run with WIDIV_DATASET=ril_v5).
#
# This has to finish before the RIL diffusion models can be trained
# (train_ril_hellbender.sbatch): they learn inside this autoencoder's latent space.
#
# Needs, copied up from the laptop after organize_ril.py and prepare_widiv_images.py:
#     dataset/ril_v5/metadata/   (train_test_split.csv)
#     results/ril_v5/scaled_512/ and results/ril_v5/scaled_512_metadata.csv
#
# Same environment as the WiDiv jobs (see train_litevae_widiv_hellbender.sbatch).
# From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/widiv/train_litevae_ril_hellbender.sbatch
#
# Weights go to models/litevae_ril_v5_v3/; a resubmitted job resumes from the
# newest checkpoint there. 320 epochs, the same number of optimiser steps as the
# WiDiv v3 autoencoder's 250.
#SBATCH --job-name=litevae_ril_v5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/training/slurm/%x_%j.out
##SBATCH --account=your_account

set -euo pipefail

export WIDIV_DATASET=ril_v5

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "dataset: $WIDIV_DATASET   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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

python -u code/widiv/train_litevae_widiv.py
