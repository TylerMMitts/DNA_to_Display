#!/bin/bash
# Trains the IMPC X-ray (impc_bcm) LiteVAE autoencoder (512 px) on Hellbender, with
# the WiDiv v3 recipe (train_litevae_widiv.py, run with WIDIV_DATASET=impc_bcm):
# linear output, background-weighted loss, background prior. For this dataset the
# foreground is the mouse's own pixels, not a disc, and augmentation is a left-right
# mirror only (widiv_paths.WIDIV_UPRIGHT).
#
# This has to finish before the IMPC diffusion models can be trained
# (train_impc_hellbender.sbatch): they learn inside this autoencoder's latent space.
#
# Needs, copied up from the laptop after fetch_impc_xray.py, organize_impc.py and
# prepare_impc_images.py:
#     dataset/impc_bcm/metadata/   (train_test_split.csv)
#     results/impc_bcm/scaled_512/ and results/impc_bcm/scaled_512_metadata.csv
# (The raw JPEGs in dataset/impc_bcm/images/ are not needed on Hellbender.)
#
# Same environment as the WiDiv and RIL jobs. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/impc/train_litevae_impc_hellbender.sbatch
#
# Weights go to models/litevae_impc_bcm_v3/; a resubmitted job resumes from the
# newest checkpoint there. 80 epochs of about 7,100 training X-rays, about the
# same number of optimiser steps as the WiDiv v3 autoencoder's 250.
#SBATCH --job-name=litevae_impc_bcm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/training/slurm/%x_%j.out
##SBATCH --account=your_account

set -euo pipefail

export WIDIV_DATASET=impc_bcm

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
