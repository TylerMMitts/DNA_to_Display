#!/bin/bash
# Trains the one-hot diffusion model on Hellbender, one model size per array task.
#
# Once, on the login node, from the project root. Compute nodes may not have
# internet, so the environment is built here rather than inside the job, and it
# needs Python 3.9 or newer - the system python on some nodes is 3.6, which
# cannot run this project at all:
#     module load miniconda3 && conda create -n dna python=3.11 && conda activate dna
#     pip install -r requirements.txt
#     mkdir -p results/training/slurm
#
# The job activates whatever environment you submitted from. If that is a venv
# somewhere unusual, pass it in: sbatch --export=ALL,VENV=/path/to/env ...
#
# Then submit from the project root:
#     sbatch code/latent_diffusion/training/train_onehot_hellbender.sbatch            # all three sizes
#     sbatch --array=1 code/latent_diffusion/training/train_onehot_hellbender.sbatch  # medium only
#
# Array index 0, 1, 2 trains small, medium, large (see SIZE_PRESETS in
# train_onehot.py). Each size has its own folder, so they run side by side and
# a resubmitted task resumes from its newest checkpoint.
#
# Check the partition name with `sinfo` and add your account if your allocation
# requires one.
#SBATCH --job-name=diffusion_onehot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

SIZES=(small medium large)
export DIFFUSION_SIZE=${SIZES[${SLURM_ARRAY_TASK_ID:-1}]}

# The Python environment to run in. SLURM exports the environment you submitted
# from, so activating before sbatch is usually enough; this picks up a venv in
# the project folder otherwise, and VENV=/path/to/env overrides both.
cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "size: $DIFFUSION_SIZE   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# Checked here, in a second, rather than after the queue wait and a half-written
# checkpoint. Python 3.6 cannot run this project at all: torch.load(weights_only)
# needs torch 1.13, and the last torch built for 3.6 is 1.10.
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

python code/latent_diffusion/training/train_onehot.py
