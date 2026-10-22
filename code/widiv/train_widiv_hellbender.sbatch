#!/bin/bash
# Trains the WiDiv diffusion model on Hellbender, one delivered split per array task.
#
# Needs the WiDiv LiteVAE first (train_litevae_widiv_hellbender.sbatch), plus
# results/widiv/scaled_512/, dataset/widiv/genetic/ and dataset/widiv/metadata/.
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
#     sbatch code/widiv/train_widiv_hellbender.sbatch             # both splits, medium
#     sbatch --array=0 code/widiv/train_widiv_hellbender.sbatch   # split_random only
#     sbatch --export=ALL,DIFFUSION_SIZE=large code/widiv/train_widiv_hellbender.sbatch
#
# Array index 0 trains split_random, 1 split_cluster. Each has its own folder,
# models/diffusion_widiv_<size>_<split>/, so they run side by side and a
# resubmitted task resumes from its newest checkpoint.
#
# Check the partition name with `sinfo` and add your account if your allocation
# requires one.
#SBATCH --job-name=diffusion_widiv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

SPLITS_TO_RUN=(random cluster)
export WIDIV_SPLIT=${SPLITS_TO_RUN[${SLURM_ARRAY_TASK_ID:-0}]}
export DIFFUSION_SIZE=${DIFFUSION_SIZE:-medium}

# The Python environment to run in. SLURM exports the environment you submitted
# from, so activating before sbatch is usually enough; this picks up a venv in
# the project folder otherwise, and VENV=/path/to/env overrides both.
cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "split: $WIDIV_SPLIT   size: $DIFFUSION_SIZE   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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

python code/widiv/train_widiv.py
