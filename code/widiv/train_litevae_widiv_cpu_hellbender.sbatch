#!/bin/bash
# Trains the WiDiv LiteVAE autoencoder (512 px) on a Hellbender CPU node.
#
# The same run as train_litevae_widiv_hellbender.sbatch, for when no GPU is
# available. It is slow: a forward pass of a batch of 8 at 512 px took 19 s on a
# 6-core laptop CPU, so with the backward pass an epoch (285 batches) is hours
# even on a large node. It saves every 5 epochs and resumes from the newest
# checkpoint, so resubmit it whenever the time limit ends a job.
#
# Needs results/widiv/scaled_512/ and scaled_512_metadata.csv, written by
# code/widiv/prepare_widiv_images.py, plus dataset/widiv/metadata/.
#
# Once, on the login node, from the project root:
#     module load miniconda3 && conda create -n dna python=3.11 && conda activate dna
#     pip install -r requirements.txt
#     mkdir -p results/training/slurm
#
# Then submit from the project root:
#     sbatch code/widiv/train_litevae_widiv_cpu_hellbender.sbatch
#
# The partition name and limits are guesses: check `sinfo` for the CPU partition
# and its longest --time, and add your account if your allocation requires one.
#SBATCH --job-name=litevae_widiv_cpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/training/slurm/%x_%j.out
##SBATCH --account=your_account

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

# paths.pick_device() reads this, so the script runs on CPU without edits.
export DNA_DEVICE=cpu
# Torch's compute threads get most of the allocated cores; the 8 data-loading
# workers the script starts decode PNGs on the rest. Without this torch would
# start one thread per core on the whole node, not per core allocated.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} - 8 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}   cpus: ${SLURM_CPUS_PER_TASK:-?}   torch threads: $OMP_NUM_THREADS"

python - <<'PYCHECK'
import sys
assert sys.version_info >= (3, 9), \
    f"needs Python 3.9 or newer, this is {sys.version.split()[0]}"
import torch
assert int(torch.__version__.split('.')[0]) >= 2, \
    f"needs torch 2.0 or newer, this is {torch.__version__}"
print(f"python {sys.version.split()[0]}   torch {torch.__version__}   "
      f"CPU, {torch.get_num_threads()} threads")
PYCHECK

python code/widiv/train_litevae_widiv.py
