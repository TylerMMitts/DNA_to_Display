#!/bin/bash
# Trains the WiDiv diffusion model on Hellbender CPU nodes, one delivered split
# per array task.
#
# The same run as train_widiv_hellbender.sbatch, for when no GPU is available.
# The first job encodes every image once in all eight views with the frozen
# LiteVAE (about 33,000 encodings; roughly 2 h on a 6-core laptop CPU, less on
# a node) and saves them to results/cache/widiv_latents/; every later job and
# the other split reuse that file. Training then runs from the cached latents:
# about 2 s per step (batch 16) on the laptop, 180 steps an epoch. It saves
# every 25 epochs and resumes from the newest checkpoint, so resubmit it
# whenever the time limit ends a job; work since the last checkpoint is redone.
# Submit one split first (--array=0) if you want the cache built only once.
#
# Needs the WiDiv LiteVAE first (models/litevae_widiv_v2/litevae_widiv_v2_best.pt),
# plus results/widiv/scaled_512/, dataset/widiv/genetic/ and dataset/widiv/metadata/.
#
# Once, on the login node, from the project root:
#     module load miniconda3 && conda create -n dna python=3.11 && conda activate dna
#     pip install -r requirements.txt
#     mkdir -p results/training/slurm
#
# Then submit from the project root:
#     sbatch code/widiv/train_widiv_cpu_hellbender.sbatch             # both splits, medium
#     sbatch --array=0 code/widiv/train_widiv_cpu_hellbender.sbatch   # split_random only
#     sbatch --export=ALL,DIFFUSION_SIZE=small code/widiv/train_widiv_cpu_hellbender.sbatch
#
# Array index 0 trains split_random, 1 split_cluster, each in its own folder.
#
# The partition name and limits are guesses: check `sinfo` for the CPU partition
# and its longest --time, and add your account if your allocation requires one.
#SBATCH --job-name=diffusion_widiv_cpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

SPLITS_TO_RUN=(random cluster)
export WIDIV_SPLIT=${SPLITS_TO_RUN[${SLURM_ARRAY_TASK_ID:-0}]}
export DIFFUSION_SIZE=${DIFFUSION_SIZE:-medium}

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

echo "split: $WIDIV_SPLIT   size: $DIFFUSION_SIZE   node: $(hostname)   job: ${SLURM_JOB_ID:-none}   torch threads: $OMP_NUM_THREADS"

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

python code/widiv/train_widiv.py
