#!/bin/bash
# Trains the WiDiv diffusion model with SNP encoding option F - 32 PCs plus
# gated windows of 200 SNPs (widiv_window_encoding.py) - on split_random, one
# sparsity strength per array task:
#
#   0  sparsity 1e-7   keeps more windows
#   1  sparsity 1e-6   the default
#   2  sparsity 1e-5   keeps fewer windows
#
# Same LiteVAE (v3), latent cache, validation lines and test lines as
# diffusion_widiv_medium_random_vaev3, so results compare directly with the PCA
# model. Every gate is held open with no penalty for the first 100 epochs, and
# the penalty ramps to full over the next 100 (train_widiv.py says why). Each
# run has its own folder, models/diffusion_widiv_medium_random_windows200_pc32_sp1e-06_vaev3/
# and so on, and writes window_gates_epoch_N.csv at every checkpoint to
# results/training/<run>/ listing which windows are still open. In the log,
# "windows open" should fall during the ramp to a number above zero; a warning
# is printed if nearly all shut.
#
# Same environment as train_widiv_hellbender.sbatch. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/widiv/train_widiv_windows_hellbender.sbatch             # all three
#     sbatch --array=1 code/widiv/train_widiv_windows_hellbender.sbatch   # default only
#     sbatch --export=ALL,WIDIV_SPLIT=cluster --array=1 code/widiv/train_widiv_windows_hellbender.sbatch
#
# Without SLURM, run one directly:
#     WIDIV_SNP_ENCODING=windows WIDIV_WINDOW_SPARSITY=1e-6 python code/widiv/train_widiv.py
#SBATCH --job-name=diffusion_widiv_windows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

SPARSITY=(1e-7 1e-6 1e-5)
export WIDIV_SPLIT=${WIDIV_SPLIT:-random}
export DIFFUSION_SIZE=${DIFFUSION_SIZE:-medium}
export WIDIV_SNP_ENCODING=windows
export WIDIV_WINDOW_SIZE=200
export WIDIV_WINDOW_PCS=32
export WIDIV_WINDOW_SPARSITY=${SPARSITY[${SLURM_ARRAY_TASK_ID:-1}]}
export WIDIV_WINDOW_WARMUP=100
export WIDIV_WINDOW_RAMP=100

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "split: $WIDIV_SPLIT   windows of $WIDIV_WINDOW_SIZE + $WIDIV_WINDOW_PCS PCs, sparsity $WIDIV_WINDOW_SPARSITY, warm-up $WIDIV_WINDOW_WARMUP, ramp $WIDIV_WINDOW_RAMP   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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
