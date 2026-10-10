#!/bin/bash
# Trains the medium seed model with the founder-window encoder, one sparsity
# strength per array task (see window_encoding.py and train_seeds.py):
#
#   0  sparsity 3e-5   keeps more windows
#   1  sparsity 1e-4   the default
#   2  sparsity 3e-4   keeps fewer windows
#
# All use windows of 200 genes and the same 93 held-out lines as the PCA seed
# model, so results compare directly with diffusion_seeds_medium. Each run has
# its own folder (models/diffusion_seeds_medium_windows200_sp1e-04, ...) and a
# window_gates_epoch_N.csv per checkpoint in results/training/<run>/ listing
# which windows are still open.
#
# Same environment as train_seeds_hellbender.sbatch. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/latent_diffusion/training/train_seeds_windows_hellbender.sbatch
#     sbatch --array=1 code/latent_diffusion/training/train_seeds_windows_hellbender.sbatch   # default only
#
# Without a GPU, run one directly:
#     DIFFUSION_ENCODING=windows DIFFUSION_WINDOW_SPARSITY=1e-4 python code/latent_diffusion/training/train_seeds.py
#SBATCH --job-name=diffusion_seeds_windows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

SPARSITY=(3e-5 1e-4 3e-4)
export DIFFUSION_SIZE=medium
export DIFFUSION_ENCODING=windows
export DIFFUSION_WINDOW_SIZE=200
export DIFFUSION_WINDOW_SPARSITY=${SPARSITY[${SLURM_ARRAY_TASK_ID:-1}]}

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "windows of $DIFFUSION_WINDOW_SIZE, sparsity $DIFFUSION_WINDOW_SPARSITY   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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

python -u code/latent_diffusion/training/train_seeds.py
