#!/bin/bash
# Trains the planted-stele model with the founder-window encoder: the check that
# the window method can find a known gene in the root data before it is trusted
# on the real root genome.
#
# Same table, model size and held-out lines as the PCA planted-stele run
# (train_synthetic_stele_hellbender.sbatch); only the encoder differs. Windows
# follow the parquet's genome order and are cut at chromosome boundaries, and the
# planted block forms its own 20 windows of 200. Gates are held open for 100
# epochs, then the penalty ramps in over 100 (see train_seeds.py for why).
#
# Weights go to models/diffusion_onehot_synthetic_stele_windows200_sp1e-04_wu100r100/,
# a window_gates_epoch_N.csv per checkpoint to results/training/<that run>/.
# Expected if the method works: at least one of the planted windows stays open,
# and test_synthetic_genes.py shows the same stele dose response as the PCA run.
#
# From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/synthetic_stele/train_synthetic_stele_windows_hellbender.sbatch
#SBATCH --job-name=diffusion_synthetic_stele_windows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/training/slurm/%x_%A.out
##SBATCH --account=your_account

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

unset DIFFUSION_SIZE
export DIFFUSION_ENCODING=windows
export DIFFUSION_WINDOW_SIZE=200
export DIFFUSION_WINDOW_SPARSITY=1e-4
export DIFFUSION_WINDOW_WARMUP=100
export DIFFUSION_WINDOW_RAMP=100

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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

python -u code/synthetic_stele/train_synthetic_stele.py
