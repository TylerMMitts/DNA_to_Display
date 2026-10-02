#!/bin/bash
# Trains the medium diffusion model on the synthetic-stele copy of the SNP table.
#
# The synthetic table is built on the laptop and copied over with the code:
#     dataset/synthetic_stele/MEMA_gene_matrix_synthetic_stele_k<block_size>.parquet
# Set the block size in code/synthetic_stele/train_synthetic_stele.py first.
#
# Same environment as train_onehot_hellbender.sbatch. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/synthetic_stele/train_synthetic_stele_hellbender.sbatch
#
# Weights go to models/diffusion_onehot_synthetic_stele/, apart from the real
# models, and a resubmitted job resumes from the newest checkpoint there.
#SBATCH --job-name=diffusion_synthetic_stele
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/training/slurm/%x_%A.out
##SBATCH --account=your_account

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# The size is fixed by train_synthetic_stele.py, not by the array variable the
# size-sweep job uses.
unset DIFFUSION_SIZE

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
