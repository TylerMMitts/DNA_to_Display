#!/bin/bash
# Runs code/widiv/train_widiv_noqtl.py, one test per array task: 0 literal (the 40 QTL
# SNPs removed), 1 strict (+/-1 Mb removed and orthogonalized). What the tests are and
# where each run's files go are in that file.
#
# Same environment as train_widiv_hellbender.sbatch. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/widiv/train_widiv_noqtl_hellbender.sbatch             # both
#     sbatch --array=1 code/widiv/train_widiv_noqtl_hellbender.sbatch   # strict only
#SBATCH --job-name=diffusion_widiv_noqtl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

TESTS=(literal strict)
TEST=${TESTS[${SLURM_ARRAY_TASK_ID:-0}]}

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "No-QTL test: $TEST   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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

python -u code/widiv/train_widiv_noqtl.py "$TEST"
