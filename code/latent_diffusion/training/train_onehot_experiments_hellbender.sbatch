#!/bin/bash
# Two experiments on why the root model's genotype knowledge does not carry over
# to held-out lines, one run per array task, all at the medium size:
#
#   0  lines50     train on 50 of the 110 training lines
#   1  lines80     train on 80 of them
#   2  cnoise0p5   all 110 lines, condition_noise 0.5
#   3  cnoise1     all 110 lines, condition_noise 1.0
#   4  cnoise2     all 110 lines, condition_noise 2.0
#
# The held-out 27 lines are the same in every run and in the existing
# diffusion_onehot_medium model, which is the 110-line, no-noise reference.
# Compare runs at the same epoch (300 matches the deck) with
# analyze_genotype_memorization.py: does held-out genotype gain rise with more
# lines, or with more noise?
#
# Same environment as train_onehot_hellbender.sbatch. From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/latent_diffusion/training/train_onehot_experiments_hellbender.sbatch
#     sbatch --array=0,1 code/latent_diffusion/training/train_onehot_experiments_hellbender.sbatch   # subsets only
#
# Without a GPU, run one directly instead - the same settings through
# environment variables:
#     DIFFUSION_TRAIN_LINES=50 python code/latent_diffusion/training/train_onehot.py
#     DIFFUSION_CONDITION_NOISE=0.5 python code/latent_diffusion/training/train_onehot.py
#
# Each run gets its own folder (models/diffusion_onehot_medium_lines50, ...),
# and a resubmitted task resumes from its newest checkpoint.
#SBATCH --job-name=diffusion_onehot_experiments
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

LINES=(50 80 "" "" "")
NOISE=("" "" 0.5 1.0 2.0)
i=${SLURM_ARRAY_TASK_ID:-0}
export DIFFUSION_SIZE=medium
export DIFFUSION_TRAIN_LINES=${LINES[$i]}
export DIFFUSION_CONDITION_NOISE=${NOISE[$i]}

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "task $i: lines=${DIFFUSION_TRAIN_LINES:-all} noise=${DIFFUSION_CONDITION_NOISE:-0}   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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

python -u code/latent_diffusion/training/train_onehot.py
