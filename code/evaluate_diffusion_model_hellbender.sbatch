#!/bin/bash
# Runs every per-model analysis for one root diffusion checkpoint on Hellbender.
#
# Pick the checkpoint by editing cfg.checkpoint in evaluate_diffusion_model.py
# before submitting; the job reads it from there like any other run. Results go
# to results/model_analysis/<checkpoint name>/, so evaluating a second model
# never overwrites the first.
#
# Needs the environment from train_onehot_hellbender.sbatch. The LiteVAE ceiling
# in train_vs_test comes from evaluation/reconstruction_fidelity_test.py, run once
# and shared by every model; without it that one section is skipped.
#
# Submit from the project root:
#     mkdir -p results/model_analysis/slurm
#     sbatch code/evaluate_diffusion_model_hellbender.sbatch
#
# Rerunning is safe: each analysis rewrites its own subfolder, and a failed step
# is logged in pipeline_log.csv without stopping the ones that do not need it.
#SBATCH --job-name=evaluate_diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/model_analysis/slurm/%x_%A.out
##SBATCH --account=your_account

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# Checked before any analysis starts, for the same reasons as the training job.
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

# Unbuffered, which each analysis inherits: written to a file, Python otherwise
# holds output back, and the step headers land after the output they announce.
python -u code/evaluate_diffusion_model.py
