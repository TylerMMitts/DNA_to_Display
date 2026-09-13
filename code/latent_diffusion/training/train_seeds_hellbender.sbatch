#!/bin/bash
# Trains the seed diffusion model on Hellbender, one model size per array task.
#
# Once, on the login node, from the project root (compute nodes may not have
# internet, so the environment is built here rather than inside the job):
#     uv sync
#     mkdir -p results/training/slurm
#
# Then submit from the project root:
#     sbatch code/latent_diffusion/training/train_seeds_hellbender.sbatch            # all three sizes
#     sbatch --array=1 code/latent_diffusion/training/train_seeds_hellbender.sbatch  # medium only
#
# Array index 0, 1, 2 trains small, medium, large (see SIZE_PRESETS in
# train_seeds.py). Each size has its own folder, so they run side by side and
# a resubmitted task resumes from its newest checkpoint.
#
# This trains on seeds. train_onehot_hellbender.sbatch trains on roots.
#
# Check the partition name with `sinfo` and add your account if your allocation
# requires one.
#SBATCH --job-name=diffusion_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

SIZES=(small medium large)
export DIFFUSION_SIZE=${SIZES[${SLURM_ARRAY_TASK_ID:-1}]}
export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

echo "size: $DIFFUSION_SIZE   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# --frozen --offline uses the environment already synced on the login node and
# fails straight away, instead of hanging, if anything is missing from it.
uv run --frozen --offline python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
uv run --frozen --offline code/latent_diffusion/training/train_seeds.py
