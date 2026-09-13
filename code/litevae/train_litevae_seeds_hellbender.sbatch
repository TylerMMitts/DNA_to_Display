#!/bin/bash
# Trains the seed LiteVAE autoencoder on Hellbender.
#
# This has to finish before the seed diffusion model can be trained: the
# diffusion model learns inside this autoencoder's latent space.
#
# Once, on the login node, from the project root (compute nodes may not have
# internet, so the environment is built here rather than inside the job):
#     uv sync
#     mkdir -p results/training/slurm
#
# Then submit from the project root:
#     sbatch code/litevae/train_litevae_seeds_hellbender.sbatch
#
# A resubmitted job resumes from the newest checkpoint in models/litevae_seeds/.
#
# Check the partition name with `sinfo` and add your account if your allocation
# requires one.
#SBATCH --job-name=litevae_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/training/slurm/%x_%j.out
##SBATCH --account=your_account

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
nvidia-smi -L

# --frozen --offline uses the environment already synced on the login node and
# fails straight away, instead of hanging, if anything is missing from it.
uv run --frozen --offline python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
uv run --frozen --offline code/litevae/train_litevae_seeds.py
