#!/bin/bash
# Trains the genotype-to-X-ray diffusion model on the IMPC BCM mice on Hellbender
# (train_widiv.py with WIDIV_DATASET=impc_bcm), one model size per array task:
#
#   0  medium   8.8M UNet  (the RIL and WiDiv size; about 1,300 parameters per
#               training image here)
#   1  large    29.3M UNet (about 4,500 per image: the ratio medium had on the RIL)
#
# results/impc_bcm/size_analysis/ says why both: this dataset has 3.7x the RIL's
# training images, about 4 mice per genotype condition instead of 1, and more
# global variety (69 PCs for 90% of image variance against 26-40 for the roots), so
# medium would run at a quarter of the capacity per image any earlier model had.
# Choose between them on validation loss, genotype gain and memorisation ratio.
#
# Encoding: snp_encoding='qtl_only' with every genotype column - the encoder reads
# the 718 knockout columns (0 / 1 / 2) and SEX_male directly, a learned embedding
# per gene; the PCA and window encodings are built for SNP panels and do not apply
# to one knockout per mouse. (The run folder ends in _qtlonly for that reason.)
#
# Split: mouse-level (organize_impc.py). 15% of every gene x zygosity x sex group is
# test and never trained on; 10% of the training mice are validation. Held-out mice
# share their genotype with training mice: the model cannot generate a gene it
# never saw, so it is evaluated on new mice of known genotypes.
# 210 epochs of about 6,600 training mice: about the RIL's 800 in optimiser steps.
# A large run may need a second submission at 2 days; it resumes.
#
# Needs the IMPC LiteVAE first (train_litevae_impc_hellbender.sbatch), plus, copied
# up from the laptop:
#     dataset/impc_bcm/genetic/    (genotypes_012.csv.gz, snp_map.csv, encoder_columns.csv)
#     dataset/impc_bcm/metadata/   (train_test_split.csv)
#     results/impc_bcm/scaled_512/ and results/impc_bcm/scaled_512_metadata.csv
# dataset/impc_bcm/answer_key/ and metadata/mouse_table.csv are for scoring and are
# never read by training.
#
# From the project root:
#     mkdir -p results/training/slurm
#     sbatch code/impc/train_impc_hellbender.sbatch             # medium and large
#     sbatch --array=1 code/impc/train_impc_hellbender.sbatch   # large only
#
# Folders: models/diffusion_impc_bcm_<size>_random_qtlonly_vaev3/. The first task
# to start encodes every image into the latent cache (results/cache/impc_bcm_latents/).
#SBATCH --job-name=diffusion_impc_bcm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=results/training/slurm/%x_%a_%A.out
##SBATCH --account=your_account

set -euo pipefail

export WIDIV_DATASET=impc_bcm
export WIDIV_SPLIT=random
export WIDIV_SNP_ENCODING=qtl_only
SIZES=(medium large)
export DIFFUSION_SIZE=${DIFFUSION_SIZE:-${SIZES[${SLURM_ARRAY_TASK_ID:-0}]}}

cd "$SLURM_SUBMIT_DIR"

VENV="${VENV:-$SLURM_SUBMIT_DIR/myenv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

echo "dataset: $WIDIV_DATASET   encoding: $WIDIV_SNP_ENCODING   size: $DIFFUSION_SIZE   node: $(hostname)   job: ${SLURM_JOB_ID:-none}"
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
