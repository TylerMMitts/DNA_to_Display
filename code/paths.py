# Every input and output location in the project, defined once.
#
# Scripts import their paths from here instead of hardcoding them, which is
# what lets any script be launched from any working directory and still read
# and write the same places - and what keeps outputs out of code/.

import sys
from pathlib import Path

# The folder that holds code/, dataset/, models/ and results/.
# This file lives directly inside code/, so the root is always one level up.
# Every path below is derived from it, which is what lets a script be launched
# from any working directory and still read and write the same places.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / 'code'

# The four top level folders. Nothing is ever written next to the script that
# produced it, so code/ stays source only.
DATASET_DIR = PROJECT_ROOT / 'dataset'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Inputs.
IMAGES_DIR = DATASET_DIR / 'images'
CROPPED_IMAGES_DIR = RESULTS_DIR / 'cropped_images'
METADATA_DIR = DATASET_DIR / 'metadata'
SNP_PARQUET = METADATA_DIR / 'MEMA_gene_matrix.parquet'
KINSHIP_MATRIX = METADATA_DIR / 'kinship_matrix.csv'
IMAGE_METADATA = METADATA_DIR / 'image_metadata.csv'
SEGMENTATION_DATASET = DATASET_DIR / 'root_features_256'
# The dataset the current segmenter trains on: every annotated image, squashed to
# 256 px and upscaled to 512, as cross_validate_segmentation.py chose.
SEGMENTATION_DATASET_512 = DATASET_DIR / 'root_features_512_all'

# The seed-kernel dataset. Same eight founders and the same one-hot treatment as
# the root data, but a different locus set (33,527 against 43,788), founders
# named rather than numbered, and about 500 genotypes instead of 200. Everything
# it produces stays under its own folder so the two datasets never mix.
SEED_SCANS_DIR = DATASET_DIR / 'seed_scans'
SEED_GENETIC_DIR = DATASET_DIR / 'seed_genetic'
SEED_SNP_PARQUET = SEED_GENETIC_DIR / 'MEMA_complete_pop_B73v5_gene_matrix.parquet'

SEED_RESULTS_DIR = RESULTS_DIR / 'seeds'
SEED_CROPPED_DIR = SEED_RESULTS_DIR / 'cropped_images'
# Written by crop_seed_scans.py: one row per image, carrying the scale bar it
# measured before cropping the bar away.
SEED_IMAGE_METADATA = SEED_RESULTS_DIR / 'seed_image_metadata.csv'

# What the models actually train on: every kernel at one common scale, on a
# square canvas. rescale_seed_crops.py writes both.
SEED_SCALED_DIR = SEED_RESULTS_DIR / 'scaled_images'
SEED_SCALED_METADATA = SEED_RESULTS_DIR / 'seed_scaled_metadata.csv'

# The seed dataset gets its own autoencoder. The root LiteVAE was fitted to
# photographed cross-sections and would be compressing something it has never
# seen, so the diffusion model would be learning in a latent space that does not
# represent kernels well.
SEED_LITEVAE_DIR = MODELS_DIR / 'litevae_seeds'
SEED_LITEVAE_MODEL = SEED_LITEVAE_DIR / 'litevae_seeds_best.pt'

# The seed diffusion model evaluate_seed_diffusion_model.py analyses by default.
# train_seeds.py writes one folder per size; medium is the default size there.
SEED_DIFFUSION_DIR = MODELS_DIR / 'diffusion_seeds_medium'
SEED_DIFFUSION_MODEL = SEED_DIFFUSION_DIR / 'diffusion_seeds_medium_best.pt'

# Size, shape and colour of the real kernels, measured from their pixels by
# kernel_traits/measure_kernel_traits.py. It needs no model, so it is measured
# once here rather than per model.
SEED_KERNEL_TRAITS_DIR = SEED_RESULTS_DIR / 'kernel_traits'

# Per-checkpoint analysis folders for seed models, kept apart from the root ones.
SEED_MODEL_ANALYSIS_DIR = SEED_RESULTS_DIR / 'model_analysis'

# One folder of weights per trainable model, and every checkpoint inside it is
# prefixed with that model's name. A loose checkpoint_epoch_100.pt says nothing
# about which model wrote it once it has been copied somewhere else.
ROOT_DETECTION_MODEL = MODELS_DIR / 'root_detection.pt'
SEGMENTATION_DIR = MODELS_DIR / 'feature_segmentation'
LITEVAE_DIR = MODELS_DIR / 'litevae'
DIFFUSION_ONEHOT_DIR = MODELS_DIR / 'diffusion_onehot'
DIFFUSION_NUMERIC_DIR = MODELS_DIR / 'diffusion_numeric'

# The specific weights each script reaches for when its config is left alone.
SEGMENTATION_MODEL = SEGMENTATION_DIR / 'feature_segmentation_best.pt'
LITEVAE_MODEL = LITEVAE_DIR / 'litevae_epoch_250.pt'
DIFFUSION_ONEHOT_MODEL = DIFFUSION_ONEHOT_DIR / 'diffusion_onehot_epoch_100.pt'
DIFFUSION_NUMERIC_MODEL = DIFFUSION_NUMERIC_DIR / 'diffusion_numeric_epoch_500.pt'

# Training writes checkpoints to models/ but its figures, previews and loss
# curves are results, so they land here instead of beside the weights.
TRAINING_RESULTS_DIR = RESULTS_DIR / 'training'

# Every analysis evaluate_diffusion_model.py runs for one checkpoint lands in a
# folder named after that checkpoint, so two models never share an output folder
# or, more importantly, a cache - several analyses cache results that are only
# valid for the model that produced them.
MODEL_ANALYSIS_DIR = RESULTS_DIR / 'model_analysis'

# The stem used for checkpoint filenames, keyed by the folder holding them.
# Kept in one place so a rename only has to happen here.
MODEL_NAMES = {
    SEGMENTATION_DIR: 'feature_segmentation',
    LITEVAE_DIR: 'litevae',
    SEED_LITEVAE_DIR: 'litevae_seeds',
    DIFFUSION_ONEHOT_DIR: 'diffusion_onehot',
    DIFFUSION_NUMERIC_DIR: 'diffusion_numeric',
}


# Replaces values in a script's cfg block, so one script can drive another
# without the second growing command-line arguments.
#
# A name that is not already a setting is an error rather than a new attribute.
# Otherwise a misspelt override - output_dr for output_dir - would be accepted,
# ignored, and leave the script writing to its default folder, which is exactly
# the silent mix-up this exists to prevent. Called with nothing, it does nothing,
# so a script run on its own behaves as it always has.
def apply_overrides(cfg, overrides):
    for name, value in (overrides or {}).items():
        if not hasattr(cfg, name):
            raise SystemExit(f"unknown setting {name!r} - not in this script's cfg block")
        setattr(cfg, name, value)
    return cfg


# Picks the fastest backend actually present on this machine: CUDA if there is
# an NVIDIA GPU, Apple's Metal (MPS) on Apple silicon, CPU otherwise. Every
# script asks here rather than testing torch.cuda.is_available() itself, so a
# machine with no CUDA still uses its GPU instead of silently dropping to CPU.
#
# Set DNA_DEVICE to force one ('cpu', 'mps', 'cuda') when a backend is present
# but unwanted - an MPS operator gap, or comparing against a CPU baseline.
def pick_device():
    import os
    import torch

    forced = os.environ.get('DNA_DEVICE')
    if forced:
        return forced

    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# Puts code/ on the import path so a script can be launched directly by file
# path and still resolve latent_diffusion, litevae and feature_segmentation.
def add_code_to_path():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))


# Turns a config value into a real file or folder that must already exist.
# Relative paths are read against the project root first so a script gives the
# same answer wherever it was launched from. The working directory and code/
# are tried afterwards so paths written before this refactor still resolve.
def resolve_input(path_str, description):
    p = Path(path_str).expanduser()
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [PROJECT_ROOT / p, Path.cwd() / p, CODE_DIR / p]

    for c in candidates:
        if c.exists():
            return c

    tried = '\n  '.join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"could not find {description} '{path_str}'. Tried:\n  {tried}")


# Turns a config value into somewhere to write. Relative paths always land
# under the project root, which is what keeps every output in results/.
def resolve_output(path_str):
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


# Finds the highest numbered <model_name>_epoch_N.pt in a weights folder.
# Falls back to the old checkpoint_epoch_N.pt name so a folder of weights
# trained before the rename still loads without being renamed by hand.
def find_latest_checkpoint(model_dir, model_name=None):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"weights folder not found: {model_dir}")

    if model_name is None:
        model_name = MODEL_NAMES.get(model_dir, model_dir.name)

    best_path, best_epoch = None, -1
    for pattern in (f'{model_name}_epoch_*.pt', 'checkpoint_epoch_*.pt'):
        for p in model_dir.glob(pattern):
            digits = ''.join(c for c in p.stem.split('_')[-1] if c.isdigit())
            if digits and int(digits) > best_epoch:
                best_path, best_epoch = p, int(digits)
        if best_path is not None:
            return best_path

    # No numbered checkpoint, so fall back to whichever best.pt is present.
    for name in (f'{model_name}_best.pt', 'best.pt', 'best_model.pt'):
        if (model_dir / name).exists():
            return model_dir / name

    raise FileNotFoundError(f"no checkpoints found in {model_dir}")


# Builds the path a training run should write one checkpoint to.
def checkpoint_path(model_dir, epoch, model_name=None):
    model_dir = Path(model_dir)
    if model_name is None:
        model_name = MODEL_NAMES.get(model_dir, model_dir.name)
    return model_dir / f'{model_name}_epoch_{epoch}.pt'


# Builds the path for a run's best scoring checkpoint.
def best_checkpoint_path(model_dir, model_name=None):
    model_dir = Path(model_dir)
    if model_name is None:
        model_name = MODEL_NAMES.get(model_dir, model_dir.name)
    return model_dir / f'{model_name}_best.pt'
