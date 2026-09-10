# Answers one question - can this machine run the pipeline, and if not, what is
# the single next command - so a new machine is diagnosed in one run instead of
# by hitting each failure in turn.
#
# Reads only. Nothing here writes, downloads or trains. Exit code is 0 when
# every script in the repo would run, 1 when something must be fixed first, so
# an agent can branch on it rather than parse the text.
#
#   uv run code/preflight.py

import importlib
import os
import platform
import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (
    CROPPED_IMAGES_DIR, DIFFUSION_ONEHOT_DIR, IMAGES_DIR, IMAGE_METADATA,
    LITEVAE_DIR, MODELS_DIR, ROOT_DETECTION_MODEL, SEGMENTATION_DIR,
    SNP_PARQUET, SYNTHETIC_MARKER, find_latest_checkpoint,
)

# (import name, pip name, what stops working without it)
REQUIRED = [
    ('numpy', 'numpy', 'everything'),
    ('pandas', 'pandas', 'metadata and SNP loading'),
    ('torch', 'torch', 'everything'),
    ('torchvision', 'torchvision', 'image transforms'),
    ('sklearn', 'scikit-learn', 'the PCA behind SNP encoding'),
    ('pyarrow', 'pyarrow', 'reading the SNP .parquet'),
    ('PIL', 'Pillow', 'image IO'),
    ('cv2', 'opencv-python', 'root detection'),
    ('skimage', 'scikit-image', 'watershed vessel splitting'),
    ('pywt', 'PyWavelets', "LiteVAE's wavelet transform"),
    ('ultralytics', 'ultralytics', 'YOLO detection and segmentation'),
    ('matplotlib', 'matplotlib', 'every figure'),
    ('tqdm', 'tqdm', 'progress bars'),
]

# Not required to run the pipeline, so a miss is a note rather than a failure.
OPTIONAL = [('lpips', 'lpips', 'only litevae/evaluation/test_litevae.py')]

PASS, WARN, FAIL = 'ok  ', 'note', 'FAIL'


def line(status, label, detail=''):
    print(f"  [{status}] {label:<34} {detail}")


# O(1) - a version tuple compare.
def check_python():
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro} ({platform.machine()})"
    if v < (3, 9):
        line(FAIL, 'python', f"{got} - 3.9 or newer needed")
        return False
    line(PASS, 'python', got)
    return True


# O(n packages) - an import each, which is the only honest test.
def check_packages():
    missing = []
    for module, pip_name, why in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pip_name)
            line(FAIL, pip_name, f"missing - breaks {why}")
    for module, pip_name, why in OPTIONAL:
        try:
            importlib.import_module(module)
        except ImportError:
            line(WARN, pip_name, f"missing - {why}")
    if not missing:
        line(PASS, 'packages', f"all {len(REQUIRED)} required imports present")
    return missing


# The trap this whole function exists for: `pip install torch` on Windows
# yields a CPU-only wheel that reports cuda unavailable on a machine with a
# perfectly good GPU. torch.version.cuda is None only for such a build, which
# separates 'no GPU here' from 'wrong wheel installed'.
# O(1)
def check_torch_build():
    import torch

    line(PASS, 'torch', torch.__version__)
    built_for_cuda = torch.version.cuda is not None
    has_cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()

    if has_cuda:
        line(PASS, 'gpu', f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}")
    elif mps:
        line(PASS, 'gpu', 'Apple Metal (MPS)')
    elif built_for_cuda:
        line(WARN, 'gpu', 'CUDA build installed but no GPU visible - will use CPU')
    else:
        note = 'CPU-only torch wheel'
        if platform.system() == 'Windows':
            note += ' - if this box has an NVIDIA GPU, see step 2 of RUNNING_ON_WINDOWS.md'
        line(WARN, 'gpu', note)

    from paths import pick_device
    forced = os.environ.get('DNA_DEVICE')
    line(PASS, 'device chosen', pick_device() + (f"  (forced by DNA_DEVICE={forced})" if forced else ''))
    return True


# O(n files) - a directory listing per input.
def check_dataset():
    ok = True
    n_images = len(list(IMAGES_DIR.glob('*.JPG'))) + len(list(IMAGES_DIR.glob('*.jpg'))) if IMAGES_DIR.exists() else 0
    n_cropped = len(list(CROPPED_IMAGES_DIR.glob('*.JPG'))) + len(list(CROPPED_IMAGES_DIR.glob('*.jpg'))) if CROPPED_IMAGES_DIR.exists() else 0

    if n_images:
        line(PASS, 'dataset/images', f"{n_images} images")
    else:
        line(FAIL, 'dataset/images', 'empty')
        ok = False
    line(PASS if n_cropped else WARN, 'results/cropped_images',
         f"{n_cropped} crops" if n_cropped else 'empty - run code/crop_root_model.py')

    for label, path in [('image_metadata.csv', IMAGE_METADATA), ('SNP matrix (.parquet)', SNP_PARQUET)]:
        if path.exists():
            line(PASS, label, f"{path.stat().st_size / 1e6:.1f} MB")
        else:
            line(FAIL, label, f"missing at {path}")
            ok = False
    return ok


# Loads each checkpoint for real rather than checking the filename, because a
# truncated or half-downloaded .pt only announces itself on load.
# O(n checkpoints), dominated by reading the weights off disk.
def check_weights():
    import torch

    targets = [
        ('root_detection.pt', ROOT_DETECTION_MODEL),
        ('litevae', find_latest_checkpoint(LITEVAE_DIR) if LITEVAE_DIR.exists() else None),
        ('diffusion_onehot', find_latest_checkpoint(DIFFUSION_ONEHOT_DIR) if DIFFUSION_ONEHOT_DIR.exists() else None),
        ('feature_segmentation', find_latest_checkpoint(SEGMENTATION_DIR) if SEGMENTATION_DIR.exists() else None),
    ]
    found = 0
    for label, path in targets:
        if path is None or not Path(path).exists():
            line(FAIL, label, 'missing')
            continue
        try:
            # weights_only=False because the diffusion checkpoints carry the
            # fitted PCA basis alongside the tensors. map_location='cpu' is
            # what makes a checkpoint written on any machine load on this one -
            # there is no conversion step, this is it.
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else 'n/a'
            line(PASS, label, f"{Path(path).name}  epoch {epoch}")
            found += 1
        except Exception as exc:
            line(FAIL, label, f"{Path(path).name} will not load - {type(exc).__name__}")
    return found, len(targets)


def main():
    print(f"\nDNA_to_Display preflight - {platform.system()} {platform.release()}")
    print(f"project root: {MODELS_DIR.parent}\n")

    print("environment")
    py_ok = check_python()
    missing = check_packages()
    if missing:
        print("\n  next:  uv run code/preflight.py   (uv installs these for you)")
        print("  or, without uv:  pip install " + ' '.join(missing) + "\n")
        return 1
    check_torch_build()

    print("\ninputs")
    data_ok = check_dataset()

    print("\nweights")
    found, total = check_weights()

    print()
    if SYNTHETIC_MARKER.exists():
        print("This project is SYNTHETIC. Weights are random and images are drawn,")
        print("so it runs end to end but no number it produces means anything.")
        print(f"Delete {SYNTHETIC_MARKER.name} in models/ when real weights arrive.\n")

    ready = py_ok and data_ok and found == total
    if ready:
        print("READY. Every script should run. Try:")
        print("  uv run code/latent_diffusion/generation/generate_from_dataset.py\n")
        return 0

    print("NOT READY.")
    if not data_ok or found == 0:
        print("  Either drop the handed-over dataset/ and models/ folders in place,")
        print("  or mint a stand-in project to rehearse against:")
        print("    uv run code/make_synthetic_project.py")
    elif found < total:
        print(f"  {total - found} of {total} checkpoints missing - copy the rest of the")
        print("  handed-over models/ folder in, then rerun.")
    print()
    return 1


if __name__ == '__main__':
    sys.exit(main())
