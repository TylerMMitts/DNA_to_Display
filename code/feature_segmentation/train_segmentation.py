# Trains the root/stele/vessel segmentation model.
#
# Ultralytics writes its run folder to results/training/, and the best weights
# are copied into models/feature_segmentation/ afterwards so every other
# script picks up the retrain without any config edit.

import shutil
import sys
from pathlib import Path

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import (
    DATASET_DIR, RESULTS_DIR, SEGMENTATION_DATASET, SEGMENTATION_DIR,
    SEGMENTATION_MODEL,
)

from ultralytics import YOLO

from feature_segmentation.prepare_dataset import prepare

# The trained weights are copied out under this name, so a loose .pt file still
# says which model produced it once it has been copied elsewhere.
MODEL_NAME = 'feature_segmentation'


def main():
    # Edit these values, then run:
    #     python code/feature_segmentation/train_segmentation.py
    class cfg:
        # Data.
        source_dataset = DATASET_DIR / 'root_features_new.yolov8'
        prepared_dataset = SEGMENTATION_DATASET
        force_reprepare = False   # True to rebuild the 256x256 dataset from source

        val_fraction = 0.2        # 34 images -> 27 train / 7 val
        rot90_augment = True      # lossless 4x expansion of the training split

        # Nano: with a small annotated set, a larger backbone overfits almost
        # immediately. '-seg' is required; the plain detection checkpoint has no
        # mask head and will not train against segmentation labels. Confirmed
        # by measurement, not just this reasoning: on the 34-image annotated
        # set, nano beat yolov8s-seg on every mask-IoU class (root/stele/vessel)
        # and on vessel-count bias - see results/segmentation_model_comparison.
        model = 'yolov8n-seg.pt'

        imgsz = 256               # matches the latent diffusion model's output
        epochs = 50
        patience = 10             # stop early after 10 epochs with no improvement
        batch = 4
        seed = 0

        # CPU by default. The dataset is ~115 images at 256x256 against a nano
        # backbone, so the GPU buys little, and the 4 GB laptop GPU here is
        # shared with the desktop - training contends with it and fails with
        # "CUDA-capable device(s) is/are busy or unavailable". Set to 0 or
        # 'cuda' on a machine with a dedicated card.
        device = 'cpu'

        # Dataloader workers. Ultralytics defaults to 8; on Windows those are
        # spawned processes that each re-import torch (~2 GB of commit apiece),
        # which exhausts the system commit limit and surfaces as
        # "OSError [WinError 1455]: The paging file is too small". 0 keeps
        # loading in-process, which is plenty for a dataset this small.
        workers = 0

        # Ultralytics writes its run folder of plots and logs here, and the best
        # weights are copied out of it into models/ once training finishes.
        project = RESULTS_DIR / 'training' / MODEL_NAME
        name = 'root_seg'
        weights_dir = SEGMENTATION_DIR

        # Augmentation. Banned: anything that translates or rescales the root.
        translate = 0.0
        scale = 0.0
        mosaic = 0.0
        shear = 0.0
        perspective = 0.0
        copy_paste = 0.0
        mixup = 0.0
        degrees = 0.0             # handled losslessly offline as 90-degree turns

        # Allowed: lossless flips and photometric jitter.
        fliplr = 0.5
        flipud = 0.5
        hsv_h = 0.015
        hsv_s = 0.4
        hsv_v = 0.3

    data_yaml = Path(cfg.prepared_dataset) / 'data.yaml'
    if cfg.force_reprepare or not data_yaml.exists():
        print("Preparing dataset...")
        data_yaml = prepare(
            source_dir=cfg.source_dataset,
            output_dir=cfg.prepared_dataset,
            imgsz=cfg.imgsz,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            rot90_augment=cfg.rot90_augment,
        )
    else:
        print(f"Using prepared dataset at {data_yaml}")
        print("  (set force_reprepare = True to rebuild it)")

    model = YOLO(cfg.model)
    model.train(
        data=str(data_yaml),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch=cfg.batch,
        seed=cfg.seed,
        device=cfg.device,
        workers=cfg.workers,
        project=str(cfg.project),
        name=cfg.name,
        exist_ok=True,
        deterministic=True,
        plots=True,

        translate=cfg.translate,
        scale=cfg.scale,
        mosaic=cfg.mosaic,
        shear=cfg.shear,
        perspective=cfg.perspective,
        copy_paste=cfg.copy_paste,
        mixup=cfg.mixup,
        degrees=cfg.degrees,
        fliplr=cfg.fliplr,
        flipud=cfg.flipud,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
    )

    run_weights = Path(cfg.project) / cfg.name / 'weights' / 'best.pt'
    print(f"\nBest weights from this run: {run_weights}")

    # Ultralytics names every run's output best.pt, so it is copied into models/
    # under this model's name. Every script that loads a segmenter reads that
    # copy, so retraining is picked up without editing any other config.
    weights_dir = Path(cfg.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    published = weights_dir / f'{MODEL_NAME}_best.pt'
    shutil.copy2(run_weights, published)
    print(f"Copied to {published}")
    if published != SEGMENTATION_MODEL:
        print(f"NOTE: paths.SEGMENTATION_MODEL points at {SEGMENTATION_MODEL}, "
              "so other scripts will not pick this up until that matches.")

    print("Validating best checkpoint:")
    YOLO(str(published)).val(data=str(data_yaml), imgsz=cfg.imgsz, split='val',
                             device=cfg.device)


if __name__ == '__main__':
    main()
