# Turns the annotation export into a 256x256 YOLO dataset.
#
# Adds 90-degree rotations to the training split only, which is a lossless 4x
# expansion of a small annotated set.

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Puts code/ on the import path so this file can be run directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import DATASET_DIR


def rot90_polygon(coords, k):
    # Rotates normalised polygon coordinates by k * 90 degrees CCW.
    #
    # Only valid for square images, where a 90-degree turn maps the image onto
    # itself and no content is lost. In image coordinates (x right, y down), a
    # single CCW turn sends (x, y) -> (y, 1 - x).
    # np.array (not asarray) so an ndarray argument is copied rather than
    # aliased - otherwise this rotates the caller's coordinates in place, and
    # successive calls compound instead of starting fresh.
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
    for _ in range(k % 4):
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = y
        pts[:, 1] = 1.0 - x
    return pts.reshape(-1)


def read_label(path):
    # Reads a YOLO segmentation label file into [(class_id, coords), ...].
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:          # need a class plus at least 3 xy pairs
            continue
        rows.append((int(parts[0]), [float(v) for v in parts[1:]]))
    return rows


def write_label(path, rows):
    lines = []
    for class_id, coords in rows:
        formatted = ' '.join(f'{v:.6f}' for v in coords)
        lines.append(f'{class_id} {formatted}')
    Path(path).write_text('\n'.join(lines) + ('\n' if lines else ''))


def resize_square(src_image, dst_image, imgsz):
    # Squash-resizes to imgsz x imgsz, matching transforms.Resize((256, 256)).
    with Image.open(src_image) as im:
        im = im.convert('RGB').resize((imgsz, imgsz), Image.LANCZOS)
        im.save(dst_image, quality=95)


def prepare(source_dir, output_dir, imgsz=256, val_fraction=0.2, seed=0,
            rot90_augment=True):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    src_images = sorted((source_dir / 'train' / 'images').glob('*'))
    src_images = [p for p in src_images if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not src_images:
        raise SystemExit(f"no images found under {source_dir / 'train' / 'images'}")

    label_dir = source_dir / 'train' / 'labels'

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(src_images))
    n_val = max(1, int(round(len(src_images) * val_fraction)))
    val_idx = set(order[:n_val].tolist())

    if output_dir.exists():
        shutil.rmtree(output_dir)

    counts = {'train': 0, 'val': 0}
    for split in ('train', 'val'):
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(src_images):
        split = 'val' if i in val_idx else 'train'
        label_path = label_dir / f'{image_path.stem}.txt'
        if not label_path.exists():
            print(f"  skipping {image_path.name}: no matching label file")
            continue

        rows = read_label(label_path)
        stem = image_path.stem

        dst_image = output_dir / split / 'images' / f'{stem}.jpg'
        resize_square(image_path, dst_image, imgsz)
        write_label(output_dir / split / 'labels' / f'{stem}.txt', rows)
        counts[split] += 1

        # Rotations go on the training split only. Augmenting validation would
        # inflate the metrics with near-duplicates of images already scored.
        if split == 'train' and rot90_augment:
            with Image.open(dst_image) as im:
                base = np.array(im.convert('RGB'))
            for k in (1, 2, 3):
                rotated = np.rot90(base, k=k)
                Image.fromarray(rotated).save(
                    output_dir / split / 'images' / f'{stem}_rot{k * 90}.jpg', quality=95)
                write_label(
                    output_dir / split / 'labels' / f'{stem}_rot{k * 90}.txt',
                    [(cid, rot90_polygon(coords, k)) for cid, coords in rows])
                counts[split] += 1

    names = ['root', 'stele', 'vessel']
    # Absolute paths, so training works regardless of working directory.
    yaml_path = output_dir / 'data.yaml'
    yaml_path.write_text(
        f"path: {output_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )

    print(f"Prepared {counts['train']} train / {counts['val']} val images "
          f"at {imgsz}x{imgsz} in {output_dir}")
    if rot90_augment:
        print(f"  (train includes 90/180/270-degree rotations of each source image)")
    return yaml_path


def _self_test():
    # Verifies the rotation transform keeps labels aligned with the pixels.
    #
    # Rasterises a polygon, rotates image and label independently, and checks the
    # two still agree. A sign error in rot90_polygon would silently produce
    # plausible-looking but misaligned training data, which is the kind of bug
    # that only shows up as mysteriously bad metrics much later.
    from PIL import ImageDraw

    # Rasterised at high resolution so edge discretisation does not dominate
    # the IoU; the residual gap from 1.0 is antialiasing at the polygon border.
    size = 512
    poly = np.array([0.1, 0.1, 0.7, 0.2, 0.5, 0.6, 0.15, 0.45])

    def rasterise(coords):
        img = Image.new('L', (size, size), 0)
        pts = (np.asarray(coords).reshape(-1, 2) * size).flatten().tolist()
        ImageDraw.Draw(img).polygon(pts, fill=255)
        return np.array(img) > 127

    base = rasterise(poly)
    ok = True
    for k in (1, 2, 3):
        from_image = np.rot90(base, k=k)
        from_label = rasterise(rot90_polygon(poly, k))
        iou = (from_image & from_label).sum() / max((from_image | from_label).sum(), 1)
        print(f"  rot{k * 90}: IoU(image-rotated, label-rotated) = {iou:.4f}")
        ok &= iou > 0.97
    print("rotation self-test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        raise SystemExit(0 if _self_test() else 1)

    prepare(
        source_dir=DATASET_DIR / 'root_features.yolov8',
        output_dir=DATASET_DIR / 'root_features_256',
        imgsz=256,
        val_fraction=0.2,
        seed=0,
        rot90_augment=True,
    )
