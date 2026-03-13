"""
prepare_lisa.py
---------------
Converts the LISA Traffic Sign dataset (Kaggle release) from its native
CSV annotation format into the YOLO flat-file layout expected by ultralytics:

    <out_root>/
        images/
            train/  *.jpg
            val/
        labels/
            train/  *.txt  (one per image, YOLO xywh normalised)
            val/
        dataset.yaml

Actual dataset layout on disk
------------------------------
    training_data/Lisa/
        Annotations/Annotations/
            dayTrain/dayClip1/frameAnnotationsBOX.csv
            dayTrain/dayClip2/frameAnnotationsBOX.csv
            ...
            nightTrain/...
            daySequence1/...
            daySequence2/...
            nightSequence1/...
            nightSequence2/...
        dayTrain/dayTrain/dayClip1/frames/dayClip1--00000.jpg
        nightTrain/nightTrain/...
        daySequence1/daySequence1/frames/...
        ...

CSV columns (semicolon-separated, no Occluded column in this release)
----------------------------------------------------------------------
    Filename ; Annotation tag ;
    Upper left corner X ; Upper left corner Y ;
    Lower right corner X ; Lower right corner Y ;
    Origin file ; Origin frame number ;
    Origin track ; Origin track frame number

The Filename column stores paths like "dayTraining/dayClip1--00000.jpg".
The bare filename (e.g. "dayClip1--00000.jpg") is unique across the dataset,
so we build a basename → absolute-path index once from all frames/ dirs.

Usage
-----
    python training/prepare_lisa.py \\
        --lisa_root  training_data/Lisa \\
        --out_root   training/lisa_yolo \\
        --val_split  0.15 \\
        --seed       42
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from pathlib import Path

try:
    from .lisa_classes import CLASS_NAMES, TAG_TO_IDX
except ImportError:
    # Allows direct script execution: python training/prepare_lisa.py
    from lisa_classes import CLASS_NAMES, TAG_TO_IDX

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image index
# ---------------------------------------------------------------------------

def _build_image_index(lisa_root: Path) -> dict[str, Path]:
    """
    Walk every frames/ directory and map bare filename → absolute Path.
    This sidesteps the mismatch between CSV Filename paths and real paths.
    """
    index: dict[str, Path] = {}
    for img_path in lisa_root.rglob("frames/*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            index[img_path.name] = img_path
    log.info("Image index built: %d images found.", len(index))
    return index


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _read_annotations(
    lisa_root: Path,
    image_index: dict[str, Path],
) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """
    Parse all frameAnnotationsBOX.csv files.

    Returns
    -------
    dict mapping absolute image path string →
        list of (class_idx, cx, cy, w, h) in normalised YOLO format.
    """
    import cv2

    ann: dict[str, list[tuple[int, float, float, float, float]]] = {}
    skipped_tags: set[str] = set()
    missing_images: int = 0

    csv_files = list(
        (lisa_root / "Annotations").rglob("frameAnnotationsBOX.csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No frameAnnotationsBOX.csv files found under "
            f"{lisa_root / 'Annotations'}. "
            "Check your --lisa_root path."
        )
    log.info("Found %d annotation CSV file(s).", len(csv_files))

    for csv_path in csv_files:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}

                raw_fname = row.get("Filename", "")
                tag = row.get("Annotation tag", "")
                if not raw_fname or not tag:
                    continue

                if tag not in TAG_TO_IDX:
                    skipped_tags.add(tag)
                    continue

                # Resolve image path using bare filename only
                bare = Path(raw_fname).name
                img_path = image_index.get(bare)
                if img_path is None:
                    missing_images += 1
                    log.debug("Image not in index: %s", bare)
                    continue

                try:
                    x1 = float(row["Upper left corner X"])
                    y1 = float(row["Upper left corner Y"])
                    x2 = float(row["Lower right corner X"])
                    y2 = float(row["Lower right corner Y"])
                except (KeyError, ValueError):
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                cx = max(0.0, min(1.0, (x1 + x2) / 2.0 / img_w))
                cy = max(0.0, min(1.0, (y1 + y2) / 2.0 / img_h))
                w = max(0.0, min(1.0, (x2 - x1) / img_w))
                h = max(0.0, min(1.0, (y2 - y1) / img_h))

                class_idx = TAG_TO_IDX[tag]
                key = str(img_path)
                ann.setdefault(key, []).append((class_idx, cx, cy, w, h))

    if skipped_tags:
        log.warning(
            "Skipped %d unknown tag(s): %s",
            len(skipped_tags),
            sorted(skipped_tags),
        )
    if missing_images:
        log.warning("%d annotation rows had no matching image.", missing_images)

    log.info("Collected annotations for %d unique images.", len(ann))
    return ann


# ---------------------------------------------------------------------------
# Dataset writing
# ---------------------------------------------------------------------------

def _write_split(
    items: list[tuple[str, list]],
    split: str,
    out_root: Path,
) -> None:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_src, boxes in items:
        src = Path(img_src)
        shutil.copy2(src, img_dir / src.name)

        lbl_path = lbl_dir / (src.stem + ".txt")
        with lbl_path.open("w") as f:
            for cls, cx, cy, bw, bh in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    log.info("  %s: %d images written.", split, len(items))


def _write_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "dataset.yaml"
    names_block = "\n".join(
        f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)
    )
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(CLASS_NAMES)}\n"
        f"\nnames:\n{names_block}\n"
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def prepare(
    lisa_root: str,
    out_root: str,
    val_split: float,
    seed: int,
) -> Path:
    lisa_path = Path(lisa_root)
    out_path = Path(out_root)

    if not lisa_path.exists():
        raise FileNotFoundError(f"LISA root not found: {lisa_path}")

    image_index = _build_image_index(lisa_path)
    ann = _read_annotations(lisa_path, image_index)

    if not ann:
        raise RuntimeError(
            "No annotated images found. Check your --lisa_root path."
        )

    items = list(ann.items())
    random.seed(seed)
    random.shuffle(items)

    n_val = max(1, int(len(items) * val_split))
    val_items = items[:n_val]
    train_items = items[n_val:]

    log.info(
        "Split → train: %d  val: %d", len(train_items), len(val_items)
    )

    _write_split(train_items, "train", out_path)
    _write_split(val_items, "val", out_path)

    yaml_path = _write_yaml(out_path)
    log.info("Dataset YAML written to: %s", yaml_path)
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare LISA dataset for YOLOv8 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lisa_root",
        default="training_data/Lisa",
        help="Root directory of the LISA dataset",
    )
    p.add_argument(
        "--out_root",
        default="training/lisa_yolo",
        help="Output directory for the converted YOLO dataset",
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of images reserved for validation",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    yaml_path = prepare(
        args.lisa_root, args.out_root, args.val_split, args.seed
    )
    print(
        f"\nDataset ready. Pass this to train_yolov8.py:\n"
        f"  --data {yaml_path}"
    )
