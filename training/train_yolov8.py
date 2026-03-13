"""
train_yolov8.py
---------------
Fine-tunes YOLOv8n on the LISA traffic-sign dataset (or any YOLO-formatted
dataset) and saves the best weights under trained_models/.

Typical workflow
----------------
1. Download the LISA dataset and run prepare_lisa.py once:

       python training/prepare_lisa.py \\
           --lisa_root /path/to/lisaData \\
           --out_root  training/lisa_yolo

2. Train (defaults shown):

       python training/train_yolov8.py \\
           --data    training/lisa_yolo/dataset.yaml \\
           --base_model models/yolov8n.pt \\
           --epochs  50 \\
           --imgsz   640 \\
           --batch   16 \\
           --device  0          # GPU index, or 'cpu'

3. The best model is copied to:
       trained_models/yolov8n_lisa_best.pt

Command-line arguments
----------------------
--data          Path to dataset YAML (required)
--base_model    Starting weights (default: models/yolov8n.pt)
--epochs        Training epochs (default: 50)
--imgsz         Input image size (default: 640)
--batch         Batch size; -1 → ultralytics auto-batch (default: 16)
--workers       DataLoader workers (default: 4)
--device        'cpu', '0', '0,1', etc. (default: '0')
--lr0           Initial learning rate (default: 0.01)
--patience      Early-stopping patience in epochs (default: 15)
--project       Ultralytics run directory (default: runs/train)
--name          Run sub-folder name (default: lisa_yolov8n)
--out_dir       Folder to copy the final model into (default: trained_models)
--resume        Path to a previous run's last.pt to continue training
--augment       Enable mosaic + copy-paste augmentation (flag)
--freeze        Number of backbone layers to freeze (default: 0 = none)
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Run:  pip install ultralytics"
        )

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}\n"
            "Run training/prepare_lisa.py first to create it."
        )

    # Resolve starting weights
    if args.resume:
        weights = args.resume
        log.info("Resuming from checkpoint: %s", weights)
    else:
        weights = args.base_model
        log.info("Starting from base model: %s", weights)

    model = YOLO(weights)

    # Build kwargs for model.train()
    train_kwargs: dict = dict(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Augmentation
        mosaic=1.0 if args.augment else 0.0,
        copy_paste=0.3 if args.augment else 0.0,
        # Freeze backbone layers
        freeze=args.freeze if args.freeze > 0 else None,
        # Verbosity
        verbose=True,
        # Save best + last checkpoints
        save=True,
        save_period=-1,  # only save best/last, not every N epochs
    )

    if args.resume:
        train_kwargs["resume"] = True

    log.info("Starting training with config:\n%s",
             "\n".join(f"  {k}: {v}" for k, v in train_kwargs.items()))

    results = model.train(**train_kwargs)

    # Locate best.pt produced by this run
    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt not found at {best_pt}. "
            "Check the ultralytics run directory."
        )

    # Copy to trained_models/
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "yolov8n_lisa_best.pt"
    shutil.copy2(best_pt, dest)
    log.info("Best model saved to: %s", dest)

    # Also copy last.pt for potential resumption
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.exists():
        shutil.copy2(last_pt, out_dir / "yolov8n_lisa_last.pt")
        log.info("Last checkpoint saved to: %s", out_dir / "yolov8n_lisa_last.pt")

    _print_metrics(results)
    return dest


def _print_metrics(results) -> None:
    """Log a compact summary of the final validation metrics."""
    try:
        metrics = results.results_dict
        log.info(
            "Final metrics → mAP50: %.4f  mAP50-95: %.4f  "
            "Precision: %.4f  Recall: %.4f",
            metrics.get("metrics/mAP50(B)", float("nan")),
            metrics.get("metrics/mAP50-95(B)", float("nan")),
            metrics.get("metrics/precision(B)", float("nan")),
            metrics.get("metrics/recall(B)", float("nan")),
        )
    except Exception:
        pass  # metrics not critical to report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the LISA traffic-sign dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default="training/lisa_yolo/dataset.yaml",
        help="Path to dataset.yaml produced by prepare_lisa.py",
    )
    p.add_argument("--base_model", default="models/yolov8n.pt",
                   help="Base YOLOv8 weights to fine-tune from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size; use -1 for ultralytics auto-batch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0",
                   help="CUDA device index(es) or 'cpu'")
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial learning rate")
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stopping patience (epochs without improvement)")
    p.add_argument("--project", default="runs/train",
                   help="Ultralytics run output directory")
    p.add_argument("--name", default="lisa_yolov8n",
                   help="Sub-folder name inside --project")
    p.add_argument("--out_dir", default="trained_models",
                   help="Directory to copy the final best.pt into")
    p.add_argument("--resume", default=None, metavar="LAST_PT",
                   help="Path to last.pt from a previous run to resume training")
    p.add_argument("--augment", action="store_true",
                   help="Enable mosaic + copy-paste augmentation")
    p.add_argument("--freeze", type=int, default=0,
                   help="Number of backbone layers to freeze (0 = unfreeze all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dest = train(args)
    print(f"\nTraining complete. Fine-tuned model: {dest}")
    print(
        f"\nTo run inference with the new model:\n"
        f"  python test_mp4_yolov8.py "
        f"--model {dest} --input <video.mp4> --output runs/annotated/out.mp4"
    )
