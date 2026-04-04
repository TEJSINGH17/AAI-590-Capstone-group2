"""
Test module: run YOLOv8 on an MP4 video and display detections.
Supports dual-model inference — a base model (COCO) and an optional
second model (e.g. LISA traffic-sign), with merged annotations on one frame.

Examples:
  # Single model
  python3 test_mp4_yolov8.py \
    --input test_data/output_ds_1.mp4 \
    --output runs/annotated/ds1.mp4

  # Dual model (COCO + LISA)
  python3 test_mp4_yolov8.py \
    --input  test_data/output_ds_1.mp4 \
    --output runs/annotated/ds1_dual.mp4 \
    --model  models/yolov8n.pt \
    --model2 models/yolov8n_lisa_v1.1.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ── per-model colour schemes (BGR) ───────────────────────────────────────────
# Model 1 (COCO): white boxes + black text
COCO_BOX  = (255, 140,   0)   # blue (BGR)
COCO_TXT  = (255, 255, 255)
COCO_BG   = (40,  40,  40)

# Model 2 (LISA traffic signs): colour-coded by sign type
_LISA_COLORS = {
    "go":          (50,  230,  80),   # green
    "goforward":   (50,  230,  80),
    "goleft":      (80,  200, 255),   # cyan-blue
    "stop":        (50,   50, 255),   # red
    "stopleft":    (80,   80, 255),
    "warning":     ( 0,  210, 255),   # yellow/amber
    "warningleft": ( 0,  185, 230),
}
_LISA_DEFAULT = (0, 200, 255)


def _lisa_color(label: str) -> Tuple[int, int, int]:
    return _LISA_COLORS.get(label.strip().lower().replace(" ", ""), _LISA_DEFAULT)


# ── drawing helper ────────────────────────────────────────────────────────────

def _draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    labels: List[str],
    box_color,
    label_fn=None,          # callable(label) -> color, or None for fixed color
    thickness: int = 2,
) -> None:
    """Draw bounding boxes + label badges onto frame in-place."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        label = labels[i]
        conf  = confs[i]
        color = label_fn(label) if label_fn else box_color

        # box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # label badge
        txt = f"{label}  {conf*100:.0f}%"
        (tw, th), bl = cv2.getTextSize(txt, font, scale, 1)
        by1 = max(0, y1 - th - bl - 6)
        by2 = y1
        cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, by2), color, -1)
        cv2.putText(frame, txt, (x1 + 4, by2 - bl - 2),
                    font, scale, (10, 10, 10), 1, cv2.LINE_AA)


# ── legend overlay ────────────────────────────────────────────────────────────

def _draw_legend(frame: np.ndarray, model1_name: str, model2_name: str | None) -> None:
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    items = [(COCO_BOX, f"M1: {Path(model1_name).stem}")]
    if model2_name:
        items.append((_LISA_DEFAULT, f"M2: {Path(model2_name).stem}"))

    x, y = w - 260, 14
    for color, txt in items:
        cv2.rectangle(frame, (x, y), (x + 18, y + 14), color, -1)
        cv2.putText(frame, txt, (x + 24, y + 12), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLOv8 on an MP4 — single or dual model."
    )
    p.add_argument("--model",   default="models/yolov8n.pt",
                   help="Primary model (COCO vehicles/persons).")
    p.add_argument("--model2",  default="",
                   help="Optional second model (e.g. LISA traffic signs).")
    p.add_argument("--input",   required=True,  help="Input .mp4 path.")
    p.add_argument("--output",  default="",     help="Output .mp4 path.")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--conf2",   type=float, default=0.30,
                   help="Confidence threshold for model2 (default 0.30).")
    p.add_argument("--device",  default="")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    model1 = YOLO(args.model)
    model2 = YOLO(args.model2) if args.model2 else None

    if model2:
        print(f"Model 1 (COCO) : {args.model}")
        print(f"Model 2 (LISA) : {args.model2}  classes: {list(model2.names.values())}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_count = 0
    start       = time.time()
    device      = args.device or None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── model 1: COCO ────────────────────────────────────────────────────
        r1 = model1.predict(source=frame, conf=args.conf,
                            device=device, verbose=False)
        annotated = frame.copy()

        b1 = r1[0].boxes
        if b1 is not None and len(b1):
            _draw_detections(
                annotated,
                b1.xyxy.cpu().numpy(),
                b1.conf.cpu().numpy(),
                [model1.names[int(c)] for c in b1.cls.cpu().numpy()],
                box_color=COCO_BOX,
            )

        # ── model 2: LISA traffic signs ──────────────────────────────────────
        if model2 is not None:
            r2 = model2.predict(source=frame, conf=args.conf2,
                                device=device, verbose=False)
            b2 = r2[0].boxes
            if b2 is not None and len(b2):
                _draw_detections(
                    annotated,
                    b2.xyxy.cpu().numpy(),
                    b2.conf.cpu().numpy(),
                    [model2.names[int(c)] for c in b2.cls.cpu().numpy()],
                    box_color=_LISA_DEFAULT,
                    label_fn=_lisa_color,
                    thickness=3,
                )

        _draw_legend(annotated, args.model, args.model2 or None)
        frame_count += 1

        if writer is not None:
            writer.write(annotated)

        if not args.no_show:
            cv2.imshow("YOLOv8 Dual Model", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = max(time.time() - start, 1e-6)
    print(f"Processed {frame_count} frames in {elapsed:.2f}s "
          f"({frame_count/elapsed:.2f} FPS).")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
