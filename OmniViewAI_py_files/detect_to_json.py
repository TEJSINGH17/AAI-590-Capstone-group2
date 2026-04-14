"""
detect_to_json.py — Step 1 of 2
=================================
Run YOLOv8 + ByteTrack on an MP4 and write a single compact JSON file
containing every frame's detections.  No rendering is done here.

Output JSON schema
------------------
{
  "meta": {
    "source":        "test_data/output_ds_1.mp4",
    "model":         "models/yolov8n.pt",
    "conf":          0.25,
    "width":         1920,
    "height":        1080,
    "fps":           30.0,
    "total_frames":  1107
  },
  "frames": [
    {
      "idx": 1,
      "detections": [
        {
          "track_id": 3,
          "cls_id":   2,
          "label":    "car",
          "conf":     0.923,
          "x1": 412.5, "y1": 230.1, "x2": 687.3, "y2": 445.8
        },
        ...
      ]
    },
    ...
  ]
}

All bbox coords are absolute pixels (x1, y1, x2, y2).

Usage
-----
python3 detect_to_json.py \\
    --model  models/yolov8n.pt \\
    --input  test_data/output_ds_1.mp4 \\
    --output runs/json/ds_1_detections.json \\
    --conf   0.25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLOv8+tracker on MP4 and save detections to JSON."
    )
    p.add_argument("--model",  required=True, help="Path to .pt / .engine / .onnx")
    p.add_argument("--input",  required=True, help="Input .mp4 path")
    p.add_argument("--output", required=True, help="Output .json path")
    p.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--device", default="", help="Device: cpu / 0 / 0,1 …")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    input_path = Path(args.input)
    out_path   = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_data: list[dict] = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame,
            conf=args.conf,
            device=args.device or None,
            persist=True,
            verbose=False,
        )
        frame_idx += 1

        detections: list[dict] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            xyxy     = boxes.xyxy.cpu().numpy()
            confs    = boxes.conf.cpu().numpy()
            cls_ids  = boxes.cls.cpu().numpy()
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else range(len(xyxy))
            )
            for i in range(len(xyxy)):
                cid = int(cls_ids[i])
                x1, y1, x2, y2 = xyxy[i].tolist()
                detections.append({
                    "track_id": int(track_ids[i]),
                    "cls_id":   cid,
                    "label":    names.get(cid, str(cid)),
                    "conf":     round(float(confs[i]), 4),
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                })

        frames_data.append({"idx": frame_idx, "detections": detections})

        if frame_idx % 100 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            print(f"  frame {frame_idx:5d}  {frame_idx/elapsed:.1f} fps")

    cap.release()
    elapsed = max(time.time() - t0, 1e-6)

    output = {
        "meta": {
            "source":       str(input_path),
            "model":        str(args.model),
            "conf":         args.conf,
            "width":        width,
            "height":       height,
            "fps":          fps,
            "total_frames": frame_idx,
        },
        "frames": frames_data,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {frame_idx} frames → {out_path}  ({size_mb:.1f} MB)")
    print(f"Total time: {elapsed:.1f}s  ({frame_idx/elapsed:.2f} fps)")


if __name__ == "__main__":
    main()
