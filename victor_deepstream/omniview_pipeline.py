import cv2
import json
import socket
import time
import os
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path

# Configuration
VIDEO_PATH   = "/home/logicpro09/omniview_ai/output_ds_3_reenc.mp4"
MODEL_PATH   = "/home/logicpro09/omniview_ai/yolov8n_lisa_v1.1.pt"
CONF_THRESH  = 0.25
UDP_HOST     = "127.0.0.1"
UDP_PORT     = 5055
DISPLAY      = True

# Dual output modes
UDP_ENABLED  = True
SAVE_JSON    = True
JSON_OUT_DIR = "/home/logicpro09/omniview_ai/runs/hud/ds_3"

@dataclass
class DetectionPayload:
    class_id:   int
    confidence: float
    x_center:   float
    y_center:   float
    width:      float
    height:     float
    source_id:  int
    frame_num:  int
    label:      str

def build_message(detections: List[DetectionPayload], sequence: int) -> bytes:
    payload = {
        "schema_version": 1,
        "timestamp_ms":   int(time.time() * 1000),
        "sequence":       sequence,
        "detections":     [asdict(d) for d in detections],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")

def print_benchmark_report(metrics: dict):
    print("\n" + "="*50)
    print("  OMNIVIEW AI - PERFORMANCE BENCHMARK REPORT")
    print("="*50)
    print(f"  Total frames processed : {metrics['total_frames']}")
    print(f"  Total detections       : {metrics['total_detections']}")
    print(f"  Total runtime          : {metrics['total_time']:.2f}s")
    print(f"  Average FPS            : {metrics['avg_fps']:.2f}")
    print(f"  Min FPS                : {metrics['min_fps']:.2f}")
    print(f"  Max FPS                : {metrics['max_fps']:.2f}")
    print("-"*50)
    print(f"  Avg inference latency  : {metrics['avg_infer_ms']:.2f}ms")
    print(f"  Min inference latency  : {metrics['min_infer_ms']:.2f}ms")
    print(f"  Max inference latency  : {metrics['max_infer_ms']:.2f}ms")
    print("-"*50)
    print(f"  Avg end-to-end latency : {metrics['avg_e2e_ms']:.2f}ms")
    print(f"  Min end-to-end latency : {metrics['min_e2e_ms']:.2f}ms")
    print(f"  Max end-to-end latency : {metrics['max_e2e_ms']:.2f}ms")
    print("-"*50)
    print(f"  Avg UDP payload size   : {metrics['avg_payload_bytes']:.0f} bytes")
    print(f"  Frames with detections : {metrics['frames_with_detections']}")
    print(f"  Detection rate         : {metrics['detection_rate']:.1f}%")
    print("="*50)

def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)
    sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if UDP_ENABLED else None

    # Create JSON output directory if needed
    if SAVE_JSON:
        Path(JSON_OUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"JSON files -> {JSON_OUT_DIR}")

    # Benchmark tracking
    frame_num        = 0
    sequence         = 0
    total_detections = 0
    frames_with_dets = 0
    infer_times      = []
    e2e_times        = []
    fps_list         = []
    payload_sizes    = []

    print("Starting OmniView AI pipeline with benchmarking...")
    if UDP_ENABLED:
        print(f"UDP stream -> {UDP_HOST}:{UDP_PORT}")
    print(f"Video: {VIDEO_PATH}")
    print("-"*50)

    pipeline_start = time.time()

    while True:
        e2e_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]

        infer_start = time.time()
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        infer_end = time.time()
        infer_ms = (infer_end - infer_start) * 1000
        infer_times.append(infer_ms)

        boxes      = results[0].boxes
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = model.names[cls_id]

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width    = (x2 - x1) / img_w
            height   = (y2 - y1) / img_h

            detections.append(DetectionPayload(
                class_id=cls_id,
                confidence=round(conf, 4),
                x_center=round(x_center, 4),
                y_center=round(y_center, 4),
                width=round(width, 4),
                height=round(height, 4),
                source_id=0,
                frame_num=frame_num,
                label=label,
            ))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Build message
        message = build_message(detections, sequence)
        payload_sizes.append(len(message))

        # Send UDP
        if UDP_ENABLED and sock:
            sock.sendto(message, (UDP_HOST, UDP_PORT))

        # Save JSON file
        if SAVE_JSON:
            json_path = os.path.join(JSON_OUT_DIR, f"frame_{frame_num:06d}.json")
            with open(json_path, "w") as f:
                f.write(message.decode("utf-8"))

        if detections:
            total_detections += len(detections)
            frames_with_dets += 1

        e2e_end = time.time()
        e2e_ms  = (e2e_end - e2e_start) * 1000
        e2e_times.append(e2e_ms)

        if len(e2e_times) >= 2:
            fps = 1000 / e2e_ms
            fps_list.append(fps)

        if len(fps_list) > 0:
            cv2.putText(frame, f"FPS: {fps_list[-1]:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Inference: {infer_ms:.1f}ms",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detections)}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        if DISPLAY:
            cv2.imshow("OmniView AI - Traffic Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1
        sequence  += 1

    pipeline_end = time.time()
    total_time   = pipeline_end - pipeline_start

    cap.release()
    if sock:
        sock.close()
    if DISPLAY:
        cv2.destroyAllWindows()

    if SAVE_JSON:
        print(f"JSON files saved to: {JSON_OUT_DIR}")

    metrics = {
        "total_frames":           frame_num,
        "total_detections":       total_detections,
        "total_time":             total_time,
        "avg_fps":                np.mean(fps_list) if fps_list else 0,
        "min_fps":                np.min(fps_list) if fps_list else 0,
        "max_fps":                np.max(fps_list) if fps_list else 0,
        "avg_infer_ms":           np.mean(infer_times),
        "min_infer_ms":           np.min(infer_times),
        "max_infer_ms":           np.max(infer_times),
        "avg_e2e_ms":             np.mean(e2e_times),
        "min_e2e_ms":             np.min(e2e_times),
        "max_e2e_ms":             np.max(e2e_times),
        "avg_payload_bytes":      np.mean(payload_sizes),
        "frames_with_detections": frames_with_dets,
        "detection_rate":         (frames_with_dets / frame_num * 100) if frame_num > 0 else 0,
    }
    print_benchmark_report(metrics)

if __name__ == "__main__":
    main()
