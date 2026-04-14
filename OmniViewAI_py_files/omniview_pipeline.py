# =============================================================================
# OmniView AI - Dual Model Detection Pipeline
# Author: Victor Salcedo
# Course: AAI-590 Capstone - University of San Diego
# Description: Real-time traffic sign detection (LISA model) and blind spot
#              detection (COCO model) with live UDP streaming and per-frame
#              JSON file output for AR HUD integration.
#
# Output:
#   - Live UDP stream to port 5055 (consumed by AR HUD)
#   - Per-frame JSON files saved to runs/hud/ds_3/
#   - Annotated MP4 video output
#   - Performance benchmark report
# =============================================================================

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
LISA_MODEL   = "/home/logicpro09/omniview_ai/yolov8n_lisa_v1.1.pt"   # Fine-tuned LISA traffic sign model
COCO_MODEL   = "/home/logicpro09/omniview_ai/yolov8n.pt"              # Base COCO model for vehicle detection
CONF_THRESH  = 0.25   # LISA confidence threshold
COCO_THRESH  = 0.40   # COCO confidence threshold (higher to reduce false positives)
UDP_HOST     = "127.0.0.1"
UDP_PORT     = 5055   # Must match AR HUD listener port
DISPLAY      = True

# Output modes
UDP_ENABLED  = True   # Stream detections live to HUD
SAVE_JSON    = True   # Save per-frame JSON files for offline HUD use
JSON_OUT_DIR = "/home/logicpro09/omniview_ai/runs/hud/ds_3"

# Video recording
RECORD       = True
OUTPUT_VIDEO = "/home/logicpro09/omniview_ai/omniview_output.mp4"

# Blind spot alert thresholds
BLIND_SPOT_YELLOW = 0.25  # Yellow alert: low confidence detection in zone
BLIND_SPOT_RED    = 0.50  # Red alert: high confidence detection in zone

# COCO classes relevant to blind spot detection
BLIND_SPOT_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

@dataclass
class DetectionPayload:
    """Single detection result with normalized coordinates and metadata."""
    class_id:   int
    confidence: float
    x_center:   float   # Normalized 0-1
    y_center:   float   # Normalized 0-1
    width:      float   # Normalized 0-1
    height:     float   # Normalized 0-1
    source_id:  int
    frame_num:  int
    label:      str
    blind_spot: str = "none"  # none, left, right
    model:      str = "lisa"  # lisa or coco

def build_message(detections: List[DetectionPayload], sequence: int) -> bytes:
    """Build JSON payload matching AR HUD schema."""
    payload = {
        "schema_version": 1,
        "timestamp_ms":   int(time.time() * 1000),
        "sequence":       sequence,
        "detections":     [asdict(d) for d in detections],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")

def get_blind_spot_zones(width, height):
    """
    Define blind spot zones matching Sunitha's HUD coordinates.
    Zones cover bottom 52% of frame, 22% width on each side.
    """
    left_zone  = (0,                 int(height * 0.48), int(width * 0.22), height)
    right_zone = (int(width * 0.78), int(height * 0.48), width,            height)
    return left_zone, right_zone

def point_in_zone(cx, cy, zone):
    """Check if bounding box center falls inside a blind spot zone."""
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2

def get_box_color(conf, in_blind_spot):
    """
    Return bounding box color based on detection location and confidence.
    Green: normal detection
    Yellow: low confidence blind spot detection
    Red: high confidence blind spot detection
    """
    if not in_blind_spot:
        return (0, 255, 0)
    elif conf < BLIND_SPOT_RED:
        return (0, 255, 255)
    else:
        return (0, 0, 255)

def draw_blind_spot_zones(frame, left_zone, right_zone):
    """Draw semi-transparent blind spot zone overlays on frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (left_zone[0], left_zone[1]),
                  (left_zone[2], left_zone[3]),
                  (0, 255, 255), -1)
    cv2.rectangle(overlay,
                  (right_zone[0], right_zone[1]),
                  (right_zone[2], right_zone[3]),
                  (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame,
                  (left_zone[0], left_zone[1]),
                  (left_zone[2], left_zone[3]),
                  (0, 255, 255), 2)
    cv2.rectangle(frame,
                  (right_zone[0], right_zone[1]),
                  (right_zone[2], right_zone[3]),
                  (0, 255, 255), 2)
    cv2.putText(frame, "LEFT BLIND SPOT",
                (10, left_zone[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, "RIGHT BLIND SPOT",
                (right_zone[0], right_zone[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

def draw_alerts(frame, left_alert, right_alert):
    """Draw red alert banners when vehicles detected in blind spot zones."""
    if left_alert:
        cv2.rectangle(frame, (20, 75), (420, 115), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: BLIND SPOT LEFT",
                    (35, 103), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    if right_alert:
        cv2.rectangle(frame, (20, 125), (430, 165), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: BLIND SPOT RIGHT",
                    (35, 153), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

def print_benchmark_report(metrics: dict):
    """Print final performance benchmark summary."""
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
    print(f"  Blind spot alerts      : {metrics['blind_spot_alerts']}")
    print(f"  LISA detections        : {metrics['lisa_detections']}")
    print(f"  COCO detections        : {metrics['coco_detections']}")
    print("="*50)

def main():
    # Load both models
    lisa_model = YOLO(LISA_MODEL)
    coco_model = YOLO(COCO_MODEL)
    cap        = cv2.VideoCapture(VIDEO_PATH)
    sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if UDP_ENABLED else None

    # Initialize video writer for annotated output
    writer = None
    if RECORD:
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
        writer  = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_vid, (width, height))
        print(f"Recording -> {OUTPUT_VIDEO}")

    # Create JSON output directory
    if SAVE_JSON:
        Path(JSON_OUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"JSON files -> {JSON_OUT_DIR}")

    # Benchmark tracking variables
    frame_num         = 0
    sequence          = 0
    total_detections  = 0
    frames_with_dets  = 0
    blind_spot_alerts = 0
    lisa_det_count    = 0
    coco_det_count    = 0
    infer_times       = []
    e2e_times         = []
    fps_list          = []
    payload_sizes     = []

    print("Starting OmniView AI pipeline - dual model...")
    print(f"LISA model: traffic sign detection")
    print(f"COCO model: vehicle/pedestrian blind spot detection")
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
        left_zone, right_zone = get_blind_spot_zones(img_w, img_h)

        # Draw blind spot zone overlays
        draw_blind_spot_zones(frame, left_zone, right_zone)

        # Run inference on both models simultaneously
        infer_start  = time.time()
        lisa_results = lisa_model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        coco_results = coco_model.predict(source=frame, conf=COCO_THRESH, verbose=False)
        infer_end    = time.time()
        infer_ms     = (infer_end - infer_start) * 1000
        infer_times.append(infer_ms)

        detections  = []
        left_alert  = False
        right_alert = False

        # Process LISA detections (traffic signs only)
        for box in lisa_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = lisa_model.names[cls_id]

            # Normalize coordinates for HUD
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width    = (x2 - x1) / img_w
            height   = (y2 - y1) / img_h

            detections.append(DetectionPayload(
                class_id=cls_id, confidence=round(conf, 4),
                x_center=round(x_center, 4), y_center=round(y_center, 4),
                width=round(width, 4), height=round(height, 4),
                source_id=0, frame_num=frame_num, label=label,
                blind_spot="none", model="lisa",
            ))

            # Draw green box for traffic signs
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            lisa_det_count += 1

        # Process COCO detections (vehicles and pedestrians for blind spot)
        for box in coco_results[0].boxes:
            cls_id  = int(box.cls[0])
            label   = coco_model.names[cls_id]

            # Only process blind spot relevant classes
            if label not in BLIND_SPOT_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf    = float(box.conf[0])

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width    = (x2 - x1) / img_w
            height   = (y2 - y1) / img_h

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check if detection falls in blind spot zone
            in_left         = point_in_zone(cx, cy, left_zone)
            in_right        = point_in_zone(cx, cy, right_zone)
            in_blind_spot   = in_left or in_right
            blind_spot_side = "left" if in_left else "right" if in_right else "none"

            # Trigger alerts for high confidence blind spot detections
            if in_left and conf >= BLIND_SPOT_RED:
                left_alert = True
                blind_spot_alerts += 1
            if in_right and conf >= BLIND_SPOT_RED:
                right_alert = True
                blind_spot_alerts += 1

            color = get_box_color(conf, in_blind_spot)

            detections.append(DetectionPayload(
                class_id=cls_id, confidence=round(conf, 4),
                x_center=round(x_center, 4), y_center=round(y_center, 4),
                width=round(width, 4), height=round(height, 4),
                source_id=0, frame_num=frame_num, label=label,
                blind_spot=blind_spot_side, model="coco",
            ))

            # Draw colored box based on blind spot status
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            coco_det_count += 1

        # Draw alert banners if blind spot triggered
        draw_alerts(frame, left_alert, right_alert)

        # Build and send UDP payload
        message = build_message(detections, sequence)
        payload_sizes.append(len(message))

        if UDP_ENABLED and sock:
            sock.sendto(message, (UDP_HOST, UDP_PORT))

        # Save per-frame JSON file
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
            fps_val = 1000 / e2e_ms
            fps_list.append(fps_val)

        # Display performance metrics on frame
        if len(fps_list) > 0:
            cv2.putText(frame, f"FPS: {fps_list[-1]:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Inference: {infer_ms:.1f}ms",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detections)}",
                        (20, 215), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        if RECORD and writer:
            writer.write(frame)

        if DISPLAY:
            cv2.imshow("OmniView AI - Traffic Sign + Blind Spot Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1
        sequence  += 1

    # Cleanup
    pipeline_end = time.time()
    total_time   = pipeline_end - pipeline_start

    cap.release()
    if writer:
        writer.release()
        print(f"Video saved to: {OUTPUT_VIDEO}")
    if sock:
        sock.close()
    if DISPLAY:
        cv2.destroyAllWindows()
    if SAVE_JSON:
        print(f"JSON files saved to: {JSON_OUT_DIR}")

    # Print final benchmark report
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
        "blind_spot_alerts":      blind_spot_alerts,
        "lisa_detections":        lisa_det_count,
        "coco_detections":        coco_det_count,
    }
    print_benchmark_report(metrics)

if __name__ == "__main__":
    main()
