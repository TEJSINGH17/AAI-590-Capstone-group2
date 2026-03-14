import cv2
import json
import socket
import time
from ultralytics import YOLO
from dataclasses import dataclass, asdict
from typing import List

# Configuration
VIDEO_PATH   = "/home/logicpro09/omniview_ai/output_ds_3_reenc.mp4"
MODEL_PATH   = "/home/logicpro09/omniview_ai/yolov8n_lisa_best.pt"
CONF_THRESH  = 0.25
UDP_HOST     = "127.0.0.1"
UDP_PORT     = 5055
DISPLAY      = True

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

def main():
    model  = YOLO(MODEL_PATH)
    cap    = cv2.VideoCapture(VIDEO_PATH)
    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    frame_num = 0
    sequence  = 0
    total_dets = 0

    print("Starting OmniView AI pipeline...")
    print(f"UDP stream -> {UDP_HOST}:{UDP_PORT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        boxes   = results[0].boxes

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = model.names[cls_id]

            # Normalized coordinates for HUD
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

            # Draw bounding box on frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Send UDP payload
        message = build_message(detections, sequence)
        sock.sendto(message, (UDP_HOST, UDP_PORT))

        if detections:
            total_dets += len(detections)
            print(f"Frame {frame_num}: {len(detections)} detections -> UDP sent")

        # Display video
        if DISPLAY:
            cv2.imshow("OmniView AI - Traffic Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1
        sequence  += 1

    cap.release()
    sock.close()
    if DISPLAY:
        cv2.destroyAllWindows()
    print(f"\nPipeline complete: {frame_num} frames | {total_dets} total detections")

if __name__ == "__main__":
    main()
