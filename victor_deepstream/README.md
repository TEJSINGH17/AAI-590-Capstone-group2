# OmniView AI — DeepStream Pipeline

**Role:** DeepStream Pipeline Engineer  
**Developer:** Victor  

## Overview
This module handles real-time traffic sign detection using NVIDIA DeepStream 7.0 and YOLOv8, fine-tuned on the LISA Traffic Sign Dataset.

## Pipeline Architecture
## Tech Stack
- DeepStream 7.0 (Docker on PC, Native on Jetson)
- YOLOv8n fine-tuned on LISA Traffic Sign Dataset
- TensorRT FP16 (~2100 FPS on RTX 4080 Super)
- Python 3.10 / GStreamer / pyds 1.1.11
- UDP JSON output compatible with publish_to_hud.py

## Model
- Architecture: YOLOv8n
- Dataset: LISA Traffic Sign Dataset
- Classes: go, goForward, goLeft, stop, stopLeft, warning, warningLeft
- Input resolution: 640x640
- Precision: FP16 TensorRT

## Detection Output Format (UDP JSON)
```json
{
  "schema_version": 1,
  "timestamp_ms": 1234567890000,
  "sequence": 42,
  "detections": [
    {
      "class_id": 0,
      "confidence": 0.87,
      "x_center": 0.52,
      "y_center": 0.43,
      "width": 0.15,
      "height": 0.12,
      "source_id": 0,
      "frame_num": 42,
      "label": "stop"
    }
  ]
}
```

## Files
- `omniview_pipeline.py` — main pipeline script (video input + inference + UDP output)
- `pipeline.py` — DeepStream GStreamer pipeline (development version)
- `config_infer_primary.txt` — DeepStream nvinfer configuration
- `labels.txt` — LISA traffic sign class names
- `yolov8n_lisa_best.onnx` — exported ONNX model for TensorRT conversion
- `setup_jetson.sh` — automated Jetson deployment script

## Running the Pipeline
```bash
python3 omniview_pipeline.py
```

## Performance
- Throughput: ~2100 FPS (TensorRT FP16 on RTX 4080 Super)
- Latency: ~0.76ms average
- UDP output: per-frame JSON to port 5055
