# DeepStream Pipeline — Jetson Setup & Usage

Real-time dual-model object detection pipeline using NVIDIA DeepStream 7.x.
Detects vehicles/pedestrians (COCO) and traffic signs (LISA) simultaneously,
with optional AR HUD output over UDP and RTSP streaming to iPad/VLC.

---

## Prerequisites

- NVIDIA Jetson (Orin / AGX / NX) running **JetPack 6.x**
- Docker installed with NVIDIA runtime (`nvidia-container-toolkit`)
- USB camera connected at `/dev/video0` (or CSI camera)
- TensorRT engine files: `yolov8n.engine` and `yolov8n_lisa_v1.1.engine`

---

## Step 1 — Copy Project to Jetson

From your Mac/PC, `scp` the project folder to the Jetson:

```bash
scp -r /path/to/AAI-590-Capstone-group2-1 tej@<jetson-ip>:/home/tej/capstone
```

Or clone directly on the Jetson:

```bash
git clone <repo-url> /home/tej/capstone
```

---

## Step 2 — Copy Model Files to Jetson

Place your TensorRT engine files and ONNX files in the `models/` folder on the Jetson:

```
/home/tej/capstone/
├── models/
│   ├── yolov8n.onnx
│   ├── yolov8n.engine               ← COCO model
│   ├── yolov8n_lisa_v1.1.onnx
│   └── yolov8n_lisa_v1.1.engine     ← LISA traffic sign model
```

> **Note:** TensorRT `.engine` files are device-specific and must be generated
> on the Jetson itself — they cannot be copied from a Mac/PC.
> ONNX files (`.onnx`) can be copied from Mac/PC and are used as the source for engine export.

To export on the Jetson (inside Docker):

```bash
yolo export model=yolov8n.pt format=engine device=0
```

---

## Step 3 — Build the Custom YOLOv8 Parser (Run Once)

The pipeline requires a custom DeepStream-Yolo parser `.so` library.
Build it once inside the DeepStream Docker container:

```bash
sudo docker run --runtime nvidia --rm \
  -v /home/tej/capstone:/capstone \
  dustynv/deepstream:7.0-r36.4.0 \
  bash /capstone/deepstream/build_parser.sh
```

This clones [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo),
compiles it, and places `libnvdsinfer_custom_impl_Yolo.so` in `deepstream/configs/`.

---

## Step 4 — Run the Pipeline

Start the DeepStream Docker container with GPU and camera access:

```bash
sudo docker run --runtime nvidia -it --rm \
  --privileged \
  -v /home/tej/capstone:/capstone \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  --device /dev/video0 \
  dustynv/deepstream:7.0-r36.4.0 \
  bash
```

> **Note:** Remove `--device /dev/video0` if using a CSI camera or RTSP stream.

Then inside the container, run the pipeline:

### USB Camera → Monitor Display

```bash
python3 /capstone/deepstream/ds_pipeline.py \
  --source /dev/video0 \
  --coco-engine /capstone/models/yolov8n.engine \
  --lisa-engine /capstone/models/yolov8n_lisa_v1.1.engine \
  --output display
```

### USB Camera → RTSP Stream (iPad / VLC)

```bash
python3 /capstone/deepstream/ds_pipeline.py \
  --source /dev/video0 \
  --coco-engine /capstone/models/yolov8n.engine \
  --lisa-engine /capstone/models/yolov8n_lisa_v1.1.engine \
  --output rtsp \
  --rtsp-port 8554
```

Open on iPad VLC: `rtsp://<jetson-ip>:8554/ds-test`

### USB Camera + UDP AR HUD

```bash
python3 /capstone/deepstream/ds_pipeline.py \
  --source /dev/video0 \
  --coco-engine /capstone/models/yolov8n.engine \
  --lisa-engine /capstone/models/yolov8n_lisa_v1.1.engine \
  --output display \
  --udp-host 192.168.4.50 \
  --udp-port 5055
```

### CSI Camera (Raspberry Pi Camera V2)

```bash
python3 /capstone/deepstream/ds_pipeline.py \
  --source csi \
  --width 1920 --height 1080 \
  --coco-engine /capstone/models/yolov8n.engine \
  --lisa-engine /capstone/models/yolov8n_lisa_v1.1.engine \
  --output display
```

### Local MP4 File (Testing)

```bash
python3 /capstone/deepstream/ds_pipeline.py \
  --source /capstone/test_data/video_test1.mp4 \
  --coco-engine /capstone/models/yolov8n.engine \
  --lisa-engine /capstone/models/yolov8n_lisa_v1.1.engine \
  --output display
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--source` | *(required)* | `csi`, `csi:1`, `0`, `/dev/video0`, `rtsp://...`, or `file.mp4` |
| `--coco-engine` | `models/yolov8n.engine` | Path to COCO TensorRT engine |
| `--lisa-engine` | `models/yolov8n_lisa_v1.1.engine` | Path to LISA TensorRT engine |
| `--width` | `1920` | Input frame width |
| `--height` | `1080` | Input frame height |
| `--output` | `display` | `display` (HDMI monitor) or `rtsp` (stream) |
| `--rtsp-port` | `8554` | RTSP port (when `--output rtsp`) |
| `--udp-host` | *(empty)* | AR HUD UDP target IP. Leave empty to disable. |
| `--udp-port` | `5055` | AR HUD UDP port |

---

## File Structure

```
deepstream/
├── ds_pipeline.py              ← Main pipeline script
├── pipeline.py                 ← Lightweight pipeline variant
├── dashboard.py                ← Detection dashboard
├── build_parser.sh             ← One-time parser build script
├── README.md                   ← This file
├── __init__.py
├── configs/
│   ├── pgie_coco.txt           ← nvinfer config for COCO model
│   ├── pgie_lisa.txt           ← nvinfer config for LISA model
│   ├── tracker_nvdcf.txt       ← NvDCF tracker config
│   └── libnvdsinfer_custom_impl_Yolo.so   ← Built by build_parser.sh
└── labels/
    ├── coco.txt                ← 80 COCO class names
    └── lisa.txt                ← LISA traffic sign class names
```

---

## UDP Output Format (AR HUD)

When `--udp-host` is set, each frame emits a JSON payload over UDP:

```json
{
  "schema_version": 1,
  "timestamp_ms": 1712900000000,
  "sequence": 42,
  "detections": [
    {
      "class_id": 2,
      "label": "car",
      "confidence": 0.87,
      "x_center": 0.52,
      "y_center": 0.61,
      "width": 0.18,
      "height": 0.14,
      "tracker_id": 5,
      "source_id": 1,
      "frame_num": 120
    }
  ]
}
```

`source_id=1` = COCO model, `source_id=2` = LISA model. All coordinates are normalized `[0, 1]`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Cannot create GStreamer element 'nvinfer'` | Run inside the DeepStream Docker container |
| `pyds not found` | Ensure you are using `dustynv/deepstream:7.0-r36.4.0` image |
| Engine file not found | Engine must be built on the Jetson — see Step 2 |
| `libnvdsinfer_custom_impl_Yolo.so` missing | Run `build_parser.sh` — see Step 3 |
| Black screen / no display | Pass `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` to Docker |
