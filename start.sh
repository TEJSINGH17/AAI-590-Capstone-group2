#!/bin/bash
# ============================================================
# start.sh — DeepStream pipeline launcher
# ============================================================
# Auto-detects EMEET C950 USB camera (/dev/video0).
# Falls back to MP4 if camera not found.
#
# Usage:
#   ./start.sh              # auto-detect
#   ./start.sh camera       # force USB camera
#   ./start.sh mp4          # force MP4 test file
# ============================================================

CAPSTONE=/home/tej/capstone
COCO_ENGINE=$CAPSTONE/models/yolov8n_deepstream.engine
LISA_ENGINE=$CAPSTONE/models/yolov8n_lisa_v1.1_deepstream.engine
TEST_MP4=$CAPSTONE/test_data/video_test1.mp4
USB_DEVICE=/dev/video0
OUTPUT_MODE=display    # display (monitor) or rtsp (iPad)

cd "$CAPSTONE" || { echo "[ERROR] $CAPSTONE not found"; exit 1; }

# Verify engines exist
if [ ! -f "$COCO_ENGINE" ] || [ ! -f "$LISA_ENGINE" ]; then
    echo "[ERROR] TensorRT engines not found. Run trtexec first (see jetson_run.txt)."
    exit 1
fi

# Determine source
MODE="${1:-auto}"

WIDTH=1920
HEIGHT=1080

if [ "$MODE" = "camera" ]; then
    if [ -e "$USB_DEVICE" ]; then
        SOURCE=$USB_DEVICE
    else
        echo "[WARN] Camera not found at $USB_DEVICE — launching dashboard with no-source message."
        SOURCE=none
    fi
elif [ "$MODE" = "mp4" ]; then
    if [ -f "$TEST_MP4" ]; then
        SOURCE=$TEST_MP4
    else
        echo "[WARN] MP4 not found at $TEST_MP4 — launching dashboard with no-source message."
        SOURCE=none
    fi
else
    # Auto-detect
    if [ -e "$USB_DEVICE" ]; then
        echo "[INFO] Camera detected at $USB_DEVICE — using live feed."
        SOURCE=$USB_DEVICE
    elif [ -f "$TEST_MP4" ]; then
        echo "[INFO] No camera found — using MP4 test file."
        SOURCE=$TEST_MP4
    else
        echo "[WARN] No camera and no MP4 found — launching dashboard with no-source message."
        SOURCE=none
    fi
fi

echo "[INFO] Source  : $SOURCE"
echo "[INFO] Engines : $COCO_ENGINE"
echo "[INFO] Mode    : ${OUTPUT_MODE:-display}"
echo ""

if [ "${OUTPUT_MODE:-display}" = "rtsp" ]; then
    exec python3 "$CAPSTONE/deepstream/ds_pipeline.py" \
        --source      "$SOURCE" \
        --width       "$WIDTH" \
        --height      "$HEIGHT" \
        --coco-engine "$COCO_ENGINE" \
        --lisa-engine "$LISA_ENGINE" \
        --output      rtsp
else
    exec python3 "$CAPSTONE/deepstream/dashboard.py" \
        --source      "$SOURCE" \
        --width       "$WIDTH" \
        --height      "$HEIGHT" \
        --coco-engine "$COCO_ENGINE" \
        --lisa-engine "$LISA_ENGINE"
fi
