#!/usr/bin/env bash
# =============================================================================
# build_parser.sh  --  Build DeepStream-Yolo custom YOLOv8 bbox parser
# =============================================================================
#
# Run ONCE on the Jetson inside the DeepStream Docker container.
# The compiled .so is placed in deepstream/configs/ and loaded by nvinfer.
#
# Usage (on Jetson):
#   sudo docker run --runtime nvidia --rm \
#     -v /home/tej/capstone:/capstone \
#     dustynv/deepstream:7.0-r36.4.0 \
#     bash /capstone/deepstream/build_parser.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/configs"
BUILD_DIR="/tmp/DeepStream-Yolo"

echo "=============================================="
echo " Building DeepStream-Yolo custom YOLOv8 parser"
echo "=============================================="

# ── detect CUDA version ───────────────────────────────────────────────────────
CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
echo "CUDA version   : $CUDA_VER"
echo "Output dir     : $OUT_DIR"

# ── clone DeepStream-Yolo ─────────────────────────────────────────────────────
if [ -d "$BUILD_DIR" ]; then
    echo "Updating existing DeepStream-Yolo clone..."
    git -C "$BUILD_DIR" pull --quiet
else
    echo "Cloning DeepStream-Yolo..."
    git clone --depth 1 \
        https://github.com/marcoslucianops/DeepStream-Yolo.git \
        "$BUILD_DIR"
fi

# ── build ─────────────────────────────────────────────────────────────────────
echo "Building parser library (CUDA_VER=$CUDA_VER)..."
cd "$BUILD_DIR"
CUDA_VER=$CUDA_VER make -C nvdsinfer_custom_impl_Yolo -j"$(nproc)"

# ── copy .so to project ───────────────────────────────────────────────────────
cp "$BUILD_DIR/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" \
   "$OUT_DIR/"

echo ""
echo "Done!  Parser library:"
echo "  $OUT_DIR/libnvdsinfer_custom_impl_Yolo.so"
echo ""
echo "Next: run ds_pipeline.py"
