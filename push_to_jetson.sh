#!/usr/bin/env bash
# =============================================================================
# push_to_jetson.sh  --  Auto-sync project files to Jetson on every save
# =============================================================================
# Usage:
#   bash push_to_jetson.sh          # watch mode (auto-push on file change)
#   bash push_to_jetson.sh once     # push once and exit
# =============================================================================

JETSON_USER="tej"
JETSON_IP="10.0.0.119"
JETSON_PATH="/home/tej/capstone"
LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)"

RSYNC_CMD="rsync -avz --exclude='.git/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='training_data/' --exclude='runs/' \
    --include='deepstream/***' \
    --include='jetson_run.txt' \
    --exclude='*' \
    \"$LOCAL_PATH/\" \
    \"$JETSON_USER@$JETSON_IP:$JETSON_PATH/\""

_push() {
    echo "[$(date '+%H:%M:%S')] Pushing to Jetson..."
    eval $RSYNC_CMD && echo "[$(date '+%H:%M:%S')] Done." || echo "[ERROR] rsync failed."
}

# ── one-shot mode ─────────────────────────────────────────────────────────────
if [ "$1" = "once" ]; then
    _push
    exit 0
fi

# ── watch mode ────────────────────────────────────────────────────────────────
if command -v fswatch &>/dev/null; then
    echo "Watching for changes in: $LOCAL_PATH/deepstream/"
    echo "Auto-pushing to: $JETSON_USER@$JETSON_IP:$JETSON_PATH"
    echo "Press Ctrl+C to stop."
    echo ""
    _push  # initial push
    fswatch -o "$LOCAL_PATH/deepstream/" | while read; do
        _push
    done
else
    echo "[INFO] fswatch not found — install it for auto-watch mode:"
    echo "       brew install fswatch"
    echo ""
    echo "Doing a one-time push instead..."
    _push
fi
