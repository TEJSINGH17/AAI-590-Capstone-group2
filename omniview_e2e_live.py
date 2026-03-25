"""
omniview_e2e_live.py -- OmniView AI End-to-End Live HUD
Version: 3.0 (clean, tested, working)
========================================================
AAI-590 Capstone Group 2 | University of San Diego

Self-contained pipeline. Run with:
    !python3 /content/omniview_e2e_live.py

Modes:
    file  -- reads per-frame JSON from runs/hud/ds_1/  (default)
    udp   -- receives live JSON from Jetson on port 5055

What it does:
    1. git clone repo (skips if already present)
    2. pip install dependencies
    3. Import HUD helpers from omniview_pipeline.py (Victor)
    4. Read per-frame JSON detections
    5. Overlay HUD on video frames
    6. Save annotated MP4 + GIF
    7. Display GIF inline in Colab

Usage:
    !python3 /content/omniview_e2e_live.py
    !python3 /content/omniview_e2e_live.py --no-bootstrap
    !python3 /content/omniview_e2e_live.py --mode udp
    !python3 /content/omniview_e2e_live.py --video /content/my_video.mp4
    !python3 /content/omniview_e2e_live.py --json /content/AAI-590-Capstone-group2/runs/hud/ds_1
"""

from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, threading, time
from glob import glob
from pathlib import Path
from typing import List, Optional

# =============================================================================
# CLI -- parse before anything else
# parse_known_args ignores Colab's -f kernel.json argument
# =============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="OmniView AI -- End-to-End Live HUD")
    _c = os.path.isdir("/content")  # True in Colab
    p.add_argument("--repo",       default="/content/AAI-590-Capstone-group2" if _c else "AAI-590-Capstone-group2")
    p.add_argument("--mode",       choices=["file","udp"], default="file")
    p.add_argument("--json",       default="")
    p.add_argument("--video",      default="")
    p.add_argument("--output",     default="/content/omniview_live_hud.mp4" if _c else "omniview_live_hud.mp4")
    p.add_argument("--gif-frames", type=int,   default=40)
    p.add_argument("--gif-fps",    type=int,   default=10)
    p.add_argument("--udp-host",   default="127.0.0.1")
    p.add_argument("--udp-port",   type=int,   default=5055)
    p.add_argument("--udp-wait",   type=float, default=10.0)
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--max-frames", type=int,   default=0)
    p.add_argument("--no-bootstrap", action="store_true")
    args, _ = p.parse_known_args()
    return args

args = _parse_args()
REPO = os.path.abspath(args.repo)

# =============================================================================
# STEP 1 -- BOOTSTRAP
# =============================================================================
def _run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True)

if not args.no_bootstrap:
    print("\n" + "="*60)
    print("  STEP 1 -- Bootstrap")
    print("="*60)
    if not os.path.isdir(REPO):
        print(f"\n[Bootstrap] Cloning repo -> {REPO}")
        _run(f"git clone https://github.com/TEJSINGH17/AAI-590-Capstone-group2.git {REPO}")
    else:
        print(f"[Bootstrap] Repo present: {REPO} -- pulling latest")
        _run(f"git -C {REPO} pull --quiet")
    print("\n[Bootstrap] Installing dependencies...")
    _run("pip install ultralytics opencv-python-headless imageio imageio-ffmpeg --quiet")
    print("\n[Bootstrap] Done.\n")

# Add repo to sys.path
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# =============================================================================
# STEP 2 -- IMPORTS (after bootstrap guarantees packages)
# =============================================================================
import cv2
import numpy as np
try:
    import imageio.v2 as imageio
    _IMAGEIO = True
except ImportError:
    _IMAGEIO = False

try:
    from IPython.display import Image as IPImage, display as ip_display
    _IPYTHON = True
except ImportError:
    _IPYTHON = False

# =============================================================================
# STEP 3 -- RESOLVE PATHS
# =============================================================================
# Video -- support multiple videos matched to JSON dirs
# If --video not specified, auto-match based on JSON dir names
_video_candidates = [
    os.path.join(REPO, "test_data", "output_ds_1.mp4"),
    os.path.join(REPO, "test_data", "output_ds_2.mp4"),
    "/content/output_ds_1.mp4",
    "/content/drive/MyDrive/output_ds_1.mp4",
]
VIDEO_PATH = args.video or next(
    (p for p in _video_candidates if os.path.exists(p)), None)

# Build a map of ds_N -> video path for multi-video support
_DS_VIDEO_MAP = {}
for _n in ["1", "2", "3"]:
    for _vdir in [os.path.join(REPO, "test_data"), "/content"]:
        _vp = os.path.join(_vdir, f"output_ds_{_n}.mp4")
        if os.path.exists(_vp):
            _DS_VIDEO_MAP[f"ds_{_n}"] = _vp
print(f"  Found videos: {list(_DS_VIDEO_MAP.values())}")

# JSON dir -- runs/hud/ds_1 is confirmed in repo
JSON_DIR = args.json or next((p for p in [
    os.path.join(REPO, "runs", "hud", "ds_1"),
    os.path.join(REPO, "victor_deepstream", "runs", "hud", "ds_3"),
] if os.path.isdir(p)), os.path.join(REPO, "runs", "hud", "ds_1"))

OUTPUT_PATH = args.output
GIF_PATH    = str(Path(OUTPUT_PATH).with_suffix(".gif"))

print("="*60)
print("  OmniView AI -- End-to-End Live HUD  v3.0")
print("  AAI-590 Capstone Group 2 | USD")
print("="*60)
print(f"  Mode   : {args.mode.upper()}")
print(f"  Video  : {VIDEO_PATH or 'NOT FOUND'}")
print(f"  JSON   : {JSON_DIR}")
print(f"  Output : {OUTPUT_PATH}")
print("="*60 + "\n")

# =============================================================================
# STEP 4 -- IMPORT FROM omniview_pipeline.py (Victor)
# These are the confirmed-working functions used in the tested inline cell
# =============================================================================
print("="*60)
print("  STEP 4 -- Loading pipeline helpers")
print("="*60)

_pipeline_loaded = False
for _try_path in [
    os.path.join(REPO, "victor_deepstream"),
    os.path.join(REPO, "application"),
]:
    if _try_path not in sys.path:
        sys.path.insert(0, _try_path)

try:
    from omniview_pipeline import (
        get_blind_spot_zones,
        draw_blind_spot_zones,
        get_box_color,
        draw_alerts,
    )
    _pipeline_loaded = True
    print("  get_blind_spot_zones   OK")
    print("  draw_blind_spot_zones  OK")
    print("  get_box_color          OK")
    print("  draw_alerts            OK")
except Exception as e:
    print(f"  [WARN] omniview_pipeline import failed: {e}")
    print("  Using built-in fallbacks")

if not _pipeline_loaded:
    def get_blind_spot_zones(w, h):
        return ((0, int(h*0.48), int(w*0.22), h),
                (int(w*0.78), int(h*0.48), w, h))

    def draw_blind_spot_zones(frame, lz, rz):
        for z in (lz, rz):
            ov = frame.copy()
            cv2.rectangle(ov, (z[0],z[1]), (z[2],z[3]), (0,255,255), -1)
            cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
            cv2.rectangle(frame, (z[0],z[1]), (z[2],z[3]), (0,255,255), 2)
        h, w = frame.shape[:2]
        cv2.putText(frame,"LEFT BLIND SPOT", (10,lz[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        cv2.putText(frame,"RIGHT BLIND SPOT",(rz[0],rz[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

    def get_box_color(conf, in_blind):
        if not in_blind: return (0,255,0)
        return (0,0,255) if conf >= 0.5 else (0,255,255)

    def draw_alerts(frame, left, right):
        if left:
            cv2.rectangle(frame,(20,75),(460,115),(0,0,200),-1)
            cv2.putText(frame,"ALERT: LEFT BLIND SPOT",(35,103),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if right:
            cv2.rectangle(frame,(20,125),(470,165),(0,0,200),-1)
            cv2.putText(frame,"ALERT: RIGHT BLIND SPOT",(35,153),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

# INSTRUCTION_MAP -- from ar_glasses_hud.py (Sunitha)
INSTRUCTION_MAP = {
    "go":          "GO",
    "goForward":   "GO FORWARD",
    "goLeft":      "GO LEFT",
    "stop":        "STOP",
    "stopLeft":    "STOP LEFT",
    "warning":     "WARNING",
    "warningLeft": "WARNING LEFT",
}

# LISA labels set for model source detection
_LISA_LABELS = {
    "go","goforward","goleft","stop","stopleft","warning","warningleft"
}

print()

# =============================================================================
# STEP 5 -- UDP LISTENER (live mode only)
# =============================================================================
_latest_payload = {"sequence": -1, "detections": []}
_payload_lock   = threading.Lock()
_stop_udp       = threading.Event()

def _udp_listener(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(1.0)
    print(f"[UDP] Listening on {host}:{port}")
    while not _stop_udp.is_set():
        try:
            data, _ = sock.recvfrom(65535)
            parsed = json.loads(data.decode())
            with _payload_lock:
                _latest_payload.update(parsed)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] Error: {e}")
            break
    sock.close()

# =============================================================================
# STEP 6 -- MAIN PIPELINE
# Exactly the logic that worked in the tested inline cell
# =============================================================================
def run():
    print("="*60)
    print("  STEP 6 -- Running Pipeline")
    print("="*60)

    # -- Video ----------------------------------------------------------------
    # Build per-frame video map: each frame knows which video file to use
    # This lets ds_1 frames use output_ds_1.mp4 and ds_2 frames use output_ds_2.mp4
    def _get_video_for_json(json_path):
        """Return the matching video path for a JSON file based on ds_N in its path."""
        for ds_key, vpath in _DS_VIDEO_MAP.items():
            if ds_key in str(json_path):
                return vpath
        return VIDEO_PATH  # fallback to default

    if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
        print("\n[Pipeline] No video found -- rendering on black canvas")
        _canvas_only = True
        w, h, fps = 1920, 1080, 30.0
        cap = None
        _current_video = None
    else:
        _canvas_only   = False
        _current_video = VIDEO_PATH
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Pipeline] Video  : {VIDEO_PATH}  {w}x{h} @ {fps:.1f}fps")

    # -- JSON source ----------------------------------------------------------
    if args.mode == "file":
        # Support multiple JSON dirs: --json dir1,dir2
        # Default: try ds_1 then ds_2 to get more frames
        _json_dirs = [d.strip() for d in JSON_DIR.split(",") if d.strip()]

        # Auto-add ds_2 if it exists and only ds_1 was specified
        if len(_json_dirs) == 1 and _json_dirs[0].endswith("ds_1"):
            _ds2 = _json_dirs[0].replace("ds_1", "ds_2")
            if os.path.isdir(_ds2):
                _json_dirs.append(_ds2)
                print(f"[Pipeline] Auto-added ds_2: {_ds2}")

        json_files = []
        for _d in _json_dirs:
            _f = sorted(glob(os.path.join(_d, "frame_*.json")))
            json_files.extend(_f)
            print(f"[Pipeline] JSON   : {len(_f)} frames from {_d}")

        if not json_files:
            raise FileNotFoundError(f"No JSON files in {JSON_DIR}")

        max_frames = args.max_frames or len(json_files)
        json_files = json_files[:max_frames]
        print(f"[Pipeline] Total  : {len(json_files)} frames")

        def get_payload(i):
            if i >= len(json_files):
                return None
            with open(json_files[i]) as f:
                return json.load(f)

    else:  # UDP mode
        threading.Thread(
            target=_udp_listener,
            args=(args.udp_host, args.udp_port),
            daemon=True).start()
        print(f"[Pipeline] UDP    : waiting {args.udp_wait}s for first packet...")
        time.sleep(args.udp_wait)
        max_frames = args.max_frames or 99999

        def get_payload(i):
            with _payload_lock:
                return _latest_payload.copy()

    # -- Output ---------------------------------------------------------------
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"[Pipeline] Output : {out_path}\n")

    # -- Zones ----------------------------------------------------------------
    left_zone, right_zone = get_blind_spot_zones(w, h)

    # -- Main loop ------------------------------------------------------------
    total = 0
    t0    = time.time()

    try:
        for i in range(max_frames):
            # Get frame -- switch video when JSON dir changes (ds_1 -> ds_2 etc.)
            if _canvas_only:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                # Check if this frame needs a different video
                _needed_video = _get_video_for_json(json_files[i])
                if _needed_video and _needed_video != _current_video:
                    print(f"\n[Pipeline] Switching video -> {_needed_video}")
                    if cap: cap.release()
                    cap = cv2.VideoCapture(_needed_video)
                    _current_video = _needed_video

                ok, frame = cap.read()
                if not ok:
                    # End of this video -- loop back
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break

            # Get detections
            payload = get_payload(i)
            if payload is None:
                break
            dets = payload.get("detections", [])

            left_alert  = False
            right_alert = False
            instruction = None
            instr_conf  = 0.0

            # -- Top bar ------------------------------------------------------
            cv2.rectangle(frame, (0,0), (w,60), (20,20,20), -1)
            cv2.putText(frame, "OmniView AI | Live HUD", (20,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            seq = payload.get("sequence", i)
            cv2.putText(frame, f"seq:{seq}", (w-150,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)

            # -- Blind spot overlays ------------------------------------------
            draw_blind_spot_zones(frame, left_zone, right_zone)

            # -- Detections ---------------------------------------------------
            for det in dets:
                conf  = float(det.get("confidence", det.get("conf", 0.0)))
                label = str(det.get("label", "object")).strip()

                # Normalise model source -- handles all pipeline formats:
                #   Victor: det["model"] = "coco"|"lisa"
                #   Tej detect_to_json: source_id=0, no model field
                #   Tej ds_pipeline: source_id=1 (COCO) or 2 (LISA)
                _raw_model = str(det.get("model", "")).strip().lower()
                _src_id    = det.get("source_id", 0)
                if _raw_model in ("lisa", "coco"):
                    model_src = _raw_model
                elif label.lower().replace(" ", "") in _LISA_LABELS:
                    model_src = "lisa"
                elif _src_id == 2:
                    model_src = "lisa"
                else:
                    model_src = "coco"

                # Confidence filter
                _thresh = 0.15 if model_src == "lisa" else args.conf
                if conf < _thresh:
                    continue

                # LISA -- capture instruction
                lk = label.replace(" ", "")
                if model_src == "lisa" and lk in INSTRUCTION_MAP:
                    if conf > instr_conf:
                        instr_conf  = conf
                        instruction = INSTRUCTION_MAP[lk]

                # Pixel box
                xc = float(det.get("x_center", det.get("x1", 0) / w)) * w \
                    if "x_center" in det else (det["x1"] + det["x2"]) / 2
                yc = float(det.get("y_center", det.get("y1", 0) / h)) * h \
                    if "y_center" in det else (det["y1"] + det["y2"]) / 2
                bw = float(det.get("width",  (det.get("x2",0)-det.get("x1",0)) / w)) * w \
                    if "width" in det else det["x2"] - det["x1"]
                bh = float(det.get("height", (det.get("y2",0)-det.get("y1",0)) / h)) * h \
                    if "height" in det else det["y2"] - det["y1"]

                x1 = max(0, min(w-1, int(xc - bw/2)))
                y1 = max(0, min(h-1, int(yc - bh/2)))
                x2 = max(0, min(w-1, int(xc + bw/2)))
                y2 = max(0, min(h-1, int(yc + bh/2)))
                if x2 <= x1 or y2 <= y1:
                    continue

                bs = str(det.get("blind_spot", "none")).strip().lower()
                # Fallback zone check if blind_spot not set
                if bs == "none" and model_src == "coco":
                    cx = (x1+x2)//2; cy = (y1+y2)//2
                    lz, rz = left_zone, right_zone
                    if lz[0]<=cx<=lz[2] and lz[1]<=cy<=lz[3]:
                        bs = "left"
                    elif rz[0]<=cx<=rz[2] and rz[1]<=cy<=rz[3]:
                        bs = "right"

                color = get_box_color(conf, bs in ("left","right"))
                if bs == "left":  left_alert  = True
                if bs == "right": right_alert = True

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.circle(frame, ((x1+x2)//2,(y1+y2)//2), 4, color, -1)

                # Label
                txt = f"{label[:10]} {conf:.2f}"
                if bs != "none": txt += f" | {bs.upper()}"
                (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = y1-5 if y1 > th+10 else y1+th+5
                cv2.rectangle(frame, (x1, ty-th-4), (x1+tw+6, ty+2), color, -1)
                cv2.putText(frame, txt, (x1+3, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                total += 1

            # -- Traffic sign banner ------------------------------------------
            if instruction:
                cv2.rectangle(frame, (w-340,10), (w-20,50), (255,140,0), -1)
                cv2.putText(frame, f"SIGN: {instruction}", (w-325,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)

            # -- Alert banners ------------------------------------------------
            draw_alerts(frame, left_alert, right_alert)

            # -- Footer -------------------------------------------------------
            cv2.putText(frame, f"Frame:{i}  Dets:{len(dets)}",
                        (20, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

            writer.write(frame)

            if i % 100 == 0:
                elapsed = max(time.time()-t0, 1e-6)
                print(f"frame={i}/{max_frames}  dets={len(dets)}"
                      f"  total={total}  render={i/elapsed:.1f}fps"
                      f"  playback=30fps")

    except KeyboardInterrupt:
        print("\n[Pipeline] Stopped by user.")

    finally:
        elapsed = max(time.time()-t0, 1e-6)
        print(f"\n[Pipeline] Done. {total} detections in {i+1} frames")
        print(f"[Pipeline] Render speed : {(i+1)/elapsed:.1f} fps (CPU rendering)")
        print(f"[Pipeline] Playback speed: 30.0 fps (video plays at normal speed)")
        if cap is not None:
            cap.release()
            cap = None
        writer.release()
        _stop_udp.set()
        print(f"[Pipeline] MP4 saved: {out_path} "
              f"({os.path.getsize(out_path)//1024} KB)")

    # -- GIF ------------------------------------------------------------------
    if args.gif_frames > 0 and _IMAGEIO:
        print("\n[Pipeline] Creating GIF...")
        gif_buf = []
        reader  = imageio.get_reader(str(out_path))
        for k, f in enumerate(reader):
            if k >= args.gif_frames: break
            gif_buf.append(f)
        reader.close()
        imageio.mimsave(GIF_PATH, gif_buf, fps=args.gif_fps)
        print(f"[Pipeline] GIF saved: {GIF_PATH} "
              f"({len(gif_buf)/args.gif_fps:.1f}s)")
        if _IPYTHON:
            ip_display(IPImage(filename=GIF_PATH))


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run()
