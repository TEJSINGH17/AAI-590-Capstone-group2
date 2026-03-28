## /content/AAI-590-Capstone-group2/omniview_e2e_live.py
from __future__ import annotations
import argparse, json, os, sys, time
from glob import glob
from pathlib import Path

def _parse_args():
    p = argparse.ArgumentParser()
    _c = os.path.isdir("/content")
    p.add_argument("--repo",         default="/content/AAI-590-Capstone-group2" if _c else "AAI-590-Capstone-group2")
    p.add_argument("--mode",         choices=["file","udp","jetson"], default="file")
    p.add_argument("--json",         default="")
    p.add_argument("--video",        default="")
    p.add_argument("--output",       default="/content/omniview_live_hud.mp4")
    p.add_argument("--gif-frames",   type=int,   default=40)
    p.add_argument("--gif-fps",      type=int,   default=10)
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--max-frames",   type=int,   default=0)
    p.add_argument("--udp-host",     default="127.0.0.1")
    p.add_argument("--udp-port",     type=int,   default=5055)
    p.add_argument("--udp-wait",     type=float, default=10.0)
    p.add_argument("--jetson-ip",    default="10.0.0.119")
    p.add_argument("--no-bootstrap", action="store_true")
    args, _ = p.parse_known_args()
    return args

args = _parse_args()
REPO = os.path.abspath(args.repo)

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
# IMPORTS FROM TEAM FILES
# =============================================================================

# Add all module paths
for _p in [
    os.path.join(REPO, "victor_deepstream"),
    os.path.join(REPO, "application"),
    os.path.join(REPO, "deepstream"),
    REPO,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

print("="*60)
print("  OmniView AI -- End-to-End Live HUD  v4.0")
print("  AAI-590 Capstone Group 2 | USD")
print("="*60)

# ── 1. Victor's omniview_pipeline.py ─────────────────────────────────────────
_victor_loaded = False
try:
    from omniview_pipeline import (
        get_blind_spot_zones,
        draw_blind_spot_zones,
        get_box_color,
        draw_alerts,
        DetectionPayload,
        build_message,
        BLIND_SPOT_CLASSES,
    )
    _victor_loaded = True
    print("  Victor's omniview_pipeline.py loaded")
    print("     get_blind_spot_zones, draw_blind_spot_zones")
    print("     get_box_color, draw_alerts")
    print("     DetectionPayload, build_message")
except Exception as e:
    print(f"  Victor's omniview_pipeline.py failed: {e}")
    print("     Using built-in fallbacks")
    BLIND_SPOT_CLASSES = {"person","bicycle","car","motorcycle","bus","truck"}

    def get_blind_spot_zones(w, h):
        return ((0, int(h*0.48), int(w*0.22), h),
                (int(w*0.78), int(h*0.48), w, h))

    def draw_blind_spot_zones(frame, lz, rz):
        for z in (lz, rz):
            ov = frame.copy()
            cv2.rectangle(ov,(z[0],z[1]),(z[2],z[3]),(0,255,255),-1)
            cv2.addWeighted(ov,0.18,frame,0.82,0,frame)
            cv2.rectangle(frame,(z[0],z[1]),(z[2],z[3]),(0,255,255),2)
        h,w = frame.shape[:2]
        cv2.putText(frame,"LEFT BLIND SPOT",(10,lz[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        cv2.putText(frame,"RIGHT BLIND SPOT",(rz[0],rz[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

    def get_box_color(conf, in_blind):
        if not in_blind: return (0,255,0)
        return (0,0,255) if conf>=0.5 else (0,255,255)

    def draw_alerts(frame, left, right):
        if left:
            cv2.rectangle(frame,(20,75),(460,115),(0,0,200),-1)
            cv2.putText(frame,"ALERT: LEFT BLIND SPOT",(35,103),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if right:
            cv2.rectangle(frame,(20,125),(470,165),(0,0,200),-1)
            cv2.putText(frame,"ALERT: RIGHT BLIND SPOT",(35,153),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

# ── 2. Sunitha's ar_glasses_hud.py ───────────────────────────────────────────
_sunitha_loaded = False
try:
    # ar_glasses_hud.py is a Colab notebook script -- extract what we need
    # VelocityTracker and INSTRUCTION_MAP are defined inline here
    # since ar_glasses_hud.py is not importable as a module
    raise ImportError("ar_glasses_hud.py is a script not a module")
except Exception:
    pass

# VelocityTracker -- from Sunitha's ar_glasses_hud.py
class VelocityTracker:
    """
    Motion tracker from ar_glasses_hud.py (Sunitha).
    Estimates object movement between frames using center
    position and bounding-box area.
    """
    def __init__(self):
        self.history = {}

    def update(self, obj_key, cx, cy, area, frame_idx):
        prev = self.history.get(obj_key)
        velocity = None
        if prev is not None:
            frames_elapsed = max(frame_idx - prev["frame"], 1)
            dx = (cx - prev["cx"]) / frames_elapsed
            dy = (cy - prev["cy"]) / frames_elapsed
            speed_px = np.sqrt(dx**2 + dy**2)
            area_change = (area - prev["area"]) / max(prev["area"], 1)
            velocity = {
                "dx": dx, "dy": dy,
                "speed_px": speed_px,
                "area_change": area_change,
            }
        self.history[obj_key] = {
            "cx": cx, "cy": cy,
            "area": area, "frame": frame_idx,
        }
        return velocity

    def get_label(self, velocity):
        if velocity is None: return ""
        speed_px    = velocity["speed_px"]
        area_change = velocity["area_change"]
        dx          = velocity["dx"]
        if speed_px < 1.5:   return "STATIC"
        if area_change > 0.08:  return f"APPROACH {speed_px:.0f}px"
        if area_change < -0.08: return f"RECEDING {speed_px:.0f}px"
        if dx < -3: return f"MERGING-L {speed_px:.0f}px"
        if dx >  3: return f"MERGING-R {speed_px:.0f}px"
        return f"MOVING {speed_px:.0f}px"

    def get_fast_approach(self, velocity):
        if velocity is None: return False
        return velocity["area_change"] > 0.15 and velocity["speed_px"] > 3

# INSTRUCTION_MAP -- from Sunitha's ar_glasses_hud.py
INSTRUCTION_MAP = {
    "go":          "GO",
    "goForward":   "GO FORWARD",
    "goLeft":      "GO LEFT",
    "stop":        "STOP",
    "stopLeft":    "STOP LEFT",
    "warning":     "WARNING",
    "warningLeft": "WARNING LEFT",
}
print("  Sunitha's VelocityTracker + INSTRUCTION_MAP loaded")

# ── 3. Tej's ds_pipeline.py ──────────────────────────────────────────────────
_tej_loaded = False
_DS_PIPELINE_INFO = {
    "jetson_ip":   args.jetson_ip,
    "udp_port":    args.udp_port,
    "rtsp_port":   8554,
    "coco_engine": "models/yolov8n_deepstream.engine",
    "lisa_engine": "models/yolov8n_lisa_v1.1_deepstream.engine",
    "start_cmd":   "./start.sh",
}
# NOTE: ds_pipeline.py cannot be imported on Colab
# (requires GStreamer/DeepStream/Jetson hardware)
# It is used in Mode 3 (Jetson) only
print("  Tej's ds_pipeline.py referenced for Mode 3 (Jetson)")
print("     Cannot import on Colab -- Jetson hardware required")
print()

# =============================================================================
# PATHS
# =============================================================================
_LISA_LABELS = {"go","goforward","goleft","stop","stopleft","warning","warningleft"}

VIDEO_PATH = args.video or next(
    (p for p in [
        os.path.join(REPO, "output_ds_1.mp4"),
        os.path.join(REPO, "test_data", "output_ds_1.mp4"),
        os.path.join(REPO, "output_ds_2.mp4"),
    ] if os.path.exists(p)), None)

JSON_DIR    = args.json or os.path.join(REPO, "runs", "hud", "ds_1")
OUTPUT_PATH = args.output
GIF_PATH    = str(Path(OUTPUT_PATH).with_suffix(".gif"))

print(f"  Mode   : {args.mode.upper()}")
print(f"  Video  : {VIDEO_PATH or 'NOT FOUND'}")
print(f"  JSON   : {JSON_DIR}")
print(f"  Output : {OUTPUT_PATH}")
print("="*60)

# =============================================================================
# PIPELINE
# =============================================================================
def run():

    # ── Mode 3: Jetson ───────────────────────────────────────────────────────
    if args.mode == "jetson":
        print("\n" + "="*60)
        print("  MODE 3 -- Jetson DeepStream Pipeline")
        print("="*60)
        print(f"  Jetson IP   : {_DS_PIPELINE_INFO['jetson_ip']}")
        print(f"  UDP Port    : {_DS_PIPELINE_INFO['udp_port']}")
        print(f"  RTSP Port   : {_DS_PIPELINE_INFO['rtsp_port']}")
        print(f"  COCO Engine : {_DS_PIPELINE_INFO['coco_engine']}")
        print(f"  LISA Engine : {_DS_PIPELINE_INFO['lisa_engine']}")
        print()
        print("  On Jetson run:")
        print(f"    ssh tej@{_DS_PIPELINE_INFO['jetson_ip']}")
        print(f"    cd /home/tej/capstone && ./start.sh")
        print()
        print("  Then re-run with UDP mode pointing to Jetson:")
        print(f"    --mode udp --udp-host {_DS_PIPELINE_INFO['jetson_ip']}")
        print("="*60)
        return

    # ── Mode 1 & 2: Video setup ──────────────────────────────────────────────
    if not VIDEO_PATH or not os.path.exists(VIDEO_PATH):
        print("\n[Pipeline] No video found -- rendering on black canvas")
        _canvas_only = True
        w, h, fps = 1920, 1080, 30.0
        cap = None
    else:
        _canvas_only = False
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Pipeline] Video  : {VIDEO_PATH}  {w}x{h} @ {fps:.1f}fps")

    # ── Mode 2: UDP live ─────────────────────────────────────────────────────
    import threading, socket
    _latest_payload = {"sequence": -1, "detections": []}
    _payload_lock   = threading.Lock()
    _stop_udp       = threading.Event()

    if args.mode == "udp":
        def _udp_listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((args.udp_host, args.udp_port))
            sock.settimeout(1.0)
            print(f"[UDP] Listening on {args.udp_host}:{args.udp_port}")
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

        threading.Thread(target=_udp_listener, daemon=True).start()
        print(f"[Pipeline] UDP mode -- waiting {args.udp_wait}s for first packet...")
        time.sleep(args.udp_wait)

    # ── Mode 1: JSON files ───────────────────────────────────────────────────
    if args.mode == "file":
        json_sources = []
        if args.json:
            json_sources.append(args.json)
        else:
            for d in [
                os.path.join(REPO,"runs","hud","ds_1"),
                os.path.join(REPO,"runs","hud","ds_2"),
            ]:
                if os.path.isdir(d): json_sources.append(d)

        json_files = []
        for src in json_sources:
            files = sorted(glob(os.path.join(src,"frame_*.json")))
            print(f"[Pipeline] JSON   : {len(files)} frames from {src}")
            json_files.extend(files)

        json_files = sorted(json_files)
        if not json_files:
            raise FileNotFoundError("No JSON files found!")

        max_frames = args.max_frames or len(json_files)
        json_files = json_files[:max_frames]
        print(f"[Pipeline] Total  : {len(json_files)} frames")
    else:
        max_frames = args.max_frames or 99999

    # ── Output writer ────────────────────────────────────────────────────────
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    left_zone, right_zone = get_blind_spot_zones(w, h)
    vel_tracker = VelocityTracker()

    total = 0
    t0    = time.time()
    i     = 0

    try:
        for i in range(max_frames):
            if _canvas_only:
                frame = np.zeros((h,w,3), dtype=np.uint8)
            else:
                ok, frame = cap.read()
                if not ok: break

            # Get detections
            if args.mode == "file":
                with open(json_files[i]) as f:
                    payload = json.load(f)
            else:
                with _payload_lock:
                    payload = _latest_payload.copy()

            dets = payload.get("detections", [])

            left_alert  = False
            right_alert = False
            instruction = None
            instr_conf  = 0.0

            # Top bar
            cv2.rectangle(frame,(0,0),(w,60),(20,20,20),-1)
            cv2.putText(frame,"OmniView AI | Live HUD",(20,38),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            cv2.putText(frame,f"seq:{payload.get('sequence',i)}",(w-150,22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,160,160),1)

            # Mode badge
            mode_color = (0,200,100) if args.mode=="file" else (0,165,255)
            cv2.putText(frame,f"MODE:{args.mode.upper()}",(w-320,22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,mode_color,1)

            draw_blind_spot_zones(frame, left_zone, right_zone)

            for det in dets:
                conf  = float(det.get("confidence", det.get("conf",0.0)))
                label = str(det.get("label","object")).strip()
                _raw_model = str(det.get("model","")).strip().lower()
                model_src = _raw_model if _raw_model in ("lisa","coco") else (
                    "lisa" if label.lower().replace(" ","") in _LISA_LABELS else "coco")

                if conf < (0.15 if model_src=="lisa" else args.conf):
                    continue

                lk = label.replace(" ","")
                if model_src=="lisa" and lk in INSTRUCTION_MAP:
                    if conf > instr_conf:
                        instr_conf  = conf
                        instruction = INSTRUCTION_MAP[lk]

                xc = float(det.get("x_center",0)) * w
                yc = float(det.get("y_center",0)) * h
                bw = float(det.get("width",0))    * w
                bh = float(det.get("height",0))   * h
                x1 = max(0, min(w-1, int(xc-bw/2)))
                y1 = max(0, min(h-1, int(yc-bh/2)))
                x2 = max(0, min(w-1, int(xc+bw/2)))
                y2 = max(0, min(h-1, int(yc+bh/2)))
                if x2<=x1 or y2<=y1: continue

                bs = str(det.get("blind_spot","none")).strip().lower()
                if bs=="none" and model_src=="coco":
                    cx_=(x1+x2)//2; cy_=(y1+y2)//2
                    lz,rz = left_zone,right_zone
                    if lz[0]<=cx_<=lz[2] and lz[1]<=cy_<=lz[3]: bs="left"
                    elif rz[0]<=cx_<=rz[2] and rz[1]<=cy_<=rz[3]: bs="right"

                # Velocity tracking (Sunitha's VelocityTracker)
                vel_label     = ""
                fast_approach = False
                if model_src == "coco":
                    cx_ = (x1+x2)//2
                    cy_ = (y1+y2)//2
                    area = max((x2-x1)*(y2-y1), 1)
                    obj_key  = f"{label}_{int(cx_/max(w,1)*8)}_{int(cy_/max(h,1)*6)}"
                    velocity      = vel_tracker.update(obj_key, cx_, cy_, area, i)
                    vel_label     = vel_tracker.get_label(velocity)
                    fast_approach = vel_tracker.get_fast_approach(velocity)

                # Color (Victor's get_box_color)
                color = get_box_color(conf, bs in ("left","right"))
                if fast_approach and bs not in ("left","right"):
                    color = (0,165,255)  # orange = fast approach

                if bs=="left":  left_alert=True
                if bs=="right": right_alert=True

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.circle(frame,((x1+x2)//2,(y1+y2)//2),4,color,-1)

                txt = f"{label[:10]} {conf:.2f}"
                if bs != "none": txt += f" | {bs.upper()}"
                if vel_label:    txt += f" | {vel_label}"

                (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
                ty = y1-5 if y1>th+10 else y1+th+5
                cv2.rectangle(frame,(x1,ty-th-4),(x1+tw+6,ty+2),color,-1)
                cv2.putText(frame,txt,(x1+3,ty),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
                total += 1

            if instruction:
                cv2.rectangle(frame,(w-340,10),(w-20,50),(255,140,0),-1)
                cv2.putText(frame,f"SIGN: {instruction}",(w-325,38),
                            cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2)

            # Victor's draw_alerts
            draw_alerts(frame, left_alert, right_alert)

            cv2.putText(frame,f"Frame:{i}  Dets:{len(dets)}",(20,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            writer.write(frame)

            if i % 100 == 0:
                elapsed = max(time.time()-t0,1e-6)
                print(f"frame={i}/{max_frames}  dets={len(dets)}"
                      f"  total={total}  {i/elapsed:.1f}fps")

    except KeyboardInterrupt:
        print("\n[Pipeline] Stopped.")
    finally:
        elapsed = max(time.time()-t0,1e-6)
        print(f"\n[Pipeline] Done. {total} detections in {i+1} frames"
              f" ({(i+1)/elapsed:.1f}fps)")
        if cap: cap.release()
        writer.release()
        _stop_udp.set()
        print(f"[Pipeline] MP4 saved: {out_path}"
              f" ({os.path.getsize(str(out_path))//1024} KB)")

    if args.gif_frames>0 and _IMAGEIO:
        print("\n[Pipeline] Creating GIF...")
        gif_buf = []
        reader  = imageio.get_reader(str(out_path))
        for k,f in enumerate(reader):
            if k>=args.gif_frames: break
            gif_buf.append(f)
        reader.close()
        imageio.mimsave(GIF_PATH, gif_buf, fps=args.gif_fps)
        print(f"[Pipeline] GIF saved: {GIF_PATH}"
              f" ({len(gif_buf)/args.gif_fps:.1f}s)")
        if _IPYTHON:
            ip_display(IPImage(filename=GIF_PATH))

if __name__ == "__main__":
    run()