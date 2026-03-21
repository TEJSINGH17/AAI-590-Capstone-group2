"""
hud_from_json.py — Step 2 of 2
================================
Read a detections JSON produced by detect_to_json.py and render a
Tesla-style perspective BEV video.  No YOLO / no GPU needed here —
pure OpenCV rendering, typically 50-200x faster than inference.

Supports two JSON formats:
  • Combined  – single file with a "frames" array  (from detect_to_json.py)
  • Per-frame – a directory of frame_XXXXXX.json    (existing pipeline output)
    In per-frame mode coords must be normalised (x_center/y_center/w/h)
    OR absolute (x1/y1/x2/y2).

Usage
-----
# From combined JSON:
python3 hud_from_json.py \\
    --json   runs/json/ds_1_detections.json \\
    --output runs/hud/from_json_ds1.mp4

# From per-frame JSON directory (existing format):
python3 hud_from_json.py \\
    --json   runs/hud/ds_1/ \\
    --width  1920 --height 1080 --fps 30 \\
    --output runs/hud/from_json_ds1.mp4

# Side-by-side mode (BEV only, no camera feed available):
python3 hud_from_json.py --json ... --output ...
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
BG         = ( 10,  10,  16)
SKY_TOP    = (  8,   8,  14)
SKY_BOT    = ( 22,  22,  34)
TERRAIN    = ( 14,  14,  20)
ROAD_FAR   = ( 22,  22,  34)
ROAD_NEAR  = ( 40,  40,  56)
ROAD_EDGE  = (210, 210, 215)
LANE_DASH  = (200, 200, 200)
LANE_CTR   = ( 45, 185, 230)
GRID_LINE  = ( 38,  38,  54)
HOR_GLOW   = ( 65,  65,  90)
EGO_COL    = (  0, 220, 255)
VEH_COL    = (235, 235, 248)
PED_COL    = ( 75, 255, 138)
SIG_GO     = ( 45, 225,  85)
SIG_STOP   = ( 55,  55, 250)
SIG_WARN   = (  0, 205, 255)
HUD_TXT    = (195, 208, 218)
HUD_DIM    = (125, 138, 150)
PANEL_BG   = ( 10,  10,  14)

HEADER_H       = 54
FOOTER_H       = 46
ROAD_HALF_FRAC = 0.42

_VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}
_PERSONS  = {"person"}


# ─────────────────────────────────────────────────────────────────────────────
#  Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kind(label: str) -> str:
    l = label.strip().lower()
    if l in _VEHICLES:
        return "vehicle"
    if l in _PERSONS:
        return "person"
    if any(k in l for k in ("stop", "go", "warn", "light", "sign")):
        return "signal"
    return "other"


def _sig_color(label: str) -> Tuple:
    l = label.lower()
    if "stop" in l or "red" in l:
        return SIG_STOP
    if "warn" in l or "yellow" in l:
        return SIG_WARN
    return SIG_GO


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────────────

def _vp(w: int, h: int) -> Tuple[int, int]:
    content_h = h - HEADER_H - FOOTER_H
    return (w // 2, HEADER_H + int(content_h * 0.30))


def _bot(h: int) -> int:
    return h - FOOTER_H


def _road_half(y: int, w: int, h: int) -> int:
    vpy = _vp(w, h)[1]
    b   = _bot(h)
    if y <= vpy:
        return 0
    t = (y - vpy) / max(1, b - vpy)
    return int(w * ROAD_HALF_FRAC * t)


def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ─────────────────────────────────────────────────────────────────────────────
#  Scene rendering
# ─────────────────────────────────────────────────────────────────────────────

def draw_scene(canvas: np.ndarray, frame_idx: int = 0) -> None:
    h, w = canvas.shape[:2]
    vpx, vpy = _vp(w, h)
    b = _bot(h)

    canvas[:] = BG

    # sky
    for y in range(HEADER_H, vpy + 1):
        t = (y - HEADER_H) / max(1, vpy - HEADER_H)
        cv2.line(canvas, (0, y), (w, y), _lerp(SKY_TOP, SKY_BOT, t), 1)

    # terrain
    for y in range(vpy, b + 1):
        rh = _road_half(y, w, h)
        if vpx - rh > 0:
            cv2.line(canvas, (0, y), (vpx - rh, y), TERRAIN, 1)
        if vpx + rh < w:
            cv2.line(canvas, (vpx + rh, y), (w, y), TERRAIN, 1)

    # road gradient
    for y in range(vpy, b + 1):
        t  = (y - vpy) / max(1, b - vpy)
        rh = _road_half(y, w, h)
        cv2.line(canvas, (vpx - rh, y), (vpx + rh, y),
                 _lerp(ROAD_FAR, ROAD_NEAR, t), 1)

    # horizon glow
    ov = canvas.copy()
    cv2.line(ov, (0, vpy), (w, vpy), HOR_GLOW, 3)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)

    # perspective grid
    n_grid = 16
    for i in range(1, n_grid + 1):
        t  = (i / n_grid) ** 1.8
        y  = int(vpy + (b - vpy) * t)
        rh = _road_half(y, w, h)
        alpha = 0.10 + 0.50 * t
        ov2 = canvas.copy()
        cv2.line(ov2, (vpx - rh, y), (vpx + rh, y), GRID_LINE, 1)
        cv2.addWeighted(ov2, alpha, canvas, 1 - alpha, 0, canvas)

    # road shoulder lines
    rh_bot = _road_half(b, w, h)
    cv2.line(canvas, (vpx, vpy), (vpx - rh_bot, b), ROAD_EDGE, 2, cv2.LINE_AA)
    cv2.line(canvas, (vpx, vpy), (vpx + rh_bot, b), ROAD_EDGE, 2, cv2.LINE_AA)

    # animated lane dashes
    n_dash = 18
    phase  = (frame_idx * 0.10) % 1.0
    x_left_bot  = vpx - rh_bot
    x_right_bot = vpx + rh_bot
    for frac in (1 / 3, 2 / 3):
        x_bot = int(x_left_bot + frac * (x_right_bot - x_left_bot))
        for d in range(n_dash + 1):
            d_eff = (d + phase) % n_dash
            t0 = float(np.clip((d_eff          / n_dash) ** 1.6, 0.0, 0.99))
            t1 = float(np.clip(((d_eff + 0.48) / n_dash) ** 1.6, 0.0, 1.00))
            if t1 <= t0:
                continue
            y0 = int(vpy + (b - vpy) * t0)
            y1 = int(vpy + (b - vpy) * t1)
            x0 = int(vpx + (x_bot - vpx) * t0)
            x1 = int(vpx + (x_bot - vpx) * t1)
            cv2.line(canvas, (x0, y0), (x1, y1), LANE_DASH, 2, cv2.LINE_AA)

    # centre reference
    for d in range(n_dash + 1):
        d_eff = (d + phase) % n_dash
        t0 = float(np.clip((d_eff          / n_dash) ** 1.6, 0.0, 0.99))
        t1 = float(np.clip(((d_eff + 0.28) / n_dash) ** 1.6, 0.0, 1.00))
        if t1 <= t0:
            continue
        y0 = int(vpy + (b - vpy) * t0)
        y1 = int(vpy + (b - vpy) * t1)
        ov3 = canvas.copy()
        cv2.line(ov3, (vpx, y0), (vpx, y1), LANE_CTR, 1)
        cv2.addWeighted(ov3, 0.22, canvas, 0.78, 0, canvas)


def draw_header(canvas: np.ndarray, frame_idx: int, fps: float, n_det: int) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), PANEL_BG, -1)
    cv2.line(canvas, (0, HEADER_H), (w, HEADER_H), (55, 55, 78), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "AUTOPILOT  PERCEPTION", (18, 35), font, 0.74, HUD_TXT, 2, cv2.LINE_AA)
    rtxt = f"FR {frame_idx:05d}   {fps:4.1f} fps   {n_det} obj"
    cv2.putText(canvas, rtxt, (w - 315, 35), font, 0.52, HUD_DIM, 1, cv2.LINE_AA)
    cv2.circle(canvas, (w - 332, 28), 7, SIG_GO, -1, cv2.LINE_AA)


def draw_footer(canvas: np.ndarray, n_det: int) -> None:
    h, w = canvas.shape[:2]
    y0 = h - FOOTER_H
    cv2.rectangle(canvas, (0, y0), (w, h), PANEL_BG, -1)
    cv2.line(canvas, (0, y0), (w, y0), (55, 55, 78), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "45 mph", (20, h - 14), font, 0.64, HUD_TXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "AUTOSTEER  ACTIVE", (w // 2 - 105, h - 14),
                font, 0.58, EGO_COL, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{n_det} OBJECTS DETECTED", (w - 270, h - 14),
                font, 0.52, HUD_DIM, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Object icons
# ─────────────────────────────────────────────────────────────────────────────

def _fill_alpha(canvas, pts: np.ndarray, color, alpha: float = 0.30) -> None:
    ov = canvas.copy()
    cv2.fillPoly(ov, [pts.astype(np.int32)], color)
    cv2.addWeighted(ov, alpha, canvas, 1.0 - alpha, 0, canvas)


def draw_car_icon(canvas, cx: int, cy: int, iw: int, ih: int, color) -> None:
    hw, hh = iw / 2.0, ih / 2.0
    body = np.array([
        [-hw*0.38, -hh*0.50], [ hw*0.38, -hh*0.50],
        [ hw*0.50, -hh*0.16], [ hw*0.50,  hh*0.32],
        [ hw*0.33,  hh*0.50], [-hw*0.33,  hh*0.50],
        [-hw*0.50,  hh*0.32], [-hw*0.50, -hh*0.16],
    ], np.float32)
    body[:, 0] += cx; body[:, 1] += cy
    _fill_alpha(canvas, body, color, 0.34)
    cv2.polylines(canvas, [body.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (int(body[0,0]), int(body[0,1])),
                     (int(body[1,0]), int(body[1,1])), color, 1, cv2.LINE_AA)
    ww = max(2, iw // 7); wh = max(3, ih // 9)
    for sx, sy in ((-1,-1),(1,-1),(-1,1),(1,1)):
        cv2.rectangle(canvas,
                      (int(cx + sx*hw*0.56) - ww, int(cy + sy*hh*0.30) - wh),
                      (int(cx + sx*hw*0.56) + ww, int(cy + sy*hh*0.30) + wh),
                      color, -1)


def draw_truck_icon(canvas, cx: int, cy: int, iw: int, ih: int, color) -> None:
    hw, hh = iw / 2.0, ih / 2.0
    trailer = np.array([
        [-hw*0.50, -hh*0.10], [ hw*0.50, -hh*0.10],
        [ hw*0.44,  hh*0.50], [-hw*0.44,  hh*0.50],
    ], np.float32)
    cab = np.array([
        [-hw*0.42, -hh*0.50], [ hw*0.42, -hh*0.50],
        [ hw*0.50, -hh*0.10], [-hw*0.50, -hh*0.10],
    ], np.float32)
    for pts in (trailer, cab):
        pts[:, 0] += cx; pts[:, 1] += cy
    _fill_alpha(canvas, trailer, color, 0.26)
    _fill_alpha(canvas, cab,     color, 0.42)
    cv2.polylines(canvas, [trailer.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [cab.astype(np.int32)],     True, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (int(cx-hw*0.50), int(cy-hh*0.10)),
                     (int(cx+hw*0.50), int(cy-hh*0.10)), color, 1, cv2.LINE_AA)
    ww = max(2, iw//7); wh = max(2, ih//10)
    for sx, sy in ((-1,-0.68),(1,-0.68),(-1,0.04),(1,0.04),(-1,0.66),(1,0.66)):
        wx = int(cx + sx*hw*0.58); wy = int(cy + sy*hh*0.50)
        cv2.rectangle(canvas, (wx-ww, wy-wh), (wx+ww, wy+wh), color, -1)


def draw_person_icon(canvas, cx: int, cy: int, iw: int, ih: int, color) -> None:
    r = max(4, iw // 4); body = max(8, ih // 2)
    ov = canvas.copy()
    cv2.circle(ov, (cx, cy - body), r, color, -1)
    cv2.addWeighted(ov, 0.32, canvas, 0.68, 0, canvas)
    cv2.circle(canvas, (cx, cy - body), r, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy-body+r), (cx, cy), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx-r*2, cy-body//2), (cx+r*2, cy-body//2), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy), (cx-r*2, cy+body), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy), (cx+r*2, cy+body), color, 2, cv2.LINE_AA)


def draw_glow_ring(canvas, cx: int, cy: int, scale: float, color) -> None:
    rx = max(14, int(24 * scale)); ry = max(5, int(9 * scale))
    ov = canvas.copy()
    cv2.ellipse(ov, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
    cv2.addWeighted(ov, 0.10, canvas, 0.90, 0, canvas)
    cv2.ellipse(canvas, (cx, cy), (rx,   ry  ), 0, 0, 360, color, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (rx+4, ry+2), 0, 0, 360, color, 1, cv2.LINE_AA)


def draw_signal_badge(canvas, x: int, y: int, label: str, color) -> None:
    txt = label.upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(txt, font, 0.52, 1)
    pw, ph = tw + 22, th + bl + 14
    ov = canvas.copy()
    cv2.rectangle(ov, (x, y), (x+pw, y+ph), color, -1)
    cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x+pw, y+ph), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x+11, y+ph-9), font, 0.52, (15,15,20), 1, cv2.LINE_AA)


def draw_motion_trail(canvas, positions: List[Tuple[int,int]], color) -> None:
    n = len(positions)
    if n < 2:
        return
    for i in range(n - 1):
        alpha = (i + 1) / n * 0.55
        thick = max(1, int((i + 1) / n * 3))
        ov = canvas.copy()
        cv2.line(ov, positions[i], positions[i+1], color, thick, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, canvas, 1.0 - alpha, 0, canvas)


def draw_ego_speed_lines(canvas, cx: int, cy: int, iw: int, ih: int, frame_idx: int) -> None:
    offsets = [-int(iw*0.38), -int(iw*0.14), int(iw*0.14), int(iw*0.38)]
    cycle = frame_idx % 6
    for k, x_off in enumerate(offsets):
        phase  = (cycle + k * 2) % 6
        length = int(ih * (0.30 + 0.25 * (phase / 5)))
        x = cx + x_off
        y1 = cy + int(ih * 0.54); y2 = y1 + length
        ov = canvas.copy()
        cv2.line(ov, (x, y1), (x, y2), EGO_COL, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.45, canvas, 0.55, 0, canvas)


# ─────────────────────────────────────────────────────────────────────────────
#  Projection
# ─────────────────────────────────────────────────────────────────────────────

def project_to_bev(
    x1: float, y1: float, x2: float, y2: float,
    frame_w: int, frame_h: int,
    canvas_w: int, canvas_h: int,
) -> Tuple[int, int, float]:
    foot_cx = (x1 + x2) / 2.0
    box_h   = max(1.0, y2 - y1)
    lateral = (foot_cx - frame_w / 2.0) / (frame_w / 2.0)

    bottom_ratio = float(np.clip(y2 / frame_h, 0.0, 1.0))
    height_ratio = float(np.clip(box_h / (frame_h * 0.50), 0.0, 1.0))
    closeness    = float(np.clip(0.60 * bottom_ratio + 0.40 * height_ratio, 0.05, 1.0))

    vpx, vpy = _vp(canvas_w, canvas_h)
    b        = _bot(canvas_h)
    far_y    = vpy + int((b - vpy) * 0.04)
    near_y   = b   - int((b - vpy) * 0.10)
    bev_y    = int(far_y + closeness * (near_y - far_y))

    rh    = _road_half(bev_y, canvas_w, canvas_h)
    bev_x = int(vpx + lateral * rh * 0.82)
    scale = 0.40 + closeness * 1.25
    return bev_x, bev_y, scale


# ─────────────────────────────────────────────────────────────────────────────
#  Master render
# ─────────────────────────────────────────────────────────────────────────────

def render_bev(
    detections: List[Dict],
    frame_idx: int,
    fps: float,
    canvas_w: int,
    canvas_h: int,
    src_w: int,
    src_h: int,
    track_history: Dict[int, Deque],
) -> np.ndarray:
    canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    draw_scene(canvas, frame_idx)
    draw_header(canvas, frame_idx, fps, len(detections))
    draw_footer(canvas, len(detections))

    vpx, _ = _vp(canvas_w, canvas_h)
    ego_y  = _bot(canvas_h) - 52
    ego_iw = max(20, int(canvas_w * 0.060))
    ego_ih = max(34, int(canvas_w * 0.100))
    draw_ego_speed_lines(canvas, vpx, ego_y, ego_iw, ego_ih, frame_idx)
    draw_car_icon(canvas, vpx, ego_y, ego_iw, ego_ih, EGO_COL)

    sig_x = 18; sig_y = HEADER_H + 10
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        label = str(det["label"])
        kind  = _kind(label)
        conf  = float(det["conf"])
        color = (VEH_COL if kind == "vehicle" else
                 PED_COL if kind == "person"  else
                 _sig_color(label))

        if kind == "signal":
            draw_signal_badge(canvas, sig_x, sig_y, label, color)
            sig_x += 148
            continue

        bx, by, scale = project_to_bev(
            det["x1"], det["y1"], det["x2"], det["y2"],
            src_w, src_h, canvas_w, canvas_h,
        )

        tid = det.get("track_id", -1)
        if tid >= 0:
            if tid not in track_history:
                track_history[tid] = deque(maxlen=14)
            track_history[tid].append((bx, by))
            draw_motion_trail(canvas, list(track_history[tid]), color)

        draw_glow_ring(canvas, bx, by, scale, color)
        iw = max(14, int(26 * scale)); ih = max(22, int(44 * scale))
        l  = label.lower()

        if "truck" in l or "bus" in l:
            draw_truck_icon(canvas, bx, by, int(iw*1.30), int(ih*1.50), color)
        elif kind == "vehicle":
            draw_car_icon(canvas, bx, by, iw, ih, color)
        elif kind == "person":
            draw_person_icon(canvas, bx, by, iw, ih, color)
        else:
            cv2.circle(canvas, (bx, by), max(6, int(12*scale)), color, 2, cv2.LINE_AA)

        badge = f"{l}  {conf*100:.0f}%"
        (tw, _), _ = cv2.getTextSize(badge, font, 0.40, 1)
        cv2.putText(canvas, badge, (bx - tw//2, by - int(ih*0.62) - 6),
                    font, 0.40, color, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  JSON loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_combined_json(path: Path):
    """Load a single combined JSON (from detect_to_json.py)."""
    with open(path) as f:
        data = json.load(f)
    meta   = data["meta"]
    frames = data["frames"]   # list of {idx, detections}
    return meta, frames


def _normalised_to_abs(det: dict, w: int, h: int) -> dict:
    """Convert normalised xywh → absolute xyxy in-place."""
    if "x1" in det:
        return det   # already absolute
    cx = det["x_center"] * w
    cy = det["y_center"] * h
    bw = det["width"]    * w
    bh = det["height"]   * h
    return {
        "track_id": det.get("track_id", -1),
        "cls_id":   det.get("class_id", 0),
        "label":    det.get("label", "unknown"),
        "conf":     det.get("confidence", det.get("conf", 0.0)),
        "x1": cx - bw/2, "y1": cy - bh/2,
        "x2": cx + bw/2, "y2": cy + bh/2,
    }


def _load_perframe_dir(directory: Path, w: int, h: int):
    """Load a directory of per-frame JSON files (existing pipeline output)."""
    files = sorted(directory.glob("frame_*.json"))
    if not files:
        raise FileNotFoundError(f"No frame_*.json files found in {directory}")
    frames = []
    for fp in files:
        with open(fp) as f:
            raw = json.load(f)
        idx  = raw.get("sequence", raw.get("frame_num",
               int(fp.stem.split("_")[1])))
        dets = [_normalised_to_abs(d, w, h) for d in raw.get("detections", [])]
        frames.append({"idx": idx, "detections": dets})
    meta = {"width": w, "height": h, "fps": 30.0, "total_frames": len(frames),
            "source": str(directory)}
    return meta, frames


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render Tesla-style HUD video from a detections JSON."
    )
    p.add_argument("--json",   required=True,
                   help="Path to combined .json file OR directory of per-frame JSONs.")
    p.add_argument("--output", required=True, help="Output .mp4 path.")
    p.add_argument("--width",  type=int, default=0,
                   help="Source frame width  (required for per-frame JSON directories).")
    p.add_argument("--height", type=int, default=0,
                   help="Source frame height (required for per-frame JSON directories).")
    p.add_argument("--fps",    type=float, default=0.0,
                   help="Output FPS override (default: taken from JSON meta).")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    json_path = Path(args.json)
    out_path  = Path(args.output)

    # ── load detections ──────────────────────────────────────────────────────
    if json_path.is_dir():
        if not args.width or not args.height:
            raise ValueError(
                "Per-frame JSON directory requires --width and --height."
            )
        meta, frames = _load_perframe_dir(json_path, args.width, args.height)
    else:
        meta, frames = _load_combined_json(json_path)

    src_w  = args.width  or meta["width"]
    src_h  = args.height or meta["height"]
    src_fps = args.fps   or meta.get("fps", 30.0)
    total   = meta.get("total_frames", len(frames))

    print(f"Source: {meta.get('source', json_path)}")
    print(f"Frames: {total}  |  {src_w}×{src_h}  @  {src_fps:.1f} fps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    track_history: Dict[int, Deque] = {}
    t0 = time.time()

    for fi, frame_data in enumerate(frames, start=1):
        idx  = frame_data["idx"]
        dets = frame_data["detections"]

        # render_fps = how fast we're rendering (not source fps)
        elapsed     = max(time.time() - t0, 1e-6)
        render_fps  = fi / elapsed

        canvas = render_bev(
            detections=dets,
            frame_idx=idx,
            fps=render_fps,
            canvas_w=src_w,
            canvas_h=src_h,
            src_w=src_w,
            src_h=src_h,
            track_history=track_history,
        )

        writer.write(canvas)

        if fi % 100 == 0 or fi == total:
            pct = fi / max(total, 1) * 100
            print(f"  [{pct:5.1f}%]  frame {idx:5d} / {total}  "
                  f"render {render_fps:.1f} fps")

    writer.release()
    elapsed = max(time.time() - t0, 1e-6)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone → {out_path}  ({size_mb:.1f} MB)")
    print(f"Rendered {fi} frames in {elapsed:.1f}s  ({fi/elapsed:.1f} fps)")


if __name__ == "__main__":
    main()
