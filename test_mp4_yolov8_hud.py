"""
Tesla-style Autonomous Driving Perception Visualizer
=====================================================
Generates a fully synthetic perspective BEV video from YOLOv8 detections.
No real camera image in default mode — all objects rendered as top-down
stylized icons on a perspective road scene (like Tesla FSD v12).

Usage
-----
# BEV only:
python3 test_mp4_yolov8_hud.py \\
    --model  models/yolov8n.pt \\
    --input  test_data/output_ds_1.mp4 \\
    --output runs/hud/tesla_bev.mp4

# Side-by-side (camera left, BEV right):
python3 test_mp4_yolov8_hud.py ... --show-camera
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  Palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
BG         = ( 10,  10,  16)   # canvas background
SKY_TOP    = (  8,   8,  14)   # horizon sky (dark)
SKY_BOT    = ( 22,  22,  34)   # sky at horizon  (slightly lighter)
TERRAIN    = ( 14,  14,  20)   # off-road surface
ROAD_FAR   = ( 22,  22,  34)   # road colour near horizon
ROAD_NEAR  = ( 40,  40,  56)   # road colour near camera
ROAD_EDGE  = (210, 210, 215)   # solid white road shoulders
LANE_DASH  = (200, 200, 200)   # white dashes
LANE_CTR   = ( 45, 185, 230)   # subtle blue centre reference
GRID_LINE  = ( 38,  38,  54)   # perspective grid
HOR_GLOW   = ( 65,  65,  90)   # thin horizon glow
EGO_COL    = (  0, 220, 255)   # cyan  – ego vehicle
VEH_COL    = (235, 235, 248)   # near-white – other vehicles
PED_COL    = ( 75, 255, 138)   # green – pedestrians
SIG_GO     = ( 45, 225,  85)   # signal: go / green
SIG_STOP   = ( 55,  55, 250)   # signal: stop / red
SIG_WARN   = (  0, 205, 255)   # signal: warn / yellow
HUD_TXT    = (195, 208, 218)
HUD_DIM    = (125, 138, 150)
PANEL_BG   = ( 10,  10,  14)


# ─────────────────────────────────────────────────────────────────────────────
#  Layout
# ─────────────────────────────────────────────────────────────────────────────
HEADER_H       = 54     # px  – top status bar
FOOTER_H       = 46     # px  – bottom status bar
ROAD_HALF_FRAC = 0.42   # road half-width at the bottom as fraction of W


# ─────────────────────────────────────────────────────────────────────────────
#  Label taxonomy
# ─────────────────────────────────────────────────────────────────────────────
_VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}
_PERSONS  = {"person"}


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
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vp(w: int, h: int) -> Tuple[int, int]:
    """Vanishing point: horizontally centred, ~30% into the content zone."""
    content_h = h - HEADER_H - FOOTER_H
    return (w // 2, HEADER_H + int(content_h * 0.30))


def _bot(h: int) -> int:
    return h - FOOTER_H


def _road_half(y: int, w: int, h: int) -> int:
    """Road half-width at canvas row y (0 at VP, full at bottom)."""
    vpy = _vp(w, h)[1]
    b   = _bot(h)
    if y <= vpy:
        return 0
    t = (y - vpy) / max(1, b - vpy)
    return int(w * ROAD_HALF_FRAC * t)


def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


# ─────────────────────────────────────────────────────────────────────────────
#  Scene: background + road + grid + lane lines
# ─────────────────────────────────────────────────────────────────────────────

def draw_scene(canvas: np.ndarray, frame_idx: int = 0) -> None:
    h, w = canvas.shape[:2]
    vpx, vpy = _vp(w, h)
    b = _bot(h)

    # fill whole canvas with base BG
    canvas[:] = BG

    # ── sky gradient (header bottom → horizon) ──────────────────────────────
    for y in range(HEADER_H, vpy + 1):
        t = (y - HEADER_H) / max(1, vpy - HEADER_H)
        cv2.line(canvas, (0, y), (w, y), _lerp(SKY_TOP, SKY_BOT, t), 1)

    # ── terrain strips (off-road, horizon → bottom) ─────────────────────────
    for y in range(vpy, b + 1):
        rh = _road_half(y, w, h)
        if vpx - rh > 0:
            cv2.line(canvas, (0, y), (vpx - rh, y), TERRAIN, 1)
        if vpx + rh < w:
            cv2.line(canvas, (vpx + rh, y), (w, y), TERRAIN, 1)

    # ── road surface gradient ───────────────────────────────────────────────
    for y in range(vpy, b + 1):
        t  = (y - vpy) / max(1, b - vpy)
        rh = _road_half(y, w, h)
        cv2.line(canvas, (vpx - rh, y), (vpx + rh, y),
                 _lerp(ROAD_FAR, ROAD_NEAR, t), 1)

    # ── horizon glow ────────────────────────────────────────────────────────
    ov = canvas.copy()
    cv2.line(ov, (0, vpy), (w, vpy), HOR_GLOW, 3)
    cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)

    # ── perspective grid (horizontal) ───────────────────────────────────────
    n_grid = 16
    for i in range(1, n_grid + 1):
        t  = (i / n_grid) ** 1.8      # log spacing: dense near camera
        y  = int(vpy + (b - vpy) * t)
        rh = _road_half(y, w, h)
        alpha = 0.10 + 0.50 * t
        ov2 = canvas.copy()
        cv2.line(ov2, (vpx - rh, y), (vpx + rh, y), GRID_LINE, 1)
        cv2.addWeighted(ov2, alpha, canvas, 1 - alpha, 0, canvas)

    # ── road shoulder lines ─────────────────────────────────────────────────
    rh_bot = _road_half(b, w, h)
    cv2.line(canvas, (vpx, vpy), (vpx - rh_bot, b), ROAD_EDGE, 2, cv2.LINE_AA)
    cv2.line(canvas, (vpx, vpy), (vpx + rh_bot, b), ROAD_EDGE, 2, cv2.LINE_AA)

    # ── dashed lane dividers — animated scroll (forward-motion feel) ─────────
    n_dash = 18
    # phase shifts one full dash-slot every 10 frames → smooth forward scroll
    phase  = (frame_idx * 0.10) % 1.0   # 0..1 fractional shift
    x_left_bot  = vpx - rh_bot
    x_right_bot = vpx + rh_bot
    for frac in (1 / 3, 2 / 3):
        x_bot = int(x_left_bot + frac * (x_right_bot - x_left_bot))
        for d in range(n_dash + 1):
            d_eff = (d + phase) % n_dash
            t0 = float(np.clip((d_eff        / n_dash) ** 1.6, 0.0, 0.99))
            t1 = float(np.clip(((d_eff + 0.48) / n_dash) ** 1.6, 0.0, 1.00))
            if t1 <= t0:
                continue
            y0 = int(vpy + (b - vpy) * t0)
            y1 = int(vpy + (b - vpy) * t1)
            x0 = int(vpx + (x_bot - vpx) * t0)
            x1 = int(vpx + (x_bot - vpx) * t1)
            cv2.line(canvas, (x0, y0), (x1, y1), LANE_DASH, 2, cv2.LINE_AA)

    # ── subtle centre reference line (blue, also scrolling) ─────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
#  HUD bars
# ─────────────────────────────────────────────────────────────────────────────

def draw_header(canvas: np.ndarray, frame_idx: int, fps: float, n_det: int) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), PANEL_BG, -1)
    cv2.line(canvas, (0, HEADER_H), (w, HEADER_H), (55, 55, 78), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "AUTOPILOT  PERCEPTION", (18, 35),
                font, 0.74, HUD_TXT, 2, cv2.LINE_AA)
    rtxt = f"FR {frame_idx:05d}   {fps:4.1f} fps   {n_det} obj"
    cv2.putText(canvas, rtxt, (w - 315, 35), font, 0.52, HUD_DIM, 1, cv2.LINE_AA)
    cv2.circle(canvas, (w - 332, 28), 7, SIG_GO, -1, cv2.LINE_AA)   # green dot


def draw_footer(canvas: np.ndarray, n_det: int) -> None:
    h, w = canvas.shape[:2]
    y0 = h - FOOTER_H
    cv2.rectangle(canvas, (0, y0), (w, h), PANEL_BG, -1)
    cv2.line(canvas, (0, y0), (w, y0), (55, 55, 78), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "45 mph", (20, h - 14), font, 0.64, HUD_TXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "AUTOSTEER  ACTIVE",
                (w // 2 - 105, h - 14), font, 0.58, EGO_COL, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{n_det} OBJECTS DETECTED",
                (w - 270, h - 14), font, 0.52, HUD_DIM, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Object icons  (top-down view)
# ─────────────────────────────────────────────────────────────────────────────

def _fill_alpha(canvas, pts: np.ndarray, color, alpha: float = 0.30) -> None:
    ov = canvas.copy()
    cv2.fillPoly(ov, [pts.astype(np.int32)], color)
    cv2.addWeighted(ov, alpha, canvas, 1.0 - alpha, 0, canvas)


def draw_car_icon(canvas, cx: int, cy: int, iw: int, ih: int, color) -> None:
    """Tesla-style top-down passenger car."""
    hw, hh = iw / 2.0, ih / 2.0
    body = np.array([
        [-hw*0.38, -hh*0.50],
        [ hw*0.38, -hh*0.50],
        [ hw*0.50, -hh*0.16],
        [ hw*0.50,  hh*0.32],
        [ hw*0.33,  hh*0.50],
        [-hw*0.33,  hh*0.50],
        [-hw*0.50,  hh*0.32],
        [-hw*0.50, -hh*0.16],
    ], np.float32)
    body[:, 0] += cx
    body[:, 1] += cy

    _fill_alpha(canvas, body, color, 0.34)
    cv2.polylines(canvas, [body.astype(np.int32)], True, color, 2, cv2.LINE_AA)

    # windshield  (line across front)
    cv2.line(canvas,
             (int(body[0, 0]), int(body[0, 1])),
             (int(body[1, 0]), int(body[1, 1])),
             color, 1, cv2.LINE_AA)

    # four wheel blocks
    ww = max(2, iw // 7)
    wh = max(3, ih // 9)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        wx = int(cx + sx * hw * 0.56)
        wy = int(cy + sy * hh * 0.30)
        cv2.rectangle(canvas, (wx - ww, wy - wh), (wx + ww, wy + wh), color, -1)


def draw_truck_icon(canvas, cx: int, cy: int, iw: int, ih: int, color) -> None:
    """Top-down truck / bus."""
    hw, hh = iw / 2.0, ih / 2.0
    # trailer (rear 55 %)
    trailer = np.array([
        [-hw*0.50, -hh*0.10],
        [ hw*0.50, -hh*0.10],
        [ hw*0.44,  hh*0.50],
        [-hw*0.44,  hh*0.50],
    ], np.float32)
    # cab (front 45 %)
    cab = np.array([
        [-hw*0.42, -hh*0.50],
        [ hw*0.42, -hh*0.50],
        [ hw*0.50, -hh*0.10],
        [-hw*0.50, -hh*0.10],
    ], np.float32)
    for pts in (trailer, cab):
        pts[:, 0] += cx
        pts[:, 1] += cy

    _fill_alpha(canvas, trailer, color, 0.26)
    _fill_alpha(canvas, cab, color, 0.42)
    cv2.polylines(canvas, [trailer.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [cab.astype(np.int32)],    True, color, 2, cv2.LINE_AA)
    # cab–trailer divider
    cv2.line(canvas,
             (int(cx - hw*0.50), int(cy - hh*0.10)),
             (int(cx + hw*0.50), int(cy - hh*0.10)),
             color, 1, cv2.LINE_AA)
    # six wheels
    ww = max(2, iw // 7)
    wh = max(2, ih // 10)
    for sx, sy in ((-1, -0.68), (1, -0.68), (-1, 0.04), (1, 0.04), (-1, 0.66), (1, 0.66)):
        wx = int(cx + sx * hw * 0.58)
        wy = int(cy + sy * hh * 0.50)
        cv2.rectangle(canvas, (wx - ww, wy - wh), (wx + ww, wy + wh), color, -1)


def draw_person_icon(canvas, cx: int, cy: int, iw: int, ih: int, color) -> None:
    r    = max(4, iw // 4)
    body = max(8, ih // 2)
    # head
    ov = canvas.copy()
    cv2.circle(ov, (cx, cy - body), r, color, -1)
    cv2.addWeighted(ov, 0.32, canvas, 0.68, 0, canvas)
    cv2.circle(canvas, (cx, cy - body), r, color, 2, cv2.LINE_AA)
    # spine
    cv2.line(canvas, (cx, cy - body + r), (cx, cy), color, 2, cv2.LINE_AA)
    # arms
    cv2.line(canvas, (cx - r*2, cy - body//2), (cx + r*2, cy - body//2), color, 2, cv2.LINE_AA)
    # legs
    cv2.line(canvas, (cx, cy), (cx - r*2, cy + body), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy), (cx + r*2, cy + body), color, 2, cv2.LINE_AA)


def draw_glow_ring(canvas, cx: int, cy: int, scale: float, color) -> None:
    """Soft elliptical ground-shadow under each detected object."""
    rx = max(14, int(24 * scale))
    ry = max(5,  int( 9 * scale))
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
    cv2.rectangle(ov, (x, y), (x + pw, y + ph), color, -1)
    cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + pw, y + ph), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x + 11, y + ph - 9), font, 0.52, (15, 15, 20), 1, cv2.LINE_AA)


def draw_motion_trail(
    canvas: np.ndarray,
    positions: List[Tuple[int, int]],
    color: Tuple,
) -> None:
    """Fading trail of past BEV positions — gives sense of movement."""
    n = len(positions)
    if n < 2:
        return
    for i in range(n - 1):
        alpha = (i + 1) / n * 0.55       # older = more transparent
        thick = max(1, int((i + 1) / n * 3))
        ov = canvas.copy()
        cv2.line(ov, positions[i], positions[i + 1], color, thick, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, canvas, 1.0 - alpha, 0, canvas)


def draw_ego_speed_lines(
    canvas: np.ndarray,
    cx: int, cy: int,
    iw: int, ih: int,
    frame_idx: int,
) -> None:
    """Animated speed streaks below the ego car to simulate forward motion."""
    # Use frame_idx to cycle streak lengths so they feel alive
    offsets = [-int(iw * 0.38), -int(iw * 0.14), int(iw * 0.14), int(iw * 0.38)]
    cycle = frame_idx % 6   # 0-5 cycle
    for k, x_off in enumerate(offsets):
        phase = (cycle + k * 2) % 6
        length = int(ih * (0.30 + 0.25 * (phase / 5)))
        x = cx + x_off
        y1 = cy + int(ih * 0.54)
        y2 = y1 + length
        ov = canvas.copy()
        cv2.line(ov, (x, y1), (x, y2), EGO_COL, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.45, canvas, 0.55, 0, canvas)


# ─────────────────────────────────────────────────────────────────────────────
#  Projection: camera bbox → BEV position
# ─────────────────────────────────────────────────────────────────────────────

def project_to_bev(
    box: np.ndarray,
    frame_w: int, frame_h: int,
    canvas_w: int, canvas_h: int,
) -> Tuple[int, int, float]:
    """
    Map a 2D bbox from the camera frame to (bev_x, bev_y, scale) in
    the perspective BEV canvas using single-camera depth heuristics:
      - bbox bottom-y  → primary depth cue (lower = closer)
      - bbox height    → secondary depth cue (taller = closer)
      - bbox centre-x  → lateral position
    """
    x1, y1, x2, y2 = box
    foot_cx = (x1 + x2) / 2.0
    box_h   = max(1.0, float(y2 - y1))

    # --- depth / closeness  (0 = far horizon, 1 = right in front) -----------
    bottom_ratio = float(np.clip(y2 / frame_h, 0.0, 1.0))
    height_ratio = float(np.clip(box_h / (frame_h * 0.50), 0.0, 1.0))
    closeness    = float(np.clip(0.60 * bottom_ratio + 0.40 * height_ratio,
                                 0.05, 1.0))

    # --- canvas y  (perspective: close objects near bottom) -----------------
    vpx, vpy = _vp(canvas_w, canvas_h)
    b        = _bot(canvas_h)
    far_y    = vpy + int((b - vpy) * 0.04)
    near_y   = b   - int((b - vpy) * 0.10)
    bev_y    = int(far_y + closeness * (near_y - far_y))

    # --- canvas x  (lateral, scaled to road width at bev_y) -----------------
    lateral  = (foot_cx - frame_w / 2.0) / (frame_w / 2.0)   # −1 … +1
    rh       = _road_half(bev_y, canvas_w, canvas_h)
    bev_x    = int(vpx + lateral * rh * 0.82)

    # --- icon scale  ---------------------------------------------------------
    scale = 0.40 + closeness * 1.25

    return bev_x, bev_y, scale


# ─────────────────────────────────────────────────────────────────────────────
#  Master render functions
# ─────────────────────────────────────────────────────────────────────────────

def render_bev(
    frame: np.ndarray,
    detections: List[Dict],
    frame_idx: int,
    fps: float,
    track_history: Dict[int, Deque] | None = None,
) -> np.ndarray:
    """Fully synthetic Tesla-style BEV frame."""
    h, w = frame.shape[:2]
    canvas = np.empty((h, w, 3), dtype=np.uint8)

    draw_scene(canvas, frame_idx)          # animated lane dashes
    draw_header(canvas, frame_idx, fps, len(detections))
    draw_footer(canvas, len(detections))

    # ego vehicle – fixed at bottom centre with speed lines
    vpx, _ = _vp(w, h)
    ego_y  = _bot(h) - 52
    ego_iw = max(20, int(w * 0.060))
    ego_ih = max(34, int(w * 0.100))
    draw_ego_speed_lines(canvas, vpx, ego_y, ego_iw, ego_ih, frame_idx)
    draw_car_icon(canvas, vpx, ego_y, ego_iw, ego_ih, EGO_COL)

    # signal badges – top-left strip
    sig_x = 18
    sig_y = HEADER_H + 10
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

        bx, by, scale = project_to_bev(det["box"], w, h, w, h)

        # motion trail (requires tracking IDs)
        if track_history is not None:
            tid = det.get("track_id", -1)
            if tid >= 0:
                if tid not in track_history:
                    track_history[tid] = deque(maxlen=14)
                track_history[tid].append((bx, by))
                draw_motion_trail(canvas, list(track_history[tid]), color)

        draw_glow_ring(canvas, bx, by, scale, color)

        iw = max(14, int(26 * scale))
        ih = max(22, int(44 * scale))
        l  = label.lower()

        if "truck" in l or "bus" in l:
            draw_truck_icon(canvas, bx, by, int(iw * 1.30), int(ih * 1.50), color)
        elif kind == "vehicle":
            draw_car_icon(canvas, bx, by, iw, ih, color)
        elif kind == "person":
            draw_person_icon(canvas, bx, by, iw, ih, color)
        else:
            cv2.circle(canvas, (bx, by), max(6, int(12 * scale)), color, 2, cv2.LINE_AA)

        # label  +  confidence
        badge = f"{l}  {conf*100:.0f}%"
        (tw, _), _ = cv2.getTextSize(badge, font, 0.40, 1)
        lx = bx - tw // 2
        ly = by - int(ih * 0.62) - 6
        cv2.putText(canvas, badge, (lx, ly), font, 0.40, color, 1, cv2.LINE_AA)

    return canvas


def render_split(
    frame: np.ndarray,
    detections: List[Dict],
    frame_idx: int,
    fps: float,
    track_history: Dict[int, Deque] | None = None,
) -> np.ndarray:
    """Left = annotated camera feed  |  Right = BEV."""
    h, w = frame.shape[:2]
    out  = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # left pane: real camera + coloured boxes
    cam = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(int)
        label = str(det["label"])
        color = (_sig_color(label) if _kind(label) == "signal" else
                 PED_COL           if _kind(label) == "person"  else
                 VEH_COL)
        cv2.rectangle(cam, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(cam, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
    out[:, :w] = cam

    # right pane: BEV
    out[:, w:] = render_bev(frame, detections, frame_idx, fps, track_history)

    # vertical divider
    cv2.line(out, (w, 0), (w, h), (80, 80, 105), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tesla-style BEV perception visualizer for MP4 videos."
    )
    p.add_argument("--model",       required=True,
                   help="Path to YOLOv8 .pt / .engine / .onnx file.")
    p.add_argument("--input",       required=True,
                   help="Input .mp4 path.")
    p.add_argument("--output",      default="",
                   help="Output .mp4 path (optional; skipped if empty).")
    p.add_argument("--conf",        type=float, default=0.25,
                   help="Detection confidence threshold.")
    p.add_argument("--device",      default="",
                   help="Inference device: 'cpu', '0', etc.")
    p.add_argument("--show-camera", action="store_true",
                   help="Side-by-side: camera left, BEV right.")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w   = W * 2 if args.show_camera else W
    out_h   = H

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))

    frame_idx     = 0
    t0            = time.time()
    track_history: Dict[int, Deque] = {}   # tid → deque of (bev_x, bev_y)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # track() keeps consistent IDs across frames; persist=True is required
        results = model.track(
            source=frame,
            conf=args.conf,
            device=args.device or None,
            persist=True,
            verbose=False,
        )

        frame_idx += 1
        elapsed    = max(time.time() - t0, 1e-6)
        live_fps   = frame_idx / elapsed

        detections: List[Dict] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            xyxy    = boxes.xyxy.cpu().numpy()
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            # track IDs may be None on the first frame or if tracker loses lock
            track_ids = (boxes.id.cpu().numpy().astype(int)
                         if boxes.id is not None
                         else range(len(xyxy)))
            for i in range(len(xyxy)):
                cid = int(cls_ids[i])
                detections.append({
                    "box":      xyxy[i],
                    "conf":     float(confs[i]),
                    "cls_id":   cid,
                    "label":    names.get(cid, str(cid)),
                    "track_id": int(track_ids[i]),
                })

        display = (render_split if args.show_camera else render_bev)(
            frame, detections, frame_idx, live_fps, track_history
        )

        if writer is not None:
            writer.write(display)

        cv2.imshow("Tesla Perception HUD", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = max(time.time() - t0, 1e-6)
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.2f} FPS)")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
