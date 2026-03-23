"""
deepstream/ds_pipeline.py  --  Full NVIDIA DeepStream 7.x Pipeline
===================================================================

Uses NVIDIA DeepStream natively — inference stays entirely on the GPU:

  nvinfer            TensorRT inference (no Python in the inference loop)
  nvtracker          NvDCF object tracker (persistent IDs across frames)
  nvdsosd            GPU-side bounding-box / label drawing
  nvrtspoutsinkbin   RTSP output  →  iPad VLC

Pipeline
--------
  Source
    → nvstreammux
    → nvinfer  (COCO: vehicles, persons, …)
    → nvinfer  (LISA: traffic signs)
    → nvtracker
    → nvdsosd
    → tee ─┬─ queue → nvrtspoutsinkbin   (RTSP → iPad)
            └─ queue → fakesink + probe   (UDP JSON → AR HUD)

Requirements on Jetson
----------------------
  1. DeepStream 7.x  (JetPack 6.x)
  2. pyds Python bindings  (part of DeepStream SDK)
  3. libnvdsinfer_custom_impl_Yolo.so  (build once via build_parser.sh)
  4. Docker image:  dustynv/deepstream:7.0-r36.4.0

Usage
-----
  python3 deepstream/ds_pipeline.py \\
      --source test_data/video_test1.mp4 \\
      --coco-engine models/yolov8n.engine \\
      --lisa-engine models/yolov8n_lisa_v1.1.engine \\
      --rtsp-port 8554

  # Live CSI camera + UDP HUD
  python3 deepstream/ds_pipeline.py \\
      --source csi \\
      --width 1920 --height 1080 \\
      --coco-engine models/yolov8n.engine \\
      --lisa-engine models/yolov8n_lisa_v1.1.engine \\
      --rtsp-port 8554 \\
      --udp-host 192.168.4.50 --udp-port 5055
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib

# Try multiple pyds locations (DeepStream installs it here on Jetson)
for _p in ["/opt/nvidia/deepstream/deepstream/lib",
           "/usr/lib/python3/dist-packages"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pyds
    _PYDS_AVAILABLE = True
except ImportError:
    _PYDS_AVAILABLE = False

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ── paths ─────────────────────────────────────────────────────────────────────

HERE        = Path(__file__).resolve().parent
CONFIGS     = HERE / "configs"
LABELS      = HERE / "labels"
TRACKER_LIB = (
    "/opt/nvidia/deepstream/deepstream/lib"
    "/libnvds_nvmultiobjecttracker.so"
)
PARSER_LIB  = str(CONFIGS / "libnvdsinfer_custom_impl_Yolo.so")

LISA_STOP_IDS = {3, 4}   # stop, stopLeft
LISA_GO_IDS   = {0, 1, 2} # go, goForward, goLeft


# ── STOP/GO shared state ───────────────────────────────────────────────────────

class _DriveState:
    """Thread-safe STOP/GO state updated by probe, read by overlay thread."""
    def __init__(self):
        self._lock  = threading.Lock()
        self._state = None          # None | "STOP" | "GO"
        self._ts    = 0.0           # timestamp of last update

    def update(self, state: str | None) -> None:
        with self._lock:
            self._state = state
            self._ts    = time.time()

    def get(self) -> str | None:
        with self._lock:
            # Clear state after 1.5 s of no detections
            if self._state and (time.time() - self._ts) > 1.5:
                self._state = None
            return self._state


# ── OpenCV STOP/GO overlay thread ─────────────────────────────────────────────

def _run_stopgo_overlay(state: _DriveState, stop_event: threading.Event) -> None:
    """Runs in background thread — draws STOP/GO indicator using OpenCV."""
    if not _CV2_AVAILABLE:
        print("[INFO] OpenCV not available — STOP/GO overlay disabled.")
        return

    WIN = "Driver Alert"
    W, H = 260, 120
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.moveWindow(WIN, 20, 20)   # top-left corner of desktop

    COLORS = {
        "STOP": (0,   0,   220),   # red   (BGR)
        "GO":   (0,   180, 0  ),   # green (BGR)
        None:   (40,  40,  40 ),   # dark grey
    }
    LABELS = {"STOP": "STOP", "GO": "GO", None: "---"}

    while not stop_event.is_set():
        s = state.get()
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = COLORS[s]

        label = LABELS[s]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2.4 if s else 1.2
        thick = 5    if s else 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx = (W - tw) // 2
        ty = (H + th) // 2
        cv2.putText(canvas, label, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

COCO_GIE_ID = 1
LISA_GIE_ID = 2


# ── element factory ───────────────────────────────────────────────────────────

def _make(factory: str, name: str) -> Gst.Element:
    elm = Gst.ElementFactory.make(factory, name)
    if elm is None:
        raise RuntimeError(
            f"Cannot create GStreamer element '{factory}'. "
            "Is DeepStream installed and NVMM plugins available?"
        )
    return elm


# ── source builder ────────────────────────────────────────────────────────────

def _add_source(
    pipeline: Gst.Pipeline,
    source: str,
    width: int,
    height: int,
    mux: Gst.Element,
) -> None:
    """Create source elements, add them to pipeline, and link to mux.sink_0."""
    sink_pad = mux.get_request_pad("sink_0")

    # ── RPi Camera V2 via CSI ribbon cable ────────────────────────────
    if source.lower().startswith("csi"):
        sensor = int(source.split(":")[1]) if ":" in source else 0
        src  = _make("nvarguscamerasrc", "src")
        caps = _make("capsfilter", "src_caps")
        src.set_property("sensor-id", sensor)
        caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,"
                f"width={width},height={height},framerate=30/1"
            ),
        )
        pipeline.add(src, caps)
        src.link(caps)
        caps.get_static_pad("src").link(sink_pad)
        return

    # ── USB camera / V4L2 ─────────────────────────────────────────────
    if source.isdigit() or source.startswith("/dev/video"):
        dev  = source if source.startswith("/dev/") else f"/dev/video{source}"
        src  = _make("v4l2src",       "src")
        conv = _make("nvvideoconvert","src_conv")
        caps = _make("capsfilter",    "src_caps")
        src.set_property("device", dev)
        caps.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,"
                f"width={width},height={height},framerate=30/1"
            ),
        )
        pipeline.add(src, conv, caps)
        src.link(conv)
        conv.link(caps)
        caps.get_static_pad("src").link(sink_pad)
        return

    # ── RTSP network stream ───────────────────────────────────────────
    if source.startswith("rtsp://") or source.startswith("rtsps://"):
        src  = _make("rtspsrc",       "src")
        dep  = _make("rtph264depay",  "rtpdep")
        par  = _make("h264parse",     "h264parse")
        dec  = _make("nvv4l2decoder", "decoder")
        conv = _make("nvvideoconvert","src_conv")
        caps = _make("capsfilter",    "src_caps")
        src.set_property("location", source)
        src.set_property("latency", 100)
        caps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"),
        )
        pipeline.add(src, dep, par, dec, conv, caps)

        def _rtspsrc_pad_added(s, pad):
            sink = dep.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)

        src.connect("pad-added", _rtspsrc_pad_added)
        dep.link(par)
        par.link(dec)
        dec.link(conv)
        conv.link(caps)
        caps.get_static_pad("src").link(sink_pad)
        return

    # ── Local MP4 / H.264 file ────────────────────────────────────────
    src   = _make("filesrc",      "src")
    demux = _make("qtdemux",      "demux")
    par   = _make("h264parse",    "h264parse")
    dec   = _make("nvv4l2decoder","decoder")
    conv  = _make("nvvideoconvert", "src_conv")
    src.set_property("location", os.path.abspath(source))
    pipeline.add(src, demux, par, dec, conv)
    src.link(demux)

    def _demux_pad_added(s, pad):
        if "video" in pad.get_name():
            sink = par.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)

    demux.connect("pad-added", _demux_pad_added)
    par.link(dec)
    dec.link(conv)
    conv.get_static_pad("src").link(sink_pad)


# ── config patcher ────────────────────────────────────────────────────────────

def _patch_config(cfg_path: str, updates: dict[str, str]) -> None:
    """Update key=value lines in a DeepStream .txt config at runtime."""
    with open(cfg_path) as f:
        lines = f.readlines()
    patched = []
    for line in lines:
        key = line.split("=")[0].strip() if "=" in line else ""
        if key in updates:
            patched.append(f"{key}={updates[key]}\n")
        else:
            patched.append(line)
    with open(cfg_path, "w") as f:
        f.writelines(patched)


# ── labels loader ─────────────────────────────────────────────────────────────

def _load_labels(path: Path) -> dict[int, str]:
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return {i: ln for i, ln in enumerate(lines)}


# ── bus callback ──────────────────────────────────────────────────────────────

def _bus_call(bus, message, loop: GLib.MainLoop) -> bool:
    t = message.type
    if t == Gst.MessageType.EOS:
        print("\n[DS] End of stream.")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"[DS] Error: {err}\n      Debug: {dbg}")
        loop.quit()
    return True


# ── probe (reads NvDsBatchMeta → publishes UDP JSON) ─────────────────────────

def _make_probe(
    udp_sock: socket.socket | None,
    udp_host: str,
    udp_port: int,
    coco_labels: dict,
    lisa_labels: dict,
    drive_state: _DriveState,
):
    seq = [0]

    def probe_fn(pad, info, _):
        if not _PYDS_AVAILABLE:
            return Gst.PadProbeReturn.OK

        gst_buf = info.get_buffer()
        if gst_buf is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buf))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        detections = []
        l_frame = batch_meta.frame_meta_list

        while l_frame:
            try:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = fm.obj_meta_list
            while l_obj:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                r   = obj.rect_params
                gie = obj.unique_component_id
                lbl_map = coco_labels if gie == COCO_GIE_ID else lisa_labels
                lbl = lbl_map.get(obj.class_id, str(obj.class_id))

                # Normalise coordinates to [0,1] using frame dims
                fw = fm.source_frame_width  or 1
                fh = fm.source_frame_height or 1

                detections.append({
                    "class_id":   obj.class_id,
                    "label":      lbl,
                    "confidence": round(float(obj.confidence), 4),
                    "x_center":   round(float(r.left + r.width  / 2) / fw, 5),
                    "y_center":   round(float(r.top  + r.height / 2) / fh, 5),
                    "width":      round(float(r.width)  / fw, 5),
                    "height":     round(float(r.height) / fh, 5),
                    "tracker_id": int(obj.object_id),
                    "source_id":  gie,
                    "frame_num":  fm.frame_num,
                })

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        # ── update STOP/GO overlay state ──────────────────────────────
        lisa_ids = {d["class_id"] for d in detections if d["source_id"] == LISA_GIE_ID}
        if lisa_ids & LISA_STOP_IDS:
            drive_state.update("STOP")
        elif lisa_ids & LISA_GO_IDS:
            drive_state.update("GO")
        else:
            drive_state.update(None)

        if udp_sock and detections:
            payload = json.dumps({
                "schema_version": 1,
                "timestamp_ms":   int(time.time() * 1000),
                "sequence":       seq[0],
                "detections":     detections,
            })
            try:
                udp_sock.sendto(payload.encode(), (udp_host, udp_port))
            except Exception:
                pass

        seq[0] += 1
        return Gst.PadProbeReturn.OK

    return probe_fn


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full DeepStream 7.x pipeline: nvinfer + nvtracker + nvdsosd"
    )
    p.add_argument("--source", required=True,
        help="csi | csi:1 | 0 | /dev/video0 | rtsp://... | file.mp4")
    p.add_argument("--coco-engine",  default="models/yolov8n.engine")
    p.add_argument("--lisa-engine",  default="models/yolov8n_lisa_v1.1.engine")
    p.add_argument("--width",        type=int, default=1920)
    p.add_argument("--height",       type=int, default=1080)
    p.add_argument("--output",       default="display",
        choices=["display", "rtsp"],
        help="Output mode: display (monitor) or rtsp (iPad). Default: display.")
    p.add_argument("--rtsp-port",    type=int, default=8554,
        help="RTSP port (only used when --output rtsp). Default 8554.")
    p.add_argument("--udp-host",     default="",
        help="UDP host for AR HUD. Leave empty to disable.")
    p.add_argument("--udp-port",     type=int, default=5055)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    Gst.init(None)

    # ── pyds check ────────────────────────────────────────────────────
    if not _PYDS_AVAILABLE:
        print("[INFO] pyds not found — UDP HUD output disabled.")
        if args.udp_host:
            print("[WARN] --udp-host ignored (pyds required for JSON output).")

    # ── labels ────────────────────────────────────────────────────────
    coco_labels = _load_labels(LABELS / "coco.txt")
    lisa_labels = _load_labels(LABELS / "lisa.txt")
    print(f"COCO classes : {len(coco_labels)}")
    print(f"LISA classes : {list(lisa_labels.values())}")

    # ── patch config files with absolute engine + parser paths ────────
    coco_cfg    = str(CONFIGS / "pgie_coco.txt")
    lisa_cfg    = str(CONFIGS / "pgie_lisa.txt")
    tracker_cfg = str(CONFIGS / "tracker_nvdcf.txt")

    # Derive absolute ONNX paths (same dir as engine, strip _deepstream suffix)
    coco_engine_path = Path(os.path.abspath(args.coco_engine))
    lisa_engine_path = Path(os.path.abspath(args.lisa_engine))
    coco_onnx = coco_engine_path.parent / (coco_engine_path.stem.replace("_deepstream", "") + ".onnx")
    lisa_onnx = lisa_engine_path.parent / (lisa_engine_path.stem.replace("_deepstream", "") + ".onnx")

    _patch_config(coco_cfg, {
        "onnx-file":         str(coco_onnx),
        "model-engine-file": str(coco_engine_path),
        "custom-lib-path":   PARSER_LIB,
        "labelfile-path":    str(LABELS / "coco.txt"),
    })
    _patch_config(lisa_cfg, {
        "onnx-file":         str(lisa_onnx),
        "model-engine-file": str(lisa_engine_path),
        "custom-lib-path":   PARSER_LIB,
        "labelfile-path":    str(LABELS / "lisa.txt"),
    })

    # ── build pipeline ────────────────────────────────────────────────
    pipeline = Gst.Pipeline.new("ds-pipeline")
    is_live  = (
        args.source.lower().startswith("csi")
        or args.source.isdigit()
        or args.source.startswith("/dev/video")
    )

    # nvstreammux
    mux = _make("nvstreammux", "mux")
    mux.set_property("batch-size",            1)
    mux.set_property("width",                 args.width)
    mux.set_property("height",                args.height)
    mux.set_property("batched-push-timeout",  33333)
    mux.set_property("live-source",           int(is_live))

    # nvinfer × 2
    pgie = _make("nvinfer", "pgie")
    sgie = _make("nvinfer", "sgie")
    pgie.set_property("config-file-path", coco_cfg)
    sgie.set_property("config-file-path", lisa_cfg)

    # nvtracker
    tracker = _make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file",          TRACKER_LIB)
    tracker.set_property("ll-config-file",        tracker_cfg)
    tracker.set_property("tracker-width",         640)
    tracker.set_property("tracker-height",        384)
    tracker.set_property("display-tracking-id",   1)

    # nvdsosd
    osd = _make("nvdsosd", "osd")
    osd.set_property("process-mode",  1)
    osd.set_property("display-text",  1)

    # ── output sink ───────────────────────────────────────────────────
    if args.output == "rtsp":
        q_out    = _make("queue",            "q_out")
        sink     = _make("nvrtspoutsinkbin", "sink")
        sink.set_property("rtsp-port", args.rtsp_port)
        sink.set_property("bitrate",   2000000)
    else:
        # display mode — render directly on HDMI monitor
        q_out = _make("queue",          "q_out")
        sink  = _make("nveglglessink",  "sink")
        sink.set_property("sync",         False)
        sink.set_property("window-x",     0)
        sink.set_property("window-y",     0)
        sink.set_property("window-width",  args.width)
        sink.set_property("window-height", args.height)

    # add all to pipeline
    for elm in (mux, pgie, sgie, tracker, osd, q_out, sink):
        pipeline.add(elm)

    # link chain
    mux.link(pgie)
    pgie.link(sgie)
    sgie.link(tracker)
    tracker.link(osd)
    osd.link(q_out)
    q_out.link(sink)

    # source → mux
    _add_source(pipeline, args.source, args.width, args.height, mux)

    # ── STOP/GO state + overlay thread ────────────────────────────────
    drive_state = _DriveState()
    stop_event  = threading.Event()
    overlay_thread = threading.Thread(
        target=_run_stopgo_overlay,
        args=(drive_state, stop_event),
        daemon=True,
    )
    overlay_thread.start()
    if _CV2_AVAILABLE:
        print("Driver alert → STOP/GO overlay window active")
    else:
        print("[INFO] OpenCV not found — STOP/GO overlay disabled.")

    # ── probe on osd sink pad → drive state + UDP JSON ────────────────
    udp_sock: socket.socket | None = None
    if args.udp_host:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP HUD     → {args.udp_host}:{args.udp_port}")

    osd.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        _make_probe(udp_sock, args.udp_host, args.udp_port,
                    coco_labels, lisa_labels, drive_state),
        0,
    )

    # ── bus ───────────────────────────────────────────────────────────
    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _bus_call, loop)

    # ── start ─────────────────────────────────────────────────────────
    print(f"\nStarting DeepStream pipeline...")
    if args.output == "rtsp":
        print(f"RTSP ready  → rtsp://0.0.0.0:{args.rtsp_port}/ds-test")
        print(f"iPad VLC    → rtsp://10.0.0.119:{args.rtsp_port}/ds-test")
    else:
        print("Output      → HDMI monitor (nveglglessink)")
    print("Press Ctrl+C to stop.\n")

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        overlay_thread.join(timeout=2)
        pipeline.set_state(Gst.State.NULL)
        if udp_sock:
            udp_sock.close()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
