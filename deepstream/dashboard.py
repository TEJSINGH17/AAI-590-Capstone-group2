"""
deepstream/dashboard.py — PyQt5 Dashboard for DeepStream Autonomous Driving Pipeline
=====================================================================================

Two-panel dashboard:
  Left  — Annotated camera / MP4 video via nveglglessink embedded in Qt
  Right — STOP/GO driver alert updated in real-time from LISA detections

Usage (same args as ds_pipeline.py):
  python3 deepstream/dashboard.py --source test_data/video_test1.mp4
  python3 deepstream/dashboard.py --source /dev/video0
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import gi
gi.require_version("Gst",      "1.0")
gi.require_version("GLib",     "2.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GLib, GstVideo

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QFrame, QSizePolicy,
)
from PyQt5.QtCore  import Qt, QTimer
from PyQt5.QtGui   import QFont, QPalette, QColor

# ── try optional deps ──────────────────────────────────────────────────────────
for _p in ["/opt/nvidia/deepstream/deepstream/lib",
           "/usr/lib/python3/dist-packages"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pyds
    _PYDS_AVAILABLE = True
except ImportError:
    _PYDS_AVAILABLE = False

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
CONFIGS     = HERE / "configs"
LABELS      = HERE / "labels"
TRACKER_LIB = (
    "/opt/nvidia/deepstream/deepstream/lib"
    "/libnvds_nvmultiobjecttracker.so"
)
PARSER_LIB  = str(CONFIGS / "libnvdsinfer_custom_impl_Yolo.so")

COCO_GIE_ID   = 1
LISA_GIE_ID   = 2
LISA_STOP_IDS = {3, 4}    # stop, stopLeft
LISA_GO_IDS   = {0, 1, 2} # go, goForward, goLeft

# ── helpers (same as ds_pipeline.py) ──────────────────────────────────────────

def _make(factory: str, name: str) -> Gst.Element:
    elm = Gst.ElementFactory.make(factory, name)
    if elm is None:
        raise RuntimeError(f"Cannot create GStreamer element '{factory}'.")
    return elm


def _patch_config(cfg_path: str, updates: dict[str, str]) -> None:
    with open(cfg_path) as f:
        lines = f.readlines()
    patched = []
    for line in lines:
        key = line.split("=")[0].strip() if "=" in line else ""
        patched.append(f"{key}={updates[key]}\n" if key in updates else line)
    with open(cfg_path, "w") as f:
        f.writelines(patched)


def _load_labels(path: Path) -> dict[int, str]:
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return {i: ln for i, ln in enumerate(lines)}


def _add_source(pipeline, source, width, height, mux):
    sink_pad = mux.get_request_pad("sink_0")

    if source.lower().startswith("csi"):
        sensor = int(source.split(":")[1]) if ":" in source else 0
        src  = _make("nvarguscamerasrc", "src")
        caps = _make("capsfilter",       "src_caps")
        src.set_property("sensor-id", sensor)
        caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),format=NV12,"
            f"width={width},height={height},framerate=30/1"))
        pipeline.add(src, caps)
        src.link(caps)
        caps.get_static_pad("src").link(sink_pad)
        return

    if source.isdigit() or source.startswith("/dev/video"):
        dev  = source if source.startswith("/dev/") else f"/dev/video{source}"
        src  = _make("v4l2src",        "src")
        conv = _make("nvvideoconvert", "src_conv")
        caps = _make("capsfilter",     "src_caps")
        src.set_property("device", dev)
        caps.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),format=NV12,"
            f"width={width},height={height},framerate=30/1"))
        pipeline.add(src, conv, caps)
        src.link(conv); conv.link(caps)
        caps.get_static_pad("src").link(sink_pad)
        return

    if source.startswith("rtsp://") or source.startswith("rtsps://"):
        src  = _make("rtspsrc",       "src")
        dep  = _make("rtph264depay",  "rtpdep")
        par  = _make("h264parse",     "h264parse")
        dec  = _make("nvv4l2decoder", "decoder")
        conv = _make("nvvideoconvert","src_conv")
        caps = _make("capsfilter",    "src_caps")
        src.set_property("location", source)
        src.set_property("latency",  100)
        caps.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=NV12"))
        pipeline.add(src, dep, par, dec, conv, caps)
        def _rtspsrc_pad_added(s, pad):
            sink = dep.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)
        src.connect("pad-added", _rtspsrc_pad_added)
        dep.link(par); par.link(dec); dec.link(conv); conv.link(caps)
        caps.get_static_pad("src").link(sink_pad)
        return

    # MP4
    src   = _make("filesrc",       "src")
    demux = _make("qtdemux",       "demux")
    par   = _make("h264parse",     "h264parse")
    dec   = _make("nvv4l2decoder", "decoder")
    conv  = _make("nvvideoconvert","src_conv")
    src.set_property("location", os.path.abspath(source))
    pipeline.add(src, demux, par, dec, conv)
    src.link(demux)
    def _demux_pad_added(s, pad):
        if "video" in pad.get_name():
            sink = par.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)
    demux.connect("pad-added", _demux_pad_added)
    par.link(dec); dec.link(conv)
    conv.get_static_pad("src").link(sink_pad)


# ── STOP/GO state ──────────────────────────────────────────────────────────────

class _DriveState:
    def __init__(self):
        self._lock  = threading.Lock()
        self._state = None
        self._label = ""
        self._conf  = 0.0
        self._ts    = 0.0

    def update(self, state, label="", conf=0.0):
        with self._lock:
            self._state = state
            self._label = label
            self._conf  = conf
            self._ts    = time.time()

    def get(self):
        with self._lock:
            if self._state and (time.time() - self._ts) > 1.5:
                self._state = None
                self._label = ""
                self._conf  = 0.0
            return self._state, self._label, self._conf


# ── GStreamer probe ────────────────────────────────────────────────────────────

def _make_probe(coco_labels, lisa_labels, drive_state):
    def probe_fn(pad, info, _):
        if not _PYDS_AVAILABLE:
            return Gst.PadProbeReturn.OK
        gst_buf = info.get_buffer()
        if gst_buf is None:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buf))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        best_stop = None   # (conf, label)
        best_go   = None

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
                if obj.unique_component_id == LISA_GIE_ID:
                    lbl  = lisa_labels.get(obj.class_id, str(obj.class_id))
                    conf = float(obj.confidence)
                    if obj.class_id in LISA_STOP_IDS:
                        if best_stop is None or conf > best_stop[0]:
                            best_stop = (conf, lbl)
                    elif obj.class_id in LISA_GO_IDS:
                        if best_go is None or conf > best_go[0]:
                            best_go = (conf, lbl)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if best_stop:
            drive_state.update("STOP", best_stop[1], best_stop[0])
        elif best_go:
            drive_state.update("GO",   best_go[1],   best_go[0])
        else:
            drive_state.update(None)

        return Gst.PadProbeReturn.OK
    return probe_fn


# ── ultralytics LISA inference thread ─────────────────────────────────────────

def _run_lisa_thread(appsink, lisa_pt_path, drive_state, stop_event):
    """
    Two-sub-thread design to eliminate lag:
      - Puller : always stores the LATEST frame from appsink (drops old ones)
      - Inferrer: runs ultralytics on the latest frame as fast as possible
    """
    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        print("[WARN] ultralytics not available — STOP/GO alert disabled.")
        return

    if not Path(lisa_pt_path).exists():
        print(f"[WARN] LISA .pt not found at {lisa_pt_path} — STOP/GO alert disabled.")
        return

    # Shared latest frame
    latest_frame      = [None]
    latest_frame_lock = threading.Lock()
    frame_ready       = threading.Event()

    def puller():
        """Continuously pulls latest frame from appsink."""
        while not stop_event.is_set():
            sample = appsink.emit("try-pull-sample", 100 * Gst.MSECOND)
            if sample is None:
                continue
            buf  = sample.get_buffer()
            caps = sample.get_caps()
            h    = caps.get_structure(0).get_value("height")
            w    = caps.get_structure(0).get_value("width")
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                frame = (np.frombuffer(mapinfo.data, dtype=np.uint8)
                         .reshape(h, w, 4)[:, :, :3].copy())  # BGRx → BGR
            finally:
                buf.unmap(mapinfo)
            with latest_frame_lock:
                latest_frame[0] = frame
            frame_ready.set()

    model = YOLO(lisa_pt_path)
    print("[Dashboard] LISA model loaded.")

    t_pull = threading.Thread(target=puller, daemon=True)
    t_pull.start()

    while not stop_event.is_set():
        frame_ready.wait(timeout=0.5)
        frame_ready.clear()

        with latest_frame_lock:
            frame = latest_frame[0]
        if frame is None:
            continue

        results = model(frame, verbose=False, device="cpu", imgsz=320)[0]

        best_stop, best_go = None, None
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in LISA_STOP_IDS:
                if best_stop is None or conf > best_stop[0]:
                    best_stop = (conf, results.names[cls])
            elif cls in LISA_GO_IDS:
                if best_go is None or conf > best_go[0]:
                    best_go = (conf, results.names[cls])

        if best_stop:
            drive_state.update("STOP", best_stop[1], best_stop[0])
        elif best_go:
            drive_state.update("GO",   best_go[1],   best_go[0])
        else:
            drive_state.update(None)


# ── Qt Widgets ─────────────────────────────────────────────────────────────────

class VideoWidget(QWidget):
    """Hosts nveglglessink output via X11 window handle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sink = None
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Overlay message (shown when no source is available)
        self._msg_label = QLabel(self)
        self._msg_label.setAlignment(Qt.AlignCenter)
        self._msg_label.setFont(QFont("Arial", 22, QFont.Bold))
        self._msg_label.setStyleSheet("color: #888888; background-color: transparent;")
        self._msg_label.setWordWrap(True)
        self._msg_label.hide()

    def show_message(self, text: str):
        self._msg_label.setText(text)
        self._msg_label.show()
        self._msg_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._msg_label.setGeometry(0, 0, self.width(), self.height())
        self._apply_handle()

    def set_sink(self, sink):
        self._sink = sink

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_handle()

    def _apply_handle(self):
        if self._sink and self.winId():
            GstVideo.VideoOverlay.set_window_handle(self._sink, int(self.winId()))


class AlertPanel(QWidget):
    """Right-side STOP/GO alert panel."""

    def __init__(self, drive_state: _DriveState, parent=None):
        super().__init__(parent)
        self._state = drive_state
        self.setFixedWidth(260)
        self.setStyleSheet("background-color: #1a1a1a;")
        self._build_ui()

        timer = QTimer(self)
        timer.timeout.connect(self._refresh)
        timer.start(100)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        title = QLabel("DRIVER ALERT")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color: #aaaaaa; letter-spacing: 2px;")
        layout.addWidget(title)

        self.alert_box = QLabel("---")
        self.alert_box.setAlignment(Qt.AlignCenter)
        self.alert_box.setFont(QFont("Arial", 52, QFont.Bold))
        self.alert_box.setFixedSize(220, 130)
        self.alert_box.setStyleSheet(
            "background-color: #333333; color: #666666; border-radius: 12px;")
        layout.addWidget(self.alert_box, alignment=Qt.AlignCenter)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        self.class_label = QLabel("Class: —")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setFont(QFont("Arial", 11))
        self.class_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.class_label)

        self.conf_label = QLabel("Conf: —")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setFont(QFont("Arial", 11))
        self.conf_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.conf_label)

        layout.addStretch()

        footer = QLabel("YOLOv8 + DeepStream")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFont(QFont("Arial", 9))
        footer.setStyleSheet("color: #444444;")
        layout.addWidget(footer)

    def _refresh(self):
        state, label, conf = self._state.get()
        if state == "STOP":
            self.alert_box.setText("STOP")
            self.alert_box.setStyleSheet(
                "background-color: #cc0000; color: white; border-radius: 12px;")
        elif state == "GO":
            self.alert_box.setText("GO")
            self.alert_box.setStyleSheet(
                "background-color: #00aa00; color: white; border-radius: 12px;")
        else:
            self.alert_box.setText("---")
            self.alert_box.setStyleSheet(
                "background-color: #333333; color: #666666; border-radius: 12px;")

        self.class_label.setText(f"Class: {label or '—'}")
        self.conf_label.setText(
            f"Conf:  {conf:.2f}" if conf else "Conf:  —")


class Dashboard(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Autonomous Driving — DeepStream Dashboard")
        self.setStyleSheet("background-color: #111111;")

        self._drive_state = _DriveState()
        self._video_widget = VideoWidget()
        self._alert_panel  = AlertPanel(self._drive_state)

        central = QWidget()
        layout  = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._video_widget)
        layout.addWidget(self._alert_panel)
        self.setCentralWidget(central)

        self.showFullScreen()

        # Build pipeline after window is shown
        QTimer.singleShot(500, lambda: self._start_pipeline(args))

    def _start_pipeline(self, args):
        # No source available — show message and skip pipeline
        if args.source == "none":
            self._video_widget.show_message(
                "No video source found.\n\n"
                "Connect the EMEET C950 camera\n"
                "or add a test MP4 file, then restart.")
            return

        Gst.init(None)

        coco_cfg    = str(CONFIGS / "pgie_coco.txt")
        lisa_cfg    = str(CONFIGS / "pgie_lisa.txt")
        tracker_cfg = str(CONFIGS / "tracker_nvdcf.txt")

        coco_engine_path = Path(os.path.abspath(args.coco_engine))
        lisa_engine_path = Path(os.path.abspath(args.lisa_engine))
        coco_onnx = coco_engine_path.parent / (
            coco_engine_path.stem.replace("_deepstream", "") + ".onnx")
        lisa_onnx = lisa_engine_path.parent / (
            lisa_engine_path.stem.replace("_deepstream", "") + ".onnx")

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

        pipeline = Gst.Pipeline.new("ds-dashboard")
        is_live  = (
            args.source.lower().startswith("csi")
            or args.source.isdigit()
            or args.source.startswith("/dev/video")
        )

        mux = _make("nvstreammux", "mux")
        mux.set_property("batch-size",           1)
        mux.set_property("width",                args.width)
        mux.set_property("height",               args.height)
        mux.set_property("batched-push-timeout", 33333)
        mux.set_property("live-source",          int(is_live))

        pgie = _make("nvinfer",  "pgie")
        sgie = _make("nvinfer",  "sgie")
        pgie.set_property("config-file-path", coco_cfg)
        sgie.set_property("config-file-path", lisa_cfg)

        tracker = _make("nvtracker", "tracker")
        tracker.set_property("ll-lib-file",        TRACKER_LIB)
        tracker.set_property("ll-config-file",     tracker_cfg)
        tracker.set_property("tracker-width",      640)
        tracker.set_property("tracker-height",     384)
        tracker.set_property("display-tracking-id", 1)

        osd = _make("nvdsosd", "osd")
        osd.set_property("process-mode", 1)
        osd.set_property("display-text", 1)

        q_out = _make("queue",         "q_out")
        sink  = _make("nveglglessink", "sink")
        sink.set_property("sync", False)

        # appsink branch for LISA inference (tee after mux)
        tee       = _make("tee",           "tee")
        q_main    = _make("queue",         "q_main")
        q_app     = _make("queue",         "q_app")
        conv_app  = _make("nvvideoconvert","conv_app")
        caps_app  = _make("capsfilter",    "caps_app")
        appsink   = _make("appsink",       "appsink")

        caps_app.set_property("caps", Gst.Caps.from_string(
            "video/x-raw,format=BGRx,width=640,height=360"))
        appsink.set_property("emit-signals",  True)
        appsink.set_property("max-buffers",   1)
        appsink.set_property("drop",          True)
        appsink.set_property("sync",          False)

        for elm in (mux, tee, q_main, pgie, sgie, tracker, osd,
                    q_out, sink, q_app, conv_app, caps_app, appsink):
            pipeline.add(elm)

        # main branch: mux → tee → q_main → pgie → sgie → tracker → osd → q_out → sink
        mux.link(tee)
        tee.get_request_pad("src_%u").link(q_main.get_static_pad("sink"))
        q_main.link(pgie); pgie.link(sgie); sgie.link(tracker)
        tracker.link(osd); osd.link(q_out); q_out.link(sink)

        # appsink branch: tee → q_app → conv_app → caps_app → appsink
        tee.get_request_pad("src_%u").link(q_app.get_static_pad("sink"))
        q_app.link(conv_app); conv_app.link(caps_app); caps_app.link(appsink)

        _add_source(pipeline, args.source, args.width, args.height, mux)

        # Attach video widget as render target
        self._video_widget.set_sink(sink)
        GstVideo.VideoOverlay.set_window_handle(sink, int(self._video_widget.winId()))

        # Launch ultralytics LISA inference thread
        self._stop_event = threading.Event()
        lisa_thread = threading.Thread(
            target=_run_lisa_thread,
            args=(appsink, args.lisa_engine.replace("_deepstream.engine", ".pt"),
                  self._drive_state, self._stop_event),
            daemon=True,
        )
        lisa_thread.start()

        # Bus
        loop = GLib.MainLoop()
        bus  = pipeline.get_bus()
        bus.add_signal_watch()

        def on_message(bus, msg):
            if msg.type == Gst.MessageType.EOS:
                loop.quit()
                QApplication.instance().quit()
            elif msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"[DS] Error: {err}\n     Debug: {dbg}")
                loop.quit()
                QApplication.instance().quit()
            elif GstVideo.is_video_overlay_prepare_window_handle_message(msg):
                GstVideo.VideoOverlay.set_window_handle(
                    msg.src, int(self._video_widget.winId()))

        bus.connect("message", on_message)

        pipeline.set_state(Gst.State.PLAYING)
        self._pipeline = pipeline

        # Run GLib loop in background thread
        self._glib_loop = loop
        t = threading.Thread(target=loop.run, daemon=True)
        t.start()

        print("[Dashboard] Pipeline running.")
        print("[Dashboard] LISA alert thread running (ultralytics CPU).")

    def closeEvent(self, event):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "_pipeline"):
            self._pipeline.set_state(Gst.State.NULL)
        if hasattr(self, "_glib_loop"):
            self._glib_loop.quit()
        super().closeEvent(event)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="PyQt5 DeepStream Dashboard")
    p.add_argument("--source",       required=True,
        help="csi | 0 | /dev/video0 | file.mp4")
    p.add_argument("--coco-engine",  default="models/yolov8n_deepstream.engine")
    p.add_argument("--lisa-engine",  default="models/yolov8n_lisa_v1.1_deepstream.engine")
    p.add_argument("--width",        type=int, default=1920)
    p.add_argument("--height",       type=int, default=1080)
    return p.parse_args()


def main():
    args = _parse_args()
    app  = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window,    QColor(17,  17,  17))
    palette.setColor(QPalette.WindowText,QColor(220, 220, 220))
    app.setPalette(palette)

    Dashboard(args)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
