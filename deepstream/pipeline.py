"""
deepstream/pipeline.py  --  GStreamer + TensorRT Inference Pipeline
===================================================================

Runs on NVIDIA Jetson Orin Nano (JetPack 6.x).

Uses GStreamer for hardware-accelerated video decode (nvv4l2decoder /
nvarguscamerasrc / v4l2src) and feeds BGR frames to both YOLOv8
TensorRT .engine models via the ultralytics API.  Detection payloads
are published over UDP to the AR HUD consumer, optionally saved to a
JSON file, drawn on an annotated output video, and/or streamed live
as an RTSP feed viewable in VLC on an iPad / any RTSP client.

Sources supported
-----------------
  --source 0               USB/CSI camera by V4L2 index
  --source /dev/video0     V4L2 device path
  --source rtsp://...      RTSP stream (H.264)
  --source video.mp4       Local MP4 file

Examples
--------
  # USB camera -> RTSP stream (view in VLC on iPad)
  python3 deepstream/pipeline.py \\
      --source 0 \\
      --model  models/yolov8n.engine \\
      --model2 models/yolov8n_lisa_v1.1.engine \\
      --rtsp-port 8554 --no-show
  # Then open VLC on iPad -> Network -> rtsp://JETSON_IP:8554/live

  # USB camera -> RTSP + UDP HUD (fully headless)
  python3 deepstream/pipeline.py \\
      --source 0 \\
      --model  models/yolov8n.engine \\
      --model2 models/yolov8n_lisa_v1.1.engine \\
      --rtsp-port 8554 --udp-host 192.168.1.50 --no-show

  # Local file -> annotated output + JSON detections
  python3 deepstream/pipeline.py \\
      --source test_data/output_ds_1.mp4 \\
      --model  models/yolov8n.engine \\
      --model2 models/yolov8n_lisa_v1.1.engine \\
      --output runs/annotated/deepstream_out.mp4 \\
      --json   runs/json/deepstream_detections.json \\
      --no-show

Notes
-----
* Run from the project root so relative model/output paths resolve.
* .engine files must be exported ON this Jetson (device-specific).
* If GStreamer is unavailable the pipeline falls back to
  cv2.VideoCapture so you can test on a dev machine without JetPack.
* RTSP encoder: uses nvv4l2h264enc (Jetson HW). Pass
  --rtsp-encoder x264enc to use software encoding on a dev machine.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from application.publish_to_hud import (  # noqa: E402
    DetectionPayload,
    UdpPublisher,
    build_message,
)


# ── colour scheme (BGR) ──────────────────────────────────────────────────────

COCO_BOX = (255, 140, 0)
_LISA_COLORS = {
    "go": (50, 230, 80),
    "goforward": (50, 230, 80),
    "goleft": (80, 200, 255),
    "stop": (50, 50, 255),
    "stopleft": (80, 80, 255),
    "warning": (0, 210, 255),
    "warningleft": (0, 185, 230),
}
_LISA_DEFAULT = (0, 200, 255)


def _lisa_color(label: str) -> Tuple[int, int, int]:
    key = label.strip().lower().replace(" ", "")
    return _LISA_COLORS.get(key, _LISA_DEFAULT)


# ── drawing helpers ───────────────────────────────────────────────────────────

def _draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    labels: List[str],
    box_color,
    label_fn=None,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        label = labels[i]
        conf = confs[i]
        color = label_fn(label) if label_fn else box_color
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA
        )
        txt = f"{label}  {conf * 100:.0f}%"
        (tw, th), bl = cv2.getTextSize(txt, font, scale, 1)
        by1 = max(0, y1 - th - bl - 6)
        cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame, txt, (x1 + 4, y1 - bl - 2),
            font, scale, (10, 10, 10), 1, cv2.LINE_AA,
        )


def _draw_overlay(
    frame: np.ndarray,
    model1_name: str,
    model2_name: Optional[str],
    fps: float,
) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    items = [(COCO_BOX, f"M1: {Path(model1_name).stem}")]
    if model2_name:
        items.append((_LISA_DEFAULT, f"M2: {Path(model2_name).stem}"))
    x, y = w - 280, 14
    for color, txt in items:
        cv2.rectangle(frame, (x, y), (x + 18, y + 14), color, -1)
        cv2.putText(
            frame, txt, (x + 24, y + 12),
            font, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
        )
        y += 22
    cv2.putText(
        frame, f"{fps:.1f} FPS", (12, h - 12),
        font, 0.52, (180, 220, 180), 1, cv2.LINE_AA,
    )


# ── GStreamer source builder ──────────────────────────────────────────────────

def _build_gst_pipeline(source: str, width: int, height: int) -> str:
    """
    Build a GStreamer pipeline string that decodes the given source
    and exposes BGR frames via appsink.

    Jetson-specific elements:
      nvarguscamerasrc -- CSI camera capture (RPi Cam V2 / IMX219)
      nvv4l2decoder    -- hardware H.264/H.265 decode
      nvvideoconvert   -- GPU-side colour-space conversion

    Source formats accepted:
      csi              RPi Cam V2 via CSI ribbon cable, sensor-id 0
      csi:1            CSI camera sensor-id 1 (second camera)
      0 / /dev/video0  RPi Cam V2 via USB adapter, or any USB camera
      rtsp://...       RTSP network stream
      file.mp4         Local video file
    """
    tail = (
        "nvvideoconvert ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=false "
        "max-buffers=1 drop=true sync=false"
    )

    # ── RPi Cam V2 / IMX219 via CSI ribbon cable ──────────────────────
    # Use:  --source csi      (sensor-id 0)
    #       --source csi:1    (sensor-id 1)
    if source.lower().startswith("csi"):
        sensor_id = int(source.split(":")[1]) if ":" in source else 0
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),format=NV12,"
            f"width={width},height={height},framerate=30/1 ! "
            + tail
        )

    # ── RTSP network stream ───────────────────────────────────────────
    if source.startswith("rtsp://") or source.startswith("rtsps://"):
        return (
            f"rtspsrc location={source} latency=100 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! " + tail
        )

    # ── Local video file ──────────────────────────────────────────────
    if Path(source).exists():
        ext = Path(source).suffix.lower()
        if ext in {".mp4", ".mkv", ".avi", ".mov"}:
            return (
                f"filesrc location={source} ! qtdemux ! "
                "h264parse ! nvv4l2decoder ! " + tail
            )
        return f"filesrc location={source} ! decodebin ! " + tail

    # ── RPi Cam V2 via USB adapter / any V4L2 USB camera ─────────────
    # Use:  --source 0   or   --source /dev/video0
    dev = source if source.startswith("/dev/") else f"/dev/video{source}"
    return (
        f"v4l2src device={dev} ! "
        f"video/x-raw,width={width},height={height},framerate=30/1 ! "
        + tail
    )


# ── GStreamer frame reader ────────────────────────────────────────────────────

def _try_gstreamer(pipeline_str: str):
    """
    Attempt to initialise a GStreamer pipeline.
    Returns (pipeline, appsink) on success, (None, None) on failure.
    """
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        pipeline = Gst.parse_launch(pipeline_str)
        appsink = pipeline.get_by_name("sink")
        if appsink is None:
            return None, None
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            return None, None
        return pipeline, appsink
    except Exception as exc:
        print(
            f"[GStreamer] init failed ({exc}); "
            "falling back to cv2.VideoCapture"
        )
        return None, None


def _pull_frame_gst(appsink) -> Optional[np.ndarray]:
    """Pull one BGR frame from a GStreamer appsink."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        sample = appsink.emit("pull-sample")
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        frame = (
            np.frombuffer(info.data, dtype=np.uint8)
            .reshape(h, w, 3)
            .copy()
        )
        buf.unmap(info)
        return frame
    except Exception:
        return None


# ── detection -> payload converter ───────────────────────────────────────────

def _boxes_to_payloads(
    boxes,
    names: dict,
    frame_num: int,
    source_id: int,
    img_w: int,
    img_h: int,
) -> List[DetectionPayload]:
    if boxes is None or len(boxes) == 0:
        return []
    payloads = []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        cls_id = int(cls_ids[i])
        payloads.append(DetectionPayload(
            class_id=cls_id,
            confidence=float(confs[i]),
            x_center=float((x1 + x2) / 2 / img_w),
            y_center=float((y1 + y2) / 2 / img_h),
            width=float((x2 - x1) / img_w),
            height=float((y2 - y1) / img_h),
            source_id=source_id,
            frame_num=frame_num,
            label=names.get(cls_id, str(cls_id)),
        ))
    return payloads


# ── RTSP server ───────────────────────────────────────────────────────────────

class RTSPStreamer:
    """
    Serves annotated BGR frames as a live RTSP stream via GstRtspServer.

    On Jetson uses nvv4l2h264enc (hardware H.264 encoder).
    Pass encoder='x264enc' on a dev machine without JetPack.

    View on iPad:
      1. Install VLC from the App Store (free)
      2. VLC -> Network -> rtsp://JETSON_IP:8554/live
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        port: int = 8554,
        mount: str = "/live",
        encoder: str = "nvv4l2h264enc",
    ) -> None:
        self._width = width
        self._height = height
        self._fps = int(fps)
        self._port = port
        self._mount = mount
        self._encoder = encoder
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._appsrc = None
        self._frame_n = 0
        self._ns_per_frame = int(1e9 / self._fps)
        self._Gst = None
        self._loop = None
        self._started = False
        self._start()

    def _launch_string(self) -> str:
        enc = (
            f"{self._encoder} bitrate=4000000 ! "
            "video/x-h264,stream-format=byte-stream ! h264parse ! "
        )
        return (
            "( appsrc name=source is-live=true block=true format=3 "
            f"caps=video/x-raw,format=BGR,"
            f"width={self._width},height={self._height},"
            f"framerate={self._fps}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"{enc}"
            "rtph264pay name=pay0 pt=96 config-interval=1 )"
        )

    def _start(self) -> None:
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtspServer", "1.0")
            from gi.repository import Gst, GstRtspServer, GLib
        except Exception as exc:
            print(f"[RTSP] GstRtspServer unavailable ({exc}). Disabled.")
            return

        self._Gst = Gst

        server = GstRtspServer.RTSPServer()
        server.set_service(str(self._port))

        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(self._launch_string())
        factory.set_shared(True)
        factory.connect("media-configure", self._on_media_configure)

        server.get_mount_points().add_factory(self._mount, factory)
        server.attach(None)

        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(
            target=self._loop.run, daemon=True
        )
        self._thread.start()
        self._started = True
        print(
            f"[RTSP] Stream ready  rtsp://0.0.0.0:"
            f"{self._port}{self._mount}"
        )
        print(
            f"       iPad VLC      rtsp://JETSON_IP:"
            f"{self._port}{self._mount}"
        )

    def _on_media_configure(self, factory, media) -> None:
        appsrc = media.get_element().get_child_by_name("source")
        if appsrc:
            self._appsrc = appsrc
            appsrc.connect("need-data", self._on_need_data)

    def _on_need_data(self, src, length) -> None:
        try:
            frame = self._q.get(timeout=0.1)
        except queue.Empty:
            return
        data = frame.tobytes()
        buf = self._Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self._ns_per_frame
        buf.pts = self._frame_n * self._ns_per_frame
        buf.dts = buf.pts
        buf.offset = self._frame_n
        self._frame_n += 1
        src.emit("push-buffer", buf)

    def push(self, frame: np.ndarray) -> None:
        """Push an annotated BGR frame into the RTSP stream."""
        if not self._started or self._appsrc is None:
            return
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame.copy())
            except queue.Empty:
                pass

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.quit()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Jetson pipeline: GStreamer decode + "
            "TRT inference + RTSP stream + UDP HUD."
        )
    )
    p.add_argument(
        "--source", required=True,
        help="Camera index, /dev/videoN, rtsp://..., or file path.",
    )
    p.add_argument(
        "--model", default="models/yolov8n.engine",
        help="Primary TRT model (COCO). Default: models/yolov8n.engine",
    )
    p.add_argument(
        "--model2", default="",
        help="Optional second TRT model (LISA traffic signs).",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--conf2", type=float, default=0.30)
    p.add_argument(
        "--output", default="",
        help="Annotated output .mp4 path (optional).",
    )
    p.add_argument(
        "--json", default="",
        help="Save all detections to a .json file.",
    )
    p.add_argument("--udp-host", default="", help="UDP host for HUD.")
    p.add_argument("--udp-port", type=int, default=5055)
    p.add_argument(
        "--rtsp-port", type=int, default=0,
        help="Enable RTSP server on this port (e.g. 8554). 0 = off.",
    )
    p.add_argument(
        "--rtsp-encoder", default="nvv4l2h264enc",
        help="GStreamer H.264 encoder (default: nvv4l2h264enc). "
             "Use x264enc on non-Jetson machines.",
    )
    p.add_argument(
        "--width", type=int, default=1280,
        help="Camera capture width (ignored for file/RTSP sources).",
    )
    p.add_argument(
        "--height", type=int, default=720,
        help="Camera capture height (ignored for file/RTSP sources).",
    )
    p.add_argument("--device", default="0", help="CUDA device index.")
    p.add_argument(
        "--no-show", action="store_true",
        help="Disable cv2.imshow preview (use for headless operation).",
    )
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── load models ───────────────────────────────────────────────────
    print(f"Loading model 1 : {args.model}")
    model1 = YOLO(args.model)
    model2 = None
    if args.model2:
        print(f"Loading model 2 : {args.model2}")
        model2 = YOLO(args.model2)
        print(f"  LISA classes  : {list(model2.names.values())}")

    device = args.device or None

    # ── UDP publisher ─────────────────────────────────────────────────
    udp: Optional[UdpPublisher] = None
    if args.udp_host:
        udp = UdpPublisher(host=args.udp_host, port=args.udp_port)
        print(f"UDP HUD stream  : {args.udp_host}:{args.udp_port}")

    # ── GStreamer decode pipeline ──────────────────────────────────────
    pipeline_str = _build_gst_pipeline(
        args.source, args.width, args.height
    )
    print(f"\nGStreamer pipeline:\n  {pipeline_str}\n")

    gst_pipeline, appsink = _try_gstreamer(pipeline_str)
    use_gst = gst_pipeline is not None

    cap: Optional[cv2.VideoCapture] = None
    if not use_gst:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        print("Using cv2.VideoCapture fallback (no hardware decode).")

    # ── probe first frame for dimensions ──────────────────────────────
    frame0: Optional[np.ndarray] = None
    for _ in range(30):
        frame0 = (
            _pull_frame_gst(appsink)
            if use_gst
            else (lambda: (lambda ok, f: f if ok else None)(*cap.read()))()
        )
        if frame0 is not None:
            break
        time.sleep(0.05)

    if frame0 is None:
        raise RuntimeError(
            "No frames received -- check pipeline / source path."
        )

    h, w = frame0.shape[:2]
    fps_src = 30.0 if use_gst else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    print(f"Frame size: {w}x{h}  |  FPS (src): {fps_src:.1f}")

    # ── RTSP server ───────────────────────────────────────────────────
    rtsp: Optional[RTSPStreamer] = None
    if args.rtsp_port:
        rtsp = RTSPStreamer(
            width=w,
            height=h,
            fps=fps_src,
            port=args.rtsp_port,
            encoder=args.rtsp_encoder,
        )

    # ── output video writer ───────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))
        print(f"Output video    : {out_path}")

    json_frames: list = []
    save_json = bool(args.json)

    # ── inference loop ────────────────────────────────────────────────
    frame_count = 0
    sequence = 0
    t_start = time.time()
    cur_frame = frame0

    try:
        while cur_frame is not None:
            img_h, img_w = cur_frame.shape[:2]
            annotated = cur_frame.copy()
            all_payloads: List[DetectionPayload] = []

            # model 1: COCO
            r1 = model1.predict(
                source=cur_frame, conf=args.conf,
                device=device, verbose=False,
            )
            b1 = r1[0].boxes
            if b1 is not None and len(b1):
                _draw_detections(
                    annotated,
                    b1.xyxy.cpu().numpy(),
                    b1.conf.cpu().numpy(),
                    [model1.names[int(c)] for c in b1.cls.cpu().numpy()],
                    box_color=COCO_BOX,
                )
                all_payloads += _boxes_to_payloads(
                    b1, model1.names, frame_count,
                    source_id=0, img_w=img_w, img_h=img_h,
                )

            # model 2: LISA traffic signs
            if model2 is not None:
                r2 = model2.predict(
                    source=cur_frame, conf=args.conf2,
                    device=device, verbose=False,
                )
                b2 = r2[0].boxes
                if b2 is not None and len(b2):
                    _draw_detections(
                        annotated,
                        b2.xyxy.cpu().numpy(),
                        b2.conf.cpu().numpy(),
                        [
                            model2.names[int(c)]
                            for c in b2.cls.cpu().numpy()
                        ],
                        box_color=_LISA_DEFAULT,
                        label_fn=_lisa_color,
                        thickness=3,
                    )
                    all_payloads += _boxes_to_payloads(
                        b2, model2.names, frame_count,
                        source_id=1, img_w=img_w, img_h=img_h,
                    )

            elapsed = max(time.time() - t_start, 1e-6)
            live_fps = frame_count / elapsed
            _draw_overlay(
                annotated, args.model, args.model2 or None, live_fps
            )

            if udp is not None:
                udp.send(build_message(all_payloads, sequence))

            if rtsp is not None:
                rtsp.push(annotated)

            if save_json:
                json_frames.append({
                    "idx": frame_count,
                    "detections": [
                        {
                            "cls_id": p.class_id,
                            "label": p.label,
                            "conf": round(p.confidence, 4),
                            "x_center": round(p.x_center, 5),
                            "y_center": round(p.y_center, 5),
                            "width": round(p.width, 5),
                            "height": round(p.height, 5),
                            "source_id": p.source_id,
                        }
                        for p in all_payloads
                    ],
                })

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow("Jetson Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            sequence += 1

            if use_gst:
                cur_frame = _pull_frame_gst(appsink)
            else:
                ok, cur_frame = cap.read()
                if not ok:
                    cur_frame = None

    finally:
        if use_gst and gst_pipeline is not None:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            gst_pipeline.set_state(Gst.State.NULL)
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if udp is not None:
            udp.close()
        if rtsp is not None:
            rtsp.stop()
        cv2.destroyAllWindows()

    if save_json and json_frames:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "meta": {
                "source": args.source,
                "model": args.model,
                "model2": args.model2,
                "conf": args.conf,
                "conf2": args.conf2,
                "width": w,
                "height": h,
                "fps": fps_src,
                "total_frames": frame_count,
            },
            "frames": json_frames,
        }
        with open(json_path, "w") as f:
            json.dump(output_data, f, separators=(",", ":"))
        size_mb = json_path.stat().st_size / 1e6
        print(f"Detections JSON : {json_path}  ({size_mb:.1f} MB)")

    elapsed = max(time.time() - t_start, 1e-6)
    print(
        f"\nProcessed {frame_count} frames in {elapsed:.2f}s "
        f"({frame_count / elapsed:.2f} FPS)."
    )


if __name__ == "__main__":
    main()
