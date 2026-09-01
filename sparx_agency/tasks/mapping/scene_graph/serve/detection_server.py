"""Scene-graph YOLO-World detection HTTP server (conda side, GPU, no ROS).

On this machine the ROS2 python (host jazzy / ``.venv``) has no torch and conda
(``navdp``) has no ``rclpy``, so open-vocabulary detection runs here as a plain
HTTP server and a thin ROS2 client node posts frames to it. stdlib
``http.server`` only — no flask — and single-threaded on purpose (one client,
one GPU, one CUDA context).

Routes (wire types in :mod:`.contract`):
  * ``GET /health`` -> ``{ok, model, device, classes, frames_served}``
  * ``POST /set_classes`` body ``{"classes": [...]}`` -> re-prompts the detector
  * ``POST /detect`` body = raw JPEG bytes ->
    ``{w, h, ms, detections: [{cls, conf, xyxy}]}``

Failure policy (repo rule: raise, never silently degrade): a missing model
path, a missing torch, or ``--device cuda:*`` without a visible CUDA device all
abort at STARTUP with a fatal message — there is no CPU fallback unless
``--device cpu`` is explicit. The model is warm-loaded before serving so a
broken checkpoint never becomes a 500 on the first frame. Check the GPU is
actually free first (``nvidia-smi``): a resident VLA server owns the card.

Run (conda ``navdp`` env; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \\
        --model /path/to/yolov8s-world.pt

``--selftest`` exercises the request routing against a stub detector with no
model and no torch — it runs in the plain ``.venv`` (implementation in the
sibling :mod:`.selftest` module).
"""
from __future__ import annotations

import argparse
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.tasks.mapping.scene_graph.serve.contract import (
    DEFAULT_HOSPITAL_VOCABULARY,
    DEFAULT_PORT,
    DetectionWire,
    decode_frame,
    detections_to_json,
)

_TAG = "[scene-graph-detect]"


class _ServerContext:
    """Mutable state shared by the request handlers across server threads.

    ``lock`` serialises everything that touches the model -- inference and
    re-prompting -- because there is one detector on one GPU. The server is
    threaded so a second client is never refused; the lock is what keeps the
    model itself single-writer.
    """

    def __init__(self, detector: DetectionModel, model_name: str, device: str,
                 classes: Sequence[str]) -> None:
        self.detector = detector
        self.model_name = model_name
        self.device = device
        self.classes: List[str] = list(classes)
        self.frames_served = 0
        self.lock = threading.Lock()


def _wire_from_core(dets: Sequence[Detection2D]) -> List[DetectionWire]:
    """Convert core :class:`Detection2D` results to wire dataclasses."""
    return [
        DetectionWire(
            cls=d.label,
            conf=float(d.score),
            xyxy=tuple(float(v) for v in d.bbox_xyxy),
        )
        for d in dets
    ]


class _DetectionHandler(BaseHTTPRequestHandler):
    """Request handler; the bound :class:`_ServerContext` is ``self.server.ctx``."""

    protocol_version = "HTTP/1.1"

    # ── plumbing ─────────────────────────────────────────────────────
    def log_message(self, fmt: str, *log_args: Any) -> None:
        print("%s %s" % (_TAG, fmt % log_args))

    def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    # ── routes ───────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/health":
            self._send_json({"ok": False, "error": "unknown path %s" % self.path}, 404)
            return
        ctx = self.server.ctx  # type: ignore[attr-defined]
        self._send_json({
            "ok": True,
            "model": ctx.model_name,
            "device": ctx.device,
            "classes": list(ctx.classes),
            "frames_served": ctx.frames_served,
        })

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        try:
            if self.path == "/detect":
                self._handle_detect()
            elif self.path == "/set_classes":
                self._handle_set_classes()
            else:
                self._send_json(
                    {"ok": False, "error": "unknown path %s" % self.path}, 404)
        except ValueError as exc:                      # bad request data
            self._send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:                       # inference/server fault
            print("%s ERROR %s: %s" % (_TAG, self.path, exc))
            self._send_json({"ok": False, "error": str(exc)}, 500)

    def _handle_detect(self) -> None:
        ctx = self.server.ctx  # type: ignore[attr-defined]
        bgr = decode_frame(self._read_body())          # raises ValueError -> 400
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])    # DetectionModel wants RGB
        t0 = time.perf_counter()
        # One model, one GPU: inference is serialised even though the server
        # is threaded. Threading is about not REFUSING a second client, not
        # about parallel inference.
        with ctx.lock:
            dets = ctx.detector.detect(rgb)
        ms = (time.perf_counter() - t0) * 1000.0
        ctx.frames_served += 1
        self._send_json({
            "w": int(bgr.shape[1]),
            "h": int(bgr.shape[0]),
            "ms": float(ms),
            "detections": detections_to_json(_wire_from_core(dets)),
        })

    def _handle_set_classes(self) -> None:
        ctx = self.server.ctx  # type: ignore[attr-defined]
        try:
            body = json.loads(self._read_body().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("set_classes body is not JSON: %s" % exc)
        classes = body.get("classes") if isinstance(body, dict) else None
        if not isinstance(classes, list) or not classes:
            raise ValueError('set_classes expects {"classes": [<non-empty list>]}')
        cleaned = [str(c).strip() for c in classes if str(c).strip()]
        if not cleaned:
            raise ValueError("set_classes: every class string was empty")
        with ctx.lock:                                 # never mid-detect
            ctx.detector.set_prompts(cleaned)          # re-prompts a loaded model
            ctx.classes = cleaned
        print("%s vocabulary set to %d classes" % (_TAG, len(cleaned)))
        self._send_json({"ok": True, "classes": cleaned})


def _make_server(ctx: _ServerContext, host: str, port: int) -> ThreadingHTTPServer:
    """Bind the HTTP server and attach the context.

    THREADED, and the reason is a measured failure rather than a preference.
    With the single-threaded ``HTTPServer`` a second client simply cannot be
    served: the detector client posts a frame every second, and while that
    request is in flight every other connection waits in the accept backlog
    until it times out. When the target-approach node joined at ~2 POST/s the
    whole approach ran with ``posts=2 dets=0 conn_err=2`` -- every request
    timed out, the servo never got a box, and the approach hit its 120 s
    timeout without ever seeing the object. ``/health`` was unanswerable for
    the same reason.

    Inference itself stays serialised behind ``ctx.lock``: there is one model
    on one GPU. Threading buys concurrent *connections*, not concurrent
    inference -- at ~7 ms a frame the queue drains far faster than either
    client fills it.
    """
    server = ThreadingHTTPServer((host, port), _DetectionHandler)
    server.daemon_threads = True
    server.ctx = ctx  # type: ignore[attr-defined]
    return server


# ── startup (the torch-touching side; all heavy imports live in here) ────────
def _build_real_detector(args: argparse.Namespace) -> DetectionModel:
    """Construct and warm-load the YOLO-World detector; fail LOUDLY on any gap.

    Startup aborts (non-zero exit) when the checkpoint is missing, torch cannot
    be imported, or a ``cuda:*`` device is requested without CUDA available.
    The warm-up detect forces the (otherwise lazy) model load so a bad
    checkpoint dies here, not on the first client frame.
    """
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit("%s [fatal] model checkpoint not found: %s"
                         % (_TAG, model_path))
    if args.device.startswith("cuda"):
        try:
            import torch  # lazy: conda-side only
        except ImportError as exc:
            raise SystemExit("%s [fatal] --device %s but torch is not importable "
                             "(wrong interpreter? use the conda env): %s"
                             % (_TAG, args.device, exc))
        if not torch.cuda.is_available():
            raise SystemExit("%s [fatal] --device %s but CUDA is unavailable; "
                             "no silent CPU fallback — pass --device cpu "
                             "explicitly if that is what you want"
                             % (_TAG, args.device))
    from sparx_agency.core.mapping.detection.yolo_world import (
        YoloWorldConfig,
        YoloWorldDetector,
    )
    detector = YoloWorldDetector(YoloWorldConfig(
        model_path=str(model_path), device=args.device, conf_thresh=args.conf))
    detector.set_prompts(_parse_classes(args.classes))
    print("%s warm-loading %s on %s ..." % (_TAG, model_path.name, args.device))
    detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))  # forces model load
    print("%s model ready" % _TAG)
    return detector


def _parse_classes(spec: str) -> List[str]:
    """Split the ``--classes`` comma list; empty spec means the default vocab."""
    classes = [c.strip() for c in spec.split(",") if c.strip()]
    if not classes:
        raise SystemExit("%s [fatal] --classes parsed to an empty list" % _TAG)
    return classes


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="Path to the YOLO-World .pt checkpoint (REQUIRED "
                        "unless --selftest)")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device (default cuda:0; pass cpu explicitly "
                        "for a CPU run)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (downstream mapper filters again)")
    p.add_argument("--classes", default=",".join(DEFAULT_HOSPITAL_VOCABULARY),
                   help="Comma-separated vocabulary (default: the hospital list)")
    p.add_argument("--selftest", action="store_true",
                   help="Exercise request routing against a stub detector "
                        "(no model, no torch) and exit")
    args = p.parse_args(argv)
    if not args.selftest and not args.model:
        p.error("--model is required (unless --selftest)")
    return args


def main() -> None:
    args = parse_args()
    if args.selftest:
        from sparx_agency.tasks.mapping.scene_graph.serve.selftest import run_selftest
        run_selftest()
        return
    detector = _build_real_detector(args)
    ctx = _ServerContext(detector, model_name=Path(args.model).name,
                         device=args.device, classes=_parse_classes(args.classes))
    server = _make_server(ctx, args.host, args.port)
    print("%s serving on %s:%d  conf>=%.2f  vocab=%d"
          % (_TAG, args.host, args.port, args.conf, len(ctx.classes)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("%s shutting down" % _TAG)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
