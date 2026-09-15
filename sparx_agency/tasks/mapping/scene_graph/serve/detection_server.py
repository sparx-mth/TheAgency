"""Shared YOLO-World / LLMDet HTTP service; no ROS, lazy model imports.

Threaded requests share one serialized detector. GET /health returns model,
device, classes, metadata and frames_served. POST /detect accepts JPEG bytes
and returns w, h, ms, detections, classes and metadata, captured atomically
with inference. Metadata identifies checkpoint bytes, configuration and
library versions. POST /set_classes changes the vocabulary explicitly.

Use --device cpu for a CPU detector while Habitat owns the rendering GPU.
The default is the local yolov8x-worldv2.pt checkpoint in the working directory.
A missing checkpoint or unavailable requested device fails at startup; there
is no checkpoint substitution or CPU fallback. --selftest needs no model.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.core.mapping.detection.registry import default_detection_registry
from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig
from sparx_agency.tasks.mapping.scene_graph.serve.backends import (
    build_detector, refresh_vocabulary_metadata)
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
                 classes: Sequence[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.detector = detector
        self.model_name = model_name
        self.device = device
        self.classes: List[str] = list(classes)
        self.metadata = dict(metadata or {})
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
        with ctx.lock:
            payload = {"ok": True, "model": ctx.model_name, "device": ctx.device,
                       "classes": list(ctx.classes), "metadata": dict(ctx.metadata),
                       "frames_served": ctx.frames_served}
        self._send_json(payload)

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
        bgr = decode_frame(self._read_body())
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        t0 = time.perf_counter()
        with ctx.lock:
            dets = ctx.detector.detect(rgb)
            classes = list(ctx.classes)
            metadata = dict(ctx.metadata)
            ctx.frames_served += 1
        self._send_json({
            "w": int(bgr.shape[1]), "h": int(bgr.shape[0]),
            "ms": float((time.perf_counter() - t0) * 1000.0),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "classes": classes, "metadata": metadata,
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
            ctx.metadata = refresh_vocabulary_metadata(ctx.detector, ctx.metadata)
        print("%s vocabulary set to %d classes" % (_TAG, len(cleaned)))
        self._send_json({"ok": True, "classes": cleaned})


def _make_server(ctx: _ServerContext, host: str, port: int) -> ThreadingHTTPServer:
    """Threaded connections, serialized inference, one detector instance."""
    server = ThreadingHTTPServer((host, port), _DetectionHandler)
    server.daemon_threads = True
    server.ctx = ctx  # type: ignore[attr-defined]
    return server


# ── startup (the torch-touching side; all heavy imports live in here) ────────
def _build_real_detector(args: argparse.Namespace):
    """Fail at startup, rather than serving a substituted or half-loaded model."""
    print("%s warm-loading %s on %s ..." % (_TAG, args.backend, args.device))
    try:
        result = build_detector(args, _parse_classes(args.classes))
    except Exception as exc:
        raise SystemExit("%s [fatal] %s: %s" % (_TAG, args.backend, exc)) from exc
    print("%s model ready" % _TAG)
    return result


def _parse_classes(spec: str) -> List[str]:
    """Split the ``--classes`` comma list; empty spec means the default vocab."""
    classes = [c.strip() for c in spec.split(",") if c.strip()]
    if not classes:
        raise SystemExit("%s [fatal] --classes parsed to an empty list" % _TAG)
    return classes


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=default_detection_registry().names(),
                   default="yolo_world", help="Detector backend (default: YOLO-World X-v2)")
    p.add_argument("--model", default=None,
                   help="Local checkpoint; YOLO defaults to %s in the working directory; "
                        "LLMDet requires an explicit HF snapshot directory" % YoloWorldConfig().model_path)
    p.add_argument("--device", default="cuda:0",
                   help="Torch device (default cuda:0; pass cpu explicitly "
                        "for a CPU run)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--conf", type=float, default=None,
                   help="Required for LLMDet; YOLO default 0.25. Downstream filters again")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    p.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU")
    p.add_argument("--shortest-edge", type=int, default=800, help="LLMDet resize short edge")
    p.add_argument("--longest-edge", type=int, default=1333, help="LLMDet resize long edge cap")
    p.add_argument("--chunk-size", type=int, default=80, help="LLMDet categories per caption at most")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--torch-threads", type=int, default=None, help="Explicit CPU intra-op thread count")
    p.add_argument("--classes", default=",".join(DEFAULT_HOSPITAL_VOCABULARY),
                   help="Comma-separated vocabulary (default: the hospital list)")
    p.add_argument("--selftest", action="store_true",
                   help="Exercise request routing against a stub detector "
                        "(no model, no torch) and exit")
    args = p.parse_args(argv)
    if args.model is None:
        if args.backend == "yolo_world":
            args.model = YoloWorldConfig().model_path
        elif not args.selftest:
            p.error("--model is required for LLMDet")
    elif not args.model.strip():
        p.error("--model must be a non-empty local checkpoint path")
    if args.conf is None:
        if args.backend == "llmdet" and not args.selftest:
            p.error("--conf is required for LLMDet; do not inherit YOLO thresholds")
        args.conf = 0.25
    return args


def main() -> None:
    args = parse_args()
    if args.selftest:
        from sparx_agency.tasks.mapping.scene_graph.serve.selftest import run_selftest
        run_selftest()
        return
    detector, metadata = _build_real_detector(args)
    ctx = _ServerContext(detector, model_name=Path(args.model).name,
                         device=args.device, classes=_parse_classes(args.classes),
                         metadata=metadata)
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
