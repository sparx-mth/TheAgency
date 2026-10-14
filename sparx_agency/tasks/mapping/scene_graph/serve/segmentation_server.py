"""Box-prompted Segment Anything HTTP service; lazy model imports.

The segmentation sibling of ``detection_server.py``, deliberately a SEPARATE
process: the detector service pins its model identity per run and a client
refuses any drift, so a segmenter inside it would have changed that identity
for every run, segmentation on or off. Threaded requests share one serialized
model. GET /health returns model, device, metadata and frames_served. POST
/segment accepts the contract's JSON (base64 JPEG frame + boxes) and returns
w, h, ms, one PNG mask per box in box order, metadata and the request hash,
captured atomically with inference.

Use --device cpu while a simulator owns the rendering GPU (MobileSAM runs at
~0.3 s per 640 x 480 frame on four CPU threads). A missing checkpoint or an
unavailable requested device fails at startup; there is no checkpoint
substitution, no download and no CPU fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sparx_agency.core.mapping.interfaces.segmentation_model import SegmentationModel
from sparx_agency.tasks.mapping.scene_graph.serve.backends import _configure_runtime, checkpoint_identity
from sparx_agency.tasks.mapping.scene_graph.serve.contract import JPEG_QUALITY
from sparx_agency.tasks.mapping.scene_graph.serve.segmentation_contract import (
    DEFAULT_SEGMENTATION_PORT, SEGMENTATION_BACKEND, masks_to_json, parse_segment_request)

_TAG = "[scene-graph-segment]"


class _SegmentationContext:
    """Mutable state shared by the request handlers across server threads; ``lock`` serialises the model."""

    def __init__(self, segmenter: SegmentationModel, model_name: str, device: str,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        self.segmenter = segmenter
        self.model_name = model_name
        self.device = device
        self.metadata = dict(metadata or {})
        self.frames_served = 0
        self.lock = threading.Lock()


class _SegmentationHandler(BaseHTTPRequestHandler):
    """Request handler; the bound :class:`_SegmentationContext` is ``self.server.ctx``."""

    protocol_version = "HTTP/1.1"

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

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/health":
            self._send_json({"ok": False, "error": "unknown path %s" % self.path}, 404)
            return
        ctx = self.server.ctx  # type: ignore[attr-defined]
        with ctx.lock:
            payload = {"ok": True, "model": ctx.model_name, "device": ctx.device,
                       "metadata": dict(ctx.metadata), "frames_served": ctx.frames_served}
        self._send_json(payload)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        try:
            if self.path == "/segment":
                self._handle_segment()
            else:
                self._send_json({"ok": False, "error": "unknown path %s" % self.path}, 404)
        except ValueError as exc:                      # bad request data
            self._send_json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:                       # inference/server fault
            print("%s ERROR %s: %s" % (_TAG, self.path, exc))
            self._send_json({"ok": False, "error": str(exc)}, 500)

    def _handle_segment(self) -> None:
        ctx = self.server.ctx  # type: ignore[attr-defined]
        raw = self._read_body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("segment body is not JSON: %s" % exc)
        bgr, jpeg, boxes = parse_segment_request(body)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        t0 = time.perf_counter()
        with ctx.lock:
            masks = ctx.segmenter.segment(rgb, boxes) if boxes else []
            metadata = dict(ctx.metadata)
            ctx.frames_served += 1
        self._send_json({
            "w": int(bgr.shape[1]), "h": int(bgr.shape[0]),
            "ms": float((time.perf_counter() - t0) * 1000.0),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "model": ctx.model_name, "metadata": metadata,
            "masks": masks_to_json(masks, boxes),
            "request_sha256": hashlib.sha256(jpeg).hexdigest(),
        })


def _make_server(ctx: _SegmentationContext, host: str, port: int) -> ThreadingHTTPServer:
    """Threaded connections, serialized inference, one segmenter instance."""
    server = ThreadingHTTPServer((host, port), _SegmentationHandler)
    server.daemon_threads = True
    server.ctx = ctx  # type: ignore[attr-defined]
    return server


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Box-prompted Segment Anything service")
    parser.add_argument("--model", required=True, help="Local ultralytics SAM-family checkpoint (mobile_sam.pt, sam2.1_t.pt, ...)")
    parser.add_argument("--device", default="cpu", help="cpu beside a simulator; cuda:N only with the card accounted for")
    parser.add_argument("--allow-shared-gpu", action=argparse.BooleanOptionalAction, default=False,
                        help="Explicit operator authorization to share CUDA with rendering/desktop")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_SEGMENTATION_PORT)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-boxes", type=int, default=64)
    parser.add_argument("--torch-threads", type=int)
    return parser.parse_args(argv)


def segmenter_metadata(segmenter, identity: Dict[str, Any], *, allow_shared_gpu: bool = False) -> Dict[str, Any]:
    """Static identity; request timing and live memory do not cause identity drift."""
    import torch
    import cv2

    packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "ultralytics", "requests")}
    packages["opencv_loaded"] = cv2.__version__
    return dict(identity, backend=SEGMENTATION_BACKEND, segmenter_config=asdict(segmenter.cfg), packages=packages,
                preprocessing={"input": "RGB uint8 HWC -> BGR for ultralytics", "resize": "ultralytics SAM letterbox",
                               "imgsz": segmenter.cfg.imgsz, "prompt": "boxes, original pixels XYXY",
                               "masks": "boolean, original frame resolution, one per box in box order"},
                wire={"encoding": "JPEG", "quality": JPEG_QUALITY, "decoder": "OpenCV BGR -> RGB",
                      "masks": "PNG 0/255 base64"},
                runtime={"torch_threads": torch.get_num_threads(), "cuda_version": torch.version.cuda,
                         "allow_shared_gpu": bool(allow_shared_gpu)})


def build_segmenter(args: argparse.Namespace):
    """Fail at startup, rather than serving a substituted or half-loaded model."""
    from sparx_agency.core.mapping.segmentation import SamPromptedConfig, SamPromptedSegmenter

    path = Path(args.model).expanduser()
    if not path.is_file():
        raise FileNotFoundError("SAM checkpoint missing: %s (no download; provision it explicitly)" % path)
    identity = checkpoint_identity(path)
    _configure_runtime(args)
    segmenter = SamPromptedSegmenter(SamPromptedConfig(model_path=str(path), device=args.device,
                                                       imgsz=args.imgsz, max_boxes=args.max_boxes))
    segmenter.segment(np.zeros((64, 64, 3), dtype=np.uint8), [(8.0, 8.0, 56.0, 56.0)])    # warm load
    if args.torch_threads is not None:
        import torch
        torch.set_num_threads(args.torch_threads)      # ultralytics resets the thread count on its first predict
    if checkpoint_identity(path) != identity:
        raise RuntimeError("Checkpoint changed during model loading")
    return segmenter, segmenter_metadata(segmenter, identity, allow_shared_gpu=args.allow_shared_gpu)


def main() -> None:
    args = parse_args()
    print("%s warm-loading %s on %s ..." % (_TAG, args.model, args.device))
    try:
        segmenter, metadata = build_segmenter(args)
    except Exception as exc:
        raise SystemExit("%s [fatal] %s" % (_TAG, exc)) from exc
    print("%s model ready" % _TAG)
    ctx = _SegmentationContext(segmenter, model_name=Path(args.model).name, device=args.device, metadata=metadata)
    server = _make_server(ctx, args.host, args.port)
    print("%s serving on %s:%d  imgsz=%d" % (_TAG, args.host, args.port, args.imgsz))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("%s shutting down" % _TAG)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
