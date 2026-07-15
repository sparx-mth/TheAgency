#!/usr/bin/env python3
"""On-demand DA3 depth HTTP server — one request, one frame, one depth result.

Drop-in replacement for the (broken) C++ depth_anything_http_server's /bbox_depth
endpoint: comm_manager_vllm.py already POSTs each frame here with the same
multipart fields (image, detections, image_dir). This computes fresh DA3 depth
for exactly that image and saves a colorized PNG to
<image_dir's-parent>/<image_dir's-name>_depth/<stem>_depth.png — the same
convention display_server.py's _depth_variant() already looks for, so no
display-side changes are needed once this is running.

No ROS topics, no timestamp pairing: each request runs inference on exactly the
bytes it was given, so there is no "depth from a different frame" possible.

Usage:
    python3 da3_depth_http_server.py \\
      --engine /path/to/DA3METRIC-LARGE.fp16-294x504.depth_only.v2.engine \\
      --calib  /path/to/camera_xtend_ros_calib_504_294_resize.yaml \\
      --host 192.0.0.89 --port 5071
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401 — creates the CUDA context we push/pop below
from flask import Flask, jsonify, request

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.robots.common.helpers import sanitize_depth

app = Flask(__name__)
model: DA3TensorRTModel | None = None
args: argparse.Namespace | None = None

METRIC_SCALE_DIVISOR = 300.0  # matches depth_processor_node.py's large_metric conversion


def _colorize(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth_clipped = np.clip(depth_m, 0.0, max_depth_m)
    depth_norm = (depth_clipped / max(max_depth_m, 1e-6) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)


@app.post("/bbox_depth")
def bbox_depth():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "missing 'image' file"}), 400

    image_file = request.files["image"]
    image_dir = request.form.get("image_dir", "").strip()

    buf = np.frombuffer(image_file.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "failed to decode image"}), 400

    # Flask's dev server invokes this handler from a context where pycuda's
    # CUDA context isn't reliably "current" (observed: TensorRT enqueueV3
    # fails with "invalid resource handle" without this, even though the
    # exact same model call works fine in a plain script). Explicitly push/pop
    # around each inference call to make it current for this call.
    pycuda.autoinit.context.push()
    try:
        raw = model.infer_all(img)
    finally:
        pycuda.autoinit.context.pop()
    metric_depth = (model.intrinsics.fx * raw) / METRIC_SCALE_DIVISOR
    metric_depth = sanitize_depth(metric_depth, clip_min=args.clip_min_m, clip_max=args.clip_max_m)

    depth_png_path = None
    if image_dir:
        stem = Path(image_file.filename).stem
        src_dir = Path(image_dir)
        depth_dir = src_dir.parent / f"{src_dir.name}_depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_png_path = str(depth_dir / f"{stem}_depth.png")
        cv2.imwrite(depth_png_path, _colorize(metric_depth, args.max_depth_m))

    return jsonify({
        "ok": True,
        "depth_png": depth_png_path,
        "min_m": float(np.nanmin(metric_depth)),
        "max_m": float(np.nanmax(metric_depth)),
        "mean_m": float(np.nanmean(metric_depth)),
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, help="Path to DA3 TensorRT engine")
    p.add_argument("--calib", required=True, help="Path to camera intrinsics YAML")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5071)
    p.add_argument("--clip-min-m", type=float, default=0.2)
    p.add_argument("--clip-max-m", type=float, default=5.0)
    p.add_argument("--max-depth-m", type=float, default=5.0, help="Colormap scaling ceiling")
    return p.parse_args()


def main() -> None:
    global model, args
    args = parse_args()
    model = DA3TensorRTModel(engine_path=args.engine, yaml_path=args.calib)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
