"""FlowNav image-goal TensorRT host server (loopback HTTP).

The FALCON Noetic adapter container ships no TensorRT, so FlowNav inference runs
here as a HOST process; the in-container ``flownav_node.py`` reaches it via
``--network host`` + ``127.0.0.1:<port>`` loopback. Flask is single-threaded /
single-process (one drone, one CUDA context, one rolling frame buffer).

Contract (image-goal; leaner than NavDP's point-goal server):
  * ``POST /imagegoal_step`` -- multipart ``image`` (current RGB PNG) +
    ``goal_image`` (target RGB PNG). The server keeps the rolling
    ``context_size+1`` frame history, preprocesses (``transform_images`` parity),
    runs :class:`FlowNavTRTPolicy`, and returns the body-frame waypoints.
  * ``POST /reset`` -- clears the rolling frame buffer (new goal / episode).
  * ``GET  /health`` -- liveness probe.

Policy on failures (repo rule: raise, don't silently degrade): the engines are
deserialized + version-locked at startup (not lazily), so an engine built for
another GPU / TensorRT build fails loud before the server serves.

Run (host, flownav_trt env; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.flownav.server.flownav_trt_server \
        --engine-dir sparx_agency/tasks/planning/flownav/engines/<target_tag> --port 8889
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from sparx_agency.tasks.planning.flownav.server import preprocess

app = Flask(__name__)
_POLICY = None          # FlowNavTRTPolicy, built at startup
_FRAMES = None          # deque(maxlen=context_size+1) of RGB uint8 frames
_CFG = {}


def _require_trt_artifacts(engine_dir, head_params):
    """Fail loud if the engine directory is not a complete, selected build."""
    engine_dir = Path(engine_dir)
    if not (engine_dir / "selected.json").exists():
        raise SystemExit("[fatal] %s has no selected.json; run the benchmark to "
                         "choose a precision + K first." % engine_dir)
    missing = [] if Path(head_params).exists() else [str(head_params)]
    sel = json.loads((engine_dir / "selected.json").read_text())
    for name in sel.get("engines", {}).values():
        if not (engine_dir / name).exists():
            missing.append(str(engine_dir / name))
    if missing:
        raise SystemExit("[fatal] missing TRT artifacts: %s" % missing)


def _build_policy(args):
    """Deserialize + version-lock the engines now (not on first request)."""
    from sparx_agency.core.planning.flownav.trt.policy import FlowNavTRTPolicy
    head = args.head_params or str(Path(args.engine_dir) / "flownav_head_params.npz")
    _require_trt_artifacts(args.engine_dir, head)
    policy = FlowNavTRTPolicy(args.engine_dir, head, num_steps=args.num_steps)
    print("[flownav-trt] engines loaded (precision=%s N=%d K=%d)"
          % (policy.precision, policy.num_samples, policy.num_steps))
    return policy


def _read_rgb(file_storage):
    """Decode an uploaded PNG into an HxWx3 uint8 RGB array."""
    return np.asarray(Image.open(file_storage.stream).convert("RGB"))


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": _POLICY is not None, "algo": "flownav-trt"})


@app.route("/reset", methods=["POST"])
def reset():
    _FRAMES.clear()
    return jsonify({"algo": "flownav-trt"})


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    if _POLICY is None:
        return jsonify({"error": "server not initialized"}), 503
    if "image" not in request.files or "goal_image" not in request.files:
        return jsonify({"error": "need multipart 'image' and 'goal_image'"}), 400

    _FRAMES.append(_read_rgb(request.files["image"]))            # current obs frame
    goal_rgb = _read_rgb(request.files["goal_image"])

    image_size = _CFG["image_size"]
    obs_img = preprocess.build_obs_stack(list(_FRAMES), image_size,
                                         _CFG["context_size"] + 1)
    goal_img = preprocess.build_goal(goal_rgb, image_size)

    actions, distance = _POLICY.predict(obs_img, goal_img)       # (N,8,2), float
    chosen = actions[0]                                          # FlowNav executes sample 0
    return jsonify({"trajectory": chosen.tolist(),
                    "all_trajectory": actions.tolist(),
                    "distance": float(distance)})


def main():
    global _POLICY, _FRAMES, _CFG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8889)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--head-params", default=None)
    ap.add_argument("--context-size", type=int, default=3,
                    help="FlowNav context_size; the obs stack is context_size+1 frames")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override the flow-matching step count K (default: selected.json)")
    args = ap.parse_args()

    _CFG = {"context_size": int(args.context_size), "image_size": int(args.image_size)}
    _FRAMES = deque(maxlen=_CFG["context_size"] + 1)
    _POLICY = _build_policy(args)
    print("[flownav-trt] port=%d engine-dir=%s context=%d image=%d"
          % (args.port, args.engine_dir, args.context_size, args.image_size))
    app.run(host="127.0.0.1", port=args.port, threaded=False, processes=1)


if __name__ == "__main__":
    main()
