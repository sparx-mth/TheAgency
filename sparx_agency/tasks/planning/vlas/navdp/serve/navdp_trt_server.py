"""Drop-in TensorRT NavDP server: same HTTP contract, TRT backend.

Honors the EXACT wire contract of the upstream ``navdp_server.py`` for the two
routes the FALCON stack uses -- ``/navigator_reset`` and ``/pointgoal_step`` --
so ``navdp_click_node.py`` and the core ``NavDPPointgoalClient`` run unchanged.
The only difference is the agent: a :class:`TRTNavDPAgent` (TensorRT engines)
instead of the torch model.

This is a HOST process (the FALCON Noetic container ships no TensorRT); FALCON
reaches it via ``--network host`` + ``127.0.0.1:<port>`` loopback, so it must run
on the FALCON host. Flask is single-threaded/single-process (one drone, one CUDA
context).

Policy on failures (repo rule: raise, don't silently degrade):
  * ``--backend trt`` (default) fails loud and exits non-zero if engines /
    head-params are missing or version-locked to another GPU -- a broken engine
    never silently becomes the 10x-slower torch path.
  * ``--backend torch`` (or ``--allow-torch-fallback`` when TRT is unavailable)
    runs the original torch ``NavDP_Agent`` for an intentional comparison.
  * ``/pixelgoal_step`` / ``/imagegoal_step`` / ``/nogoal_step`` return 501 --
    their goal encoders were never exported (point-goal only).

Run (host, TRT venv; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.vlas.navdp.serve.navdp_trt_server \
        --engine-dir .../engines/orin_sm87 --navdp-repo ~/PycharmProjects/NavDP/baselines/navdp
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
_AGENT = None        # constructed on first /navigator_reset
_BUILD_AGENT = None  # closure: intrinsic -> agent
_CFG = {}
# serve/configs/inference_speed.yaml — the LIVE sampler settings (ddim / 4 steps).
# This file decides what the drone actually flies, so it is kept next to the server
# that reads it; `trt/configs/build_policy.json` is build-time and lives with the
# builder. Losing this path silently reverts the server to the 10-step DDPM
# baseline, which is a flight-behaviour change, not a config nit.
_DEFAULT_SPEED_CONFIG = (Path(__file__).resolve().parent
                         / "configs" / "inference_speed.yaml")


def load_speed_config(path=None):
    """Read the runtime sampler settings from the inference-speed YAML.

    Returns ``(sampler, num_inference_steps)``. A missing file / missing pyyaml /
    ``runtime.wired != true`` all fall back to the bit-faithful default
    (``ddpm``, full trained steps), so an absent or not-yet-live config never
    silently changes behaviour. Parsed on the tasks side only -- ``core`` never
    gains a YAML dependency (the resolved values are passed in as plain kwargs).
    """
    default = ("ddpm", None)
    p = Path(path) if path else _DEFAULT_SPEED_CONFIG
    if not p.exists():
        print("[navdp-trt] no speed config at %s; using ddpm baseline" % p)
        return default
    try:
        import yaml
    except ImportError:
        print("[warn] pyyaml missing; ignoring %s, using ddpm baseline" % p)
        return default
    rt = ((yaml.safe_load(p.read_text()) or {}).get("runtime", {})) or {}
    if not rt.get("wired", False):
        print("[navdp-trt] %s runtime.wired=false; using ddpm baseline" % p.name)
        return default
    return str(rt.get("sampler", "ddpm")).lower(), rt.get("num_inference_steps")


def load_gate_config(path=None):
    """Read the ``runtime.validate_against_frozen_gold`` startup-gate block.

    Returns the block dict (``{}`` if absent, disabled, not a mapping, or pyyaml
    is missing) so a stale ``false`` value simply disables the gate.
    """
    p = Path(path) if path else _DEFAULT_SPEED_CONFIG
    if not p.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    rt = ((yaml.safe_load(p.read_text()) or {}).get("runtime", {})) or {}
    g = rt.get("validate_against_frozen_gold")
    return g if isinstance(g, dict) else {}


def _frozen_gold_gate(policy, gate_cfg):
    """Pre-flight gate: if enabled and a reduced-step sampler is configured,
    compare it to the 10-step DDPM gold and REFUSE to serve if the trajectory
    drift exceeds ``max_chosen_l2`` -- a too-aggressive K never reaches the drone.
    A no-op for ``sampler: ddpm`` (nothing to check) or when disabled."""
    if not gate_cfg.get("enabled"):
        return
    if policy.sampler == "ddpm":
        print("[navdp-trt] frozen-gold gate: nothing to check (sampler=ddpm)")
        return
    from sparx_agency.tasks.planning.vlas.navdp.trt.benchmark.validate_steps import frozen_gold_check
    k = len(policy.scheduler.timesteps)
    m = frozen_gold_check(policy, k, num_scenarios=int(gate_cfg.get("scenarios", 16)),
                          frames=gate_cfg.get("frames"))
    limit = float(gate_cfg.get("max_chosen_l2", 5.0))
    inputs = "real" if gate_cfg.get("frames") else "RANDOM(smoke)"
    print("[navdp-trt] frozen-gold gate: ddim/%d vs ddpm/%d  chosen_L2=%.3f "
          "(limit %.3f, inputs=%s, flip=%.3f)"
          % (k, m["num_train"], m["chosen_l2"], limit, inputs, m["argmax_flip"]))
    if m["chosen_l2"] > limit:
        raise SystemExit(
            "[fatal] frozen-gold gate FAILED: ddim/%d trajectory drift %.3f > %.3f "
            "vs the 10-step baseline -- too aggressive to fly. Lower "
            "num_inference_steps, raise max_chosen_l2, or validate on real "
            "--frames first." % (k, m["chosen_l2"], limit))


def _require_trt_artifacts(engine_dir, head_params):
    """Fail loud if the engine directory is not a complete, selected build."""
    engine_dir = Path(engine_dir)
    missing = [str(p) for p in
               [engine_dir / "selected.json", Path(head_params)] if not p.exists()]
    if not (engine_dir / "selected.json").exists():
        raise SystemExit("[fatal] %s has no selected.json; run the benchmark to "
                         "choose a precision first." % engine_dir)
    sel = json.loads((engine_dir / "selected.json").read_text())
    for name in sel.get("engines", {}).values():
        if not (engine_dir / name).exists():
            missing.append(str(engine_dir / name))
    if missing:
        raise SystemExit("[fatal] missing TRT artifacts: %s" % missing)


def _trt_agent_builder(args):
    """Return a closure ``intrinsic -> TRTNavDPAgent`` after loading the engines.

    The engines are deserialized and version-locked HERE (at startup), not lazily
    at the first request, so an incompatible / version-locked engine fails loud
    before the server starts serving instead of 500-ing every reset.
    """
    from sparx_agency.core.planning.vlas.navdp.trt.policy import NavDPTRTPolicy
    from sparx_agency.tasks.planning.vlas.navdp.serve.trt_agent import make_trt_agent
    head = args.head_params or str(Path(args.engine_dir) / "navdp_head_params.npz")
    _require_trt_artifacts(args.engine_dir, head)
    sampler, steps = load_speed_config(args.speed_config)
    if args.sampler:                          # CLI overrides the YAML
        sampler = args.sampler
    if args.num_inference_steps is not None:
        steps = args.num_inference_steps
    policy = NavDPTRTPolicy(args.engine_dir, head, sampler=sampler,
                            num_inference_steps=steps)   # loads + version-locks now
    print("[navdp-trt] engines loaded (precision=%s, sampler=%s, steps=%d)"
          % (policy.precision, policy.sampler, len(policy.scheduler.timesteps)))
    _frozen_gold_gate(policy, load_gate_config(args.speed_config))   # refuses to serve a bad K
    return lambda intrinsic: make_trt_agent(
        intrinsic, args.engine_dir, head, navdp_repo=args.navdp_repo,
        render_cam_height=args.render_cam_height, policy=policy)


def _torch_agent_builder(args):
    """Return a closure ``intrinsic -> NavDP_Agent`` (the original torch model)."""
    from sparx_agency.tasks.planning.vlas.navdp.trt.export.build_policy import resolve_navdp_repo
    repo = resolve_navdp_repo(args.navdp_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from policy_agent import NavDP_Agent
    if not args.ckpt:
        raise SystemExit("[fatal] --backend torch needs --ckpt")
    # No render_cam_height here: upstream's NavDP_Agent.__init__ does not accept
    # it (only our own TRT agent does), and passing it raises TypeError. The
    # agent is built lazily inside /navigator_reset, so the server would start
    # cleanly and then return HTTP 500 on the first reset and 409 on every step
    # after it -- which reads as a dead policy, not a bad keyword argument. The
    # /navigator_reset handler still assigns the attribute for the mask draw,
    # which works on any object.
    return lambda intrinsic: NavDP_Agent(
        intrinsic, image_size=224, memory_size=8, predict_size=24,
        temporal_depth=16, heads=8, token_dim=384, navi_model=args.ckpt,
        device="cuda:0")


def _select_builder(args):
    """Choose the TRT or torch agent builder, honoring the fallback policy.

    Catches both the pre-flight ``SystemExit`` (missing files) and the
    ``NavDPError`` / ``ImportError`` raised when the engines fail to load or are
    version-locked to another GPU. With ``--allow-torch-fallback`` it switches to
    the torch backend; otherwise it exits non-zero with the real cause.
    """
    from sparx_agency.core.planning.vlas.navdp.trt.errors import NavDPError
    if args.backend == "torch":
        return _torch_agent_builder(args)
    try:
        return _trt_agent_builder(args)
    except (SystemExit, NavDPError, ImportError, FileNotFoundError) as e:
        if args.allow_torch_fallback:
            print("[warn] TRT backend unavailable (%s); falling back to torch" % e)
            return _torch_agent_builder(args)
        raise SystemExit("[fatal] TRT backend failed to initialize: %s" % e)


# ── routes (contract-identical to navdp_server.py) ────────────────────────────
@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    global _AGENT
    body = request.get_json()
    intrinsic = np.array(body.get("intrinsic"))
    threshold = np.array(body.get("stop_threshold"))
    batch_size = int(np.array(body.get("batch_size")))
    if batch_size != 1:
        return jsonify({"error": "TRT server is single-drone (batch_size==1)"}), 400
    if _AGENT is None:
        _AGENT = _BUILD_AGENT(intrinsic)
    _AGENT.reset(batch_size, threshold)
    return jsonify({"algo": "navdp-trt"})


@app.route("/pointgoal_step", methods=["POST"])
def pointgoal_step():
    if _AGENT is None:
        return jsonify({"error": "call /navigator_reset first"}), 409
    goal_data = json.loads(request.form.get("goal_data"))
    goal_x = np.array(goal_data["goal_x"])
    goal_y = np.array(goal_data["goal_y"])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = _AGENT.batch_size

    # Render-height passthrough (mask draw only), mirroring the upstream server.
    client_alt = goal_data.get("altitude")
    if client_alt is not None and float(client_alt) > 0:
        _AGENT.render_cam_height = float(client_alt)

    image = Image.open(request.files["image"].stream).convert("RGB")
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)               # kept verbatim (#4)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(request.files["depth"].stream).convert("I")
    depth = np.asarray(depth)[:, :, np.newaxis].astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    execute, all_traj, all_values, mask = _AGENT.step_pointgoal(goal, image, depth)
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return jsonify({"trajectory": execute.tolist(),
                    "all_trajectory": all_traj.tolist(),
                    "all_values": all_values.tolist(),
                    "trajectory_mask": base64.b64encode(buf.getvalue()).decode("ascii")})


@app.route("/pixelgoal_step", methods=["POST"])
@app.route("/imagegoal_step", methods=["POST"])
@app.route("/nogoal_step", methods=["POST"])
def _unsupported_mode():
    return jsonify({"error": "this TRT server supports point-goal only; the "
                    "image/pixel/nogoal encoders were not exported"}), 501


def main():
    global _BUILD_AGENT, _CFG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--engine-dir", default=None)
    ap.add_argument("--head-params", default=None)
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--render-cam-height", type=float, default=0.2)
    ap.add_argument("--backend", choices=["trt", "torch"], default="trt")
    ap.add_argument("--allow-torch-fallback", action="store_true")
    ap.add_argument("--speed-config", default=None,
                    help="inference-speed YAML (default: configs/inference_speed.yaml)")
    ap.add_argument("--sampler", default=None, choices=["ddpm", "ddim"],
                    help="override runtime.sampler from the speed config")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="override runtime.num_inference_steps (ddim only)")
    args = ap.parse_args()
    if args.backend == "trt" and not args.engine_dir:
        raise SystemExit("[fatal] --backend trt needs --engine-dir")
    _CFG = vars(args)
    _BUILD_AGENT = _select_builder(args)
    print("[navdp-trt] backend=%s port=%d engine-dir=%s"
          % (args.backend, args.port, args.engine_dir))
    app.run(host="127.0.0.1", port=args.port, threaded=False, processes=1)


if __name__ == "__main__":
    main()
