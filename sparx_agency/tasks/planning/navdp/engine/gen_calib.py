"""Capture an INT8 calibration ``.npz`` for the NavDP engines from the torch model.

INT8 entropy calibration (``engine/calibrator.py``) needs representative inputs
for each engine, keyed ``"<engine_key>/<input_name>"`` -> ``(num_samples,
*engine_input_shape)`` -- exactly what ``build_engine._load_calibrators``
consumes. Per the calibrator docstring:

  * **encoder**: ``process_image`` / ``process_depth`` captures. Pass real frames
    via ``--frames`` for a shippable INT8 encoder; random RGB-D is a *bootstrap*
    only (uniform noise is out-of-distribution for the DINOv2 ViT, so its INT8
    activation ranges will be off and the encoder may miss the stricter gate).
  * **denoise / critic**: tensors captured from a **full denoise loop across all
    timesteps** (so the ``last_actions`` distribution spans noisy -> clean), NOT a
    single step.

This runs the **baseline torch model** -- ``build_navdp_policy`` + the same three
export wrappers ``benchmark.bench.TorchReference`` uses -- so it does NOT require
working TensorRT engines. That breaks the INT8 chicken-and-egg: you can calibrate
straight from the checkpoint. (Capturing from a built FP16 TRT-in-loop run is a
touch higher-fidelity, but the FP16-vs-FP32 loop divergence is far smaller than
the quantization granularity, so torch-loop data is representative for range
estimation and the on-device INT8 gate is the backstop.)

Dev/host only (imports torch + the external NavDP repo); never imported by
``core``.

Run (on the Orin, ``navdp`` env; ``PYTHONPATH`` = repo root):
    python -m sparx_agency.tasks.planning.navdp.engine.gen_calib \
        --ckpt $NAVDP_REPO/checkpoints/best.pth --navdp-repo $NAVDP_REPO \
        --out sparx_agency/tasks/planning/navdp/engines/onnx/calib.npz \
        --num-scenarios 64            # add --frames real_rgbd.npz for a shippable encoder
then feed the ``.npz`` to the INT8 build:
    python -m sparx_agency.tasks.planning.navdp.engine.build_engine \
        --onnx-dir .../engines/onnx --precision int8 --calib-npz .../engines/onnx/calib.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sparx_agency.core.planning.navdp.trt.point_encoder import NavDPPointEncoder
from sparx_agency.core.planning.navdp.trt.scheduler import NumpyDDPMScheduler
from sparx_agency.tasks.planning.navdp.export import io_spec
from sparx_agency.tasks.planning.navdp.export.build_policy import build_navdp_policy
from sparx_agency.tasks.planning.navdp.export.wrappers import (
    CriticWrapper, DenoiseStepWrapper, EncoderWrapper,
)

# One .npz key per engine input, matching build_engine._load_calibrators
# ("<engine_key>/<input_name>") and io_spec's names/shapes exactly.
ENC, DEN, CRI = io_spec.ENCODER, io_spec.DENOISE, io_spec.CRITIC
CALIB_KEYS = tuple(
    "%s/%s" % (k, n) for k in (ENC, DEN, CRI) for n in io_spec.input_names(k))


def _scenarios(frames_path, num, rng):
    """Yield ``(images_bthwc, depth_bhwc, goal)`` scenarios.

    Real captures from ``frames_path`` (an ``.npz`` with ``images``
    ``(K,8,224,224,3)``, ``depth`` ``(K,224,224,1)`` and optional ``goals``
    ``(K,3)``) when given, else random RGB-D (bootstrap; see the module docstring).
    """
    if frames_path:
        data = np.load(frames_path)
        imgs = np.asarray(data["images"], np.float32)
        deps = np.asarray(data["depth"], np.float32)
        goals = np.asarray(data["goals"], np.float32) if "goals" in data else None
        k = imgs.shape[0]
        if k == 0:
            raise SystemExit("--frames %s has no frames" % frames_path)
        for i in range(num):
            j = i % k
            img = imgs[j][None] if imgs.ndim == 4 else imgs[j:j + 1]   # -> (1,8,224,224,3)
            dep = deps[j][None] if deps.ndim == 3 else deps[j:j + 1]   # -> (1,224,224,1)
            if goals is not None:
                goal = goals[j % goals.shape[0]].reshape(1, 3).astype(np.float32)
            else:
                goal = np.array([[rng.uniform(0.5, 8.0), rng.uniform(-3, 3), 0.0]], np.float32)
            yield img, dep, goal
        return
    for _ in range(num):
        img = rng.rand(1, io_spec.MEM, io_spec.IMG, io_spec.IMG, 3).astype(np.float32)
        dep = (rng.rand(1, io_spec.IMG, io_spec.IMG, 1) * 5.0).astype(np.float32)
        goal = np.array([[rng.uniform(0.5, 8.0), rng.uniform(-3, 3), 0.0]], np.float32)
        yield img, dep, goal


def capture(policy, head_npz, scenarios_iter, device):
    """Run the torch model over scenarios, returning ``{calib_key: stacked array}``.

    Mirrors ``benchmark.bench.TorchReference.run`` (same wrappers, same numpy
    scheduler) but records each engine input at its exact ``io_spec`` shape.
    """
    enc = EncoderWrapper(policy.rgbd_encoder).to(device).eval()
    den = DenoiseStepWrapper(policy).to(device).eval()
    cri = CriticWrapper(policy).to(device).eval()

    p = np.load(head_npz)
    point_encoder = NavDPPointEncoder(p["point_encoder_weight"], p["point_encoder_bias"])
    time_table = np.asarray(p["time_table"], np.float32)
    scheduler = NumpyDDPMScheduler(p["alphas_cumprod"])
    n = io_spec.N
    rng = np.random.RandomState(0)

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)

    buf = {k: [] for k in CALIB_KEYS}
    with torch.no_grad():
        for img_bthwc, dep_bhwc, goal in scenarios_iter:
            nchw_img = t(img_bthwc).permute(0, 1, 4, 2, 3).contiguous()   # (1,8,3,224,224)
            nchw_dep = t(dep_bhwc).permute(0, 3, 1, 2).contiguous()       # (1,1,224,224)
            buf["%s/%s" % (ENC, io_spec.ENC_IN_IMAGES)].append(
                nchw_img.cpu().numpy().astype(np.float32))               # (1,8,3,224,224)
            buf["%s/%s" % (ENC, io_spec.ENC_IN_DEPTH)].append(
                nchw_dep.cpu().numpy().astype(np.float32))               # (1,1,224,224)

            rgbd = enc(nchw_img, nchw_dep)                                # (1,128,384)
            rgbd_n = np.repeat(rgbd.cpu().numpy().astype(np.float32), n, axis=0)  # (16,128,384)
            goal_embed = point_encoder(goal)                             # (1,384)
            goal_n = np.repeat(goal_embed[:, None, :], n, axis=0).astype(np.float32)  # (16,1,384)
            rgbd_nt, goal_nt = t(rgbd_n), t(goal_n)

            naction = rng.randn(n, io_spec.PREDICT, 3).astype(np.float32)
            for k in scheduler.timesteps:
                k = int(k)
                tt = np.repeat(time_table[k][None, None, :], n, axis=0).astype(np.float32)  # (16,1,384)
                buf["%s/%s" % (DEN, io_spec.DEN_IN_ACTIONS)].append(naction.copy())
                buf["%s/%s" % (DEN, io_spec.DEN_IN_TIME)].append(tt)
                buf["%s/%s" % (DEN, io_spec.DEN_IN_GOAL)].append(goal_n)
                buf["%s/%s" % (DEN, io_spec.DEN_IN_RGBD)].append(rgbd_n)
                noise = den(t(naction), t(tt), goal_nt, rgbd_nt).cpu().numpy().astype(np.float32)
                naction = scheduler.step(noise, k, naction)

            buf["%s/%s" % (CRI, io_spec.CRI_IN_TRAJ)].append(naction.astype(np.float32))
            buf["%s/%s" % (CRI, io_spec.CRI_IN_RGBD)].append(rgbd_n)

    stacked = {k: np.stack(v, axis=0).astype(np.float32) for k, v in buf.items() if v}
    _assert_shapes(stacked)
    return stacked


def _assert_shapes(stacked):
    """Fail loud if any stacked array's per-sample shape != the engine input shape."""
    for key, arr in stacked.items():
        engine, name = key.split("/", 1)
        want = tuple(io_spec.shapes(engine)[name])
        got = tuple(arr.shape[1:])
        if got != want:
            raise SystemExit("calib key %r per-sample shape %r != engine input %r"
                             % (key, got, want))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--out", required=True, help="calib .npz path (feed to build_engine --calib-npz)")
    ap.add_argument("--head-params", default=None,
                    help="head params npz (default: navdp_head_params.npz next to --out)")
    ap.add_argument("--frames", default=None,
                    help="npz of real {images:(K,8,224,224,3), depth:(K,224,224,1)[, goals:(K,3)]} "
                         "for the ENCODER; random RGB-D bootstrap if omitted")
    ap.add_argument("--num-scenarios", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    head = Path(args.head_params) if args.head_params else out.parent / "navdp_head_params.npz"
    if not head.exists():
        raise SystemExit("head params npz not found at %s; run export_onnx first "
                         "(it writes navdp_head_params.npz) or pass --head-params." % head)
    if args.frames is None:
        print("[warn] no --frames: ENCODER calibrated on RANDOM RGB-D (out-of-distribution "
              "for the ViT). INT8 will BUILD but may miss the stricter accuracy gate; pass "
              "--frames with real captures for a shippable INT8 encoder.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = build_navdp_policy(args.ckpt, navdp_repo=args.navdp_repo, device=device)
    rng = np.random.RandomState(args.seed)
    scenarios = _scenarios(args.frames, args.num_scenarios, rng)
    stacked = capture(policy, head, scenarios, device)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **stacked)
    print("[done] wrote", out)
    for k in CALIB_KEYS:
        print("  %-34s %s" % (k, stacked[k].shape))


if __name__ == "__main__":
    main()
