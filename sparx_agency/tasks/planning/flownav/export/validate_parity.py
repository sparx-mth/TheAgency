"""Authoritative FP32 parity proof for the FlowNav export (x86, CPU EP).

Two deterministic FP32 checks, before any FP16/TensorRT step muddies the picture:

  1. **Graph parity** -- for each of the three engines, ONNXRuntime (CPU EP) vs the
     torch wrapper it was exported from, on identical seeded inputs. This blesses
     the export-time graph surgery (DINOv2 pos-embed pre-bake, EfficientNet swish
     swap, SDPA decomposition, constant folding).
  2. **Scheduler parity** -- the pure-numpy :class:`FlowMatchEulerScheduler` vs
     ``torchdiffeq.odeint(method="euler")`` driving the SAME torch velocity field
     from the SAME initial state. This blesses the numpy integrator that the
     ROS-free runtime uses in place of torchdiffeq.

Run on x86 (FlowNav build env with onnxruntime + torchdiffeq):
    python -m sparx_agency.tasks.planning.flownav.export.validate_parity \
        --onnx-dir .../engines/onnx --ckpt .../flownav_weights.pth \
        --flownav-repo ~/PycharmProjects/flownav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sparx_agency.core.planning.flownav.trt.policy import (
    DIST_IN_COND, ENC_IN_GOAL, ENC_IN_OBS, VF_IN_COND, VF_IN_SAMPLE, VF_IN_TIME,
)
from sparx_agency.core.planning.flownav.trt.scheduler import FlowMatchEulerScheduler
from sparx_agency.tasks.planning.flownav.engine.inspect_onnx import load_build_policy
from sparx_agency.tasks.planning.flownav.export import io_spec
from sparx_agency.tasks.planning.flownav.export.build_model import build_flownav_model
from sparx_agency.tasks.planning.flownav.export.wrappers import (
    DistWrapper, EncoderWrapper, VFieldWrapper,
)

_DEFAULT_TOL = {"encoder_rel_l2": 1.0e-3, "vfield_rel_l2": 2.0e-3, "dist_rel_l2": 2.0e-3}
TOL = load_build_policy().get("parity_tolerances", _DEFAULT_TOL)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _session(onnx_path):
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _report(name, rel, tol):
    ok = rel <= tol
    print("  %-16s rel_l2=%.2e (tol %.1e) %s" % (name, rel, tol, "OK" if ok else "FAIL"))
    return ok


@torch.no_grad()
def check_encoder(model, onnx_dir, rng):
    enc = EncoderWrapper(model.vision_encoder).eval()
    obs = rng.randn(*io_spec.shapes(io_spec.ENCODER)[ENC_IN_OBS]).astype(np.float32)
    goal = rng.randn(*io_spec.shapes(io_spec.ENCODER)[ENC_IN_GOAL]).astype(np.float32)
    ref = enc(torch.from_numpy(obs), torch.from_numpy(goal)).cpu().numpy()
    out = _session(Path(onnx_dir) / (io_spec.ENCODER + ".onnx")).run(
        None, {ENC_IN_OBS: obs, ENC_IN_GOAL: goal})[0]
    return _report("encoder", _rel_l2(out, ref), TOL["encoder_rel_l2"])


@torch.no_grad()
def check_vfield(model, onnx_dir, rng):
    vf = VFieldWrapper(model.noise_pred_net).eval()
    sh = io_spec.shapes(io_spec.VFIELD)
    sample = rng.randn(*sh[VF_IN_SAMPLE]).astype(np.float32)
    t = np.array([0.37], np.float32)
    cond = rng.randn(*sh[VF_IN_COND]).astype(np.float32)
    ref = vf(torch.from_numpy(sample), torch.from_numpy(t), torch.from_numpy(cond)).cpu().numpy()
    out = _session(Path(onnx_dir) / (io_spec.VFIELD + ".onnx")).run(
        None, {VF_IN_SAMPLE: sample, VF_IN_TIME: t, VF_IN_COND: cond})[0]
    return _report("vfield", _rel_l2(out, ref), TOL["vfield_rel_l2"])


@torch.no_grad()
def check_dist(model, onnx_dir, rng):
    dist = DistWrapper(model.dist_pred_net).eval()
    cond = rng.randn(*io_spec.shapes(io_spec.DIST)[DIST_IN_COND]).astype(np.float32)
    ref = dist(torch.from_numpy(cond)).cpu().numpy()
    out = _session(Path(onnx_dir) / (io_spec.DIST + ".onnx")).run(
        None, {DIST_IN_COND: cond})[0]
    return _report("dist", _rel_l2(out, ref), TOL["dist_rel_l2"])


@torch.no_grad()
def check_scheduler(model, rng, num_steps=6):
    """Numpy Euler vs torchdiffeq Euler driving the same torch velocity field."""
    import torchdiffeq

    vf = VFieldWrapper(model.noise_pred_net).eval()
    sh = io_spec.shapes(io_spec.VFIELD)
    x0 = rng.randn(*sh[VF_IN_SAMPLE]).astype(np.float32)
    cond = torch.from_numpy(rng.randn(*sh[VF_IN_COND]).astype(np.float32))

    def field(t, x):
        tt = t.reshape(1).float()
        return vf(x, tt, cond)

    traj = torchdiffeq.odeint(field, torch.from_numpy(x0),
                              torch.linspace(0, 1, num_steps), method="euler")
    ref = traj[-1].cpu().numpy()

    sched = FlowMatchEulerScheduler(num_steps)
    x = x0.copy()
    for i in range(sched.num_field_evals):
        tt = torch.tensor([sched.timesteps[i]], dtype=torch.float32)
        v = vf(torch.from_numpy(x), tt, cond).cpu().numpy()
        x = sched.step(v, i, x)
    return _report("euler_scheduler", _rel_l2(x, ref), 1e-5)


def validate(onnx_dir, ckpt, flownav_repo=None, seed=0):
    """Run all parity checks; raise if any fails."""
    model = build_flownav_model(ckpt, flownav_repo=flownav_repo, device="cpu")
    rng = np.random.RandomState(seed)
    print("[parity] FP32 ONNXRuntime vs torch + numpy-vs-torchdiffeq scheduler")
    results = [
        check_encoder(model, onnx_dir, rng),
        check_vfield(model, onnx_dir, rng),
        check_dist(model, onnx_dir, rng),
        check_scheduler(model, rng),
    ]
    if not all(results):
        raise RuntimeError("FlowNav parity FAILED; see report above.")
    print("[parity] all checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flownav-repo", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    validate(args.onnx_dir, args.ckpt, flownav_repo=args.flownav_repo, seed=args.seed)


if __name__ == "__main__":
    main()
