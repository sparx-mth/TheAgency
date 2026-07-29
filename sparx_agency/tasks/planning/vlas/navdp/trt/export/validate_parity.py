"""Validate the exported ONNX graphs against the PyTorch reference (FP32, CPU).

Two deterministic comparisons per engine, both at FP32 on the CPU EP (so there
is no silent GPU/sm_120 divergence and the diffusion stochasticity is excluded):

  1. **wrapper vs original** -- the export wrapper's output vs the original
     ``NavDP`` module computation on identical inputs. This blesses the two
     deliberate changes: ``-inf`` -> ``-1e4`` attention masks and precomputing
     the sinusoidal time embedding in Python.
  2. **onnx vs wrapper** -- onnxruntime (CPU) vs the torch wrapper on identical
     inputs. This blesses the export itself (op decomposition, constant folding).

The encoder is gated on relative-L2 / max-abs (the critic is scale/offset
sensitive, so cosine alone is not enough); denoiser/critic on relative-L2.
Raises on any gate failure. This is the authoritative numeric proof and runs
fully at home on x86 (the FP16/INT8 ranking gate is a separate on-target step).

Run (navdp conda env with onnx + onnxruntime; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.vlas.navdp.trt.export.validate_parity \
        --onnx-dir .../engines/onnx --ckpt ... --navdp-repo ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sparx_agency.tasks.planning.vlas.navdp.trt.export import io_spec
from sparx_agency.tasks.planning.vlas.navdp.trt.export.build_policy import build_navdp_policy
from sparx_agency.tasks.planning.vlas.navdp.trt.export.wrappers import (
    CriticWrapper, DenoiseStepWrapper, EncoderWrapper,
)

ENC_REL_L2 = 1.0e-3
DEN_REL_L2 = 2.0e-3
CRI_REL_L2 = 2.0e-3


def rel_l2(a, b):
    """Relative L2 error ``||a-b|| / (||b|| + eps)``."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def max_abs(a, b):
    """Max absolute elementwise error."""
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _session(onnx_path):
    import onnxruntime as ort
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _run_onnx(sess, feeds):
    return sess.run(None, {k: np.asarray(v, np.float32) for k, v in feeds.items()})[0]


def check_encoder(policy, onnx_dir, rng):
    """Encoder: wrapper==original rgbd_encoder, and onnx==wrapper."""
    wrapper = EncoderWrapper(policy.rgbd_encoder).eval()
    images = rng.randn(1, io_spec.MEM, io_spec.IMG, io_spec.IMG, 3).astype(np.float32)
    depth = np.abs(rng.randn(1, io_spec.IMG, io_spec.IMG, 1).astype(np.float32))
    with torch.no_grad():
        original = policy.rgbd_encoder(torch.from_numpy(images),
                                       torch.from_numpy(depth)).cpu().numpy()
        nchw_img = torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()
        nchw_dep = torch.from_numpy(depth).permute(0, 3, 1, 2).contiguous()
        wrapped = wrapper(nchw_img, nchw_dep).cpu().numpy()
    onnx_out = _run_onnx(_session(onnx_dir / "navdp_encoder.onnx"),
                         {io_spec.SPECS[io_spec.ENCODER][0][0]: nchw_img.numpy(),
                          io_spec.SPECS[io_spec.ENCODER][0][1]: nchw_dep.numpy()})
    return _report("encoder", [("wrapper_vs_original", original, wrapped),
                               ("onnx_vs_wrapper", wrapped, onnx_out)], ENC_REL_L2)


def check_denoise(policy, onnx_dir, rng):
    """Denoiser: wrapper==predict_noise, and onnx==wrapper."""
    wrapper = DenoiseStepWrapper(policy).eval()
    sh = io_spec.shapes(io_spec.DENOISE)
    la = rng.randn(*sh[io_spec.SPECS[io_spec.DENOISE][0][0]]).astype(np.float32)
    goal = rng.randn(*sh[io_spec.SPECS[io_spec.DENOISE][0][2]]).astype(np.float32)
    rgbd = rng.randn(*sh[io_spec.SPECS[io_spec.DENOISE][0][3]]).astype(np.float32)
    timestep = torch.tensor([7], dtype=torch.float32)
    with torch.no_grad():
        time_token = policy.time_emb(timestep).unsqueeze(1).tile((la.shape[0], 1, 1))
        original = policy.predict_noise(torch.from_numpy(la), timestep,
                                        torch.from_numpy(goal), torch.from_numpy(rgbd)).cpu().numpy()
        wrapped = wrapper(torch.from_numpy(la), time_token,
                          torch.from_numpy(goal), torch.from_numpy(rgbd)).cpu().numpy()
    names = io_spec.SPECS[io_spec.DENOISE][0]
    onnx_out = _run_onnx(_session(onnx_dir / "navdp_denoise.onnx"),
                         {names[0]: la, names[1]: time_token.numpy(),
                          names[2]: goal, names[3]: rgbd})
    return _report("denoise", [("wrapper_vs_original", original, wrapped),
                               ("onnx_vs_wrapper", wrapped, onnx_out)], DEN_REL_L2)


def check_critic(policy, onnx_dir, rng):
    """Critic: wrapper==predict_critic, and onnx==wrapper."""
    wrapper = CriticWrapper(policy).eval()
    sh = io_spec.shapes(io_spec.CRITIC)
    names = io_spec.SPECS[io_spec.CRITIC][0]
    traj = rng.randn(*sh[names[0]]).astype(np.float32)
    rgbd = rng.randn(*sh[names[1]]).astype(np.float32)
    with torch.no_grad():
        original = policy.predict_critic(torch.from_numpy(traj),
                                         torch.from_numpy(rgbd)).cpu().numpy().reshape(-1, 1)
        wrapped = wrapper(torch.from_numpy(traj), torch.from_numpy(rgbd)).cpu().numpy()
    onnx_out = _run_onnx(_session(onnx_dir / "navdp_critic.onnx"),
                         {names[0]: traj, names[1]: rgbd})
    return _report("critic", [("wrapper_vs_original", original, wrapped),
                              ("onnx_vs_wrapper", wrapped, onnx_out)], CRI_REL_L2)


def _report(name, pairs, threshold):
    """Print metrics for each comparison and return True if all pass."""
    ok = True
    for label, ref, got in pairs:
        r = rel_l2(got, ref)
        m = max_abs(got, ref)
        passed = r < threshold
        ok = ok and passed
        print("  [%s] %-20s rel_l2=%.2e max_abs=%.2e  %s"
              % (name, label, r, m, "PASS" if passed else "FAIL"))
    return ok


def validate(onnx_dir, ckpt, navdp_repo=None, seed=0):
    """Run all three engine parity checks; raise on any failure."""
    onnx_dir = Path(onnx_dir)
    policy = build_navdp_policy(ckpt, navdp_repo=navdp_repo, device="cpu")
    rng = np.random.RandomState(seed)
    results = [check_encoder(policy, onnx_dir, rng),
               check_denoise(policy, onnx_dir, rng),
               check_critic(policy, onnx_dir, rng)]
    if not all(results):
        raise RuntimeError("ONNX parity validation FAILED -- see metrics above.")
    print("[done] ONNX parity validation PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    validate(args.onnx_dir, args.ckpt, navdp_repo=args.navdp_repo, seed=args.seed)


if __name__ == "__main__":
    main()
