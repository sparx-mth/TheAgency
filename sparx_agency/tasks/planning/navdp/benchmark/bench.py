"""Benchmark NavDP TRT engines: FPS + accuracy gate, then pick the precision.

For each available precision (FP16, optionally INT8) this measures end-to-end and
per-engine FPS at the real operating point (1 encode + 10 denoise + 1 critic),
and runs an accuracy gate whose metrics are the *decision* failures that matter
for navigation, not raw MSE:

  * **argmax flip rate** -- does the executed (highest-critic) sample differ from
    the FP32 torch reference? (the single most important metric.)
  * **stop-decision match** -- does ``critic.max() < stop_threshold`` agree?
  * **chosen-sample <0.5 zeroing match** -- does the executed trajectory get
    zeroed the same way (this changes what the drone flies)?
  * chosen-trajectory L2 (diagnostic).

Both paths use IDENTICAL injected noise (the only way the comparison is
meaningful) and the SAME numpy scheduler / point-encoder / post-processing; only
the three transformer forwards differ (TRT engines vs FP32 torch wrappers). The
fastest precision passing its gate is written to ``selected.json``; INT8 is only
chosen if it passes a stricter gate ON THIS device. Run on the target with the
TRT venv.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparx_agency.core.planning.navdp.trt.errors import NavDPError
from sparx_agency.core.planning.navdp.trt.point_encoder import NavDPPointEncoder
from sparx_agency.core.planning.navdp.trt.policy import NavDPTRTPolicy
from sparx_agency.core.planning.navdp.trt.postprocess import finalize_trajectories
from sparx_agency.core.planning.navdp.trt.scheduler import NumpyDDPMScheduler
from sparx_agency.tasks.planning.navdp.engine import build_engine
from sparx_agency.tasks.planning.navdp.engine.inspect_onnx import load_build_policy
from sparx_agency.tasks.planning.navdp.export import io_spec
from sparx_agency.tasks.planning.navdp.export.build_policy import build_navdp_policy
from sparx_agency.tasks.planning.navdp.export.wrappers import (
    CriticWrapper, DenoiseStepWrapper, EncoderWrapper,
)
from sparx_agency.tasks.planning.navdp.hardware.detect import detect

# Gate thresholds (declared, tune on target). INT8 must clear a stricter bar.
# Sourced from configs/build_policy.json; these are the fallback defaults.
_DEFAULT_GATES = {"fp16": {"argmax_flip": 0.05, "stop_match": 0.98, "zero_match": 0.98},
                  "int8": {"argmax_flip": 0.02, "stop_match": 0.99, "zero_match": 0.99}}
GATES = load_build_policy().get("accuracy_gates", _DEFAULT_GATES)


class TorchReference:
    """FP32 torch reference pipeline (same numpy orchestration as the runtime)."""

    def __init__(self, policy, head_npz, device):
        self.device = device
        self.enc = EncoderWrapper(policy.rgbd_encoder).to(device).eval()
        self.den = DenoiseStepWrapper(policy).to(device).eval()
        self.cri = CriticWrapper(policy).to(device).eval()
        p = np.load(head_npz)
        self.point_encoder = NavDPPointEncoder(p["point_encoder_weight"], p["point_encoder_bias"])
        self.time_table = np.asarray(p["time_table"], np.float32)
        self.scheduler = NumpyDDPMScheduler(p["alphas_cumprod"])
        self.n = io_spec.N

    def _t(self, a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(self.device)

    @torch.no_grad()
    def run(self, goal, images_bthwc, depth_bhwc, init_noise, variance_noises):
        nchw_img = self._t(images_bthwc).permute(0, 1, 4, 2, 3).contiguous()
        nchw_dep = self._t(depth_bhwc).permute(0, 3, 1, 2).contiguous()
        rgbd = self.enc(nchw_img, nchw_dep)                          # (1,128,384)
        rgbd_n = rgbd.repeat_interleave(self.n, dim=0)
        goal_embed = self.point_encoder(np.asarray(goal, np.float32))
        goal_n = self._t(np.repeat(goal_embed[:, None, :], self.n, axis=0))
        naction = np.asarray(init_noise, np.float32)
        for step, k in enumerate(self.scheduler.timesteps):
            tt = self._t(np.repeat(self.time_table[int(k)][None, None, :], self.n, axis=0))
            noise = self.den(self._t(naction), tt, goal_n, rgbd_n).cpu().numpy()
            naction = self.scheduler.step(noise, int(k), naction, variance_noise=variance_noises[step])
        critic = self.cri(self._t(naction), rgbd_n).cpu().numpy().reshape(-1)
        return finalize_trajectories(naction, critic, 1, self.n)


def gen_scenarios(num, seed=0):
    """Random RGB-D + goal scenarios with fixed injected noise (per scenario)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(num):
        images = rng.rand(1, io_spec.MEM, io_spec.IMG, io_spec.IMG, 3).astype(np.float32)
        depth = (rng.rand(1, io_spec.IMG, io_spec.IMG, 1) * 5.0).astype(np.float32)
        goal = np.array([[rng.uniform(0.5, 8.0), rng.uniform(-3, 3), 0.0]], np.float32)
        init_noise = rng.randn(io_spec.N, io_spec.PREDICT, 3).astype(np.float32)
        var = rng.randn(10, io_spec.N, io_spec.PREDICT, 3).astype(np.float32)
        out.append((goal, images, depth, init_noise, var))
    return out


def _chosen(result):
    """(executed trajectory, argmax index, critic.max, chosen-sample zeroed?)."""
    all_traj, critic, positive, _ = result
    # Use the SAME ordering that produced the executed trajectory (positive[0,0]
    # comes from argsort(-critic)), so the flip/zero checks describe the sample
    # the drone would actually fly, even on an exact critic tie.
    idx = int(np.argsort(-critic[0])[0])
    length = float(np.linalg.norm(all_traj[0, idx, -1, 0:2]))
    return positive[0, 0], idx, float(critic[0].max()), length < 0.5


def accuracy(policy, ref, scenarios, stop_threshold):
    """Compare TRT vs torch over scenarios; return decision-flip metrics."""
    flips = stop_mismatch = zero_mismatch = 0
    l2s = []
    for goal, img, dep, init_noise, var in scenarios:
        rt = policy.predict_pointgoal_action(goal, img, dep, init_noise=init_noise, variance_noises=var)
        rr = ref.run(goal, img, dep, init_noise, var)
        et, it, ct, zt = _chosen(rt)
        er, ir, cr, zr = _chosen(rr)
        flips += int(it != ir)
        stop_mismatch += int((ct < stop_threshold) != (cr < stop_threshold))
        zero_mismatch += int(zt != zr)
        l2s.append(float(np.linalg.norm(et - er)))
    n = len(scenarios)
    return {"argmax_flip": flips / n, "stop_match": 1 - stop_mismatch / n,
            "zero_match": 1 - zero_mismatch / n, "traj_l2_mean": float(np.mean(l2s))}


def fps(policy, scenario, warmup=3, iters=20):
    """End-to-end Hz over repeated inference on one scenario."""
    goal, img, dep, init_noise, var = scenario
    for _ in range(warmup):
        policy.predict_pointgoal_action(goal, img, dep, init_noise=init_noise, variance_noises=var)
    t0 = time.perf_counter()
    for _ in range(iters):
        policy.predict_pointgoal_action(goal, img, dep, init_noise=init_noise, variance_noises=var)
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms": dt * 1000.0, "hz": 1.0 / dt}


def _write_selected(engine_dir, precision):
    sel = {"precision": precision, "engines": {
        k: "%s.%s.engine" % (n, precision) for k, n in
        (("encoder", io_spec.ENCODER), ("denoise", io_spec.DENOISE), ("critic", io_spec.CRITIC))}}
    (engine_dir / "selected.json").write_text(json.dumps(sel, indent=2))


def evaluate(precision, engine_dir, npz, ref, scenarios, stop_threshold):
    """Build a runtime for one precision, measure FPS + accuracy, return a row."""
    _write_selected(engine_dir, precision)
    policy = NavDPTRTPolicy(engine_dir, npz)
    acc = accuracy(policy, ref, scenarios, stop_threshold)
    perf = fps(policy, scenarios[0])
    gate = GATES[precision]
    passed = (acc["argmax_flip"] <= gate["argmax_flip"]
              and acc["stop_match"] >= gate["stop_match"]
              and acc["zero_match"] >= gate["zero_match"])
    return {"precision": precision, "passed": bool(passed), **acc, **perf}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-dir", required=True, help="dir with <name>.<prec>.engine + npz")
    ap.add_argument("--head-params", default=None, help="head params npz (default: <engine-dir>/navdp_head_params.npz)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--num-scenarios", type=int, default=32)
    ap.add_argument("--stop-threshold", type=float, default=-999.0)
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir)
    npz = Path(args.head_params) if args.head_params else engine_dir / "navdp_head_params.npz"
    if not npz.exists():
        npz = next(engine_dir.glob("*head_params*.npz"), None)
    if npz is None or not npz.exists():
        raise NavDPError("no head_params npz in %s; pass --head-params or copy it "
                         "from the onnx export dir (build_engine now copies it)." % engine_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_torch = build_navdp_policy(args.ckpt, navdp_repo=args.navdp_repo, device=device)
    ref = TorchReference(policy_torch, npz, device)
    scenarios = gen_scenarios(args.num_scenarios)

    rows = []
    try:
        for precision in ("fp16", "int8"):
            if all((engine_dir / ("%s.%s.engine" % (k, precision))).exists()
                   for k in build_engine.ENGINE_KEYS):
                print("[bench] evaluating", precision)
                rows.append(evaluate(precision, engine_dir, npz, ref, scenarios, args.stop_threshold))
        if not rows:
            raise RuntimeError("No engines found in %s" % engine_dir)
        passing = [r for r in rows if r["passed"]]
        if not passing:
            raise RuntimeError("No precision passed the accuracy gate: %s" % rows)
    except BaseException:
        # evaluate() writes selected.json per-precision as a side effect (the
        # policy resolves its engines only from that file). If no precision is
        # blessed -- or a scenario crashes mid-loop -- never leave selected.json
        # pointing at an unvalidated / gate-failed engine for the server to load.
        (engine_dir / "selected.json").unlink(missing_ok=True)
        raise
    # Prefer the fastest passing precision; FP16 wins ties (safer).
    winner = max(passing, key=lambda r: (r["hz"], r["precision"] == "fp16"))
    _write_selected(engine_dir, winner["precision"])
    report = {"target": detect().target_tag, "winner": winner["precision"], "rows": rows}
    (engine_dir / "bench_report.json").write_text(json.dumps(report, indent=2))
    print("[done] selected", winner["precision"], "->", engine_dir / "selected.json")
    for r in rows:
        print("  %-5s pass=%s hz=%.1f flip=%.3f stop=%.3f zero=%.3f"
              % (r["precision"], r["passed"], r["hz"], r["argmax_flip"],
                 r["stop_match"], r["zero_match"]))


if __name__ == "__main__":
    main()
