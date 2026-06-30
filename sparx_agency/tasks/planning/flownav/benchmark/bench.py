"""Benchmark FlowNav TRT engines: accuracy gate, K-sweep, and runtime comparison.

This is the on-target validation + selection step. It answers three questions and
writes ``selected.json`` (consumed by ``FlowNavTRTPolicy``):

1. **TRT fidelity (the accuracy gate).** At a fixed number of Euler steps K, do the
   TensorRT engines reproduce the FP32 torch model? Both paths run the IDENTICAL
   numpy Euler loop with the SAME injected initial noise; only the three forward
   passes differ (TRT engines vs FP32 torch wrappers). Metrics are decision-level:
   the executed-waypoint L2, the full-trajectory relative L2, and the distance-head
   error. FP16 is selected only if it clears the gate.

2. **The K question (low-K quality).** FlowNav's speed comes from using few flow-
   matching steps. For each K in the sweep we measure how far the torch trajectory
   at K drifts from the high-K reference trajectory (same noise) -- so a *low* K
   can be chosen with data, not guessed. The smallest K whose drift stays within
   the configured tolerance AND whose TRT engines pass the fidelity gate is written
   as ``num_steps``.

3. **With vs without TRT (runtime).** End-to-end latency of the TRT path and the
   eager-torch path at each K, so the speed-up is measured, not assumed.

Run on the target with the FlowNav build env (torch + tensorrt + the FlowNav repo):
    python -m sparx_agency.tasks.planning.flownav.benchmark.bench \
        --engine-dir .../engines/<target_tag> \
        --ckpt .../flownav_weights.pth --flownav-repo ~/PycharmProjects/flownav
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparx_agency.core.planning.flownav.trt.errors import FlowNavError
from sparx_agency.core.planning.flownav.trt.policy import FlowNavTRTPolicy
from sparx_agency.core.planning.flownav.trt.postprocess import chosen_waypoint, get_action
from sparx_agency.core.planning.flownav.trt.scheduler import FlowMatchEulerScheduler
from sparx_agency.tasks.planning.flownav.engine import build_engine
from sparx_agency.tasks.planning.flownav.engine.inspect_onnx import load_build_policy
from sparx_agency.tasks.planning.flownav.export import io_spec
from sparx_agency.tasks.planning.flownav.export.build_model import build_flownav_model
from sparx_agency.tasks.planning.flownav.export.wrappers import (
    DistWrapper, EncoderWrapper, VFieldWrapper,
)
from sparx_agency.tasks.planning.flownav.hardware.detect import detect

# Defaults (overridden by configs/build_policy.json).
_DEFAULT_GATES = {"fp16": {"waypoint_l2": 0.05, "traj_rel_l2": 0.05, "dist_abs_err": 0.25}}
_DEFAULT_K_SWEEP = [2, 3, 4, 5, 8, 10]
_DEFAULT_K_QUALITY_L2 = 0.10     # max executed-waypoint drift (m) vs the high-K ref
_DEFAULT_WAYPOINT_INDEX = 2

_CFG = load_build_policy()
GATES = _CFG.get("accuracy_gates", _DEFAULT_GATES)
K_SWEEP = _CFG.get("k_sweep", _DEFAULT_K_SWEEP)
K_QUALITY_L2 = _CFG.get("k_quality_waypoint_l2", _DEFAULT_K_QUALITY_L2)
WAYPOINT_INDEX = _CFG.get("waypoint_index", _DEFAULT_WAYPOINT_INDEX)


class TorchReference:
    """FP32 eager-torch FlowNav pipeline (same numpy Euler loop as the runtime)."""

    def __init__(self, model, head_npz, device):
        self.device = device
        self.enc = EncoderWrapper(model.vision_encoder).to(device).eval()
        self.vf = VFieldWrapper(model.noise_pred_net).to(device).eval()
        self.dist = DistWrapper(model.dist_pred_net).to(device).eval()
        p = np.load(head_npz)
        self.action_min = np.asarray(p["action_min"], np.float32)
        self.action_max = np.asarray(p["action_max"], np.float32)
        self.n = io_spec.N

    def _t(self, a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(self.device)

    @torch.no_grad()
    def encode(self, obs_img, goal_img):
        return self.enc(self._t(obs_img), self._t(goal_img)).cpu().numpy()

    @torch.no_grad()
    def run(self, obs_img, goal_img, init_noise, num_steps):
        """Return (actions (N,8,2), distance) for the eager-torch model at K steps."""
        cond = self.enc(self._t(obs_img), self._t(goal_img))            # (1,256)
        cond_n = cond.repeat_interleave(self.n, dim=0)                  # (N,256)
        dist = float(self.dist(cond).cpu().numpy().reshape(-1)[0])
        sched = FlowMatchEulerScheduler(num_steps)
        x = np.asarray(init_noise, np.float32)
        for i in range(sched.num_field_evals):
            t = self._t(np.array([sched.timesteps[i]], np.float32))
            vfield = self.vf(self._t(x), t, cond_n).cpu().numpy()
            x = sched.step(vfield, i, x)
        return get_action(x, self.action_min, self.action_max), dist


def gen_scenarios(num, seed=0):
    """Random obs/goal + fixed injected initial noise (per scenario)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(num):
        obs = rng.randn(1, io_spec.OBS_CH, io_spec.IMG, io_spec.IMG).astype(np.float32)
        goal = rng.randn(1, 3, io_spec.IMG, io_spec.IMG).astype(np.float32)
        init_noise = rng.randn(io_spec.N, io_spec.HORIZON, io_spec.ACT_DIM).astype(np.float32)
        out.append((obs, goal, init_noise))
    return out


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _waypoint_l2(actions_a, actions_b):
    wa = chosen_waypoint(actions_a, WAYPOINT_INDEX)
    wb = chosen_waypoint(actions_b, WAYPOINT_INDEX)
    return float(np.linalg.norm(wa - wb))


def fidelity(policy, ref, scenarios, num_steps):
    """TRT vs torch agreement at K steps (the accuracy gate)."""
    wp_l2, tr_l2, d_err = [], [], []
    for obs, goal, init_noise in scenarios:
        a_trt, d_trt = policy.predict(obs, goal, init_noise=init_noise)
        a_ref, d_ref = ref.run(obs, goal, init_noise, num_steps)
        wp_l2.append(_waypoint_l2(a_trt, a_ref))
        tr_l2.append(_rel_l2(a_trt, a_ref))
        d_err.append(abs(d_trt - d_ref))
    return {"waypoint_l2": float(np.mean(wp_l2)), "traj_rel_l2": float(np.mean(tr_l2)),
            "dist_abs_err": float(np.mean(d_err))}


def k_quality(ref, scenarios, num_steps, ref_steps):
    """Torch-at-K vs torch-at-ref_steps trajectory drift (the low-K cost)."""
    wp_l2, tr_l2 = [], []
    for obs, goal, init_noise in scenarios:
        a_k, _ = ref.run(obs, goal, init_noise, num_steps)
        a_ref, _ = ref.run(obs, goal, init_noise, ref_steps)
        wp_l2.append(_waypoint_l2(a_k, a_ref))
        tr_l2.append(_rel_l2(a_k, a_ref))
    return {"waypoint_l2": float(np.mean(wp_l2)), "traj_rel_l2": float(np.mean(tr_l2))}


def latency_ms(fn, scenario, warmup=3, iters=20):
    """Median-free mean latency (ms) of ``fn(scenario)``."""
    for _ in range(warmup):
        fn(scenario)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(scenario)
    return (time.perf_counter() - t0) / iters * 1000.0


def _write_selected(engine_dir, precision, num_steps):
    sel = {"precision": precision,
           "engines": {"encoder": "%s.%s.engine" % (io_spec.ENCODER, precision),
                       "vfield": "%s.%s.engine" % (io_spec.VFIELD, precision),
                       "dist": "%s.%s.engine" % (io_spec.DIST, precision)},
           "num_samples": io_spec.N, "num_steps": int(num_steps),
           "horizon": io_spec.HORIZON, "action_dim": io_spec.ACT_DIM}
    (engine_dir / "selected.json").write_text(json.dumps(sel, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--head-params", default=None,
                    help="default: <engine-dir>/flownav_head_params.npz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flownav-repo", default=None)
    ap.add_argument("--num-scenarios", type=int, default=16)
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir)
    npz = Path(args.head_params) if args.head_params else engine_dir / build_engine.HEAD_PARAMS_NPZ
    if not npz.exists():
        npz = next(engine_dir.glob("*head_params*.npz"), None)
    if npz is None or not npz.exists():
        raise FlowNavError("no head_params npz in %s; pass --head-params." % engine_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_flownav_model(args.ckpt, flownav_repo=args.flownav_repo, device=device)
    ref = TorchReference(model, npz, device)
    scenarios = gen_scenarios(args.num_scenarios)
    precision = "fp16"
    gate = GATES.get(precision, _DEFAULT_GATES["fp16"])
    ref_steps = max(K_SWEEP)

    rows, chosen_k = [], None
    for k in sorted(set(K_SWEEP)):
        _write_selected(engine_dir, precision, k)
        policy = FlowNavTRTPolicy(engine_dir, npz, num_steps=k)
        fid = fidelity(policy, ref, scenarios, k)
        qual = k_quality(ref, scenarios, k, ref_steps)
        trt_ms = latency_ms(lambda s: policy.predict(s[0], s[1], init_noise=s[2]), scenarios[0])
        torch_ms = latency_ms(lambda s: ref.run(s[0], s[1], s[2], k), scenarios[0])
        passed_fid = (fid["waypoint_l2"] <= gate["waypoint_l2"]
                      and fid["traj_rel_l2"] <= gate["traj_rel_l2"]
                      and fid["dist_abs_err"] <= gate["dist_abs_err"])
        passed_qual = qual["waypoint_l2"] <= K_QUALITY_L2
        row = {"K": k, "fid": fid, "quality_vs_K%d" % ref_steps: qual,
               "trt_ms": trt_ms, "torch_ms": torch_ms,
               "speedup": torch_ms / trt_ms if trt_ms else 0.0,
               "passed_fidelity": bool(passed_fid), "passed_quality": bool(passed_qual)}
        rows.append(row)
        print("  K=%-2d trt=%.2fms torch=%.2fms x%.2f | fid wp=%.4f trel=%.4f derr=%.3f %s "
              "| qual wp=%.4f %s"
              % (k, trt_ms, torch_ms, row["speedup"], fid["waypoint_l2"],
                 fid["traj_rel_l2"], fid["dist_abs_err"], "PASS" if passed_fid else "FAIL",
                 qual["waypoint_l2"], "ok" if passed_qual else "drift"))
        if chosen_k is None and passed_fid and passed_qual:
            chosen_k = k

    if chosen_k is None:
        passing = [r["K"] for r in rows if r["passed_fidelity"]]
        if not passing:
            raise FlowNavError("No K passed the TRT fidelity gate: %s" % rows)
        chosen_k = max(passing)
        print("[warn] no low K met the quality drift bound %.3f m; falling back to "
              "the largest fidelity-passing K=%d" % (K_QUALITY_L2, chosen_k))

    _write_selected(engine_dir, precision, chosen_k)
    report = {"target": detect().target_tag, "precision": precision,
              "chosen_num_steps": chosen_k, "ref_steps": ref_steps, "rows": rows}
    (engine_dir / "bench_report.json").write_text(json.dumps(report, indent=2))
    print("[done] selected precision=%s num_steps(K)=%d -> %s"
          % (precision, chosen_k, engine_dir / "selected.json"))


if __name__ == "__main__":
    main()
