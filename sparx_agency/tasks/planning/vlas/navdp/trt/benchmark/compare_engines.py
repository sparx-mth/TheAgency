"""A/B-compare denoiser engine variants: latency + accuracy, side by side.

Runs the full point-goal pipeline once per available denoiser engine
(``navdp_denoise*.<prec>.engine`` -- e.g. the baseline and the ``tgt_is_causal``
variant), sharing the same encoder + critic engines, and reports:

  * denoise-step latency (the 10x-per-decision inner loop, the measured bottleneck),
  * end-to-end Hz,
  * the accuracy gate vs the FP32 torch reference (argmax-flip / stop / zeroing).

It then writes ``selected.json`` pointing at the fastest variant that still passes
the gate, so the server runs the winner. Use this to decide whether a variant is
actually worth it -- measured, not assumed.

Run (target device, TRT venv; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.vlas.navdp.trt.benchmark.compare_engines \
        --engine-dir .../engines/<target_tag> --ckpt ... --navdp-repo ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sparx_agency.core.planning.vlas.navdp.trt.policy import NavDPTRTPolicy
from sparx_agency.tasks.planning.vlas.navdp.trt.benchmark.bench import (
    GATES, TorchReference, accuracy, fps, gen_scenarios,
)
from sparx_agency.tasks.planning.vlas.navdp.trt.export import io_spec


def _write_selected(engine_dir, denoise_file, precision):
    """Point selected.json at a specific denoise engine (encoder/critic baseline)."""
    sel = {"precision": precision, "engines": {
        "encoder": "%s.%s.engine" % (io_spec.ENCODER, precision),
        "denoise": denoise_file,
        "critic": "%s.%s.engine" % (io_spec.CRITIC, precision)}}
    (engine_dir / "selected.json").write_text(json.dumps(sel, indent=2))


def _denoise_step_ms(policy, n=50, warmup=5):
    """Median latency of one denoise engine call (conditioning resident)."""
    nn = policy.sample_num
    rgbd = np.random.rand(nn, io_spec.MEM_TOK, io_spec.TOK).astype(np.float32)
    goal = np.random.rand(nn, 1, io_spec.TOK).astype(np.float32)
    la = np.random.rand(nn, io_spec.PREDICT, 3).astype(np.float32)
    tt = np.random.rand(nn, 1, io_spec.TOK).astype(np.float32)
    policy._den.upload({"rgbd_embed": rgbd, "goal_embed": goal})
    run = lambda: policy._den.infer({"last_actions": la, "time_token": tt})
    for _ in range(warmup):
        run()
    t0 = time.perf_counter()
    for _ in range(n):
        run()
    return (time.perf_counter() - t0) / n * 1000.0


def evaluate_variant(engine_dir, npz, denoise_file, precision, ref, scenarios, stop):
    """Measure one denoise variant end-to-end + the gate. Returns a row dict."""
    _write_selected(engine_dir, denoise_file, precision)
    policy = NavDPTRTPolicy(engine_dir, npz)
    acc = accuracy(policy, ref, scenarios, stop)
    perf = fps(policy, scenarios[0])
    step_ms = _denoise_step_ms(policy)
    gate = GATES.get(precision, GATES["fp16"])
    passed = (acc["argmax_flip"] <= gate["argmax_flip"]
              and acc["stop_match"] >= gate["stop_match"]
              and acc["zero_match"] >= gate["zero_match"])
    return {"denoise": denoise_file, "step_ms": step_ms, "passed": bool(passed),
            **acc, **perf}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--head-params", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--num-scenarios", type=int, default=32)
    ap.add_argument("--stop-threshold", type=float, default=-999.0)
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir)
    npz = Path(args.head_params) if args.head_params else engine_dir / "navdp_head_params.npz"
    variants = sorted(p.name for p in engine_dir.glob("navdp_denoise*.%s.engine" % args.precision))
    if not variants:
        raise SystemExit("no denoise engines (%s) in %s" % (args.precision, engine_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_torch = __import__(
        "sparx_agency.tasks.planning.vlas.navdp.trt.export.build_policy",
        fromlist=["build_navdp_policy"]).build_navdp_policy(
            args.ckpt, navdp_repo=args.navdp_repo, device=device)
    ref = TorchReference(policy_torch, npz, device)
    scenarios = gen_scenarios(args.num_scenarios)

    rows = []
    for v in variants:
        print("[compare] evaluating", v)
        rows.append(evaluate_variant(engine_dir, npz, v, args.precision, ref,
                                     scenarios, args.stop_threshold))

    print("\n%-30s %10s %8s %8s %6s" % ("denoise engine", "step_ms", "hz", "flip", "pass"))
    for r in rows:
        print("%-30s %10.2f %8.1f %8.3f %6s"
              % (r["denoise"], r["step_ms"], r["hz"], r["argmax_flip"], r["passed"]))

    passing = [r for r in rows if r["passed"]]
    if not passing:
        raise SystemExit("no denoise variant passed the gate")
    winner = min(passing, key=lambda r: r["step_ms"])     # fastest inner-loop step
    _write_selected(engine_dir, winner["denoise"], args.precision)
    (engine_dir / "compare_report.json").write_text(json.dumps(rows, indent=2))
    print("\n[done] winner: %s  ->  selected.json" % winner["denoise"])


if __name__ == "__main__":
    main()
