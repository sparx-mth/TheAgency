"""Which graph costs the FP16 gate? Try every per-graph precision combination.

A whole-pipeline FP16 build is 4.4x and fails the action-sequence gate; a
whole-pipeline FP32 build passes at 2.3x. "FP16 is too inaccurate" is not a
useful conclusion from that, because the three graphs are not equally at risk:
the DINOv2 trunk is 12 residual blocks whose output feeds everything downstream,
while the denoiser is re-entered ten times with its error partly re-corrected by
the Euler update each time.

TensorRT 11 is strongly typed, so precision is a property of each ONNX and each
engine independently -- which means the combinations need no rebuild at all.
Both precisions of all three engines already exist on disk; this loads each of
the eight assignments and gates it.

The point is not to find a loophole. It is that "FP16 failed" and "the *vision*
graph in FP16 failed" are different findings, and only the second one tells the
next person what to do.

Usage::

    python -m sparx_agency.tasks.planning.vlas.internvla_n1.trt.benchmark.mixed_precision \\
        --ckpt <InternVLA-N1-DualVLN> --engine-dir <.../engines/<target_tag>>
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer import adapter as adapter_mod
from sparx_agency.tasks.common.trt_optimizer.bench import latency
from sparx_agency.tasks.common.trt_optimizer.engine.runner import EngineRunner
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import adapter as ia

#: Graph order in every printed assignment, shortest-name-first for legibility.
KEYS = (ia.VISION_KEY, ia.CONDITION_KEY, ia.DENOISE_KEY)
#: Short labels for the table header.
SHORT = {ia.VISION_KEY: "vision", ia.CONDITION_KEY: "cond", ia.DENOISE_KEY: "denoise"}


def load_runtimes(engine_dir, assignment, cache):
    """Load (or reuse) one runner per graph for a precision assignment.

    Args:
        engine_dir: the per-target engine directory.
        assignment: mapping engine key -> ``"fp16"`` / ``"fp32"``.
        cache: dict reused across assignments, so each engine is deserialized
            once rather than eight times.

    Returns:
        Mapping engine key -> :class:`EngineRunner`.

    Raises:
        FileNotFoundError: naming the engine that is absent.
    """
    runtimes = {}
    for key, precision in assignment.items():
        path = Path(engine_dir) / ("%s.%s.engine" % (key, precision))
        if str(path) not in cache:
            if not path.is_file():
                raise FileNotFoundError(
                    "no %s engine for %s at %s; build that precision first"
                    % (precision, key, path))
            cache[str(path)] = EngineRunner(path)
        runtimes[key] = cache[str(path)]
    return runtimes


def evaluate(adapter, runtimes, reference_outputs, scenarios, sync):
    """Latency and averaged decision metrics for one assignment.

    Args:
        adapter: the System-1 adapter.
        runtimes: engine key -> runner.
        reference_outputs: precomputed torch references, one per scenario, so
            the reference is identical across assignments and cannot drift.
        scenarios: the scenarios to gate over.
        sync: the CUDA synchronise callable.

    Returns:
        ``(LatencyStats, {metric: mean})``.
    """
    stats = latency.measure(lambda: adapter.run_engines(runtimes, scenarios[0]),
                            warmup=5, iters=30, sync=sync)
    totals = {}
    for scenario, reference in zip(scenarios, reference_outputs):
        for name, value in adapter.decision_metrics(
                reference, adapter.run_engines(runtimes, scenario)).items():
            totals.setdefault(name, []).append(float(value))
    return stats, dict((k, sum(v) / len(v)) for k, v in totals.items())


def main(argv=None):
    """Run the sweep and print one row per assignment, fastest first."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--scenarios", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write the sweep as JSON here")
    args = ap.parse_args(argv)

    adapter = ia.InternVLAN1System1Adapter()
    model = adapter.patch(adapter.load(args.ckpt, device=args.device))
    scenarios = adapter.scenarios(args.scenarios)
    sync = latency.cuda_sync()

    # The torch reference is computed once and reused, so every assignment is
    # gated against the same numbers and a difference between rows is entirely
    # the engines'.
    references = [adapter.run_reference(model, s) for s in scenarios]
    torch_before = latency.measure(lambda: adapter.run_reference(model, scenarios[0]),
                                   warmup=5, iters=20, sync=sync)

    cache, rows = {}, []
    for combination in itertools.product(("fp32", "fp16"), repeat=len(KEYS)):
        assignment = dict(zip(KEYS, combination))
        stats, metrics = evaluate(
            adapter, load_runtimes(args.engine_dir, assignment, cache),
            references, scenarios, sync)
        passed, gate_rows = adapter_mod.evaluate_gates(adapter, metrics)
        rows.append({
            "assignment": assignment, "p50_ms": stats.p50_ms, "p99_ms": stats.p99_ms,
            "hz": stats.hz, "speedup": torch_before.mean_ms / stats.mean_ms,
            "passed": bool(passed), "metrics": metrics,
            "failed": [r[0] for r in gate_rows if not r[-1]],
        })

    rows.sort(key=lambda r: -r["hz"])
    header = "  ".join("%-7s" % SHORT[k] for k in KEYS)
    print("\ntorch reference (fp32, batch %d): p50 %.2f ms (%.2f Hz)\n"
          % (adapter.batch, torch_before.p50_ms, torch_before.hz))
    print("%s   %8s %8s %8s  %-6s %s"
          % (header, "p50 ms", "p99 ms", "speedup", "gate", "failed / traj_rel_l2"))
    for row in rows:
        print("%s   %8.2f %8.2f %8.2fx  %-6s %s"
              % ("  ".join("%-7s" % row["assignment"][k] for k in KEYS),
                 row["p50_ms"], row["p99_ms"], row["speedup"],
                 "PASS" if row["passed"] else "FAIL",
                 ", ".join(row["failed"]) or "-- rel_l2 %.3f" % row["metrics"]["traj_rel_l2"]))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"torch_p50_ms": torch_before.p50_ms, "rows": rows}, indent=2))
        print("\n[written]", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
