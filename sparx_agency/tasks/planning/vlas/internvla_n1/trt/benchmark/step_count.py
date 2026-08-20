"""The step-count lever: what fewer flow-matching steps cost in agreement.

``decide`` flags this before it flags any kernel, and it is right to: the
denoise loop is 94% of a System-1 call, the engine is **one** step rather than
ten unrolled, so the count is a runtime knob needing no rebuild, and halving it
halves the loop.

But it changes behaviour, so it is not a decision this tooling gets to make. This
measures the trade so a human can. Every row is gated against the **10-step FP32
torch reference** -- not against the same step count in torch, which would only
prove the engine reproduces a different algorithm faithfully.

Usage::

    python -m sparx_agency.tasks.planning.vlas.internvla_n1.trt.benchmark.step_count \\
        --ckpt <InternVLA-N1-DualVLN> --engine-dir <.../engines/<target_tag>>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer import adapter as adapter_mod
from sparx_agency.tasks.common.trt_optimizer.bench import latency
from sparx_agency.tasks.common.trt_optimizer.engine import selection
from sparx_agency.tasks.common.trt_optimizer.engine.runner import EngineRunner
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import adapter as ia

#: Step counts worth measuring. 10 is what ships; 2 is the floor the
#: flow-matching ladder is defined at.
STEPS = (10, 8, 6, 5, 4, 3, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--scenarios", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    reference_adapter = ia.InternVLAN1System1Adapter()
    model = reference_adapter.patch(reference_adapter.load(args.ckpt, device="cuda"))
    scenarios = reference_adapter.scenarios(args.scenarios)
    references = [reference_adapter.run_reference(model, s) for s in scenarios]
    del model                                   # free the torch copy before timing

    chosen = selection.read(args.engine_dir)
    runtimes = dict((k, EngineRunner(Path(args.engine_dir) / n))
                    for k, n in chosen["engines"].items())
    sync = latency.cuda_sync()

    rows = []
    for steps in STEPS:
        adapter = ia.InternVLAN1System1Adapter(steps=steps)
        stats = latency.measure(lambda: adapter.run_engines(runtimes, scenarios[0]),
                                warmup=5, iters=30, sync=sync)
        totals = {}
        for scenario, reference in zip(scenarios, references):
            for name, value in adapter.decision_metrics(
                    reference, adapter.run_engines(runtimes, scenario)).items():
                totals.setdefault(name, []).append(float(value))
        metrics = dict((k, sum(v) / len(v)) for k, v in totals.items())
        passed, gate_rows = adapter_mod.evaluate_gates(adapter, metrics)
        rows.append({"steps": steps, "p50_ms": stats.p50_ms, "p99_ms": stats.p99_ms,
                     "hz": stats.hz, "passed": bool(passed), "metrics": metrics,
                     "failed": [r[0] for r in gate_rows if not r[-1]]})

    print("\ngated against the 10-step FP32 torch reference, %d scenarios\n"
          % args.scenarios)
    print("%6s %9s %9s %8s  %-18s %-18s %-12s %s"
          % ("steps", "p50 ms", "p99 ms", "Hz", "first_action_match",
             "action_seq_match", "endpoint_m", "gate"))
    for row in rows:
        m = row["metrics"]
        print("%6d %9.2f %9.2f %8.2f  %-18.3f %-18.3f %-12.4f %s"
              % (row["steps"], row["p50_ms"], row["p99_ms"], row["hz"],
                 m["first_action_match"], m["action_seq_match"],
                 m["endpoint_err_m"], "PASS" if row["passed"] else
                 "FAIL (%s)" % ", ".join(row["failed"])))
    print("\nThis table is a trade, not a recommendation: fewer steps is a "
          "behaviour change,\nso the choice belongs to whoever owns the aircraft.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
