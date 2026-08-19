"""Command line for the TensorRT optimizer: ``python -m ...trt_optimizer <cmd>``.

Argument parsing and dispatch only -- every command is a thin call into
:mod:`..trt_optimizer.pipeline`. Commands that need a network speak to it
through a registered :class:`..adapter.ModelAdapter`, named with ``--adapter``
and loaded from a module given with ``--adapter-module`` (that module registers
itself on import, which is why it is named rather than imported here).

``inspect``, ``dla`` and ``budget`` need no adapter and no checkpoint, so the
questions worth asking before committing to any of this -- can this machine
build FP16 at all, is DLA reachable, will the thing even fit in 8 GB -- can be
answered in seconds.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer import (
    adapter as adapter_mod, decide, memory_budget, pipeline, target as target_mod,
)


def _load_adapter(args):
    """Import the registering module, then instantiate and check the adapter."""
    if args.adapter_module:
        importlib.import_module(args.adapter_module)
    if not args.adapter:
        raise SystemExit("--adapter is required for this command (available: %s)"
                         % (", ".join(adapter_mod.available()) or "none; pass "
                            "--adapter-module to register one)"))
    return adapter_mod.check(adapter_mod.create(args.adapter))


def cmd_inspect(args):
    """Print what this machine can build."""
    print(pipeline.inspect())
    return 0


def cmd_plan(args):
    """Dissect, profile and decide; write the plan and print the reasoning."""
    adapter = _load_adapter(args)
    plan_obj, verdicts = pipeline.plan(
        adapter, args.ckpt, device=args.device, scenarios=args.scenarios,
        max_depth=args.max_depth)
    if args.out:
        print("[plan]", pipeline.save_plan(plan_obj, args.out))
    print("baseline: %.1f Hz" % (plan_obj.baseline_hz or 0.0))
    total = plan_obj.decision_ms() or 0.0
    for component in sorted(plan_obj.components,
                            key=lambda c: -(c.decision_ms or 0.0)):
        share = 100.0 * (component.decision_ms or 0.0) / total if total else 0.0
        print("  %-28s %7.2f ms  %5.1f%%  %s"
              % (component.name, component.decision_ms or 0.0, share,
                 component.cadence))
    print()
    for verdict in verdicts:
        print("  %-28s %-16s %s" % (verdict.component, verdict.action,
                                    verdict.why))
    projected, bound = decide.ceiling(plan_obj, verdicts)
    print("\nprojected %.2fx, Amdahl ceiling %.2fx" % (projected, bound))
    return 0


def cmd_export(args):
    """Export the adapter's graphs to portable FP32 ONNX."""
    adapter = _load_adapter(args)
    manifest = pipeline.export(adapter, args.ckpt, args.out_dir,
                               slim=not args.no_slim)
    for key in sorted(manifest.get("graphs", {})):
        print("[ok] exported", key)
    return 0


def cmd_build(args):
    """Build one engine per exported graph on this device."""
    built = pipeline.build(args.onnx_dir, args.out_dir, precision=args.precision)
    for key, path in sorted(built.items()):
        print("[ok] built %s -> %s" % (key, path.name))
    return 0


def cmd_dla(args):
    """Report whether any of a graph should run on the Jetson's DLA."""
    from sparx_agency.tasks.common.trt_optimizer.engine import dla

    tgt = target_mod.resolve()
    verdict = dla.evaluate(args.onnx, tgt.hardware)
    print("use_dla: %s" % verdict.use_dla)
    print("why:     %s" % verdict.why)
    print("eligible fraction: %.2f, contiguous prefix: %d"
          % (verdict.eligible_fraction, verdict.contiguous_prefix))
    note = dla.power_note(tgt.hardware)
    if note:
        print(note)
    return 0


def cmd_budget(args):
    """Report whether a plan fits this device's memory, and what to do if not."""
    from sparx_agency.tasks.common.trt_optimizer.spec import Component, Plan

    payload = json.loads(Path(args.plan).read_text())
    components = [Component(**{k: v for k, v in c.items()
                               if k in Component.__dataclass_fields__})
                  for c in payload["components"]]
    plan_obj = Plan(model=payload.get("model", "?"), components=components)
    tgt = target_mod.resolve()
    budget = memory_budget.estimate(plan_obj, args.precision, tgt.hardware,
                                    resident=args.resident)
    print("required %.2f GiB, free %.2f GiB, headroom %.2f GiB -> %s"
          % (budget.required_bytes / (1 << 30), budget.free_bytes / (1 << 30),
             budget.headroom_bytes / (1 << 30),
             "FITS" if budget.fits else "DOES NOT FIT"))
    for line in memory_budget.recommendations(budget, tgt.hardware):
        print("  -", line)
    return 0


def cmd_acquire(args):
    """Clone or locate the model and inventory it, without importing anything."""
    from sparx_agency.tasks.common.trt_optimizer import acquire as acq

    source = acq.acquire(args.source, depth=args.depth, dry_run=args.dry_run)
    code_dir = (source.workspace / "code" if source.kind == "git"
                else Path(source.url_or_path))
    if not code_dir.is_dir():
        # A dry run resolves and slugs the source without fetching it, so there
        # is nothing on disk to inventory. Report the plan rather than failing.
        print("%s -> %s (kind %s)" % (args.source, source.workspace, source.kind))
        print("would clone into %s" % code_dir)
        print("nothing to inventory yet: re-run without --dry-run to fetch it.")
        return 0
    print(acq.summarize(source, acq.find_entrypoints(code_dir)))
    return 0


def cmd_bench(args):
    """Measure built engines against the torch reference and gate them."""
    adapter = _load_adapter(args)
    model = adapter.patch(adapter.load(args.ckpt, device=args.device))
    scenarios = adapter.scenarios(args.scenarios)
    reference = lambda scenario: adapter.run_reference(model, scenario)

    from sparx_agency.tasks.common.trt_optimizer.bench import latency
    before = latency.measure(lambda: reference(scenarios[0]),
                             sync=latency.cuda_sync())
    plan_obj = pipeline.load_plan(args.plan) if args.plan else None
    winner, reports = pipeline.race(
        adapter, args.onnx_dir, args.engine_dir, scenarios, before, reference,
        precisions=args.precisions.split(",") if args.precisions else None,
        specs=list(adapter.graphs()), plan_obj=plan_obj)
    return _finish(reports, winner, args.report_dir or args.engine_dir)


def cmd_run(args):
    """Every stage, in order: plan, export, build, race, report."""
    adapter = _load_adapter(args)
    tgt = target_mod.resolve()
    print(target_mod.describe(tgt), "\n")

    plan_obj, verdicts = pipeline.plan(adapter, args.ckpt, device=args.device,
                                       scenarios=args.scenarios, target=tgt,
                                       max_depth=args.max_depth)
    if args.plan_out:
        pipeline.save_plan(plan_obj, args.plan_out)
    _print_plan(plan_obj, verdicts)

    onnx_dir = Path(args.out_dir) / "onnx"
    engine_dir = Path(args.out_dir) / tgt.target_tag
    pipeline.export(adapter, args.ckpt, onnx_dir, slim=not args.no_slim)

    model = adapter.patch(adapter.load(args.ckpt, device=args.device))
    scenarios = adapter.scenarios(args.scenarios)
    reference = lambda scenario: adapter.run_reference(model, scenario)
    from sparx_agency.tasks.common.trt_optimizer.bench import latency
    before = latency.measure(lambda: reference(scenarios[0]),
                             sync=latency.cuda_sync())

    winner, reports = pipeline.race(
        adapter, onnx_dir, engine_dir, scenarios, before, reference, target=tgt,
        precisions=args.precisions.split(",") if args.precisions else None,
        specs=list(adapter.graphs()), plan_obj=plan_obj)
    return _finish(reports, winner, args.report_dir or engine_dir)


def _print_plan(plan_obj, verdicts):
    """Print the measured breakdown and the verdict for each component."""
    total = plan_obj.decision_ms() or 0.0
    print("baseline: %.1f Hz" % (plan_obj.baseline_hz or 0.0))
    for component in sorted(plan_obj.components,
                            key=lambda c: -(c.decision_ms or 0.0)):
        share = 100.0 * (component.decision_ms or 0.0) / total if total else 0.0
        print("  %-28s %7.2f ms  %5.1f%%  %s"
              % (component.name, component.decision_ms or 0.0, share,
                 component.cadence))
    print()
    for verdict in verdicts:
        print("  %-28s %-16s %s" % (verdict.component, verdict.action,
                                    verdict.why))
    projected, bound = decide.ceiling(plan_obj, verdicts)
    print("\nprojected %.2fx, Amdahl ceiling %.2fx\n" % (projected, bound))


def _finish(reports, winner, report_dir):
    """Write every report, print the summary, and return the exit code."""
    from sparx_agency.tasks.common.trt_optimizer.bench import report as report_mod

    for report in reports:
        md, _ = report_mod.write_report(report, report_dir,
                                        stem="trt_report_%s" % report.precision)
        print("[report] %s -- %s" % (report_mod.summarize(report), md))
    if winner is None:
        print("\nNO precision passed its quality gate. Nothing was blessed; the "
              "selection file has been withdrawn so no engine is served.")
        return 1
    print("\nwinner: %s" % winner)
    return 0


def build_parser():
    """The argument parser for every command."""
    parser = argparse.ArgumentParser(
        prog="python -m sparx_agency.tasks.common.trt_optimizer",
        description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default=None, help="registered adapter name")
    parser.add_argument("--adapter-module", default=None,
                        help="module to import so it registers its adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="what this machine can build").set_defaults(
        func=cmd_inspect)

    p = sub.add_parser("plan", help="dissect, profile and decide")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="write the plan JSON here")
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenarios", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=2,
                   help="how deep the component inventory descends. 2 suits a\n"
                        "model whose depth-2 frontier is real modules; drop to 1\n"
                        "when it is nn.ModuleList containers, which have no\n"
                        "forward to time and would leave the plan unprofiled")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("export", help="export graphs to portable FP32 ONNX")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-slim", action="store_true",
                   help="skip onnxslim; REQUIRED on Jetson/aarch64, where it aborts")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("build", help="build engines on this device")
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--precision", default="fp16")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("dla", help="should this graph use the Jetson DLA?")
    p.add_argument("--onnx", required=True)
    p.set_defaults(func=cmd_dla)

    p = sub.add_parser("budget", help="will this plan fit in device memory?")
    p.add_argument("--plan", required=True)
    p.add_argument("--precision", default="fp16")
    p.add_argument("--resident", default="concurrent",
                   choices=["concurrent", "sequential"])
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("acquire", help="clone/locate a model and inventory it")
    p.add_argument("source", help="a GitHub URL or a local path")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_acquire)

    p = sub.add_parser("bench", help="measure built engines and gate them")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--onnx-dir", required=True)
    p.add_argument("--engine-dir", required=True)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenarios", type=int, default=16)
    p.add_argument("--precisions", default=None,
                   help="comma-separated candidates; default is the measured "
                        "ladder for this silicon")
    p.add_argument("--plan", default=None,
                   help="plan.json from the plan stage. Without it the report "
                        "has no component table and no 'deliberately not "
                        "converted' section")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("run", help="every stage: plan, export, build, race, report")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True,
                   help="engines/ root; onnx/ and <target_tag>/ are created under it")
    p.add_argument("--plan-out", default=None)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenarios", type=int, default=16)
    p.add_argument("--precisions", default=None)
    p.add_argument("--no-slim", action="store_true")
    p.add_argument("--max-depth", type=int, default=2,
                   help="how deep the component inventory descends. 2 suits a\n"
                        "model whose depth-2 frontier is real modules; drop to 1\n"
                        "when it is nn.ModuleList containers, which have no\n"
                        "forward to time and would leave the plan unprofiled")
    p.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
