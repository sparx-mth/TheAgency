"""The five stages, in the one order that makes the result trustworthy.

    inspect -> plan -> export -> build -> bench

The ordering is the method, not a convenience. Each stage refuses to run
without what the previous one produced, so the two failure modes that make a
TensorRT project waste weeks are structurally impossible:

* **Converting before measuring.** :func:`plan` profiles the unmodified model
  first and :func:`..decide.decide` refuses an unprofiled plan, so nothing is
  ever converted because it looked easy.
* **Claiming a speedup with no baseline.** The BEFORE number is captured in
  :func:`plan`, long before any engine exists, and carried through to the
  report. There is no code path that produces an "after" without a "before".

Stages split across machines the same way the artifacts do. ``export`` runs
anywhere torch runs and produces a *portable* ONNX; ``build`` and ``bench`` must
run on the target device in the same interpreter that will serve the engine,
because a serialized engine deserializes only under the exact TensorRT build and
compute capability that wrote it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer import (
    adapter as adapter_mod, decide, dissect, profile_torch,
    target as target_mod,
)
from sparx_agency.tasks.common.trt_optimizer.bench import latency, report as report_mod
from sparx_agency.tasks.common.trt_optimizer.engine import build as build_mod
from sparx_agency.tasks.common.trt_optimizer.engine import precision as prec_mod
from sparx_agency.tasks.common.trt_optimizer.engine import selection
from sparx_agency.tasks.common.trt_optimizer.export import onnx_export
from sparx_agency.tasks.common.trt_optimizer.spec import (
    Component, GraphSpec, Plan, ShapeProfile, Verdict,
)


def inspect(target=None):
    """Stage 0 -- what this machine can build, as a human-readable block."""
    tgt = target or target_mod.resolve()
    return target_mod.describe(tgt)


def plan(adapter, checkpoint, device="cuda", scenarios=8, target=None,
         max_depth=2, warmup=3, iters=10):
    """Stages 1-3 -- dissect, profile, decide. Returns ``(Plan, verdicts)``.

    Args:
        adapter: a checked :class:`..adapter.ModelAdapter`.
        checkpoint: path passed to ``adapter.load``.
        device: torch device for the reference run.
        scenarios: how many representative scenarios to profile over.
        target: the :class:`..target.Target`; resolved when omitted.
        max_depth: how deep to descend when building the component inventory.
        warmup: untimed iterations before measuring.
        iters: timed iterations.

    Returns:
        ``(plan, verdicts)`` with every component profiled and a verdict for
        each.

    Raises:
        ValueError: from :func:`..decide.decide` if profiling produced no
            measurements -- which means the run function never touched the
            model, and is a bug in the adapter rather than a soft condition.
    """
    tgt = target or target_mod.resolve()
    model = adapter.patch(adapter.load(checkpoint, device=device))
    components = dissect.inventory(model, max_depth=max_depth,
                                   cadences=adapter.cadences())
    dissect.check_accounting(components, model)

    cases = adapter.scenarios(scenarios)
    # Synthetic ".other" buckets are sums over the tree, not modules: they have
    # no forward to hook and profile_components rightly raises on a name it
    # cannot resolve. They stay in the inventory (the parameter total depends
    # on them) and simply go unmeasured, which fill_latencies reports as a note.
    names = [c.name for c in components if not dissect.is_synthetic(c)]
    overlaps = profile_torch.detect_overlap(model, names)
    run = lambda: adapter.run_reference(model, cases[0])
    measured = profile_torch.profile_components(model, run, names,
                                                warmup=warmup, iters=iters)
    components, notes = profile_torch.fill_latencies(components, measured)
    for ancestor, descendant in overlaps:
        notes.append("component %r contains %r; their times overlap and must "
                     "not be summed" % (ancestor, descendant))

    baseline = profile_torch.profile_end_to_end(run, warmup=warmup, iters=iters,
                                                sync=latency.cuda_sync())
    built = Plan(model=adapter.name, target_tag=tgt.target_tag,
                 components=components, graphs=list(adapter.graphs()),
                 baseline_hz=baseline.hz, notes=notes)
    built.notes.extend(latency.clock_warnings(tgt.hardware))
    verdicts = decide.decide(built, tgt)
    built.verdicts = verdicts
    built.notes.extend(decide.coverage_notes(built, verdicts))
    return built, verdicts


def export(adapter, checkpoint, out_dir, device="cpu", slim=True, keys=None):
    """Stage 4 -- export the chosen graphs to portable FP32 ONNX.

    Args:
        adapter: the model adapter.
        checkpoint: path passed to ``adapter.load``.
        out_dir: the ``engines/onnx`` directory.
        device: export device; CPU is correct and hardware-agnostic.
        slim: run onnxslim. Must be False on Jetson/aarch64.
        keys: restrict to these engine keys; all of them when omitted.

    Returns:
        The manifest dict written beside the graphs.
    """
    model = adapter.patch(adapter.load(checkpoint, device=device))
    graphs = [g for g in adapter.graphs() if keys is None or g.key in keys]
    wrappers = adapter.wrappers(model)
    return onnx_export.export_all(graphs, wrappers, out_dir, slim=slim,
                                  extra={"model": adapter.name})


def build(onnx_dir, out_dir, target=None, precision="fp16", options=None,
          param_counts=None, keys=None, specs=None, calibrators=None):
    """Stage 5 -- build one engine per exported graph, on this device.

    Args:
        onnx_dir: directory holding the exported graphs and their manifest.
        out_dir: per-target engine directory.
        target: the :class:`..target.Target`; resolved when omitted.
        precision: requested precision, checked against the toolchain first.
            A graph whose :class:`..spec.GraphSpec` (or manifest entry) marks it
            ``precision_sensitive`` is built FP32 regardless, so one numerically
            fragile graph does not force the whole pipeline down.
        options: :class:`..engine.builder_config.BuildOptions`.
        param_counts: engine key -> parameter count, enabling the post-build
            precision verification. Defaults to the counts the exporter recorded
            in the manifest, so verification happens unless a caller opts out by
            passing counts of its own.
        keys: restrict to these engine keys.
        specs: the :class:`..spec.GraphSpec` objects, keyed or listed. Required
            for any graph with dynamic axes, which needs its min/opt/max profile
            at build time.
        calibrators: engine key -> INT8 calibrator, for the weakly-typed route
            (TensorRT <= 10). Build them with
            :func:`..engine.calibrate.make_entropy_calibrator` over arrays from
            :func:`..engine.calibrate.collect_calibration_arrays`. Unused on a
            strongly-typed TensorRT, which needs a Q/DQ graph instead.

    Returns:
        Mapping of engine key -> written engine path.

    Raises:
        RuntimeError: if this toolchain cannot produce ``precision`` at all.
    """
    tgt = target or target_mod.resolve()
    target_mod.require_buildable(tgt, precision)
    manifest = onnx_export.read_manifest(onnx_dir)
    options = options or build_mod.BuildOptions(precision=precision)
    options.precision = precision
    if param_counts is None:
        param_counts = dict(
            (key, entry["params"])
            for key, entry in manifest.get("graphs", {}).items()
            if entry.get("params"))

    built = {}
    for key in sorted(manifest.get("graphs", {})):
        if keys is not None and key not in keys:
            continue
        spec = _spec_for(specs, key)
        key_options = options
        if precision != "fp32" and _is_precision_sensitive(spec, manifest, key):
            # The flag exists precisely so one graph can opt out of the
            # pipeline's precision without dragging the others down with it.
            # Honouring it here rather than in build_engine keeps the decision
            # where the graph list is: build_engine sees one graph and cannot
            # know it is part of a mixed-precision set.
            key_options = replace(options, precision="fp32")
        built[key] = build_mod.build_engine(
            Path(onnx_dir) / ("%s.onnx" % key), out_dir, tgt, options=key_options,
            param_count=(param_counts or {}).get(key),
            spec=spec,
            calibrator=(calibrators or {}).get(key))
    if not built:
        raise RuntimeError("no graphs to build in %s" % onnx_dir)
    return built



def _is_precision_sensitive(spec, manifest, key):
    """Whether ``key`` must be built FP32 whatever precision was requested.

    The GraphSpec wins when one was supplied; otherwise the flag the exporter
    recorded in the manifest is used, so a build driven from an ONNX directory
    alone still honours it. TensorRT 11 is strongly typed, so precision is a
    property of each engine independently and pinning one graph costs the others
    nothing.

    Args:
        spec: the :class:`..spec.GraphSpec` for this key, or None.
        manifest: the export manifest.
        key: engine key.

    Returns:
        bool
    """
    if spec is not None:
        return bool(spec.precision_sensitive)
    return bool(manifest.get("graphs", {}).get(key, {}).get("precision_sensitive"))

def _spec_for(specs, key):
    """Find a GraphSpec by engine key in a list or mapping, or None."""
    if not specs:
        return None
    if isinstance(specs, dict):
        return specs.get(key)
    for spec in specs:
        if getattr(spec, "key", None) == key:
            return spec
    return None


def bench(adapter, runtimes, reference_run, scenarios, before, target=None,
          precision="fp16", plan_obj=None, memory=None):
    """Stage 6 -- measure the engines, gate them, and build the report.

    Args:
        adapter: the model adapter, supplying the decision metrics and gates.
        runtimes: engine key -> loaded ``TRTEngineRunner``.
        reference_run: callable ``scenario -> reference output``.
        scenarios: the scenarios to compare over.
        before: the baseline :class:`..bench.latency.LatencyStats`.
        target: the :class:`..target.Target`.
        precision: the precision that was built, for the report.
        plan_obj: the :class:`..spec.Plan`, so the report can show the
            per-component table and the reasoning.
        memory: an optional :class:`..memory_budget.Budget`.

    Returns:
        An :class:`..bench.report.OptimizationReport`.
    """
    tgt = target or target_mod.resolve()
    sync = latency.cuda_sync()
    after = latency.measure(lambda: adapter.run_engines(runtimes, scenarios[0]),
                            sync=sync)

    totals = {}
    for scenario in scenarios:
        metrics = adapter.decision_metrics(reference_run(scenario),
                                           adapter.run_engines(runtimes, scenario))
        for name, value in metrics.items():
            totals.setdefault(name, []).append(float(value))
    averaged = dict((k, sum(v) / len(v)) for k, v in totals.items())
    passed, rows = adapter_mod.evaluate_gates(adapter, averaged)

    quality = [report_mod.QualityRow(metric=m, reference="torch fp32",
                                    measured=value, threshold="%s %s" % (op, thr),
                                    passed=ok, note="")
               for (m, value, op, thr, ok) in rows]
    for name, value in sorted(averaged.items()):
        if name not in adapter.gates():
            quality.append(report_mod.QualityRow(
                metric=name, reference="torch fp32", measured=value,
                threshold="(diagnostic)", passed=True, note="not gated"))

    return report_mod.OptimizationReport(
        model=adapter.name, target_tag=tgt.target_tag,
        gpu_name=tgt.hardware.gpu_name, trt_version=tgt.trt_version or "unknown",
        precision=precision, before=before, after=after,
        components=_component_rows(plan_obj),
        quality=quality,
        memory=_memory_dict(memory),
        warnings=list(latency.clock_warnings(tgt.hardware)),
        notes=list(plan_obj.notes) if plan_obj else [])


def _component_rows(plan_obj):
    """Turn a Plan's components and verdicts into report rows."""
    if plan_obj is None:
        return []
    verdict_by_name = {}
    for v in plan_obj.verdicts:
        verdict_by_name.setdefault(v.component, v)
    rows = []
    for c in plan_obj.components:
        v = verdict_by_name.get(c.name)
        rows.append(report_mod.ComponentRow(
            name=c.name, params=c.params, cadence=c.cadence,
            calls_per_decision=c.calls_per_decision, before_ms=c.decision_ms,
            after_ms=None, action=v.action if v else "",
            why=v.why if v else ""))
    return rows


def _memory_dict(budget):
    """Serialize a Budget for the report, tolerating None."""
    if budget is None:
        return {}
    if is_dataclass(budget):
        out = asdict(budget)
        out["required_bytes"] = budget.required_bytes
        out["headroom_bytes"] = budget.headroom_bytes
        out["fits"] = budget.fits
        return out
    return dict(budget)


def race(adapter, onnx_dir, out_dir, scenarios, before, reference_run,
         target=None, precisions=None, param_counts=None, specs=None,
         runner_factory=None, calibrators=None, plan_obj=None):
    """Build every candidate precision, gate each, and bless the fastest winner.

    This is what "maximum optimization" means operationally: not picking a
    precision up front and hoping, but building each one this toolchain can
    actually produce, measuring it, holding it to the adapter's gates, and
    letting the measurement choose. A quantized format must clear the SAME gates
    as FP16 -- and in practice a stricter operator should set stricter gates for
    it, because a format that is faster and slightly wrong is the worst trade
    available.

    Args:
        adapter: the :class:`..adapter.ModelAdapter`, supplying scenarios,
            decision metrics and gates.
        onnx_dir: directory of exported graphs.
        out_dir: per-target engine directory.
        scenarios: the scenarios to gate over.
        before: the baseline :class:`..bench.latency.LatencyStats`.
        reference_run: callable ``scenario -> reference output``, closing over
            the loaded torch model. Passed in rather than rebuilt here so the
            reference is the same object the baseline was measured on.
        target: the :class:`..target.Target`; resolved when omitted.
        precisions: candidates to try, in order. Defaults to
            :func:`..decide.precision_ladder`, which orders by *measured* payoff
            on this silicon rather than by width.
        param_counts: engine key -> parameter count, enabling the post-build
            precision verification.
        specs: GraphSpecs, needed for any graph with dynamic axes.
        plan_obj: the profiled :class:`..spec.Plan`. Pass it -- without it the
            report has no component table and no "deliberately not converted"
            section, which is the half of the document worth keeping.
        runner_factory: callable ``engine_path -> runner``; defaults to the
            toolkit's own :class:`..engine.runner.EngineRunner`.

    Returns:
        ``(winner, reports)`` -- the winning precision name (or None if none
        passed) and the per-precision :class:`..bench.report.OptimizationReport`
        list, in the order they were tried.

    Raises:
        Exception: whatever a build or gate raised, after withdrawing any
            selection this function had written. A half-written selection would
            point the runtime at an engine nothing blessed.
    """
    tgt = target or target_mod.resolve()
    candidates = list(precisions or decide.precision_ladder(tgt))
    reports, winner = [], None
    try:
        for precision in candidates:
            report = _race_one(adapter, onnx_dir, out_dir, scenarios, before,
                               reference_run, tgt, precision, param_counts,
                               specs, runner_factory, calibrators, plan_obj)
            if report is None:
                continue
            reports.append(report)
        passing = [r for r in reports if r.passed]
        if not passing:
            selection.clear(out_dir)
            return None, reports
        best = max(passing, key=lambda r: r.after.hz)
        winner = best.precision
        wanted = [s.key for s in specs] if specs else None
        selection.write(out_dir, winner, _engine_files(out_dir, winner, keys=wanted))
        return winner, reports
    except BaseException:
        # Selection is a side effect of evaluation, so an exception mid-race can
        # leave a file blessing an engine that never passed. Withdraw it.
        selection.clear(out_dir)
        raise


def _race_one(adapter, onnx_dir, out_dir, scenarios, before, reference_run, tgt,
              precision, param_counts, specs, runner_factory, calibrators,
              plan_obj):
    """Build and gate one precision. Returns its report, or None if unbuildable."""
    try:
        target_mod.require_buildable(tgt, precision)
    except RuntimeError as exc:
        log = "%s skipped: %s" % (precision, exc)
        print("[race]", log)
        return None
    try:
        built = build(onnx_dir, out_dir, target=tgt, precision=precision,
                      param_counts=param_counts, specs=specs,
                      calibrators=calibrators)
    except prec_mod.PrecisionUnavailable as exc:
        # Not a build failure: this precision simply cannot be expressed for
        # this graph here. Move on to the next candidate rather than losing the
        # whole race, and say so out loud.
        print("[race] %s skipped: %s" % (precision, exc))
        return None
    factory = runner_factory or _default_runner
    runtimes = dict((key, factory(path)) for key, path in built.items())
    return bench(adapter, runtimes, reference_run, scenarios, before,
                 target=tgt, precision=precision, plan_obj=plan_obj)


def _default_runner(engine_path):
    """The toolkit's own runner, which handles static and dynamic engines."""
    from sparx_agency.tasks.common.trt_optimizer.engine.runner import EngineRunner
    return EngineRunner(engine_path)


def _engine_files(out_dir, precision, keys=None):
    """Map engine key -> filename for the blessed precision in a directory.

    A ``precision_sensitive`` graph is built FP32 even when the race blessed
    FP16, so globbing for one suffix silently drops it -- and a selection
    missing an engine points the runtime at a pipeline it cannot complete. Any
    key not found at ``precision`` therefore falls back to its FP32 build, which
    is the only other precision :func:`build` can have produced for it.

    Args:
        out_dir: the per-target engine directory.
        precision: the blessed precision.
        keys: the engine keys the pipeline needs. When given, a key with no
            engine at either precision is an error rather than an omission.

    Returns:
        Mapping of engine key -> filename.

    Raises:
        RuntimeError: naming any requested key with no engine at all.
    """
    out_dir = Path(out_dir)

    def _collect(suffix):
        found = {}
        for path in sorted(out_dir.glob("*.%s.engine" % suffix)):
            found[path.name[: -len(".%s.engine" % suffix)]] = path.name
        return found

    files = _collect(precision)
    if precision != "fp32":
        for key, name in _collect("fp32").items():
            files.setdefault(key, name)
    if keys is not None:
        missing = [k for k in keys if k not in files]
        if missing:
            raise RuntimeError(
                "no engine built for %s at %s or fp32; the selection would send "
                "the runtime looking for a file that does not exist"
                % (", ".join(sorted(missing)), precision))
        files = dict((k, files[k]) for k in keys)
    return files


def save_plan(plan_obj, path):
    """Write a Plan to JSON so it can be reviewed, edited and replayed."""
    payload = {
        "model": plan_obj.model,
        "target_tag": plan_obj.target_tag,
        "baseline_hz": plan_obj.baseline_hz,
        "notes": list(plan_obj.notes),
        "components": [asdict(c) for c in plan_obj.components],
        "graphs": [asdict(g) for g in plan_obj.graphs],
        "verdicts": [asdict(v) for v in plan_obj.verdicts],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    return Path(path)


def load_plan(path):
    """Read back a Plan written by :func:`save_plan`.

    The staged path (``plan`` then ``export`` then ``bench``) needs this: without
    it, ``bench`` has no component inventory, so the report loses its
    per-component table *and* its "deliberately not converted" section -- which
    is the half of the document worth keeping. ``run`` keeps the Plan in memory
    and never needed a loader, which is why there was none.

    Args:
        path: a JSON file written by :func:`save_plan`.

    Returns:
        A :class:`..spec.Plan`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    payload = json.loads(Path(path).read_text())
    profiles = lambda raw: dict(
        (name, ShapeProfile(tuple(v["min"]), tuple(v["opt"]), tuple(v["max"])))
        for name, v in (raw or {}).items())
    graphs = []
    for raw in payload.get("graphs", []):
        raw = dict(raw)
        raw["inputs"] = dict((k, tuple(v)) for k, v in raw.get("inputs", {}).items())
        raw["profiles"] = profiles(raw.get("profiles"))
        graphs.append(GraphSpec(**raw))
    return Plan(
        model=payload.get("model", "unknown"),
        target_tag=payload.get("target_tag", "unknown"),
        components=[Component(**c) for c in payload.get("components", [])],
        graphs=graphs,
        verdicts=[Verdict(**v) for v in payload.get("verdicts", [])],
        baseline_hz=payload.get("baseline_hz"),
        notes=list(payload.get("notes", [])),
    )
