"""The deliverable: what the model did before, what it does now, and what it cost.

Every other module in this package produces evidence; this one produces the
document a human signs off on. Three decisions shape it.

**A speedup with no quality column is not a result.** :meth:`OptimizationReport.passed`
is the gate, and a report carrying zero :class:`QualityRow` entries is *not*
passing -- it is unverified, which is a different and more dangerous thing than
failing. When the gate is red the headline says ``FAILED`` and prints the
speedup as ``NOT ACCEPTED`` rather than burying the regression under a big
number. That inversion is the whole reason the renderer exists instead of an
f-string at the call site.

**The components that were deliberately *not* converted are the most valuable
part of the report.** A reader six months from now does not need to be told that
the ViT trunk became an FP16 engine -- the engine file says so. They need to know
why the language head is still in PyTorch, so they do not spend a week
rediscovering that it is autoregressive. ``## Deliberately not converted`` is
therefore emitted unconditionally, with an explicit "nothing was skipped" line
when the list is empty, so its absence can never be read as "nobody thought
about it".

**Shares are computed by the renderer, not trusted from the caller.** Each
:class:`ComponentRow` gets ``share_before`` assigned during rendering from the
row totals actually present, so the percentages always add up to the table above
them even when a caller hands over a partial inventory.

Numbers are printed to three significant figures and never in scientific
notation: this is read in a terminal and pasted into a review.

Pure standard library. :class:`LatencyStats` from :mod:`..bench.latency` is
duck-typed on ``.mean_ms`` / ``.hz`` and imported only under ``TYPE_CHECKING``,
so this module stays importable while the rest of the package is still landing.
Python-3.8-compatible syntax, like everything under ``trt_optimizer``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sparx_agency.tasks.common.trt_optimizer.bench.format import (
    MIB as _MIB, _fmt_cell, _fmt_params, _hz, _sig3, _stats_value, _table,
)
from sparx_agency.tasks.common.trt_optimizer.spec import ACTIONS, Cadence

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from sparx_agency.tasks.common.trt_optimizer.bench.latency import LatencyStats

#: Actions from :data:`..spec.ACTIONS` that build an engine. The same three
#: :func:`..decide.convertible` counts, kept in one place so the two modules
#: cannot drift apart about what "converted" means.
CONVERTED_ACTIONS = ("trt_fp16", "trt_fp32", "trt_int8")

#: Every other declared action means "measured, judged, left alone". Derived
#: from :data:`..spec.ACTIONS` rather than listed by hand: a hand-written list
#: silently drops any action added later (``reduce_calls`` was already missing
#: from it), and a dropped action is a component that vanishes from the
#: ``Deliberately not converted`` section -- the one section that exists so a
#: decision can never go unrecorded.
NOT_CONVERTED_ACTIONS = tuple(a for a in ACTIONS if a not in CONVERTED_ACTIONS)


def _stats_dict(stats):
    """Serialize a LatencyStats-like object to a JSON-safe dict, or None."""
    if stats is None:
        return None
    out = {"mean_ms": _stats_value(stats, "mean_ms"), "hz": _hz(stats)}
    for extra in ("std_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms",
                  "max_ms", "n", "iters", "warmup"):
        value = _stats_value(stats, extra)
        if value is not None:
            out[extra] = value
    return out


@dataclass
class ComponentRow:
    """One component's before/after line in the report.

    Args:
        name: the :class:`..spec.Component` name this row reports on.
        params: parameter count under that component.
        cadence: one of :class:`..spec.Cadence`.
        calls_per_decision: executions per control decision.
        before_ms: measured per-decision wall time before optimization.
        after_ms: the same after optimization, or None when the component was
            left alone (there is no "after" to state).
        action: the :data:`..spec.ACTIONS` verdict that produced this row.
        why: the verdict's reasoning, verbatim. Carried even for converted rows
            so the table and the prose never drift apart.
        share_before: fraction of the total ``before_ms`` budget. Left None by
            the caller and **assigned by** :func:`render_markdown`, which is the
            only place that knows the totals of the table it is drawing.
    """

    name: str
    params: int = 0
    cadence: str = Cadence.PER_FRAME
    calls_per_decision: float = 1.0
    before_ms: Optional[float] = None
    after_ms: Optional[float] = None
    action: str = ""
    why: str = ""
    share_before: Optional[float] = None

    @property
    def speedup(self):
        """Per-component speedup ``before_ms / after_ms``, or None.

        Returns:
            None when either side is missing or ``after_ms`` is zero -- an
            unconverted component has no speedup, and reporting 1.0 for it would
            claim a measurement that was never made.
        """
        if self.before_ms is None or self.after_ms is None:
            return None
        if self.after_ms <= 0.0:
            return None
        return self.before_ms / self.after_ms

    @classmethod
    def from_component(cls, component, verdict=None, after_ms=None):
        """Build a row from a :class:`..spec.Component` and its verdict.

        Args:
            component: the profiled component; ``decision_ms`` becomes
                ``before_ms``, so cadence is already folded into the budget.
            verdict: the matching :class:`..spec.Verdict`, or None.
            after_ms: measured per-decision wall time after conversion.

        Returns:
            A :class:`ComponentRow`.
        """
        return cls(
            name=component.name,
            params=component.params,
            cadence=component.cadence,
            calls_per_decision=component.calls_per_decision,
            before_ms=component.decision_ms,
            after_ms=after_ms,
            action="" if verdict is None else verdict.action,
            why="" if verdict is None else verdict.why,
        )


@dataclass
class QualityRow:
    """One accuracy check: what was measured against what, and did it hold.

    Args:
        metric: what was compared (``"trajectory_l2"``, ``"cosine_sim"``).
        reference: the PyTorch/FP32 value the engine is judged against.
        measured: the value the optimized pipeline produced.
        threshold: the pass/fail bound this metric was gated on.
        passed: whether the check held. Set by the checker, never inferred here
            -- only the checker knows whether the bound is an upper or a lower
            one.
        note: how the number was obtained, or what a near-miss means.
    """

    metric: str
    reference: Any = None
    measured: Any = None
    threshold: Any = None
    passed: bool = False
    note: str = ""


@dataclass
class OptimizationReport:
    """The full before/after record for one model on one target.

    Args:
        model: model name, matching :attr:`..spec.Plan.model`.
        target_tag: the device identity the numbers are valid for. Engines and
            timings are not portable across tags.
        gpu_name: human-readable GPU name.
        trt_version: the TensorRT version that built the engines.
        precision: the headline build precision (``fp16``, ``fp32``, ``int8``).
        before: LatencyStats-like object (``.mean_ms`` / ``.hz``) for baseline.
        after: the same for the optimized pipeline.
        components: per-component rows, in any order (rendered sorted).
        quality: the accuracy gate. An empty list means *unverified*.
        memory: budget dict; ``required_bytes`` / ``available_bytes`` (or the
            ``_mib`` / ``_mb`` spellings) are understood, anything else is
            printed as-is.
        warnings: reproducibility caveats -- clock locking, thermal drift,
            an unlocked SM clock, a shared GPU.
        notes: free-form remarks carried from the plan.
    """

    model: str
    target_tag: str = "unknown"
    gpu_name: str = "unknown"
    trt_version: str = "unknown"
    precision: str = "unknown"
    before: Optional["LatencyStats"] = None
    after: Optional["LatencyStats"] = None
    components: List[ComponentRow] = field(default_factory=list)
    quality: List[QualityRow] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def speedup(self):
        """End-to-end speedup, or None when either measurement is missing."""
        before_ms = _stats_value(self.before, "mean_ms")
        after_ms = _stats_value(self.after, "mean_ms")
        if before_ms is not None and after_ms is not None and after_ms > 0.0:
            return before_ms / after_ms
        before_hz, after_hz = _hz(self.before), _hz(self.after)
        if before_hz is None or after_hz is None or before_hz <= 0.0:
            return None
        return after_hz / before_hz

    @property
    def passed(self):
        """True only when there is a quality gate and every row of it held.

        Returns:
            False for an empty :attr:`quality` list. That is deliberate and is
            the one place this class refuses a vacuous truth: "nothing was
            checked" must not render as PASS.
        """
        if not self.quality:
            return False
        for row in self.quality:
            if not row.passed:
                return False
        return True

    def as_dict(self):
        """Serialize the whole report to a JSON-safe dict.

        Returns:
            A dict of plain types only, safe for ``json.dumps``. Derived values
            (``speedup``, ``passed``, per-row ``speedup``) are included so a
            consumer never has to reimplement the gate.
        """
        return {
            "model": self.model,
            "target_tag": self.target_tag,
            "gpu_name": self.gpu_name,
            "trt_version": self.trt_version,
            "precision": self.precision,
            "before": _stats_dict(self.before),
            "after": _stats_dict(self.after),
            "speedup": self.speedup,
            "passed": self.passed,
            "components": [
                {
                    "name": r.name, "params": int(r.params),
                    "cadence": r.cadence,
                    "calls_per_decision": float(r.calls_per_decision),
                    "before_ms": r.before_ms, "after_ms": r.after_ms,
                    "share_before": r.share_before, "speedup": r.speedup,
                    "action": r.action, "why": r.why,
                }
                for r in self.components
            ],
            "quality": [
                {
                    "metric": q.metric, "reference": q.reference,
                    "measured": q.measured, "threshold": q.threshold,
                    "passed": bool(q.passed), "note": q.note,
                }
                for q in self.quality
            ],
            "memory": dict(self.memory),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def render_markdown(report):
    """Render the report as Markdown.

    Re-exported from :mod:`..bench.markdown`, which owns the document shape, so
    callers keep importing one module.
    """
    from sparx_agency.tasks.common.trt_optimizer.bench.markdown import (
        render_markdown as _render)
    return _render(report)


def write_report(report, out_dir, stem="trt_report"):
    """Write ``<stem>.md`` and ``<stem>.json`` into ``out_dir``.

    Note:
        Engine directories under ``tasks/planning/vlas/*/trt/engines/`` are
        gitignored, so a report written there is build output and disappears on
        a clean checkout. Pass a tracked directory when the report has to
        survive one.

    Args:
        report: the :class:`OptimizationReport` to write.
        out_dir: destination directory; created with parents if missing.
        stem: filename stem shared by both files.

    Returns:
        ``(md_path, json_path)`` as :class:`pathlib.Path` objects.

    Raises:
        ValueError: propagated from :func:`render_markdown` on a one-sided
            report -- nothing is written in that case.
    """
    markdown = render_markdown(report)
    payload = json.dumps(report.as_dict(), indent=2, sort_keys=False)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / (stem + ".md")
    json_path = directory / (stem + ".json")
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(payload + "\n", encoding="utf-8")
    return md_path, json_path


def summarize(report):
    """One-line terminal summary of the whole report.

    Example:
        ``3.40x (12.1 -> 41.2 Hz) on nvidiageforcertx_sm120, fp16, quality PASS``

    Args:
        report: the :class:`OptimizationReport` to summarize.

    Returns:
        A single line, no trailing newline.

    Raises:
        ValueError: if either measurement is missing, for the same reason
            :func:`render_markdown` does.
    """
    before_hz, after_hz = _hz(report.before), _hz(report.after)
    if before_hz is None or after_hz is None:
        raise ValueError(
            "OptimizationReport for %r cannot be summarized: both a before and "
            "an after measurement are required." % report.model)
    return "%sx (%s -> %s Hz) on %s, %s, quality %s" % (
        _sig3(report.speedup), _sig3(before_hz), _sig3(after_hz),
        report.target_tag, report.precision,
        "PASS" if report.passed else "FAIL")
