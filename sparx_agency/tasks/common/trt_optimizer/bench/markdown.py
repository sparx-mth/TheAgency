"""Render an :class:`..bench.report.OptimizationReport` as Markdown.

Split from the report's own module because the *shape of the document* is a
separate concern from the data it presents, and the shape is the part that gets
argued about: which section leads, whether a failed gate is allowed to be a
footnote, how a number is rounded.

One rule is enforced structurally rather than left to the writer. The
``Deliberately not converted`` section is emitted from
:data:`..bench.report.NOT_CONVERTED_ACTIONS`, which is *derived* from
:data:`..spec.ACTIONS` rather than listed by hand -- so an action added later
cannot quietly stop appearing. That section is the most reusable part of the
whole report: it is what stops the next person redoing the analysis that already
concluded "not worth it".
"""
from __future__ import annotations

from sparx_agency.tasks.common.trt_optimizer.bench.report import (
    NOT_CONVERTED_ACTIONS,
)
from sparx_agency.tasks.common.trt_optimizer.bench.format import (
    MIB as _MIB, _fmt_cell, _fmt_params, _hz, _sig3, _stats_value, _table,
)


def _sorted_rows(report):
    """Rows sorted by ``before_ms`` descending, unmeasured ones last."""
    def key(row):
        return -row.before_ms if row.before_ms is not None else float("inf")
    return sorted(report.components, key=key)


def _assign_shares(rows):
    """Assign ``share_before`` in place from the rows actually present.

    Args:
        rows: the :class:`ComponentRow` objects about to be rendered.

    Returns:
        The summed ``before_ms`` of the measured rows, or None if none were.
    """
    total = 0.0
    measured = False
    for row in rows:
        if row.before_ms is not None:
            total += row.before_ms
            measured = True
    for row in rows:
        if measured and total > 0.0 and row.before_ms is not None:
            row.share_before = row.before_ms / total
        else:
            row.share_before = None
    return total if measured else None


def _headline_lines(report):
    """Section 1: the one line a reviewer reads, gate first."""
    before_hz, after_hz = _hz(report.before), _hz(report.after)
    if before_hz is None or after_hz is None:
        raise ValueError(
            "OptimizationReport for %r cannot be rendered: it needs both a "
            "before and an after measurement (.mean_ms or .hz). Report the "
            "baseline honestly rather than publishing a one-sided speedup."
            % report.model)
    speedup = report.speedup
    rate = "%s Hz -> %s Hz" % (_sig3(before_hz), _sig3(after_hz))
    lines = ["# TensorRT optimization report -- %s" % report.model, ""]
    if report.passed:
        lines.append("**PASS** -- %s (%sx faster), quality gate PASS over %d "
                     "check(s)." % (rate, _sig3(speedup), len(report.quality)))
    else:
        failed = [q for q in report.quality if not q.passed]
        if report.quality:
            cause = "%d of %d quality check(s) failed" % (len(failed),
                                                          len(report.quality))
        else:
            cause = "no quality checks were run, so accuracy is unverified"
        lines.append("**FAILED** -- %s (%sx) is **NOT ACCEPTED**: %s."
                     % (rate, _sig3(speedup), cause))
    lines.append("")
    return lines


def _identity_lines(report):
    """Section 2: what hardware and toolchain these numbers belong to."""
    rows = [
        ["gpu", report.gpu_name],
        ["target tag", report.target_tag],
        ["TensorRT", report.trt_version],
        ["precision", report.precision],
    ]
    lines = ["## Hardware and build", ""]
    lines.extend(_table(["field", "value"], rows))
    lines.append("")
    lines.append("Latencies and engines are valid for this target tag only.")
    lines.append("")
    return lines


def _component_lines(rows, total_before_ms):
    """Section 3: the latency inventory, worst offender first."""
    headers = ["name", "params", "cadence", "calls/decision", "before ms",
               "share", "after ms", "speedup", "action"]
    body = []
    for row in rows:
        share = "-" if row.share_before is None else "%.1f%%" % (
            100.0 * row.share_before)
        speedup = "-" if row.speedup is None else _sig3(row.speedup) + "x"
        body.append([
            row.name, _fmt_params(row.params), row.cadence,
            _sig3(row.calls_per_decision), _sig3(row.before_ms), share,
            _sig3(row.after_ms), speedup, row.action or "-",
        ])
    lines = ["## Components", ""]
    lines.extend(_table(headers, body, right=(1, 3, 4, 5, 6, 7)))
    lines.append("")
    if total_before_ms is not None:
        lines.append("Total measured decision budget before: %s ms."
                     % _sig3(total_before_ms))
        lines.append("")
    return lines


def _not_converted_lines(rows):
    """Section 4: mandatory -- every component left alone, and why."""
    lines = ["## Deliberately not converted", ""]
    kept = [r for r in rows if r.action in NOT_CONVERTED_ACTIONS]
    if not rows:
        # An empty inventory is not evidence that nothing was skipped -- it is
        # evidence that nobody passed the plan. Say which one it is.
        lines.append("No component inventory was supplied with this report, so "
                     "nothing can be said about what was left alone. Pass the "
                     "profiled Plan through to the bench stage.")
        lines.append("")
        return lines
    if not kept:
        lines.append("Nothing was skipped: every component in the inventory "
                     "above was converted.")
        lines.append("")
        return lines
    for row in kept:
        share = "" if row.share_before is None else (
            " (%.1f%% of the budget)" % (100.0 * row.share_before))
        lines.append("- **%s** -- `%s`%s: %s"
                     % (row.name, row.action, share,
                        row.why or "no reason recorded (fix this)"))
    lines.append("")
    return lines


def _quality_lines(report):
    """Section 5: the accuracy gate, or a loud note that there is none."""
    lines = ["## Quality", ""]
    if not report.quality:
        lines.append("No quality checks were recorded. This report is "
                     "**unverified** and the speedup above is not accepted.")
        lines.append("")
        return lines
    body = []
    for q in report.quality:
        body.append([q.metric, _fmt_cell(q.reference), _fmt_cell(q.measured),
                     _fmt_cell(q.threshold), "PASS" if q.passed else "FAIL"])
    lines.extend(_table(["metric", "reference", "measured", "threshold",
                         "pass"], body, right=(1, 2, 3)))
    lines.append("")
    for q in report.quality:
        if q.note:
            lines.append("- %s: %s" % (q.metric, q.note))
    if any(q.note for q in report.quality):
        lines.append("")
    return lines


def _mem_mib(memory, kind, consumed):
    """Read one memory figure in MiB, recording which key it came from."""
    for suffix, scale in (("_bytes", 1.0 / _MIB), ("_mib", 1.0),
                          ("_mb", 1.0), ("", 1.0)):
        key = kind + suffix
        if key in memory and isinstance(memory[key], (int, float)):
            consumed.add(key)
            return float(memory[key]) * scale
    return None


def _memory_lines(report):
    """Section 6: does it fit on the target, and by how much."""
    lines = ["## Memory", ""]
    memory = report.memory
    if not memory:
        lines.append("Not measured. Nothing here certifies that these engines "
                     "fit alongside the rest of the stack on the target.")
        lines.append("")
        return lines
    consumed = set()
    required = _mem_mib(memory, "required", consumed)
    available = _mem_mib(memory, "available", consumed)
    body = []
    if required is not None:
        body.append(["required", _sig3(required) + " MiB"])
    if available is not None:
        body.append(["available", _sig3(available) + " MiB"])
    if required is not None and available is not None:
        headroom = available - required
        body.append(["headroom", "%s MiB (%s)" % (
            _sig3(headroom),
            "fits" if headroom >= 0.0 else "OVER BUDGET")])
    for key in sorted(memory):
        if key not in consumed:
            body.append([key, _fmt_cell(memory[key])])
    lines.extend(_table(["field", "value"], body))
    lines.append("")
    return lines


def _warning_lines(report):
    """Section 7: everything that could make these numbers irreproducible."""
    lines = ["## Warnings", ""]
    if report.warnings:
        for warning in report.warnings:
            lines.append("- %s" % warning)
    else:
        lines.append("- None recorded. Confirm the SM clock was locked and the "
                     "device was thermally settled before trusting the deltas.")
    lines.append("")
    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for note in report.notes:
            lines.append("- %s" % note)
        lines.append("")
    return lines


def render_markdown(report):
    """Render the human-facing Markdown report.

    Sections are emitted in a fixed order: headline, hardware and build,
    components, deliberately not converted, quality, memory, warnings. The
    fourth is never omitted -- see the module docstring.

    Side effect: assigns :attr:`ComponentRow.share_before` on every row of
    ``report``, because the shares are a property of the table being drawn.

    Args:
        report: the :class:`OptimizationReport` to render.

    Returns:
        The report as one Markdown string.

    Raises:
        ValueError: if either the before or the after measurement is missing.
            A one-sided report has no honest headline.
    """
    rows = _sorted_rows(report)
    total_before_ms = _assign_shares(rows)
    lines = []
    lines.extend(_headline_lines(report))
    lines.extend(_identity_lines(report))
    lines.extend(_component_lines(rows, total_before_ms))
    lines.extend(_not_converted_lines(rows))
    lines.extend(_quality_lines(report))
    lines.extend(_memory_lines(report))
    lines.extend(_warning_lines(report))
    return "\n".join(lines).rstrip() + "\n"
