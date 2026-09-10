"""Put one run beside the numbers papers reported, in a Markdown table that refuses to mix splits.

The table is where a result meets the literature, and the error it exists to
catch is the comparison that is not one: a val-split figure beside a test-split
paper number, an HM3D v1 run beside a v2 figure. Every reported row must name
the run's benchmark and split, or the table is refused.

Our row carries its uncertainty -- success rate and SPL with their intervals --
because a gap narrower than the interval is not a result. Scores are fractions
everywhere else in the harness; this is the only place that prints percent. A
value a paper did not report prints as an em dash, never as 0, and a reported
result's notes stay in the table, since they are exactly the caveats that make
a number not directly comparable.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.checks import is_name
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
    ReportedResult,
)

#: Printed where a value was not reported (an em dash).
MISSING_CELL = "—"

#: The table's columns, in order.
COLUMNS = ("Method", "SR (%)", "SPL (%)", "SoftSPL (%)", "DTG (m)", "Source")

_ALIGNMENT = ("---", "---:", "---:", "---:", "---:", "---")


def _cell(text: str) -> str:
    """One line of text, with ``|`` escaped so it cannot split the cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _percent(value: Optional[float]) -> str:
    return MISSING_CELL if value is None else "%.1f" % (100.0 * value)


def _metres(value: Optional[float]) -> str:
    return MISSING_CELL if value is None else "%.2f" % value


def _with_interval(value: float, interval: Tuple[float, float]) -> str:
    return "%.1f [%.1f, %.1f]" % (100.0 * value, 100.0 * interval[0],
                                  100.0 * interval[1])


def _reported_cells(result: ReportedResult) -> Tuple[str, ...]:
    if not isinstance(result.notes, str):
        raise HarnessError(
            "ReportedResult.notes of %r must be a string, got %r"
            % (result.method, result.notes))
    source = result.source
    if result.notes.strip():
        source = "%s; %s" % (source, result.notes)
    return (_cell(result.method), _percent(result.success_rate),
            _percent(result.spl), _percent(result.soft_spl),
            _metres(result.distance_to_goal_m), _cell(source))


def _our_cells(summary: BenchmarkSummary, label: str) -> Tuple[str, ...]:
    overall = summary.overall
    notes = ["%d episodes" % overall.n_episodes,
             "%g%% intervals" % (100.0 * summary.confidence)]
    if overall.n_unreachable_end:
        notes.append("DTG over the %d that ended with a goal reachable"
                     % (overall.n_episodes - overall.n_unreachable_end))
    if summary.agent_errors:
        notes.append("%d agent errors counted as failures"
                     % summary.agent_errors)
    return (_cell("%s (ours)" % label),
            _with_interval(overall.success_rate, overall.success_rate_ci),
            _with_interval(overall.spl, overall.spl_ci),
            _percent(overall.soft_spl),
            _metres(overall.distance_to_goal_m),
            _cell("this run: " + ", ".join(notes)))


def comparison_table(summary: BenchmarkSummary,
                     reported: Sequence[ReportedResult], *,
                     our_label: Optional[str] = None) -> str:
    """A Markdown table of papers' reported results and this run, ours last.

    Columns ``| Method | SR (%) | SPL (%) | SoftSPL (%) | DTG (m) | Source |``.
    Reported rows keep the given order. Our row comes last, labelled with the
    agent name (or ``our_label``) and marked "(ours)"; its SR and SPL carry
    their intervals (``46.2 [43.1, 49.3]``), and its source cell gives the
    episode count and confidence level. A reported result's notes follow its
    source.

    Args:
        summary: This run.
        reported: Paper numbers for the same benchmark and split.
        our_label: The name of our row; defaults to the agent name.

    Returns:
        The table, one row per line, without a trailing newline.

    Raises:
        TypeError: If ``summary`` is not a :class:`BenchmarkSummary` or an
            entry of ``reported`` is not a :class:`ReportedResult`.
        HarnessError: If a reported result is for another benchmark or split,
            its notes are not a string, or ``our_label`` is blank.
    """
    if not isinstance(summary, BenchmarkSummary):
        raise TypeError("comparison_table needs a BenchmarkSummary, got %s"
                        % type(summary).__name__)
    results = list(reported)
    for result in results:
        if not isinstance(result, ReportedResult):
            raise TypeError("reported entries must be ReportedResult, got %s"
                            % type(result).__name__)
        if (result.benchmark, result.split) != (summary.benchmark,
                                                summary.split):
            raise HarnessError(
                "reported result %r (%s) is for %s/%s but this run is %s/%s; "
                "a table that mixes benchmarks or splits compares numbers "
                "measured on different episodes -- drop the row or find the "
                "paper's %s/%s figure"
                % (result.method, result.source, result.benchmark,
                   result.split, summary.benchmark, summary.split,
                   summary.benchmark, summary.split))
    label = summary.agent if our_label is None else our_label
    if not is_name(label):
        raise HarnessError("our_label must be a non-empty string, got %r"
                           % (our_label,))
    lines = [_row(COLUMNS), _row(_ALIGNMENT)]
    lines.extend(_row(_reported_cells(result)) for result in results)
    lines.append(_row(_our_cells(summary, label)))
    return "\n".join(lines)


def reported_results_from_rows(
        rows: Sequence[Mapping[str, Any]]) -> Tuple[ReportedResult, ...]:
    """Paper numbers from plain rows, such as a JSON list typed in from the papers.

    Every key must be a :class:`ReportedResult` field. An unknown key is
    refused rather than ignored: a typo such as ``"sucess_rate"`` would
    otherwise surface far from the file that holds it. Values are not
    coerced; a percent typed where a fraction belongs is refused by
    :class:`ReportedResult` itself, and the message names the row.

    Args:
        rows: One mapping per reported result.

    Returns:
        The results, in row order.

    Raises:
        HarnessError: If ``rows`` is one mapping instead of a sequence of
            them, or a row is not a mapping.
        ResultsError: On an unknown or missing key, notes that are not a
            string, or a value :class:`ReportedResult` refuses.
    """
    if isinstance(rows, (Mapping, str, bytes)):
        raise HarnessError(
            "reported_results_from_rows needs a sequence of rows, got a "
            "single %s" % type(rows).__name__)
    allowed = [f.name for f in fields(ReportedResult)]
    required = [f.name for f in fields(ReportedResult)
                if f.default is MISSING and f.default_factory is MISSING]
    results: List[ReportedResult] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise HarnessError(
                "reported row %d must be a mapping of ReportedResult fields, "
                "got %r" % (index, row))
        unknown = sorted((key for key in row if key not in allowed), key=repr)
        missing = [name for name in required if name not in row]
        if unknown or missing:
            raise ResultsError(
                "reported row %d (%r) has unknown keys %r and lacks %r; the "
                "keys are %r" % (index, row.get("method"), unknown, missing,
                                 allowed))
        if "notes" in row and not isinstance(row["notes"], str):
            raise ResultsError("reported row %d: notes must be a string, got "
                               "%r" % (index, row["notes"]))
        try:
            results.append(ReportedResult(**dict(row)))
        except ResultsError as exc:
            raise ResultsError("reported row %d: %s" % (index, exc)) from exc
    return tuple(results)
