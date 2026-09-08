"""Turning result rows into tables a person can read.

Medians, not means. The runtime of an exponential search is heavy-tailed: one
instance in twenty can cost more than the other nineteen together, and a mean
reports that instance rather than the algorithm. Every summary here is a
median unless it is explicitly a rate or a count.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def median(values):
    # type: (Sequence[float]) -> float
    """The middle value, averaging the two middles on an even count."""
    ordered = sorted(v for v in values if v == v)   # drops NaN
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def group_by(rows, *keys):
    # type: (Iterable[Dict], str) -> Dict[Tuple, List[Dict]]
    """Bucket rows by the values of one or more columns."""
    out = {}                                    # type: Dict[Tuple, List[Dict]]
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        out.setdefault(key, []).append(row)
    return out


def table(headers, rows, aligns=None):
    # type: (Sequence[str], Sequence[Sequence[str]], Optional[Sequence[str]]) -> str
    """Render a fixed-width text table.

    Args:
        headers: Column titles.
        rows: Cell text, already formatted.
        aligns: ``"l"`` or ``"r"`` per column. Defaults to left for the first
            column and right for the rest, which is what a table of numbers
            with a name in front wants.

    Returns:
        The table as a string, no trailing newline.
    """
    if aligns is None:
        aligns = ["l"] + ["r"] * (len(headers) - 1)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def render(cells):
        # type: (Sequence[str]) -> str
        out = []
        for i, cell in enumerate(cells):
            text = str(cell)
            out.append(text.ljust(widths[i]) if aligns[i] == "l"
                       else text.rjust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [render(headers), render(["-" * w for w in widths])]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def summarise_fidelity(rows):
    # type: (Sequence[Dict]) -> str
    """Every claim as a pass count out of the instances tested."""
    total = len(rows)
    if not total:
        return "no instances"
    checks = [
        ("RPT* == exhaustive optimum (Thm. 2)", "rpt_is_optimal"),
        ("RPT*_noh == exhaustive optimum", "noh_is_optimal"),
        ("Lemma 1 == Eq. 1", "lemma1_holds"),
        ("h admissible (Lemma 5)", "heuristic_admissible"),
    ]
    out = [["%s" % label, "%d/%d" % (sum(1 for r in rows if r[key]), total),
            "PASS" if all(r[key] for r in rows) else "FAIL"]
           for label, key in checks]

    focal_total = sum(len(r["focal"]) for r in rows)
    focal_ok = sum(1 for r in rows for f in r["focal"] if f["within_bound"])
    out.append(["F-RPT* within (1+eps) (Thm. 3)",
                "%d/%d" % (focal_ok, focal_total),
                "PASS" if focal_ok == focal_total else "FAIL"])

    speedup = median([r["expansions_noh"] / r["expansions"]
                      for r in rows if r["expansions"]])
    out.append(["heuristic expansion saving",
                "%.2fx" % speedup, ""])
    return table(["claim", "result", ""], out)


def summarise_ablation(rows):
    # type: (Sequence[Dict]) -> str
    """Runtime and success rate per variant per size."""
    sizes = sorted({r["n"] for r in rows})
    variants = []                               # type: List[str]
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])

    grouped = group_by(rows, "variant", "n")
    body = []
    for variant in variants:
        cells = [variant]
        for n in sizes:
            bucket = grouped.get((variant, n), [])
            if not bucket:
                cells.append("-")
                continue
            ms = median([r["planning_s"] for r in bucket]) * 1000.0
            rate = sum(1 for r in bucket if r["solved"]) / float(len(bucket))
            cells.append("%.1f (%.0f%%)" % (ms, 100.0 * rate))
        body.append(cells)
    return table(["variant"] + ["n=%d" % n for n in sizes], body)


def summarise_baselines(rows):
    # type: (Sequence[Dict]) -> str
    """Cost ratio to the best planner, and planning time, per size."""
    sizes = sorted({r["n"] for r in rows})
    planners = []                               # type: List[str]
    for row in rows:
        if row["planner"] not in planners:
            planners.append(row["planner"])

    grouped = group_by(rows, "planner", "n")
    body = []
    for planner in planners:
        cells = [planner]
        for n in sizes:
            bucket = grouped.get((planner, n), [])
            if not bucket:
                cells.append("-")
                continue
            ratio = median([r["ratio_to_best"] for r in bucket])
            ms = median([r["planning_s"] for r in bucket]) * 1000.0
            cells.append("%.3fx / %.1fms" % (ratio, ms))
        body.append(cells)
    return table(["planner (cost ratio / plan time)"]
                 + ["n=%d" % n for n in sizes], body)


def summarise_reliability(rows, column="distance"):
    # type: (Sequence[Dict], str) -> str
    """A metric per planner per reliability regime, normalised to RPT*.

    Args:
        rows: Rows from :func:`~...experiments.run_reliability`.
        column: Which measured column to summarise.

    Returns:
        The table. Each cell is the median value and, in brackets, the ratio to
        RPT* in the same regime -- so ``1.00x`` means "same as RPT*" and
        ``2.10x`` means "flies twice as far". Planning times are shown in
        milliseconds, because at these sizes every planner is sub-second and a
        table of zeroes would hide the three orders of magnitude between them.
    """
    regimes = []                                # type: List[str]
    for row in rows:
        if row["regime"] not in regimes:
            regimes.append(row["regime"])
    planners = []                               # type: List[str]
    for row in rows:
        if row["planner"] not in planners:
            planners.append(row["planner"])

    scale, unit, digits = _scale_for(column)
    grouped = group_by(rows, "planner", "regime")
    reference = {regime: median([r[column]
                                 for r in grouped.get(("RPT*", regime), [])])
                 for regime in regimes}

    body = []
    for planner in planners:
        cells = [planner]
        for regime in regimes:
            bucket = grouped.get((planner, regime), [])
            if not bucket:
                cells.append("-")
                continue
            value = median([r[column] for r in bucket])
            base = reference.get(regime) or 0.0
            ratio = (value / base) if base > 0.0 else float("nan")
            cells.append("%.*f (%.2fx)" % (digits, value * scale, ratio))
        body.append(cells)
    return table(["planner (%s%s)" % (column, unit)] + list(regimes), body)


def _scale_for(column):
    # type: (str) -> Tuple[float, str, int]
    """Multiplier, unit suffix and decimal places for a measured column."""
    if column.endswith("_s") and column != "flight_time_s":
        return 1000.0, ", ms", 2
    if column == "places_searched":
        return 1.0, "", 2
    return 1.0, "", 0


def summarise_sharpness(rows):
    # type: (Sequence[Dict]) -> str
    """Cost ratio to the best planner as the belief goes from flat to spiked."""
    sharpnesses = sorted({r["sharpness"] for r in rows}, reverse=True)
    planners = []                               # type: List[str]
    for row in rows:
        if row["planner"] not in planners:
            planners.append(row["planner"])

    grouped = group_by(rows, "planner", "sharpness")
    peak = group_by(rows, "sharpness")
    headers = ["planner (cost ratio)"]
    for s in sharpnesses:
        headers.append("a=%g\nmax p=%.2f"
                       % (s, median([r["max_prob"] for r in peak[(s,)]])))

    body = []
    for planner in planners:
        cells = [planner]
        for s in sharpnesses:
            bucket = grouped.get((planner, s), [])
            cells.append("%.3fx" % median([r["ratio_to_best"] for r in bucket])
                         if bucket else "-")
        body.append(cells)

    # Two header rows, so the concentration and the resulting peak line up.
    top = [h.split("\n")[0] for h in headers]
    bottom = [h.split("\n")[1] if "\n" in h else "" for h in headers]
    rendered = table(top, [bottom] + body)
    lines = rendered.splitlines()
    # Move the separator below the second header line.
    return "\n".join([lines[0], lines[2], lines[1]] + lines[3:])


def summarise_objective(rows):
    # type: (Sequence[Dict]) -> str
    """How far the Eq. 1 route flies beyond the best one, per belief shape."""
    sharpnesses = sorted({r["sharpness"] for r in rows}, reverse=True)
    grouped = group_by(rows, "sharpness")

    body = []
    for sharpness in sharpnesses:
        bucket = grouped[(sharpness,)]
        excess = [r["excess_pct"] for r in bucket]
        agree = sum(1 for r in bucket if r["same_route"])
        body.append([
            "%g" % sharpness,
            "%.2f" % median([r["max_prob"] for r in bucket]),
            "%.2f%%" % median(excess),
            "%.2f%%" % (sum(excess) / len(excess)),
            "%.2f%%" % max(excess),
            "%d/%d" % (agree, len(bucket)),
        ])
    return table(["Dirichlet a", "median max p", "median excess",
                  "mean excess", "worst excess", "same route"], body)


def summarise_tsplib(rows):
    # type: (Sequence[Dict]) -> str
    """Violations, whether they cost anything, and the guarantee they void."""
    instances = []                              # type: List[str]
    for row in rows:
        if row["instance"] not in instances:
            instances.append(row["instance"])

    grouped = group_by(rows, "instance", "matrix")
    body = []
    for name in instances:
        published = grouped.get((name, "as-published"), [])
        if not published:
            continue
        penalty = median([r["penalty_pct"] for r in published])
        body.append([
            name,
            str(published[0]["n"]),
            str(published[0]["triangle_violations"]),
            "%.0f" % published[0]["worst_violation"],
            "%+.2f%%" % penalty,
            published[0]["guarantee"],
        ])
    return table(["instance", "n", "triangle viol.", "worst",
                  "route penalty", "guarantee"], body)



