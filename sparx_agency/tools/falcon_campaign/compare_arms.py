"""Compare two interleaved A/B arms on pre-registered metrics.

Every verdict this loop reaches goes through here, so the things that have gone
wrong before are enforced rather than remembered:

* **Runs are deduplicated by ``truth.jsonl`` MD5.** A cycle that fails before the
  recorder produces new data copies the PREVIOUS flight's telemetry and still
  writes ``ended: completed`` (Finding J). ``ended`` is necessary evidence that a
  flight happened, not sufficient.
* **A bootstrap interval on the ratio, not a bare median.** Baseline coverage
  varies with CV 27 % and distance with CV 51 % (Finding G): at n=3 a median
  ratio can only detect ~60 % effects, and reading anything smaller off one is
  how five leads died here. An interval that spans 1.0 is reported as spanning
  1.0, not rounded into a verdict.
* **Metrics are addressed by dotted path**, so the pre-registration in
  LOOP_FIXES.md and the comparison run on literally the same string.

Usage::

    python3 -m sparx_agency.tools.falcon_campaign.compare_arms \\
        --a-rev v9.0-velwindow --b-rev v2.1d-legacyvel \\
        --metric tracking.pos_err_m.p90 --metric trace.route.frac_outside_safe_distance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

from sparx_agency.tools.falcon_campaign import config as C

RUN_NAME = re.compile(r"^\d{8}_\d{6}Z$")

#: Resamples for the ratio interval. 2000 is plenty for a 90 % interval and
#: keeps the whole comparison under a second.
BOOTSTRAP_N = 2000

#: Below this many flights in an arm the bootstrap interval is meaningless and
#: is reported as such. Resampling a single value always returns that value, so
#: n=1 yields a ZERO-WIDTH interval that looks like certainty and is the exact
#: opposite. Finding G puts the real requirement far higher -- ~8 flights per arm
#: for a 40 % effect, ~52 for 15 % -- so 3 is only the floor below which the
#: arithmetic itself lies.
MIN_N_FOR_INTERVAL = 3

#: Seeded so a verdict is reproducible: the same runs must always give the same
#: interval, or "it moved" and "I re-rolled" are indistinguishable.
BOOTSTRAP_SEED = 20260902


def _dig(obj, path):
    """Value at a dotted path, or None if any step is missing."""
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj if isinstance(obj, (int, float)) else None


def _md5(path):
    """MD5 of a file, or None if unreadable."""
    try:
        digest = hashlib.md5()
        with open(str(path), "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def load_runs(runs_dir=None, require_completed=True):
    """Every run folder with a usable summary, deduplicated by telemetry hash.

    Args:
        runs_dir: Directory of run folders. Defaults to the campaign's.
        require_completed: Drop runs whose ``ended`` is not "completed".

    Returns:
        A list of ``(name, summary)``, oldest first, with duplicate flights and
        flights that never happened removed.
    """
    runs_dir = runs_dir or C.RUNS_DIR
    out, seen = [], {}
    for path in sorted(p for p in runs_dir.glob("*")
                       if p.is_dir() and RUN_NAME.match(p.name)):
        try:
            with open(str(path / "summary.json")) as handle:
                summary = json.load(handle)
        except (OSError, ValueError):
            continue
        if require_completed and summary.get("ended") != "completed":
            continue
        # truth.jsonl is the primary identity, but the ROS 2 recorder and the
        # ROS 1 trace probe live in different containers and fail independently:
        # a run can carry a complete flight_trace and no truth.jsonl at all
        # (2026-09-02). Falling back keeps such a flight in the sample while
        # still catching Finding J's stale-copy duplicate, which is what the
        # hash is for.
        digest = _md5(path / "truth.jsonl") or _md5(path / "flight_trace.jsonl")
        if digest is None:
            continue
        if digest in seen:
            summary["duplicate_of"] = seen[digest]
            continue
        seen[digest] = path.name
        summary["run"] = path.name
        # metrics.json is authoritative: summary.json embeds a COPY taken at
        # cycle end, so any later re-analysis (a new metric, a corrected
        # formula) is invisible through the summary. Reading the stale copy is
        # how a metric that had just been added kept reading as absent.
        try:
            with open(str(path / "metrics.json")) as handle:
                summary["metrics"] = json.load(handle)
        except (OSError, ValueError):
            pass
        out.append((path.name, summary))
    return out


def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def ratio_interval(a_values, b_values, level=0.90):
    """Bootstrap interval on median(a)/median(b).

    Returns:
        ``(ratio, low, high)``, or ``(None, None, None)`` when either arm is
        empty or the control's median is zero.
    """
    a = [v for v in a_values if v is not None]
    b = [v for v in b_values if v is not None]
    if not a or not b:
        return (None, None, None)
    base_b = _median(b)
    if not base_b:
        return (None, None, None)
    point = _median(a) / base_b
    rng = random.Random(BOOTSTRAP_SEED)
    ratios = []
    for _ in range(BOOTSTRAP_N):
        ra = _median([rng.choice(a) for _ in a])
        rb = _median([rng.choice(b) for _ in b])
        if rb:
            ratios.append(ra / rb)
    if not ratios:
        return (point, None, None)
    ratios.sort()
    tail = (1.0 - level) / 2.0
    return (point, ratios[int(tail * len(ratios))],
            ratios[min(len(ratios) - 1, int((1.0 - tail) * len(ratios)))])


#: Stationary share above which a flight is treated as planner-locked and
#: excluded from a tracking comparison.
#:
#: Not a quality filter -- an EXCLUSION OF A DIFFERENT FAILURE. FALCON's terminal
#: A*-plan-fail lock parks the aircraft for minutes at a time (B6/Finding I:
#: 21.7 % of all flight time corpus-wide, worst case 254 s on one cell), and
#: nothing in the control stack causes or cures it. A locked flight barely moves,
#: so it scores a near-perfect tracking error while mapping almost nothing:
#: measured 2026-09-02, the flight that tracked best (route p50 0.043 m) mapped
#: 233 m3 and the one at 0.478 m mapped 3894 m3. Leaving them in lets the arm
#: that happened to lock more often "win".
#:
#: Applied identically to both arms, and the per-arm exclusion count is reported
#: -- if the arms lock at different rates, that is itself the finding and the
#: comparison must not be read as a tracking result.
LOCKED_STATIONARY_FRAC = 0.50


def _stationary(metrics):
    """Stationary share, from truth.jsonl if present and the trace otherwise."""
    value = _dig(metrics, "motion.frac_time_below_stop_speed")
    if value is None:
        value = _dig(metrics, "trace.motion.frac_time_below_stop_speed")
    return value


def compare(a_rev, b_rev, metrics, runs_dir=None, exclude_locked=True,
            exclude_degraded=False, since=None):
    """Compare the two arms on each metric.

    Args:
        a_rev: ``controller_rev`` of the candidate arm.
        b_rev: ``controller_rev`` of the control arm.
        metrics: Dotted paths into each run's ``metrics`` block.

    Returns:
        A report dict: the run names per arm and, per metric, both medians, the
        ratio and its interval.
    """
    runs = load_runs(runs_dir)
    arms = {a_rev: [], b_rev: []}
    excluded = {a_rev: [], b_rev: []}
    degraded = {a_rev: [], b_rev: []}
    for name, summary in runs:
        rev = summary.get("controller_rev")
        if rev not in arms:
            continue
        # Scope an experiment to its own window. A revision string identifies a
        # CONFIGURATION, and the same configuration is often the candidate of one
        # experiment and the control of the next -- so without this the control
        # arm silently absorbs flights from before the previous experiment ended,
        # reintroducing exactly the drift that interleaving exists to remove.
        if since is not None and name < since:
            continue
        metrics_block = summary.get("metrics") or {}
        if (metrics_block.get("health") or {}).get("dual_publisher_active"):
            degraded[rev].append(name)
            if exclude_degraded:
                continue
        stationary = _stationary(metrics_block)
        if (exclude_locked and stationary is not None
                and stationary > LOCKED_STATIONARY_FRAC):
            excluded[rev].append((name, round(stationary, 3)))
            continue
        arms[rev].append((name, metrics_block))
    report = {"a_rev": a_rev, "b_rev": b_rev, "since": since,
              "a_runs": [n for n, _ in arms[a_rev]],
              "b_runs": [n for n, _ in arms[b_rev]],
              "a_excluded_locked": excluded[a_rev],
              "b_excluded_locked": excluded[b_rev],
              "a_excluded_degraded": degraded[a_rev],
              "b_excluded_degraded": degraded[b_rev],
              "metrics": {}}
    for path in metrics:
        a_vals = [_dig(m, path) for _, m in arms[a_rev]]
        b_vals = [_dig(m, path) for _, m in arms[b_rev]]
        ratio, low, high = ratio_interval(a_vals, b_vals)
        a_n = len([v for v in a_vals if v is not None])
        b_n = len([v for v in b_vals if v is not None])
        informative = min(a_n, b_n) >= MIN_N_FOR_INTERVAL
        report["metrics"][path] = dict(
            a_n=a_n, b_n=b_n, interval_informative=informative,
            a_median=_median(a_vals), b_median=_median(b_vals),
            a_values=a_vals, b_values=b_vals,
            ratio=ratio, ratio_low=low, ratio_high=high,
            #: True when the interval contains 1.0, i.e. the data cannot
            #: distinguish the arms and no verdict may be read off the point.
            spans_one=(low is not None and low <= 1.0 <= high))
    return report


def render(report):
    """The report as a table for a human."""
    lines = ["A = %s  (n=%d: %s)" % (report["a_rev"], len(report["a_runs"]),
                                     ", ".join(report["a_runs"]) or "none"),
             "B = %s  (n=%d: %s)" % (report["b_rev"], len(report["b_runs"]),
                                     ", ".join(report["b_runs"]) or "none"),
             "excluded as planner-locked (stationary > %.2f):  A %d %s   B %d %s"
             % (LOCKED_STATIONARY_FRAC,
                len(report["a_excluded_locked"]), report["a_excluded_locked"] or "",
                len(report["b_excluded_locked"]), report["b_excluded_locked"] or ""),
             "flown against Sphera's duplicate pawn (excluded only with "
             "--drop-degraded):  A %d   B %d"
             % (len(report["a_excluded_degraded"]), len(report["b_excluded_degraded"])),
             "",
             "%-46s %9s %9s %8s %-18s" % ("metric", "A med", "B med", "A/B", "90% interval")]
    for path, row in report["metrics"].items():
        def fmt(v):
            return "-" if v is None else ("%.4g" % v)
        if row["ratio_low"] is None:
            interval = "-"
        elif not row.get("interval_informative", True):
            interval = "n=%d/%d TOO FEW" % (row["a_n"], row["b_n"])
        else:
            interval = "[%.2f, %.2f]%s" % (row["ratio_low"], row["ratio_high"],
                                           "  SPANS 1" if row["spans_one"] else "")
        lines.append("%-46s %9s %9s %8s %-18s"
                     % (path[:46], fmt(row["a_median"]), fmt(row["b_median"]),
                        fmt(row["ratio"]), interval))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-rev", required=True, help="candidate controller_rev")
    parser.add_argument("--b-rev", required=True, help="control controller_rev")
    parser.add_argument("--metric", action="append", required=True,
                        help="dotted path into metrics.json; repeatable")
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    parser.add_argument("--keep-locked", action="store_true",
                        help="do not exclude planner-locked flights")
    parser.add_argument("--since", default=None,
                        help="only runs whose folder name sorts at or after this "
                             "(e.g. 20260903_08) -- scopes an experiment to its "
                             "own window; run names are chronological")
    parser.add_argument("--drop-degraded", action="store_true",
                        help="exclude flights flown against Sphera's duplicate "
                             "pawn (reported either way; NOT dropped by default "
                             "since the stack now handles the defect)")
    args = parser.parse_args()
    report = compare(args.a_rev, args.b_rev, args.metric,
                     exclude_locked=not args.keep_locked,
                     exclude_degraded=args.drop_degraded, since=args.since)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))


if __name__ == "__main__":
    main()
