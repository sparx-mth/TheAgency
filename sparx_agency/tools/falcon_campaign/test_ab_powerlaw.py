"""The pre-registered A/B verdict for the powerlaw+lateral controller.

Written 2026-08-31, BEFORE any A/B run existed, in the same spirit as
test_p38.py: the sample, the metrics, the thresholds and the decision rule are
all fixed here in advance, so running it is a yes/no with nothing left to
tune after the numbers are visible.

Sample: every run folder whose summary.json carries a ``controller_variant``
key (only A/B cycles write it) and completed a flight. Refuses to judge below
3 baseline + 3 candidate runs. The scheduled set is 3 legacy + 4 candidate,
interleaved (see AB_RUNBOOK_powerlaw_lateral.md).

Decision rule (medians across runs, per arm):
  PRIMARY   candidate mean tracking error (tracking.pos_err_m.mean) is LOWER
            than baseline. This is the goal metric -- the brief's target is
            "follow the spline as closely as possible".
  GUARD-1   candidate distance_m >= 0.80 x baseline (it must still explore).
  GUARD-2   candidate coverage final_m3 >= 0.85 x baseline.
  GUARD-3   candidate stops_per_min <= 1.5 x baseline.
  GUARD-4   every candidate run flew: ended normally and distance_m > 20.

  ADOPT   = primary holds and every guard passes.
  REVERT  = primary fails, or GUARD-4 fails, or two-plus guards fail.
  MORE-RUNS = anything else (primary holds but one soft guard is marginal).

Usage:  PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_ab_powerlaw
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
MIN_PER_ARM = 3


def _runs():
    out = {"legacy": [], "powerlaw_lateral": []}
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "summary.json"))):
        try:
            with open(path) as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            continue
        variant = summary.get("controller_variant")
        # "Completed a flight" per the docstring: a cycle that died in
        # bring-up ("unhealthy stack; flew nothing") writes a summary with no
        # "flight" section and is a void attempt, not a sample. Found by the
        # A/B operator before any data was judged (runs/AB_LEDGER.md).
        if variant in out and isinstance(summary.get("flight"), dict):
            out[variant].append(summary)
    return out


def _metric(summary, *keys):
    node = summary.get("metrics") or {}
    for key in keys:
        node = (node or {}).get(key)
        if node is None:
            return None
    return node


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def main():
    arms = _runs()
    for arm, runs in arms.items():
        print("%s: %d runs" % (arm, len(runs)))
    if any(len(runs) < MIN_PER_ARM for runs in arms.values()):
        print("VERDICT: NOT ENOUGH DATA (need %d per arm)" % MIN_PER_ARM)
        return 2

    med = {}
    for arm, runs in arms.items():
        med[arm] = dict(
            pos_err=_median([_metric(r, "tracking", "pos_err_m", "mean")
                             for r in runs]),
            distance=_median([_metric(r, "motion", "distance_m")
                              for r in runs]),
            coverage=_median([_metric(r, "coverage", "final_m3")
                              for r in runs]),
            stops=_median([_metric(r, "motion", "stops_per_min")
                           for r in runs]),
        )
        print(arm, json.dumps(med[arm]))

    base, cand = med["legacy"], med["powerlaw_lateral"]
    if None in base.values() or None in cand.values():
        print("VERDICT: NOT ENOUGH DATA (a median metric is missing)")
        return 2

    primary = cand["pos_err"] < base["pos_err"]
    guards = {
        "distance>=0.80x": cand["distance"] >= 0.80 * base["distance"],
        "coverage>=0.85x": cand["coverage"] >= 0.85 * base["coverage"],
        "stops<=1.5x": cand["stops"] <= 1.5 * base["stops"],
        "every candidate flew": all(
            (_metric(r, "motion", "distance_m") or 0) > 20
            for r in arms["powerlaw_lateral"]),
    }
    print("primary (tracking improves): %s" % primary)
    for name, ok in guards.items():
        print("guard %-22s %s" % (name, ok))

    failed = [name for name, ok in guards.items() if not ok]
    if primary and not failed:
        print("VERDICT: ADOPT")
        return 0
    if (not primary) or ("every candidate flew" in failed) or len(failed) >= 2:
        print("VERDICT: REVERT (take it out of the code)")
        return 1
    print("VERDICT: MORE-RUNS (primary holds; %s marginal)" % failed)
    return 2


if __name__ == "__main__":
    sys.exit(main())
