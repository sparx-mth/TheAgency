"""The pre-registered P38 test, written before its data existed.

Five hypotheses in this campaign looked strong on small or reused samples and
died. The pattern behind all five was the same: the analysis was chosen, or
adjusted, after the numbers were in view. So this file fixes the analysis in
advance -- the filter, the metric, the threshold and the sample size are all
decided here, and running it is a single yes/no with nothing left to tune.

Hypothesis (from a seven-way scan of runs already in hand, which can generate a
hypothesis but cannot confirm one):

    corr(tracking.pos_err_m.mean, coverage.final_m3) <= -0.5

Sample: runs whose directory timestamp is after 2026-08-20 13:30 UTC -- i.e.
flown after the hypothesis was written down -- with ``config.max_vel == 0.8``.
Refuses to report anything below n=15.

Usage:  PYTHONPATH=. .venv/bin/python -m sparx_agency.tools.falcon_campaign.test_p38
"""

import datetime
import glob
import json
import os
import sys

#: Runs before this were used to GENERATE the hypothesis and cannot test it.
CUTOFF = datetime.datetime(2026, 8, 20, 13, 30)
MIN_RUNS = 15
THRESHOLD = -0.5
MAX_VEL = 0.8


def _pearson(xs, ys):
    """Pearson correlation of two equal-length sequences."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def sample(runs_dir="runs"):
    """The pre-registered sample: (run, pos_err mean, final volume) per run."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "2026*Z"))):
        name = os.path.basename(path)
        try:
            stamp = datetime.datetime.strptime(name[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if stamp < CUTOFF:
            continue
        metrics = os.path.join(path, "metrics.json")
        if not os.path.isfile(metrics):
            continue
        try:
            m = json.load(open(metrics))
        except ValueError:
            continue
        config = m.get("config") or {}
        coverage = m.get("coverage") or {}
        error = ((m.get("tracking") or {}).get("pos_err_m") or {}).get("mean")
        if config.get("max_vel") != MAX_VEL:
            continue
        if coverage.get("final_m3") is None or error is None:
            continue
        rows.append((name, error, coverage["final_m3"]))
    return rows


def main():
    """Run the test once and print its verdict."""
    rows = sample()
    print("P38: corr(pos_err mean, final_m3) <= %.1f, max_vel %.1f, runs after %s"
          % (THRESHOLD, MAX_VEL, CUTOFF))
    print("sample: n=%d (needs %d)" % (len(rows), MIN_RUNS))
    for name, error, final in rows:
        print("   %-22s pos_err %5.2f m   final %6.0f m3" % (name, error, final))
    if len(rows) < MIN_RUNS:
        print("\nNOT ENOUGH DATA -- no coefficient reported. This is the point:")
        print("five earlier leads died because they were read at n<15.")
        return 2
    r = _pearson([x[1] for x in rows], [x[2] for x in rows])
    print("\ncorrelation: %+.3f" % r)
    print("VERDICT: %s" % ("PASS -- hypothesis survives; NOW ask whether the error causes "
                           "low coverage or merely accompanies it"
                           if r <= THRESHOLD else
                           "FAIL -- record it as the fifth dead lead and move on"))
    return 0 if r <= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
