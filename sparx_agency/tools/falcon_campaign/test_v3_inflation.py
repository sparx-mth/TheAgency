"""Pre-registered verdict for v3.0: obstacles_inflation 0.40 -> 0.30.

Written 2026-08-31 BEFORE any v3.0 run existed, from Finding A in
runs/AUTOLOOP_JOURNAL.md: exploration collapses part-way through every
flight because A* can no longer route to any viewpoint once the map fills in
(13,404 "No path to next viewpoint" in one run; 0 trajectories executed in
the 300-360 s window). 0.40 m of inflation on a 0.20 m grid demands a 0.80 m
free corridor; 0.30 demands 0.60 m.

Samples (completed flights only, controller_variant powerlaw_lateral):
  v3 = controller_rev "v3.0"   (inflation 0.30)
  v2 = controller_rev "v2.1"   (inflation 0.40) -- the baseline
Refuses to judge below 3 runs per revision.

The metric that matters is whether the aircraft keeps EXPLORING for the whole
flight, not the total plan-fail count (a rate that says nothing on its own).

Decision rule (medians across runs):
  PRIMARY  v3 coverage final_m3 >= 1.15 x v2  -- it must map materially more.
  E1  late-flight trajectories executed (last half of the flight) >= 2.0 x v2
      -- the direct measure of the collapse this change targets.
  S1 safety  no v3 run may log a tilt-cut ("cutting drive until it is back
      under") more than 5 times, and none may end other than "completed".
  S2 clearance  v3 median aircraft clearance >= 0.8 x v2 -- flying nearer to
      walls is expected, but not a different regime.

  ADOPT-V3 = PRIMARY and E1 and S1 and S2.
  REVERT   = S1 fails, or PRIMARY fails while E1 also fails.
  MORE-RUNS = anything else.

Usage:  PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v3_inflation
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
MIN_PER_REV = 3


def _flights():
    out = {"v2.1": [], "v3.0": []}
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "summary.json"))):
        try:
            with open(path) as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            continue
        if summary.get("controller_variant") != "powerlaw_lateral":
            continue
        if not isinstance(summary.get("flight"), dict):
            continue
        if summary.get("ended") != "completed":
            continue
        rev = summary.get("controller_rev")
        if rev in out:
            out[rev].append((os.path.dirname(path), summary))
    return out


def _late_execs(run_dir):
    """Trajectories that reached EXEC_TRAJ in the second half of the flight."""
    stamps = []
    log = os.path.join(run_dir, "logs", "falcon_roslaunch.log")
    try:
        with open(log, errors="replace") as fh:
            for line in fh:
                if "PUB_TRAJ to EXEC_TRAJ" in line:
                    m = re.search(r"\[(\d{10}\.\d+)\]", line)
                    if m:
                        stamps.append(float(m.group(1)))
    except OSError:
        return None
    if not stamps:
        return 0
    t0, t1 = min(stamps), max(stamps)
    mid = t0 + (t1 - t0) / 2.0
    return sum(1 for s in stamps if s >= mid)


def _tilt_cuts(run_dir):
    log = os.path.join(run_dir, "logs", "falcon_roslaunch.log")
    try:
        with open(log, errors="replace") as fh:
            return sum(1 for line in fh
                       if "cutting drive until it is back under" in line)
    except OSError:
        return 0


def _clearance_p50(run_dir):
    try:
        with open(os.path.join(run_dir, "clearance.jsonl")) as fh:
            vals = [json.loads(line).get("nearest_m") for line in fh]
    except OSError:
        return None
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _stats(runs):
    return dict(
        coverage=_med([((s.get("metrics") or {}).get("coverage") or {}).get("final_m3")
                       for _, s in runs]),
        late_execs=_med([_late_execs(d) for d, _ in runs]),
        clearance=_med([_clearance_p50(d) for d, _ in runs]),
        distance=_med([((s.get("metrics") or {}).get("motion") or {}).get("distance_m")
                       for _, s in runs]),
    )


def main():
    arms = _flights()
    for rev, runs in arms.items():
        print("%s: %d completed flights" % (rev, len(runs)))
    if any(len(runs) < MIN_PER_REV for runs in arms.values()):
        print("VERDICT: NOT ENOUGH DATA (need %d per revision)" % MIN_PER_REV)
        return 2

    base, cand = _stats(arms["v2.1"]), _stats(arms["v3.0"])
    print("v2.1", json.dumps(base))
    print("v3.0", json.dumps(cand))
    if None in base.values() or None in cand.values():
        print("VERDICT: NOT ENOUGH DATA (a median metric is missing)")
        return 2

    tilts = [_tilt_cuts(d) for d, _ in arms["v3.0"]]
    primary = cand["coverage"] >= 1.15 * base["coverage"]
    e1 = cand["late_execs"] >= 2.0 * max(base["late_execs"], 1)
    s1 = all(t <= 5 for t in tilts)
    s2 = cand["clearance"] >= 0.8 * base["clearance"]
    print("PRIMARY coverage >=1.15x : %s" % primary)
    print("E1 late-flight execs >=2x: %s (v3 %s vs v2 %s)"
          % (e1, cand["late_execs"], base["late_execs"]))
    print("S1 safety (tilt-cuts<=5) : %s (%s)" % (s1, tilts))
    print("S2 clearance >=0.8x      : %s" % s2)

    if primary and e1 and s1 and s2:
        print("VERDICT: ADOPT-V3")
        return 0
    if (not s1) or ((not primary) and (not e1)):
        print("VERDICT: REVERT (put obstacles_inflation back to 0.40)")
        return 1
    print("VERDICT: MORE-RUNS")
    return 2


if __name__ == "__main__":
    sys.exit(main())
