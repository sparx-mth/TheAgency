"""Pre-registered verdict for v7.0: a slow yaw scan while the plan is parked.

Written 2026-08-31 BEFORE any v6.0 run existed.

Rationale (Findings C/F, runs/AUTOLOOP_JOURNAL.md). Both the coverage
shortfall and the tracking error are produced by the time the aircraft spends
STATIONARY (coverage vs distance r=0.95; along-track error is made almost
entirely while stopped). And while parked, nothing sweeps the camera: course
yaw only aims the nose when there is travel to aim it along. So no new
geometry enters the map, no frontier resolves, the planner re-picks the same
spot, and the aircraft stays parked -- a self-sustaining loop. The v4.0
experiment demonstrated the converse by accident: removing the sweep
deadlocked exploration outright.

v7.0 sweeps yaw slowly (0.5 rad/s) once the plan has been parked for 2 s and
the aircraft is not moving. It is yaw-only, so unlike every other lever tried
tonight it cannot fly the aircraft into anything, and it yields the instant
the plan asks for travel. It defers to the escape reflex and the tilt cut.

Samples: completed powerlaw_lateral flights by controller_rev.
  v6 = "v6.0"   v2 = "v2.1" (baseline, n=5)
Refuses to judge below 3 v6.0 runs.

Decision rule (medians across runs):
  PRIMARY  coverage >= 1.15 x baseline.
  E1       the mechanism: the fraction of the flight with a MOVING reference
           >= 1.15 x baseline. Sweeping should resolve frontiers, which is
           what produces plans; if that does not move, the idea is wrong even
           if coverage happens to drift up.
  S1 safety no run may log more than 5 tilt-cuts (a yaw-only change should
           never trip the attitude reflex; if it does, something is coupling).
  S2       distance >= 0.9 x baseline -- scanning must not eat travel time.

  ADOPT     = PRIMARY and E1 and S1 and S2
  REVERT    = S1 fails, or S2 fails, or (PRIMARY and E1 both fail)
  MORE-RUNS = otherwise

Usage: PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v7_park_scan
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
BASE_REV, CAND_REV = "v2.1", "v7.0"
MIN_PER_REV = 3


def _flights(rev):
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "summary.json"))):
        try:
            with open(path) as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        if (s.get("controller_variant") == "powerlaw_lateral"
                and isinstance(s.get("flight"), dict)
                and s.get("ended") == "completed"
                and s.get("controller_rev") == rev):
            out.append((os.path.dirname(path), s))
    return out


def _ref_moving(run_dir):
    """Fraction of clearance samples where the reference was actually moving.

    The probe omits along_m exactly when the reference velocity is ~0, so its
    presence is a direct read of "the planner is asking for travel".
    """
    try:
        with open(os.path.join(run_dir, "clearance.jsonl")) as fh:
            rows = [json.loads(line) for line in fh]
    except OSError:
        return None
    if not rows:
        return None
    return sum(1 for r in rows if r.get("along_m") is not None) / len(rows)


def _truth_stats(run_dir):
    """Stationary fraction and ranger spread over airborne samples."""
    airborne = False
    slow = n = 0
    rangers = []
    try:
        with open(os.path.join(run_dir, "truth.jsonl")) as fh:
            for line in fh:
                row = json.loads(line)
                st = row.get("state") or {}
                if st.get("age") is not None:
                    airborne = bool(st.get("airborne", airborne))
                    if airborne and st.get("ranger") is not None:
                        rangers.append(st["ranger"])
                vel = row.get("velocity") or {}
                if airborne and vel.get("age") is not None:
                    n += 1
                    if math.hypot(vel["vx"], vel["vy"]) < 0.05:
                        slow += 1
    except OSError:
        return None, None
    return (slow / n if n else None,
            statistics.pstdev(rangers) if len(rangers) > 2 else None)


def _tilts(run_dir):
    try:
        with open(os.path.join(run_dir, "logs", "falcon_roslaunch.log"),
                  errors="replace") as fh:
            return sum(1 for l in fh if "cutting drive until it is back under" in l)
    except OSError:
        return 0


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _stats(runs):
    ts = [_truth_stats(d) for d, _ in runs]
    return dict(
        coverage=_med([((s.get("metrics") or {}).get("coverage") or {}).get("final_m3")
                       for _, s in runs]),
        distance=_med([((s.get("metrics") or {}).get("motion") or {}).get("distance_m")
                       for _, s in runs]),
        stationary=_med([t[0] for t in ts]),
        ref_moving=_med([_ref_moving(d) for d, _ in runs]),
        ranger_sd=_med([t[1] for t in ts]),
    )


def main():
    base, cand = _flights(BASE_REV), _flights(CAND_REV)
    print("%s: %d | %s: %d" % (BASE_REV, len(base), CAND_REV, len(cand)))
    if len(base) < MIN_PER_REV or len(cand) < MIN_PER_REV:
        print("VERDICT: NOT ENOUGH DATA (need %d per revision)" % MIN_PER_REV)
        return 2
    a, b = _stats(base), _stats(cand)
    print(BASE_REV, json.dumps(a))
    print(CAND_REV, json.dumps(b))
    if None in a.values() or None in b.values():
        print("VERDICT: NOT ENOUGH DATA (a median metric is missing)")
        return 2
    tilts = [_tilts(d) for d, _ in cand]
    primary = b["coverage"] >= 1.15 * a["coverage"]
    e1 = b["ref_moving"] >= 1.15 * a["ref_moving"]
    s1 = all(t <= 5 for t in tilts)
    s2 = b["distance"] >= 0.9 * a["distance"]
    print("PRIMARY coverage >=1.15x  : %s" % primary)
    print("E1 ref-moving >=1.15x     : %s (%.3f vs %.3f)"
          % (e1, b["ref_moving"], a["ref_moving"]))
    print("S1 no tilt-cuts           : %s (%s)" % (s1, tilts))
    print("S2 distance >=0.9x        : %s" % s2)
    if primary and e1 and s1 and s2:
        print("VERDICT: ADOPT-V7")
        return 0
    if (not s1) or (not s2) or ((not primary) and (not e1)):
        print("VERDICT: REVERT (park_scan_rate back to 0.0)")
        return 1
    print("VERDICT: MORE-RUNS")
    return 2


if __name__ == "__main__":
    sys.exit(main())
