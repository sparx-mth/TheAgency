"""Pre-registered verdict for v6.0: altitude_band_m 0.30 -> 0.60.

Written 2026-08-31 BEFORE any v6.0 run existed.

Rationale (Finding E + F, runs/AUTOLOOP_JOURNAL.md). FALCON's reference sits
ABOVE the aircraft in every flight measured (mean dz +0.06..+0.38 m, p90 up
to 1.13 m) because it plans inside a flight_band the aircraft cannot use: the
aircraft is pinned near 1.2 m and the twist adapter could bias that by only
+/-0.3 m in total, so viewpoints beyond that are unreachable by construction.
Finding F then showed that BOTH the coverage shortfall and the tracking error
are produced by time spent stationary. Giving the aircraft more of the
vertical band should let it actually arrive at viewpoints instead of sitting
under them.

Deliberate caution: band 1.0 with a COARSE 0.3 m nudge was flown historically
and railed the live altitude target, driving the hold loop hard (z sd 42 ->
114) and costing horizontal speed. The nudge is now 0.15 m, so v6.0 doubles
the range at half the step, and stops well short of the old 1.0.

Samples: completed powerlaw_lateral flights by controller_rev.
  v6 = "v6.0"   v2 = "v2.1" (baseline, n=5)
Refuses to judge below 3 v6.0 runs.

Decision rule (medians across runs):
  PRIMARY  coverage >= 1.15 x baseline.
  E1       stationary fraction (airborne samples below 0.05 m/s) <= 0.85 x
           baseline -- the mechanism this is supposed to fix.
  S1 safety altitude hold must not destabilise: median ranger standard
           deviation <= 1.5 x baseline, AND no run may log >5 tilt-cuts.
           This is the exact regression the historical band=1.0 caused.
  S2       distance >= 0.9 x baseline (a hold loop fighting itself costs
           horizontal speed -- the other half of that historical regression).

  ADOPT     = PRIMARY and E1 and S1 and S2
  REVERT    = S1 fails, or S2 fails, or (PRIMARY and E1 both fail)
  MORE-RUNS = otherwise

Usage: PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v6_altitude_band
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
BASE_REV, CAND_REV = "v2.1", "v6.0"
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
    e1 = b["stationary"] <= 0.85 * a["stationary"]
    s1 = b["ranger_sd"] <= 1.5 * a["ranger_sd"] and all(t <= 5 for t in tilts)
    s2 = b["distance"] >= 0.9 * a["distance"]
    print("PRIMARY coverage >=1.15x  : %s" % primary)
    print("E1 stationary <=0.85x     : %s (%.3f vs %.3f)"
          % (e1, b["stationary"], a["stationary"]))
    print("S1 altitude stable        : %s (ranger sd %.3f vs %.3f, tilts %s)"
          % (s1, b["ranger_sd"], a["ranger_sd"], tilts))
    print("S2 distance >=0.9x        : %s" % s2)
    if primary and e1 and s1 and s2:
        print("VERDICT: ADOPT-V6")
        return 0
    if (not s1) or (not s2) or ((not primary) and (not e1)):
        print("VERDICT: REVERT (altitude_band back to 0.30)")
        return 1
    print("VERDICT: MORE-RUNS")
    return 2


if __name__ == "__main__":
    sys.exit(main())
