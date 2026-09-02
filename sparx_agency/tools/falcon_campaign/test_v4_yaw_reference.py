"""Pre-registered verdict for v4.0: yaw_mode=reference + yaw_dot feedforward.

Written 2026-08-31 BEFORE any v4.0 run existed.

Rationale. "course" yaw (nose along travel) is a workaround from the era when
the lateral axis was disabled and sideways demand had to be turned into
forward demand. Lateral now works, so the workaround can go, and FALCON's own
yaw plan -- which exists to point the depth camera at the frontiers it wants
to map -- can be followed instead, with traj_server's analytic `yaw_dot` fed
forward so it does not lag. Measured baseline in course mode: heading error
p50 18-28 deg, p90 46-50, beyond 45 deg for 10-15% of the flight.

This is a MAPPING hypothesis as much as a control one: aiming the sensor
where the planner wants to look should map more per metre flown.

Samples: completed powerlaw_lateral flights, by controller_rev.
  v4 = "v4.0"  (reference yaw + yaw_dot feedforward)
  v2 = "v2.1"  (course yaw)   -- the baseline
Refuses to judge below 3 per revision.

Decision rule (medians across runs):
  PRIMARY  coverage PER METRE FLOWN (final_m3 / distance_m) >= 1.15 x baseline
           -- the direct statement of "aiming the camera better maps more".
  G1 guard total coverage >= 0.95 x baseline. Better aiming must not be paid
           for by flying so much less that the map ends up smaller: with the
           nose off the travel direction more demand lands on the lateral
           axis, which is capped at 600 counts (~0.43 m/s).
  S1 safety no run may log more than 5 tilt-cuts; roll time above 5 deg
           <= 1.3 x baseline (reference yaw must not undo the gentle roll).

  ADOPT     = PRIMARY and G1 and S1
  REVERT    = S1 fails, or G1 fails, or PRIMARY fails
  MORE-RUNS = only when PRIMARY is within 5% of its bar and the guards hold

Usage:
  PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v4_yaw_reference
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
BASE_REV, CAND_REV = "v2.1", "v4.0"
MIN_PER_REV = 3
ROLL_DEG = 5.0


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


def _roll_frac(run_dir):
    airborne = False
    n = hot = 0
    try:
        with open(os.path.join(run_dir, "truth.jsonl")) as fh:
            for line in fh:
                row = json.loads(line)
                st = row.get("state") or {}
                if st.get("age") is not None:
                    airborne = bool(st.get("airborne", airborne))
                tr = row.get("truth") or {}
                if airborne and tr.get("age") is not None:
                    n += 1
                    if abs(math.degrees(tr["roll"])) > ROLL_DEG:
                        hot += 1
    except OSError:
        return None
    return hot / n if n else None


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
    cov, dist, per_m = [], [], []
    for _, s in runs:
        m = s.get("metrics") or {}
        c = (m.get("coverage") or {}).get("final_m3")
        d = (m.get("motion") or {}).get("distance_m")
        cov.append(c)
        dist.append(d)
        if c is not None and d:
            per_m.append(c / d)
    return dict(coverage=_med(cov), distance=_med(dist),
                cov_per_m=_med(per_m),
                roll=_med([_roll_frac(d) for d, _ in runs]))


def main():
    base, cand = _flights(BASE_REV), _flights(CAND_REV)
    print("%s: %d flights | %s: %d flights"
          % (BASE_REV, len(base), CAND_REV, len(cand)))
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
    primary = b["cov_per_m"] >= 1.15 * a["cov_per_m"]
    near = b["cov_per_m"] >= 1.09 * a["cov_per_m"]      # within 5% of the bar
    g1 = b["coverage"] >= 0.95 * a["coverage"]
    s1 = all(t <= 5 for t in tilts) and b["roll"] <= 1.3 * a["roll"]
    print("PRIMARY coverage/metre >=1.15x : %s (%.2f vs %.2f)"
          % (primary, b["cov_per_m"], a["cov_per_m"]))
    print("G1 total coverage >=0.95x      : %s" % g1)
    print("S1 safety/roll                 : %s (tilt-cuts %s)" % (s1, tilts))

    if primary and g1 and s1:
        print("VERDICT: ADOPT-V4")
        return 0
    if s1 and g1 and near:
        print("VERDICT: MORE-RUNS")
        return 2
    print("VERDICT: REVERT (back to course yaw)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
