"""Pre-registered verdict for the INTERLEAVED v7.1 vs v2.1c campaign.

Written 2026-08-31 before the interleaved campaign produced any flight.

Design, and why it differs from every earlier test tonight (Finding G):
baseline coverage has CV 27% and distance CV 51%, so a 15% coverage change
needs ~52 flights per arm at 80% power. Earlier verdicts were run at n=3,
which can only resolve changes of roughly 60% -- the test structure was fine
but the sample size never was. So this one:

  * **interleaves** the arms (alternating every cycle) instead of comparing
    against a baseline collected hours earlier, because the stack drifts and
    has been restarted repeatedly;
  * leads on **low-variance** metrics and treats coverage as a guard;
  * reports a bootstrap confidence interval rather than a bare median ratio,
    and refuses to call anything whose interval spans 1.0.

Arms, taken from summary.json (completed flights only):
  v7.1  = parked yaw scan 0.25 rad/s, trigger 4 s
  v2.1c = identical build, scan off

Metrics, in order of how well the platform resolves them:
  PRIMARY  moving-reference fraction (from clearance.jsonl: the probe omits
           along_m exactly when the reference velocity is ~0, so this reads
           "the planner is asking for travel"). This is the mechanism, and it
           was the only metric that moved cleanly at n=3.
  M2       stationary fraction (airborne samples under 0.05 m/s) -- should FALL.
  G1 guard coverage ratio must not be worse than 0.85x (it is too noisy to
           carry a positive claim, but a large loss is still detectable).
  G2 guard distance ratio >= 0.85x.
  S1 guard no run above 5 tilt-cuts; ranger sd <= 2x (v7.0 at 0.5 rad/s
           destabilised altitude to 0.345 vs 0.085 -- this guard exists
           because dropping it was a mistake in the v7.0 test).

  ADOPT     = PRIMARY interval entirely above 1.0, M2 improved, all guards hold
  REVERT    = any guard fails, or PRIMARY interval entirely below 1.0
  MORE-RUNS = interval spans 1.0 (report how many more flights are needed)

Usage: PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v71_interleaved
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
ARMS = {"v7.1": [], "v2.1c": []}
MIN_PER_ARM = 6
random.seed(20260831)


def _load():
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "summary.json"))):
        try:
            with open(path) as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        rev = s.get("controller_rev")
        if (rev in ARMS and isinstance(s.get("flight"), dict)
                and s.get("ended") == "completed"):
            ARMS[rev].append((os.path.dirname(path), s))


def _metrics(run_dir, summary):
    try:
        with open(os.path.join(run_dir, "clearance.jsonl")) as fh:
            rows = [json.loads(l) for l in fh]
    except OSError:
        return None
    if not rows:
        return None
    ref_moving = sum(1 for r in rows if r.get("along_m") is not None) / len(rows)

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
        return None
    m = summary.get("metrics") or {}
    tilts = 0
    try:
        with open(os.path.join(run_dir, "logs", "falcon_roslaunch.log"),
                  errors="replace") as fh:
            tilts = sum(1 for l in fh if "cutting drive until it is back under" in l)
    except OSError:
        pass
    return dict(
        ref_moving=ref_moving,
        stationary=(slow / n if n else None),
        coverage=(m.get("coverage") or {}).get("final_m3"),
        distance=(m.get("motion") or {}).get("distance_m"),
        ranger_sd=(statistics.pstdev(rangers) if len(rangers) > 2 else None),
        tilts=tilts,
    )


def _boot_ratio(a, b, iters=20000):
    """Bootstrap CI for median(a)/median(b)."""
    out = []
    for _ in range(iters):
        ra = [random.choice(a) for _ in a]
        rb = [random.choice(b) for _ in b]
        mb = statistics.median(rb)
        if mb:
            out.append(statistics.median(ra) / mb)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main():
    _load()
    data = {k: [_metrics(d, s) for d, s in v] for k, v in ARMS.items()}
    data = {k: [m for m in v if m] for k, v in data.items()}
    for k, v in data.items():
        print("%s: %d completed flights" % (k, len(v)))
    if any(len(v) < MIN_PER_ARM for v in data.values()):
        print("VERDICT: NOT ENOUGH DATA (need %d per arm; interleaving in "
              "progress)" % MIN_PER_ARM)
        return 2

    def col(arm, key):
        return [m[key] for m in data[arm] if m.get(key) is not None]

    for key in ("ref_moving", "stationary", "coverage", "distance", "ranger_sd"):
        a, b = col("v7.1", key), col("v2.1c", key)
        print("%-11s v7.1 median %8.3f | v2.1c median %8.3f | ratio %.2f"
              % (key, statistics.median(a), statistics.median(b),
                 statistics.median(a) / statistics.median(b)))

    lo, hi = _boot_ratio(col("v7.1", "ref_moving"), col("v2.1c", "ref_moving"))
    print("PRIMARY ref_moving ratio 95%% CI: [%.2f, %.2f]" % (lo, hi))
    m2 = (statistics.median(col("v7.1", "stationary"))
          < statistics.median(col("v2.1c", "stationary")))
    g1 = (statistics.median(col("v7.1", "coverage"))
          >= 0.85 * statistics.median(col("v2.1c", "coverage")))
    g2 = (statistics.median(col("v7.1", "distance"))
          >= 0.85 * statistics.median(col("v2.1c", "distance")))
    s1 = (all(m["tilts"] <= 5 for m in data["v7.1"])
          and statistics.median(col("v7.1", "ranger_sd"))
          <= 2.0 * statistics.median(col("v2.1c", "ranger_sd")))
    print("M2 stationary improved : %s" % m2)
    print("G1 coverage >=0.85x    : %s" % g1)
    print("G2 distance >=0.85x    : %s" % g2)
    print("S1 tilt/altitude       : %s" % s1)

    if not (g1 and g2 and s1):
        print("VERDICT: REVERT")
        return 1
    if lo > 1.0 and m2:
        print("VERDICT: ADOPT-V7.1")
        return 0
    if hi < 1.0:
        print("VERDICT: REVERT")
        return 1
    print("VERDICT: MORE-RUNS (the interval still spans 1.0)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
