"""Pre-registered round-2 verdict: lateral cap 600 vs the round-1 candidate.

Written 2026-08-31, BEFORE any round-2 run existed. Round 1 (see
test_ab_powerlaw.py, runs/AB_LEDGER.md) ended MORE-RUNS: tracking error
halved vs legacy, wedging collapsed, but the operator flagged the roll as too
aggressive -- traced to the turn-crab feedforward slamming the lateral axis
to p50 ~600 / p90 ~900 counts on every course change. Round 2 changes ONE
variable: config.LATERAL_AXIS_CAP 900 -> 600. The calibration curve is
untouched by explicit instruction.

Samples (from summary.json; COMPLETED flights only -- a cycle aborted by the
liveness guard is a void attempt under the runbook's own void-and-retry
protocol, and round 2's one abort was caused by the since-reverted map
enlargement, not the controller):
  v2 = controller_variant powerlaw_lateral AND controller_rev == "v2.1"
       (cap 600 + gentle lateral slew + small map -- the configuration of
       record; recorded per run since 2026-08-31 after round 2's sample
       silently mixed two slew configs and two maps)
  v1 = controller_variant powerlaw_lateral AND no controller_rev key AND
       (no lateral_axis_cap key or 900) -- the four round-1 candidate flights.
The interim cap-600 runs 8-10 (old slew and/or big map) belong to NEITHER
sample. Refuses to judge below 3 runs of v2.

Decision rule (medians across runs):
  R1 calm    v2 time with |roll| > 5 deg (airborne)  <= 0.6 x v1.
  R2 track   v2 clearance pos_err_m p50              <= 1.2 x v1.
  R3 go      v2 motion distance_m                    >= 1.0 x v1.

  ADOPT-V2      = all three hold.
  REVERT-TO-V1  = R2 fails (the cap costs tracking -- the goal metric).
  MORE-RUNS     = anything else.

Usage:  PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_ab_powerlaw_v2
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
MIN_V2 = 3
ROLL_DEG = 5.0


def _flights():
    v1, v2 = [], []
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
            continue          # aborted cycles are void attempts, not samples
        rev = summary.get("controller_rev")
        cap = summary.get("lateral_axis_cap")
        if rev == "v2.1":
            v2.append((os.path.dirname(path), summary))
        elif rev is None and cap in (None, 900):
            v1.append((os.path.dirname(path), summary))
        # rev-less cap-600 runs (interim 8-10) belong to neither sample
    return v1, v2


def _roll_frac(run_dir):
    """Airborne fraction with |roll| beyond ROLL_DEG, from truth.jsonl."""
    airborne = False
    n = hot = 0
    try:
        with open(os.path.join(run_dir, "truth.jsonl")) as fh:
            for line in fh:
                row = json.loads(line)
                state = row.get("state") or {}
                if state.get("age") is not None:
                    airborne = bool(state.get("airborne", airborne))
                truth = row.get("truth") or {}
                if airborne and truth.get("age") is not None:
                    n += 1
                    if abs(math.degrees(truth["roll"])) > ROLL_DEG:
                        hot += 1
    except OSError:
        return None
    return hot / n if n else None


def _pos_err_p50(run_dir):
    try:
        with open(os.path.join(run_dir, "clearance.jsonl")) as fh:
            errs = [json.loads(line).get("pos_err_m") for line in fh]
    except OSError:
        return None
    errs = [e for e in errs if e is not None]
    return statistics.median(errs) if errs else None


def _stats(runs):
    roll = [_roll_frac(d) for d, _ in runs]
    err = [_pos_err_p50(d) for d, _ in runs]
    dist = [((s.get("metrics") or {}).get("motion") or {}).get("distance_m")
            for _, s in runs]
    med = lambda xs: (statistics.median([x for x in xs if x is not None])
                      if any(x is not None for x in xs) else None)
    return dict(roll=med(roll), err=med(err), dist=med(dist))


def main():
    v1, v2 = _flights()
    print("v1 (cap 900): %d flights, v2 (cap 600): %d flights" % (len(v1), len(v2)))
    if len(v2) < MIN_V2 or len(v1) < MIN_V2:
        print("VERDICT: NOT ENOUGH DATA (need %d per revision)" % MIN_V2)
        return 2
    a, b = _stats(v1), _stats(v2)
    print("v1", json.dumps(a), "\nv2", json.dumps(b))
    if None in a.values() or None in b.values():
        print("VERDICT: NOT ENOUGH DATA (a median metric is missing)")
        return 2
    calm = b["roll"] <= 0.6 * a["roll"]
    track = b["err"] <= 1.2 * a["err"]
    go = b["dist"] >= 1.0 * a["dist"]
    print("R1 calm  (roll>5deg time <=0.6x): %s" % calm)
    print("R2 track (pos_err p50 <=1.2x):    %s" % track)
    print("R3 go    (distance >=1.0x):       %s" % go)
    if calm and track and go:
        print("VERDICT: ADOPT-V2")
        return 0
    if not track:
        print("VERDICT: REVERT-TO-V1 (the cap costs tracking)")
        return 1
    print("VERDICT: MORE-RUNS")
    return 2


if __name__ == "__main__":
    sys.exit(main())
