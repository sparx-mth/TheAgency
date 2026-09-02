"""Pre-registered verdict for v8.0: blocked_region_radius 1.5 -> 2.75 m.

Written 2026-09-01 BEFORE any v8.0 flight existed.

WHY THIS AND NOT THE OTHER CANDIDATES (Finding I, runs/AUTOLOOP_JOURNAL.md).
Measured over 41 flights: 91.9% of the time with no moving reference is
A*-plan-fail flooding, and 86% of that flood time is ONE terminal lock on a
single unreachable viewpoint -- worst case 254 s emitting the identical
"Next pos" line 17,551 times against 17,925 plan fails, with the reference
never moving once. The blacklist exists to retire such a viewpoint, but
`sweepBlockedFrontiers` retired ZERO clusters in the locked runs while 10-11
shadows were re-struck: a first strike only shadows 1.5 m (the C++ default,
never set) while viewpoints are sampled to candidate_rmax 5.5 m, so the tour
re-offers the same target forever. 2.75 makes strike 1 cover 2.75 m and
strike >=2 escalate to exactly 5.5 m, which is the geometric requirement the
launch file's own comment already states.

Explicitly NOT chosen: raising TOUR_COMMIT_MAX_S. It was measured to make
things worse in the same investigation -- tour churn correlates with PROGRESS
(r=+0.68 with coverage, -0.36 with stationarity), 81% of starved windows
already have a frozen target, and an 8.0 s commit was already flown on
2026-08-20 and moved the mediator 4-5x while coverage FELL.

ARMS (interleaved, alternating every cycle so drift hits both equally):
  v8.0  = blocked_region_radius 2.75
  v2.1d = blocked_region_radius 1.5 (the C++ default; otherwise identical)

METRICS. Coverage has CV 27% and cannot decide anything at feasible n
(Finding G), so it is a guard only. The primary is a BINARY endpoint with a
large expected effect, which is cheap to power:

  PRIMARY  fraction of flights containing a plan-fail flood >= 120 s.
           Baseline: 6 of 8 recent flights. Expect <= 1 of 8 if the shadow
           now retires the frontier. Fisher exact on 6/8 vs 1/8 gives p~0.04.
  M2       "re-reported on strike" count per flight (r=-0.98 with the
           moving-reference fraction): expect a fall from ~6 to <= 2.
  M3       longest single plan-fail flood, seconds (r=-0.96 with distance):
           expect the mean to fall from ~168 s to under 60 s.

GUARDS -- this change can sterilise the map by blacklisting too much, which
is a documented failure (192 shadows at a 2 m radius once confined a mission
to one corner at 146 m3):
  G1  coverage median >= 0.85 x control median.
  G2  no flight may log `finished: True` with zero reopens AND a coverage
      plateau > 60 s (the recorded premature-finish signature).
  G3  no flight may reach zero frontiers-to-visit more than once.
  G4  max simultaneous standing shadows <= 10.

  ADOPT     = PRIMARY improves AND M2 improves AND all guards hold
  REVERT    = any guard fails, or PRIMARY does not improve
  MORE-RUNS = mixed mechanism signals with guards intact

Usage: PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v8_blocked_radius
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
ARMS = ("v8.0", "v2.1d")
MIN_PER_ARM = 6
FLOOD_GAP_S = 2.0          # a gap this long ends a flood
FLOOD_MIN_S = 120.0        # what counts as a flood for the primary endpoint


def _flights(rev):
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*", "summary.json"))):
        try:
            with open(path) as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            continue
        if (s.get("controller_rev") == rev
                and isinstance(s.get("flight"), dict)
                and s.get("ended") == "completed"):
            out.append((os.path.dirname(path), s))
    return out


def _log_metrics(run_dir):
    """Flood structure and blacklist activity from the FALCON log."""
    stamps = []
    re_reported = 0
    zero_to_visit = 0
    finished_no_reopen = 0
    path = os.path.join(run_dir, "logs", "falcon_roslaunch.log")
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if "No path to next viewpoint using default" in line:
                    m = re.search(r"\[(\d{10}\.\d+)\]", line)
                    if m:
                        stamps.append(float(m.group(1)))
                elif "re-reported on strike" in line:
                    re_reported += 1
                elif "still to visit" in line and re.search(r";\s*0\s+still to visit", line):
                    zero_to_visit += 1
                elif "finished: True" in line and "reopened: 0" in line:
                    finished_no_reopen += 1
    except OSError:
        return None
    floods = []
    if stamps:
        start = prev = stamps[0]
        for t in stamps[1:]:
            if t - prev > FLOOD_GAP_S:
                floods.append(prev - start)
                start = t
            prev = t
        floods.append(prev - start)
    longest = max(floods) if floods else 0.0
    return dict(longest_flood_s=longest,
                had_flood=1 if longest >= FLOOD_MIN_S else 0,
                re_reported=re_reported,
                zero_to_visit=zero_to_visit,
                finished_no_reopen=finished_no_reopen)


def _coverage(summary):
    return ((summary.get("metrics") or {}).get("coverage") or {}).get("final_m3")


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    data = {}
    for rev in ARMS:
        rows = []
        for d, s in _flights(rev):
            m = _log_metrics(d)
            if m:
                m["coverage"] = _coverage(s)
                m["run"] = os.path.basename(d)
                rows.append(m)
        data[rev] = rows
        print("%s: %d completed flights" % (rev, len(rows)))
    if any(len(v) < MIN_PER_ARM for v in data.values()):
        print("VERDICT: NOT ENOUGH DATA (need %d per arm)" % MIN_PER_ARM)
        return 2

    cand, ctrl = data["v8.0"], data["v2.1d"]
    for name, rows in (("v8.0", cand), ("v2.1d", ctrl)):
        print("%-6s floods>=120s %d/%d | longest_flood mean %.0f s | "
              "re-reported med %.1f | coverage med %.0f"
              % (name, sum(r["had_flood"] for r in rows), len(rows),
                 statistics.fmean(r["longest_flood_s"] for r in rows),
                 _med([r["re_reported"] for r in rows]),
                 _med([r["coverage"] for r in rows]) or 0))

    prim = (sum(r["had_flood"] for r in cand) / len(cand)
            < sum(r["had_flood"] for r in ctrl) / len(ctrl))
    m2 = _med([r["re_reported"] for r in cand]) < _med([r["re_reported"] for r in ctrl])
    g1 = (_med([r["coverage"] for r in cand])
          >= 0.85 * _med([r["coverage"] for r in ctrl]))
    g2 = all(r["finished_no_reopen"] == 0 for r in cand)
    g3 = all(r["zero_to_visit"] <= 1 for r in cand)
    print("PRIMARY flood rate down : %s" % prim)
    print("M2 re-reported down     : %s" % m2)
    print("G1 coverage >=0.85x     : %s" % g1)
    print("G2 no premature finish  : %s" % g2)
    print("G3 frontiers not emptied: %s" % g3)

    if not (g1 and g2 and g3):
        print("VERDICT: REVERT (guard failed -- likely map sterilisation)")
        return 1
    if prim and m2:
        print("VERDICT: ADOPT-V8")
        return 0
    if not prim:
        print("VERDICT: REVERT (the lock is not being cleared)")
        return 1
    print("VERDICT: MORE-RUNS")
    return 2


if __name__ == "__main__":
    sys.exit(main())
