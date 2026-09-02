"""Pre-registered verdict for the reference-acceleration lead (accel_lead_s).

Written 2026-08-31 BEFORE any run with a raised lead existed.

Motivation (measured over five v2.1 flights, 700 moving-reference samples):
the remaining tracking error is **74% along-track** and only 26% cross-track.
|along| is p50 0.17 m / p90 0.45 m, and at cruise 0.45 m is ~0.9 s of timing
error -- essentially the plant's own lag (tau ~1.15 s + ~0.14 s dead time)
going uncompensated. The mean signed lag is ~0, so the aircraft is not
biased behind; it oscillates around the schedule, which is what an
un-anticipated lag looks like. `accel_lead_s` is the term that projects the
reference acceleration forward to cancel exactly that.

Compares the two most recent controller revisions by `controller_rev`:
whichever revision carries the raised lead against its immediate predecessor,
passed in as REV_NEW / REV_OLD below. Completed flights only, >=3 per side.

Decision rule (medians across runs):
  PRIMARY  |along| p90 <= 0.80 x baseline   (the error this term targets)
  T1       pos_err p50 <= 1.05 x baseline   (total error must not worsen)
  S1       no run may log more than 5 tilt-cuts, and cross-track p90 must not
           exceed 1.25 x baseline (over-anticipation shows up as swerve)
  K1       coverage >= 0.9 x baseline       (must not cost exploration)

  ADOPT  = PRIMARY and T1 and S1 and K1
  REVERT = S1 fails, or PRIMARY fails and T1 fails
  MORE-RUNS otherwise

Usage:
  PYTHONPATH=. python3 -m sparx_agency.tools.falcon_campaign.test_v4_accel_lead
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "runs")
REV_OLD = os.environ.get("SPARX_REV_OLD", "v3.0")
REV_NEW = os.environ.get("SPARX_REV_NEW", "v4.0")
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


def _track(run_dir):
    along, cross, err = [], [], []
    try:
        with open(os.path.join(run_dir, "clearance.jsonl")) as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("along_m") is not None:
                    along.append(abs(d["along_m"]))
                    cross.append(abs(d["cross_m"]))
                if d.get("pos_err_m") is not None:
                    err.append(d["pos_err_m"])
    except OSError:
        return None
    if not along or not err:
        return None
    q = lambda xs, p: statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 2 else xs[0]
    return dict(along_p90=q(along, 90), cross_p90=q(cross, 90),
                err_p50=statistics.median(err))


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
    t = [_track(d) for d, _ in runs]
    return dict(
        along_p90=_med([x["along_p90"] for x in t if x]),
        cross_p90=_med([x["cross_p90"] for x in t if x]),
        err_p50=_med([x["err_p50"] for x in t if x]),
        coverage=_med([((s.get("metrics") or {}).get("coverage") or {}).get("final_m3")
                       for _, s in runs]),
    )


def main():
    old, new = _flights(REV_OLD), _flights(REV_NEW)
    print("%s: %d flights | %s: %d flights" % (REV_OLD, len(old), REV_NEW, len(new)))
    if len(old) < MIN_PER_REV or len(new) < MIN_PER_REV:
        print("VERDICT: NOT ENOUGH DATA (need %d per revision)" % MIN_PER_REV)
        return 2
    a, b = _stats(old), _stats(new)
    print(REV_OLD, json.dumps(a))
    print(REV_NEW, json.dumps(b))
    if None in a.values() or None in b.values():
        print("VERDICT: NOT ENOUGH DATA (a median metric is missing)")
        return 2
    tilts = [_tilts(d) for d, _ in new]
    primary = b["along_p90"] <= 0.80 * a["along_p90"]
    t1 = b["err_p50"] <= 1.05 * a["err_p50"]
    s1 = all(t <= 5 for t in tilts) and b["cross_p90"] <= 1.25 * a["cross_p90"]
    k1 = b["coverage"] >= 0.9 * a["coverage"]
    print("PRIMARY along_p90 <=0.80x : %s" % primary)
    print("T1 err_p50 <=1.05x        : %s" % t1)
    print("S1 safety/cross           : %s (tilt-cuts %s)" % (s1, tilts))
    print("K1 coverage >=0.9x        : %s" % k1)
    if primary and t1 and s1 and k1:
        print("VERDICT: ADOPT")
        return 0
    if (not s1) or ((not primary) and (not t1)):
        print("VERDICT: REVERT")
        return 1
    print("VERDICT: MORE-RUNS")
    return 2


if __name__ == "__main__":
    sys.exit(main())
