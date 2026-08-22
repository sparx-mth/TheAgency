"""Pre-registered test: does the TIME of the planner's death set the damage?

Written 2026-08-22 10:35, with n=2, deliberately before the data exists — for
the same reason ``test_p38.py`` and ``test_p41.py`` were. Two runs suggested it
and two runs cannot confirm it:

    093058Z  planner died  90 s into the flight ->   805 m3  (collapse)
    092027Z  planner died 400 s into the flight ->  1657 m3  (above median)

The mechanism is plausible, which is exactly why it needs testing rather than
believing: the map is fused from depth by ``mapping_sync`` and the planner only
chooses where to go, so an abort stops directed exploration while mapping
continues. An abort in the last minute should therefore cost almost nothing,
and one in the first minute should cost most of the flight.

THE HYPOTHESIS — one claim, tested on planner-death runs ONLY:

    corr(seconds_of_flight_before_death, final_volume_m3) >= +0.5,
    with n >= 8 such runs

Nothing else is scanned on this set. If it passes, the campaign gains a real
mechanism: the cost of a planner death is proportional to the flight it steals,
and the fix priority becomes restarting the planner fast rather than only
preventing the abort. If it fails, the damage is set by something else and the
apparent pattern in those two runs was luck.

NOT the hypothesis: that planner death predicts collapse at all. That is P42,
already measured over 266 runs (52 % vs 12 %, Fisher p=1.1e-06).

Requires ``metrics.json`` to carry ``planner_death`` (added 2026-08-22); older
runs are skipped rather than guessed at.
"""

import datetime
import glob
import json
import os

#: Runs started after this instant carry the planner_death block.
CUTOFF = datetime.datetime(2026, 8, 22, 10, 30)
#: Refuse to report below this many planner-death runs.
MIN_RUNS = 8
#: The pre-registered bar.
MIN_CORRELATION = 0.5
#: Bring-up takes about this long before the flight starts; the run directory
#: is named for the CYCLE start, not the takeoff. Only used as a fallback when
#: the supervisor log cannot be read.
TYPICAL_BRINGUP_S = 330.0
MAX_VEL = 0.8
RAYCAST_MAX = 8.0


def _flight_seconds_before_death(run_dir, metrics):
    """Seconds of flight elapsed when the planner died, or None."""
    death = (metrics.get("planner_death") or {}).get("wall")
    if not death:
        return None
    try:
        died = datetime.datetime.strptime(death, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    name = os.path.basename(run_dir)
    try:
        started = datetime.datetime.strptime(name[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    elapsed = (died - started).total_seconds() - TYPICAL_BRINGUP_S
    duration = (metrics.get("motion") or {}).get("duration_s") or 0.0
    if elapsed < 0 or (duration and elapsed > duration + 60):
        return None            # outside the flight: do not guess
    return elapsed


def _cases(runs_dir):
    """Planner-death runs after the cutoff, with a usable death time."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "2026*Z"))):
        metrics_path = os.path.join(path, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        try:
            started = datetime.datetime.strptime(
                os.path.basename(path)[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if started <= CUTOFF:
            continue
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        config = metrics.get("config") or {}
        coverage = metrics.get("coverage") or {}
        if config.get("max_vel") != MAX_VEL:
            continue
        if config.get("raycast_max") != RAYCAST_MAX:
            continue
        if not coverage.get("reliable"):
            continue
        if not (metrics.get("planner_death") or {}).get("died"):
            continue
        elapsed = _flight_seconds_before_death(path, metrics)
        if elapsed is None:
            continue
        out.append((os.path.basename(path), elapsed, coverage["final_m3"]))
    return out


def _pearson(xs, ys):
    """Pearson correlation, or 0.0 when a series has no spread."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else 0.0


def main(runs_dir):
    """Report the pre-registered result, or refuse if the sample is too small."""
    cases = _cases(runs_dir)
    print("planner-death runs since %s with a usable death time: %d (need %d)"
          % (CUTOFF, len(cases), MIN_RUNS))
    for name, elapsed, volume in cases:
        print("   %s  died %5.0f s into the flight  ->  %6.0f m3"
              % (name, elapsed, volume))
    if len(cases) < MIN_RUNS:
        print("NOT YET TESTABLE -- no coefficient is reported below n=%d, by design."
              % MIN_RUNS)
        return
    corr = _pearson([c[1] for c in cases], [c[2] for c in cases])
    print("corr(seconds before death, final volume) = %+.3f  (bar: >= %+.2f)"
          % (corr, MIN_CORRELATION))
    print("RESULT: %s" % ("PASSED -- the cost of a planner death is the flight it steals"
                          if corr >= MIN_CORRELATION else
                          "FAILED -- timing does not set the damage"))


if __name__ == "__main__":
    from sparx_agency.tools.falcon_campaign import config as C
    main(str(C.RUNS_DIR))
