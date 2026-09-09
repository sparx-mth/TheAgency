"""Alternate a follower parameter during a flight, for within-flight paired runs.

Every outcome this campaign cares about is unaffordable to measure between
flights: voxels per metre needs 43 flights per arm for a 15 % effect, coverage
48, collision exposure 134, and ``locked_s`` — the metric that best matches the
documented failure mode — needs **55 per arm for a 70 % reduction** because it is
dominated by rare severe events (LOOP_BUGS.md B39, and the F33 closure).

Nearly all of that variance is *between* flights: battery state, where the tour
starts, which cells the aircraft meets, whether it gets wedged. Alternating the
arm inside a single flight makes each flight its own matched pair, so that
variance cancels instead of being averaged over.

This is the driver half. The follower half already exists: ``~yaw_mode_poll_s``
makes the node re-read its mode on a timer (0 = disabled = shipped behaviour).
Only parameters the Python follower reads can be alternated this way — FALCON's
C++ reads its own in constructors, so those need a patch to re-read (the pattern
`sparx_opt_result.sh` uses for the optimiser budget).

**The confound this must respect.** A flight is not stationary in time: early
segments explore virgin space and late ones revisit, so voxels-per-metre falls
through a flight regardless of arm (the measured 4.2x discovery decay, B10).
Segments must therefore be short enough that *neighbours* face comparable maps,
and the analysis must compare **adjacent** segments rather than pooling all A
against all B. Pooling would manufacture an effect out of the decay curve alone.

Usage (inside the falcon container's ROS environment)::

    python3 -m sparx_agency.tools.falcon_campaign.paired_flipper \\
        --param ~yaw_mode --values blend course --period 60 --duration 430
"""
from __future__ import annotations

import argparse
import subprocess
import time

import sparx_agency.tools.falcon_campaign.config as C

#: Node whose private parameters are flipped.
NODE = "/falcon_exploration_follower"


def set_param(name, value):
    """Set one rosparam inside the falcon container.

    Args:
        name: Parameter name; ``~x`` is resolved against :data:`NODE`.
        value: Value to set, as a string.

    Returns:
        True when the call returned zero.
    """
    full = name.replace("~", NODE + "/", 1) if name.startswith("~") else name
    cmd = ("docker exec %s bash -lc %s"
           % (C.FALCON_CONTAINER,
              "'" + C.FALCON_ENV + "rosparam set %s %s'" % (full, value)))
    return subprocess.call(cmd, shell=True) == 0


def run(param, values, period_s, duration_s, log=print):
    """Alternate ``param`` through ``values`` until ``duration_s`` elapses.

    Args:
        param: Parameter to flip.
        values: Values to cycle through, in order.
        period_s: Seconds each value is held.
        duration_s: Total run length.
        log: Where to report each switch; the timestamps are what the offline
            analysis segments on, so they must be recorded.

    Returns:
        The list of ``(wall_time, value)`` switches actually applied.
    """
    switches = []
    start = time.time()
    i = 0
    while time.time() - start < duration_s:
        value = values[i % len(values)]
        if set_param(param, value):
            switches.append((time.time(), value))
            log("[paired] %s := %s at t=%.1f s" % (param, value, time.time() - start))
        else:
            log("[paired] FAILED to set %s := %s -- segment is void" % (param, value))
        i += 1
        # Sleep the remainder of this period, not a fixed interval, so a slow
        # rosparam call does not drift the segment boundaries.
        target = start + i * period_s
        while time.time() < target and time.time() - start < duration_s:
            time.sleep(0.5)
    return switches


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--param", default="~yaw_mode")
    ap.add_argument("--values", nargs="+", default=["blend", "course"])
    ap.add_argument("--period", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=430.0)
    args = ap.parse_args(argv)
    sw = run(args.param, args.values, args.period, args.duration)
    print("[paired] %d switches applied" % len(sw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
