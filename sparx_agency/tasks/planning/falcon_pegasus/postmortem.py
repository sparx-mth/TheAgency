#!/usr/bin/env python3
"""Work out what actually ended a flight, from the recording it left behind.

    .venv/bin/python sparx_agency/tasks/planning/falcon_pegasus/postmortem.py <dir>

``<dir>`` is a soak attempt directory (or anything holding ``result.json`` and
``recording/poses.npy``).

**The outcome field is not the diagnosis.** Six soak rounds in, four of them
were something other than what they reported: two `stalled` verdicts that were a
mapper abort and a wall strike, one that was Isaac Sim running out of VRAM
before the aircraft existed, and one `crashed` that was the outer loop
limit-cycling. Every one of those cost an hour of reading logs to establish, and
every one of them is visible in the recording in seconds if you know which four
questions to ask:

* **Did it touch something?** A velocity that reverses within a couple of ticks
  is not a control response -- no loop on this aircraft can turn 1.4 m/s around
  in 200 ms. It is a contact, and it is the single most useful thing to know,
  because a contact makes every downstream number (tilt, tracking error,
  "stalled") a *consequence* rather than a cause.
* **Where was it, and what is there?** Answered against the surveyed voxel map
  rather than against FALCON's, so the answer does not depend on the mapper
  being right. Index those arrays ``v[k, j, i]`` -- they are ``(nz, ny, nx)``,
  and getting that backwards once produced a confident, wrong "it was in free
  space".
* **Was it upset, or was it parked?** Tilt past 60 degrees with the aircraft
  pinned at one spot near the floor is post-crash debris, not the failure.
* **Was it looking where it flew?** Interesting, and usually *not* a fault --
  FALCON aims the camera at frontiers, not along travel. Reported so the number
  stops being rediscovered as if it were a bug.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from sparx_agency.core.control.constants import GRAVITY_MPS2

# A contact, expressed as the thing that makes it unmistakable: the aircraft's
# horizontal velocity reverses through more than this angle inside REVERSAL_S
# while it was moving at least REVERSAL_SPEED. Cornering cannot do this -- at
# 0.7 g the fastest a 1 m/s turn can swing its course is about 40 deg in 0.1 s.
REVERSAL_DEG = 120.0
REVERSAL_S = 0.25
REVERSAL_SPEED = 0.6

AIRBORNE_Z_M = 0.4
"""Below this the aircraft is on the ground and its "contacts" are the floor.

Not pedantry: take-off registers a textbook reversal -- the aircraft settles,
touches, and bounces -- and it was being reported as the flight's first contact,
which then truncated the whole pre-contact analysis to nothing.
"""

ARREST_MPS2 = 8.0
"""Deceleration above which nothing but a collision can be responsible.

The tilt ceiling is 35 degrees, so the most the controller can ask for is
g*tan(35) = 6.9 m/s^2, and it cannot even ask instantly because the thrust axis
lags. Anything past 8 m/s^2 was applied by the building.

This is the case the reversal test misses. A glancing blow spins the velocity
round and shows up as a course reversal; a square-on strike simply STOPS the
aircraft, leaving no course to reverse -- and with the speed through the floor,
the reversal test rejects the sample for being too slow.
"""

UPSET_TILT_DEG = 45.0
PARKED_DISPLACEMENT_M = 0.5
"""How far the aircraft must travel in the last window to count as still flying.

NET displacement, not mean speed. An aircraft lying on its side against a wall
still registers a fifth of a metre per second of scraping and bouncing -- enough
to clear any speed threshold loose enough to be useful -- while going nowhere at
all. Measured on a real crash: mean speed 0.18 m/s, net displacement 6 cm.
"""


def _columns(meta):
    """Column index by name, from the recording's own schema."""
    names = meta.get("pose_columns")
    if not names:
        raise SystemExit(
            "this recording predates the named pose schema; it cannot be read "
            "positionally without guessing which column is z")
    return {name: i for i, name in enumerate(names)}


def _tilt_deg(poses, col):
    """Angle between the thrust axis and vertical, from the logged quaternion."""
    q = np.stack([poses[:, col[n]] for n in ("qx", "qy", "qz", "qw")], axis=1)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    body_z_up = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    return np.degrees(np.arccos(np.clip(body_z_up, -1.0, 1.0)))


def _find_contacts(t, velocity, z):
    """Where the aircraft was stopped or turned by something other than control.

    Two signatures, because a collision has two shapes. See REVERSAL_DEG and
    ARREST_MPS2. Both are gated on being airborne, or take-off is reported as
    the first contact of the flight.
    """
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    course = np.arctan2(velocity[:, 1], velocity[:, 0])
    found = []
    for i in range(1, len(t)):
        if z[i] < AIRBORNE_Z_M:
            continue
        j = i - 1
        while j > 0 and t[i] - t[j] < REVERSAL_S:
            j -= 1
        dt = t[i] - t[j]
        if dt <= 0.0:
            continue
        # (a) the velocity turned round faster than flight allows
        if speed[i] >= REVERSAL_SPEED and speed[j] >= REVERSAL_SPEED:
            swing = abs(math.degrees(math.atan2(math.sin(course[i] - course[j]),
                                                math.cos(course[i] - course[j]))))
            if swing >= REVERSAL_DEG:
                found.append((i, swing, speed[j], speed[i]))
                continue
        # (b) it was simply stopped, harder than the tilt ceiling permits
        if (speed[j] - speed[i]) / dt >= ARREST_MPS2:
            found.append((i, 0.0, speed[j], speed[i]))
    collapsed = []
    for entry in found:
        if not collapsed or t[entry[0]] - t[collapsed[-1][0]] > 1.0:
            collapsed.append(entry)
    return collapsed


class _Survey:
    """The surveyed ground truth, indexed the way the arrays are actually laid out."""

    def __init__(self, scene):
        from sparx_agency.robots.PEGASUS.adapters import scene_map

        path = Path(scene_map.MAP_DIR) / ("%s_voxels.npz" % scene)
        if not path.exists():
            raise SystemExit("no surveyed map at %s" % path)
        data = np.load(path)
        self.voxels = data["voxels"]            # (nz, ny, nx) -- v[k, j, i]
        self.origin = data["origin"]
        self.resolution = float(data["resolution"])

    def _index(self, x, y, z):
        return (int(round((z - self.origin[2]) / self.resolution)),
                int(round((y - self.origin[1]) / self.resolution)),
                int(round((x - self.origin[0]) / self.resolution)))

    def occupied(self, x, y, z):
        """True if the surveyed map says that point is inside something."""
        k, j, i = self._index(x, y, z)
        shape = self.voxels.shape
        if not (0 <= k < shape[0] and 0 <= j < shape[1] and 0 <= i < shape[2]):
            return False
        return bool(self.voxels[k, j, i] > 0)

    def nearby(self, x, y, z, radius_m=0.6):
        """How many occupied cells sit within a box of that half-width."""
        k, j, i = self._index(x, y, z)
        cells = int(round(radius_m / self.resolution))
        block = self.voxels[max(0, k - cells):k + cells + 1,
                            max(0, j - cells):j + cells + 1,
                            max(0, i - cells):i + cells + 1]
        return int((block > 0).sum())


def _report_contacts(survey, t, x, y, z, contacts):
    print("CONTACTS  (horizontal velocity reversing faster than flight allows)")
    if not contacts:
        print("  none -- whatever ended this flight, it was not a strike")
        return
    for i, swing, before, after in contacts:
        where = "occupied" if survey.occupied(x[i], y[i], z[i]) else "clear"
        how = ("course swung %5.1f deg" % swing) if swing else "STOPPED DEAD    "
        print("  t=%7.1fs  (%6.2f,%6.2f,%5.2f)  %s, "
              "%.2f -> %.2f m/s | that point reads %s, %d occupied cells within 0.6 m"
              % (t[i], x[i], y[i], z[i], how, before, after, where,
                 survey.nearby(x[i], y[i], z[i])))
    print("  A contact makes the tilt and tracking error that follow it a "
          "CONSEQUENCE. Diagnose the approach, not the upset.")


def _report_ending(t, x, y, z, tilt, speed):
    print()
    print("HOW IT ENDED")
    last = t[-1]
    tail = t >= last - 10.0
    first = int(np.argmax(tail))
    displacement = float(math.dist((x[first], y[first], z[first]),
                                   (x[-1], y[-1], z[-1])))
    parked = displacement < PARKED_DISPLACEMENT_M
    upset = bool(tilt[tail].mean() > UPSET_TILT_DEG)
    print("  last 10 s: net displacement %.2f m (mean speed %.2f m/s), mean tilt "
          "%.1f deg, z %.2f-%.2f m"
          % (displacement, speed[tail].mean(), tilt[tail].mean(),
             z[tail].min(), z[tail].max()))
    if parked and upset:
        print("  -> ON THE GROUND, ON ITS SIDE. This is debris. The flight ended "
              "earlier; look at the first contact above.")
    elif parked:
        print("  -> STATIONARY and upright: wedged, holding, or the planner stopped.")
    elif upset:
        print("  -> UPSET while still moving: an attitude divergence in flight.")
    else:
        print("  -> still flying normally at the last sample.")


def _report_looking(t, yaw, velocity, hfov_deg=90.0):
    print()
    print("WAS IT LOOKING WHERE IT FLEW  (context, rarely a fault)")
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    moving = speed > 0.4
    if not moving.any():
        print("  never moved")
        return
    course = np.arctan2(velocity[moving, 1], velocity[moving, 0])
    off = np.abs(np.degrees(np.arctan2(np.sin(course - yaw[moving]),
                                       np.cos(course - yaw[moving]))))
    half = hfov_deg / 2.0
    print("  |travel - camera| p50 %.1f deg, p90 %.1f deg; %.0f%% outside the "
          "%.0f deg FOV" % (np.percentile(off, 50), np.percentile(off, 90),
                            100.0 * (off > half).mean(), hfov_deg))
    print("  FALCON aims the camera at frontiers, not along travel, and plans "
          "against its accumulated map. A large number here is normal.")


def _report_trace(directory, contacts_t, pose_origin_s):
    """What the plan asked for against what the aircraft did, if it was recorded.

    The split between along-track and cross-track is the whole value here.
    ``err`` on its own conflates two completely different mistakes: being late,
    which is benign and mostly an artefact of a simulator running slower than
    the clock FALCON plans on, and being sideways, which is what hits things.
    A round was misread once by treating a large ``err`` as a tracking failure
    when nearly all of it was schedule.

    Speed is reported against the plan's OWN speed rather than an absolute
    number, because that is what FALCON's clearance is computed for.
    """
    path = directory / "trace.npy"
    if not path.exists():
        print()
        print("No trace.npy -- this flight predates per-tick tracing.")
        return
    trace = np.load(path)
    columns = json.loads((directory / "trace_columns.json").read_text())
    col = {name: i for i, name in enumerate(columns)}
    take = lambda name: trace[:, col[name]]

    # BOTH clocks are raw sim_time, but the pose axis was rebased to its own
    # first sample, so the contact times handed in here are relative and the
    # trace's are absolute. They differ by the whole pre-flight period -- boot,
    # settle and PX4's arming, ~50 s on every real recording -- and comparing
    # them silently selects the wrong rows, or none, and prints a confidently
    # wrong pre-contact summary. Rebase onto the same origin.
    trace_t = take("t") - pose_origin_s
    # Hold ticks carry fabricated zero lag and cross-track -- _hold_station has
    # no curve to measure against -- so they are excluded rather than allowed to
    # flatter the averages.
    if "holding" in col:
        flying = take("holding") == 0.0
        held = int((~flying).sum())
        if held:
            print()
            print("  (%d of %d ticks were deliberate holds and are excluded below)"
                  % (held, len(trace)))
        # `take` closes over `trace` by name, so rebinding it here is enough --
        # every read below sees the filtered array.
        trace = trace[flying]
        trace_t = trace_t[flying]

    speed = np.hypot(take("vx"), take("vy"))
    planned = np.hypot(take("ref_vx"), take("ref_vy"))
    ceiling = planned + 0.25            # params.max_overspeed
    print()
    print("PLAN VERSUS AIRCRAFT  (%d ticks)" % len(trace))
    if "rtf" in col:
        rtf = take("rtf")
        print("  simulator ran at %.2fx real time (min %.2f) -- FALCON is on "
              "/clock, so this no longer becomes lag" % (rtf.mean(), rtf.min()))
    print("  planned speed  mean %.2f  p90 %.2f m/s" % (planned.mean(),
                                                        np.percentile(planned, 90)))
    print("  actual  speed  mean %.2f  p90 %.2f  max %.2f m/s"
          % (speed.mean(), np.percentile(speed, 90), speed.max()))
    # Only where the plan HAS a speed. `BsplineTrajectory.sample` zeroes every
    # derivative once the reference runs past the end of its curve, and with
    # FALCON replanning several times a second that is a large minority of
    # ticks. Counting them makes the relative ceiling 0.25 m/s and reports a
    # perfectly well-behaved aircraft as 55% over it -- which it is not.
    live = planned > 0.0
    if live.any():
        print("  over the governor's own ceiling: %.1f%% of the ticks where the "
              "plan has a speed (%.0f%% of ticks do)"
              % (100.0 * (speed[live] > ceiling[live]).mean(), 100.0 * live.mean()))
    else:
        print("  the reference was past the end of its curve for the whole flight")
    print("  along-track lag  mean %+.2f  p90 %.2f m   (benign)"
          % (take("lag_m").mean(), np.percentile(take("lag_m"), 90)))
    print("  cross-track      mean %+.2f  p90 %.2f  max %.2f m   (this is the one)"
          % (take("xte_m").mean(), np.percentile(take("xte_m"), 90),
             take("xte_m").max()))

    if contacts_t:
        first = contacts_t[0]
        before = trace_t < first
        if not before.any():
            print("  (no trace ticks before the first contact -- the aircraft hit "
                  "something before the tracker was engaged)")
        else:
            print("  BEFORE the first contact at t=%.1fs: speed mean %.2f, "
                  "xte mean %.2f, lag mean %+.2f"
                  % (first, speed[before].mean(), take("xte_m")[before].mean(),
                     take("lag_m")[before].mean()))
            print("  -- judge the controller on those numbers, not on the whole "
                  "flight: after a contact the aircraft is pinned and the "
                  "reference walks away from it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="a soak attempt directory")
    args = parser.parse_args()

    recording = args.directory / "recording"
    poses = np.load(recording / "poses.npy")
    meta = json.loads((recording / "meta.json").read_text())
    col = _columns(meta)

    result_path = args.directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text()).get("result", {})
        print("REPORTED  %s -- %s" % (result.get("outcome"), result.get("detail", "")))
        print("          %.0f m flown, mean tracking error %.2f m, max %.2f m"
              % (result.get("distance_m", 0.0),
                 result.get("mean_tracking_error_m", 0.0),
                 result.get("max_tracking_error_m", 0.0)))
        print()

    t = poses[:, col["t"]] - poses[0, col["t"]]
    x, y, z = poses[:, col["x"]], poses[:, col["y"]], poses[:, col["z"]]
    yaw = poses[:, col["yaw"]]
    velocity = np.stack([poses[:, col[n]] for n in ("vx", "vy", "vz")], axis=1)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    tilt = _tilt_deg(poses, col)
    survey = _Survey(str(meta.get("scene", "office")))

    print("%.1f s of flight, %d samples" % (t[-1], len(t)))
    print()
    contacts = _find_contacts(t, velocity, z)
    _report_contacts(survey, t, x, y, z, contacts)
    _report_ending(t, x, y, z, tilt, speed)
    _report_trace(args.directory, [t[i] for i, _, _, _ in contacts],
                  float(poses[0, col["t"]]))
    _report_looking(t, yaw, velocity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
