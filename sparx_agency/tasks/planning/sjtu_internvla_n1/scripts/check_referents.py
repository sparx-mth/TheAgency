#!/usr/bin/env python3
"""Prove that every thing an instruction names is actually in the picture.

**Why this exists.** Nearly every failure of this deployment traces back to an
order naming something the camera could not see. Measured over six flights:
"turn until you can see an open doorway" drew STOP 42 times out of 47, because
the doorway was ninety degrees off the nose; "turn and look around you" drew
STOP 59 for 59. The camera is 600x600 at fx 390.64, which is 75.05 degrees --
so a landmark more than **37.5 degrees off the boresight is not in the frame**,
however plainly it is "on the right" on a map. Writing an instruction without
checking is guessing, and every guess costs a flight.

It answers three questions per referent, and all three have to pass:

1. Is the start pose somewhere the aircraft can actually hover? (clearance)
2. Is the referent inside the cone? (bearing, against half the real HFOV)
3. Is anything between the camera and it? (occlusion)

Two and three come from
:class:`~sparx_agency.core.planning.exploration.visibility_coverage.VisibilityCoverage`
-- the same class that scores coverage during a flight, run for a single
observation from the start pose. Using the flight's own visibility model rather
than a fresh ray-caster is the point: if the two disagreed, the disagreement
would be the bug.

Usage, from the repo root::

    .venv/bin/python sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/check_referents.py
    ... --task t2_turn_right          # just one
    ... --pose -5.0 -16.5 -90 --at "the doorway" -6.47 -23.08   # ad hoc

Exit status is 1 if any referent fails, so it can gate a campaign.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "..")))

from sparx_agency.core.planning.environment.occupancy_io import occupancy_from_mask
from sparx_agency.core.planning.exploration import (VisibilityCoverage,
                                                    cone_from_intrinsics)
from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import load_map_backdrop
from sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.area_clearance import ClearanceMap

#: The SJTU front camera, from robots/SJTU/config/vla/internvla_n1.yaml.
WIDTH, FX, RANGE_M = 600, 390.642735, 10.0
#: Half the field of view. A referent beyond this is not in the picture.
HALF_FOV_DEG = math.degrees(2.0 * math.atan(0.5 * WIDTH / FX)) / 2.0
#: The airframe is 0.63 m across, so this much clearance is a hover, not a scrape.
MIN_CLEARANCE_M = 0.45

MAPS = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..",
                                    "robots", "SJTU", "maps"))


class Referent(object):
    """One thing an instruction names, and where the aircraft must be to see it.

    ``seen_from`` is the whole difference between a short task and a medium
    one. In the known-good short order -- "there is a room to your right, enter
    it, go to the center, find the table and stop near the table" -- the door
    is visible from the start and the table is not visible until the aircraft
    is inside. That is fine, and it is the shape that works.

    What is NOT fine is naming something invisible *from where the aircraft is
    standing when it is told*. So every referent declares the pose it is
    supposed to become visible from, and the check is run from there. A
    referent with no ``seen_from`` is a first sub-goal and is checked from the
    start pose -- and those are the ones that must pass, because they are what
    the model is looking at when it decides whether it can do this at all.
    """

    def __init__(self, label, x, y, seen_from=None, required=True):
        # type: (str, float, float, tuple, bool) -> None
        self.label, self.x, self.y = label, float(x), float(y)
        self.seen_from = seen_from
        self.required = required


class Task(object):
    """A start pose and everything its instruction names."""

    def __init__(self, name, start, instruction, referents, note=""):
        # type: (str, tuple, str, list, str) -> None
        self.name, self.start = name, start
        self.instruction, self.referents, self.note = instruction, referents, note


def _visibility():
    backdrop = load_map_backdrop(os.path.join(MAPS, "hospital.yaml"))
    grid = occupancy_from_mask(backdrop.occupied_mask, backdrop.resolution,
                               backdrop.origin_x, backdrop.origin_y,
                               known=backdrop.known_mask)
    return VisibilityCoverage(grid, cone_from_intrinsics(WIDTH, FX, RANGE_M, 0.2))


def _bearing_deg(x, y, yaw_deg, target):
    # type: (float, float, float, Referent) -> float
    rel = math.atan2(target.y - y, target.x - x) - math.radians(yaw_deg)
    return math.degrees((rel + math.pi) % (2.0 * math.pi) - math.pi)


def check(task, coverage, clearance):
    # type: (Task, VisibilityCoverage, ClearanceMap) -> bool
    """Print a verdict for one task; True if every check passed."""
    x, y, yaw_deg = task.start
    print("\n%s" % task.name)
    print("  instruction: %s" % task.instruction)
    if task.note:
        print("  %s" % task.note)

    ok = True
    room = clearance.clearance(x, y)
    verdict = "ok" if room >= MIN_CLEARANCE_M else "TOO TIGHT"
    print("  start (%.2f, %.2f) yaw %+.0f deg -- clearance %.2f m  %s"
          % (x, y, yaw_deg, room, verdict))
    if room < MIN_CLEARANCE_M:
        ok = False

    for ref in task.referents:
        fx_, fy_, fyaw = ref.seen_from or task.start
        if ref.seen_from:
            room_here = clearance.clearance(fx_, fy_)
            if room_here < MIN_CLEARANCE_M:
                print("    %-34s  the pose it is judged from (%.2f, %.2f) has "
                      "only %.2f m clearance" % (ref.label, fx_, fy_, room_here))
                ok = False
                continue
        bearing = _bearing_deg(fx_, fy_, fyaw, ref)
        rng = math.hypot(ref.x - fx_, ref.y - fy_)

        coverage.restore_seen(np.zeros_like(coverage.seen_mask))
        coverage.observe(fx_, fy_, math.radians(fyaw))
        cell = coverage.cell_of(ref.x, ref.y)
        looked = bool(cell is not None and coverage.seen_mask[cell[1], cell[0]])

        if abs(bearing) > HALF_FOV_DEG:
            state, good = "OUT OF FRAME (camera sees +/-%.1f)" % HALF_FOV_DEG, False
        elif rng > RANGE_M:
            state, good = "BEYOND RANGE (camera reaches %.0f m)" % RANGE_M, False
        elif not looked:
            state, good = "OCCLUDED -- something is in the way", False
        else:
            state, good = "in frame", True
        if not good and not ref.required:
            state += "  [not required]"
        else:
            ok = ok and good
        where = "" if not ref.seen_from else "  from (%.1f,%.1f,%+.0f)" % ref.seen_from
        print("    %-34s %+6.1f deg  %5.1f m  %s%s"
              % (ref.label, bearing, rng, state, where))
    return ok


#: The ladder. Every coordinate is measured off hospital.npz and the world's
#: own model placements, never estimated.
#:
#: The rungs climb ONE axis: how far ahead of the aircraft the last sub-goal
#: sits. t0 is the order that already works, and each rung adds one thing the
#: model must hold in mind past the point where it can still see it. That is
#: the difference between "enter the room and stop by the table" and "explore
#: the hospital", and it is the thing worth measuring.
TASKS = [
    # Task 2: medium horizon, VLA alone, one fixed prompt. No direction word in
    # any of them -- see record_medium_horizon.sh for why that is the design.
    Task("m1_which_room",
         (-3.90, -24.98, 180.0),
         "There are two rooms in front of you. You will find a refrigerator in "
         "one of the rooms. Go into the room that has the refrigerator and "
         "stop inside it.",
         [Referent("the lounge doorway", -6.47, -24.98),
          Referent("the refrigerator", -7.98, -24.90)],
         note="pick one of two rooms by what is in it"),

    Task("m2_find_and_look",
         (-3.90, -24.98, 180.0),
         "There are two rooms in front of you. One of them has a refrigerator "
         "in it. Find the room with the refrigerator, go inside it, and look "
         "around the whole room.",
         [Referent("the lounge doorway", -6.47, -24.98),
          Referent("the refrigerator", -7.98, -24.90)],
         note="as m1, plus a survey once inside"),

    Task("m3_both_rooms",
         (-3.90, -24.98, 180.0),
         "There are two rooms in front of you. Go into the first room and look "
         "around it, then leave that room and go into the second room and look "
         "around it too.",
         [Referent("the lounge doorway", -6.47, -24.98),
          Referent("the ward doorway", -6.47, -23.08,
                   seen_from=(-4.60, -24.30, 150.0))],
         note="two rooms in sequence -- the shape a survey is made of"),

    Task("m4_down_the_hall",
         (-5.00, -16.50, -90.0),
         "Fly along the corridor in front of you and go through the wide "
         "doorway at the far end of it, then stop inside the room.",
         [Referent("the doorway at the end", -6.47, -23.08)],
         note="THE CONTROL. Identical to the rung that failed 0/3, with the "
              "words 'turn right' removed and nothing else changed"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", action="append", default=None,
                    help="check only these (repeatable)")
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                    help="an ad-hoc start pose")
    ap.add_argument("--at", nargs=3, action="append", metavar=("LABEL", "X", "Y"),
                    help="an ad-hoc referent (repeatable); needs --pose")
    args = ap.parse_args(argv)

    coverage = _visibility()
    clearance = ClearanceMap(os.path.join(MAPS, "hospital.yaml"))

    if args.pose:
        refs = [Referent(l, float(x), float(y)) for l, x, y in (args.at or [])]
        tasks = [Task("ad hoc", tuple(args.pose), "(none)", refs)]
    else:
        tasks = [t for t in TASKS if not args.task or t.name in args.task]
        if not tasks:
            ap.error("no task matched %s; have: %s"
                     % (args.task, ", ".join(t.name for t in TASKS)))

    print("camera: %d px, fx %.1f -> %.1f deg across, so +/-%.1f off the nose"
          % (WIDTH, FX, 2 * HALF_FOV_DEG, HALF_FOV_DEG))
    failures = [t.name for t in tasks if not check(t, coverage, clearance)]
    if failures:
        print("\nFAILED: %s" % ", ".join(failures))
        print("An instruction naming something the camera cannot see is the "
              "single most common cause of STOP in this deployment.")
        return 1
    print("\nall %d task(s) name only things the camera can see." % len(tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
