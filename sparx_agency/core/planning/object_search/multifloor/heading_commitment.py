"""One stair-steering heading held from the moment it is chosen until it is walked, per standing position.

:meth:`StairTerrain.steering_options` re-reads the support grid every action, and near a riser that grid depends
on the view: a tread-edge cell's height comes from whichever surface samples the current frame left in it, so the
cell one step ahead can read as floor (and the storey map's occupancy of the flight, an obstacle) while it is
faced, and as a tread while the camera looks 30 degrees off it. 00802/000008 (HM3D v1 val, 2026-11-02) stood on a
landing for its whole 30-action traversal budget turning between -103 and -133 degrees: facing either one, only
the other was safe. So the steering decides once per standing position:

* a chosen heading is held while the agent turns to it, whatever the views on the way say, and walked once faced
  if it is still safe;
* any heading faced while unsafe -- the held one or another -- is ruled out for this position;
* every new choice skips the headings ruled out here, so the agent never turns back to one;
* moving (or a new route end) starts afresh, and so does a full scan round with nothing left.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from sparx_agency.core.common.types.geometry import normalize_angle

#: XY motion (m) that makes a new standing position -- the stair traversal's own progress test.
MOVED_M = 0.04
#: A route end moved this far (m) is a new route: its headings are decided anew.
GOAL_MOVED_M = 0.30


class HeadingCommitment:
    """The held heading and the headings ruled out where the agent stands.

    Headings are compared by angle, within half a turn, never by rounding to a lattice index: the start yaw is
    arbitrary, and a yaw half a turn off the index grid would round to either neighbour as float error drifts.

    Args:
        turn_rad: The action lattice's turn angle.
    """

    def __init__(self, turn_rad: float):
        if not turn_rad > 0.0:
            raise ValueError("turn_rad must be positive, got %r" % turn_rad)
        self.turn_rad = float(turn_rad)
        self.anchor: Optional[Tuple[float, float]] = None
        self.goal: Optional[Tuple[float, float]] = None
        self.held = None                        # the option turned to or walked
        self.ruled_out: List[float] = []

    def reset(self):
        """Forget the held heading and every heading ruled out."""
        self.held, self.ruled_out = None, []

    def same(self, a: float, b: float) -> bool:
        return abs(normalize_angle(float(a) - float(b))) < 0.5 * self.turn_rad

    def chooser(self, goal):
        """A ``choose`` for :meth:`StairTerrain.steering_path` toward the route end ``goal``."""
        goal = (float(goal[0]), float(goal[1]))
        if self.goal is not None and math.dist(goal, self.goal) > GOAL_MOVED_M:
            self.reset()
        self.goal = goal
        return self.choose

    def choose(self, pose, options):
        """The held option, a new one, or None when every safe heading here is ruled out."""
        here = (float(pose.x), float(pose.y))
        if self.anchor is None or math.dist(here, self.anchor) > MOVED_M:
            self.reset()
            self.anchor = here
        faced = [o for o in options if self.same(o[2], pose.yaw)]
        if not faced and not any(self.same(pose.yaw, r) for r in self.ruled_out):
            self.ruled_out.append(float(pose.yaw))
        if self.held is not None:
            if not self.same(self.held[2], pose.yaw):
                return self.held                # still turning to it: the views on the way do not re-decide
            if faced:
                self.held = faced[0]
                return self.held
            self.held = None
        free = [o for o in options if not any(self.same(o[2], r) for r in self.ruled_out)]
        if not free:
            return None
        self.held = max(free, key=lambda o: o[:2])
        return self.held

    def diagnostics(self):
        return {"held_deg": None if self.held is None else round(math.degrees(normalize_angle(self.held[2]))),
                "ruled_out": len(self.ruled_out)}
