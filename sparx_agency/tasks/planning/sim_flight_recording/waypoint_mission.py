"""Sequence a list of world-frame waypoints into offboard setpoints.

Deliberately dumb: PX4's own position controller does the flying, so all this
owns is *which* setpoint to stream right now and when to move on. Progress is
measured in **simulation** time, never wall-clock time -- PX4 SITL runs in
lockstep with the simulator, so a run that renders slowly still takes exactly
as many simulated seconds as one that does not.

The one piece of judgement here is that **intermediate waypoints are passed
through, not arrived at**. A position setpoint makes PX4 decelerate onto it, so
requiring the aircraft to settle inside a tight radius at every waypoint of a
route produces a stop-and-go flight -- jerky video, and a demonstration of
something no pilot would do. Handing over the next setpoint while the aircraft
is still a little short of the current one keeps it moving through corners. Only
the final waypoint, where it actually has to be before it lands, is held to a
tight radius and a dwell.
"""
from __future__ import annotations

import math

ARRIVAL_RADIUS_M = 0.8      # pass-through radius for intermediate waypoints
FINAL_RADIUS_M = 0.8        # the goal has to actually be reached
DWELL_S = 0.5               # how long to hold inside FINAL_RADIUS_M before finishing
TIMEOUT_S = 30.0            # give up on a waypoint PX4 cannot reach and move to the next
# FINAL_RADIUS_M is the airframe's demonstrated position-hold precision, not an
# aspiration. At 0.35 m -- which looks like the obviously right number -- three
# consecutive flights tracked their whole route, arrived, settled 0.66 m from
# the goal and were then judged failures for never closing the last 30 cm. The
# aircraft holds position to roughly two thirds of a metre and no tightening of
# the acceptance test changes that. Where it actually landed is recorded per
# frame either way, so a consumer that needs better can measure it.


class WaypointMission:
    """Walks a vehicle through ``waypoints`` using position setpoints.

    Args:
        waypoints: World-frame (ENU) ``(x, y, z, yaw)`` tuples. ``yaw`` is in
            radians CCW from +X, matching the repo-wide FLU convention.
        arrival_radius_m: Pass-through radius for every waypoint but the last.
        final_radius_m: Arrival radius for the last waypoint.
        dwell_s: How long the aircraft must stay inside ``final_radius_m``.
        timeout_s: Per-waypoint budget. A waypoint PX4 cannot reach is skipped
            rather than blocking the mission forever -- the aircraft may simply
            have settled just outside the radius.

    Raises:
        ValueError: If ``waypoints`` is empty.
    """

    def __init__(self, waypoints, arrival_radius_m: float = ARRIVAL_RADIUS_M,
                 final_radius_m: float = FINAL_RADIUS_M, dwell_s: float = DWELL_S,
                 timeout_s: float = TIMEOUT_S):
        if not waypoints:
            raise ValueError("a mission needs at least one waypoint")
        self._waypoints = [tuple(w) for w in waypoints]
        self._arrival_radius_m = arrival_radius_m
        self._final_radius_m = final_radius_m
        self._dwell_s = dwell_s
        self._timeout_s = timeout_s
        self._index = 0
        self._inside_since = None
        self.skipped = 0
        # Set on the first update(), not here: a mission is built well before the
        # vehicle is armed, and starting the per-waypoint timeout at construction
        # burns the first waypoint's whole budget during PX4's boot and arming.
        self._entered_at = None

    def __len__(self) -> int:
        return len(self._waypoints)

    @property
    def finished(self) -> bool:
        """True once the last waypoint has been reached (or timed out)."""
        return self._index >= len(self._waypoints)

    @property
    def index(self) -> int:
        """Index of the waypoint currently being flown to."""
        return self._index

    @property
    def on_final(self) -> bool:
        """True while flying the last leg."""
        return self._index == len(self._waypoints) - 1

    def current(self):
        """The waypoint to stream as a setpoint right now, or None if finished."""
        if self.finished:
            return None
        return self._waypoints[self._index]

    def update(self, position, sim_time: float) -> bool:
        """Advance the mission if the current waypoint has been reached.

        Args:
            position: The vehicle's current world-frame ``(x, y, z)``.
            sim_time: Elapsed simulation time, seconds.

        Returns:
            True if this call advanced to a new waypoint.
        """
        if self.finished:
            return False
        if self._entered_at is None:
            self._entered_at = sim_time

        target = self._waypoints[self._index]
        distance = math.dist(tuple(position[:3]), target[:3])
        radius = self._final_radius_m if self.on_final else self._arrival_radius_m

        if distance <= radius:
            if self._inside_since is None:
                self._inside_since = sim_time
            dwell = self._dwell_s if self.on_final else 0.0
            reached = (sim_time - self._inside_since) >= dwell
        else:
            self._inside_since = None
            reached = False

        if not reached and (sim_time - self._entered_at) < self._timeout_s:
            return False

        if not reached:
            self.skipped += 1
            print(f"    waypoint {self._index} timed out at {distance:.2f} m -- skipping",
                  flush=True)
        self._index += 1
        self._inside_since = None
        self._entered_at = sim_time
        return True
