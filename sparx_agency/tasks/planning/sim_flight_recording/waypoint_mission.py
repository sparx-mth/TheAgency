"""Sequence a list of world-frame waypoints into offboard setpoints.

Deliberately dumb: PX4's own position controller does the flying, so all this
owns is *which* setpoint to stream right now and when to move on. Progress is
measured in **simulation** time, never wall-clock time -- PX4 SITL runs in
lockstep with the simulator, so a run that renders slowly still takes exactly
as many simulated seconds as one that does not.
"""
from __future__ import annotations

import math

ARRIVAL_RADIUS_M = 0.35
DWELL_S = 0.5      # how long to stay inside ARRIVAL_RADIUS_M before advancing
TIMEOUT_S = 30.0   # give up on a waypoint PX4 cannot reach and move to the next


class WaypointMission:
    """Walks a vehicle through ``waypoints`` using position setpoints.

    Args:
        waypoints: World-frame (ENU) ``(x, y, z, yaw)`` tuples. ``yaw`` is in
            radians CCW from +X, matching the repo-wide FLU convention.
    """

    def __init__(self, waypoints):
        if not waypoints:
            raise ValueError("a mission needs at least one waypoint")
        self._waypoints = [tuple(w) for w in waypoints]
        self._index = 0
        self._inside_since = None
        # Set on the first update(), not here: a mission is built well before the
        # vehicle is armed, and starting the per-waypoint timeout at construction
        # burns the first waypoint's whole budget during PX4's boot and arming.
        self._entered_at = None

    @property
    def finished(self) -> bool:
        """True once the last waypoint has been reached (or timed out)."""
        return self._index >= len(self._waypoints)

    @property
    def index(self) -> int:
        """Index of the waypoint currently being flown to."""
        return self._index

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
        distance = math.dist(position[:3], target[:3])

        if distance <= ARRIVAL_RADIUS_M:
            if self._inside_since is None:
                self._inside_since = sim_time
            reached = (sim_time - self._inside_since) >= DWELL_S
        else:
            self._inside_since = None
            reached = False

        if not reached and (sim_time - self._entered_at) < TIMEOUT_S:
            return False

        if not reached:
            print(f"waypoint {self._index} timed out at {distance:.2f} m -- skipping", flush=True)
        self._index += 1
        self._inside_since = None
        self._entered_at = sim_time
        return True
