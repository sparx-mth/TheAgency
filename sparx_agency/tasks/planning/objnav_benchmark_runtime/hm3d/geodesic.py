"""Privileged HM3D distance-to-goal, measured on the episode's own navmesh.

habitat-lab is not installed on this workstation, so this module reproduces
``habitat.tasks.nav.nav.DistanceToGoal`` with ``DISTANCE_TO: VIEW_POINTS``
directly against habitat-sim's pathfinder. What that measure does, exactly:

* it flattens **every view point of every goal** of the episode into one list
  of positions (``goal.view_points[k].agent_state.position``) -- not the
  object centres, and not one goal at a time;
* it asks the simulator for ``geodesic_distance(current, view_points)``, which
  is a single ``habitat_sim.MultiGoalShortestPath`` whose ``requested_ends``
  are those positions;
* it reuses the same path object for the whole episode
  (``episode._shortest_path_cache``), overwriting only ``requested_start``;
* and it recomputes **only when the agent has moved**
  (``np.allclose(previous, current, atol=1e-4)``), so a turn or a tilt reuses
  the previous value. That is not just an optimisation to copy for speed: it is
  what the reference measures, and a goal set here carries over a thousand view
  points, so querying it on every one of 500 actions would dominate the run.

Success is then ``is_stop_called and distance < SUCCESS_DISTANCE`` (strictly
less), and SPL's ``l`` is this same measure evaluated at reset. So one number,
measured one way, drives success, SPL, SoftSPL and DTG -- which is why it lives
in one small class here instead of being recomputed in three places.

Nothing in this module is ever shown to a policy.

Python 3.8 syntax; habitat-sim is imported lazily, inside the call.
"""
from __future__ import annotations

import math

import numpy as np


def view_point_positions(goals):
    """Every view point of every goal, in the dataset's order, as an Nx3 array.

    Args:
        goals: The decoded ``ObjectGoal`` rows of one (scene, category), each
            holding a ``view_points`` list of ``{"agent_state": {"position":
            [x, y, z]}}`` mappings, in Habitat's own +Y-up frame.

    Returns:
        ``numpy.ndarray`` of shape ``(N, 3)``, dtype float32.

    Raises:
        ValueError: No goal has a view point, or a position is not three
            finite numbers. An episode whose goals carry no view point cannot
            be scored by this benchmark's own definition, and silently falling
            back to the object centre would quietly change the success radius.
    """
    positions = []
    for goal in goals:
        for view in goal.get("view_points") or ():
            state = view.get("agent_state") if isinstance(view, dict) else None
            position = (state or {}).get("position")
            array = np.asarray(position, dtype=np.float64)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError("A goal view point is not three finite numbers: %r"
                                 % (position,))
            positions.append(array)
    if not positions:
        raise ValueError("No goal view points; habitat-lab's ObjectNav measures "
                         "distance to view points and cannot score this episode")
    return np.asarray(positions, dtype=np.float32)


class ViewPointDistance:
    """Geodesic distance from a Habitat position to the nearest goal view point.

    Args:
        pathfinder: The ``habitat_sim`` pathfinder of the loaded scene. It must
            be the **same** pathfinder the agent navigates on -- whichever
            navmesh the protocol chose. Measuring on one mesh while moving on
            another changes ``l`` without changing anything observable, and so
            changes every SPL silently.
        view_points: What :func:`view_point_positions` returned.

    The instance is bound to one (scene, category): its ``requested_ends``
    never change, exactly as habitat-lab's per-episode cache never changes them.
    Because the cached value is keyed on the position it was measured from, the
    instance is safely shared by every episode with that goal set.
    """

    #: Positions this close together are the same position, as habitat-lab's
    #: ``DistanceToGoal.update_metric`` decides it.
    SAME_POSITION_M = 1e-4

    def __init__(self, pathfinder, view_points):
        points = np.asarray(view_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("View points must be a non-empty Nx3 array")
        if not np.isfinite(points).all():
            raise ValueError("View points must all be finite")
        self._pathfinder = pathfinder
        self._points = points
        self._path = None
        self._previous = None
        self._value = None
        self.queries = 0

    @property
    def view_point_count(self) -> int:
        """How many view points this distance is measured to."""
        return int(len(self._points))

    def distance(self, position) -> float:
        """Metres along the navmesh to the nearest view point; ``inf`` if none.

        Args:
            position: A Habitat-frame ``(x, y, z)`` position, +Y up -- the raw
                agent position, not our ENU pose.

        Returns:
            The geodesic distance in metres, or ``float("inf")`` when no view
            point is reachable from ``position`` (a different navmesh island).

        Raises:
            ValueError: ``position`` is not three finite numbers.
        """
        start = np.asarray(position, dtype=np.float64)
        if start.shape != (3,) or not np.isfinite(start).all():
            raise ValueError("Position must be three finite numbers: %r" % (position,))
        if self._previous is not None and np.allclose(self._previous, start,
                                                      atol=self.SAME_POSITION_M):
            return self._value
        import habitat_sim

        if self._path is None:
            self._path = habitat_sim.MultiGoalShortestPath()
            self._path.requested_ends = self._points
        self._path.requested_start = start.astype(np.float32)
        self._pathfinder.find_path(self._path)
        self.queries += 1
        value = float(self._path.geodesic_distance)
        self._previous = start
        self._value = value if math.isfinite(value) else float("inf")
        return self._value
