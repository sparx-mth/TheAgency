"""Evaluator-only geodesic distance to the published goal view points.

Independent implementation of habitat-lab's ObjectNav distance, which is what
every MP3D number in the comparison papers is measured with:

* ``habitat/tasks/nav/nav.py::DistanceToGoal`` with ``distance_to:
  VIEW_POINTS`` calls ``sim.geodesic_distance(current_position,
  [view_point.agent_state.position for goal in episode.goals for view_point in
  goal.view_points])``;
* ``habitat/sims/habitat_simulator/habitat_simulator.py::HabitatSim.geodesic_distance``
  fills one ``habitat_sim.MultiGoalShortestPath``, keeps it on the episode
  (``_shortest_path_cache``) so the multi-goal precomputation is paid once, and
  returns ``path.geodesic_distance``.

Do not substitute Euclidean distance, distance to the object centre, or a grid
planner: each changes SR and SPL. Never hand this object to a SearchPolicy.
"""
from __future__ import annotations

import numpy as np


class ViewPointDistance:
    """Geodesic distance from a position to the nearest published view point.

    One :class:`habitat_sim.MultiGoalShortestPath` is reused for the whole
    episode, exactly as habitat-lab caches it on the episode: the ends never
    change within an episode, and rebuilding it every step would re-run the
    multi-goal precomputation for the same answer.

    Args:
        pathfinder: The simulator's pathfinder, with the published navmesh
            loaded. This class never recomputes or edits a navmesh.
        view_points: (N, 3) published view-point positions in Habitat XYZ.

    Raises:
        ValueError: On an empty or malformed view-point array.
    """

    def __init__(self, pathfinder, view_points):
        import habitat_sim

        points = np.asarray(view_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("View points must be a nonempty (N, 3) array")
        if not np.isfinite(points).all():
            raise ValueError("View points must be finite")
        self._pathfinder = pathfinder
        self._path = habitat_sim.MultiGoalShortestPath()
        self._path.requested_ends = points
        self.view_point_count = int(len(points))

    def distance(self, habitat_position):
        """Metres to the nearest view point, ``inf`` when none is reachable.

        ``inf`` is habitat-sim's answer for an unreachable or off-navmesh
        position; it is returned as-is rather than replaced by a sentinel, so
        an unreachable goal cannot quietly become a finite distance.
        """
        position = np.asarray(habitat_position, dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Need a finite Habitat XYZ position")
        self._path.requested_start = position
        self._pathfinder.find_path(self._path)
        return float(self._path.geodesic_distance)
