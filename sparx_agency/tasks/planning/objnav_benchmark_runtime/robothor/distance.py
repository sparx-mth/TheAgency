"""Geodesic distance to a goal object, asked of AI2-THOR's own navmesh.

The RoboTHOR challenge never measures a distance: it scores
``stopped and target_obj["visible"]``, and its SPL numerator is a polyline
shipped inside the episode file. But SoftSPL and the distance-to-goal figure
the ObjectNav literature reports (OSG Navigator: "average distance of the
agent from the goal instance at the end of each episode") need a distance, and
it has to be the *same* quantity at the start and at the end or the ratio
means nothing.

So both ends use the one call that produced the shipped path in the first
place -- ``GetShortestPath`` with an ``objectType`` -- through
``ai2thor.util.metrics``' own arithmetic. That makes the start distance
directly checkable: recomputed at an episode's published spawn point it must
reproduce that episode's ``shortest_path_length``, and
:meth:`RobothorGeodesic.check_start` is what a preflight uses to prove the
live build agrees with the shipped files before a sweep begins.

Two upstream behaviours are reproduced deliberately:

* **The tolerance ladder.** ``GetShortestPath`` fails outright when it thinks
  no path exists, which happens for numerical reasons near walls and
  furniture. AllenAct answers by retrying with an escalating ``allowedError``
  (0.05 up to 2.5) rather than recording an unreachable goal, and so does
  this. An exhausted ladder is reported as ``inf``, which the scorer counts
  separately and never averages.
* **It is a ``controller.step``.** The query returns an event and replaces
  the controller's ``last_event``. Nothing here reads ``last_event``, and the
  bridge keeps its own reference to the frame it last rendered, so a distance
  query never becomes the agent's observation.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math

from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.dataset import (
    path_distance,
)

#: AllenAct's escalating tolerance, in metres. The first entry is the value a
#: healthy query needs; the rest buy robustness near walls at the cost of a
#: little slack in where the path starts and ends.
ALLOWED_ERRORS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class RobothorGeodesic:
    """Distance along the navmesh from a position to the nearest instance of a type.

    Args:
        controller: Anything with AI2-THOR's ``step(dict) -> event``.
        allowed_errors: The tolerance ladder, tried in order.

    Attributes:
        queries: How many ``GetShortestPath`` calls were made, and how many
            needed a relaxed tolerance or failed outright -- recorded in the
            run manifest, because a build that needs the ladder constantly is
            not the build the numbers were published on.
    """

    def __init__(self, controller, allowed_errors=ALLOWED_ERRORS):
        if not allowed_errors:
            raise ValueError("Need at least one allowed_error")
        self._controller = controller
        self._allowed_errors = tuple(allowed_errors)
        self.queries = {"total": 0, "relaxed": 0, "unreachable": 0}

    def corners(self, position, object_type):
        """The shortest-path corners, or None when the ladder is exhausted."""
        self.queries["total"] += 1
        for index, allowed in enumerate(self._allowed_errors):
            event = self._controller.step({
                "action": "GetShortestPath",
                "objectType": object_type,
                "position": {axis: float(position[axis])
                             for axis in ("x", "y", "z")},
                "allowedError": allowed,
            })
            if event.metadata.get("lastActionSuccess"):
                if index:
                    self.queries["relaxed"] += 1
                return event.metadata["actionReturn"]["corners"]
        self.queries["unreachable"] += 1
        return None

    def distance(self, position, object_type) -> float:
        """Geodesic metres to the nearest ``object_type``; ``inf`` if unreachable."""
        corners = self.corners(position, object_type)
        if corners is None:
            return float("inf")
        return path_distance(corners)

    def check_start(self, row, tolerance_m=0.05) -> float:
        """Recompute ``l`` at a published spawn point and compare with the file.

        This is the one end-to-end proof that the installed Unity build,
        its navmesh and the shipped episode files are the same benchmark. A
        build whose navmesh differs gives a different ``l`` here while every
        episode still runs and every SPL still looks reasonable.

        Args:
            row: A :class:`~...robothor.dataset.RobothorEpisodeRow`.
            tolerance_m: How far the recomputed length may differ. The query
                is allowed to start up to ``allowed_errors[0]`` from the
                requested point, so this is slack, not equality.

        Returns:
            The recomputed length, metres.

        Raises:
            ValueError: The recomputed length disagrees with the shipped one,
                or no path was found at all from a published start.
        """
        measured = self.distance(row.start_position, row.category)
        if not math.isfinite(measured):
            raise ValueError(
                "No navmesh path from the published start of %s to a %s; this "
                "build disagrees with the shipped episode file"
                % (row.episode_id, row.category))
        if abs(measured - row.shortest_path_length_m) > tolerance_m:
            raise ValueError(
                "%s ships shortest_path_length %.4f m but this build measures "
                "%.4f m from the same start. The installed Unity build is not "
                "the one the episodes were generated on"
                % (row.episode_id, row.shortest_path_length_m, measured))
        return measured
