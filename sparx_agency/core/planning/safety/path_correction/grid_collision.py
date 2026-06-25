"""Inflated line-of-sight collision check for a path on an OccupancyGrid2D.

A small, ROS-free mirror of ``WeightedAStarPlanner2D.path_collides``: inflate the
occupied cells by a robot radius once, then Bresenham-test each path segment. It
lets a path *corrector* (which holds no planner) re-validate a reshaped path
against the same inflated-obstacle model the planner used, so the corrected path
is never less safe than the planned one.

Reuses the shared grid primitives in
``sparx_agency.core.planning.planners.common.grid_geometry_2d`` (pure numpy, no
ROS); Python 3.8 compatible (the FALCON Noetic adapter imports core under 3.8).
"""
from __future__ import annotations

from typing import Sequence

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.planners.common.grid_geometry_2d import (
    dilate_mask,
    line_of_sight_clear,
)


class InflatedGridCollisionChecker:
    """Segment / path collision test against an inflated occupancy grid.

    Build once per grid (the inflation is computed in ``__init__``), then call
    :meth:`segment_clear` / :meth:`path_collides` as often as needed -- e.g. the
    per-waypoint bisection of a trajectory clip, which probes many candidate
    segments against the same grid.
    """

    def __init__(self, grid: OccupancyGrid2D, inflate_radius_m: float) -> None:
        occ = grid.grid == grid.values.occupied
        n = max(0, int(round(inflate_radius_m / grid.resolution)))
        if n > 0:
            occ = dilate_mask(occ, n)
        self._grid = grid
        self._occ = occ
        self._h, self._w = occ.shape

    def segment_clear(self, a: Pose2D, b: Pose2D) -> bool:
        """True if the world segment ``a -> b`` clears all inflated obstacles.

        A segment with an endpoint outside the grid is treated as clear (matching
        ``WeightedAStarPlanner2D.path_collides``, which skips out-of-bounds
        segments) so the caller never reverts a waypoint merely for leaving the
        mapped area.
        """
        x0, y0 = self._grid.world_to_grid(a.x, a.y)
        x1, y1 = self._grid.world_to_grid(b.x, b.y)
        if not (0 <= x0 < self._w and 0 <= y0 < self._h
                and 0 <= x1 < self._w and 0 <= y1 < self._h):
            return True
        return line_of_sight_clear(self._occ, x0, y0, x1, y1)

    def path_collides(self, points: Sequence[Pose2D]) -> bool:
        """True if any segment of ``points`` crosses an inflated obstacle."""
        if len(points) < 2:
            return False
        for a, b in zip(points[:-1], points[1:]):
            if not self.segment_clear(a, b):
                return True
        return False
