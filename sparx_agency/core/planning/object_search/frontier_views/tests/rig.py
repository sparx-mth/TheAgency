"""A rig for the frontier-view tests: small maps drawn in metres, the benchmark's camera, frames by pose.

Every map is 20 x 20 m at the method's 0.1 m, centred on the origin, unknown
until drawn on. Rectangles are given in world metres and cover every cell
whose centre lies inside them, so a test reads as a floor plan::

    plan = Plan()
    plan.free(0.0, -2.0, 5.0, 2.0)          # a room
    plan.wall(5.0, -2.0, 5.1, -0.4)         # its east wall, south of the doorway

No policy and no simulator: the selector is exercised on the types it reads.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameRecord, frame_name
from sparx_agency.core.planning.object_search.frontier_views.nodes import FrontierNode
from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
RESOLUTION = 0.1
SIZE_M = 20.0
ORIGIN = -0.5 * SIZE_M
CELLS = int(round(SIZE_M / RESOLUTION))


def camera(width=640, height=480, hfov_deg=79.0) -> CameraSpec:
    """The benchmark's camera: 640 x 480, 79 degrees, 0.88 m up, depth 0.5 to 5 m."""
    return CameraSpec(intrinsics_from_hfov(width, height, hfov_deg), 0.88, 0.5, 5.0)


def pose(x, y, yaw_deg=0.0, pitch_deg=0.0, z=0.0) -> AgentPose:
    return AgentPose(float(x), float(y), float(z), math.radians(yaw_deg), math.radians(pitch_deg))


def record(step, at: AgentPose) -> FrameRecord:
    """The frame of ``step``, captured at ``at``."""
    return FrameRecord(frame_name(step, ".jpg"), step, at)


def cell(x, y):
    """``(row, col)`` of the cell holding the world point ``(x, y)``."""
    return int(math.floor((y - ORIGIN) / RESOLUTION)), int(math.floor((x - ORIGIN) / RESOLUTION))


class Plan:
    """A floor plan drawn in world metres on an all-unknown grid."""

    def __init__(self):
        self.grid = np.full((CELLS, CELLS), VALUES.unknown, dtype=np.int8)
        self.settled = np.zeros((CELLS, CELLS), dtype=bool)

    def _box(self, x0, y0, x1, y1):
        centres = (np.arange(CELLS) + 0.5) * RESOLUTION + ORIGIN
        cols = np.nonzero((centres >= x0) & (centres < x1))[0]
        rows = np.nonzero((centres >= y0) & (centres < y1))[0]
        if not len(cols) or not len(rows):
            raise ValueError("the rectangle (%r, %r)-(%r, %r) covers no cell centre" % (x0, y0, x1, y1))
        return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)

    def free(self, x0, y0, x1, y1):
        self.grid[self._box(x0, y0, x1, y1)] = VALUES.free
        return self

    def wall(self, x0, y0, x1, y1):
        self.grid[self._box(x0, y0, x1, y1)] = VALUES.occupied
        return self

    def unknown(self, x0, y0, x1, y1):
        self.grid[self._box(x0, y0, x1, y1)] = VALUES.unknown
        return self

    def settle(self, x0, y0, x1, y1):
        """Mark the unknown of a rectangle as settled (looked through, a pocket)."""
        self.settled[self._box(x0, y0, x1, y1)] = True
        return self

    def world(self) -> OccupancyGrid2D:
        return OccupancyGrid2D(self.grid.copy(), OccupancyGrid2DParams(RESOLUTION, ORIGIN, ORIGIN, "world"), values=VALUES)

    def cells(self, x0, y0, x1, y1):
        """``(rows, cols)`` of every cell whose centre lies in the rectangle, row-major."""
        rows, cols = self._box(x0, y0, x1, y1)
        grid_rows, grid_cols = np.meshgrid(np.arange(rows.start, rows.stop), np.arange(cols.start, cols.stop), indexing="ij")
        return grid_rows.ravel(), grid_cols.ravel()


def node(plan: Plan, x0, y0, x1, y1, node_id=0, floor_id=0) -> FrontierNode:
    """A node made of the cells of a rectangle, with its goal point at the rectangle's middle cell."""
    rows, cols = plan.cells(x0, y0, x1, y1)
    middle = len(rows) // 2
    gx, gy = int(cols[middle]), int(rows[middle])
    xy = ((gx + 0.5) * RESOLUTION + ORIGIN, (gy + 0.5) * RESOLUTION + ORIGIN)
    return FrontierNode(node_id, floor_id, xy, (gx, gy), rows, cols, 1)


def room_with_doorway() -> Plan:
    """A 5 x 4 m room whose east wall has a 0.8 m doorway, and the 2 m of corridor seen through it.

    The room: ``x`` in [0, 5), ``y`` in [-2, 2). Its east wall stands at
    ``x`` in [5.0, 5.1) but for the doorway, ``y`` in [-0.4, 0.4). Beyond,
    the floor seen through the doorway: ``x`` in [5.1, 7.0), ``y`` in
    [-0.4, 0.4). Everything else is unknown -- so the corridor's far end
    (``x`` just under 7) and its two sides are the map's boundary.
    """
    plan = Plan()
    plan.free(0.0, -2.0, 5.0, 2.0)
    plan.wall(-0.1, -2.1, 0.0, 2.1).wall(-0.1, -2.1, 5.1, -2.0).wall(-0.1, 2.0, 5.1, 2.1)
    plan.wall(5.0, -2.0, 5.1, -0.4).wall(5.0, 0.4, 5.1, 2.0)
    plan.free(5.0, -0.4, 7.0, 0.4)
    return plan
