"""Seen, floor-local stair geometry is never a room-peek destination or route.

Two rules keep the mask from swallowing the agent itself (Collierville
2026-10-08: the agent walked past the foot of a flight 0.6 m from its
centreline, the capsule written occupied around that centreline covered its
cell, and with no start cell A* failed for every goal -- 309 idle turns):

* **The floor run is not the flight.** A connector's polyline begins on the
  storey's own floor and climbs from there; its part within ``FLOOR_RUN_M``
  of the storey height is the floor in front of the stairs (the observed
  map's own free band), where the agent walks past, so only the part of the
  flight above that -- and below it, for a descending flight -- is drawn.
* **Where the agent stood is passable.** The cells within the body radius
  and one cell of every pose the agent has taken on this storey (the
  sight ledger's trail, plus the current pose) are carved out of the mask:
  the agent was physically there, and nothing written for planning may
  outvote that.
* **The capsule never closes a corridor** (``adaptive_capsule``, since
  2026-10-08). Its preferred margin -- body radius, arrival tolerance and a
  cell, 0.65 m at 0.1 m cells -- is a guess at a flight's half-width plus
  room for the agent's body; the planner inflates obstacles by the body
  radius again, and in Markleeville the end cap at a stair head reached
  into the 0.55 m corridor between the stairwell and the wall, leaving a
  0.25 m gap no body fits through and the floor beyond it unreachable. So
  the margin is bounded per flight: from the preferred value down to the
  body radius, one cell at a time, the first capsule that neither splits
  nor erases any body-passable region of free floor around it -- passable
  as the planner sees it, free cells at least a body radius from anything
  blocked -- is the one drawn.
"""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy import ndimage

from sparx_agency.core.planning.environment import OccupancyGrid2D

#: The polyline within this height of the storey plane is floor in front of the stairs, not a tread (the observed
#: map's free band is the plane +/- 0.10 m; the first tread of a flight rises 0.15-0.20 m).
FLOOR_RUN_M = 0.10
#: A body-passable region smaller than this many cells is not a corridor the capsule must spare.
CORRIDOR_MIN_CELLS = 10
#: How far around the preferred capsule passability is judged, metres: a corridor split here reconnects, if at
#: all, around something bigger than a room.
CORRIDOR_WINDOW_M = 2.5


def walked_exemption(policy, world, pose=None) -> np.ndarray:
    """``(H, W)`` bool: the cells the agent has stood on (body radius + one cell), never to be masked for planning."""
    sight = getattr(policy, "sight", None)
    if sight is None:
        return np.zeros(world.grid.shape, dtype=bool)
    radius = policy.settings.body_radius_m + world.resolution
    include = None if pose is None else [(float(pose.x), float(pose.y))]
    return sight.walked_mask(world, policy.mapping.floor_id, radius, include=include)


def stair_peek_mask(policy, world, pose=None, carve=True, report=None):
    """Rasterize seen portal polylines only near the current storey's elevation, off the floor run and the trail.

    Uses the existing OpenCV capsule rasterizer, not per-pixel iteration. The
    preferred margin includes body radius, arrival tolerance and one map cell,
    bounded per flight so no corridor closes (``adaptive_capsule``). Other-floor
    flights are clipped out before projection, so switchbacks do not erase rooms;
    the floor run within ``FLOOR_RUN_M`` of the storey height is clipped out too,
    and the cells the agent has stood on (``walked_exemption``, ``pose`` being
    the one not yet in the ledger) are never part of the mask. ``carve=False``
    returns the raw footprint for a caller that combines masks and carves once;
    ``report`` (a dict) receives the margin drawn per portal id, in metres.
    """
    mask = np.zeros(world.grid.shape, dtype=bool)
    building = getattr(policy, "building", None)
    if building is None:
        return mask
    atlas = policy.mapping.atlas
    height = atlas.elevation_m if atlas.floors else getattr(policy.mapping, "_anchor", None)
    if height is None:
        return mask
    band = building.params.departure_m
    res = world.resolution
    preferred = int(math.ceil((policy.settings.body_radius_m + policy.converter_params.goal_tolerance_m + res) / res))
    body = int(math.ceil(policy.settings.body_radius_m / res))
    for portal in building.portals:
        if portal["floor_id"] != policy.mapping.floor_id:
            continue
        points = portal.get("path") or [portal["entry"]]
        pairs = list(zip(points, points[1:])) or [(points[0], points[0])]
        segments = []
        for left, right in pairs:
            for a, b in _flight_pieces(np.asarray(left[:3], float), np.asarray(right[:3], float), height, band):
                segments.append((tuple(int(v) for v in world.world_to_grid(*a[:2])),
                                 tuple(int(v) for v in world.world_to_grid(*b[:2]))))
        if not segments:
            continue
        capsule, radius = adaptive_capsule(world, segments, preferred, body, body)
        mask |= capsule
        if report is not None:
            report[str(portal["id"])] = round(radius * res, 2)
    if carve and mask.any():
        mask &= ~walked_exemption(policy, world, pose)
    return mask


def _capsule(shape, segments, radius_cells):
    """The cells within ``radius_cells`` of the polyline ``segments`` (grid coordinates), as OpenCV draws them."""
    mask = np.zeros(shape, np.uint8)
    for start, end in segments:
        cv2.line(mask, start, end, (1,), thickness=2 * int(radius_cells) + 1)
    return mask.astype(bool)


def _passable(free, blocked, body_cells):
    """Free cells at least ``body_cells`` from the nearest blocked cell: where the planner's floor rung may stand."""
    if body_cells <= 0:
        return free & ~blocked
    distance = ndimage.distance_transform_edt(~blocked)
    return free & (distance >= body_cells)


_EIGHT = np.ones((3, 3), dtype=bool)


def _cuts(labels, count, sizes, candidate_passable):
    """Whether ``candidate_passable`` splits or erases a body-passable region of ``labels`` worth sparing."""
    for index in range(1, count + 1):
        if sizes[index] < CORRIDOR_MIN_CELLS:
            continue
        piece = candidate_passable & (labels == index)
        if not piece.any() or ndimage.label(piece, structure=_EIGHT)[1] > 1:
            return True
    return False


def adaptive_capsule(world, segments, preferred_cells, floor_cells, body_cells):
    """The widest capsule around ``segments``, from ``preferred_cells`` down to ``floor_cells``, that closes no corridor.

    Passability is judged inside a window ``CORRIDOR_WINDOW_M`` around the
    preferred capsule, on the observed grid: free cells at least the body
    radius from occupied, unknown or masked cells, 8-connected. The regions
    that exist under the ``floor_cells`` capsule are the baseline; the first
    (widest) candidate that neither splits nor erases one of at least
    ``CORRIDOR_MIN_CELLS`` cells is drawn, else the floor capsule itself.

    Returns:
        ``(mask, radius_cells)``.
    """
    preferred_cells, floor_cells = int(preferred_cells), int(floor_cells)
    if preferred_cells <= floor_cells:
        return _capsule(world.grid.shape, segments, max(0, preferred_cells)), max(0, preferred_cells)
    shape = world.grid.shape
    core = _capsule(shape, segments, 0)
    if not core.any():
        return core, 0
    ys, xs = np.nonzero(core)
    pad = preferred_cells + body_cells + int(round(CORRIDOR_WINDOW_M / world.resolution))
    y0, y1 = max(0, ys.min() - pad), min(shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(shape[1], xs.max() + pad + 1)
    window = (slice(y0, y1), slice(x0, x1))
    free = world.grid[window] == world.values.free
    blocked = ~free                                   # occupied or unknown: the planner treats both as walls
    floor_mask = _capsule(shape, segments, floor_cells)
    base = _passable(free, blocked | floor_mask[window], body_cells)
    labels, count = ndimage.label(base, structure=_EIGHT)
    if count == 0:
        return _capsule(shape, segments, preferred_cells), preferred_cells
    sizes = np.bincount(labels.ravel())
    for radius in range(preferred_cells, floor_cells, -1):
        candidate = _capsule(shape, segments, radius)
        if not _cuts(labels, count, sizes, _passable(free, blocked | candidate[window], body_cells)):
            return candidate, radius
    return floor_mask, floor_cells


def _flight_pieces(a, b, height, band):
    """The parts of segment ``a -> b`` whose height lies within ``band`` of ``height`` but outside the floor run.

    Two windows, above and below the plane: ``[height + FLOOR_RUN_M,
    height + band]`` and ``[height - band, height - FLOOR_RUN_M]``; a flat
    segment is kept whole when its height falls in one of them.
    """
    dz = b[2] - a[2]
    windows = ((height + FLOOR_RUN_M, height + band), (height - band, height - FLOOR_RUN_M))
    if abs(dz) < 1e-9:
        return [(a, b)] if any(lo <= a[2] <= hi for lo, hi in windows) else []
    pieces = []
    for lo, hi in windows:
        low, high = sorted(((lo - a[2]) / dz, (hi - a[2]) / dz))
        low, high = max(0.0, low), min(1.0, high)
        if low <= high:
            pieces.append((a + low * (b - a), a + high * (b - a)))
    return pieces


def peek_planning_world(policy, world, excluded=None, pose=None):
    """A peek-only occupancy copy; ordinary/committed stair navigation is untouched."""
    excluded = stair_peek_mask(policy, world, pose) if excluded is None else excluded
    if not excluded.any():
        return world
    grid = world.grid.copy()
    grid[excluded] = world.values.occupied
    return OccupancyGrid2D(grid, world.params, values=world.values)

