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
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D

#: The polyline within this height of the storey plane is floor in front of the stairs, not a tread (the observed
#: map's free band is the plane +/- 0.10 m; the first tread of a flight rises 0.15-0.20 m).
FLOOR_RUN_M = 0.10


def walked_exemption(policy, world, pose=None) -> np.ndarray:
    """``(H, W)`` bool: the cells the agent has stood on (body radius + one cell), never to be masked for planning."""
    sight = getattr(policy, "sight", None)
    if sight is None:
        return np.zeros(world.grid.shape, dtype=bool)
    radius = policy.settings.body_radius_m + world.resolution
    include = None if pose is None else [(float(pose.x), float(pose.y))]
    return sight.walked_mask(world, policy.mapping.floor_id, radius, include=include)


def stair_peek_mask(policy, world, pose=None, carve=True):
    """Rasterize seen portal polylines only near the current storey's elevation, off the floor run and the trail.

    Uses the existing OpenCV capsule rasterizer, not per-pixel iteration. The
    margin includes body radius, arrival tolerance and one map cell. Other-floor
    flights are clipped out before projection, so switchbacks do not erase rooms;
    the floor run within ``FLOOR_RUN_M`` of the storey height is clipped out too,
    and the cells the agent has stood on (``walked_exemption``, ``pose`` being
    the one not yet in the ledger) are never part of the mask. ``carve=False``
    returns the raw footprint for a caller that combines masks and carves once.
    """
    mask = np.zeros(world.grid.shape, np.uint8)
    building = getattr(policy, "building", None)
    if building is None:
        return mask.astype(bool)
    atlas = policy.mapping.atlas
    height = atlas.elevation_m if atlas.floors else getattr(policy.mapping, "_anchor", None)
    if height is None:
        return mask.astype(bool)
    band = building.params.departure_m
    margin = policy.settings.body_radius_m + policy.converter_params.goal_tolerance_m + world.resolution
    thickness = 2 * int(math.ceil(margin / world.resolution)) + 1
    for portal in building.portals:
        if portal["floor_id"] != policy.mapping.floor_id:
            continue
        points = portal.get("path") or [portal["entry"]]
        pairs = list(zip(points, points[1:])) or [(points[0], points[0])]
        for left, right in pairs:
            for a, b in _flight_pieces(np.asarray(left[:3], float), np.asarray(right[:3], float), height, band):
                start = tuple(int(v) for v in world.world_to_grid(*a[:2]))
                end = tuple(int(v) for v in world.world_to_grid(*b[:2]))
                cv2.line(mask, start, end, (1,), thickness=thickness)
    drawn = mask.astype(bool)
    if carve and drawn.any():
        drawn &= ~walked_exemption(policy, world, pose)
    return drawn


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

