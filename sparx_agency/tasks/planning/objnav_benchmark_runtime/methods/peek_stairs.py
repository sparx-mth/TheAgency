"""Seen, floor-local stair geometry is never a room-peek destination or route."""
from __future__ import annotations

import math
import cv2
import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D


def stair_peek_mask(policy, world):
    """Rasterize seen portal polylines only near the current storey's elevation.

    Uses the existing OpenCV capsule rasterizer, not per-pixel iteration. The
    margin includes body radius, arrival tolerance and one map cell. Other-floor
    flights are clipped out before projection, so switchbacks do not erase rooms.
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
            a, b = np.asarray(left[:3], float), np.asarray(right[:3], float)
            dz = b[2] - a[2]
            if abs(dz) < 1e-9:
                if abs(a[2] - height) > band:
                    continue
            else:
                low, high = sorted(((height - band - a[2]) / dz, (height + band - a[2]) / dz))
                low, high = max(0.0, low), min(1.0, high)
                if low > high:
                    continue
                a, b = a + low * (b - a), a + high * (b - a)
            start = tuple(int(v) for v in world.world_to_grid(*a[:2]))
            end = tuple(int(v) for v in world.world_to_grid(*b[:2]))
            cv2.line(mask, start, end, (1,), thickness=thickness)
    return mask.astype(bool)


def peek_planning_world(policy, world, excluded=None):
    """A peek-only occupancy copy; ordinary/committed stair navigation is untouched."""
    excluded = stair_peek_mask(policy, world) if excluded is None else excluded
    if not excluded.any():
        return world
    grid = world.grid.copy()
    grid[excluded] = world.values.occupied
    return OccupancyGrid2D(grid, world.params, values=world.values)

