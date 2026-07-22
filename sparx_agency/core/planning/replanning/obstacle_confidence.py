"""Confidence of the obstacle(s) a committed route now crosses.

ROS-free and 3.8-compatible. This supports a *confidence-aware* collision replan:
on a noisy monocular-depth map a slow stop-and-turn platform should reroute off
its committed route only once the map is actually SURE the on-route obstacle is
real -- not on a single low-confidence speckle that another look would clear.

The BEV temporal filter (see :class:`BevProjector`) already accumulates per-cell
evidence and only latches a cell OCCUPIED once that evidence crosses ``t_on``.
But once latched, a *stably* mis-triggered speckle is indistinguishable from a
wall in the binary grid, so a frame-count collision gate keeps rerouting on it --
and the reroute turns the drone away before it can re-observe and clean the cell.

:func:`route_obstacle_confidence` reads that evidence (normalised to ``[0, 1]``)
at the occupied cells the route crosses and returns the strongest one, so the
node can gate the reroute on "the obstacle is confirmed" rather than only "the
path is blocked this frame". A cell that stays only marginally occupied never
reaches full confidence, so the route is kept and the drone keeps observing it;
a real wall firms up quickly and reroutes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D

from .path_raster import corridor_mask


def route_obstacle_confidence(
    world: OccupancyGrid2D,
    confidence: Optional[np.ndarray],
    points: Sequence[Pose2D],
    radius_cells: int,
) -> Optional[float]:
    """Peak map confidence among the occupied cells within ``radius_cells`` of a route.

    The route is rasterized and dilated by ``radius_cells`` (the obstacle inflation
    radius) into a corridor -- the same band whose inflated obstacle cells make the
    path "collide" -- and the maximum confidence over the *truly* occupied cells in
    that band is returned.

    Args:
        world: BEV occupancy grid; its ``values.occupied`` cells are the obstacles.
        confidence: ``(H, W)`` per-cell confidence in ``[0, 1]`` co-registered with
            ``world`` (``BevProjector.last_confidence``), or ``None``.
        points: World waypoints of the route to test (e.g. the remaining committed
            route). Fewer than one point yields ``0.0``.
        radius_cells: Corridor half-width in cells; pass the obstacle inflation
            radius in cells so the cells whose inflation blocks the route are read.

    Returns:
        The maximum confidence over occupied cells inside the corridor, in
        ``[0, 1]``; ``0.0`` when no occupied cell lies on/near the route; ``None``
        when ``confidence`` is missing or its shape does not match ``world`` (the
        caller should then fall back to the frame-count gate).
    """
    if confidence is None:
        return None
    occ = world.grid == world.values.occupied
    if confidence.shape != occ.shape:
        return None
    corridor = corridor_mask(points, world, max(0, int(radius_cells)))
    region = occ & corridor
    if not region.any():
        return 0.0
    return float(np.clip(confidence[region].max(), 0.0, 1.0))
