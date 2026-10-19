"""Where to stand in a room to see it: the interior point of greatest clearance."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def vantage_point(world, mask, reachable=None, min_clearance_m=0.0, avoid=(), avoid_radius_m=0.0):
    """The room cell farthest from everything that is not the room, as a world point.

    The camera's ray ends at the nearest object, so the frame that shows the
    most of a room is the one taken from the point with the most room around
    it -- the middle of the open floor, not the doorway and not the frontier
    at the far wall. Measured as the Euclidean distance transform of the room
    mask (walls, unknown space and other rooms all count as "not the room"),
    restricted to ``reachable`` cells when given (``True`` where the observed
    passable map connects to the agent).

    Args:
        world: The floor's :class:`OccupancyGrid2D`; supplies the resolution
            and the grid-to-world transform.
        mask: ``(H, W)`` bool, the room's cells.
        reachable: Optional ``(H, W)`` bool; cells outside it are not offered.
        min_clearance_m: Below this clearance the room has no vantage worth
            walking to (None is returned, and the caller scans where it is).
        avoid: World ``(x, y)`` points a second look should NOT be taken
            from again -- the room's earlier scan points. Cells within
            ``avoid_radius_m`` of any of them are not offered; when that
            leaves no eligible cell the discs are ignored, so a small room
            is re-scanned from its one good spot rather than not at all.
        avoid_radius_m: The radius of those discs; 0 disables them.

    Returns:
        ``((x, y), clearance_m)`` or ``None`` when no eligible cell exists.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    clearance = _clearance(mask) * float(world.resolution)
    eligible = mask if reachable is None else (mask & np.asarray(reachable, dtype=bool))
    if not eligible.any():
        return None
    if avoid and avoid_radius_m > 0.0:
        kept = eligible.copy()
        ys, xs = np.nonzero(kept)
        radius_cells = float(avoid_radius_m) / float(world.resolution)
        for point in avoid:
            gx, gy = world.world_to_grid(float(point[0]), float(point[1]))
            close = (xs - gx) ** 2 + (ys - gy) ** 2 <= radius_cells * radius_cells
            kept[ys[close], xs[close]] = False
        if kept.any():
            eligible = kept
    scores = np.where(eligible, clearance, -1.0)
    gy, gx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = float(scores[gy, gx])
    if best < min_clearance_m:
        return None
    x, y = world.grid_to_world(int(gx), int(gy))
    return (float(x), float(y)), best



def _clearance(mask):
    """The distance transform of ``mask`` in cells, computed on its bounding box only.

    Every cell outside the mask is "not the room", so a one-cell False
    border around the box reproduces the full-grid transform exactly -- on
    a 10 m room of an 80 m map that is a hundredth of the work (the full
    grid cost 15 ms per room per selection, review of 2026-10-07).
    """
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # The full-grid transform knows no boundary at the grid's own edge, so a side the
    # box already touches is left unpadded (a room on the map's edge is the same either way).
    top, bottom, left, right = int(y0 > 0), int(y1 < h), int(x0 > 0), int(x1 < w)
    window = np.zeros((y1 - y0 + top + bottom, x1 - x0 + left + right), dtype=bool)
    window[top:top + y1 - y0, left:left + x1 - x0] = mask[y0:y1, x0:x1]
    clearance = np.zeros(mask.shape, dtype=float)
    clearance[y0:y1, x0:x1] = distance_transform_edt(window)[top:top + y1 - y0, left:left + x1 - x0]
    return clearance
