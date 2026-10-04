"""Where to stand in a room to see it: the interior point of greatest clearance."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def vantage_point(world, mask, reachable=None, min_clearance_m=0.0):
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

    Returns:
        ``((x, y), clearance_m)`` or ``None`` when no eligible cell exists.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    clearance = distance_transform_edt(mask) * float(world.resolution)
    eligible = mask if reachable is None else (mask & np.asarray(reachable, dtype=bool))
    if not eligible.any():
        return None
    scores = np.where(eligible, clearance, -1.0)
    gy, gx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = float(scores[gy, gx])
    if best < min_clearance_m:
        return None
    x, y = world.grid_to_world(int(gx), int(gy))
    return (float(x), float(y)), best

