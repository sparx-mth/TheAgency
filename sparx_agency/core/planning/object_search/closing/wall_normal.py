"""The wall a raised target hangs on, read from the occupancy grid: its normal into the room.

MP3D's evaluator accepts a STOP within 0.1 m of a published *view point*, and
the view points of a thin wall-mounted object -- a picture, a wall cabinet, a
towel on its rail -- stand IN FRONT of it: within the object's own width along
the wall (0.26 m beyond it at the median over the single-storey set's 78
pictures, against 0.6-0.9 m for every other category) and 0.2-0.8 m out from
the wall. The dataset generator keeps a candidate within 1 m of the object's
box only when the object's pixels show from there with the camera turned to
its centre, and a 2 cm thick picture shows nothing from beside itself.

The closing selects its standoff on a ring about the memory starting from the
agent's own bearing (``target_path.TargetPath``), so a picture first seen at a
grazing angle was approached and STOPped on from beside it: 2azQ1b91cZZ/000157,
QUCTc6BB5sX/000055 and zsNo4HB9uLZ/000045 (2026-11-10) ended 0.26-0.38 m
beside the object's width, 60-69 degrees off the wall's normal, each within
the 1 m band and 0.1-0.5 m from the nearest view point; every picture success
of the same run stood within 0.28 m of the width footprint. :func:`wall_normal`
gives the ring its first bearing instead: the direction from the memory into
the room, the normal of whatever it is mounted on.

The front is where the floor is. The map integrates obstacles in a band above
the floor the agent walks on, so a picture itself is not in it, but the wall
under it is, and so is the sideboard, sofa or vanity it hangs above -- a blob
beside the wall that defeats a line fit (the three cases above all have one).
What every mounting has in common is that the free cells around the memory lie
on the room's side: :func:`wall_normal` is the direction from the memory to the
centroid of the free cells within ``radius_m``, given that something occupied
stands within ``mount_m`` of it. A raised object with free floor all round it
-- a plant on a stand mid-room -- has no front and gets None.
"""
import math
from typing import Optional, Sequence, Tuple

import numpy as np

#: Fewer free cells than this around the memory is no room side to speak of.
MIN_CELLS = 6
#: The free centroid must stand at least this far from the memory: free floor all round is no front.
MIN_OFFSET_M = 0.15


def wall_normal(world, xy: Sequence[float], radius_m: float = 0.8,
                mount_m: float = 0.3) -> Optional[Tuple[float, float]]:
    """The unit direction from ``xy`` into the room it is mounted on the edge of; None when it has no such side.

    Args:
        world: An ``OccupancyGrid2D`` (``grid``, ``values``, ``world_to_grid``, ``resolution``, origin).
        xy: The target's memory, planar, in the grid's world frame.
        radius_m: The free cells within this planar radius of ``xy`` are the room side's evidence.
        mount_m: Something occupied must stand within this distance of ``xy``: a target with nothing
            under or behind it is free-standing, not mounted, and gets None.

    Returns:
        ``(nx, ny)`` of unit length, or None when nothing occupied is near, the free cells are too few,
        or they surround the memory evenly.
    """
    gx, gy = world.world_to_grid(float(xy[0]), float(xy[1]))
    reach = int(math.ceil(float(radius_m) / world.resolution))
    x0, x1 = max(gx - reach, 0), min(gx + reach + 1, world.width)
    y0, y1 = max(gy - reach, 0), min(gy + reach + 1, world.height)
    if x0 >= x1 or y0 >= y1:
        return None
    window = world.grid[y0:y1, x0:x1]

    def centres(mask):
        ys, xs = np.nonzero(mask)
        points = np.stack([(xs + x0 + 0.5) * world.resolution + world.origin_x,
                           (ys + y0 + 0.5) * world.resolution + world.origin_y], axis=1)
        return points, np.hypot(points[:, 0] - float(xy[0]), points[:, 1] - float(xy[1]))

    occupied, occupied_range = centres(window == world.values.occupied)
    if len(occupied) == 0 or occupied_range.min() > float(mount_m):
        return None                                                  # nothing to hang on
    free, free_range = centres(window == world.values.free)
    free = free[free_range <= float(radius_m)]
    if len(free) < MIN_CELLS:
        return None
    offset = free.mean(axis=0) - np.asarray(xy, dtype=float)
    length = float(np.hypot(offset[0], offset[1]))
    if length < MIN_OFFSET_M:
        return None                                                  # free floor all round: no front
    return float(offset[0] / length), float(offset[1] / length)
