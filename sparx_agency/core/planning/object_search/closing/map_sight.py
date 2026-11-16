"""Whether the observed map puts an obstacle between two points: the closing's one sight-line test.

Three questions of the closing are this one question, and each used to carry
its own copy of the ray: is the remembered target still in sight of the agent
(a STOP on the memory alone needs it; a failed inspection is judged blind by
it), would the target be in sight from a spot the agent might back off to, and
-- since 2026-11-15 -- does a standoff moved onto standing floor still face
the target, or has it been moved to the far side of the wall the target stands
against. The ray is tested on the agent's own occupancy grid and on OCCUPIED
cells only: an UNKNOWN cell hides nothing the map knows of, and "the map has
seen a wall here" is the whole of the claim.

A 2-D map has no heights, so a table between the agent and a picture above it
blocks the ray although the camera sees over it. The callers treat the answer
accordingly -- as a reason to prefer another spot or to distrust a memory, not
as proof that nothing can be seen.

Pure functions of the grid; numpy, Python 3.8 syntax.
"""
from __future__ import annotations

import math

from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear

#: Along the line to the 3-D memory, this much at the memory's end is the object and what it stands on, and is not read.
MEMORY_SIGHT_MARGIN_M = 0.4


def map_sight_clear(world, from_xy, to_xy, skip_from_m: float = 0.0, skip_to_m: float = 0.0,
                    off_map: bool = False) -> bool:
    """Whether the map shows nothing OCCUPIED on the straight line from ``from_xy`` to ``to_xy``.

    Args:
        world: The agent's occupancy grid.
        from_xy: Where the line starts, world metres.
        to_xy: Where it ends.
        skip_from_m: This much of the line at its start is not read (the
            body standing there).
        skip_to_m: This much at its end is not read (the object looked at
            and its support are occupied cells themselves).
        off_map: The answer when either end of the part that is read lies
            off the grid.

    Returns:
        True when no occupied cell lies on the part of the line that is
        read -- including when nothing is left to read, the two skips
        covering the whole line.
    """
    dx, dy = float(to_xy[0]) - float(from_xy[0]), float(to_xy[1]) - float(from_xy[1])
    length = math.hypot(dx, dy)
    if length - float(skip_to_m) <= float(skip_from_m):
        return True
    ux, uy = dx / length, dy / length
    x0, y0 = world.world_to_grid(float(from_xy[0]) + float(skip_from_m) * ux, float(from_xy[1]) + float(skip_from_m) * uy)
    x1, y1 = world.world_to_grid(float(from_xy[0]) + (length - float(skip_to_m)) * ux,
                                 float(from_xy[1]) + (length - float(skip_to_m)) * uy)
    if not (world.in_bounds(x0, y0) and world.in_bounds(x1, y1)):
        return bool(off_map)
    return bool(line_of_sight_clear(world.grid == world.values.occupied, x0, y0, x1, y1))
