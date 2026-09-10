"""Metric distances onto the fake building's grid: where an agent of a radius can stand, where it has found an object, and what a depth camera sees.

A **test rig, not a benchmark.** The environment decides from these grids
where the agent may be and whether it has succeeded, the oracle plans on the
same ones, and the depth camera renders the voxels, so a grid that is subtly
wrong is a pipeline test that passes for the wrong reason:

* **An agent that clips walls.** A cell is navigable only when its centre is
  at least the agent's radius from every blocked cell's *square* -- not its
  centre, which would let a 0.18 m agent stand 0.05 m from a wall. Beyond the
  grid counts as blocked: an agent must not stand half off the map.
* **A goal region that is too small.** A cell is in an object's goal region
  when its centre lies within the success distance of the object's *square*,
  as Habitat's view points surround an object's extent; to its centre, an
  object one cell wide would be harder to reach from its side than its corner.
* **A camera that sees through the floor.** The voxel grid has a floor slab
  and a ceiling across the whole grid, so a ray inside the building always
  ends on a surface or runs out of range -- never reads ``inf`` below the
  horizon.

The repo's own dilations (``core.mapping.bev.morphology.dilate4``,
``core.planning.planners.common.grid_geometry_2d.dilate_mask``) grow a mask by
whole 4-connected rings, not by a metric radius, so the kernel here is local.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.tasks.planning.objnav_benchmark.checks import is_length
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

#: A displacement from one cell to another, ``(drow, dcol)``.
Offset = Tuple[int, int]


def navigable_mask(world: GridWorld, agent_radius_m: float) -> np.ndarray:
    """Cells whose centre is at least ``agent_radius_m`` from every blocked cell's square.

    Beyond the grid counts as blocked. A blocked cell is never navigable,
    whatever the radius.

    Args:
        world: The building.
        agent_radius_m: The agent's radius, metres; 0 for a point.

    Returns:
        A new boolean mask of the world's shape.

    Raises:
        ObjNavError: On a negative or non-finite radius.
    """
    if not is_length(agent_radius_m):
        raise ObjNavError("agent_radius_m must be a finite non-negative "
                          "number, got %r" % (agent_radius_m,))
    blocked = world.blocked_mask
    offsets = _square_offsets(float(agent_radius_m), world.resolution_m, False)
    return ~(_dilate(blocked, offsets, True) | blocked)


def goal_mask(world: GridWorld, category: str, success_distance_m: float,
              agent_radius_m: float) -> np.ndarray:
    """The fake's goal "view points": navigable cells near an object of ``category``.

    A cell qualifies when its centre lies within ``success_distance_m``
    (inclusive) of the nearest cell of the category -- measured to that
    cell's square, not its centre, so an object one cell wide is as easy to
    reach from its side as from its corner.

    Args:
        world: The building.
        category: An object category of the world.
        success_distance_m: How near counts as found, metres.
        agent_radius_m: The agent's radius, for :func:`navigable_mask`.

    Returns:
        A new boolean mask, empty when no navigable cell is near enough.

    Raises:
        ObjNavError: On an unknown category or a bad distance.
    """
    objects = world.object_mask(category)
    if not (is_length(success_distance_m) and success_distance_m > 0):
        raise ObjNavError("success_distance_m must be a positive finite "
                          "number, got %r" % (success_distance_m,))
    offsets = _square_offsets(float(success_distance_m), world.resolution_m,
                              True)
    return _dilate(objects, offsets, False) & navigable_mask(world,
                                                             agent_radius_m)


def voxel_grid(world: GridWorld
               ) -> Tuple[np.ndarray, Tuple[float, float, float], float]:
    """The world as an occupancy voxel grid, for ``VoxelDepthCamera``.

    Voxel layer 0 is a floor slab at ``z`` in ``[-resolution, 0)``; walls
    fill ``z`` in ``[0, wall_height_m)`` and objects ``[0,
    object_height_m)``, each rounded up to whole voxels; a ceiling slab sits
    on the first voxel boundary at or above ``wall_height_m``. Floor and
    ceiling span the whole grid, so a ray inside the building always ends on
    a surface or runs out of range.

    Args:
        world: The building.

    Returns:
        ``(voxels, origin, resolution)``: an int8 ``(nz, rows, cols)`` grid,
        1 occupied and 0 free; the world ``(x, y, z)`` of voxel ``(0, 0,
        0)``'s corner; the voxel edge in metres.
    """
    res = world.resolution_m
    wall_layers = int(math.ceil(world.wall_height_m / res - 1e-9))
    object_layers = int(math.ceil(world.object_height_m / res - 1e-9))
    rows, cols = world.shape
    walls = world.wall_mask
    grid = np.zeros((wall_layers + 2, rows, cols), dtype=np.int8)
    grid[0] = 1
    grid[1:1 + wall_layers, walls] = 1
    grid[1:1 + object_layers, world.blocked_mask & ~walls] = 1
    grid[wall_layers + 1] = 1
    ox, oy = world.origin_xy
    return grid, (ox, oy, -res), res


def _square_offsets(radius_m: float, resolution_m: float,
                    inclusive: bool) -> List[Offset]:
    """Offsets to every cell whose square lies within ``radius_m`` of a cell's centre.

    The distance from a cell's centre to the square of the cell ``(dr, dc)``
    away is ``resolution * hypot(max(|dr| - 1/2, 0), max(|dc| - 1/2, 0))``:
    zero to its own square, half a cell to an orthogonal neighbour's.

    Args:
        radius_m: The radius, metres.
        resolution_m: The cell edge, metres.
        inclusive: Keep offsets exactly ``radius_m`` away (``<=``) or not
            (``<``).
    """
    reach = int(math.ceil(radius_m / resolution_m + 0.5))
    offsets = []
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            gap = resolution_m * math.hypot(max(abs(dr) - 0.5, 0.0),
                                            max(abs(dc) - 0.5, 0.0))
            if gap < radius_m or (inclusive and gap <= radius_m):
                offsets.append((dr, dc))
    return offsets


def _dilate(mask: np.ndarray, offsets: Sequence[Offset],
            outside: bool) -> np.ndarray:
    """Cells with a set cell at any of ``offsets``; cells beyond the grid read ``outside``."""
    hit = np.zeros(mask.shape, dtype=bool)
    if not offsets:
        return hit
    pad = max(max(abs(dr), abs(dc)) for dr, dc in offsets)
    padded = np.pad(mask, pad, mode="constant", constant_values=outside)
    rows, cols = mask.shape
    for dr, dc in offsets:
        hit |= padded[pad + dr:pad + dr + rows, pad + dc:pad + dc + cols]
    return hit
