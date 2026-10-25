"""A lower-resolution copy of an occupancy grid, for planning that does not need the fine cells."""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams

UNKNOWN_RULES = ("any", "majority", "all")


def downsample_world(world, factor, unknown_rule="any", agent_xy=None):
    """Block-reduce ``world`` by ``factor``: occupied if any cell is, else unknown by ``unknown_rule``, else free.

    ``any``: one unknown cell makes the block unknown; ``majority``: more than half; ``all``: only a wholly unseen block.
    The block under ``agent_xy`` is free: a planner cannot start from a cell the robot stands in but a wall pixel shares.
    """
    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer")
    if unknown_rule not in UNKNOWN_RULES:
        raise ValueError("unknown_rule must be one of %s" % ", ".join(UNKNOWN_RULES))
    if factor == 1:
        return world
    values, grid = world.values, world.grid
    height, width = grid.shape
    padded_h, padded_w = -(-height // factor) * factor, -(-width // factor) * factor
    padded = np.full((padded_h, padded_w), values.unknown, dtype=grid.dtype)
    padded[:height, :width] = grid
    blocks = padded.reshape(padded_h // factor, factor, padded_w // factor, factor)
    occupied = (blocks == values.occupied).any(axis=(1, 3))
    unknown_cells = (blocks == values.unknown).sum(axis=(1, 3))
    unknown = {"any": unknown_cells > 0, "majority": 2 * unknown_cells > factor * factor,
               "all": unknown_cells == factor * factor}[unknown_rule]
    coarse = np.full(occupied.shape, values.free, dtype=grid.dtype)
    coarse[unknown] = values.unknown
    coarse[occupied] = values.occupied
    params = world.params
    result = OccupancyGrid2D(coarse, OccupancyGrid2DParams(round(params.resolution * factor, 9), params.origin_x,
                                                           params.origin_y, params.frame_id), values=values)
    if agent_xy is not None:
        gx, gy = result.world_to_grid(*agent_xy)
        if result.in_bounds(gx, gy):
            result.grid[gy, gx] = values.free
    return result
