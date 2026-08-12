"""Placement picking must respect spacing and keepout, or a campaign gets a
duplicate obstacle wedged into the spawn point or stacked on another one.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import occupancy_from_mask
from sparx_agency.robots.PEGASUS.adapters import obstacle_placement


def _open_grid(size: int = 40, resolution: float = 0.5):
    occupied = np.zeros((size, size), dtype=bool)
    landable = np.ones((size, size), dtype=bool)
    origin = -size * resolution / 2.0
    grid = occupancy_from_mask(occupied, resolution, origin, origin, frame_id="world")
    return grid, landable


def test_respects_minimum_spacing():
    grid, landable = _open_grid()
    rng = np.random.default_rng(0)
    placements = obstacle_placement.sample_placements(
        grid, landable, count=15, min_spacing_m=2.0, rng=rng)

    assert len(placements) > 1
    xy = np.array([(p.x, p.y) for p in placements])
    for i in range(len(xy)):
        others = np.delete(xy, i, axis=0)
        distances = np.hypot(others[:, 0] - xy[i, 0], others[:, 1] - xy[i, 1])
        assert np.all(distances >= 2.0)


def test_respects_keepout():
    grid, landable = _open_grid()
    rng = np.random.default_rng(1)
    placements = obstacle_placement.sample_placements(
        grid, landable, count=30, min_spacing_m=0.5,
        keepout=[(0.0, 0.0, 3.0)], rng=rng)

    for p in placements:
        assert np.hypot(p.x, p.y) >= 3.0


def test_never_picks_occupied_or_unlandable_cells():
    size = 20
    resolution = 0.5
    occupied = np.zeros((size, size), dtype=bool)
    occupied[:, :size // 2] = True  # left half is obstacle
    landable = np.ones((size, size), dtype=bool)
    landable[size // 2:, :] = False  # bottom half not landable
    origin = -size * resolution / 2.0
    grid = occupancy_from_mask(occupied, resolution, origin, origin, frame_id="world")

    placements = obstacle_placement.sample_placements(
        grid, landable, count=50, min_spacing_m=0.1, rng=np.random.default_rng(2))

    for p in placements:
        gx, gy = grid.world_to_grid(p.x, p.y)
        assert grid.is_free(gx, gy)
        assert landable[gy, gx]


def test_never_fully_blocks_the_only_passage():
    """A duplicate landing in the one corridor between two rooms is exactly
    the bug this guards against: nothing checked reachability, only spacing
    between obstacles, and a "correctly" spaced obstacle sealed a hallway.
    """
    from scipy import ndimage

    size = 24
    resolution = 0.5
    occupied = np.ones((size, size), dtype=bool)
    occupied[2:10, 2:10] = False    # room A
    occupied[2:10, 14:22] = False   # room B
    occupied[5:7, 10:14] = False    # the only corridor between them, 1 m wide
    landable = ~occupied
    origin = -size * resolution / 2.0
    grid = occupancy_from_mask(occupied, resolution, origin, origin, frame_id="world")

    spawn_x, spawn_y = grid.grid_to_world(5, 5)  # inside room A

    placements = obstacle_placement.sample_placements(
        grid, landable, count=60, min_spacing_m=0.3,
        keepout=[(spawn_x, spawn_y, 1.2)], obstacle_radius_m=0.4,
        rng=np.random.default_rng(7))

    assert len(placements) > 0

    # Stamp each placement's footprint the same shape the implementation
    # itself checks against (a disk, not a bounding square -- a square is
    # stricter along the diagonals and flagged a false block here once).
    free = grid.grid == grid.values.free
    offsets = obstacle_placement._disk_offsets(round(0.4 / resolution))
    for p in placements:
        gx, gy = grid.world_to_grid(p.x, p.y)
        fy, fx = gy + offsets[:, 0], gx + offsets[:, 1]
        in_bounds = (fy >= 0) & (fy < free.shape[0]) & (fx >= 0) & (fx < free.shape[1])
        free[fy[in_bounds], fx[in_bounds]] = False

    labels, _ = ndimage.label(free, structure=np.ones((3, 3), dtype=bool))
    spawn_gx, spawn_gy = grid.world_to_grid(spawn_x, spawn_y)
    assert labels[spawn_gy, spawn_gx] != 0
    # Room B occupies grid columns 14-21 -- checking whether *any* of them
    # still shares spawn's label (rather than one fixed reference cell) means
    # this does not spuriously fail just because a legitimate placement
    # happened to land on whatever single coordinate was chosen to represent
    # "room B".
    room_b_still_reachable = np.any(labels[2:10, 14:22] == labels[spawn_gy, spawn_gx])
    assert room_b_still_reachable, (
        "room B is no longer reachable from room A -- a placement sealed the corridor"
    )


def test_raises_on_no_candidates():
    size = 10
    grid, _ = _open_grid(size=size)
    landable = np.zeros((size, size), dtype=bool)

    with pytest.raises(ValueError):
        obstacle_placement.sample_placements(grid, landable, count=1, min_spacing_m=0.5)


def test_raises_on_nonpositive_count():
    grid, landable = _open_grid()
    with pytest.raises(ValueError):
        obstacle_placement.sample_placements(grid, landable, count=0, min_spacing_m=0.5)
