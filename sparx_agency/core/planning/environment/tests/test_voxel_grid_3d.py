"""The 3D ground-truth map, its persistence, and the 2D maps derived from it.

The property that matters most: UNKNOWN is not FREE. A survey reaches a building
by flooding out from a point inside it, so everything past the walls is
unobserved -- and a planner that treats that as open air routes the aircraft out
through one.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import (
    VoxelGrid3D, indoor_mask, landable_mask, load_voxel_grid,
    project_to_occupancy_2d, restrict_to_indoor, save_voxel_grid,
)
from sparx_agency.core.planning.environment.voxel_grid_3d import FREE, OCCUPIED, UNKNOWN

RES = 0.1


def _room() -> VoxelGrid3D:
    """A 2 x 3 m room, 1.5 m tall, with a 0.7 m desk in the middle of it.

    Indices are ``[z, y, x]``; the grid spans z in [-0.3, 1.2).
    """
    voxels = np.full((15, 30, 20), FREE, dtype=np.int8)
    voxels[0, :, :] = OCCUPIED          # floor at z = -0.3
    voxels[:, 0, :] = OCCUPIED          # a wall
    voxels[:, :, 0] = OCCUPIED          # another wall
    voxels[:11, 14:18, 8:12] = OCCUPIED  # a desk, from the floor to z = 0.8
    voxels[:, 25:, :] = UNKNOWN         # never observed: outside the building
    return VoxelGrid3D(voxels, RES, (-1.0, -1.5, -0.3))


def test_shape_and_geometry():
    grid = _room()
    assert (grid.width, grid.height, grid.depth) == (20, 30, 15)
    assert grid.world_to_grid(-1.0, -1.5, -0.3) == (0, 0, 0)
    # grid_to_world returns the voxel centre, so it round-trips to the same cell.
    x, y, z = grid.grid_to_world(5, 7, 3)
    assert grid.world_to_grid(x, y, z) == (5, 7, 3)


def test_unknown_is_not_free():
    """The single most important property: a planner may not route through it."""
    grid = _room()
    unknown_cell = grid.world_to_grid(-0.5, 1.2, 0.5)
    assert grid.voxels[unknown_cell[2], unknown_cell[1], unknown_cell[0]] == UNKNOWN
    assert not grid.is_free(*unknown_cell)


def test_out_of_bounds_is_not_free():
    grid = _room()
    assert not grid.is_free(-1, 0, 0)
    assert not grid.is_free(0, 0, 999)


def test_occupied_points_are_voxel_centres_in_world_coordinates():
    voxels = np.full((2, 2, 2), FREE, dtype=np.int8)
    voxels[1, 0, 1] = OCCUPIED
    grid = VoxelGrid3D(voxels, 0.5, (10.0, 20.0, 30.0))

    points = grid.occupied_points()

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [10.0 + 0.75, 20.0 + 0.25, 30.0 + 0.75])


def test_projection_marks_a_cell_occupied_if_any_voxel_in_the_slab_is():
    """Ground truth has no noise to average away: one voxel is enough."""
    grid = _room()
    # The desk top is at z = 0.8; a slab there must see it.
    sliced = project_to_occupancy_2d(grid, altitude_m=0.7, half_height_m=0.05)
    desk_x, desk_y, _ = grid.world_to_grid(-0.15, 0.1, 0.7)
    assert sliced.is_occupied(desk_x, desk_y)


def test_projection_at_a_higher_slab_flies_over_the_desk():
    grid = _room()
    high = project_to_occupancy_2d(grid, altitude_m=1.0, half_height_m=0.05)
    desk_x, desk_y, _ = grid.world_to_grid(-0.15, 0.1, 1.0)
    assert high.is_free(desk_x, desk_y), "the desk stops at 0.8 m"


def test_projection_keeps_unobserved_space_unknown():
    grid = _room()
    sliced = project_to_occupancy_2d(grid, altitude_m=0.5, half_height_m=0.2)
    outside_x, outside_y, _ = grid.world_to_grid(-0.5, 1.2, 0.5)
    assert sliced.is_unknown(outside_x, outside_y)
    assert not sliced.is_free(outside_x, outside_y)


def test_projection_is_co_registered_with_the_voxel_grid():
    grid = _room()
    sliced = project_to_occupancy_2d(grid, altitude_m=0.5, half_height_m=0.2)
    assert sliced.resolution == grid.resolution
    assert sliced.origin_x == grid.origin_x
    assert sliced.origin_y == grid.origin_y
    assert (sliced.width, sliced.height) == (grid.width, grid.height)


def test_a_slab_outside_the_grid_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="outside the grid"):
        project_to_occupancy_2d(_room(), altitude_m=50.0, half_height_m=0.2)


def test_landable_excludes_cells_with_furniture_underneath():
    """Flyable at cruise height, and not somewhere the aircraft can be put down."""
    grid = _room()
    landable = landable_mask(grid, altitude_m=1.0)
    high = project_to_occupancy_2d(grid, altitude_m=1.0, half_height_m=0.05)

    desk_x, desk_y, _ = grid.world_to_grid(-0.15, 0.1, 1.0)
    assert high.is_free(desk_x, desk_y), "the aircraft can fly over the desk"
    assert not landable[desk_y, desk_x], "and cannot land on it"


def test_landable_includes_open_floor():
    grid = _room()
    landable = landable_mask(grid, altitude_m=1.0)
    open_x, open_y, _ = grid.world_to_grid(0.5, -0.5, 1.0)
    assert landable[open_y, open_x]


def test_round_trip_preserves_the_grid_and_its_metadata(tmp_path):
    grid = _room()
    path = save_voxel_grid(tmp_path / "scene.npz", grid,
                           metadata={"scene": "office", "resolution_m": RES})

    loaded, metadata = load_voxel_grid(path)

    np.testing.assert_array_equal(loaded.voxels, grid.voxels)
    assert loaded.resolution == pytest.approx(grid.resolution)
    assert (loaded.origin_x, loaded.origin_y, loaded.origin_z) == pytest.approx(
        (grid.origin_x, grid.origin_y, grid.origin_z))
    assert metadata["scene"] == "office"


def test_round_trip_preserves_world_coordinates(tmp_path):
    grid = _room()
    loaded, _ = load_voxel_grid(save_voxel_grid(tmp_path / "g.npz", grid))
    for cell in [(0, 0, 0), (5, 9, 3), (19, 29, 14)]:
        assert loaded.grid_to_world(*cell) == pytest.approx(grid.grid_to_world(*cell))


def test_a_missing_voxel_map_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_voxel_grid(tmp_path / "nope.npz")


def test_a_non_3d_array_is_rejected():
    with pytest.raises(ValueError, match="3D"):
        VoxelGrid3D(np.zeros((4, 4)), 0.1, (0, 0, 0))


def test_a_degenerate_resolution_is_rejected():
    with pytest.raises(ValueError, match="resolution"):
        VoxelGrid3D(np.zeros((2, 2, 2)), 0.0, (0, 0, 0))


def test_world_clearance_is_zero_until_a_field_is_computed():
    """Part of the VoxelMap3D protocol; a building-sized 3D EDT is not free."""
    grid = _room()
    assert grid.clearance is None
    assert grid.world_clearance(0.0, 0.0, 0.5) == 0.0


def test_it_satisfies_the_voxel_map_3d_protocol():
    """The 3D planners are typed against this; a missing member fails at runtime.

    Checked member by member rather than with isinstance, because ``VoxelMap3D``
    is not ``@runtime_checkable`` and making it so to satisfy a test would be
    changing the contract to fit the check.
    """
    grid = _room()
    for name in ("origin_x", "origin_y", "origin_z", "width", "height", "depth",
                 "resolution", "frame_id", "clearance"):
        assert hasattr(grid, name), f"missing attribute {name}"
    for name in ("world_to_grid", "is_free", "world_clearance"):
        assert callable(getattr(grid, name, None)), f"missing method {name}"


def _building_in_a_field() -> VoxelGrid3D:
    """A room with a ceiling, standing on open ground that has none."""
    voxels = np.full((60, 40, 40), FREE, dtype=np.int8)
    voxels[0, :, :] = OCCUPIED              # the ground, everywhere
    voxels[35, 5:20, 5:20] = OCCUPIED       # a roof, over part of it only
    return VoxelGrid3D(voxels, RES, (0.0, 0.0, 0.0))


def test_indoors_is_where_there_is_something_overhead():
    """The flood fill escapes over the roof; only a ceiling test separates them."""
    grid = _building_in_a_field()
    indoors = indoor_mask(grid, altitude_m=1.5, max_ceiling_m=4.0)

    under_roof = grid.world_to_grid(1.0, 1.0, 1.5)
    open_sky = grid.world_to_grid(3.0, 3.0, 1.5)
    assert indoors[under_roof[1], under_roof[0]]
    assert not indoors[open_sky[1], open_sky[0]]


def test_restricting_to_indoors_blanks_the_field_but_keeps_the_room():
    grid = restrict_to_indoor(_building_in_a_field(), altitude_m=1.5, max_ceiling_m=4.0)

    under_roof = grid.world_to_grid(1.0, 1.0, 1.5)
    open_sky = grid.world_to_grid(3.0, 3.0, 1.5)
    assert grid.is_free(*under_roof)
    assert not grid.is_free(*open_sky)
    assert grid.voxels[open_sky[2], open_sky[1], open_sky[0]] == UNKNOWN


def test_a_projection_of_a_restricted_grid_calls_outdoors_unknown():
    grid = restrict_to_indoor(_building_in_a_field(), altitude_m=1.5, max_ceiling_m=4.0)
    sliced = project_to_occupancy_2d(grid, altitude_m=1.5, half_height_m=0.2)

    open_sky = grid.world_to_grid(3.0, 3.0, 1.5)
    assert sliced.is_unknown(open_sky[0], open_sky[1])


def test_the_floor_slab_is_not_mistaken_for_furniture():
    """Too small a floor clearance made every single cell unlandable."""
    voxels = np.full((30, 10, 10), FREE, dtype=np.int8)
    voxels[0:3, :, :] = OCCUPIED           # a floor slab 0.3 m thick, from z = 0
    grid = VoxelGrid3D(voxels, RES, (0.0, 0.0, 0.0))

    assert not landable_mask(grid, 1.5, floor_z_m=0.0, floor_clearance_m=0.05).any()
    assert landable_mask(grid, 1.5, floor_z_m=0.0, floor_clearance_m=0.35).all()
