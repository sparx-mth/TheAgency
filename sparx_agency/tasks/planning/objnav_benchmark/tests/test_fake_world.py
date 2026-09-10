"""The fake building's geometry, pinned on maps small enough to check by hand.

A fake whose geometry is subtly wrong -- a map read upside down, a radius
measured to cell centres, a geodesic that cuts a corner -- lets every pipeline
test pass for the wrong reason, and the oracle's SR 1.0 then proves nothing.
Each test puts the answer where it can be worked out on paper: the building
(``world``), its text (``ascii_map``), its masks and voxels (``raster``) and
its distances (``geodesics``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.geodesics import (
    geodesic_field,
    legal_moves,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.raster import (
    goal_mask,
    navigable_mask,
    voxel_grid,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

RES = 0.25
SQRT2 = math.sqrt(2.0)


def world_of(rows, cols, walls=(), objects=None):
    """A world of ``rows`` x ``cols`` free cells with the given wall and object cells."""
    wall = np.zeros((rows, cols), dtype=bool)
    for cell in walls:
        wall[cell] = True
    masks = {}
    for category, cells in (objects or {}).items():
        mask = np.zeros((rows, cols), dtype=bool)
        for cell in cells:
            mask[cell] = True
        masks[category] = mask
    return GridWorld(wall, masks)


def mask(north_first):
    """A boolean mask drawn north row first, ``T`` set -- stored with row 0 south."""
    return np.array([[char == "T" for char in row] for row in reversed(north_first)])


# -- parsing --------------------------------------------------------------------

def test_ascii_row_zero_is_the_north_row_so_the_map_reads_like_a_map():
    """A missed flip mirrors every goal north-south and nothing else notices."""
    world = GridWorld.from_ascii(["c..", "...", "..#"], {"c": "chair"})
    assert world.shape == (3, 3)
    assert world.object_mask("chair")[2, 0] and world.wall_mask[0, 2]
    assert world.object_mask("chair").sum() == 1 and world.wall_mask.sum() == 1
    assert world.cell_center(2, 0) == (0.125, 0.625)
    assert world.cell_of(0.125, 0.625) == (2, 0)


def test_several_characters_may_name_one_category_and_an_absent_one_is_left_out():
    """A category with no cell would be a goal nobody can reach."""
    world = GridWorld.from_ascii(["cC.", "..."],
                                 {"c": "chair", "C": "chair", "b": "bed"})
    assert world.categories == ("chair",)
    assert world.object_mask("chair").sum() == 2
    with pytest.raises(ObjNavError, match="has no 'bed'"):
        world.object_mask("bed")


@pytest.mark.parametrize("rows,legend,message", [
    (["...", ".."], {}, "same length"),
    (["..x"], {}, "add it to the legend"),
    (["..."], {"#": "chair"}, "other than"),
    ("...", {}, "sequence of row strings"),
    ([], {}, "at least one"),
], ids=["ragged", "unknown_char", "legend_wall", "bare_string", "empty"])
def test_a_malformed_map_is_refused_with_a_message_that_says_how_to_fix_it(rows, legend, message):
    """A map that parses into something else is a building nobody drew."""
    with pytest.raises(ObjNavError, match=message):
        GridWorld.from_ascii(rows, legend)


def test_a_cell_that_holds_a_wall_and_an_object_is_refused():
    """The object would be unreachable inside the wall, and its goal region empty."""
    walls = np.zeros((2, 2), dtype=bool)
    walls[0, 0] = True
    with pytest.raises(ObjNavError, match="overlaps"):
        GridWorld(walls, {"chair": walls.copy()})


@pytest.mark.parametrize("options,message", [
    ({"resolution_m": 0.0}, "resolution_m"),
    ({"object_height_m": 3.0}, "exceeds wall_height_m"),
    ({"origin_xy": (0.0, float("nan"))}, "origin_xy"),
], ids=["zero_resolution", "object_taller_than_walls", "nan_origin"])
def test_sizes_that_cannot_be_rendered_are_refused(options, message):
    """Each would make the depth camera or the grid geometry silently wrong."""
    with pytest.raises(ObjNavError, match=message):
        GridWorld(np.zeros((2, 2), dtype=bool), {}, **options)


def test_cell_of_floors_so_points_below_the_origin_fall_out_of_bounds():
    """Truncation would put -0.01 m in cell 0 and let an agent stand off the map."""
    world = world_of(3, 3)
    assert world.cell_of(-0.01, -0.01) == (-1, -1)
    assert not world.in_bounds(-1, -1) and world.in_bounds(2, 2)
    assert world.cell_of(0.25, 0.0) == (0, 1)
    moved = GridWorld(np.zeros((2, 2), dtype=bool), {}, origin_xy=(1.0, 2.0))
    assert moved.cell_of(1.0, 2.0) == (0, 0)
    assert moved.cell_center(1, 1) == (1.375, 2.375)


def test_masks_handed_out_cannot_be_written_through():
    """A caller that edits the wall mask in place would move walls under every other reader."""
    world = world_of(3, 3, walls=[(1, 1)])
    with pytest.raises(ValueError):
        world.wall_mask[0, 0] = True
    assert not world.wall_mask[0, 0]


# -- where the agent can stand ----------------------------------------------------

def test_the_agent_radius_is_measured_to_the_square_of_a_blocked_cell():
    """To the centre, a 0.13 m agent could stand 0.005 m from a wall's face."""
    world = world_of(5, 5, walls=[(2, 2)])
    at_013 = navigable_mask(world, 0.13)
    assert not at_013[2, 1] and not at_013[1, 2]    # a face 0.125 m away
    assert at_013[1, 1] and at_013[3, 3]            # a corner 0.177 m away
    at_018 = navigable_mask(world, 0.18)
    assert not at_018[1, 1]
    at_zero = navigable_mask(world, 0.0)
    assert not at_zero[2, 2] and at_zero.sum() == 24


def test_beyond_the_grid_counts_as_blocked():
    """An agent standing on the last cell would otherwise hang half off the map."""
    world = world_of(5, 5)
    assert not navigable_mask(world, 0.13)[0, 2]
    assert navigable_mask(world, 0.13)[2, 2]
    assert navigable_mask(world, 0.1).all()


def test_the_goal_region_is_measured_to_the_objects_square_and_needs_navigable_floor():
    """Habitat's view points surround the object's extent; to its centre the region would be too small."""
    world = world_of(21, 21, objects={"chair": [(10, 10)]})
    goal = goal_mask(world, "chair", 1.0, 0.1)
    assert goal[14, 10] and not goal[15, 10]    # 0.875 m and 1.125 m from the face
    assert goal[13, 13] and not goal[14, 13]    # 0.884 m and 1.075 m from the corner
    assert goal[11, 10] and not goal[10, 10]    # beside the chair, not on it
    assert not goal_mask(world, "chair", 1.0, 0.13)[11, 10]  # too close to stand


# -- how far a goal is ----------------------------------------------------------------

def test_geodesic_distance_counts_straight_and_diagonal_steps_between_cell_centres():
    """``l`` and ``d0`` come from this field; SPL divides by them."""
    world = world_of(3, 5)
    targets = np.zeros((3, 5), dtype=bool)
    targets[0, 0] = True
    field = geodesic_field(world, targets, np.ones((3, 5), dtype=bool))
    assert field[0, 4] == 1.0
    assert field[1, 1] == pytest.approx(RES * SQRT2)
    assert field[2, 2] == pytest.approx(2 * RES * SQRT2)
    assert field[2, 3] == pytest.approx(2 * RES * SQRT2 + RES)


def test_a_diagonal_step_never_squeezes_past_a_blocked_corner():
    """A corner-cutting geodesic is shorter than any path the agent can walk."""
    world = world_of(2, 2)
    navigable = mask(["FT",
                      "TT"])
    targets = np.zeros((2, 2), dtype=bool)
    targets[0, 0] = True
    assert geodesic_field(world, targets, navigable)[1, 1] == 2 * RES
    assert legal_moves(world, 0, 0, navigable) == [(0, 1, RES)]


def test_a_cell_walled_off_from_every_target_is_infinitely_far():
    """Unreachable is inf, never a large number that would average into a mean."""
    world = world_of(3, 5)
    navigable = mask(["TTFTT",
                      "TTFTT",
                      "TTFTT"])
    targets = np.zeros((3, 5), dtype=bool)
    targets[1, 0] = True
    field = geodesic_field(world, targets, navigable)
    assert np.isinf(field[:, 3:]).all() and np.isinf(field[:, 2]).all()
    assert np.isfinite(field[:, :2]).all()


def test_a_target_the_agent_cannot_stand_on_is_refused():
    """A distance to a wall cell would read 0 where no agent can ever be."""
    world = world_of(2, 2)
    with pytest.raises(ObjNavError, match="not navigable"):
        geodesic_field(world, np.ones((2, 2), dtype=bool), mask(["TT", "TF"]))


def test_cell_costs_weigh_each_step_by_the_mean_of_its_two_cells():
    """The oracle's clearance weighting must bend its route, not its notion of a step."""
    world = world_of(1, 3)
    targets = np.array([[True, False, False]])
    field = geodesic_field(world, targets, np.ones((1, 3), dtype=bool),
                           np.array([[1.0, 3.0, 1.0]]))
    assert field.tolist() == [[0.0, 0.5, 1.0]]
    with pytest.raises(ObjNavError, match="at least 1"):
        geodesic_field(world, targets, np.ones((1, 3), dtype=bool),
                       np.array([[1.0, 0.5, 1.0]]))


# -- what the depth camera sees ------------------------------------------------------

def test_the_voxel_grid_has_a_floor_full_height_walls_table_height_objects_and_a_ceiling():
    """A missing floor reads inf below the horizon; a wall at object height hides nothing behind it."""
    world = world_of(3, 3, walls=[(0, 0)], objects={"chair": [(2, 2)]})
    voxels, origin, resolution = voxel_grid(world)
    assert voxels.shape == (12, 3, 3) and voxels.dtype == np.int8
    assert origin == (0.0, 0.0, -RES) and resolution == RES
    assert voxels[0].all() and voxels[11].all()                    # floor, ceiling
    assert voxels[1:11, 0, 0].all()                                # a 2.5 m wall
    assert voxels[1:5, 2, 2].all() and not voxels[5:11, 2, 2].any()  # 0.8 m, rounded up
    assert not voxels[1:11, 1, 1].any()                            # free floor
