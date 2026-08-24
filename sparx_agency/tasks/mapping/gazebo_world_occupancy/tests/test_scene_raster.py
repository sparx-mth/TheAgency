"""Tests for sizing the grid a scene is rasterised into."""
from __future__ import annotations

import numpy as np

from sparx_agency.core.mapping.geometry_raster import rasterise_polygons
from sparx_agency.tasks.mapping.gazebo_world_occupancy.scene_raster import (
    SceneExtent,
    grid_spec_for,
)

RESOLUTION = 0.05


def _extent(max_x, max_y, min_x=0.0, min_y=0.0):
    """A SceneExtent with the margin already applied, as the CLI passes it."""
    return SceneExtent(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        instance_count=1,
        triangle_count=1,
    )


def test_an_extent_that_divides_exactly_still_has_its_far_edge_inside():
    """5.0 m at 5 cm is 100 whole cells, and the wall at 5.0 needs the 101st."""
    spec = grid_spec_for(_extent(5.0, 5.0), RESOLUTION)
    assert (spec.width, spec.height) == (101, 101)

    corner = spec.to_cell_coords(np.array([[5.0, 5.0]]))
    assert int(np.floor(corner[0, 0])) < spec.width
    assert int(np.floor(corner[0, 1])) < spec.height


def test_a_wall_on_the_far_edge_is_drawn():
    """With --margin 0 the outermost wall lands exactly on the extent."""
    spec = grid_spec_for(_extent(5.0, 5.0), RESOLUTION)
    grid = spec.empty()
    wall = np.array([[[5.0, 1.0, 0.0], [5.0, 4.0, 0.0], [5.0, 4.0, 0.0]]])

    rasterise_polygons(grid, wall, np.array([3]), spec)

    assert grid[:, spec.width - 1].any()


def test_a_partial_last_cell_is_counted_once_not_twice():
    """The far edge inside a cell must not add a second one."""
    spec = grid_spec_for(_extent(5.02, 5.02), RESOLUTION)
    assert (spec.width, spec.height) == (101, 101)


def test_the_origin_is_snapped_below_the_extent():
    spec = grid_spec_for(_extent(2.0, 2.0, min_x=-13.58, min_y=-36.07), RESOLUTION)
    assert spec.origin_x == -13.60
    assert spec.origin_y == -36.10
