"""Tests for rasterising convex polygons into a boolean grid."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from sparx_agency.core.mapping.geometry_raster.grid_spec import GridSpec
from sparx_agency.core.mapping.geometry_raster.polygon_raster import (
    rasterise_polygon,
    rasterise_polygons,
)


def _cells(grid):
    """Return the set of ``(col, row)`` cells that are True."""
    rows, cols = np.nonzero(grid)
    return set(zip(cols.tolist(), rows.tolist()))


def test_axis_aligned_square_covers_exactly_the_expected_cells():
    """A square well inside its cells must fill exactly those cells."""
    square = np.array([[0.22, 0.22], [0.78, 0.22], [0.78, 0.78], [0.22, 0.78]])
    grid = rasterise_polygon(square, 0.1, 0.0, 0.0, 10, 10)
    expected = {(col, row) for col in range(2, 8) for row in range(2, 8)}
    assert _cells(grid) == expected


def test_row_zero_is_minimum_y():
    """A strip low in y must land in low rows, not high ones."""
    strip = np.array([[0.05, 0.05], [0.95, 0.05], [0.95, 0.15], [0.05, 0.15]])
    grid = rasterise_polygon(strip, 0.1, 0.0, 0.0, 10, 10)
    assert grid[0].any()
    assert not grid[5:].any()


def test_clockwise_winding_gives_the_same_result():
    square = np.array([[0.22, 0.22], [0.78, 0.22], [0.78, 0.78], [0.22, 0.78]])
    forward = rasterise_polygon(square, 0.1, 0.0, 0.0, 10, 10)
    backward = rasterise_polygon(square[::-1].copy(), 0.1, 0.0, 0.0, 10, 10)
    np.testing.assert_array_equal(forward, backward)


def test_thin_diagonal_wall_has_no_holes_along_its_length():
    """The wall a planner must not fly through: 2 cm thick, at 27 degrees."""
    resolution = 0.05
    start = np.array([0.30, 0.30])
    end = np.array([2.60, 1.50])
    direction = end - start
    normal = np.array([-direction[1], direction[0]])
    normal = 0.01 * normal / np.linalg.norm(normal)
    wall = np.array([start + normal, end + normal, end - normal, start - normal])

    grid = rasterise_polygon(wall, resolution, 0.0, 0.0, 60, 40)

    # Every point along the wall's centre line must be in an occupied cell.
    for step in np.linspace(0.0, 1.0, 2001):
        point = start + step * direction
        col = int(np.floor(point[0] / resolution))
        row = int(np.floor(point[1] / resolution))
        assert grid[row, col], "hole at %.3f, %.3f" % (point[0], point[1])

    # And the occupied cells must form one 4-connected run, so no planner can
    # slip between them diagonally.
    assert _is_four_connected(grid)


def test_degenerate_polygon_draws_a_line_not_a_filled_box():
    """Three collinear points are a segment; filling their bbox would be a lie."""
    segment = np.array([[0.05, 0.05], [0.50, 0.50], [0.95, 0.95]])
    grid = rasterise_polygon(segment, 0.1, 0.0, 0.0, 10, 10)
    assert grid[0, 0] and grid[9, 9]
    assert not grid[0, 9]
    assert not grid[9, 0]
    assert int(grid.sum()) < 40


def test_polygon_outside_the_grid_draws_nothing():
    far = np.array([[50.0, 50.0], [51.0, 50.0], [51.0, 51.0]])
    grid = rasterise_polygon(far, 0.1, 0.0, 0.0, 10, 10)
    assert not grid.any()


def test_polygon_partly_outside_is_clipped_to_the_grid():
    straddling = np.array([[-0.5, 0.22], [0.35, 0.22], [0.35, 0.78], [-0.5, 0.78]])
    grid = rasterise_polygon(straddling, 0.1, 0.0, 0.0, 10, 10)
    assert grid[3, 0]
    assert not grid[3, 5]


def test_existing_grid_is_accumulated_into():
    first = np.array([[0.22, 0.22], [0.38, 0.22], [0.38, 0.38], [0.22, 0.38]])
    second = np.array([[0.62, 0.62], [0.78, 0.62], [0.78, 0.78], [0.62, 0.78]])
    grid = rasterise_polygon(first, 0.1, 0.0, 0.0, 10, 10)
    rasterise_polygon(second, 0.1, 0.0, 0.0, 10, 10, grid=grid)
    assert grid[2, 2] and grid[7, 7]


def test_three_dimensional_vertices_use_only_xy():
    flat = np.array([[0.22, 0.22, 1.7], [0.78, 0.22, 1.7], [0.78, 0.78, 1.7]])
    grid = rasterise_polygon(flat, 0.1, 0.0, 0.0, 10, 10)
    assert grid[2, 7]


def test_bad_polygon_shape_raises():
    with pytest.raises(ValueError):
        rasterise_polygon(np.zeros((3, 4)), 0.1, 0.0, 0.0, 10, 10)


def test_non_positive_resolution_raises():
    with pytest.raises(ValueError):
        rasterise_polygon(np.zeros((3, 2)), 0.0, 0.0, 0.0, 10, 10)


def _is_four_connected(grid):
    """True when every occupied cell is reachable from any other 4-connectedly."""
    occupied = set(_cells(grid))
    if not occupied:
        return False
    start = next(iter(occupied))
    seen = {start}
    stack = [start]
    while stack:
        col, row = stack.pop()
        for step_col, step_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (col + step_col, row + step_row)
            if neighbour in occupied and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == occupied


def test_a_grid_of_the_wrong_shape_raises():
    """Silently drawing into the corner of someone else's map is worse."""
    square = np.array([[0.22, 0.22], [0.78, 0.22], [0.78, 0.78], [0.22, 0.78]])
    with pytest.raises(ValueError):
        rasterise_polygon(square, 0.1, 0.0, 0.0, 10, 10, grid=np.zeros((5, 5), bool))


def test_an_empty_polygon_beside_a_real_one_does_not_warn():
    """The batch API's padding row: its +/-inf bbox must never reach a cast."""
    spec = GridSpec(resolution=0.1, origin_x=0.0, origin_y=0.0, width=10, height=10)
    grid = spec.empty()
    polygons = np.zeros((2, 4, 3))
    polygons[0, :, :2] = [[0.22, 0.22], [0.78, 0.22], [0.78, 0.78], [0.22, 0.78]]
    counts = np.array([4, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rasterise_polygons(grid, polygons, counts, spec)

    assert grid[2:8, 2:8].all()
    assert int(grid.sum()) == 36
