"""Tests for slicing a triangle mesh into a 2D occupancy grid."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.geometry_raster.mesh_occupancy import rasterise_mesh_slab

RESOLUTION = 0.1
GRID = dict(resolution=RESOLUTION, origin_x=0.0, origin_y=0.0, width=10, height=10)


def _box(min_corner, max_corner):
    """Return ``(vertices, faces)`` for the closed surface of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = min_corner, max_corner
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # -y
            [2, 3, 7], [2, 7, 6],  # +y
            [1, 2, 6], [1, 6, 5],  # +x
            [3, 0, 4], [3, 4, 7],  # -x
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _cells(grid):
    rows, cols = np.nonzero(grid)
    return set(zip(cols.tolist(), rows.tolist()))


def test_box_walls_rasterise_to_a_hollow_footprint():
    """A closed box is a *surface*: sliced mid-height it leaves its four walls."""
    vertices, faces = _box((0.13, 0.13, 0.0), (0.87, 0.87, 1.0))
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.4, z_max=0.6, **GRID)

    ring = set()
    for index in range(1, 9):
        ring.update({(1, index), (8, index), (index, 1), (index, 8)})
    assert _cells(grid) == ring


def test_horizontal_plate_inside_the_band_fills_solid():
    """A table top has area, so its footprint is filled, not outlined."""
    vertices = np.array(
        [[0.22, 0.22, 0.75], [0.78, 0.22, 0.75], [0.78, 0.78, 0.75], [0.22, 0.78, 0.75]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.3, z_max=2.0, **GRID)
    expected = {(col, row) for col in range(2, 8) for row in range(2, 8)}
    assert _cells(grid) == expected


def test_geometry_below_the_band_is_ignored():
    """The floor must not become an obstacle."""
    vertices, faces = _box((0.13, 0.13, -1.0), (0.87, 0.87, 0.2))
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.3, z_max=2.0, **GRID)
    assert not grid.any()


def test_geometry_above_the_band_is_ignored():
    """So must the ceiling."""
    vertices, faces = _box((0.13, 0.13, 2.5), (0.87, 0.87, 3.0))
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.3, z_max=2.0, **GRID)
    assert not grid.any()


def test_a_wall_only_tall_enough_to_reach_the_band_still_counts():
    """A 0.35 m kerb pokes into a band starting at 0.30 m and must be drawn."""
    vertices, faces = _box((0.13, 0.13, 0.0), (0.87, 0.87, 0.35))
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.3, z_max=2.0, **GRID)
    assert grid.any()


def test_chunking_does_not_change_the_result():
    vertices, faces = _box((0.13, 0.13, 0.0), (0.87, 0.87, 1.0))
    whole = rasterise_mesh_slab(
        vertices, faces, z_min=0.4, z_max=0.6, face_chunk=10_000, **GRID
    )
    piecemeal = rasterise_mesh_slab(
        vertices, faces, z_min=0.4, z_max=0.6, face_chunk=1, **GRID
    )
    np.testing.assert_array_equal(whole, piecemeal)


def test_float32_vertices_are_accepted():
    vertices, faces = _box((0.13, 0.13, 0.0), (0.87, 0.87, 1.0))
    grid = rasterise_mesh_slab(
        vertices.astype(np.float32), faces, z_min=0.4, z_max=0.6, **GRID
    )
    assert grid.any()


def test_meshes_accumulate_into_a_shared_grid():
    left, left_faces = _box((0.03, 0.03, 0.0), (0.17, 0.17, 1.0))
    right, right_faces = _box((0.83, 0.83, 0.0), (0.97, 0.97, 1.0))
    grid = rasterise_mesh_slab(left, left_faces, z_min=0.4, z_max=0.6, **GRID)
    rasterise_mesh_slab(right, right_faces, z_min=0.4, z_max=0.6, out=grid, **GRID)
    assert grid[0, 0] and grid[9, 9]


def test_geometry_outside_the_grid_is_clipped_away():
    vertices, faces = _box((20.0, 20.0, 0.0), (21.0, 21.0, 1.0))
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.4, z_max=0.6, **GRID)
    assert not grid.any()


def test_empty_mesh_returns_an_empty_grid():
    grid = rasterise_mesh_slab(
        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), z_min=0.3, z_max=2.0, **GRID
    )
    assert grid.shape == (10, 10)
    assert not grid.any()


def test_a_long_thin_diagonal_wall_is_unbroken():
    """The failure that matters: a wall the map must not leave a gap in."""
    grid_spec = dict(resolution=0.05, origin_x=0.0, origin_y=0.0, width=80, height=60)
    start = np.array([0.20, 0.20])
    end = np.array([3.70, 2.60])
    direction = end - start
    normal = np.array([-direction[1], direction[0]])
    normal = 0.006 * normal / np.linalg.norm(normal)
    lower, upper = 0.0, 2.5
    corners = [start + normal, end + normal, end - normal, start - normal]
    vertices = np.array(
        [[c[0], c[1], lower] for c in corners] + [[c[0], c[1], upper] for c in corners]
    )
    faces = np.array(
        [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
         [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.int64,
    )
    grid = rasterise_mesh_slab(vertices, faces, z_min=0.3, z_max=2.0, **grid_spec)
    for step in np.linspace(0.0, 1.0, 2001):
        point = start + step * direction
        col = int(np.floor(point[0] / 0.05))
        row = int(np.floor(point[1] / 0.05))
        assert grid[row, col], "hole at %.3f, %.3f" % (point[0], point[1])


def test_bad_inputs_raise():
    vertices, faces = _box((0.13, 0.13, 0.0), (0.87, 0.87, 1.0))
    with pytest.raises(ValueError):
        rasterise_mesh_slab(vertices[:, :2], faces, z_min=0.4, z_max=0.6, **GRID)
    with pytest.raises(ValueError):
        rasterise_mesh_slab(vertices, faces.astype(np.float64), z_min=0.4, z_max=0.6, **GRID)
    with pytest.raises(ValueError):
        rasterise_mesh_slab(vertices, faces + 100, z_min=0.4, z_max=0.6, **GRID)
    with pytest.raises(ValueError):
        rasterise_mesh_slab(vertices, faces, z_min=2.0, z_max=0.4, **GRID)
    with pytest.raises(ValueError):
        rasterise_mesh_slab(
            vertices, faces, z_min=0.4, z_max=0.6, out=np.zeros((3, 3), dtype=bool), **GRID
        )
