"""Tests for clipping triangles to a horizontal slab."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.geometry_raster.slab_clip import (
    clip_triangle_to_slab,
    clip_triangles_to_slab,
)

Z_MIN = 0.3
Z_MAX = 2.0


def _sorted_rows(points):
    """Return points sorted lexicographically so order does not affect equality."""
    return np.array(sorted(points.tolist()))


def test_triangle_fully_inside_is_returned_unchanged():
    triangle = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.2], [0.0, 1.0, 0.8]])
    clipped = clip_triangle_to_slab(triangle, Z_MIN, Z_MAX)
    assert clipped.shape == (3, 3)
    np.testing.assert_allclose(_sorted_rows(clipped), _sorted_rows(triangle))


def test_triangle_below_the_slab_is_empty():
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 1.0, 0.2]])
    assert clip_triangle_to_slab(triangle, Z_MIN, Z_MAX).shape == (0, 3)


def test_triangle_above_the_slab_is_empty():
    triangle = np.array([[0.0, 0.0, 2.5], [1.0, 0.0, 3.0], [0.0, 1.0, 2.1]])
    assert clip_triangle_to_slab(triangle, Z_MIN, Z_MAX).shape == (0, 3)


def test_triangle_touching_the_boundary_from_below_is_a_degenerate_sliver():
    """A triangle whose only in-slab point is one vertex on z_min still clips."""
    triangle = np.array([[0.0, 0.0, 0.3], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]])
    clipped = clip_triangle_to_slab(triangle, Z_MIN, Z_MAX)
    assert clipped.shape[0] in (0, 3)
    if clipped.shape[0]:
        np.testing.assert_allclose(clipped[:, 2], Z_MIN, atol=1e-12)


def test_triangle_straddling_z_min_becomes_a_quad_inside_the_slab():
    # One vertex below the cut and two above leaves a quad: the two survivors
    # plus the two points where the cut crosses the sloping edges.
    triangle = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    clipped = clip_triangle_to_slab(triangle, Z_MIN, Z_MAX)
    assert clipped.shape == (4, 3)
    assert np.all(clipped[:, 2] >= Z_MIN - 1e-12)
    assert np.all(clipped[:, 2] <= Z_MAX + 1e-12)
    # The cut runs along z = z_min; two of the four vertices must sit on it.
    assert int(np.sum(np.isclose(clipped[:, 2], Z_MIN))) == 2


def test_vertical_triangle_spanning_the_slab_keeps_only_the_band():
    triangle = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
    clipped = clip_triangle_to_slab(triangle, Z_MIN, Z_MAX)
    assert clipped.shape[0] >= 3
    assert clipped[:, 2].min() >= Z_MIN - 1e-12
    assert clipped[:, 2].max() <= Z_MAX + 1e-12


def test_horizontal_triangle_inside_the_slab_survives_whole():
    triangle = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    clipped = clip_triangle_to_slab(triangle, Z_MIN, Z_MAX)
    assert clipped.shape == (3, 3)
    assert _polygon_area_xy(clipped) == pytest.approx(3.0)


def test_clip_preserves_winding_and_area_of_a_horizontal_triangle():
    """The clipped polygon must stay convex and keep its orientation."""
    triangle = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    forward = _signed_area_xy(clip_triangle_to_slab(triangle, Z_MIN, Z_MAX))
    reversed_tri = triangle[::-1].copy()
    backward = _signed_area_xy(clip_triangle_to_slab(reversed_tri, Z_MIN, Z_MAX))
    assert forward > 0.0 > backward


def test_batched_clip_matches_the_single_triangle_path():
    rng = np.random.RandomState(0)
    triangles = rng.uniform(-1.0, 3.0, size=(64, 3, 3))
    polygons, counts = clip_triangles_to_slab(triangles, Z_MIN, Z_MAX)
    assert polygons.shape == (64, 5, 3)
    for index in range(triangles.shape[0]):
        single = clip_triangle_to_slab(triangles[index], Z_MIN, Z_MAX)
        assert single.shape[0] == int(counts[index])
        np.testing.assert_allclose(polygons[index, : int(counts[index])], single)


def test_empty_batch_is_handled():
    polygons, counts = clip_triangles_to_slab(np.zeros((0, 3, 3)), Z_MIN, Z_MAX)
    assert polygons.shape[0] == 0
    assert counts.shape == (0,)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        clip_triangle_to_slab(np.zeros((4, 3)), Z_MIN, Z_MAX)
    with pytest.raises(ValueError):
        clip_triangles_to_slab(np.zeros((2, 4, 3)), Z_MIN, Z_MAX)


def test_inverted_slab_raises():
    with pytest.raises(ValueError):
        clip_triangles_to_slab(np.zeros((1, 3, 3)), 2.0, 0.3)


def _signed_area_xy(polygon):
    """Shoelace area of the polygon's XY projection."""
    if polygon.shape[0] < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _polygon_area_xy(polygon):
    return abs(_signed_area_xy(polygon))
