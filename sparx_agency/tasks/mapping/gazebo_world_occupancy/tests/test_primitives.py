"""Tests for tessellating SDF primitive shapes."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.mapping.gazebo_world_occupancy import primitives


@pytest.mark.parametrize(
    "vertices, faces",
    [
        primitives.box_mesh([1.0, 2.0, 3.0]),
        primitives.cylinder_mesh(0.5, 2.0),
        primitives.sphere_mesh(1.0),
    ],
)
def test_every_tessellation_is_a_valid_index_mesh(vertices, faces):
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert faces.min() >= 0
    assert faces.max() < vertices.shape[0]
    assert np.issubdtype(faces.dtype, np.integer)


def test_box_is_centred_on_its_link_origin_with_the_given_extents():
    vertices, _faces = primitives.box_mesh([1.0, 2.0, 3.0])
    np.testing.assert_allclose(vertices.min(axis=0), [-0.5, -1.0, -1.5])
    np.testing.assert_allclose(vertices.max(axis=0), [0.5, 1.0, 1.5])


def test_cylinder_spans_its_length_and_circumscribes_its_radius():
    """Facets outside the true circle: a ground-truth obstacle is never smaller."""
    vertices, _faces = primitives.cylinder_mesh(0.5, 2.0)
    assert vertices[:, 2].min() == pytest.approx(-1.0)
    assert vertices[:, 2].max() == pytest.approx(1.0)
    radii = np.hypot(vertices[:, 0], vertices[:, 1])
    assert radii.max() >= 0.5
    assert radii.max() < 0.5 * 1.01


def test_sphere_circumscribes_its_radius():
    vertices, _faces = primitives.sphere_mesh(1.0)
    radii = np.linalg.norm(vertices, axis=1)
    assert radii.max() >= 1.0
    assert radii.max() < 1.02


def test_degenerate_dimensions_raise():
    with pytest.raises(ValueError):
        primitives.box_mesh([1.0, 2.0])
    with pytest.raises(ValueError):
        primitives.cylinder_mesh(0.0, 1.0)
    with pytest.raises(ValueError):
        primitives.sphere_mesh(-1.0)
