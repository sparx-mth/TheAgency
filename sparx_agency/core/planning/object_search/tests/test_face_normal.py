"""``perception/face_normal.py``: the facing normal of a planar target from the points a sighting measured on it."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.perception.face_normal import MIN_POINTS, face_normal


def plane(normal, centre, width_m, height_m, n=400, thickness_m=0.0, seed=0):
    """``n`` points on a standing plane with unit ``normal`` through ``centre``, ``width_m`` across, ``height_m`` tall."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    across = np.cross(normal, (0.0, 0.0, 1.0))
    if np.linalg.norm(across) < 1e-6:                 # a horizontal plane: any in-plane axis will do
        across = np.array([1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    up = np.cross(across, normal)
    u = rng.uniform(-width_m / 2, width_m / 2, n)
    v = rng.uniform(-height_m / 2, height_m / 2, n)
    w = rng.normal(0.0, thickness_m, n) if thickness_m else np.zeros(n)
    return np.asarray(centre, dtype=float) + np.outer(u, across) + np.outer(v, up) + np.outer(w, normal)


def test_a_picture_on_a_wall_faces_the_camera():
    points = plane((0.0, -1.0, 0.0), (2.0, 3.0, 1.5), width_m=0.6, height_m=0.5)
    (nx, ny), thickness = face_normal(points, camera_xy=(2.0, 0.0))
    assert nx == pytest.approx(0.0, abs=1e-6) and ny == pytest.approx(-1.0, abs=1e-6) and thickness < 1e-6
    (nx, ny), _ = face_normal(points, camera_xy=(2.0, 6.0))
    assert ny == pytest.approx(1.0), "the normal points to the camera's side of the plane"
    assert math.hypot(nx, ny) == pytest.approx(1.0)


def test_a_tilted_wall_and_a_slightly_rough_surface_still_read():
    points = plane((1.0, 1.0, 0.0), (0.0, 0.0, 1.2), width_m=0.8, height_m=0.6, thickness_m=0.02)
    (nx, ny), thickness = face_normal(points, camera_xy=(3.0, 3.0))
    assert nx == pytest.approx(math.sqrt(0.5), abs=0.05) and ny == pytest.approx(math.sqrt(0.5), abs=0.05)
    assert 0.01 < thickness < 0.04


def test_a_deep_object_a_table_top_a_sliver_and_too_few_points_give_no_normal():
    rng = np.random.default_rng(1)
    sofa = rng.uniform((-0.9, -0.4, 0.0), (0.9, 0.4, 0.8), (400, 3))
    assert face_normal(sofa, camera_xy=(0.0, -3.0)) is None, "a sofa is not a plane"
    table_top = plane((0.0, 0.0, 1.0), (0.0, 0.0, 0.75), width_m=1.2, height_m=0.8)
    assert face_normal(table_top, camera_xy=(0.0, -3.0)) is None, "a horizontal plane has no front"
    sliver = plane((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), width_m=0.02, height_m=1.0)
    assert face_normal(sliver, camera_xy=(0.0, -3.0)) is None, "a line fixes no plane"
    few = plane((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), width_m=0.6, height_m=0.5, n=MIN_POINTS - 1)
    assert face_normal(few, camera_xy=(0.0, -3.0)) is None
    leaning = plane((0.0, -0.5, 1.0), (0.0, 0.0, 1.0), width_m=0.6, height_m=0.5)
    assert face_normal(leaning, camera_xy=(0.0, -3.0)) is None, "a surface leaning 63 degrees back is a top, not a face"
    assert face_normal(np.zeros((5, 2)), camera_xy=(0.0, 0.0)) is None
