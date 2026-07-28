"""Tests for the stub's voxel raycaster.

The renderer is only useful if its depth means the same thing Isaac's does:
**perpendicular distance to the image plane**, in the optical frame, with the
image right-handed the same way round. A renderer that is subtly mirrored, or
that reports ray length instead of plane distance, would produce a stub that
passes while the real aircraft fails -- the exact opposite of what it is for.
"""
import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.falcon_pegasus.stub.voxel_camera import VoxelDepthCamera

RESOLUTION = 0.1
ORIGIN = (-5.0, -5.0, -1.0)
SHAPE = (40, 100, 100)   # (nz, ny, nx) -> 4 m tall, 10 x 10 m


def _empty_world():
    return np.zeros(SHAPE, dtype=np.int8)


def _wall_at_x(voxels, x_m):
    """A solid wall spanning the whole grid at one x."""
    index = int(round((x_m - ORIGIN[0]) / RESOLUTION))
    voxels[:, :, index] = 1
    return voxels


def _camera(voxels, width=64, height=48, **kwargs):
    intrinsics = Intrinsics(width=width, height=height,
                            fx=width / 2.0, fy=width / 2.0,
                            cx=width / 2.0, cy=height / 2.0)
    return VoxelDepthCamera(voxels, ORIGIN, RESOLUTION, intrinsics,
                            ray_shape=(width, height), **kwargs)


def _looking_along_x():
    """World-from-optical rotation for a camera at zero yaw: +z_optical = +x."""
    return np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])


def test_an_empty_world_returns_no_surfaces():
    camera = _camera(_empty_world())
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    assert np.isinf(depth).all()


def test_a_wall_is_seen_at_its_distance():
    camera = _camera(_wall_at_x(_empty_world(), 3.0))
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    centre = depth[depth.shape[0] // 2, depth.shape[1] // 2]
    assert centre == pytest.approx(3.0, abs=2 * RESOLUTION)


def test_depth_is_distance_to_the_image_plane_not_ray_length():
    """A flat wall must read the SAME depth everywhere, not further at the edges.

    This is the whole convention. Ray length off a flat wall grows as 1/cos of
    the pixel's angle -- at this 90-degree field of view that is 41 % at the
    corners -- and back-projecting it as if it were plane distance bulges every
    wall into a barrel.
    """
    camera = _camera(_wall_at_x(_empty_world(), 3.0))
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    finite = depth[np.isfinite(depth)]
    assert finite.size > 0
    assert float(finite.max() - finite.min()) < 3 * RESOLUTION


def test_output_matches_the_camera_resolution_not_the_ray_grid():
    intrinsics = Intrinsics(width=320, height=240, fx=160.0, fy=160.0, cx=160.0, cy=120.0)
    camera = VoxelDepthCamera(_wall_at_x(_empty_world(), 2.0), ORIGIN, RESOLUTION,
                              intrinsics, ray_shape=(32, 24))
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    assert depth.shape == (240, 320)
    assert depth.dtype == np.float32


def test_the_default_ray_grid_is_a_fifth_of_the_image():
    intrinsics = Intrinsics(width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)
    camera = VoxelDepthCamera(_empty_world(), ORIGIN, RESOLUTION, intrinsics)
    assert camera._ray_shape == (128, 96)


def test_a_wall_to_one_side_appears_on_that_side_of_the_image():
    """Catches a mirrored render, which builds a mirrored map and raises nothing."""
    voxels = _empty_world()
    # A wall filling only the +y half of the world, 3 m ahead.
    index = int(round((3.0 - ORIGIN[0]) / RESOLUTION))
    half = int(round((0.0 - ORIGIN[1]) / RESOLUTION))
    voxels[:, half:, index] = 1

    camera = _camera(voxels)
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    row = depth[depth.shape[0] // 2]
    left_half, right_half = row[: row.size // 2], row[row.size // 2:]
    # +y is to the aircraft's LEFT, which is the LEFT of the image.
    assert np.isfinite(left_half).mean() > 0.8
    assert np.isfinite(right_half).mean() < 0.2


def test_a_floor_appears_in_the_lower_half():
    voxels = _empty_world()
    floor_index = int(round((0.0 - ORIGIN[2]) / RESOLUTION))
    voxels[floor_index, :, :] = 1

    camera = _camera(voxels)
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    top, bottom = depth[: depth.shape[0] // 2], depth[depth.shape[0] // 2:]
    assert np.isfinite(bottom).mean() > np.isfinite(top).mean()


def test_rotating_the_camera_rotates_what_it_sees():
    camera = _camera(_wall_at_x(_empty_world(), 3.0))
    facing_wall = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    yaw = np.array([[math.cos(math.pi), -math.sin(math.pi), 0.0],
                    [math.sin(math.pi), math.cos(math.pi), 0.0],
                    [0.0, 0.0, 1.0]])
    facing_away = camera.render((0.0, 0.0, 1.0), yaw.dot(_looking_along_x()))
    assert np.isfinite(facing_wall).any()
    assert np.isinf(facing_away).all()


def test_nothing_beyond_the_far_range_is_reported():
    camera = _camera(_wall_at_x(_empty_world(), 4.0), far_m=2.0)
    depth = camera.render((0.0, 0.0, 1.0), _looking_along_x())
    assert np.isinf(depth).all()


def test_a_non_3d_grid_is_rejected():
    with pytest.raises(ValueError):
        _camera(np.zeros((10, 10), dtype=np.int8))


def test_unknown_voxels_are_not_surfaces():
    """-1 means "never observed", which is not the same as "solid"."""
    voxels = _empty_world()
    index = int(round((2.0 - ORIGIN[0]) / RESOLUTION))
    voxels[:, :, index] = -1
    camera = _camera(voxels)
    assert np.isinf(camera.render((0.0, 0.0, 1.0), _looking_along_x())).all()
