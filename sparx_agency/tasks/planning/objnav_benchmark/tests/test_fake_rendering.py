"""The fake's frames: a grey, read-only RGB, and depth pinned to the pixel.

A half-pixel slip in a fake's camera is invisible until a mapper built on it
is half a pixel wrong, so depth is checked against a wall edge placed between
two pixels' rays -- and the check is itself checked against the uncorrected
ray caster, which must fail it.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.camera_geometry import (
    camera_position_world,
    world_T_camera_optical,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.falcon_pegasus.stub.voxel_camera import (
    VoxelDepthCamera,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import (
    FakeEpisodeSpec,
    FakeObjNavEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.raster import voxel_grid
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.rendering import (
    RGB_GREY,
    FrameRenderer,
    ray_caster_intrinsics,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld
from sparx_agency.tasks.planning.objnav_benchmark.tests.fake_rooms import (
    RES,
    ROOM,
    room_env,
    small_camera,
)

#: The ray caster's depth sample spacing: three quarters of a voxel.
SAMPLE_M = 0.75 * RES


def test_the_frame_is_grey_rgb_and_float_depth_of_the_cameras_shape_both_read_only():
    """The fake has no appearance; a policy that wrote into a shared frame would corrupt the next."""
    observation = room_env().reset("out")[1]
    assert observation.rgb.shape == (5, 9, 3) and observation.rgb.dtype == np.uint8
    assert (observation.rgb == RGB_GREY).all()
    assert observation.depth_m.shape == (5, 9)
    assert np.issubdtype(observation.depth_m.dtype, np.floating)
    finite = observation.depth_m[np.isfinite(observation.depth_m)]
    assert finite.size and (finite >= 0.1).all() and (finite < 5.0).all()
    for image in (observation.rgb, observation.depth_m):
        assert not image.flags.writeable


def test_a_camera_the_ray_caster_cannot_render_with_is_refused():
    """From inside the ceiling every ray reads the slab; with no far limit the caster has no sample range."""
    world = GridWorld.from_ascii(ROOM, {"c": "chair"})
    high = CameraSpec(small_camera().intrinsics, height_m=2.5, min_depth_m=0.1,
                      max_depth_m=5.0)
    with pytest.raises(ObjNavError, match="inside the ceiling"):
        FrameRenderer(world, high)
    with pytest.raises(ObjNavError, match="must be finite"):
        FrameRenderer(world, small_camera(max_depth_m=float("inf")))


# -- depth, to the pixel ------------------------------------------------------------

#: The half-pixel scene: a wall block whose north edge is a quarter pixel to
#: the left of pixel 3's ray (in the doctrine's pixel centres), 2 m ahead.
EDGE_DISTANCE_M = 2.0
EDGE_Y_M = 3.0


def edge_env():
    """A level camera facing +x at a wall block whose edge falls between pixels 3 and 4."""
    walls = np.zeros((24, 48), dtype=bool)
    walls[:12, 30:34] = True          # x in [7.5, 8.5), y in [0, 3.0)
    chair = np.zeros((24, 48), dtype=bool)
    chair[20, 5] = True
    world = GridWorld(walls, {"chair": chair})
    camera = small_camera()
    k = camera.intrinsics
    # Pixel u's ray meets the plane x = 7.5 at y0 + (cx - u) / fx * 2; the
    # block's edge is placed where u = 3.25 would meet it.
    y0 = EDGE_Y_M - (k.cx - 3.25) / k.fx * EDGE_DISTANCE_M
    start = AgentPose(7.5 - EDGE_DISTANCE_M, y0, 0.0, 0.0)
    return FakeObjNavEnv(world, [FakeEpisodeSpec("edge", "edge", "chair", start)],
                         camera=camera)


def test_a_level_camera_reads_the_wall_ahead_within_one_ray_sample():
    """Depth is the optical z of the first sample inside the wall: never short, at most a sample long."""
    depth = edge_env().reset("edge")[1].depth_m
    assert EDGE_DISTANCE_M <= depth[2, 4] < EDGE_DISTANCE_M + SAMPLE_M
    assert np.isinf(depth[2, 0])      # nothing within 5 m to the left: inf, not a number


def test_depth_pixels_are_centred_where_backproject_depth_reads_them():
    """The voxel camera centres pixel u at u + 0.5; uncorrected, every depth pixel sits half a pixel off."""
    env = edge_env()
    depth = env.reset("edge")[1].depth_m
    assert np.isinf(depth[2, 3]) and np.isfinite(depth[2, 4])
    # Guard the guard: handed the raw intrinsics, the stub puts pixel 3's ray
    # past the edge -- the test above would catch that regression.
    camera = env.camera
    pose = env.reset("edge")[1].pose
    raw = VoxelDepthCamera(*voxel_grid(env.privileged_world()), camera.intrinsics,
                           ray_shape=(9, 5), near_m=0.1, far_m=5.0)
    shifted = raw.render(camera_position_world(pose, camera),
                         world_T_camera_optical(pose, camera)[:3, :3])
    assert np.isfinite(shifted[2, 3])
    k = ray_caster_intrinsics(camera.intrinsics)
    assert (k.cx, k.cy, k.fx) == (camera.intrinsics.cx + 0.5,
                                  camera.intrinsics.cy + 0.5, camera.intrinsics.fx)
