"""The per-step types -- pose, camera and observation -- accept what they promise and refuse what they forbid.

A depth image in integer millimetres, a swapped image size or a NaN pose
crashes nothing downstream: each moves SR and SPL. So each documented refusal
is exercised here, with the error class the docstring names, and each
documented acceptance too, because a type that refuses a legal input (a
numpy-integer image size, an unlimited depth range) breaks every adapter that
meets it.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.objnav.errors import (
    ObjNavError,
    ObservationError,
)
from sparx_agency.core.planning.objnav.tests.helpers import (
    INF,
    NAN,
    camera,
    intrinsics,
)
from sparx_agency.core.planning.objnav.types.angles import MAX_ANGLE_RAD
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def observation(**overrides):
    """A valid step-0 observation from :func:`camera`, with any field replaced."""
    size = (6, 8)
    fields = dict(rgb=np.zeros(size + (3,), np.uint8),
                  depth_m=np.ones(size, np.float32), pose=AgentPose(0.0, 0.0),
                  camera=camera(), target_category="chair", step=0)
    fields.update(overrides)
    return ObjNavObservation(**fields)


# -- AgentPose ------------------------------------------------------------------

def test_a_pose_defaults_to_level_and_exposes_its_planar_parts():
    """Planners consume ``pose2d``; the harness sums ``position`` chords for SPL."""
    pose = AgentPose(1.0, 2.0)
    assert (pose.z, pose.yaw, pose.camera_pitch) == (0.0, 0.0, 0.0)
    full = AgentPose(1.0, 2.0, z=3.0, yaw=0.5, camera_pitch=0.25)
    assert full.pose2d() == Pose2D(1.0, 2.0, 0.5)
    assert full.position() == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("field", ["x", "y", "z", "yaw", "camera_pitch"])
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_non_finite_pose_field_is_refused(field, value):
    """A NaN pose poisons every distance and projection downstream without raising."""
    fields = dict(x=0.0, y=0.0)
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        AgentPose(**fields)


@pytest.mark.parametrize("field", ["yaw", "camera_pitch"])
@pytest.mark.parametrize("value", [1e17, -2.0 * MAX_ANGLE_RAD])
def test_a_pose_angle_too_large_to_wrap_is_refused(field, value):
    """normalize_angle subtracts one turn per loop: at 1e17 rad the converter hung instead of raising."""
    with pytest.raises(ObjNavError, match="wrap it at the adapter"):
        AgentPose(0.0, 0.0, **{field: value})


def test_a_pose_angle_at_the_bound_is_accepted():
    """The bound refuses garbage, not an unwrapped yaw accumulated over an episode."""
    pose = AgentPose(0.0, 0.0, yaw=MAX_ANGLE_RAD, camera_pitch=-MAX_ANGLE_RAD)
    assert (pose.yaw, pose.camera_pitch) == (MAX_ANGLE_RAD, -MAX_ANGLE_RAD)


# -- CameraSpec -----------------------------------------------------------------

def test_a_valid_camera_is_accepted():
    """The helper every other test builds on must itself be valid."""
    assert camera().intrinsics.width == 8


def test_numpy_integer_image_sizes_are_accepted():
    """Sizes read from an array's shape are numpy integers; refusing them refuses every adapter."""
    cam = camera(intrinsics=intrinsics(width=np.int64(8), height=np.int32(6)))
    assert (cam.intrinsics.width, cam.intrinsics.height) == (8, 6)


@pytest.mark.parametrize("overrides", [
    dict(width=0), dict(height=-6), dict(width=8.0), dict(height=True),
    dict(fx=0.0), dict(fy=NAN), dict(fx=INF), dict(fy=True),
    dict(cx=NAN), dict(cy=INF),
], ids=lambda o: "%s=%r" % next(iter(o.items())))
def test_bad_intrinsics_are_refused(overrides):
    """A zero focal length or a float width breaks every pixel projection later."""
    with pytest.raises(ObjNavError, match="intrinsics"):
        camera(intrinsics=intrinsics(**overrides))


def test_intrinsics_of_another_type_are_refused():
    """A bare tuple has no field names, so its order is anyone's guess."""
    with pytest.raises(ObjNavError, match="Intrinsics"):
        camera(intrinsics=(8, 6, 4.0, 4.0, 3.5, 2.5))


@pytest.mark.parametrize("height", [-0.1, NAN, INF])
def test_a_negative_or_non_finite_mount_height_is_refused(height):
    """A camera below its own base is a sign error in the adapter."""
    with pytest.raises(ObjNavError, match="height_m"):
        camera(height_m=height)


def test_a_camera_at_floor_level_is_accepted():
    """Zero height is odd but physical, so it is not an error."""
    assert camera(height_m=0.0).height_m == 0.0


@pytest.mark.parametrize("low,high", [(-0.1, 5.0), (5.0, 5.0), (5.0, 0.1),
                                      (0.1, NAN), (INF, INF)])
def test_a_negative_or_empty_depth_range_is_refused(low, high):
    """An empty range would drop every depth pixel without a word."""
    with pytest.raises(ObjNavError, match="depth"):
        camera(min_depth_m=low, max_depth_m=high)


def test_an_unlimited_depth_range_is_accepted():
    """A sensor without a range limit reports ``inf`` as its maximum; that is legal."""
    assert camera(min_depth_m=0.0, max_depth_m=INF).max_depth_m == INF


# -- ObjNavObservation -------------------------------------------------------------

def test_a_valid_observation_shares_its_arrays():
    """Copying every frame would double the memory of a run for nothing."""
    rgb = np.zeros((6, 8, 3), np.uint8)
    depth = np.full((6, 8), NAN, np.float64)
    obs = observation(rgb=rgb, depth_m=depth, step=np.int64(3))
    assert obs.rgb is rgb and obs.depth_m is depth
    assert obs.step == 3


@pytest.mark.parametrize("rgb", [
    np.zeros((6, 8, 3), np.float32), np.zeros((6, 8, 4), np.uint8),
    np.zeros((8, 6, 3), np.uint8), np.zeros((6, 8), np.uint8), [[0]],
], ids=["float-dtype", "rgba", "width-height-swapped", "grey", "not-an-array"])
def test_a_malformed_rgb_image_is_refused(rgb):
    """A swapped size or a leftover alpha channel is cheapest to catch before a pixel is read."""
    with pytest.raises(ObservationError, match="rgb"):
        observation(rgb=rgb)


@pytest.mark.parametrize("depth", [
    np.zeros((6, 8), np.uint16), np.zeros((6, 8), np.int32),
    np.zeros((8, 6), np.float32), np.zeros((6, 8, 1), np.float32),
], ids=["uint16-millimetres", "int", "width-height-swapped", "trailing-channel"])
def test_a_malformed_depth_image_is_refused(depth):
    """Integer depth is millimetres; read as metres it puts every wall a kilometre away."""
    with pytest.raises(ObservationError, match="depth_m"):
        observation(depth_m=depth)


@pytest.mark.parametrize("target", ["", None, 3])
def test_a_blank_target_is_refused(target):
    """An observation must say what it is searching for."""
    with pytest.raises(ObservationError, match="target_category"):
        observation(target_category=target)


@pytest.mark.parametrize("step", [-1, True, 1.0, "0"])
def test_a_negative_bool_or_non_integer_step_is_refused(step):
    """The runner checks step order; ``True`` would pass as step 1."""
    with pytest.raises(ObservationError, match="step"):
        observation(step=step)


def test_a_pose_or_camera_of_the_wrong_type_is_refused():
    """A ``Pose2D`` has no camera pitch, and a camera must match the images."""
    with pytest.raises(ObservationError, match="pose"):
        observation(pose=Pose2D(0.0, 0.0, 0.0))
    with pytest.raises(ObservationError, match="camera"):
        observation(camera=intrinsics())


def test_observations_compare_by_identity():
    """Generated equality would compare images element-wise and raise on the result."""
    first = observation()
    second = observation(rgb=first.rgb, depth_m=first.depth_m)
    assert first == first
    assert first != second
    assert len({first, second}) == 2
