"""Pin the topic and frame names against what the plugin actually publishes.

These are not tautologies. ``topics.py`` composes almost everything from
:data:`NAMESPACE` and :data:`FRAME_PREFIX`, which is what makes a rename in the
simulator's ``drone.yaml`` a one-line change here -- but it also means a typo in
the composition would move every name at once and still look internally
consistent. So the literals are written out a second time, from the plugin
source and from a live topic listing, and compared.

The three exceptions are the interesting part: ``world`` and ``camera`` are not
namespaced at all, and the IMU and sonar sit one level deeper than their
constants' names suggest (``imu/out``, not ``imu``).
"""
from __future__ import annotations

import pytest

from sparx_agency.robots.SJTU.adapters import topics

WIRE_TOPICS = {
    "CMD_VEL": "/simple_drone/cmd_vel",
    "TAKEOFF": "/simple_drone/takeoff",
    "LAND": "/simple_drone/land",
    "RESET": "/simple_drone/reset",
    "POSCTRL": "/simple_drone/posctrl",
    "DRONEVEL_MODE": "/simple_drone/dronevel_mode",
    "ODOM": "/simple_drone/odom",
    "GT_POSE": "/simple_drone/gt_pose",
    "GT_VEL": "/simple_drone/gt_vel",
    "GT_ACC": "/simple_drone/gt_acc",
    "IMU": "/simple_drone/imu/out",
    "SONAR": "/simple_drone/sonar/out",
    "BUMPER_STATES": "/simple_drone/bumper_states",
    "STATE": "/simple_drone/state",
    "CMD_MODE": "/simple_drone/cmd_mode",
    "FRONT_IMAGE": "/simple_drone/front/image_raw",
    "FRONT_CAMERA_INFO": "/simple_drone/front/camera_info",
    "FRONT_DEPTH_IMAGE": "/simple_drone/front_depth/depth/image_raw",
    "FRONT_DEPTH_CAMERA_INFO": "/simple_drone/front_depth/depth/camera_info",
    "FRONT_DEPTH_POINTS": "/simple_drone/front_depth/points",
    "BOTTOM_IMAGE": "/simple_drone/bottom/image_raw",
}

WIRE_FRAMES = {
    "FRAME_ODOM": "simple_drone/odom",
    "FRAME_BASE_FOOTPRINT": "simple_drone/base_footprint",
    "FRAME_BASE_LINK": "simple_drone/base_link",
    "FRAME_SONAR": "simple_drone/sonar_link",
    "FRAME_FRONT_CAM": "simple_drone/front_cam_link",
    "FRAME_BOTTOM_CAM": "simple_drone/bottom_cam_link",
}


@pytest.mark.parametrize("name,expected", sorted(WIRE_TOPICS.items()))
def test_topic_matches_the_wire(name, expected):
    """Each constant is exactly the string the simulator publishes."""
    assert getattr(topics, name) == expected


@pytest.mark.parametrize("name,expected", sorted(WIRE_FRAMES.items()))
def test_frame_matches_the_wire(name, expected):
    """Each namespaced frame is the URDF/plugin name with no leading slash."""
    assert getattr(topics, name) == expected


@pytest.mark.parametrize("name", sorted(WIRE_TOPICS))
def test_every_topic_is_namespaced(name):
    """No topic escapes the namespace, and none doubles a separator.

    A doubled slash is the failure a composition bug produces: ROS 2 rejects it
    at publisher creation with a name-validation error that says nothing about
    which constant was wrong.
    """
    value = getattr(topics, name)
    assert value.startswith(topics.NAMESPACE + "/")
    assert "//" not in value


@pytest.mark.parametrize("name", sorted(WIRE_FRAMES))
def test_every_frame_is_prefixed_and_slashless(name):
    """TF frames carry the prefix without the namespace's leading slash."""
    value = getattr(topics, name)
    assert value.startswith(topics.FRAME_PREFIX + "/")
    assert not value.startswith("/")


def test_frame_prefix_is_the_namespace_without_its_slash():
    """The one relationship the plugin encodes by string concatenation."""
    assert topics.NAMESPACE == "/" + topics.FRAME_PREFIX


def test_unnamespaced_frames_stay_unnamespaced():
    """``world`` and ``camera`` are outside the robot's prefix, deliberately.

    ``camera`` is what ``gazebo_ros_camera`` puts in both front sensors' image
    headers; nothing publishes a transform for it. Prefixing it here would make
    a TF lookup look correct and still return nothing.
    """
    assert topics.FRAME_WORLD == "world"
    assert topics.FRAME_CAMERA_SENSOR == "camera"


def test_front_camera_offset_is_the_urdf_joint():
    """The camera sits 20 cm ahead of the body origin, body FLU, no rotation."""
    assert topics.FRONT_CAMERA_OFFSET_FLU == (0.2, 0.0, 0.0)
