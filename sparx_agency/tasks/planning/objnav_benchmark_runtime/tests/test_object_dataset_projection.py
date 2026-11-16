import math

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.projection import box_floor_point

SIZE = dict(width=640, height=480, hfov_deg=79.0)


def flat_depth(metres):
    return np.full((480, 640), metres)


def test_a_centred_box_two_metres_ahead_is_two_metres_in_front_of_the_camera():
    x, z = box_floor_point((300, 220, 340, 260), flat_depth(2.0), (0.0, 0.0, 0.0), **SIZE)
    assert (x, z) == pytest.approx((0.0, -2.0), abs=0.02)


def test_a_box_right_of_the_centre_lies_to_the_cameras_right():
    x, _ = box_floor_point((500, 220, 540, 260), flat_depth(2.0), (0.0, 0.0, 0.0), **SIZE)
    assert x > 0.5


def test_the_camera_pose_moves_and_turns_the_point():
    x, z = box_floor_point((300, 220, 340, 260), flat_depth(2.0), (1.0, 3.0, math.pi / 2), **SIZE)
    assert (x, z) == pytest.approx((-1.0, 3.0), abs=0.02)


def test_the_median_of_the_valid_depth_inside_the_box_is_used():
    depth = flat_depth(2.0)
    depth[225:240, 305:335] = 0.0
    depth[245:248, 320:325] = 9.0
    x, z = box_floor_point((300, 220, 340, 260), depth, (0.0, 0.0, 0.0), **SIZE)
    assert z == pytest.approx(-2.0, abs=0.02)


def test_a_box_with_no_valid_depth_has_no_point():
    assert box_floor_point((300, 220, 340, 260), flat_depth(0.0), (0.0, 0.0, 0.0), **SIZE) is None
