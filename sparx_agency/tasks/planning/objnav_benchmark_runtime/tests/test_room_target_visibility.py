import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_visibility import steps_in_index, target_visible

WIDTH, HEIGHT, HFOV, MAX_DEPTH = 64, 48, 79.0, 5.0
FACING_PLUS_X = -math.pi / 2
CAMERA = (0.0, 1.0, 0.0, FACING_PLUS_X)


def depth(value):
    return np.full((HEIGHT, WIDTH), float(value))


def visible(center, depth_image=None, margin=0):
    return target_visible(CAMERA, depth(MAX_DEPTH + 1) if depth_image is None else depth_image, center, WIDTH, HEIGHT, HFOV, margin, 0.7, MAX_DEPTH)


def test_an_object_straight_ahead_in_open_space_is_visible():
    assert visible((3.0, 1.0, 0.0), depth(10.0))


def test_an_object_behind_the_camera_or_outside_the_field_of_view_is_not():
    assert not visible((-3.0, 1.0, 0.0))
    assert not visible((1.0, 1.0, 3.0))


def test_an_object_hidden_behind_a_nearer_surface_is_not_visible():
    assert not visible((3.0, 1.0, 0.0), depth(1.5))


def test_an_object_at_the_depth_of_its_own_surface_is_visible():
    assert visible((3.0, 1.0, 0.0), depth(2.8))


def test_an_object_beyond_the_sensor_range_cannot_be_checked_for_occlusion_so_counts_as_visible():
    assert visible((8.0, 1.0, 0.0), depth(2.0))


def test_an_object_far_above_or_below_the_view_is_outside_the_picture():
    assert not visible((3.0, 6.0, 0.0))
    assert not visible((3.0, -4.0, 0.0))


def test_a_margin_keeps_an_object_just_outside_the_picture_as_visible():
    center = (2.0, 1.0, 1.9)
    assert not visible(center, depth(10.0), margin=0)
    assert visible(center, depth(10.0), margin=400)


def test_the_index_after_the_forward_steps_skips_turns_in_place():
    poses = [(np.array([0.0, 0, 0]), 0.0), (np.array([0.0, 0, 0]), 0.5), (np.array([0.25, 0, 0]), 0.5), (np.array([0.5, 0, 0]), 0.5), (np.array([0.75, 0, 0]), 0.5)]
    assert steps_in_index(poses, 2) == 3
    assert steps_in_index(poses, 4) == 4
    assert steps_in_index(poses, 9) == 4
    assert steps_in_index(poses[:1], 4) == 0
