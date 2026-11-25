import math

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import (
    APPROACH_DISTANCES_M, approach_positions, forward_xz, is_held_out, jittered_yaws, touches_border, yaw_toward)


@pytest.mark.parametrize("target, yaw", [((0, -1), 0.0), ((-1, 0), math.pi / 2), ((1, 0), -math.pi / 2)])
def test_the_yaw_toward_a_target_follows_habitats_convention(target, yaw):
    assert yaw_toward((0, 0), target) == pytest.approx(yaw)


def test_facing_the_yaw_toward_a_target_points_at_it():
    yaw = yaw_toward((2.0, 3.0), (-1.0, 7.0))
    direction = forward_xz(yaw)
    expected = np.array([-3.0, 4.0]) / 5.0
    assert np.allclose(direction, expected)


def test_the_approach_comes_from_far_to_near_on_a_line_toward_the_object():
    points = approach_positions((10.0, 5.0), math.pi / 3)
    distances = [math.dist(p, (10.0, 5.0)) for p in points]
    assert distances == pytest.approx(list(APPROACH_DISTANCES_M)) and distances == sorted(distances, reverse=True)
    bearings = {round(math.atan2(p[1] - 5.0, p[0] - 10.0), 6) for p in points}
    assert len(bearings) == 1


def test_the_first_view_looks_straight_at_the_object_and_the_others_stay_within_the_jitter():
    views = jittered_yaws(1.0, 4, 0.5, np.random.default_rng(0))
    assert views[0] == (1.0, 0.0) and len(views) == 4
    assert all(abs(jitter) <= 0.5 and yaw == pytest.approx(1.0 + jitter) for yaw, jitter in views)


def test_the_same_seed_gives_the_same_jitter():
    a = jittered_yaws(0.0, 5, 0.4, np.random.default_rng(7))
    b = jittered_yaws(0.0, 5, 0.4, np.random.default_rng(7))
    assert a == b


def test_the_scene_split_is_stable_and_near_the_requested_fraction():
    scenes = ["%05d-scene%d" % (i, i) for i in range(400)]
    held = [is_held_out(s, 0.25) for s in scenes]
    assert held == [is_held_out(s, 0.25) for s in scenes]
    assert 0.18 < sum(held) / len(held) < 0.32
    assert not any(is_held_out(s, 0.0) for s in scenes)


@pytest.mark.parametrize("box, border", [((0, 10, 50, 60), True), ((10, 10, 50, 60), False), ((10, 10, 640, 60), True),
                                         ((10, 10, 50, 480), True)])
def test_a_box_on_the_frame_edge_is_truncated(box, border):
    assert touches_border(box, 640, 480) is border
