import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.grid_views import leg_actions
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.walk_views import even_picks, keep_best, walk_poses

FACING_PLUS_X = -math.pi / 2


def line(*xz):
    return [np.array([x, 0.0, z]) for x, z in xz]


def test_a_straight_walk_has_a_pose_per_forward_step_after_the_start():
    poses = walk_poses(line((0, 0), (1, 0)), FACING_PLUS_X)
    assert len(poses) == 5
    assert [round(float(p[0][0]), 2) for p in poses] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(abs(p[1] - FACING_PLUS_X) < 1e-9 for p in poses)


def test_turning_to_face_the_walk_gives_a_pose_per_thirty_degrees_in_place():
    poses = walk_poses(line((0, 0), (0.25, 0)), FACING_PLUS_X + math.pi / 2)
    assert [round(float(p[0][0]), 2) for p in poses] == [0.0, 0.0, 0.0, 0.0, 0.25]
    assert abs(poses[-1][1] - FACING_PLUS_X) < 1e-9
    assert abs(poses[1][1] - (FACING_PLUS_X + math.pi / 2 - math.radians(30))) < 1e-9


def test_the_number_of_poses_after_the_start_is_the_number_of_actions_of_the_walk():
    path = line((0, 0), (1.1, 0), (1.1, 2.0))
    assert len(walk_poses(path, FACING_PLUS_X)) - 1 == leg_actions([(p[0], p[2]) for p in path], FACING_PLUS_X, walk_poses(path, FACING_PLUS_X)[-1][1])


def test_the_walk_ends_at_the_end_of_the_path_with_a_last_short_step():
    poses = walk_poses(line((0, 0), (0.6, 0)), FACING_PLUS_X)
    assert round(float(poses[-1][0][0]), 2) == 0.6 and len(poses) == 4


def test_even_picks_are_spread_through_the_walk():
    assert even_picks(9, 3) == [2, 4, 6]
    assert even_picks(2, 3) == [0, 1]
    assert even_picks(0, 3) == []


def frame(seen, near=0.0, inside=1.0, cells=8):
    return {"cells": np.array([i in seen for i in range(cells)]), "near": near, "inside": inside}


def test_the_first_usable_frames_are_kept_until_there_are_three():
    assert keep_best([frame({0, 1}), frame({2, 3}), frame({4, 5})], 3) == [0, 1, 2]


def test_a_later_frame_that_shows_more_new_floor_replaces_the_one_that_adds_least():
    frames = [frame({0, 1}), frame({0, 1, 2}), frame({3, 4}), frame({4, 5, 6, 7})]
    assert set(keep_best(frames, 3)) == {1, 2, 3}


def test_close_ups_and_views_out_of_the_room_are_never_kept():
    frames = [frame({0, 1, 2, 3}, near=0.9), frame({4, 5}, inside=0.1), frame({6})]
    assert keep_best(frames, 3) == [2]


def test_a_frame_that_adds_nothing_does_not_displace_a_kept_one():
    frames = [frame({0, 1, 2, 3}), frame({4, 5, 6, 7}), frame({0, 1}), frame({4})]
    assert set(keep_best(frames, 2)) == {0, 1}
