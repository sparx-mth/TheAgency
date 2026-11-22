import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.frontier_membership import MapFrame, cluster_membership, in_view, plan_steps, sees


def open_floor(size=60):
    return np.zeros((size, size), dtype=np.int8)


FRAME = MapFrame(origin=(0.0, 0.0), resolution=0.1)


def test_a_point_ahead_within_range_is_in_view_and_one_behind_or_far_is_not():
    pose = (1.0, 1.0, 0.0)
    assert in_view(pose, (3.0, 1.0)) and not in_view(pose, (-1.0, 1.0)) and not in_view(pose, (7.0, 1.0))


def test_the_edge_of_the_field_of_view_is_half_the_horizontal_angle_each_side():
    pose = (1.0, 1.0, 0.0)
    inside, outside = np.radians(39.0), np.radians(41.0)
    assert in_view(pose, (1.0 + 2.0 * np.cos(inside), 1.0 + 2.0 * np.sin(inside)))
    assert not in_view(pose, (1.0 + 2.0 * np.cos(outside), 1.0 + 2.0 * np.sin(outside)))


def test_yaw_turns_the_field_of_view_counter_clockwise_from_plus_x():
    assert in_view((1.0, 1.0, np.pi / 2), (1.0, 3.0)) and not in_view((1.0, 1.0, np.pi / 2), (3.0, 1.0))


def test_a_wall_between_the_pose_and_the_point_hides_it():
    grid = open_floor()
    assert sees((1.0, 1.0, 0.0), (3.0, 1.0), grid, FRAME)
    grid[10, 20] = 100
    assert not sees((1.0, 1.0, 0.0), (3.0, 1.0), grid, FRAME)


def test_unknown_cells_on_the_way_also_hide_the_point():
    grid = open_floor()
    grid[10, 20] = -1
    assert not sees((1.0, 1.0, 0.0), (3.0, 1.0), grid, FRAME)


def test_plan_steps_are_where_the_state_becomes_select():
    states = ["warmup", "transit", "select", "select", "transit", "select"]
    assert plan_steps(states) == [2, 5]


def test_membership_counts_the_frames_of_the_window_that_see_each_cluster():
    poses = [(3.0, 3.0, 0.0), (3.0, 3.0, np.pi), (3.0, 3.0, 0.0)]
    counts = cluster_membership(poses, [(5.0, 3.0), (1.0, 3.0)], open_floor(), FRAME)
    assert counts == [[0, 2], [1]]
