import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_peeks import parse_events, peek_steps, region_at, visit_intervals

LINES = """  10 excluded(rooms={"1": "scanned:scan_point_inside"})
  36 release(asked=true, local_steps=0, room=0, verdict=exhausted)
  47 arrive(room=2)
  53 release(asked=true, local_steps=2, room=4, verdict=exhausted)
  60 arrive(room=6)
 123 excluded(rooms={"0": "scanned:scan_point_inside", "2": "scanned:seen_thr...)
 130 release(asked=true, local_steps=1, room=2, verdict=exhausted)
"""


def test_events_are_read_with_their_step_name_and_room():
    events = parse_events(LINES)
    assert (47, "arrive", 2) in events and (36, "release", 0) in events
    assert (10, "excluded", None) in events


def test_a_shortened_line_carries_no_room():
    assert (123, "excluded", None) in parse_events(LINES)


def test_a_visit_runs_from_arrival_to_the_next_arrival_or_its_release():
    assert visit_intervals(parse_events(LINES), last_step=200) == [(2, 47, 60), (6, 60, 200)]


def test_a_second_arrival_ends_the_visit_before_it():
    events = [(5, "arrive", 1), (9, "arrive", 2), (12, "release", 2)]
    assert visit_intervals(events, last_step=40) == [(1, 5, 9), (2, 9, 12)]


def test_only_the_first_visit_to_a_room_is_kept_by_peek_steps():
    assert peek_steps(47, 130) == (47, 51, list(range(47, 78)))
    assert peek_steps(47, 50) == (47, 50, [47, 48, 49, 50])


def test_a_pose_belongs_to_the_nearest_floor_cell_region_on_its_floor():
    cells = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    regions = np.array([1, 2, 3])
    assert region_at((0.2, 0.0, 0.0), cells, regions) == 1
    assert region_at((4.8, 0.1, 0.0), cells, regions) == 2
    assert region_at((0.0, 3.0, 0.1), cells, regions) == 3


def test_a_pose_far_from_every_cell_belongs_to_no_region():
    cells = np.array([[0.0, 0.0, 0.0]])
    assert region_at((9.0, 0.0, 9.0), cells, np.array([1])) is None


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_peeks import pose_visits


def test_a_room_is_visited_at_the_first_run_of_steps_inside_it():
    regions = [None, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1]
    assert pose_visits(regions, min_steps=3) == {1: (1, 4), 2: (5, 7)}


def test_a_run_shorter_than_the_minimum_is_a_pass_through_and_not_a_visit():
    assert pose_visits([1, 1, 2, 1, 1, 1], min_steps=3) == {1: (3, 5)}


def test_a_room_left_and_entered_again_keeps_its_first_visit_only():
    assert pose_visits([1, 1, 1, None, 1, 1, 1], min_steps=3) == {1: (0, 2)}


def test_no_steps_in_any_region_gives_no_visits():
    assert pose_visits([None, None, None], min_steps=3) == {}
