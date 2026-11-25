import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import far_fraction, in_room_share, pick_frames


def view(frame, spot, visible, region=7):
    return {"frame": frame, "spot": spot, "heading": 0, "region": region,
            "visible": [{"instance": i, "region": r, "pixels": p} for i, (r, p) in enumerate(visible)]}


def test_the_share_is_the_pixels_of_the_rooms_own_objects_over_all_visible_pixels():
    assert in_room_share(view("a", 0, [(7, 300), (3, 100)])) == 0.75


def test_a_frame_with_nothing_annotated_has_no_share():
    assert in_room_share(view("a", 0, [])) == 0.0


def test_a_frame_that_looks_mostly_at_the_room_beats_one_that_sees_many_objects_in_another_room():
    inside = view("inside", 0, [(7, 900)])
    out_the_door = view("door", 1, [(7, 100)] + [(3, 200)] * 6)
    assert [v["frame"] for v in pick_frames([out_the_door, inside], count=1)] == ["inside"]


def test_at_most_the_allowed_frames_come_from_one_standing_spot():
    views = [view("s0a", 0, [(7, 900)]), view("s0b", 0, [(7, 800)]), view("s0c", 0, [(7, 700)]), view("s1", 1, [(7, 600), (3, 400)])]
    assert [v["frame"] for v in pick_frames(views, count=3, per_spot=2)] == ["s0a", "s0b", "s1"]


def test_a_frame_that_mostly_shows_another_room_is_never_picked_even_when_slots_are_free():
    views = [view("inside", 0, [(7, 900)]), view("elsewhere", 1, [(7, 100), (3, 900)])]
    assert [v["frame"] for v in pick_frames(views, count=6)] == ["inside"]


def test_with_no_minimum_the_best_frames_of_a_mixed_room_are_still_returned():
    views = [view("a", 0, [(7, 100), (3, 900)]), view("b", 1, [(3, 1000)])]
    assert [v["frame"] for v in pick_frames(views, count=6, min_share=0.0)] == ["a", "b"]


def test_the_far_fraction_is_the_share_of_valid_depth_beyond_the_distance():
    depth = np.array([[1.0, 2.0, 5.0, 6.0], [0.0, 0.0, 3.0, 9.0]])
    assert far_fraction(depth, far_m=4.0) == 0.5


def test_a_frame_without_valid_depth_has_no_far_part():
    assert far_fraction(np.zeros((2, 2)), far_m=4.0) == 0.0


def test_a_frame_looking_far_away_is_not_picked_even_when_it_is_all_in_the_room():
    near, far = dict(view("near", 0, [(7, 900)]), far_fraction=0.1), dict(view("far", 1, [(7, 900)]), far_fraction=0.6)
    assert [v["frame"] for v in pick_frames([far, near], count=6)] == ["near"]


def test_with_no_far_limit_a_far_frame_is_returned():
    far = dict(view("far", 0, [(7, 900)]), far_fraction=0.9)
    assert [v["frame"] for v in pick_frames([far], count=6, max_far=1.0)] == ["far"]


def test_the_map_draws_a_view_where_it_stands_on_the_habitat_floor_even_though_its_pose_is_stored_in_enu():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import SPOT_COLOUR, floor_map
    cells = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 4.0]])
    room = {"region": 1, "floor_y": 0.0}
    standing_at_habitat_2_2 = {"x": -2.0, "y": -2.0, "z": 0.9, "yaw": 0.0}
    image = floor_map(cells, np.array([1, 1]), room, [standing_at_habitat_2_2])
    assert tuple(image[280, 280]) == SPOT_COLOUR
