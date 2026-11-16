from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import in_room_share, pick_frames


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
