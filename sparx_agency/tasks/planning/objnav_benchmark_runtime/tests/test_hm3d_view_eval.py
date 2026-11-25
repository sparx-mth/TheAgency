from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.hm3d_view_eval import accepted_groups, correct_rooms
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.label_store import OPEN_PLAN, ROOM_TYPES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_room_types import ROOM_POOL


def test_every_room_type_on_the_page_maps_to_groups_the_model_can_name_or_is_left_out():
    for room_type in ROOM_TYPES:
        groups = accepted_groups(room_type)
        assert groups is None or (groups and groups <= set(ROOM_POOL)), room_type


def test_unsure_and_not_a_room_are_left_out():
    assert accepted_groups("unsure") is None and accepted_groups("not a room") is None


def test_several_names_share_a_group_and_an_open_plan_room_accepts_its_three_functions():
    assert accepted_groups("toilet") == accepted_groups("bathroom") == {"bathroom"}
    assert accepted_groups("study") == {"office"} and accepted_groups("family room or lounge") == {"living room"}
    assert accepted_groups(OPEN_PLAN) == {"kitchen", "living room", "dining room"}


def test_strict_scoring_accepts_one_group_for_the_two_ambiguous_types():
    assert accepted_groups(OPEN_PLAN, strict=True) == {"living room"}
    assert accepted_groups("playroom or kids room") == {"recreation room", "bedroom"}
    assert accepted_groups("playroom or kids room", strict=True) == {"recreation room"}


def test_rooms_are_scored_against_the_accepted_groups_and_left_out_rooms_are_missing():
    labels = {"a": "bedroom", "b": OPEN_PLAN, "c": "unsure", "d": "garage"}
    predicted = {"a": "bedroom", "b": "kitchen", "c": "bedroom", "d": None}
    assert correct_rooms(predicted, labels) == {"a": True, "b": True, "d": False}
    assert correct_rooms(predicted, labels, strict=True) == {"a": True, "b": False, "d": False}
