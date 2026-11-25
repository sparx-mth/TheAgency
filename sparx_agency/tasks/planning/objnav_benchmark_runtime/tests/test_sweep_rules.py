import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_rules import frame_picks, usable_frames

FRAMES = {
    "a": {"near": 0.1, "inside": 0.9, "floor_share": 0.2, "entrance_m": 0.5},
    "b": {"near": 0.1, "inside": 0.9, "floor_share": 0.6, "entrance_m": 3.0},
    "c": {"near": 0.1, "inside": 0.9, "floor_share": 0.4, "entrance_m": 6.0},
    "d": {"near": 0.9, "inside": 0.9, "floor_share": 0.9, "entrance_m": 2.0},
    "e": {"near": 0.1, "inside": 0.3, "floor_share": 0.8, "entrance_m": 1.0},
}


def test_close_ups_and_views_mostly_outside_the_room_are_not_usable():
    assert sorted(usable_frames(FRAMES)) == ["a", "b", "c"]


def test_floor_rule_takes_the_usable_views_with_most_floor():
    assert frame_picks(FRAMES, "floor", 2, np.random.default_rng(0)) == ["b", "c"]


def test_door_and_far_rules_order_by_distance_to_the_entrance():
    assert frame_picks(FRAMES, "door", 2, np.random.default_rng(0)) == ["a", "b"]
    assert frame_picks(FRAMES, "far", 1, np.random.default_rng(0)) == ["c"]


def test_random_rule_picks_k_usable_views_and_depends_on_the_generator():
    picks = frame_picks(FRAMES, "random", 2, np.random.default_rng(1))
    assert len(picks) == 2 and set(picks) <= {"a", "b", "c"}


def test_fewer_usable_views_than_asked_gives_what_there_is():
    assert len(frame_picks(FRAMES, "floor", 10, np.random.default_rng(0))) == 3


def test_gate_rule_takes_the_usable_views_with_the_highest_gate_logit():
    gate = {"a": 2.0, "b": -1.0, "c": 0.5, "d": 9.0, "e": 9.0}
    assert frame_picks(FRAMES, "gate", 2, np.random.default_rng(0), gate=gate) == ["a", "c"]


def test_gate_rule_leaves_out_views_the_gate_never_scored():
    assert frame_picks(FRAMES, "gate", 3, np.random.default_rng(0), gate={"c": 1.0}) == ["c"]
