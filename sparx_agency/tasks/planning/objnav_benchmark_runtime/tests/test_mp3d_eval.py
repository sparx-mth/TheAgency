import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import (
    accuracy_by_frames, objects_text, predict_group, spread_frames, unique_objects)


def frames(*spots):
    return [{"frame": "%d_%d" % (s, i), "spot": s} for s, n in enumerate(spots) for i in range(n)]


def test_spread_frames_takes_them_round_robin_across_standing_spots():
    assert [f["frame"] for f in spread_frames(frames(3, 3), 4)] == ["0_0", "1_0", "0_1", "1_1"]


def test_spread_frames_returns_all_when_there_are_fewer_than_asked():
    assert len(spread_frames(frames(1, 1), 6)) == 2


def test_the_predicted_group_is_the_majority_of_the_answers_read_as_groups():
    assert predict_group(["living room", "family room", "kitchen", "spaceship"]) == "living room"


def test_a_tie_goes_to_the_group_that_comes_first_in_the_pool_and_no_readable_answer_is_none():
    assert predict_group(["kitchen", "bedroom"]) == "bedroom"
    assert predict_group(["spaceship", ""]) is None


def test_objects_are_counted_most_frequent_first():
    assert objects_text(["chair", "table", "chair", "chair", "bed", "table"]) == "chair x3, table x2, bed"
    assert objects_text([]) == "(no objects)"


def test_one_object_seen_in_several_frames_is_listed_once_and_two_far_apart_stay_two():
    seen = [{"cls": "chair", "point": (0.0, 0.0)}, {"cls": "chair", "point": (0.3, 0.1)}, {"cls": "chair", "point": (3.0, 0.0)},
            {"cls": "bed", "point": (0.1, 0.0)}]
    assert sorted(unique_objects(seen, radius_m=0.6)) == ["bed", "chair", "chair"]


def test_accuracy_by_frames_votes_over_random_frame_subsets():
    answers = {"r1": ["kitchen", "kitchen", "bedroom"], "r2": ["bedroom", "bedroom", "bedroom"]}
    truth = {"r1": "kitchen", "r2": "bedroom"}
    result = accuracy_by_frames(answers, truth, ks=(1, 3), repeats=50, rng=np.random.default_rng(0))
    assert result[3] == 1.0 and 0.75 < result[1] < 1.0


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import OPEN_PROMPT, list_prompt, objects_prompt
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_room_types import ROOM_POOL


def test_the_open_prompt_offers_no_names_and_the_list_prompt_offers_exactly_the_pool():
    assert not any(option in OPEN_PROMPT for option in ROOM_POOL)
    assert all(option in list_prompt() for option in ROOM_POOL) and '{"room"' in list_prompt()


def test_the_objects_prompt_carries_the_object_list_and_the_same_question():
    assert "chair x3, table" in objects_prompt("chair x3, table", with_list=False)
    assert all(option in objects_prompt("bed", with_list=True) for option in ROOM_POOL)


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import agreement_summary, fused_prompt, vote_counts
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_house import clean_category


def test_vote_counts_read_each_answer_as_a_group_and_list_the_most_voted_first():
    assert vote_counts(["Living room", "family room", "kitchen", "spaceship"]) == [("living room", 2), ("kitchen", 1)]


def test_the_fused_prompt_carries_the_objects_and_how_the_photos_voted_and_the_pool_when_asked():
    prompt = fused_prompt("bed, lamp", [("bedroom", 4), ("hallway", 2)], with_list=True)
    assert "bed, lamp" in prompt and "bedroom 4" in prompt and "hallway 2" in prompt
    assert all(option in prompt for option in ROOM_POOL)
    assert "Choose one of" not in fused_prompt("bed", [("bedroom", 1)], with_list=False)


def test_agreement_summary_splits_rooms_by_whether_the_two_methods_agree_and_who_was_right():
    truth = {"a": "kitchen", "b": "kitchen", "c": "bedroom", "d": "bedroom", "e": "hallway"}
    first = {"a": "kitchen", "b": "kitchen", "c": "kitchen", "d": "bedroom", "e": "hallway"}
    second = {"a": "kitchen", "b": "bedroom", "c": "bedroom", "d": "kitchen", "e": "bathroom"}
    assert agreement_summary(first, second, truth) == {"rooms": 5, "agree": 1, "agree_right": 1, "only_first": 3, "only_second": 1, "neither": 0}


def test_delete_is_a_placeholder_not_an_object():
    assert clean_category("delete") is None and clean_category("lamp") == "lamp"


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.build_mp3d_frames import interior_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import first_k_accuracy, paired_difference


def test_a_zero_margin_keeps_floor_cells_right_at_the_region_edge():
    cells = np.array([[0.1, 0.0, 2.0], [2.0, 0.0, 2.0]])
    keep_all = lambda points: np.ones(len(points), dtype=bool)
    assert len(interior_cells(cells, (0.0, 0.0, 4.0, 4.0), 0.0, keep_all)) == 1
    assert len(interior_cells(cells, (0.0, 0.0, 4.0, 4.0), 0.0, keep_all, margin=0.0)) == 2


def test_first_k_accuracy_scores_the_vote_over_each_rooms_first_k_answers():
    answers = {"a": ["kitchen", "bedroom", "bedroom"], "b": ["bedroom", "bedroom", "kitchen"]}
    truth = {"a": "bedroom", "b": "bedroom"}
    assert first_k_accuracy(answers, truth, 1) == {"a": False, "b": True}
    assert first_k_accuracy(answers, truth, 3) == {"a": True, "b": True}


def test_paired_difference_is_the_mean_gain_with_an_interval_that_shrinks_with_more_rooms():
    gain = {r: True for r in range(60)}
    same = {r: r % 2 == 0 for r in range(60)}
    mean, low, high = paired_difference(gain, same, np.random.default_rng(0))
    assert mean == 0.5 and 0.3 < low <= mean <= high < 0.7
    assert paired_difference(same, same, np.random.default_rng(0)) == (0.0, 0.0, 0.0)
