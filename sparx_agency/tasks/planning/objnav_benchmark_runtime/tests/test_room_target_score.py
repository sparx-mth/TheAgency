import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import (
    TARGETS, auroc, auroc_difference, parse_percent, present, room_score, scene_top1)


def test_a_room_holds_a_target_when_an_annotated_name_belongs_to_its_category():
    assert present(["curtain", "armchair", "lamp"], "chair")
    assert present(["couch"], "sofa") and present(["tv"], "tv_monitor") and present(["potted plant"], "plant")
    assert not present(["curtain", "toilet paper"], "toilet")
    assert not present(["bedside table", "pillow"], "bed")


def test_a_name_that_only_contains_the_category_word_is_not_the_category():
    assert not present(["chair rail", "tv stand", "flowerbed"], "chair")
    assert not present(["tv stand"], "tv_monitor")


def test_all_six_objectnav_targets_have_names():
    assert set(TARGETS) == {"chair", "bed", "plant", "toilet", "tv_monitor", "sofa"}


def test_a_percent_is_read_from_the_json_reply_and_clipped_to_0_100():
    assert parse_percent('{"percent": 70}') == 70.0
    assert parse_percent('{"percent": "35%"}') == 35.0
    assert parse_percent('{"percent": 250}') == 100.0
    assert parse_percent('{"percent": -5}') == 0.0


def test_an_unreadable_reply_has_no_score():
    assert parse_percent("") is None and parse_percent("{}") is None and parse_percent('{"percent": "high"}') is None


def test_a_room_scores_the_mean_over_its_frames_and_none_without_any():
    assert room_score([10.0, 30.0, None]) == 20.0
    assert room_score([None]) is None


def test_auroc_is_one_for_a_perfect_ranking_half_for_ties_and_zero_for_the_reverse():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_auroc_needs_both_classes():
    assert np.isnan(auroc([0.1, 0.2], [True, True]))


def test_the_top_room_of_a_house_counts_when_it_holds_the_target_and_ties_split_the_credit():
    houses = {"a": [(0.9, True), (0.1, False)], "b": [(0.2, True), (0.8, False)], "c": [(0.5, True), (0.5, False)], "d": [(0.3, True), (0.2, True)]}
    assert scene_top1(houses) == (1 + 0 + 0.5) / 3


def test_the_difference_interval_covers_zero_for_two_equal_rankings():
    rng = np.random.default_rng(0)
    scores = list(np.random.default_rng(1).random(60))
    labels = [bool(i % 2) for i in range(60)]
    mean, low, high = auroc_difference(scores, scores, labels, rng)
    assert mean == 0.0 and low == 0.0 and high == 0.0


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import yes_no_percent


def test_a_yes_that_is_much_likelier_than_no_gives_a_high_percent_and_a_positive_logit():
    percent, logit = yes_no_percent([{"token": "yes", "logprob": -0.01}, {"token": "no", "logprob": -4.6}])
    assert percent > 98 and logit > 4


def test_spelling_and_case_variants_of_the_answer_are_pooled():
    low = yes_no_percent([{"token": "No", "logprob": -0.1}, {"token": " no", "logprob": -3.0}, {"token": "yes", "logprob": -5.0}])
    assert low[0] < 2 and low[1] < 0


def test_equal_chances_give_fifty_percent():
    percent, logit = yes_no_percent([{"token": "yes", "logprob": -0.69}, {"token": "no", "logprob": -0.69}])
    assert abs(percent - 50.0) < 1e-6 and abs(logit) < 1e-9


def test_a_missing_answer_is_treated_as_very_unlikely_and_no_answer_at_all_has_no_score():
    assert yes_no_percent([{"token": "yes", "logprob": -0.2}])[1] > 20
    assert yes_no_percent([{"token": "maybe", "logprob": -0.2}]) is None
    assert yes_no_percent([]) is None
