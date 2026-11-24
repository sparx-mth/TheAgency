import math

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_within_scene import within_scene_metrics


def test_a_house_that_ranks_its_holding_room_first_scores_one_against_a_chance_of_a_third():
    houses = {"h": [(0.9, True), (0.2, False), (0.1, False)]}
    assert within_scene_metrics(houses) == {"auroc": 1.0, "top1": 1.0, "chance": 1 / 3, "houses": 1}


def test_houses_where_every_room_or_no_room_holds_the_target_say_nothing():
    houses = {"all": [(0.9, True), (0.2, True)], "none": [(0.9, False), (0.2, False)]}
    metrics = within_scene_metrics(houses)
    assert metrics["houses"] == 0 and math.isnan(metrics["auroc"]) and math.isnan(metrics["top1"])


def test_a_tie_for_the_top_splits_the_credit_and_the_auroc_counts_it_half():
    houses = {"h": [(0.5, True), (0.5, False)]}
    metrics = within_scene_metrics(houses)
    assert metrics["top1"] == 0.5 and metrics["auroc"] == 0.5 and metrics["chance"] == 0.5


def test_the_metrics_average_over_houses_not_over_rooms():
    houses = {"a": [(0.9, True), (0.1, False)], "b": [(0.1, True), (0.9, False), (0.8, False), (0.7, False)]}
    metrics = within_scene_metrics(houses)
    assert metrics["top1"] == 0.5 and metrics["auroc"] == 0.5 and metrics["chance"] == (0.5 + 0.25) / 2
