import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import (
    KIND_OF_CATEGORY, accepted_by_vote, score_view, sequence_vote_summary, summarise)

TARGET = {"instance": 7, "kind": "chair", "name": "chair", "xyxy": [100, 100, 200, 200], "pixels": 5000, "is_goal": True, "truncated": False}
OTHER_GOAL = {"instance": 8, "kind": "chair", "name": "chair", "xyxy": [300, 100, 400, 200], "pixels": 5000, "is_goal": True, "truncated": False}
NOT_GOAL = {"instance": 9, "kind": "chair", "name": "wall chair", "xyxy": [450, 100, 550, 200], "pixels": 5000, "is_goal": False, "truncated": False}


def det(cls, conf, xyxy):
    return {"cls": cls, "conf": conf, "xyxy": list(xyxy)}


def view(boxes, visible=True, role="object", category="chair"):
    return {"role": role, "category": category, "target_instance": 7, "target_visible": visible, "boxes": boxes,
            "distance_m": 1.2, "frame": "f.jpg", "scene": "s"}


def test_the_categories_map_to_the_dataset_kinds():
    assert KIND_OF_CATEGORY == {"chair": "chair", "bed": "bed", "sofa": "sofa", "tv_monitor": "tv", "toilet": "toilet", "plant": "plant"}


def test_a_box_on_the_target_is_a_hit():
    score = score_view(view([TARGET]), [det("chair", 0.8, (105, 105, 195, 195))], 0.5)
    assert score["hit_target"] and score["wrong"] == 0 and score["on_other_goal"] == 0


def test_a_confident_box_on_another_object_of_the_goal_set_is_not_the_target_but_not_wrong_either():
    score = score_view(view([TARGET, OTHER_GOAL]), [det("chair", 0.8, (305, 105, 395, 195))], 0.5)
    assert not score["hit_target"] and score["on_other_goal"] == 1 and score["wrong"] == 0


def test_a_box_on_an_object_of_the_kind_outside_the_goal_set_is_a_wrong_stop_candidate():
    score = score_view(view([TARGET, NOT_GOAL]), [det("chair", 0.8, (455, 105, 545, 195))], 0.5)
    assert score["on_non_goal"] == 1 and score["wrong"] == 1


def test_a_box_on_nothing_is_wrong():
    score = score_view(view([TARGET]), [det("chair", 0.9, (0, 300, 50, 350))], 0.5)
    assert score["on_nothing"] == 1 and score["wrong"] == 1


def test_a_box_below_the_threshold_or_of_another_kind_is_ignored():
    detections = [det("chair", 0.3, (105, 105, 195, 195)), det("bed", 0.9, (105, 105, 195, 195))]
    score = score_view(view([TARGET]), detections, 0.5)
    assert not score["hit_target"] and score["wrong"] == 0 and score["detections"] == 0


def test_the_detectors_other_words_for_the_kind_count_as_the_kind():
    tv = dict(TARGET, kind="tv")
    score = score_view(view([tv], category="tv_monitor"), [det("television", 0.8, (105, 105, 195, 195))], 0.5)
    assert score["hit_target"]


def test_a_random_view_has_no_target_so_any_box_of_the_goal_kind_not_on_a_goal_is_wrong():
    score = score_view({"role": "random", "boxes": [], "category": None}, [det("chair", 0.9, (0, 0, 50, 50))], 0.5, category="chair")
    assert score["wrong"] == 1 and not score["hit_target"]


def test_the_summary_counts_recall_over_visible_views_only_and_false_alarms_per_view():
    rows = [dict(score_view(view([TARGET]), [det("chair", 0.9, (105, 105, 195, 195))], 0.5), target_visible=True, bucket=1.2, category="chair"),
            dict(score_view(view([TARGET]), [], 0.5), target_visible=True, bucket=1.2, category="chair"),
            dict(score_view(view([TARGET], visible=False), [det("chair", 0.9, (0, 300, 50, 350))], 0.5), target_visible=False, bucket=1.2,
                 category="chair")]
    table = summarise(rows, key="category")
    assert table["chair"]["views"] == 3 and table["chair"]["visible"] == 2
    assert table["chair"]["recall"] == pytest.approx(0.5) and table["chair"]["wrong_per_view"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("flags, need, expected", [([True, True, False], 2, True), ([True, False, False], 2, False), ([False, False], 1, False),
                                                   ([True], 1, True), ([], 1, False)])
def test_a_vote_accepts_when_enough_of_the_views_have_a_box(flags, need, expected):
    assert accepted_by_vote(flags, need) is expected


def scored(sequence, step, hit, wrong, visible=True):
    return {"scene": "s", "target_instance": 7, "sequence": sequence, "step": step, "hit_target": hit, "wrong": wrong,
            "target_visible": visible, "category": "chair", "bucket": 1.2}


def test_voting_groups_the_headings_of_one_step_and_compares_with_one_view():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import vote_summary
    rows = [scored(0, 0, True, 0)] * 2 + [scored(0, 0, False, 1)]
    rows += [scored(0, 1, False, 1)] * 2 + [scored(0, 1, False, 0)]
    rows += [scored(0, 2, True, 0)]
    summary = vote_summary(rows, need=2)
    assert summary == {"groups": 2, "hit": 1, "wrong": 1, "views": 6, "single_hit": 2, "single_wrong": 3}


def test_a_box_on_another_goal_object_counts_as_a_hit_on_the_goal_set():
    score = score_view(view([TARGET, OTHER_GOAL]), [det("chair", 0.8, (305, 105, 395, 195))], 0.5)
    assert score["hit_goal"] and not score["hit_target"]
    assert not score_view(view([TARGET, NOT_GOAL]), [det("chair", 0.8, (455, 105, 545, 195))], 0.5)["hit_goal"]


def test_the_score_carries_the_targets_size_and_whether_the_frame_cuts_it():
    cut = dict(TARGET, pixels=800, truncated=True)
    score = score_view(view([cut]), [], 0.5)
    assert score["target_pixels"] == 800 and score["target_truncated"] is True
    assert score_view(view([], visible=False), [], 0.5)["target_pixels"] == 0


def test_the_summary_reports_recall_on_the_goal_set_next_to_recall_on_the_target():
    rows = [{"hit_target": False, "hit_goal": True, "wrong": 0, "target_visible": True, "category": "chair"},
            {"hit_target": False, "hit_goal": False, "wrong": 0, "target_visible": True, "category": "chair"}]
    cell = summarise(rows, key="category")["chair"]
    assert cell["recall"] == 0 and cell["recall_goal"] == pytest.approx(0.5)


@pytest.mark.parametrize("boxes, xyxy, expected", [([TARGET], (105, 105, 195, 195), "target"),
                                                   ([TARGET, OTHER_GOAL], (305, 105, 395, 195), "other_goal"),
                                                   ([TARGET, NOT_GOAL], (455, 105, 545, 195), "non_goal"),
                                                   ([TARGET], (0, 300, 50, 350), "nothing")])
def test_one_box_is_classified_by_what_it_covers(boxes, xyxy, expected):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import classify_box
    assert classify_box(view(boxes), det("chair", 0.9, xyxy), "chair") == expected


def test_the_vlm_summary_counts_kept_right_boxes_and_rejected_wrong_ones_per_category():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import vlm_summary
    rows = [{"category": "chair", "label": "target", "answer": "yes"}, {"category": "chair", "label": "other_goal", "answer": "no"},
            {"category": "chair", "label": "non_goal", "answer": "no"}, {"category": "chair", "label": "nothing", "answer": "unsure"},
            {"category": "bed", "label": "target", "answer": "yes"}]
    table = vlm_summary(rows)
    assert table["chair"] == {"right": 2, "kept": 1, "wrong": 2, "rejected": 1, "unsure_wrong": 1, "unsure_right": 0}
    assert table["all"]["right"] == 3 and table["all"]["kept"] == 2


def step_view(step, boxes_hit=None, bucket=1.2, sequence=0):
    flags = {"detections": 0, "hit_goal": False, "wrong": 0} if boxes_hit is None else \
        {"detections": 1, "hit_goal": boxes_hit, "wrong": 0 if boxes_hit else 1}
    return dict(flags, scene="s", target_instance=7, sequence=sequence, step=step, bucket=bucket)


@pytest.mark.parametrize("need, expected", [(1, {"stopped_right": 0, "stopped_wrong": 1, "never": 0}),
                                            (2, {"stopped_right": 1, "stopped_wrong": 0, "never": 0}),
                                            (3, {"stopped_right": 0, "stopped_wrong": 0, "never": 1})])
def test_a_stop_needs_that_many_consecutive_steps_with_a_box_and_is_judged_at_the_last_of_them(need, expected):
    steps = [step_view(0, boxes_hit=False), step_view(1), step_view(2, boxes_hit=True), step_view(3, boxes_hit=True)]
    result = sequence_vote_summary(steps, need=need, max_distance_m=2.5)
    assert {k: result[k] for k in expected} == expected and result["sequences"] == 1


def test_steps_farther_than_the_limit_do_not_count():
    steps = [step_view(0, boxes_hit=False, bucket=3.5), step_view(1, boxes_hit=True, bucket=1.2)]
    assert sequence_vote_summary(steps, need=1, max_distance_m=2.5)["stopped_right"] == 1
