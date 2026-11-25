from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_spins import room_scores


def test_room_score_is_the_mean_percent_over_the_views_of_the_method():
    entry = {"methods": {"spin0": ["a"], "spin3": ["a", "b"]}}
    answers = {("r", "bed"): {"a": 20.0, "b": 60.0, "c": 99.0}}
    assert room_scores({"r": entry}, answers, "spin3", ["bed"]) == {("r", "bed"): 40.0}
    assert room_scores({"r": entry}, answers, "spin0", ["bed"]) == {("r", "bed"): 20.0}


def test_a_view_without_an_answer_is_left_out_of_the_mean():
    entry = {"methods": {"spin3": ["a", "b"]}}
    assert room_scores({"r": entry}, {("r", "bed"): {"a": 10.0, "b": None}}, "spin3", ["bed"]) == {("r", "bed"): 10.0}
