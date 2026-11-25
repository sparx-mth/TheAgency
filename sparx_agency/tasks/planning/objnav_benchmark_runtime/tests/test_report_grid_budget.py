from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_grid_budget import summary_row


def test_a_method_row_reports_views_actions_and_accuracy_over_regions_that_got_a_view():
    answers = {"a": ["kitchen", "kitchen", "bedroom"], "b": ["bedroom"], "c": []}
    actions = {"a": 30, "b": 10, "c": 0}
    truth = {"a": "kitchen", "b": "kitchen", "c": "kitchen"}
    row = summary_row(answers, actions, truth, k=3)
    assert row["regions"] == 3 and row["with_views"] == 2
    assert row["mean_views"] == 4 / 3 and row["mean_actions"] == 40 / 3
    assert row["accuracy"] == 0.5
