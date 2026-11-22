import json

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_prompts import reply_value, verdict


def test_a_percent_reply_is_read_as_its_percent():
    assert reply_value(json.dumps({"percent": 35}), "percent") == 35


def test_a_yes_no_reply_is_read_as_its_logit_when_it_has_one():
    assert reply_value(json.dumps({"percent": 99.9, "logit": 6.9}), "keep") == 6.9


def test_an_unreadable_reply_is_nothing():
    assert reply_value("", "percent") is None and reply_value("{}", "keep") is None


def test_a_wording_is_kept_only_when_its_interval_clears_the_set_bar():
    assert verdict((0.72, 0.62, 0.80)) == "keep"
    assert verdict((0.72, 0.55, 0.80)) == "not shown"
    assert verdict((0.66, 0.62, 0.70)) == "not shown"
