"""``hm3d/campaign_breakdown.py``: the storey join and the failure-mechanism tags, on synthetic rows."""
from __future__ import annotations

import csv

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign_breakdown import (
    DEFAULT_LEVELS, MECHANISMS, load_levels, mechanism, storey_of)


def _row(stop, dtg, closing=None, blocked=0, lean=False, error=None, pursuit=None):
    policy = {}
    if closing is not None:
        policy["target_closing"] = closing
    if pursuit is not None:
        policy["target_pursuit"] = {"stats": pursuit}
    info = {"policy": policy, "blocked_forward": blocked}
    if lean:
        info["lean"] = True
    return {"episode_id": "00800-TEEsavR23oF/000001", "stop_called": stop, "distance_to_goal_m": dtg,
            "termination": "agent_error" if error else ("stop" if stop else "step_limit"), "agent_error": error,
            "agent_info": info}


def test_the_shipped_v1_val_survey_covers_the_split_and_splits_it_by_storey():
    levels = load_levels(DEFAULT_LEVELS)
    assert len(levels) == 2000 and len({k.split("/")[0] for k in levels}) == 20
    same = sum(1 for row in levels.values() if row["goal_same_level"] == 1)
    assert same == 1652 and len(levels) - same == 348
    assert storey_of("00877-4ok3usBNeis/000000", levels) == "same"
    assert storey_of("nowhere/000000", levels) is None
    with DEFAULT_LEVELS.open() as stream:
        assert "goal_same_level" in csv.DictReader(stream).fieldnames


def test_every_tag_is_reachable_and_in_the_declared_order():
    unseen = {"last_seen_step": -1, "releases": 0, "stall_releases": 0, "bearing_starts": 0}
    seen = dict(unseen, last_seen_step=120, releases=2)
    assert mechanism(_row(False, 9.0, error="RuntimeError: boom"), "same") == "agent_error"
    assert mechanism(_row(False, 9.0, closing=seen), "cross") == "limit_cross_storey_goal"
    assert mechanism(_row(False, 9.0, closing=unseen), "same") == "limit_never_seen"
    assert mechanism(_row(False, 9.0, closing=dict(unseen, stall_releases=1)), "same") == "limit_wedged"
    assert mechanism(_row(False, 9.0, closing=seen, blocked=40), "same") == "limit_wedged"
    assert mechanism(_row(False, 9.0, closing=seen), "same") == "limit_seen_not_converted"
    assert mechanism(_row(False, 9.0, closing=unseen, pursuit={"queries": 1}), "same") == "limit_seen_not_converted"
    assert mechanism(_row(False, 9.0), "same") == "limit_unknown", "a record without the closing's counters says so"
    assert mechanism(_row(True, 3.0, closing=seen), "cross") == "stop_cross_storey_goal"
    assert mechanism(_row(True, 0.4, closing=seen), "same") == "stop_near_miss"
    assert mechanism(_row(True, 3.0, closing=seen), "same") == "stop_far"
    assert mechanism(_row(True, 3.0, lean=True), None) == "stop_far"
    assert mechanism(_row(True, 3.0), None) == "stop_unknown"
    tags = {mechanism(_row(False, 9.0, closing=unseen), "same"), mechanism(_row(True, 0.4, closing=seen), "same")}
    assert tags <= set(MECHANISMS) and MECHANISMS.index("limit_cross_storey_goal") < MECHANISMS.index("limit_never_seen")
