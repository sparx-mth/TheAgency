"""``hm3d/campaign_breakdown.py``: the storey join and the failure-mechanism tags, on synthetic rows."""
from __future__ import annotations

import csv

import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d import compare_campaigns
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.campaign_breakdown import (
    DEFAULT_LEVELS, MECHANISMS, load_levels, mechanism, storey_of, transitions_summary)


def _row(stop, dtg, closing=None, blocked=0, lean=False, error=None, pursuit=None, building=None, episode_id="00800-TEEsavR23oF/000001",
         success=None, spl=None, steps=100, path=5.0, category="toilet"):
    policy = {}
    if closing is not None:
        policy["target_closing"] = closing
    if pursuit is not None:
        policy["target_pursuit"] = {"stats": pursuit}
    if building is not None:
        policy["building"] = building
    info = {"policy": policy, "blocked_forward": blocked}
    if lean:
        info["lean"] = True
    return {"episode_id": episode_id, "stop_called": stop, "distance_to_goal_m": dtg,
            "termination": "agent_error" if error else ("stop" if stop else "step_limit"), "agent_error": error,
            "agent_info": info, "success": bool(stop and dtg <= 0.1) if success is None else success,
            "spl": (0.5 if (stop and dtg <= 0.1) else 0.0) if spl is None else spl, "steps": steps, "path_length_m": path,
            "target_category": category, "wall_s": 10.0}


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


def test_traversal_on_rows_get_their_own_tags_and_cross_storey_is_no_longer_unreachable():
    unseen = {"last_seen_step": -1, "releases": 0, "stall_releases": 0, "bearing_starts": 0}
    on = {"allow_stair_traversal": True, "phase": "SEARCH", "completed_transitions": [], "events": []}
    halted = dict(on, phase="SAFE_HALT")
    mid = dict(on, phase="RETREAT")
    arrived = dict(on, completed_transitions=[{"action": 139}])
    assert mechanism(_row(True, 7.9, closing=unseen, building=halted), "cross") == "stop_safe_halt"
    assert mechanism(_row(True, 7.9, closing=unseen, building=halted), "same") == "stop_safe_halt"
    assert mechanism(_row(False, 9.0, closing=unseen, building=mid), "cross") == "limit_in_traversal"
    assert mechanism(_row(False, 9.0, closing=unseen, building=on), "cross") == "limit_no_transition"
    assert mechanism(_row(False, 9.0, closing=unseen, building=arrived), "cross") == "limit_never_seen", "arrived on the goal's storey, then the usual reading"
    assert mechanism(_row(True, 3.0, closing=unseen, building=on), "cross") == "stop_far", "a STOP across a level is not 'cross_storey_goal' once traversal is allowed"
    off = {"allow_stair_traversal": False, "phase": "SEARCH", "completed_transitions": [], "events": []}
    assert mechanism(_row(False, 9.0, closing=unseen, building=off), "cross") == "limit_cross_storey_goal"
    assert all(tag in MECHANISMS for tag in ("stop_safe_halt", "limit_in_traversal", "limit_no_transition"))


def test_transitions_are_counted_per_episode_by_storey_and_outcome():
    on = lambda n: {"allow_stair_traversal": True, "phase": "SEARCH", "completed_transitions": [{}] * n, "events": []}
    rows = [_row(True, 0.05, building=on(1), episode_id="s/000000"), _row(False, 9.0, building=on(0), episode_id="s/000001"),
            _row(True, 0.05, building=on(0), episode_id="s/000002"), _row(False, 9.0, building=on(2), episode_id="s/000003")]
    storeys = {"s/000000": "cross", "s/000001": "cross", "s/000002": "same", "s/000003": "same"}
    t = transitions_summary(rows, storeys)
    assert t["total"] == 3 and t["episodes"] == 4 and t["mean"] == 0.75 and t["max"] == 2 and t["episodes_with_any"] == 2
    assert t["distribution"] == {"0": 2, "1": 1, "2": 1}
    assert t["mean_by_storey"] == {"same": 1.0, "cross": 0.5} and t["mean_by_outcome"] == {"success": 0.5, "failure": 1.0}
    assert t["cross_storey_successes_with_transition"] == 1 and t["same_storey_episodes_with_transition"] == 1
    assert t["allowed"] == ["True"]


def test_the_paired_comparison_reads_two_roots_episode_by_episode(tmp_path):
    levels = {"x/%06d" % i: {"goal_same_level": 1 if i % 2 == 0 else 0} for i in range(6)}
    off = {"allow_stair_traversal": False, "phase": "SEARCH", "completed_transitions": [], "events": []}
    on = lambda n, phase="SEARCH": {"allow_stair_traversal": True, "phase": phase, "completed_transitions": [{}] * n,
                                    "events": [{"event": "traversal_started"}] if (n or phase != "SEARCH") else []}
    guard_quiet = {"spawn_floor_guard": {"stats": {"step_cells": 0, "drop_cells": 0, "stair_cells": 0}}}
    a_rows, b_rows = [], []
    for i in range(6):
        eid = "x/%06d" % i
        cross = i % 2 == 1
        # A: every same-storey episode succeeds, every cross-storey one hits the limit
        ra = _row(not cross, 0.05 if not cross else 9.0, closing={"last_seen_step": 5}, building=off, episode_id=eid, steps=100 + i, spl=0.6 if not cross else 0.0)
        ra["agent_info"]["policy"].update(guard_quiet)
        # B: cross-storey 1 and 3 now succeed after a transition; same-storey 4 is lost to a SAFE_HALT; the rest identical
        if cross and i < 5:
            rb = _row(True, 0.05, closing={"last_seen_step": 5}, building=on(1), episode_id=eid, steps=200, spl=0.3)
        elif i == 4:
            rb = _row(True, 6.0, closing={"last_seen_step": -1}, building=on(0, "SAFE_HALT"), episode_id=eid, steps=150, spl=0.0, success=False)
        else:
            rb = _row(not cross, 0.05 if not cross else 9.0, closing={"last_seen_step": 5}, building=on(0), episode_id=eid, steps=100 + i, spl=0.6 if not cross else 0.0)
        rb["agent_info"]["policy"]["llm_answers"] = {"asked": 0, "replayed": 3, "recorded": 0, "repeated": 0}
        a_rows.append(ra); b_rows.append(rb)
    for name, rows in (("A", a_rows), ("B", b_rows)):
        (tmp_path / name / "x").mkdir(parents=True)
        (tmp_path / name / "x" / "episodes.jsonl").write_text("".join(json.dumps(_record(r)) + "\n" for r in rows))
    report = compare_campaigns.compare(tmp_path / "A", tmp_path / "B", levels, "off", "on")
    g = report["groups"]
    two_thirds = pytest.approx(2 / 3, abs=1e-3)
    assert report["pairs"] == 6 and g["overall"]["SR_a"] == 0.5 and g["overall"]["SR_b"] == two_thirds
    assert g["cross-storey"]["SR_a"] == 0.0 and g["cross-storey"]["SR_b"] == two_thirds and g["cross-storey"]["transitions_b"] == two_thirds
    assert g["same-storey"]["SR_a"] == 1.0 and g["same-storey"]["SR_b"] == two_thirds
    assert report["flips"]["to_success"] == ["x/000001", "x/000003"] and report["flips"]["to_failure"] == ["x/000004"]
    assert report["flips"]["to_failure_mechanisms_in_b"] == {"stop_safe_halt": 1}
    assert report["flips"]["to_success_mechanisms_in_a"] == {"limit_cross_storey_goal": 2}
    assert report["integrity"]["b_quiet"] == 3 and report["integrity"]["b_quiet_but_differs"] == 0, "the untouched episodes 0, 2 and 5 end as in A"
    text = compare_campaigns.markdown(report)
    assert "cross-storey" in text and "stop_safe_halt" in text and "transitions" in text


def _record(row):
    """A complete episodes.jsonl row around the synthetic policy diagnostics, as results_io reads it."""
    from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts["MOVE_FORWARD"] = int(row["steps"])
    return {"schema": "objnav_episode/1", "benchmark": "hm3d", "split": "val", "episode_id": row["episode_id"],
            "scene_id": row["episode_id"].split("/")[0], "target_category": row["target_category"], "agent": "rpt",
            "success": bool(row["success"]), "spl": float(row["spl"]), "soft_spl": float(row["spl"]),
            "distance_to_goal_m": float(row["distance_to_goal_m"]), "path_length_m": float(row["path_length_m"]),
            "observed_path_length_m": float(row["path_length_m"]), "shortest_path_m": 4.0, "start_distance_to_goal_m": 4.0,
            "steps": int(row["steps"]), "stop_called": bool(row["stop_called"]), "termination": row["termination"],
            "action_counts": counts, "wall_s": float(row["wall_s"]), "agent_error": row["agent_error"],
            "agent_info": row["agent_info"], "native_metrics": {}}
