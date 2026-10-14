"""The live progress file, the console line, the lean record, and their place in a real run."""
from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES, TERMINATION_STEP_LIMIT, TERMINATION_STOP, EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.progress import (
    LEAN_DROP_KEYS, LeanHeadlessObjNavAgent, ProgressEnv, ProgressTracker, human_duration, lean_episode_info)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_gibson import (
    LinePolicy, LineSimulator, semantic, write_dataset)  # noqa: F401  (the fixture)


def record(episode_id, scene, category, success, spl, dtg, steps=100, wall_s=60.0, termination=TERMINATION_STOP,
           error=None):
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts["MOVE_FORWARD"] = steps
    return EpisodeRecord(benchmark="gibson", split="val", episode_id=episode_id, scene_id=scene, target_category=category,
                         agent="rpt", success=success, spl=spl if success else 0.0, soft_spl=spl, distance_to_goal_m=dtg,
                         path_length_m=5.0, observed_path_length_m=5.0, shortest_path_m=4.0, start_distance_to_goal_m=4.0,
                         steps=steps, stop_called=termination != TERMINATION_STEP_LIMIT, termination=termination,
                         action_counts=counts, wall_s=wall_s, agent_error=error)


def test_the_tracker_keeps_running_metrics_per_scene_and_category_and_writes_the_file_after_every_episode(tmp_path):
    path = tmp_path / "progress.json"
    tracker = ProgressTracker(path, total=4, label="unit")
    first = json.loads(path.read_text())
    assert first["total"] == 4 and first["done"] == 0 and first["percent"] == 0.0 and first["eta_s"] is None
    assert first["current"] is None and first["last"] is None and first["finished_utc"] is None
    tracker.begin("Collierville/000000")
    assert json.loads(path.read_text())["current"]["episode_id"] == "Collierville/000000"
    line = tracker.update(1, 4, record("Collierville/000000", "Collierville", "toilet", True, 0.8, 0.0, steps=120, wall_s=90.0))
    snap = json.loads(path.read_text())
    assert snap["done"] == 1 and snap["percent"] == 25.0 and snap["current"] is None
    assert snap["overall"] == {"episodes": 1, "SR": 1.0, "SPL": 0.8, "SoftSPL": 0.8, "DTG_m": 0.0, "unreachable_end": 0,
                               "mean_steps": 120.0, "stop_rate": 1.0, "agent_errors": 0, "mean_wall_s": 90.0}
    assert snap["eta_s"] == pytest.approx(3 * 90.0) and snap["eta_human"] == "4m30s"
    assert snap["last"]["episode_id"] == "Collierville/000000" and snap["last"]["SR"] == 1
    assert line.startswith("[") and "1/4  25.0%" in line and "SR 1.000 SPL 0.800 DTG 0.00" in line and "ETA 4m30s" in line
    assert "last Collierville/000000 toilet SR=1 SPL=0.800 steps=120 (90s)" in line
    tracker.update(2, 4, record("Corozal/000001", "Corozal", "chair", False, 0.0, None, steps=500, wall_s=30.0,
                                termination=TERMINATION_STEP_LIMIT))
    tracker.update(3, 4, record("Corozal/000002", "Corozal", "toilet", False, 0.0, 2.5, steps=500, wall_s=30.0,
                                termination="agent_error", error="RuntimeError: boom"))
    snap = json.loads(path.read_text())
    assert snap["overall"]["SR"] == pytest.approx(1 / 3, abs=1e-4) and snap["overall"]["unreachable_end"] == 1
    assert snap["overall"]["DTG_m"] == pytest.approx(1.25) and snap["overall"]["agent_errors"] == 1
    assert snap["by_scene"]["Corozal"]["episodes"] == 2 and snap["by_scene"]["Collierville"]["SR"] == 1.0
    assert snap["by_category"]["toilet"]["episodes"] == 2 and snap["by_category"]["chair"]["SR"] == 0.0
    assert snap["terminations"] == {"agent_error": 1, TERMINATION_STEP_LIMIT: 1, TERMINATION_STOP: 1}
    assert snap["eta_s"] == pytest.approx(150.0 / 3), "the session's mean episode time times the episodes left"
    assert "AGENT ERROR" in tracker.console_line()
    tracker.finish()
    assert json.loads(path.read_text())["finished_utc"] is not None


def test_a_resumed_run_is_seeded_with_the_rows_on_disk_but_the_eta_comes_from_this_session(tmp_path):
    done = [record("Darden/%06d" % i, "Darden", "bed", i % 2 == 0, 0.5, 0.0 if i % 2 == 0 else 1.0, wall_s=1000.0)
            for i in range(6)]
    tracker = ProgressTracker(tmp_path / "p.json", total=10, completed=done)
    snap = tracker.snapshot()
    assert snap["done"] == 6 and snap["resumed_from"] == 6 and snap["percent"] == 60.0 and snap["overall"]["SR"] == 0.5
    assert snap["eta_s"] is None, "no episode of this session yet"
    tracker.update(7, 10, record("Darden/000006", "Darden", "bed", True, 1.0, 0.0, wall_s=20.0))
    snap = tracker.snapshot()
    assert snap["done"] == 7 and snap["session_episodes"] == 1 and snap["eta_s"] == pytest.approx(3 * 20.0)
    assert human_duration(3 * 3600 + 12 * 60) == "3h12m" and human_duration(47 * 60 + 5) == "47m05s" and human_duration(55) == "55s"
    assert human_duration(74 * 3600 + 120) == "3d02h"


def test_lean_episode_info_keeps_the_counters_and_drops_the_series():
    full = {"agent": "rpt", "steps": 120, "idle_steps": 3, "statuses": {"ok": 120}, "blocked_notifications": 2,
            "policy": {"method": "rpt", "rooms": 5, "llm_queries": 7, "oracle_reuses": 2,
                       "target_closing": {"active": False, "locks": 1},
                       "room_search_loop": {"stats": {"scans_completed": 4}, "events": [{"e": i} for i in range(500)],
                                            "estimate_events": [1, 2], "estimates": {"0": {}}, "second_pass": True,
                                            "excluded": {}, "finished": {"0": {"how": "scan_point_inside", "step": 7}},
                                            "order": [1, 0], "settings": {"x": 1}},
                       "exploration_fallback": {"stats": {"frontier": 3}, "failures": [1, 2, 3], "settings": {}},
                       "exploration_metrics": {"coverage_curve": [{"a": 1}] * 400, "observed_gain_m2": 42.0,
                                               "latency": {"astar": {"count": 3}}, "coverage_definition": "..."},
                       "last_reasoning": {"oracle": {"source": "llm", "reading": {"pass": "second"}, "pass_verdict": "second",
                                                     "probs": {"0": 0.3, "1": 0.1}}, "nodes": [1, 2, 3]},
                       "solver_records": [1] * 50, "room_label_history": [1] * 50,
                       "room_scans": {"records": [1] * 20, "finished_here": [0, 1]},
                       "building": {"big": [0] * 1000}}}
    lean = lean_episode_info(full)
    assert lean["lean"] is True and lean["steps"] == 120 and lean["statuses"] == {"ok": 120}
    policy = lean["policy"]
    assert policy["method"] == "rpt" and policy["rooms"] == 5 and policy["llm_queries"] == 7
    assert policy["target_closing"] == {"active": False, "locks": 1}
    assert policy["room_search_loop"] == {"stats": {"scans_completed": 4}, "second_pass": True, "excluded": {},
                                          "finished": {"0": {"how": "scan_point_inside", "step": 7}}, "order": [1, 0]}
    assert policy["exploration_fallback"] == {"stats": {"frontier": 3}, "failures": 3}
    assert policy["exploration_metrics"] == {"observed_gain_m2": 42.0, "latency": {"astar": {"count": 3}}}
    assert policy["last_oracle"] == {"source": "llm", "reading": {"pass": "second"}, "pass_verdict": "second", "nodes": 2}
    assert policy["room_scans"] == {"finished_here": [0, 1]}, "the records series is dropped inside a kept block"
    for gone in ("solver_records", "room_label_history", "building", "last_reasoning"):
        assert gone not in policy
    assert len(json.dumps(lean)) < 0.05 * len(json.dumps(full))
    assert "coverage_curve" in LEAN_DROP_KEYS and "events" in LEAN_DROP_KEYS
    assert lean_episode_info({"agent": "x"}) == {"agent": "x"}, "no policy block: nothing to trim"


def test_the_real_runner_writes_progress_and_lean_rows(tmp_path, semantic):
    paths = write_dataset(tmp_path, semantic)
    scorer = GibsonEnv(GibsonDataset(*paths, full=False), simulator=LineSimulator())
    scorer.validate_starts()
    tracker = ProgressTracker(tmp_path / "progress.json", total=len(scorer.episode_ids()))
    env = ProgressEnv(scorer, tracker)
    assert env.episode_ids() == scorer.episode_ids() and env.name == scorer.name
    agent = LeanHeadlessObjNavAgent(LinePolicy(), gibson_label_mapper(), name="synthetic-line-test")
    config = {"protocol": asdict(PROTOCOL), "selected_episode_ids": list(env.episode_ids()), "full_split": False,
              "method": {"method": agent.name}, "reference_sim_version_match": False}
    seen = []

    def progress(i, n, row):
        seen.append(tracker.update(i, n, row))

    with MetricsLogger(tmp_path / "results", config) as logger:
        summary = run_benchmark(env, agent, logger=logger, require_stop_for_success=False, path_length_dimension="planar",
                                path_length_epsilon_m=1e-5, kinematics=PROTOCOL.kinematics(), progress=progress)
    tracker.finish()
    assert summary.overall.success_rate == 1 and len(seen) == 1 and "1/1 100.0%" in seen[0]
    snap = json.loads((tmp_path / "progress.json").read_text())
    assert snap["done"] == 1 and snap["finished_utc"] and snap["overall"]["SR"] == 1.0 and snap["current"] is None
    row = json.loads((tmp_path / "results" / "episodes.jsonl").read_text().splitlines()[0])
    assert row["agent_info"]["lean"] is True and "policy" in row["agent_info"]
    assert env.evaluation_diagnostics()["distance_to_goal_m"] == 0.0, "the scorer's diagnostics pass through the wrapper"
    env.close()
