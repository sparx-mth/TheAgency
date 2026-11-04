"""The detached-run monitor: the scalar CSV, the status line, and how it tells a finished run from a running one."""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, Optional

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.monitor import (
    CSV_COLUMNS, EXIT_CODE_UNKNOWN, Monitor, episodes_csv_text, launcher_started, main, read_episode_scalars,
    run_finished, status_line, write_episodes_csv)


def episode_row(episode_id: str = "Darden/000000", success: bool = True, dtg: Optional[float] = 0.0,
                error: Optional[str] = None) -> Dict[str, Any]:
    """A lean ``episodes.jsonl`` row as the harness writes it, diagnostics included."""
    return {"schema": "objnav_episode/1", "benchmark": "gibson", "split": "val", "episode_id": episode_id,
            "scene_id": episode_id.split("/")[0], "target_category": "bed", "agent": "rpt", "success": success,
            "spl": 0.4511 if success else 0.0, "soft_spl": 0.4511, "distance_to_goal_m": dtg,
            "path_length_m": 16.22, "observed_path_length_m": 16.22, "shortest_path_m": 7.32,
            "start_distance_to_goal_m": 6.32, "steps": 283, "stop_called": True, "termination": "stop",
            "action_counts": {"MOVE_FORWARD": 65, "STOP": 1}, "wall_s": 143.1, "agent_error": error,
            "native_metrics": {"success": 1.0}, "agent_info": {"policy": {"rooms": 7, "llm_queries": 3}}}


def write_jsonl(path, rows, truncated_tail=False):
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if truncated_tail:
        text += json.dumps(episode_row("Darden/000099"))[:40]
    path.write_text(text)


def test_the_csv_holds_only_the_scalar_columns_with_booleans_as_digits_and_none_as_empty(tmp_path):
    source = tmp_path / "episodes.jsonl"
    write_jsonl(source, [episode_row(), episode_row("Darden/000001", success=False, dtg=None, error="Boom: x")])
    target = tmp_path / "episodes.csv"
    assert write_episodes_csv(source, target) == 2
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    assert "agent_info" not in rows[0] and "action_counts" not in rows[0] and "native_metrics" not in rows[0]
    assert rows[0]["episode_id"] == "Darden/000000" and rows[0]["success"] == "1" and rows[0]["stop_called"] == "1"
    assert rows[0]["path_length_m"] == "16.22" and rows[0]["start_distance_to_goal_m"] == "6.32"
    assert rows[0]["distance_to_goal_m"] == "0.0" and rows[0]["agent_error"] == ""
    assert rows[1]["success"] == "0" and rows[1]["distance_to_goal_m"] == "" and rows[1]["agent_error"] == "Boom: x"


def test_a_truncated_last_line_and_a_missing_file_are_tolerated(tmp_path):
    source = tmp_path / "episodes.jsonl"
    assert read_episode_scalars(source) == []
    write_jsonl(source, [episode_row()], truncated_tail=True)
    rows = read_episode_scalars(source)
    assert [row["episode_id"] for row in rows] == ["Darden/000000"]
    assert episodes_csv_text([]).splitlines() == [",".join(CSV_COLUMNS)]


def test_the_status_line_reads_the_progress_snapshot():
    assert "no progress.json yet" in status_line(None)
    snapshot: Dict[str, Any] = {
        "done": 212, "total": 1000, "percent": 21.2, "eta_human": "3d02h",
        "overall": {"SR": 0.642, "SPL": 0.331, "DTG_m": 1.52},
        "current": {"episode_id": "Corozal/000012", "started_utc": "2026-10-08T10:00:00Z"},
        "last": {"episode_id": "Corozal/000011", "target": "chair", "SR": 1, "SPL": 0.712, "DTG_m": 0.0,
                 "steps": 143, "wall_s": 412.0, "agent_error": None}}
    line = status_line(snapshot, now=0.0)
    assert line.startswith("1970-01-01T00:00:00Z | 212/1000  21.2% | SR 0.642 SPL 0.331 DTG 1.52 | ETA 3d02h")
    assert "running Corozal/000012" in line and "last Corozal/000011 chair SR=1 SPL=0.712 DTG=0.00 steps=143 (412s)" in line
    assert "AGENT ERROR" not in line
    snapshot["last"]["agent_error"] = "RuntimeError: x"
    snapshot["overall"]["DTG_m"] = None
    assert "DTG n/a" in status_line(snapshot) and status_line(snapshot).endswith("AGENT ERROR")


def test_finished_is_judged_by_the_pid_and_an_exit_code_older_than_the_monitor_does_not_count(tmp_path):
    since = time.time()
    stale = tmp_path / "exit_code.txt"
    stale.write_text("0\n")
    os.utime(stale, (since - 100, since - 100))                       # a preflight into the same directory
    assert run_finished(tmp_path, pid=None, since=since) is None
    assert run_finished(tmp_path, pid=os.getpid(), since=since) is None   # this process is alive
    dead = 2 ** 22 - 1                                               # beyond pid_max on this machine
    assert run_finished(tmp_path, pid=dead, since=since) == EXIT_CODE_UNKNOWN
    stale.write_text("3\n")                                           # the launcher's trap wrote it just now
    assert run_finished(tmp_path, pid=dead, since=since) == 3
    assert run_finished(tmp_path, pid=None, since=since) == 3


def test_the_monitor_ticks_refreshes_the_csv_only_when_the_file_grew_and_concludes_with_the_summary(tmp_path, capsys):
    (tmp_path / "progress.json").write_text(json.dumps(
        {"label": "gibson val", "done": 1, "total": 2, "percent": 50.0, "eta_human": "2m00s", "elapsed_s": 120.0,
         "started_utc": "x", "updated_utc": "y", "overall": {"episodes": 1, "SR": 1.0, "SPL": 0.45, "SoftSPL": 0.45,
                                                              "DTG_m": 0.0, "mean_steps": 283.0, "stop_rate": 1.0,
                                                              "agent_errors": 0, "mean_wall_s": 143.0},
         "by_scene": {}, "by_category": {}, "terminations": {"stop": 1}, "current": None,
         "last": {"episode_id": "Darden/000000", "target": "bed", "SR": 1, "SPL": 0.45, "DTG_m": 0.0, "steps": 283,
                  "wall_s": 143.0, "termination": "stop", "agent_error": None}}))
    write_jsonl(tmp_path / "episodes.jsonl", [episode_row()])
    monitor = Monitor(tmp_path, pid=os.getpid())
    assert monitor.refresh_csv() == 1
    assert monitor.refresh_csv() is None                              # unchanged: not rewritten
    line = monitor.tick()
    assert "1/2  50.0%" in line and (tmp_path / "monitor.log").read_text().strip().endswith(line)
    assert monitor.finished() is None
    write_jsonl(tmp_path / "episodes.jsonl", [episode_row(), episode_row("Darden/000001")])
    assert monitor.refresh_csv() == 2
    text = monitor.conclude(0)
    assert "run completed" in text and "gibson val: 1/2 episodes" in text      # no summary.json: the progress view
    assert (tmp_path / "FINAL_SUMMARY.txt").read_text().startswith(text.splitlines()[0])
    assert main([str(tmp_path), "--once", "--pid", str(os.getpid())]) == 0
    assert "1/2  50.0%" in capsys.readouterr().out


def test_main_anchors_since_on_the_pid_file_so_a_preflight_exit_code_is_ignored_and_the_runs_own_counts(tmp_path, capsys):
    now = time.time()
    stale = tmp_path / "exit_code.txt"
    stale.write_text("0\n")                                            # the preflight's, before the launch
    os.utime(stale, (now - 100, now - 100))
    pid_file = tmp_path / "launcher.pid"
    pid_file.write_text("%d\n" % (2 ** 22 - 1))                       # the launcher wrote it, then died
    os.utime(pid_file, (now - 50, now - 50))
    assert launcher_started(tmp_path) == pytest.approx(now - 50, abs=1.0)
    assert main([str(tmp_path), "--interval", "1"]) == 1
    assert "killed (no exit code recorded)" in capsys.readouterr().out   # the stale 0 did not count
    stale.write_text("1\n")                                            # the launcher's trap, after the pid file
    assert main([str(tmp_path), "--interval", "1"]) == 1
    out = capsys.readouterr().out
    assert "FAILED with exit code 1" in out and "no progress.json" in out
    assert launcher_started(tmp_path / "elsewhere") is None


def test_a_non_positive_interval_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--interval", "0"])

