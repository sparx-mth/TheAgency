"""Five-scene campaign driver: forwarded flags, per-episode console line, final table, saved runtime."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES, EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import five_scene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene_console import (
    failure_line, recording_video, status_line, summary_table)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES


def record(scene, *, success=True, wall_s=12.5, dtg=0.4, steps=37, index=0):
    return EpisodeRecord(
        benchmark="gibson", split="val", episode_id="%s/%06d" % (scene, index), scene_id=scene,
        target_category="toilet", agent="test-agent", success=success,
        spl=0.5 if success else 0.0, soft_spl=0.5, distance_to_goal_m=dtg,
        path_length_m=6.0, observed_path_length_m=6.0, shortest_path_m=3.0,
        start_distance_to_goal_m=3.0, steps=steps, stop_called=True, termination="stop",
        action_counts={key: {"STOP": 1, "MOVE_FORWARD": steps - 1}.get(key, 0) for key in ACTION_NAMES},
        wall_s=wall_s)


def write_scored_episodes(directory, rows, allow_stair_traversal=False):
    config = dict(protocol=asdict(PROTOCOL), runtime={"habitat-sim": "test"}, source_sha256="synthetic",
                  method={"method": "test-agent", "allow_stair_traversal": allow_stair_traversal}, seed=0,
                  reference_sim_version_match=False, dataset={"synthetic": True}, shards=1, shard_index=0,
                  limit=len(rows), full_split=False, selected_episode_ids=[row.episode_id for row in rows],
                  on_agent_error="record")
    with MetricsLogger(directory, config) as logger:
        for row in rows:
            logger.log(row)
        logger.finish(summarise(rows))


def write_scored_episode(directory, row, **kwargs):
    write_scored_episodes(directory, [row], **kwargs)


def test_status_line_names_scene_goal_and_every_requested_metric(tmp_path):
    row = record("Darden", wall_s=61.25, dtg=0.75, steps=88)
    text = status_line(row, tmp_path / "video.mp4", job_s=90.4)
    for fragment in ("EPISODE COMPLETE", "scene=Darden", "goal=toilet", "SR=1", "SPL=0.5000",
                     "DTG=0.750 m", "runtime=61.2 s", "steps=88", "termination=stop", "job=90 s",
                     "video=%s" % (tmp_path / "video.mp4")):
        assert fragment in text
    assert "video=not written" in status_line(record("Darden"))
    assert "DTG=n/a m" in status_line(record("Darden", dtg=None, success=False))
    assert failure_line("Corozal", 3, tmp_path / "run.log") == (
        "EPISODE FAILED    scene=Corozal  exit=3  no scored episode; see %s" % (tmp_path / "run.log"))


def test_summary_table_lists_every_episode_and_a_mean_row():
    rows = [record("Collierville", success=True, wall_s=10.0, dtg=0.0, steps=20),
            record("Corozal", success=False, wall_s=30.0, dtg=4.0, steps=60)]
    table = summary_table(rows)
    lines = table.splitlines()
    assert lines[0].split() == ["scene", "goal", "SR", "SPL", "DTG_m", "runtime_s", "steps", "termination"]
    assert lines[2].split() == ["Collierville", "toilet", "1", "0.5000", "0.000", "10.0", "20", "stop"]
    assert lines[3].split() == ["Corozal", "toilet", "0", "0.0000", "4.000", "30.0", "60", "stop"]
    assert lines[-1].split() == ["MEAN", "(2", "ep)", "0.500", "0.2500", "2.000", "20.0", "40.0"]
    with pytest.raises(ValueError):
        summary_table([])


def test_recording_video_uses_the_recorder_key(tmp_path):
    row = record("Wiconisco")
    key = hashlib.sha256(row.episode_id.encode()).hexdigest()[:12]
    assert recording_video(tmp_path, row) is None
    video = tmp_path / "recordings" / key / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\0")
    assert recording_video(tmp_path, row) == video


class FakeProcess:
    """A finished `gibson.run` child: its side effects were applied when it was started."""

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def benchmark_run_fake(monkeypatch, on_benchmark):
    """Intercept only the `gibson.run` child; `MetricsLogger` still shells out to git."""
    real_popen = five_scene.subprocess.Popen

    def fake_popen(command, *args, **kwargs):
        if "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run" not in command:
            return real_popen(command, *args, **kwargs)
        return FakeProcess(on_benchmark(command))

    monkeypatch.setattr(five_scene.subprocess, "Popen", fake_popen)


def test_main_forwards_flags_prints_each_episode_and_a_table(tmp_path, monkeypatch, capsys):
    commands = []

    def on_benchmark(command):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        scene = command[command.index("--scene") + 1]
        write_scored_episode(output, record(scene, success=scene != "Darden", dtg=0.0 if scene != "Darden" else 2.5))
        return 0

    benchmark_run_fake(monkeypatch, on_benchmark)
    output = tmp_path / "campaign"
    code = five_scene.main(["--episodes-dir", "/episodes", "--scenes-dir", "/scenes", "--output", str(output),
                            "--detector-url", "http://127.0.0.1:18095", "--allow-shared-gpu",
                            "--allow-sim-version-mismatch", "--gpu-device", "0", "--poll-s", "0"])
    assert code == 0
    assert [c[c.index("--scene") + 1] for c in commands] == list(SCENES)
    for command in commands:
        assert "--allow-shared-gpu" in command and "--allow-sim-version-mismatch" in command
        assert command[command.index("--detector-backend") + 1] == "yolo_world"
        assert command[command.index("--detector-url") + 1] == "http://127.0.0.1:18095"
        assert command[command.index("--gpu-device") + 1] == "0"
        assert command[command.index("--limit") + 1] == "1" and "--record" in command
        policy_config = Path(command[command.index("--policy-config") + 1])
        assert json.loads(policy_config.read_text()) == {"allow_stair_traversal": False}
    out = capsys.readouterr().out
    for scene in SCENES:
        assert "EPISODE COMPLETE  scene=%s  goal=toilet  SR=%d" % (scene, scene != "Darden") in out
    assert out.index("EPISODE COMPLETE  scene=Collierville") < out.index("Starting Corozal")
    assert "FIVE-SCENE SUMMARY (5/5 episodes scored" in out
    assert out.count("RUNNING MEAN      episodes=") == 5 and "episodes=5/5  SR=0.800" in out
    assert (output / "summary.txt").read_text().splitlines()[-1].startswith("MEAN (5 ep)")
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scene"] for row in rows] == list(SCENES)
    assert all(float(row["runtime_s"]) == 12.5 for row in rows)


def test_main_reports_a_scene_that_ended_without_a_scored_row(tmp_path, monkeypatch, capsys):
    benchmark_run_fake(monkeypatch, lambda command: 2)
    code = five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(tmp_path / "out"),
                            "--poll-s", "0"])
    out = capsys.readouterr().out
    assert code == 1
    assert out.count("EPISODE FAILED    scene=") == len(SCENES)
    assert "FIVE-SCENE SUMMARY: no scene produced a scored episode" in out
    assert not (tmp_path / "out" / "summary.txt").exists()


def test_three_episodes_per_scene_stream_results_and_running_means(tmp_path, monkeypatch, capsys):
    def on_benchmark(command):
        output = Path(command[command.index("--output") + 1])
        scene = command[command.index("--scene") + 1]
        assert command[command.index("--limit") + 1] == "3"
        rows = [record(scene, index=i, success=i != 1, dtg=0.0 if i != 1 else 3.0, steps=10 * (i + 1))
                for i in range(3)]
        write_scored_episodes(output, rows)
        return 0

    benchmark_run_fake(monkeypatch, on_benchmark)
    output = tmp_path / "campaign"
    code = five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(output),
                            "--episodes-per-scene", "3", "--poll-s", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("EPISODE COMPLETE") == 15 and out.count("RUNNING MEAN") == 15
    assert "RUNNING MEAN      episodes=15/15  SR=0.667" in out
    assert out.count("SCENE DONE        scene=") == 5
    assert "FIVE-SCENE SUMMARY (15/15 episodes scored; the first 3 published episode(s) per scene; Collierville" in out
    results = json.loads((output / "benchmark_results.json").read_text())
    assert results["allow_stair_traversal"] is False and results["episodes_total"] == 15
    assert results["episodes_scored"] == 15 and len(results["episodes"]) == 15
    assert [row["index"] for row in results["episodes"]] == list(range(1, 16))
    assert results["running_summary"]["overall"]["n_episodes"] == 15
    with (output / "benchmark_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15 and rows[0]["episode_id"] == "Collierville/000000" and rows[4]["SR"] == "0"
    assert (output / "summary.txt").read_text().splitlines()[-1].startswith("MEAN (15 ep)")
    with (output / "metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 15


def test_a_job_that_allowed_stairs_is_refused(tmp_path, monkeypatch):
    def on_benchmark(command):
        output = Path(command[command.index("--output") + 1])
        write_scored_episode(output, record(command[command.index("--scene") + 1]), allow_stair_traversal=True)
        return 0

    benchmark_run_fake(monkeypatch, on_benchmark)
    with pytest.raises(RuntimeError, match="allow_stair_traversal=True"):
        five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(tmp_path / "out"),
                         "--poll-s", "0"])


def test_a_scene_subset_runs_only_those_scenes_in_the_given_order(tmp_path, monkeypatch, capsys):
    seen = []

    def on_benchmark(command):
        scene = command[command.index("--scene") + 1]
        seen.append(scene)
        write_scored_episodes(Path(command[command.index("--output") + 1]),
                              [record(scene, index=i) for i in range(2)])
        return 0

    benchmark_run_fake(monkeypatch, on_benchmark)
    output = tmp_path / "campaign"
    code = five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(output),
                            "--episodes-per-scene", "2", "--scenes", "Wiconisco", "Darden", "--poll-s", "0"])
    assert code == 0 and seen == ["Wiconisco", "Darden"]
    out = capsys.readouterr().out
    assert out.count("EPISODE COMPLETE") == 4 and "episodes=4/4" in out
    assert "FIVE-SCENE SUMMARY (4/4 episodes scored; the first 2 published episode(s) per scene; Wiconisco, Darden)" in out
    results = json.loads((output / "benchmark_results.json").read_text())
    assert results["scenes"] == ["Wiconisco", "Darden"] and results["episodes_total"] == 4
    assert [row["scene"] for row in results["episodes"]] == ["Wiconisco", "Wiconisco", "Darden", "Darden"]
    with (output / "metrics.csv").open() as stream:
        assert [row["scene"] for row in csv.DictReader(stream)] == ["Wiconisco", "Wiconisco", "Darden", "Darden"]
    with pytest.raises(SystemExit):
        five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(tmp_path / "x"),
                         "--scenes", "Darden", "Darden"])
    with pytest.raises(SystemExit):
        five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(tmp_path / "y"),
                         "--scenes", "Nowhere"])
