"""Five-scene campaign driver: forwarded flags, per-episode console line, final table, saved runtime."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES, EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import five_scene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.five_scene_console import (
    failure_line, recording_video, status_line, summary_table)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES


def record(scene, *, success=True, wall_s=12.5, dtg=0.4, steps=37):
    return EpisodeRecord(
        benchmark="gibson", split="val", episode_id="%s/000000" % scene, scene_id=scene,
        target_category="toilet", agent="test-agent", success=success,
        spl=0.5 if success else 0.0, soft_spl=0.5, distance_to_goal_m=dtg,
        path_length_m=6.0, observed_path_length_m=6.0, shortest_path_m=3.0,
        start_distance_to_goal_m=3.0, steps=steps, stop_called=True, termination="stop",
        action_counts={key: {"STOP": 1, "MOVE_FORWARD": steps - 1}.get(key, 0) for key in ACTION_NAMES},
        wall_s=wall_s)


def write_scored_episode(directory, row):
    config = dict(protocol=asdict(PROTOCOL), runtime={"habitat-sim": "test"}, source_sha256="synthetic",
                  method={"method": "test-agent"}, seed=0, reference_sim_version_match=False,
                  dataset={"synthetic": True}, shards=1, shard_index=0, limit=1, full_split=False,
                  selected_episode_ids=[row.episode_id], on_agent_error="record")
    with MetricsLogger(directory, config) as logger:
        logger.log(row)
        logger.finish(summarise([row]))


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


def benchmark_run_fake(monkeypatch, on_benchmark):
    """Intercept only the `gibson.run` child; `MetricsLogger` still shells out to git."""
    real_run = five_scene.subprocess.run

    def fake_run(command, *args, **kwargs):
        if "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run" not in command:
            return real_run(command, *args, **kwargs)
        return on_benchmark(command)

    monkeypatch.setattr(five_scene.subprocess, "run", fake_run)


def test_main_forwards_flags_prints_each_episode_and_a_table(tmp_path, monkeypatch, capsys):
    commands = []

    def on_benchmark(command):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        scene = command[command.index("--scene") + 1]
        write_scored_episode(output, record(scene, success=scene != "Darden", dtg=0.0 if scene != "Darden" else 2.5))
        return type("Result", (), {"returncode": 0})()

    benchmark_run_fake(monkeypatch, on_benchmark)
    output = tmp_path / "campaign"
    code = five_scene.main(["--episodes-dir", "/episodes", "--scenes-dir", "/scenes", "--output", str(output),
                            "--detector-url", "http://127.0.0.1:18095", "--allow-shared-gpu",
                            "--allow-sim-version-mismatch", "--gpu-device", "0"])
    assert code == 0
    assert [c[c.index("--scene") + 1] for c in commands] == list(SCENES)
    for command in commands:
        assert "--allow-shared-gpu" in command and "--allow-sim-version-mismatch" in command
        assert command[command.index("--detector-backend") + 1] == "yolo_world"
        assert command[command.index("--detector-url") + 1] == "http://127.0.0.1:18095"
        assert command[command.index("--gpu-device") + 1] == "0"
        assert command[command.index("--limit") + 1] == "1" and "--record" in command
    out = capsys.readouterr().out
    for scene in SCENES:
        assert "EPISODE COMPLETE  scene=%s  goal=toilet  SR=%d" % (scene, scene != "Darden") in out
    assert out.index("EPISODE COMPLETE  scene=Collierville") < out.index("Starting Corozal")
    assert "FIVE-SCENE SUMMARY (5/5 scenes scored" in out
    assert (output / "summary.txt").read_text().splitlines()[-1].startswith("MEAN (5 ep)")
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scene"] for row in rows] == list(SCENES)
    assert all(float(row["runtime_s"]) == 12.5 for row in rows)


def test_main_reports_a_scene_that_ended_without_a_scored_row(tmp_path, monkeypatch, capsys):
    benchmark_run_fake(monkeypatch, lambda command: type("Result", (), {"returncode": 2})())
    code = five_scene.main(["--episodes-dir", "/e", "--scenes-dir", "/s", "--output", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 1
    assert out.count("EPISODE FAILED    scene=") == len(SCENES)
    assert "FIVE-SCENE SUMMARY: no scene produced a scored episode" in out
    assert not (tmp_path / "out" / "summary.txt").exists()
