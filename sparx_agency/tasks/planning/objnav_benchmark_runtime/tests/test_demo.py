"""Single-scene selection and real recorder/report wiring with explicit test fixtures."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_dashboard, write_live_page
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.demo import evaluation_command, validate_demo
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import EpisodeRecorder, PolicyProbe, RecordingAgent, RecordingEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_gibson import LinePolicy, LineSimulator, write_dataset


def semantic_map():
    grid = np.zeros((7, 120, 120), np.uint8)
    grid[0] = 1
    grid[1:, 60, 20] = 1
    return grid


def test_selected_scene_does_not_require_other_scene_meshes(tmp_path):
    episodes, assets = write_dataset(tmp_path, semantic_map(), SCENES, 200)
    for scene in SCENES[1:]:
        for suffix in (".glb", ".navmesh"):
            (assets / (scene + suffix)).unlink()
    dataset = GibsonDataset(episodes, assets, full=False, scene=SCENES[0])
    assert len(dataset.episodes) == 200
    assert {row.scene for row in dataset.episodes.values()} == {SCENES[0]}
    assert len(dataset.manifest()["files"]) == 4
    with pytest.raises(ValueError):
        GibsonDataset(episodes, assets, full=True, scene=SCENES[0])


def test_demo_command_always_selects_one_scene_and_small_limit(monkeypatch, tmp_path):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import demo
    monkeypatch.setattr(demo, "environment_python", lambda *args: "/existing/python")
    config = dict(scene="Collierville", episodes=1, scenes_dir="/scenes", episodes_dir="/episodes",
                  allow_version_mismatch=True)
    command = evaluation_command(config, tmp_path)
    assert command[command.index("--scene") + 1] == "Collierville"
    assert command[command.index("--limit") + 1] == "1"
    assert "--record" in command and "--shards" not in command
    with pytest.raises(ValueError):
        validate_demo(dict(config, episodes=1000))
    with pytest.raises(FileNotFoundError, match="Real Gibson"):
        validate_demo(config)


class MemoryVideo:
    def __init__(self, path, fps):
        self.frames = 0

    def write(self, frame):
        assert frame.shape == (900, 1600, 3) and frame.dtype == np.uint8
        self.frames += 1

    def close(self):
        pass


def test_recording_wraps_actual_runner_and_keeps_gt_out_of_decisions(tmp_path):
    episodes, assets = write_dataset(tmp_path, semantic_map())
    base = GibsonEnv(GibsonDataset(episodes, assets, full=False, scene=SCENES[0]), simulator=LineSimulator())
    policy = LinePolicy()
    output = tmp_path / "recorded"
    output.mkdir()
    recorder = EpisodeRecorder(output, policy, writer_factory=MemoryVideo)
    probe = PolicyProbe(policy)
    agent = RecordingAgent(HeadlessObjNavAgent(probe, gibson_label_mapper()), probe, recorder)
    env = RecordingEnv(base, recorder)
    write_live_page(output)
    with MetricsLogger(output, {"test_fixture": True}) as logger:
        summary = run_benchmark(env, agent, logger=logger, require_stop_for_success=False,
                                path_length_dimension="planar", path_length_epsilon_m=1e-5,
                                progress=lambda i, n, row: recorder.complete(row),
                                kinematics=PROTOCOL.kinematics())
    env.close()
    assert summary.overall.success_rate == 1
    directory = next((output / "recordings").iterdir())
    decisions = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    assert len(decisions) == 5 and decisions[-1]["decision"]["action"] == "STOP"
    assert all("distance_to_goal_m" not in row for row in decisions)
    with (directory / "trajectory.csv").open() as stream:
        trajectory = list(csv.DictReader(stream))
    assert len(trajectory) == 6 and float(trajectory[-1]["distance_to_goal_m"]) == 0
    report = write_dashboard(output)
    text = report.read_text()
    assert "NOT a full Gibson benchmark" in text and "SoftSPL" in text
    for artifact in ("latest.jpg", "metrics.png", "metrics.csv", "demo_metrics.json", "live.html"):
        assert (output / artifact).is_file()
    assert (directory / "trajectory.png").is_file()


def test_metrics_cannot_be_fabricated_for_an_empty_recording(tmp_path):
    (tmp_path / "episodes.jsonl").write_text("")
    with pytest.raises(ValueError, match="No completed episode"):
        write_dashboard(tmp_path)


def test_step_limit_success_is_explicitly_distinguished_from_stop(tmp_path):
    from dataclasses import replace
    from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_run_control import record

    row = record("Collierville/000000")
    counts = dict.fromkeys(row.action_counts, 0)
    counts.update(MOVE_FORWARD=120, TURN_LEFT=380)
    row = replace(row, success=True, spl=0.1, soft_spl=0.1, distance_to_goal_m=0.0,
                  path_length_m=30.0, observed_path_length_m=30.0,
                  steps=500, stop_called=False, termination="step_limit", action_counts=counts)
    with MetricsLogger(tmp_path, {"test_fixture": True}) as logger:
        logger.log(row)
        logger.finish(summarise([row]))
    report = write_dashboard(tmp_path)
    assert "Protocol success without STOP: 1 episode(s)" in report.read_text()
    with (tmp_path / "metrics.csv").open() as stream:
        exported = next(csv.DictReader(stream))
    assert exported["success"] == "1"
    assert exported["stop_called"] == "False"
    assert exported["termination"] == "step_limit"


