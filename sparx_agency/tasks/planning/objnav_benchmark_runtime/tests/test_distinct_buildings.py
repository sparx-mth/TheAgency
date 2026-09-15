"""Distinct-building generation/recording contracts, without licensed data or GPU."""
import bz2
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import generate_development as generation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import SCHEMA, SPLIT, DevelopmentDataset, DevelopmentEnv, file_identity, load_training_maps
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distinct_buildings import command_for, jobs_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_gibson import LineSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import method_snapshot, render_dashboard
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy


def test_selection_is_distinct_deterministic_and_order_independent():
    names = ["Train%02d" % i for i in range(25)]
    selected = generation.select_buildings(names, 15, 0)
    assert selected == generation.select_buildings(reversed(names), 15, 0)
    assert len(selected) == len(set(selected)) == 15
    assert selected != generation.select_buildings(names, 15, 1)
    with pytest.raises(ValueError):
        generation.select_buildings(names, 26, 0)


def test_sampler_is_bounded_and_deterministic(monkeypatch):
    class Field:
        def __init__(self, *args):
            self.cells = np.full((4, 4), 50.0)
            self.unreachable = np.zeros((4, 4), bool)
        def map_cell(self, point):
            return (0, 0)
    class Pathfinder:
        def seed(self, seed):
            self.seed_value = seed
        def get_random_navigable_point(self):
            return (0.0, 0.18, 0.0)
        def is_navigable(self, point):
            return True
    monkeypatch.setattr(generation, "GibsonDistanceField", Field)
    semantic = np.zeros((7, 4, 4), np.uint8)
    semantic[0] = 1
    semantic[1, 0, 0] = 1
    floors = {0: {"sem_map": semantic, "floor_height": 0.0, "origin": [0, 0]}}
    first = generation.generate_start("TrainA", floors, Pathfinder(), 7, attempts=2)
    second = generation.generate_start("TrainA", floors, Pathfinder(), 7, attempts=2)
    assert first == second and first[0]["object_category"] == "chair"
    assert np.linalg.norm(first[0]["start_rotation"]) == pytest.approx(1.0)
    bad = Pathfinder()
    bad.get_random_navigable_point = lambda: (0.0, 9.0, 0.0)
    with pytest.raises(RuntimeError, match="No valid ObjectNav start"):
        generation.generate_start("TrainA", floors, bad, 7, attempts=2)


def development_manifest(tmp_path):
    semantic = np.zeros((7, 120, 120), np.uint8)
    semantic[0] = 1
    semantic[1, 60, 20] = 1
    maps = tmp_path / "train_info.pbz2"
    with bz2.open(maps, "wb") as stream:
        pickle.dump({"TrainA": {0: {"sem_map": semantic, "origin": np.array([0, 0]), "floor_height": 0.0}}}, stream)
    assets = []
    for suffix in (".glb", ".navmesh"):
        path = tmp_path / ("TrainA" + suffix)
        path.write_bytes(b"synthetic-test-asset")
        assets.append(file_identity(path))
    data = {"schema": SCHEMA, "split": SPLIT, "scenes": ["TrainA"], "scenes_dir": str(tmp_path),
            "train_info": file_identity(maps), "assets": assets, "generation": {"policy_feedback_used": False},
            "episodes": [{"scene": "TrainA", "object_category": "chair", "object_id": 0, "floor_id": 0,
                          "start_position": [3.0, 0.0, 3.0], "start_rotation": [1, 0, 0, 0]}]}
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(data))
    return path


def test_generated_data_is_not_validation_and_refuses_asset_drift(tmp_path):
    path = development_manifest(tmp_path)
    dataset = DevelopmentDataset(path)
    env = DevelopmentEnv(dataset, simulator=LineSimulator())
    try:
        episode, obs = env.reset("TrainA/000000")
        assert episode.split == SPLIT and episode.max_steps == 500
        assert not episode.metadata
        assert not hasattr(obs, "goal_position")
    finally:
        env.close()
    (tmp_path / "TrainA.glb").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        DevelopmentDataset(path)


def test_validation_maps_cannot_be_relabeled_training(tmp_path):
    path = tmp_path / "train_info.pbz2"
    with bz2.open(path, "wb") as stream:
        pickle.dump({SCENES[0]: {0: {}}}, stream)
    with pytest.raises(ValueError, match="separate Gibson training"):
        load_training_maps(path)


def test_each_building_has_both_recorded_jobs_with_fixed_action_protocol(tmp_path):
    jobs = jobs_for(["A", "B"])
    assert jobs == [("frontier", "A"), ("falcon", "A"), ("falcon", "B"), ("frontier", "B")]
    args = SimpleNamespace(manifest=tmp_path / "episodes.json", output=tmp_path / "out", seed=0,
                           video_fps=6, detector_url="http://127.0.0.1:18095", detector_backend="yolo_world",
                           policy_config=None, allow_sim_version_mismatch=True)
    command = command_for(args, "falcon", "A")
    assert "--record" in command and "--video-fps" in command
    assert "--max-steps" not in command
    assert command[command.index("--explorer") + 1] == "falcon"


def test_falcon_recording_shows_actual_phase_and_renders():
    old, episode = setup_policy("bed")
    policy = RPTSearchPolicy(old.detector, old.llm_client, replace(old.settings, local_exploration="falcon"))
    policy.reset(episode, old.target)
    obs = observation(episode, 0, depth=3)
    command = policy.plan(obs)
    snapshot = method_snapshot(policy)
    assert snapshot["state"] == policy.hierarchy.machine.phase
    assert snapshot["explorer"] == "falcon" and snapshot["burst_actions_left"] == 48
    frame = render_dashboard(policy, obs, [(0, 0)], {"action": "TURN_LEFT", "info": command.info}, episode.episode_id, snapshot)
    assert frame.shape == (900, 1600, 3)

