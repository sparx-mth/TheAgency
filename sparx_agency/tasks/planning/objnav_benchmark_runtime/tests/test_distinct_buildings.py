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
        clearance = 1.0
        def seed(self, seed):
            self.seed_value = seed
        def get_random_navigable_point(self):
            return (0.0, 0.18, 0.0)
        def is_navigable(self, point):
            return True
        def distance_to_closest_obstacle(self, point, max_search_radius=2.0):
            return min(self.clearance, max_search_radius)
    monkeypatch.setattr(generation, "GibsonDistanceField", Field)
    semantic = np.zeros((7, 4, 4), np.uint8)
    semantic[0] = 1
    semantic[1, 0, 0] = 1
    floors = {0: {"sem_map": semantic, "floor_height": 0.0, "origin": [0, 0]}}
    first = generation.generate_start("TrainA", floors, Pathfinder(), 7, attempts=2)
    second = generation.generate_start("TrainA", floors, Pathfinder(), 7, attempts=2)
    assert first == second and first[0]["object_category"] == "chair"
    assert np.linalg.norm(first[0]["start_rotation"]) == pytest.approx(1.0)
    assert first[1]["start_clearance_m"] == 1.0 and first[1]["min_start_clearance_m"] == generation.MIN_START_CLEARANCE_M
    bad = Pathfinder()
    bad.get_random_navigable_point = lambda: (0.0, 9.0, 0.0)
    with pytest.raises(RuntimeError, match="No valid ObjectNav start"):
        generation.generate_start("TrainA", floors, bad, 7, attempts=2)
    # A navigable point 0.2 m from the nearest obstacle is a start beside a bed: boxed in, not a start
    # (Hanson/000002, 2026-10-05). The plain reference sampler is one knob away.
    boxed = Pathfinder()
    boxed.clearance = 0.2
    with pytest.raises(RuntimeError, match="No valid ObjectNav start"):
        generation.generate_start("TrainA", floors, boxed, 7, attempts=2)
    row, audit = generation.generate_start("TrainA", floors, boxed, 7, attempts=2, min_clearance_m=0.0)
    assert row == first[0] and audit["start_clearance_m"] is None


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


@pytest.mark.parametrize("shared", [False, True])
def test_each_building_has_both_recorded_jobs_with_fixed_action_protocol(tmp_path, shared):
    jobs = jobs_for(["A", "B"])
    assert jobs == [("frontier", "A"), ("falcon", "A"), ("falcon", "B"), ("frontier", "B")]
    args = SimpleNamespace(manifest=tmp_path / "episodes.json", output=tmp_path / "out", seed=0,
                           video_fps=6, detector_url="http://127.0.0.1:18095", detector_backend="yolo_world",
                           policy_config=None, allow_sim_version_mismatch=True, allow_shared_gpu=shared)
    command = command_for(args, "falcon", "A")
    assert "--record" in command and "--video-fps" in command
    assert "--record-first" not in command
    assert "--max-steps" not in command
    assert command[command.index("--explorer") + 1] == "falcon"
    assert ("--allow-shared-gpu" in command) is shared
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run_development import parser
    assert parser().parse_args(command).allow_shared_gpu is shared


def test_falcon_recording_shows_actual_phase_and_renders():
    old, episode = setup_policy("bed")
    policy = RPTSearchPolicy(old.detector, old.llm_client, replace(old.settings, local_exploration="falcon"))
    policy.reset(episode, old.target)
    obs = observation(episode, 0, depth=3)
    command = policy.plan(obs)
    snapshot = method_snapshot(policy)
    assert snapshot["state"] == "warmup"
    assert snapshot["explorer"] == "falcon" and snapshot["burst_actions_left"] is None
    frame = render_dashboard(policy, obs, [(0, 0)], {"action": "TURN_LEFT", "info": command.info}, episode.episode_id, snapshot)
    assert frame.shape == (900, 1600, 3)



def test_cross_floor_starts_can_be_restricted_to_one_goal_category(monkeypatch):
    """A couch-only cross-floor campaign (2026-10-05): the one category never upstairs in a house."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import multifloor_generation as mf
    semantic = np.zeros((7, 8, 8), np.uint8)
    semantic[0] = 1
    semantic[1, 1, 1] = 1                     # chair
    semantic[2, 6, 6] = 1                     # couch
    floors = {0: {"sem_map": semantic, "floor_height": 0.0, "origin": [0, 0]}}
    levels = [{"height_m": 0.0, "sample_count": 100, "estimated_area_m2": 40.0},
              {"height_m": 2.7, "sample_count": 80, "estimated_area_m2": 30.0}]
    rng = np.random.RandomState(0)
    samples = np.column_stack([rng.uniform(0, 8, 400), np.where(rng.rand(400) < 0.5, 0.0, 2.7), rng.uniform(0, 8, 400)])
    monkeypatch.setattr(mf, "sampled_levels", lambda pathfinder, seed, **kw: (levels, samples))
    monkeypatch.setattr(mf, "goal_region_points", lambda pathfinder, floor, category: [[1.0, 0.0, 1.0], [1.2, 0.0, 1.0]])

    class Field:
        def __init__(self, pathfinder, goals, semantic, origin, category, goal_height):
            self.category, self.goal_height = category, goal_height
        def distance(self, position, start=False):
            return 10.0
    monkeypatch.setattr(mf, "MultiFloorDistance", Field)
    monkeypatch.setattr(mf, "start_clearance", lambda pathfinder, position: 1.0)
    rows, regions, audit = mf.cross_floor_starts("Fake", floors, object(), seed=3, count=1, categories=[1])
    assert [row["object_category"] for row in rows] == ["couch"] and list(regions) == ["1"]
    assert audit["episodes"][0]["start_height_m"] == pytest.approx(2.7), "the start is on the other storey"
    rows, _, _ = mf.cross_floor_starts("Fake", floors, object(), seed=3, count=2)
    assert len(rows) == 2 and {row["object_category"] for row in rows} <= {"chair", "couch"}, "unrestricted: every annotated category"
    with pytest.raises(ValueError, match="None of toilet annotated"):
        mf.cross_floor_starts("Fake", floors, object(), seed=3, count=1, categories=[4])
    with pytest.raises(ValueError, match="requires --multistory"):
        generation.main(["--train-info", "x", "--archive", "y", "--scenes-dir", "z", "--output", "w", "--categories", "couch"])


def test_same_storey_starts_lie_on_the_goals_storey_with_a_different_category_per_episode(monkeypatch):
    """The same-floor protocol on the multistory harness (2026-10-05): a STOP at an annotated instance scores."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import multifloor_generation as mf
    semantic = np.zeros((7, 8, 8), np.uint8)
    semantic[0] = 1
    semantic[1, 1, 1] = semantic[2, 6, 6] = semantic[5, 3, 3] = 1          # chair, couch, toilet
    floors = {0: {"sem_map": semantic, "floor_height": 0.0, "origin": [0, 0]}}
    levels = [{"height_m": 0.0, "sample_count": 100, "estimated_area_m2": 40.0}]   # a single-storey house
    rng = np.random.RandomState(1)
    samples = np.column_stack([rng.uniform(0, 40, 600), np.zeros(600), rng.uniform(0, 40, 600)])
    monkeypatch.setattr(mf, "sampled_levels", lambda pathfinder, seed, **kw: (levels, samples))
    monkeypatch.setattr(mf, "goal_region_points", lambda pathfinder, floor, category: [[1.0, 0.0, 1.0]])

    class Field:
        def __init__(self, pathfinder, goals, semantic, origin, category, goal_height):
            self.category, self.goal_height = category, goal_height
        def distance(self, position, start=False):
            return 10.0
    monkeypatch.setattr(mf, "MultiFloorDistance", Field)
    monkeypatch.setattr(mf, "start_clearance", lambda pathfinder, position: 1.0)
    with pytest.raises(ValueError, match="Fewer than two"):
        mf.cross_floor_starts("Fake", floors, object(), seed=3, count=1)
    rows, regions, audit = mf.cross_floor_starts("Fake", floors, object(), seed=3, count=3, same_storey=True)
    assert len(rows) == 3 and len({row["object_category"] for row in rows}) == 3, "three episodes, three categories"
    assert all(row["start_position"][1] == 0.0 for row in rows), "every start on the goals' storey"
    assert all(entry["start_storey"] == "same" and not entry["connected_cross_floor"] for entry in audit["episodes"])
    starts = [np.asarray(row["start_position"])[[0, 2]] for row in rows]
    assert all(np.linalg.norm(a - b) >= 2.0 for i, a in enumerate(starts) for b in starts[i + 1:]), "starts 2 m apart"
    with pytest.raises(ValueError, match="requires --multistory"):
        generation.main(["--train-info", "x", "--archive", "y", "--scenes-dir", "z", "--output", "w", "--start-storey", "same"])


def test_a_building_with_fewer_categories_than_episodes_is_ineligible(monkeypatch):
    """Onaga (2026-10-05) has one annotated category: three couch episodes are not three different objects."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import multifloor_generation as mf
    semantic = np.zeros((7, 8, 8), np.uint8)
    semantic[0] = 1
    semantic[2, 6, 6] = 1                                                   # a couch, nothing else
    floors = {0: {"sem_map": semantic, "floor_height": 0.0, "origin": [0, 0]}}
    levels = [{"height_m": 0.0, "sample_count": 100, "estimated_area_m2": 40.0}]
    samples = np.column_stack([np.linspace(0, 40, 300), np.zeros(300), np.linspace(0, 40, 300)])
    monkeypatch.setattr(mf, "sampled_levels", lambda pathfinder, seed, **kw: (levels, samples))
    monkeypatch.setattr(mf, "goal_region_points", lambda pathfinder, floor, category: [[1.0, 0.0, 1.0]])

    class Field:
        def __init__(self, pathfinder, goals, semantic, origin, category, goal_height):
            self.category, self.goal_height = category, goal_height
        def distance(self, position, start=False):
            return 10.0
    monkeypatch.setattr(mf, "MultiFloorDistance", Field)
    monkeypatch.setattr(mf, "start_clearance", lambda pathfinder, position: 1.0)
    with pytest.raises(ValueError, match="Only 1 category annotated"):
        mf.cross_floor_starts("Onaga", floors, object(), seed=3, count=3, same_storey=True)
    rows, _, _ = mf.cross_floor_starts("Onaga", floors, object(), seed=3, count=3, same_storey=True, distinct_categories=False)
    assert [row["object_category"] for row in rows] == ["couch"] * 3, "the former rotation, one knob away"
