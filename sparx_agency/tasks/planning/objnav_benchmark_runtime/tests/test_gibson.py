"""CPU regressions: real FMM/scoring/runner/logger, synthetic licensed-free data."""
from __future__ import annotations

import bz2
from dataclasses import asdict
import gzip
import json
import math
import pickle

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction as A
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset, _ArrayUnpickler
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distance import GibsonDistanceField
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.report import write_report
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import select_episodes
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import habitat_pose, metric_depth


@pytest.fixture
def semantic():
    grid = np.zeros((7, 120, 120), dtype=np.uint8)
    grid[0] = 1
    grid[1:, 60, 20] = 1
    return grid


def write_dataset(root, semantic, scenes=SCENES[:1], per_scene=1, edit=None):
    episodes = root / "val"
    assets = root / "scenes"
    (episodes / "content").mkdir(parents=True)
    assets.mkdir()
    maps = {}
    for scene in scenes:
        maps[scene] = {0: {"sem_map": semantic, "origin": np.array([-200, -200])}}
        for suffix in (".glb", ".navmesh"):
            (assets / (scene + suffix)).write_text("synthetic fixture; not renderable")
        rows = [{"start_position": [1.0, 0.0, 1.0],
                 "start_rotation": [1.0, 0.0, 0.0, 0.0],
                 "object_category": "chair", "object_id": 0, "floor_id": 0}
                for _ in range(per_scene)]
        if edit:
            edit(rows)
        with gzip.open(episodes / "content" / (scene + "_episodes.json.gz"), "wt") as stream:
            json.dump({"episodes": rows}, stream)
    with bz2.open(episodes / "val_info.pbz2", "wb") as stream:
        pickle.dump(maps, stream, protocol=4)
    return episodes, assets


def test_full_split_requires_all_five_scenes_and_exact_counts(tmp_path, semantic):
    paths = write_dataset(tmp_path, semantic)
    with pytest.raises(ValueError, match="scenes"):
        GibsonDataset(*paths)
    dataset = GibsonDataset(*paths, full=False)
    assert tuple(dataset.episodes) == ("Collierville/000000",)
    assert dataset.manifest()["episode_count"] == 1
    (paths[1] / "Collierville.navmesh").unlink()
    with pytest.raises(FileNotFoundError, match="asset"):
        GibsonDataset(*paths, full=False)


def test_published_count_and_manifest(tmp_path, semantic):
    paths = write_dataset(tmp_path, semantic, SCENES, 200)
    dataset = GibsonDataset(*paths)
    assert len(dataset.episodes) == 1000
    assert len(dataset.manifest()["files"]) == 16
    assert set(dataset.scene_counts.values()) == {200}


@pytest.mark.parametrize("edit", [
    lambda rows: rows[0].update(object_id=2),
    lambda rows: rows[0].update(start_rotation=[0, 0, 0, 0]),
    lambda rows: rows[0].update(object_categories=["chair"]),
    lambda rows: rows[0].update(floor_id=-1),
])
def test_bad_or_wrong_release_episodes_refused(tmp_path, semantic, edit):
    paths = write_dataset(tmp_path, semantic, edit=edit)
    with pytest.raises(ValueError):
        GibsonDataset(*paths, full=False)


def test_pickle_cannot_resolve_executable_globals():
    import io
    with pytest.raises(ValueError, match="Unapproved global"):
        _ArrayUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_fmm_matches_reference_computation(semantic):
    import skfmm
    from skimage.morphology import binary_dilation, disk
    field = GibsonDistanceField(semantic, (-200, -200), 0)
    free = binary_dilation(semantic[0], disk(2))
    goals = binary_dilation(semantic[1], disk(20))
    phi = np.ma.masked_values(free * 1, 0)
    phi[goals == 1] = 0
    reference = skfmm.distance(phi, dx=1)
    reference = np.ma.filled(reference, np.max(reference) + 1)
    np.testing.assert_allclose(field.cells, reference, atol=0, rtol=0)
    assert field.distance((1, 0, 0)) == 0  # exactly on success boundary
    assert field.map_cell((1, 0, 1)) == (60, 60)
    with pytest.raises(ValueError, match="outside"):
        field.distance((-2.001, 0, 1))  # never numpy negative-index wrapping


def test_depth_encoding_and_coordinate_axes():
    camera = PROTOCOL.camera()
    raw = np.array([[0, 0.5, 2.0, 5.0, np.inf, np.nan]], dtype=np.float32)
    decoded = metric_depth(raw, camera)
    assert np.isnan(decoded[0, :2]).all()
    assert decoded[0, 2] == 2.0
    assert np.isposinf(decoded[0, 3:5]).all()
    assert np.isnan(decoded[0, 5])
    np.testing.assert_allclose(metric_depth(np.array([[0, 1 / 3, 1.0]]), camera, True),
                               [[np.nan, 2.0, np.inf]], equal_nan=True)
    p = habitat_pose((3, 2, 1), np.eye(3), np.eye(3))
    assert (p.x, p.y, p.z, p.yaw) == (-1, -3, 2, 0)
    theta = math.pi / 6
    rotation = np.array([[math.cos(theta), 0, math.sin(theta)],
                         [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    assert habitat_pose((0, 0, 0), rotation, rotation).yaw == pytest.approx(theta)
    tilt = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)],
                     [0, math.sin(theta), math.cos(theta)]])
    assert habitat_pose((0, 0, 0), np.eye(3), tilt).camera_pitch == pytest.approx(-theta)


class LineSimulator:
    def reset(self, scene, mesh, position, rotation, seed):
        self.pose = AgentPose(-position[2], -position[0], position[1], 0.0)
        self.seed = seed
        return self.frame()

    def frame(self):
        k = PROTOCOL.camera().intrinsics
        return np.zeros((k.height, k.width, 3), np.uint8), np.ones((k.height, k.width), np.float32), self.pose

    def step(self, action):
        self.pose = apply_action(self.pose, action, PROTOCOL.actions())
        return self.frame()

    def close(self):
        pass


class LinePolicy:
    name = "synthetic-line-test"

    def reset(self, episode, target):
        assert episode.metadata == {}  # neither goal coordinates nor floor/map ids
        self.target = target

    def plan(self, observation):
        if observation.pose.x >= 0:
            return NavigationCommand.stop_here()
        # Aim beyond the STOP threshold, not inside the arrival dead band.
        return NavigationCommand.follow([(0.5, -1)])


def test_real_runner_scores_logs_and_resumes_gibson(tmp_path, semantic):
    paths = write_dataset(tmp_path, semantic)
    env = GibsonEnv(GibsonDataset(*paths, full=False), simulator=LineSimulator())
    env.validate_starts()
    agent = HeadlessObjNavAgent(LinePolicy(), gibson_label_mapper(), name="synthetic-line-test")
    output = tmp_path / "results"
    config = {"protocol": asdict(PROTOCOL), "selected_episode_ids": list(env.episode_ids()),
              "full_split": False, "method": {"method": agent.name},
              "reference_sim_version_match": False}
    kwargs = dict(require_stop_for_success=False, path_length_dimension="planar",
                  path_length_epsilon_m=1e-5, kinematics=PROTOCOL.kinematics())
    with MetricsLogger(output, config) as logger:
        summary = run_benchmark(env, agent, logger=logger, **kwargs)
    assert summary.overall.success_rate == 1
    assert summary.overall.spl == 1
    assert summary.overall.distance_to_goal_m == 0
    audit = write_report(output)
    assert audit["full_split_completed"] is False
    assert "NOT a full" in (output / "comparison.md").read_text()
    with MetricsLogger(output, config, resume=True) as logger:
        resumed = run_benchmark(env, agent, logger=logger, **kwargs)
    assert resumed == summary
    rows = (output / "episodes.jsonl").read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["steps"] == 5 and row["path_length_m"] == pytest.approx(1 + 1e-5)
    env.close()


def test_step_limit_success_is_not_stop_gated(tmp_path, semantic):
    paths = write_dataset(tmp_path, semantic)
    env = GibsonEnv(GibsonDataset(*paths, full=False), simulator=LineSimulator())
    env.reset(env.episode_ids()[0])
    for _ in range(4):
        env.step(A.MOVE_FORWARD)
    # Stay still to the exact budget; exercise the adapter, not a fake score.
    for _ in range(PROTOCOL.max_steps - 4):
        env.step(A.TURN_LEFT)
    measured = env.measure()
    assert measured.success and not measured.stop_called
    assert measured.termination == "step_limit" and measured.steps == 500
    with pytest.raises(ValueError):
        env.step(A.STOP)
    env.close()


def test_shards_partition_and_invalid_selection_refused():
    ids = tuple(str(i) for i in range(1000))
    shards = [select_episodes(ids, shards=3, shard_index=i) for i in range(3)]
    assert len({x for shard in shards for x in shard}) == 1000
    assert sum(map(len, shards)) == 1000
    for kwargs in ({"limit": 0}, {"shards": 0}, {"shards": 2, "shard_index": 2}):
        with pytest.raises(ValueError):
            select_episodes(ids, **kwargs)

