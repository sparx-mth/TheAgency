"""MP3D adapter: the published release's schema, protocol and evaluator.

No Matterport3D data, habitat-sim or GPU is needed. The release is built as a
tiny synthetic fixture with the published *schema*, and the simulator and the
navmesh are stubs, so these tests check the adapter's conversions and rules
rather than re-measuring a dataset.
"""
import gzip
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import (
    CATEGORIES, mp3d_label_mapper)
from sparx_agency.core.planning.objnav.labels.registry import default_label_mapper_registry
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.scoring import score_episode
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import env as env_module
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOL, PUBLISHED_VAL_EPISODES, PUBLISHED_VAL_SCENES)

SCENE = "2azQ1b91cZZ"


def enu(position, yaw=0.0, pitch=0.0):
    """The bridge's Habitat XYZ -> ENU mapping, written out for the fixture."""
    x, y, z = position
    return AgentPose(x=-z, y=-x, z=y, yaw=yaw, camera_pitch=pitch)


def release(root, scenes=(SCENE,), per_scene=2, categories=CATEGORIES,
            start=(1.0, 0.0, 2.0), rotation=(0.0, 0.0, 0.0, 1.0),
            view_point=(1.0, 0.0, 5.0), geodesic=3.0):
    """Write a synthetic release with the published layout and schema."""
    episodes_dir = root / "val"
    (episodes_dir / "content").mkdir(parents=True)
    scenes_dir = root / "scene_datasets"
    with gzip.open(episodes_dir / "val.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"episodes": [],
                   "category_to_task_category_id": {c: i for i, c in enumerate(categories)},
                   "category_to_mp3d_category_id": {c: i + 3 for i, c in enumerate(categories)}},
                  stream)
    for scene in scenes:
        mesh = scenes_dir / "mp3d" / scene
        mesh.mkdir(parents=True)
        (mesh / (scene + ".glb")).write_bytes(b"glb")
        (mesh / (scene + ".navmesh")).write_bytes(b"navmesh")
        rows = [{"episode_id": str(index), "scene_id": "mp3d/%s/%s.glb" % (scene, scene),
                 "start_position": list(start), "start_rotation": list(rotation),
                 "object_category": "chair", "goals": [],
                 "info": {"geodesic_distance": geodesic}}
                for index in range(per_scene)]
        goals = {"%s.glb_chair" % scene: [
            {"position": [1.0, 0.0, 6.0], "object_id": "7", "object_category": "chair",
             "view_points": [{"agent_state": {"position": list(view_point),
                                              "rotation": [0, 0, 0, 1]}, "iou": 0.5}]}]}
        with gzip.open(episodes_dir / "content" / (scene + ".json.gz"), "wt",
                       encoding="utf-8") as stream:
            json.dump({"episodes": rows, "goals_by_category": goals}, stream)
    return episodes_dir, scenes_dir


class StubDistance:
    """Stands in for the geodesic field; the real one needs habitat-sim."""

    def __init__(self, pathfinder, view_points):
        self.view_point_count = len(view_points)
        self.point = np.asarray(view_points, dtype=np.float64)[0]

    def distance(self, position):
        return float(np.linalg.norm(np.asarray(position, dtype=np.float64) - self.point))


class StubSimulator:
    """Replays a scripted pose sequence; never renders and never moves the start."""

    pathfinder = "published-navmesh"

    def __init__(self, positions):
        self.positions = list(positions)
        self.index = 0
        self.closed = False
        self.loaded = []

    def _frame(self):
        shape = (PROTOCOL.height, PROTOCOL.width)
        rgb = np.zeros(shape + (3,), np.uint8)
        depth = np.full(shape, 2.0, np.float32)
        return rgb, depth, enu(self.positions[self.index])

    def reset(self, mesh, navmesh, position, rotation_wxyz, seed):
        self.loaded.append((str(mesh), str(navmesh), tuple(rotation_wxyz)))
        self.index = 0
        return self._frame()

    def step(self, action):
        if action != DiscreteAction.STOP:
            self.index = min(self.index + 1, len(self.positions) - 1)
        return self._frame()

    def navmesh_provenance(self):
        return {"source": "agent", "scene": self.loaded[-1][0] if self.loaded else None,
                "agent_radius_m": 0.18, "agent_height_m": 0.88,
                "cell_size_m": 0.05, "navigable_area_m2": 44.145}

    def close(self):
        self.closed = True


def build(tmp_path, positions, monkeypatch, **kwargs):
    episodes_dir, scenes_dir = release(tmp_path, **kwargs)
    dataset = MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)
    monkeypatch.setattr(env_module, "ViewPointDistance", StubDistance)
    simulator = StubSimulator(positions)
    return dataset, MP3DEnv(dataset, simulator=simulator), simulator


# --- the published vocabulary -------------------------------------------------

def test_the_table_is_the_published_21_categories_and_is_unambiguous():
    mapper = mp3d_label_mapper()
    assert mapper.categories() == CATEGORIES and len(CATEGORIES) == 21
    assert mapper.target_labels("tv_monitor").query == "tv"
    assert mapper.target_labels("chest_of_drawers").query == "chest of drawers"
    seen = {}
    for category in CATEGORIES:
        for label in mapper.target_labels(category).accept_labels:
            assert seen.setdefault(label, category) == category
    assert default_label_mapper_registry().create("mp3d").categories() == CATEGORIES


def test_the_vocabulary_carries_the_door_prompts_the_method_needs():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS

    assert DOOR_LABELS.issubset(set(mp3d_label_mapper().vocabulary()))


# --- the published protocol ---------------------------------------------------

def test_the_protocol_is_the_published_configuration():
    """Transcribed from habitat-lab objectnav_mp3d.yaml / challenge 2021."""
    assert (PROTOCOL.max_steps, PROTOCOL.width, PROTOCOL.height) == (500, 640, 480)
    assert (PROTOCOL.hfov_deg, PROTOCOL.camera_height_m) == (79.0, 0.88)
    assert (PROTOCOL.agent_height_m, PROTOCOL.agent_radius_m) == (0.88, 0.18)
    assert (PROTOCOL.min_depth_m, PROTOCOL.max_depth_m) == (0.5, 5.0)
    assert (PROTOCOL.forward_step_m, PROTOCOL.turn_angle_deg) == (0.25, 30.0)
    assert (PROTOCOL.tilt_angle_deg, PROTOCOL.success_distance_m) == (30.0, 0.1)
    assert PROTOCOL.allow_sliding is False
    assert len(PROTOCOL.actions().actions) == 6


def test_the_harness_options_are_the_habitat_lab_rules_not_a_relaxation():
    settings = PROTOCOL.evaluation()
    assert settings.require_stop_for_success is True
    assert settings.path_length_dimension == "3d"
    assert settings.path_length_epsilon_m == 0.0


# --- the published release ----------------------------------------------------

def test_a_release_with_another_vocabulary_is_refused(tmp_path):
    episodes_dir, scenes_dir = release(tmp_path, categories=CATEGORIES[:6])
    with pytest.raises(ValueError, match="21"):
        MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)


def test_a_partial_release_cannot_pass_as_the_full_split(tmp_path):
    episodes_dir, scenes_dir = release(tmp_path)
    with pytest.raises(ValueError, match="%d episodes" % PUBLISHED_VAL_EPISODES):
        MP3DDataset(episodes_dir, scenes_dir, full=True)
    assert PUBLISHED_VAL_SCENES == 11


def test_episodes_are_scene_qualified_and_keep_the_publishers_own_id(tmp_path):
    episodes_dir, scenes_dir = release(tmp_path, per_scene=3)
    dataset = MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)
    assert list(dataset.episodes) == ["%s/%06d" % (SCENE, i) for i in range(3)]
    row = dataset.episodes["%s/000001" % SCENE]
    assert row.published_episode_id == "1" and row.goals_key == "%s.glb_chair" % SCENE
    assert dataset.view_points(row).shape == (1, 3)
    assert dataset.manifest()["episode_count"] == 3


def test_the_published_xyzw_rotation_is_converted_for_habitat_sim(tmp_path):
    """habitat-lab serializes (x, y, z, w); quaternion.from_float_array wants (w, x, y, z)."""
    episodes_dir, scenes_dir = release(tmp_path, rotation=(0.0, 0.7071068, 0.0, 0.7071068))
    dataset = MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)
    row = dataset.episodes["%s/000000" % SCENE]
    assert row.start_rotation_xyzw == pytest.approx((0.0, 0.7071068, 0.0, 0.7071068))
    assert row.start_rotation_wxyz() == pytest.approx((0.7071068, 0.0, 0.7071068, 0.0))


def test_a_missing_navmesh_is_not_regenerated(tmp_path):
    episodes_dir, scenes_dir = release(tmp_path)
    (scenes_dir / "mp3d" / SCENE / (SCENE + ".navmesh")).unlink()
    with pytest.raises(FileNotFoundError, match="navmesh"):
        MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)


# --- the evaluator ------------------------------------------------------------

def test_stopping_on_a_view_point_succeeds_and_scores_like_habitat_lab(tmp_path, monkeypatch):
    positions = [(1.0, 0.0, 2.0), (1.0, 0.0, 3.0), (1.0, 0.0, 4.0), (1.0, 0.0, 5.0)]
    dataset, env, simulator = build(tmp_path, positions, monkeypatch)
    episode, observation = env.reset("%s/000000" % SCENE)
    assert episode.max_steps == 500 and episode.target_category == "chair"
    assert habitat_position(observation.pose) == pytest.approx((1.0, 0.0, 2.0))
    for _ in range(3):
        env.step(DiscreteAction.MOVE_FORWARD)
    env.step(DiscreteAction.STOP)
    measurement = env.measure()
    assert measurement.success and measurement.stop_called
    assert measurement.shortest_path_m == pytest.approx(3.0)
    assert measurement.path_length_m == pytest.approx(3.0)
    assert measurement.final_distance_to_goal_m == pytest.approx(0.0)
    score = score_episode(measurement, observed_path_length_m=measurement.path_length_m,
                          require_stop_for_success=True)
    assert score.spl == pytest.approx(1.0) and score.success


def test_arriving_without_stop_is_not_a_success(tmp_path, monkeypatch):
    positions = [(1.0, 0.0, 2.0), (1.0, 0.0, 5.0)]
    dataset, env, simulator = build(tmp_path, positions, monkeypatch)
    env.reset("%s/000000" % SCENE)
    env.step(DiscreteAction.MOVE_FORWARD)
    measurement = env.measure() if env.episode_over else None
    assert measurement is None  # the budget has not run out and STOP was never called
    env.step(DiscreteAction.STOP)
    assert env.measure().success


def test_stopping_outside_the_success_distance_fails_but_keeps_soft_progress(tmp_path, monkeypatch):
    positions = [(1.0, 0.0, 2.0), (1.0, 0.0, 2.5)]
    dataset, env, simulator = build(tmp_path, positions, monkeypatch)
    env.reset("%s/000000" % SCENE)
    env.step(DiscreteAction.MOVE_FORWARD)
    env.step(DiscreteAction.STOP)
    measurement = env.measure()
    assert not measurement.success and measurement.stop_called
    assert measurement.final_distance_to_goal_m == pytest.approx(2.5)
    score = score_episode(measurement, observed_path_length_m=measurement.path_length_m,
                          require_stop_for_success=True)
    assert score.spl == 0.0 and score.soft_spl == pytest.approx((1 - 2.5 / 3.0) * 3.0 / 3.0)


def test_the_path_is_accumulated_in_three_dimensions(tmp_path, monkeypatch):
    """Habitat's SPL sums 3-D chords; a flattened axis would shorten every stair."""
    positions = [(0.0, 0.0, 0.0), (0.0, 3.0, 4.0)]
    dataset, env, simulator = build(tmp_path, positions, monkeypatch,
                                    start=(0.0, 0.0, 0.0), view_point=(0.0, 3.0, 4.0))
    env.reset("%s/000000" % SCENE)
    env.step(DiscreteAction.MOVE_FORWARD)
    env.step(DiscreteAction.STOP)
    assert env.measure().path_length_m == pytest.approx(5.0)


def test_a_simulator_that_moves_the_published_start_is_refused(tmp_path, monkeypatch):
    dataset, env, simulator = build(tmp_path, [(9.0, 0.0, 9.0)], monkeypatch)
    with pytest.raises(EnvContractError, match="published episode start"):
        env.reset("%s/000000" % SCENE)


def test_an_unreachable_start_is_refused_rather_than_scored(tmp_path, monkeypatch):
    dataset, env, simulator = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch)

    class Unreachable(StubDistance):
        def distance(self, position):
            return math.inf

    monkeypatch.setattr(env_module, "ViewPointDistance", Unreachable)
    with pytest.raises(EnvContractError, match="reachable"):
        env.reset("%s/000000" % SCENE)


def test_the_step_budget_ends_the_episode_without_stop(tmp_path, monkeypatch):
    dataset, env, simulator = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch)
    env.reset("%s/000000" % SCENE)
    for _ in range(PROTOCOL.max_steps):
        env.step(DiscreteAction.TURN_LEFT)
    assert env.episode_over
    measurement = env.measure()
    assert not measurement.success and measurement.termination == "step_limit"
    assert measurement.steps == PROTOCOL.max_steps


def test_goal_telemetry_reaches_the_recorder_and_not_the_observation(tmp_path, monkeypatch):
    dataset, env, simulator = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch)
    _, observation = env.reset("%s/000000" % SCENE)
    for name in ("distance_to_goal_m", "shortest_path_m", "view_points"):
        assert not hasattr(observation, name)
    diagnostics = env.evaluation_diagnostics()
    assert diagnostics["shortest_path_m"] == pytest.approx(3.0)
    assert diagnostics["view_points"] == 1


def test_closing_the_environment_releases_the_simulator(tmp_path, monkeypatch):
    dataset, env, simulator = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch)
    env.close()
    assert simulator.closed


# --- the render-free start preflight -----------------------------------------

def test_the_start_preflight_measures_on_the_navmesh_the_run_scores_on(
        tmp_path, monkeypatch):
    """A standalone PathFinder would read the published file and answer for a
    navmesh the run never uses, which is the whole bug this guards."""
    dataset, env, simulator = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch,
                                    per_scene=2, geodesic=99.0)
    report = env.validate_starts()
    assert report["checked"] == 2 and not report["unreachable_starts"]
    assert report["navmesh"][SCENE]["source"] == "agent"
    assert report["navmesh"][SCENE]["agent_radius_m"] == 0.18
    # One scene load for two episodes, not one load each.
    assert len(simulator.loaded) == 1
    # The stub distance says 3 m; the fixture's release claims 99 m.
    assert set(report["published_geodesic_gaps"]) == set(dataset.episodes)
    assert report["published_geodesic_gaps"]["%s/000000" % SCENE] == pytest.approx(-96.0)


def test_the_start_preflight_is_quiet_when_the_publisher_agrees(tmp_path, monkeypatch):
    dataset, env, simulator = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch,
                                    per_scene=2, geodesic=3.0)
    report = env.validate_starts()
    assert report["published_geodesic_gaps"] == {} and not report["unreachable_starts"]


# --- the report ---------------------------------------------------------------

def test_the_published_rows_are_fractions_with_a_source_each():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import baselines

    rows = baselines()
    assert len(rows) >= 12
    for row in rows:
        assert 0.0 <= row.spl <= row.success_rate <= 1.0
        assert row.source and row.benchmark == "mp3d" and row.split == "val"
    best = max(rows, key=lambda r: r.success_rate)
    assert best.method.startswith("SG-Nav") and best.success_rate == pytest.approx(0.402)


# --- which navmesh scores the run --------------------------------------------

def test_mp3d_scores_on_the_navmesh_habitat_lab_derives_for_this_agent():
    """habitat-lab's 0.18 m / 0.88 m agent makes habitat-sim recompute the navmesh.

    Measured on Collierville with habitat-sim 0.2.4: the recomputed navmesh has
    44.145 m2 navigable against the shipped file's 58.006 m2, and 14 of 39
    sampled point pairs are reachable only on the shipped one. Reloading the
    shipped navmesh would score a more permissive world than every published
    MP3D number was measured in.
    """
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import (
        NAVMESH_AGENT_RECOMPUTED)

    assert PROTOCOL.navmesh_source == NAVMESH_AGENT_RECOMPUTED


def test_the_environment_hands_that_choice_to_the_bridge(tmp_path):
    episodes_dir, scenes_dir = release(tmp_path)
    dataset = MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)
    env = MP3DEnv(dataset)  # builds the real bridge; habitat-sim is not imported
    assert env._simulator.navmesh == PROTOCOL.navmesh_source
    assert (env._simulator.radius_m, env._simulator.height_m) == (0.18, 0.88)


def test_the_bridge_refuses_an_unnamed_navmesh_source():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import (
        HabitatRGBDSimulator)

    with pytest.raises(ValueError, match="navmesh must be one of"):
        HabitatRGBDSimulator(PROTOCOL.camera(), PROTOCOL.actions(), 0.18, False,
                             height_m=0.88, navmesh="whatever-is-loaded")


# --- the geodesic field itself ------------------------------------------------

class FakeMultiGoalShortestPath:
    instances = 0

    def __init__(self):
        FakeMultiGoalShortestPath.instances += 1
        self.requested_start = None
        self.requested_ends = None
        self.geodesic_distance = None


class FakePathFinder:
    """Answers with the straight-line distance to the nearest end."""

    def __init__(self):
        self.queries = []

    def find_path(self, path):
        start = np.asarray(path.requested_start, dtype=np.float64)
        self.queries.append(tuple(start))
        path.geodesic_distance = min(
            float(np.linalg.norm(start - np.asarray(end, dtype=np.float64)))
            for end in path.requested_ends)


@pytest.fixture
def fake_habitat_sim(monkeypatch):
    FakeMultiGoalShortestPath.instances = 0
    monkeypatch.setitem(sys.modules, "habitat_sim", SimpleNamespace(
        MultiGoalShortestPath=FakeMultiGoalShortestPath))
    return FakeMultiGoalShortestPath


def test_the_view_point_field_reuses_one_multi_goal_path_for_the_episode(fake_habitat_sim):
    """habitat-lab caches the path object on the episode; rebuilding it per step
    would re-run the multi-goal precomputation for the same answer."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.distance import ViewPointDistance

    finder = FakePathFinder()
    field = ViewPointDistance(finder, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)])
    assert field.view_point_count == 2
    assert fake_habitat_sim.instances == 1
    assert field.distance((3.0, 0.0, 0.0)) == pytest.approx(3.0)
    assert field.distance((9.0, 0.0, 0.0)) == pytest.approx(1.0)
    assert fake_habitat_sim.instances == 1
    assert finder.queries == [(3.0, 0.0, 0.0), (9.0, 0.0, 0.0)]


def test_the_view_point_field_passes_an_unreachable_answer_through(fake_habitat_sim):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.distance import ViewPointDistance

    class Unreachable(FakePathFinder):
        def find_path(self, path):
            path.geodesic_distance = math.inf

    field = ViewPointDistance(Unreachable(), [(0.0, 0.0, 0.0)])
    assert math.isinf(field.distance((1.0, 2.0, 3.0)))


def test_the_view_point_field_refuses_malformed_input(fake_habitat_sim):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.distance import ViewPointDistance

    with pytest.raises(ValueError):
        ViewPointDistance(FakePathFinder(), np.zeros((0, 3)))
    with pytest.raises(ValueError):
        ViewPointDistance(FakePathFinder(), [(0.0, 0.0)])
    field = ViewPointDistance(FakePathFinder(), [(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError):
        field.distance((0.0, 0.0))
    with pytest.raises(ValueError):
        field.distance((0.0, math.nan, 0.0))


# --- regressions for what the first review found ------------------------------

def test_the_path_moves_on_all_three_axes(tmp_path, monkeypatch):
    """Two axes cannot tell a 3-D accumulation from a flattened one."""
    positions = [(0.0, 0.0, 0.0), (2.0, 3.0, 6.0)]
    dataset, env, simulator = build(tmp_path, positions, monkeypatch,
                                    start=(0.0, 0.0, 0.0), view_point=(2.0, 3.0, 6.0))
    env.reset("%s/000000" % SCENE)
    env.step(DiscreteAction.MOVE_FORWARD)
    env.step(DiscreteAction.STOP)
    assert env.measure().path_length_m == pytest.approx(7.0)


def test_the_publishers_episode_id_is_kept_and_is_not_the_row_index(tmp_path):
    episodes_dir, scenes_dir = release(tmp_path, per_scene=3)
    path = episodes_dir / "content" / (SCENE + ".json.gz")
    import gzip as _gzip
    with _gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    for index, row in enumerate(payload["episodes"]):
        row["episode_id"] = "published-%d" % (900 + index)
    with _gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    dataset = MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)
    row = dataset.episodes["%s/000002" % SCENE]
    assert row.published_episode_id == "published-902"
    assert list(dataset.episodes) == ["%s/%06d" % (SCENE, i) for i in range(3)]


def test_the_start_preflight_names_an_unreachable_start(tmp_path, monkeypatch):
    dataset, env, _ = build(tmp_path, [(1.0, 0.0, 2.0)], monkeypatch, per_scene=1)

    class Unreachable(StubDistance):
        def distance(self, position):
            return math.inf

    monkeypatch.setattr(env_module, "ViewPointDistance", Unreachable)
    report = env.validate_starts()
    assert report["unreachable_starts"] == ["%s/000000" % SCENE]
    assert report["published_geodesic_gaps"] == {}


def test_a_desk_counts_as_the_table_goal_rather_than_as_room_context():
    """mpcat40 has no 'desk'; MP3D annotates desks under the 'table' goal."""
    mapper = mp3d_label_mapper()
    assert "desk" in mapper.target_labels("table").detector_prompts
    assert "desk" in mapper.vocabulary()
    for category in CATEGORIES:
        if category != "table":
            assert "desk" not in mapper.target_labels(category).accept_labels
