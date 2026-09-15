"""The RoboTHOR adapter: its episode files, its challenge rules, and the harness accepting it.

Everything here runs without ai2thor, without a GPU and without the published
dataset: the episode files are written on the fly in the challenge's own
on-disk shape, and the simulator is the fake controller that moves the way
AI2-THOR moves. What is being tested is the adapter's *rules* -- which episode
is which, what counts as success, what ``l`` and ``p`` are -- and the fact that
the shared harness's contract, motion and cross-checks all accept the result.
"""
from __future__ import annotations

import gzip
import json
import math

import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.dataset import (
    RobothorDataset, path_distance,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.env import (
    SUCCESS_ANY_INSTANCE, SUCCESS_FIRST_INSTANCE, RobothorEnv, official_spl,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.fake_thor import (
    FakeThorController,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.thor.simulator import (
    AI2ThorRGBDSimulator,
)

SCENE = "FloorPlan_Val1_1"
ORIGIN_Y = 0.901


def episode_row(episode_id="ep0", scene=SCENE, category="Mug",
                start=(0.0, 0.0), corners=((0.0, 0.0), (1.0, 0.0), (1.0, 2.0)),
                orientation=270.0, horizon=30.0):
    """One episode in the challenge's own on-disk shape."""
    path = [{"x": float(x), "y": 0.0103, "z": float(z)} for x, z in corners]
    return {
        "id": episode_id, "scene": scene, "object_type": category,
        "initial_position": {"x": float(start[0]), "y": ORIGIN_Y,
                             "z": float(start[1])},
        "initial_orientation": orientation, "initial_horizon": horizon,
        "shortest_path": path, "shortest_path_length": path_distance(path),
    }


def write_split(root, rows_by_scene):
    """``<root>/val/episodes/<Scene>.json.gz``, gzipped JSON lists."""
    episodes = root / "val" / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    for scene, rows in rows_by_scene.items():
        (episodes / (scene + ".json.gz")).write_bytes(
            gzip.compress(json.dumps(rows).encode("utf-8")))
    return root


def dataset(tmp_path, rows=None, **kwargs):
    rows = rows if rows is not None else [episode_row()]
    write_split(tmp_path, {SCENE: rows})
    options = dict(scene=SCENE, require_full_split=False)
    options.update(kwargs)
    return RobothorDataset(tmp_path, **options)


def environment(data, *, controller=None, goal_visible=False,
                goal_at=(1.0, 2.0), instances=1, **kwargs):
    controller = controller or FakeThorController(
        64, 48, camera_offset_m=PROTOCOL.origin_height_m - PROTOCOL.camera_height_m,
        origin_height_m=PROTOCOL.origin_height_m)
    row = next(iter(data.episodes.values()))
    for index in range(instances):
        controller.place_object(
            row.category,
            {"x": goal_at[0] + index, "y": 0.9, "z": goal_at[1]},
            visible=goal_visible and index == instances - 1)
    camera = PROTOCOL.camera()
    simulator = AI2ThorRGBDSimulator(
        _rescaled(camera, 64, 48), PROTOCOL.actions(), PROTOCOL.agent_radius_m,
        height_m=PROTOCOL.body_height_m,
        origin_height_m=PROTOCOL.origin_height_m,
        initialize=PROTOCOL.initialize(), commit_id=PROTOCOL.thor_build_id,
        width=64, height=48, controller_factory=lambda _: controller)
    env = RobothorEnv(data, simulator, **kwargs)
    env._camera = simulator.camera
    return env, controller


def _rescaled(camera, width, height):
    """The protocol camera at a test-sized frame; the FOV is what matters."""
    from sparx_agency.core.common.types import Intrinsics
    from sparx_agency.core.planning.objnav.types.camera import CameraSpec
    k = camera.intrinsics
    scale = width / float(k.width)
    return CameraSpec(
        Intrinsics(width, height, k.fx * scale, k.fy * scale,
                   (k.cx + 0.5) * scale - 0.5, (k.cy + 0.5) * scale - 0.5),
        camera.height_m, camera.min_depth_m, camera.max_depth_m)


class Scripted(ObjNavAgent):
    """Plays a fixed action list, then STOPs. No perception, no privilege."""

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self._index = 0

    def reset(self, episode):
        self._index = 0

    def act(self, observation):
        if self._index < len(self.script):
            action = self.script[self._index]
            self._index += 1
        else:
            action = DiscreteAction.STOP
        return AgentDecision(action)


# ------------------------------------------------------------------ dataset


def test_the_published_shape_loads_with_its_privileged_columns_intact(tmp_path):
    data = dataset(tmp_path)
    row = data.episodes["ep0"]
    assert row.scene == SCENE and row.category == "Mug"
    assert row.start_horizon_deg == 30.0
    assert row.shortest_path_length_m == pytest.approx(3.0)
    assert data.categories() == ("Mug",)
    manifest = data.manifest()
    assert manifest["n_episodes"] == 1 and manifest["split"] == "val"
    assert list(manifest["episode_files_sha256"]) == [SCENE + ".json.gz"]


def test_a_regenerated_shortest_path_length_is_refused(tmp_path):
    row = episode_row()
    row["shortest_path_length"] = 99.0
    write_split(tmp_path, {SCENE: [row]})
    with pytest.raises(ValueError, match="SPL numerator"):
        RobothorDataset(tmp_path, scene=SCENE, require_full_split=False)


def test_a_category_outside_the_twelve_targets_is_refused(tmp_path):
    row = episode_row(category="Toaster")
    write_split(tmp_path, {SCENE: [row]})
    with pytest.raises(ValueError, match="12 RoboTHOR challenge targets"):
        RobothorDataset(tmp_path, scene=SCENE, require_full_split=False)


def test_a_repeated_episode_id_is_refused(tmp_path):
    write_split(tmp_path, {SCENE: [episode_row("ep0"), episode_row("ep0")]})
    with pytest.raises(ValueError, match="appears twice"):
        RobothorDataset(tmp_path, scene=SCENE, require_full_split=False)


def test_an_episode_filed_under_another_scene_is_refused(tmp_path):
    write_split(tmp_path, {SCENE: [episode_row(scene="FloorPlan_Val2_2")]})
    with pytest.raises(ValueError, match="only its own scene"):
        RobothorDataset(tmp_path, scene=SCENE, require_full_split=False)


def test_a_partial_directory_cannot_pass_as_the_published_split(tmp_path):
    write_split(tmp_path, {SCENE: [episode_row()]})
    with pytest.raises(ValueError, match="not the published RoboTHOR validation split"):
        RobothorDataset(tmp_path)


def test_a_test_split_row_without_a_length_is_refused(tmp_path):
    row = episode_row()
    del row["shortest_path_length"]
    write_split(tmp_path, {SCENE: [row]})
    with pytest.raises(ValueError, match="cannot be scored here"):
        RobothorDataset(tmp_path, scene=SCENE, require_full_split=False)


def test_zero_length_episodes_are_counted_for_the_spl_disclosure(tmp_path):
    rows = [episode_row("ep0"),
            episode_row("ep1", corners=((0.0, 0.0),))]
    data = dataset(tmp_path, rows)
    assert data.zero_length_episode_ids == ("ep1",)
    assert data.manifest()["zero_shortest_path_episodes"] == 1


# --------------------------------------------------------------- the episode


def test_the_episode_carries_no_privileged_value(tmp_path):
    env, _ = environment(dataset(tmp_path))
    episode, observation = env.reset("ep0")
    assert episode.benchmark == "robothor" and episode.split == "val"
    assert episode.target_category == "Mug"
    assert episode.max_steps == 500
    assert episode.metadata == {}
    assert observation.step == 0
    assert observation.pose.camera_pitch == pytest.approx(math.radians(30.0))
    env.close()


def test_a_scene_without_the_goal_is_refused_with_a_reason(tmp_path):
    env, _ = environment(dataset(tmp_path), instances=0)
    with pytest.raises(EnvContractError, match="holds no Mug"):
        env.reset("ep0")
    env.close()


def test_a_moved_start_is_refused(tmp_path):
    class Drifting(FakeThorController):
        def step(self, action=None, **kwargs):
            event = super().step(action=action, **kwargs)
            if isinstance(action, dict):
                self.position = dict(self.position, x=self.position["x"] + 0.5)
                return super()._event()
            return event

    data = dataset(tmp_path)
    env, _ = environment(data, controller=Drifting(
        64, 48, camera_offset_m=PROTOCOL.origin_height_m - PROTOCOL.camera_height_m,
        origin_height_m=PROTOCOL.origin_height_m))
    with pytest.raises(EnvContractError, match="published start"):
        env.reset("ep0")
    env.close()


def test_the_budget_ends_the_episode_and_a_failed_action_still_spends_a_step(tmp_path):
    controller = FakeThorController(
        64, 48, camera_offset_m=PROTOCOL.origin_height_m - PROTOCOL.camera_height_m,
        origin_height_m=PROTOCOL.origin_height_m,
        blocked=lambda x, z: True)
    env, _ = environment(dataset(tmp_path), controller=controller)
    env.reset("ep0")
    for _ in range(PROTOCOL.max_steps):
        env.step(DiscreteAction.MOVE_FORWARD)
    assert env.episode_over
    measurement = env.measure()
    assert measurement.steps == PROTOCOL.max_steps
    assert measurement.termination == "step_limit"
    assert measurement.path_length_m == pytest.approx(0.0)
    assert measurement.info["failed_actions"] == PROTOCOL.max_steps
    assert not measurement.success
    env.close()


def test_stepping_after_stop_is_refused(tmp_path):
    env, _ = environment(dataset(tmp_path))
    env.reset("ep0")
    env.step(DiscreteAction.STOP)
    with pytest.raises(EnvContractError):
        env.step(DiscreteAction.MOVE_FORWARD)
    env.close()


def test_measure_before_the_end_is_refused(tmp_path):
    env, _ = environment(dataset(tmp_path))
    env.reset("ep0")
    with pytest.raises(EnvContractError):
        env.measure()
    env.close()


# ------------------------------------------------------------------ success


def test_success_needs_both_stop_and_a_visible_goal(tmp_path):
    env, _ = environment(dataset(tmp_path), goal_visible=True)
    env.reset("ep0")
    env.step(DiscreteAction.STOP)
    assert env.measure().success

    env, _ = environment(dataset(tmp_path), goal_visible=False)
    env.reset("ep0")
    env.step(DiscreteAction.STOP)
    assert not env.measure().success


def test_a_visible_goal_without_stop_is_not_a_success(tmp_path):
    env, _ = environment(dataset(tmp_path), goal_visible=True)
    env.reset("ep0")
    for _ in range(PROTOCOL.max_steps):
        env.step(DiscreteAction.TURN_LEFT)
    measurement = env.measure()
    assert measurement.info["any_instance_visible"]
    assert not measurement.success and not measurement.stop_called
    env.close()


def test_the_two_published_readings_of_the_visibility_rule_are_both_recorded(tmp_path):
    """Upstream inspects only the first listed instance; AllenAct accepts any."""
    for rule, expected in ((SUCCESS_ANY_INSTANCE, True),
                           (SUCCESS_FIRST_INSTANCE, False)):
        env, _ = environment(dataset(tmp_path), goal_visible=True, instances=2,
                             success_rule=rule)
        env.reset("ep0")
        env.step(DiscreteAction.STOP)
        measurement = env.measure()
        assert measurement.success is expected
        assert measurement.info["any_instance_visible"] is True
        assert measurement.info["first_listed_instance_visible"] is False
        assert measurement.info["instances_of_goal_in_scene"] == 2
        assert measurement.info["success_rule"] == rule
        env.close()


def test_an_unknown_success_rule_is_refused(tmp_path):
    with pytest.raises(ValueError):
        environment(dataset(tmp_path), success_rule="whatever")


# ----------------------------------------------------------------- the score


def test_upstream_spl_arithmetic_including_its_zero_length_corner():
    assert official_spl(True, 4.0, 8.0) == pytest.approx(0.5)
    assert official_spl(True, 8.0, 4.0) == pytest.approx(1.0)
    assert official_spl(False, 4.0, 4.0) == 0.0
    # Six per cent of the published split ships l == 0: a success that moved
    # at all scores 0, and only a literal first-action STOP scores 1.
    assert official_spl(True, 0.0, 0.25) == 0.0
    assert official_spl(True, 0.0, 0.0) == pytest.approx(1.0)


def test_the_path_is_accounted_the_same_in_both_frames(tmp_path):
    """The ENU conversion is an isometry, so upstream's own accounting agrees."""
    env, _ = environment(dataset(tmp_path), goal_visible=True)
    env.reset("ep0")
    for action in (DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT,
                   DiscreteAction.MOVE_FORWARD, DiscreteAction.MOVE_FORWARD,
                   DiscreteAction.STOP):
        env.step(action)
    measurement = env.measure()
    assert measurement.path_length_m == pytest.approx(0.75, abs=1e-6)
    assert measurement.info["thor_path_length_m"] == pytest.approx(
        measurement.path_length_m, abs=1e-9)
    env.close()


def test_the_harness_scores_an_episode_end_to_end(tmp_path):
    """Contract checks, motion check, path cross-check and native cross-check."""
    env, _ = environment(dataset(tmp_path), goal_visible=True)
    agent = Scripted([DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT,
                      DiscreteAction.LOOK_UP, DiscreteAction.MOVE_FORWARD])
    summary = run_benchmark(
        env, agent, episode_ids=["ep0"],
        require_stop_for_success=PROTOCOL.require_stop_for_success,
        path_length_dimension=PROTOCOL.path_length_dimension,
        path_length_epsilon_m=PROTOCOL.path_length_epsilon_m,
        kinematics=PROTOCOL.kinematics())
    assert summary.overall.n_episodes == 1
    assert summary.overall.success_rate == 1.0
    assert summary.overall.spl == pytest.approx(official_spl(True, 3.0, 0.5))
    env.close()


def test_a_step_limited_episode_also_scores_end_to_end(tmp_path):
    env, _ = environment(dataset(tmp_path), goal_visible=False)
    agent = Scripted([DiscreteAction.TURN_RIGHT] * (PROTOCOL.max_steps + 5))
    summary = run_benchmark(
        env, agent, episode_ids=["ep0"],
        require_stop_for_success=PROTOCOL.require_stop_for_success,
        path_length_dimension=PROTOCOL.path_length_dimension,
        kinematics=PROTOCOL.kinematics())
    assert summary.overall.success_rate == 0.0
    assert summary.overall.spl == 0.0
    env.close()


def test_diagnostics_are_evaluator_only_and_name_their_distance(tmp_path):
    env, _ = environment(dataset(tmp_path))
    env.reset("ep0")
    telemetry = env.evaluation_diagnostics()
    # The cheap distance is never called distance_to_goal_m: the recorder plots
    # that key as DTG, and a Euclidean value there would not be the reported one.
    assert "euclidean_distance_to_goal_m" in telemetry
    assert "distance_to_goal_m" not in telemetry
    assert telemetry["shortest_path_m"] == pytest.approx(3.0)
    env.close()
