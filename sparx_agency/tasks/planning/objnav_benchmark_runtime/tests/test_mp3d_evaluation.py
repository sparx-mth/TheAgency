"""MP3D end to end on the shared execution path, with a scripted simulator.

This is the whole pipeline a real run uses -- loader, environment, headless
agent, action converter, the shared runner's kinematic and path cross-checks,
the metrics logger and the MP3D report -- with habitat-sim replaced by a stub
that obeys the published action geometry exactly. It proves the plumbing and
the arithmetic, not a score: no Matterport3D scene is rendered here.
"""
import json
import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import env as env_module
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import write_report
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d import (
    SCENE, StubDistance, enu, release)

GOAL_ENU = (-5.0, -1.0)


class ScriptedSimulator:
    """Executes the published action geometry: 0.25 m forward, 30 degree turns."""

    pathfinder = "published-navmesh"

    def __init__(self):
        self.pose = None
        self.closed = False
        self.actions = []

    def _frame(self):
        shape = (PROTOCOL.height, PROTOCOL.width)
        return (np.zeros(shape + (3,), np.uint8),
                np.full(shape, 3.0, np.float32), self.pose)

    def reset(self, mesh, navmesh, position, rotation_wxyz, seed):
        self.pose = enu(position)
        return self._frame()

    def step(self, action):
        self.actions.append(action)
        pose, spec = self.pose, PROTOCOL.actions()
        if action == DiscreteAction.MOVE_FORWARD:
            self.pose = AgentPose(
                x=pose.x + spec.forward_step_m * math.cos(pose.yaw),
                y=pose.y + spec.forward_step_m * math.sin(pose.yaw),
                z=pose.z, yaw=pose.yaw, camera_pitch=pose.camera_pitch)
        elif action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
            sign = 1.0 if action == DiscreteAction.TURN_LEFT else -1.0
            self.pose = AgentPose(
                x=pose.x, y=pose.y, z=pose.z,
                yaw=pose.yaw + sign * math.radians(spec.turn_angle_deg),
                camera_pitch=pose.camera_pitch)
        return self._frame()

    def close(self):
        self.closed = True


class StraightToTheGoal:
    """A plumbing policy: walk the straight line to a fixed point and stop there.

    It sees only :class:`ObjNavObservation`; the goal below is the fixture's,
    written into the test, never read from the environment.
    """

    name = "test-straight-line"
    converter_params = ActionConverterParams(goal_tolerance_m=0.1)

    def reset(self, episode, target):
        assert target.category == "chair" and "chair" in target.accept_labels
        assert not hasattr(episode, "goals")

    def plan(self, observation):
        assert not hasattr(observation, "distance_to_goal_m")
        return NavigationCommand.follow(
            [(observation.pose.x, observation.pose.y), GOAL_ENU], stop_on_arrival=True)


def prepared(tmp_path, monkeypatch, per_scene=2):
    episodes_dir, scenes_dir = release(tmp_path, per_scene=per_scene)
    monkeypatch.setattr(env_module, "ViewPointDistance", StubDistance)
    dataset = MP3DDataset(episodes_dir, scenes_dir, full=False, scene=SCENE)
    return dataset, MP3DEnv(dataset, simulator=ScriptedSimulator())


def test_a_scripted_mp3d_run_passes_every_shared_cross_check(tmp_path, monkeypatch):
    dataset, env = prepared(tmp_path, monkeypatch)
    output = tmp_path / "run"
    seen = []
    summary = run_evaluation(
        env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
        config={"protocol": {"protocol_id": PROTOCOL.protocol_id}, "purpose": "plumbing"},
        settings=PROTOCOL.evaluation(),
        progress=lambda index, total, row: seen.append((index, total, row.episode_id)))
    assert summary.overall.n_episodes == 2
    assert summary.overall.success_rate == 1.0
    assert summary.overall.spl == pytest.approx(1.0)
    assert [item[0] for item in seen] == [1, 2] and seen[0][1] == 2
    rows = [json.loads(line) for line in
            (output / "episodes.jsonl").read_text().splitlines()]
    assert {row["benchmark"] for row in rows} == {"mp3d"}
    assert {row["split"] for row in rows} == {"val"}
    assert all(row["stop_called"] for row in rows)


def test_the_run_is_scored_with_stop_required_and_a_three_dimensional_path(
        tmp_path, monkeypatch):
    dataset, env = prepared(tmp_path, monkeypatch, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config={"purpose": "plumbing"}, settings=PROTOCOL.evaluation())
    config = json.loads((output / "run.json").read_text())["config"]
    assert config["evaluation"]["require_stop_for_success"] is True
    assert config["evaluation"]["path_length_dimension"] == "3d"
    assert config["evaluation"]["path_length_epsilon_m"] == 0.0
    assert config["selected_episode_ids"] == ["%s/000000" % SCENE]


def test_the_report_refuses_to_call_a_subset_a_benchmark_result(tmp_path, monkeypatch):
    from dataclasses import asdict

    dataset, env = prepared(tmp_path, monkeypatch, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config={"protocol": asdict(PROTOCOL), "full_split": False,
                           "reference_sim_version_match": True,
                           "method": {"method": "test-straight-line"},
                           "dataset": dataset.manifest(), "start_validation": None},
                   settings=PROTOCOL.evaluation())
    audit = write_report(output)
    assert audit["full_split_completed"] is False
    assert audit["published_episodes"] == 2195 and audit["published_scenes"] == 11
    text = (output / "comparison.md").read_text()
    assert "NOT a full benchmark result" in text
    assert "SG-Nav-GPT" in text and "ApexNav" in text
    assert "arXiv:2410.08189" in text and "arXiv:2504.14478" in text


def test_a_skipped_start_preflight_does_not_read_as_a_clean_one(tmp_path, monkeypatch):
    """"Not checked" and "checked, found nothing" must not print the same number."""
    from dataclasses import asdict

    dataset, env = prepared(tmp_path, monkeypatch, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config={"protocol": asdict(PROTOCOL), "full_split": False,
                           "reference_sim_version_match": True,
                           "method": {"method": "test-straight-line"},
                           "dataset": dataset.manifest(), "start_validation": None},
                   settings=PROTOCOL.evaluation())
    audit = write_report(output)
    assert audit["starts_preflighted"] is False
    assert audit["unreachable_starts"] is None
    assert audit["published_geodesic_gaps"] is None


def test_a_preflighted_run_reports_the_counts_it_actually_checked(tmp_path, monkeypatch):
    from dataclasses import asdict

    dataset, env = prepared(tmp_path, monkeypatch, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config={"protocol": asdict(PROTOCOL), "full_split": False,
                           "reference_sim_version_match": True,
                           "method": {"method": "test-straight-line"},
                           "dataset": dataset.manifest(),
                           "start_validation": {"checked": 1, "unreachable_starts": [],
                                                "published_geodesic_gaps": {"a": 0.4}}},
                   settings=PROTOCOL.evaluation())
    audit = write_report(output)
    assert audit["starts_preflighted"] is True
    assert audit["unreachable_starts"] == 0 and audit["published_geodesic_gaps"] == 1


def test_the_protocol_recorded_in_a_run_names_the_navmesh_it_scored_on(tmp_path, monkeypatch):
    from dataclasses import asdict

    dataset, env = prepared(tmp_path, monkeypatch, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config={"protocol": asdict(PROTOCOL)}, settings=PROTOCOL.evaluation())
    config = json.loads((output / "run.json").read_text())["config"]
    assert config["protocol"]["navmesh_source"] == "agent"
