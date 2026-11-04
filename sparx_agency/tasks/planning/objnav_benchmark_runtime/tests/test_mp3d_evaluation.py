"""MP3D end to end on the shared execution path, with a scripted simulator, and the report it leaves.

This is the whole pipeline a real run uses -- loader, environment, headless
agent, action converter, the shared runner's kinematic and path cross-checks,
the metrics logger and the MP3D report -- with habitat-sim replaced by a stub
that obeys the published action geometry exactly. It proves the plumbing and
the arithmetic, not a score: no Matterport3D scene is rendered here.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.measurement import (
    TERMINATION_STEP_LIMIT, TERMINATION_STOP)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    ACTION_NAMES, TERMINATION_AGENT_ERROR, EpisodeRecord)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    APEXNAV_SUCCESS_DISTANCE_M, MP3D_V1, PUBLISHED_VAL)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import (
    merge_runs, rescore, write_report)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.sota import reported_results
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d import (
    OTHER_SCENE, SCENE, row, scripted_distances, write_release)

PROTOCOL = MP3D_V1
#: Where the plumbing policy walks to, in ENU: 3 m straight ahead of the start
#: (START_ENU is (-2, -1, 0.5) facing +x).
GOAL_ENU = (1.0, -1.0)


class ScriptedSimulator:
    """Executes the published action geometry: 0.25 m forward, 30 degree turns, no climb."""

    def __init__(self, protocol=PROTOCOL):
        self.protocol = protocol
        self.pose = None
        self.closed = False
        self.actions = []
        self._scene = None

    @property
    def pathfinder(self):
        return "navmesh of %s" % self._scene

    def _frame(self):
        shape = (self.protocol.height, self.protocol.width)
        return (np.zeros(shape + (3,), np.uint8),
                np.full(shape, 3.0, np.float32), self.pose)

    def reset(self, mesh, navmesh, position, rotation_wxyz, seed):
        self._scene = str(mesh)
        x, y, z = (float(v) for v in position)
        self.pose = AgentPose(x=-z, y=-x, z=y, yaw=2.0 * math.atan2(rotation_wxyz[2], rotation_wxyz[0]))
        return self._frame()

    def step(self, action):
        self.actions.append(action)
        pose, spec = self.pose, self.protocol.actions()
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
        assert target.category == "chair" and target.accepts("chair")
        assert not hasattr(episode, "goals")

    def plan(self, observation):
        assert not hasattr(observation, "distance_to_goal_m")
        return NavigationCommand.follow(
            [(observation.pose.x, observation.pose.y), GOAL_ENU], stop_on_arrival=True)


def prepared(tmp_path, per_scene=2, final_m=0.05):
    """A written release behind an environment whose geodesic ends at ``final_m``.

    One episode per scene: the scripted distance's first answer is ``l`` and
    a goal set's field is built once per scene, so a second episode of the
    same scene would read ``final_m`` as its ``l``.
    """
    scenes = {scene: [row()] for scene in (SCENE, OTHER_SCENE)[:per_scene]}
    dataset = write_release(tmp_path, PROTOCOL, scenes)
    env = MP3DEnv(dataset, PROTOCOL, simulator=ScriptedSimulator(),
                  distance_factory=scripted_distances((3.0, final_m)))
    return dataset, env


def run_config(dataset, *, full_split=False, method="test-straight-line", selected=None):
    """A run manifest shaped like the one ``mp3d.run.prepare`` assembles."""
    ids = list(dataset.episode_ids()) if selected is None else list(selected)
    return {"protocol": asdict(PROTOCOL), "protocol_id": PROTOCOL.protocol_id,
            "runtime": {"habitat-sim": PROTOCOL.reference_habitat_sim_version},
            "reference_habitat_sim_version_match": True,
            "method": {"method": method, "publishable": method != "diagnostic-stop"},
            "seed": 0, "gpu_device": 0, "shards": 1, "shard_index": 0, "limit": None, "scenes": [],
            "starts_validated": True, "excluded_unreachable_episode_ids": [],
            "full_split": full_split, "dataset": dataset.manifest(hash_scenes=False),
            "selected_episode_ids": ids}


def test_a_scripted_mp3d_run_passes_every_shared_cross_check(tmp_path):
    dataset, env = prepared(tmp_path)
    output = tmp_path / "run"
    seen = []
    summary = run_evaluation(
        env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
        config=run_config(dataset), settings=PROTOCOL.evaluation_settings(),
        progress=lambda index, total, row: seen.append((index, total, row.episode_id)))
    assert summary.overall.n_episodes == 2
    assert summary.overall.success_rate == 1.0
    # l = 3.0 m scripted, p = 12 strides of 0.25 m: SPL is exactly 1.
    assert summary.overall.spl == pytest.approx(1.0)
    assert [item[0] for item in seen] == [1, 2] and seen[0][1] == 2
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert {row["benchmark"] for row in rows} == {"mp3d_v1"}
    assert {row["split"] for row in rows} == {"val"}
    assert all(row["stop_called"] for row in rows)
    assert all(row["path_length_m"] == pytest.approx(3.0) for row in rows)
    assert env._simulator.closed


def test_the_run_records_the_habitat_lab_rules_and_the_protocol_it_scored_under(tmp_path):
    dataset, env = prepared(tmp_path, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config=run_config(dataset), settings=PROTOCOL.evaluation_settings())
    config = json.loads((output / "run.json").read_text())["config"]
    assert config["evaluation"]["require_stop_for_success"] is True
    assert config["evaluation"]["path_length_dimension"] == "3d"
    assert config["evaluation"]["path_length_epsilon_m"] == 0.0
    assert config["evaluation"]["kinematics"]["climb_m"] == 0.85
    assert config["protocol"]["navmesh"] == "agent"
    assert config["protocol"]["success_distance_m"] == 0.1
    assert config["selected_episode_ids"] == ["%s/000000" % SCENE]


def test_the_report_refuses_to_call_a_subset_a_benchmark_result(tmp_path):
    dataset, env = prepared(tmp_path, per_scene=1)
    output = tmp_path / "run"
    run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                   config=run_config(dataset), settings=PROTOCOL.evaluation_settings())
    audit = write_report(output)
    assert audit["full_split_completed"] is False
    assert audit["episodes_published"] == 2195 and audit["scenes_published"] == 11
    assert audit["navmesh"] == "agent" and audit["success_distance_m"] == 0.1
    assert audit["starts_validated"] is True
    text = (output / "comparison.md").read_text()
    assert "NOT a full benchmark result" in text
    assert "SG-Nav-GPT" in text and "ApexNav" in text
    assert "arXiv:2410.08189" in text and "arXiv:2504.14478" in text
    assert "0.2 m success radius" in text
    assert json.loads((output / "audit.json").read_text())["apexnav_radius_extra_successes"] == 0


def test_a_stop_just_outside_the_official_radius_is_a_success_only_under_apexnavs(tmp_path):
    """The two tables of the report come from the same rows: 0.15 m fails at
    habitat-lab's 0.1 m and succeeds at ApexNav's 0.2 m."""
    dataset, env = prepared(tmp_path, per_scene=1, final_m=0.15)
    output = tmp_path / "run"
    summary = run_evaluation(env, StraightToTheGoal(), mp3d_label_mapper(), output=output,
                             config=run_config(dataset), settings=PROTOCOL.evaluation_settings())
    assert summary.overall.success_rate == 0.0
    audit = write_report(output)
    assert audit["apexnav_radius_extra_successes"] == 1
    assert audit["apexnav_radius_success_rate"] == 1.0


# --- the re-scoring arithmetic, on rows alone ---------------------------------

def record(episode_id, *, distance=0.05, stopped=True, error=None, shortest=2.0, travelled=8.0,
           scene=SCENE, agent="rpt"):
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts.update(MOVE_FORWARD=31, TURN_LEFT=8, STOP=1 if stopped else 0)
    success = bool(error is None and stopped and distance is not None
                   and distance < PROTOCOL.success_distance_m)
    ratio = shortest / max(shortest, travelled)
    termination = (TERMINATION_AGENT_ERROR if error is not None else
                   TERMINATION_STOP if stopped else TERMINATION_STEP_LIMIT)
    return EpisodeRecord(
        benchmark=PROTOCOL.benchmark, split=PROTOCOL.split, episode_id=episode_id,
        scene_id=scene, target_category="chair", agent=agent, success=success,
        spl=float(success) * ratio, soft_spl=0.42, distance_to_goal_m=distance,
        path_length_m=travelled, observed_path_length_m=travelled,
        shortest_path_m=shortest, start_distance_to_goal_m=shortest,
        steps=sum(counts.values()), stop_called=termination != TERMINATION_STEP_LIMIT,
        termination=termination, action_counts=counts, wall_s=1.5, agent_error=error)


def test_rescoring_at_the_official_radius_reproduces_the_rows_and_a_looser_one_only_adds():
    rows = [record("a", distance=0.05), record("b", distance=0.15), record("c", distance=0.25),
            record("d", distance=None), record("e", distance=0.05, stopped=False),
            record("f", distance=0.0, error="RuntimeError: detector died")]
    same = rescore(rows, PROTOCOL.success_distance_m)
    assert [r.success for r in same] == [r.success for r in rows]
    assert [r.spl for r in same] == [r.spl for r in rows]
    loose = rescore(rows, APEXNAV_SUCCESS_DISTANCE_M)
    assert [r.success for r in loose] == [True, True, False, False, False, False]
    assert loose[1].spl == pytest.approx(0.25) and loose[1].soft_spl == rows[1].soft_spl


def test_the_published_rows_are_fractions_with_a_source_each():
    rows = reported_results("mp3d_v1")
    assert len(rows) >= 10
    assert all(0 < r.success_rate < 1 and 0 < r.spl < 1 and "arXiv" in r.source for r in rows)
    assert {r.method for r in rows} >= {"SG-Nav-GPT", "ApexNav", "VLFM (ApexNav's table)", "VLFM (SG-Nav's table)"}
    with pytest.raises(KeyError):
        reported_results("hm3d_v1")


# --- shards and completeness -----------------------------------------------------

def write_run(directory, dataset, records, **options):
    config = run_config(dataset, **options)
    with MetricsLogger(directory, config, argv=["pytest"]) as logger:
        for record_ in records:
            logger.log(record_)
        logger.finish(summarise(records))
    return config


def test_two_disjoint_complete_shards_merge_into_one_run_and_a_subset_stays_a_subset(tmp_path, monkeypatch):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row(), row()], OTHER_SCENE: [row(), row()]})
    ids = dataset.episode_ids()
    first = [record(ids[0]), record(ids[2], scene=OTHER_SCENE)]
    second = [record(ids[1]), record(ids[3], scene=OTHER_SCENE, distance=0.5)]
    for name, rows, index in (("a", first, 0), ("b", second, 1)):
        config = run_config(dataset, full_split=False, method="rpt", selected=[r.episode_id for r in rows])
        config.update(shards=2, shard_index=index)
        with MetricsLogger(tmp_path / name, config, argv=["pytest"]) as logger:
            for r in rows:
                logger.log(r)
            logger.finish(summarise(rows))
    # Four episodes are not the published 2,195 -- unless the published count is four.
    audit = merge_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "merged")
    assert audit["episodes_scored"] == 4 and audit["full_split_completed"] is False
    monkeypatch.setitem(PUBLISHED_VAL, "v1", {"scenes": 2, "episodes": 4})
    audit = merge_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "merged_full")
    assert audit["full_split_completed"] is True
    assert (tmp_path / "merged_full" / "comparison.md").read_text().startswith("# Full MP3D ObjectNav v1 val validation")
    with pytest.raises(ValueError, match="Missing shards"):
        merge_runs([tmp_path / "a"], tmp_path / "half")


def test_a_run_of_another_benchmark_or_protocol_is_not_reported_as_mp3d(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL)
    config = run_config(dataset, method="rpt")
    config["protocol"]["success_distance_m"] = 0.2
    with MetricsLogger(tmp_path / "run", config, argv=["pytest"]) as logger:
        logger.log(record(dataset.episode_ids()[0]))
        logger.finish(summarise([record(dataset.episode_ids()[0])]))
    with pytest.raises(ValueError, match="different MP3D protocol"):
        write_report(tmp_path / "run")
