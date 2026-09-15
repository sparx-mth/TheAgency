"""Shared execution, provenance and recording on the existing synthetic corridor."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import fake_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import CorridorEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import (
    EvaluationSettings, evaluation_configuration, run_evaluation,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    check_frozen_configuration, freeze_configuration, select_episodes, source_fingerprint,
)


class Env(CorridorEnv):
    def __init__(self):
        super().__init__({"one": 1.0, "two": 1.0})
        self.closed = False

    def close(self):
        self.closed = True


class LinePolicy:
    name = "synthetic-line"
    converter_params = ActionConverterParams(goal_tolerance_m=0.1)

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        assert not hasattr(observation, "distance_to_goal_m")
        return NavigationCommand.follow([(0, 0), (1, 0)], stop_on_arrival=True)


class Sink:
    def __init__(self, path, fps):
        self.frames = 0

    def write(self, frame):
        assert frame.shape == (900, 1600, 3)
        self.frames += 1

    def close(self):
        pass


def test_shared_evaluation_records_without_privileged_telemetry_and_resumes(tmp_path):
    env, output = Env(), tmp_path / "run"
    summary = run_evaluation(env, LinePolicy(), fake_label_mapper(), output=output,
                             config={"purpose": "synthetic"}, record=True, writer_factory=Sink)
    assert summary.overall.success_rate == 1.0 and env.closed
    assert len(list(output.glob("recordings/*/trajectory.csv"))) == 2
    text = (output / "index.html").read_text()
    assert "corridor / test" in text and "configured evaluator" in text
    assert "Gibson" not in text and "SemExp" not in text
    assert 'href="comparison.md"' not in text
    env = Env()
    resumed = run_evaluation(env, LinePolicy(), fake_label_mapper(), output=output,
                             config={"purpose": "synthetic"}, resume=True, dashboard=False)
    assert resumed.overall.n_episodes == 2 and env.resets == [] and env.closed
    assert (output / "episodes.jsonl").read_text().count("\n") == 2


def test_frozen_configuration_is_checked_before_reset_or_output(tmp_path):
    env = Env()
    config = evaluation_configuration({"model": "one"}, env.episode_ids(), policy=LinePolicy())
    lock = tmp_path / "lock.json"
    freeze_configuration(config, lock, development_note="synthetic unit development")
    with pytest.raises(ValueError, match="changed"):
        run_evaluation(env, LinePolicy(), fake_label_mapper(), output=tmp_path / "out",
                       config={"model": "two"}, frozen_lock=lock)
    assert env.resets == [] and env.closed and not (tmp_path / "out").exists()
    env = Env()
    result = run_evaluation(env, LinePolicy(), fake_label_mapper(), output=tmp_path / "out",
                            config={"model": "one"}, frozen_lock=lock,
                            expected_episode_ids=env.episode_ids())
    assert result.overall.n_episodes == 2


def test_locks_pin_arbitrary_episode_count_order_and_configuration(tmp_path):
    config = {"selected_episode_ids": ["a/4", "b/2", "b/9"], "protocol": {"stop": True}}
    path = tmp_path / "lock.json"
    freeze_configuration(config, path, development_note="No published benchmark used")
    assert check_frozen_configuration(config, path)["held_out_claim"] is False
    with pytest.raises(FileExistsError):
        freeze_configuration(config, path, development_note="still synthetic")
    with pytest.raises(ValueError, match="identities"):
        check_frozen_configuration(config, path, expected_episode_ids=["a/4", "b/2"])
    with pytest.raises(ValueError, match="changed"):
        check_frozen_configuration(dict(config, selected_episode_ids=list(reversed(config["selected_episode_ids"]))), path)
    with pytest.raises(ValueError, match="changed"):
        check_frozen_configuration(dict(config, extra=None), path)
    with pytest.raises(ValueError, match="exposure"):
        freeze_configuration(config, tmp_path / "other", development_note="")


def test_source_fingerprint_and_sharding_are_stable_and_explicit(tmp_path):
    file = tmp_path / "policy.py"
    file.write_text("x = 1\n")
    before = source_fingerprint(tmp_path)
    (tmp_path / "notes.md").write_text("not executable")
    assert source_fingerprint(tmp_path) == before
    file.write_text("x = 2\n")
    assert source_fingerprint(tmp_path) != before
    assert select_episodes(["a", "b", "c", "d"], shards=2, shard_index=1, limit=1) == ("b",)
    for ids in ([], ["a", "a"], [""]):
        with pytest.raises(ValueError):
            select_episodes(ids)
    with pytest.raises(ValueError):
        select_episodes(["a"], limit=True)


@pytest.mark.parametrize("dimension,epsilon", [("3d", 0.0), ("planar", 0.002)])
def test_path_accounting_is_explicit_and_defaults_remain_3d(dimension, epsilon, tmp_path):
    class SlopedEnv(Env):
        def step(self, action):
            before = self.pose
            observation = super().step(action)
            z = before.z + (0.1 if action == DiscreteAction.MOVE_FORWARD else 0.0)
            self.pose = replace(observation.pose, z=z)
            return replace(observation, pose=self.pose)

        def measure(self):
            measured = super().measure()
            length = 4 * (math.hypot(0.25, 0.1) if dimension == "3d" else 0.25) + epsilon
            return replace(measured, path_length_m=length)

    options = EvaluationSettings(path_length_dimension=dimension, path_length_epsilon_m=epsilon,
                                 kinematics=None)
    output = tmp_path / "run"
    run_evaluation(SlopedEnv(), LinePolicy(), fake_label_mapper(), output=output,
                   config={"purpose": "synthetic slope"}, settings=options)
    row = json.loads((output / "episodes.jsonl").read_text().splitlines()[0])
    assert row["path_length_m"] == pytest.approx(row["observed_path_length_m"])
    assert EvaluationSettings().require_stop_for_success
    assert EvaluationSettings().path_length_dimension == "3d"


def test_renderer_failure_still_closes_environment(tmp_path, monkeypatch):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime import recording
    monkeypatch.setattr(recording.EpisodeRecorder, "close", lambda self: (_ for _ in ()).throw(RuntimeError("writer failed")))
    env = Env()
    with pytest.raises(RuntimeError, match="writer failed"):
        run_evaluation(env, LinePolicy(), fake_label_mapper(), output=tmp_path / "out",
                       config={}, record=True, writer_factory=Sink)
    assert env.closed


def test_json_reloaded_configuration_keeps_tuple_settings_and_detects_converter_drift():
    class TuplePolicy(LinePolicy):
        def configuration(self):
            return {"prompts": ("chair", "bed"), "nested": {1: (0.2, 0.3)}}

    policy = TuplePolicy()
    prepared = evaluation_configuration({}, ["one"], policy=policy)
    reloaded = json.loads(json.dumps(prepared))
    assert evaluation_configuration(reloaded, ["one"], policy=policy) == prepared
    policy.converter_params = replace(policy.converter_params, lookahead_m=0.4)
    with pytest.raises(ValueError, match="agent_converter"):
        evaluation_configuration(reloaded, ["one"], policy=policy)




def test_a_caller_may_watch_episodes_finish_without_displacing_the_recorder(tmp_path):
    """A thousand-episode run that prints nothing until it ends is unusable, and
    an adapter's CLI is the only thing positioned to say how it is going. The
    recorder's own hook still runs first: it is what finishes writing an
    episode's artifacts, and a caller's printer must not be able to skip it."""
    seen = []
    env = Env()
    summary = run_evaluation(env, LinePolicy(), fake_label_mapper(),
                             output=tmp_path / "out", config={"experiment": "progress"},
                             progress=lambda index, total, row: seen.append(
                                 (index, total, row.episode_id)))
    assert [(index, total) for index, total, _ in seen] == [(1, 2), (2, 2)]
    assert {episode for _, _, episode in seen} == set(env.episode_ids())
    assert summary.overall.n_episodes == 2
