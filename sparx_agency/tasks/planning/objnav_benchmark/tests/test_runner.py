"""The episode loop on a scripted corridor: every action counted, every pose traced, every run resumable.

SPL divides by the path and the budget counts every action, so both must come
from the runner's own counts; a resumed run must reproduce an uninterrupted
one exactly; and a bad episode list, a nameless agent or an environment that
serves an id twice must be refused before the first episode runs, not after
the hundredth.
"""
from __future__ import annotations

import dataclasses
import itertools
import json

import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.measurement import (
    TERMINATION_STEP_LIMIT,
)
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import (
    ON_AGENT_ERROR_RECORD,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import (
    CONFIG,
    GOALS,
    WALK_AND_STOP,
    A,
    CorridorEnv,
    F,
    ScriptedAgent,
    one_episode,
)


class SplitOfE2(CorridorEnv):
    """Reports split 'train' for e2 alone, as an adapter deriving the split from a scene path might."""

    def reset(self, episode_id):
        episode, observation = super().reset(episode_id)
        if episode_id == "e2":
            episode = dataclasses.replace(episode, split="train")
        return episode, observation


class DuplicateIds(CorridorEnv):
    """Serves e1 twice, as a Habitat adapter passing raw per-scene ids through would."""

    def episode_ids(self):
        return ("e1", "e1", "e2")


class ResetCounting(ScriptedAgent):
    """The walk-and-stop script, remembering which episodes it was reset for."""

    def __init__(self, **options):
        super().__init__(**options)
        self.resets = []

    def reset(self, episode):
        self.resets.append(episode.episode_id)
        super().reset(episode)


# -- one episode -----------------------------------------------------------------

def test_every_action_counts_stop_included_and_the_path_is_what_the_poses_trace():
    """SPL divides by p and the budget counts STOP; both must come from the runner's own counts."""
    agent = ScriptedAgent((A.TURN_LEFT, A.TURN_RIGHT, A.LOOK_DOWN, A.LOOK_UP) + WALK_AND_STOP)
    record = one_episode(agent=agent)
    assert agent.seen_steps == list(range(9))
    assert record.steps == 9
    assert record.action_counts == {"STOP": 1, "MOVE_FORWARD": 4, "TURN_LEFT": 1,
                                    "TURN_RIGHT": 1, "LOOK_UP": 1, "LOOK_DOWN": 1}
    assert record.path_length_m == record.observed_path_length_m == 1.0
    assert (record.success, record.spl, record.termination) == (True, 1.0, "stop")
    assert (record.agent, record.benchmark, record.split) == ("scripted", "corridor", "test")
    assert record.native_metrics == {"success": 1.0, "distance_to_goal": 0.0}
    assert record.agent_info == {"acts": 9} and record.agent_error is None


def test_a_blocked_forward_step_counts_as_an_action_but_adds_no_path():
    """Both simulators spend the step; only the real displacement belongs in p."""
    record = one_episode(CorridorEnv({"e1": 1.0}, wall_x=0.5))
    assert (record.steps, record.path_length_m, record.observed_path_length_m) == (5, 0.5, 0.5)
    assert (record.success, record.spl, record.distance_to_goal_m) == (False, 0.0, 0.5)
    assert record.soft_spl == pytest.approx(0.5)


def test_an_agent_that_never_stops_ends_at_the_budget_as_a_step_limit():
    """The environment owns the budget; SoftSPL still credits the progress made."""
    record = one_episode(CorridorEnv({"e1": 1.0}, max_steps=6, wall_x=1.0), ScriptedAgent((F,)))
    assert (record.steps, record.action_counts["MOVE_FORWARD"]) == (6, 6)
    assert (record.termination, record.stop_called, record.success) == (
        TERMINATION_STEP_LIMIT, False, False)
    assert (record.spl, record.soft_spl) == (0.0, 1.0)


def test_an_end_with_no_goal_reachable_has_no_distance_and_stays_strict_json():
    """inf does not survive strict JSON: the record says None, and the row still serialises."""
    record = one_episode(CorridorEnv({"e1": 1.0}, unreachable_end=True))
    assert record.distance_to_goal_m is None
    assert record.native_metrics["distance_to_goal"] is None
    assert (record.success, record.soft_spl) == (False, 0.0)
    json.dumps(record.to_row(), allow_nan=False)


def test_the_wall_time_comes_from_the_injected_clock():
    """An injectable clock keeps records deterministic in tests and honest in runs."""
    ticks = itertools.count(10.0, 2.5)
    assert one_episode(clock=lambda: next(ticks)).wall_s == 2.5


def test_run_episode_refuses_an_episode_of_another_split_than_expected():
    """The run's identity is checked after the reset and before the agent sees the episode."""
    agent = ResetCounting()
    with pytest.raises(EnvContractError, match="but this run is corridor/val"):
        one_episode(CorridorEnv({"e1": 1.0}), agent, expected=("corridor", "val"))
    assert agent.resets == []
    assert one_episode(expected=("corridor", "test")).success


@pytest.mark.parametrize("expected", ["corridor/test", ("corridor",), ("corridor", " ")],
                         ids=["string", "one_item", "blank_split"])
def test_a_malformed_expected_identity_is_refused_before_the_reset(expected):
    """A malformed identity would otherwise refuse every episode, or none."""
    env = CorridorEnv({"e1": 1.0})
    with pytest.raises(HarnessError, match="expected must be None or a"):
        one_episode(env, expected=expected)
    assert env.resets == []


# -- a whole run -----------------------------------------------------------------

def test_run_benchmark_runs_logs_and_summarises_every_episode(tmp_path):
    """The loop the smoke run and every benchmark branch rely on, end to end."""
    logger = MetricsLogger(tmp_path / "run", CONFIG, argv=[])
    calls = []
    summary = run_benchmark(CorridorEnv(GOALS), ScriptedAgent(), logger=logger,
                            progress=lambda i, n, r: calls.append((i, n, r.episode_id)))
    assert calls == [(1, 3, "e1"), (2, 3, "e2"), (3, 3, "e3")]
    assert (summary.agent, summary.overall.n_episodes) == ("scripted", 3)
    assert summary.overall.success_rate == pytest.approx(2 / 3)
    assert logger.completed_episode_ids() == frozenset(GOALS)
    assert (tmp_path / "run" / "summary.json").is_file()
    assert json.loads((tmp_path / "run" / "run.json").read_text())["n_episodes"] == 3


def test_a_resumed_run_runs_only_the_missing_episodes_and_matches_a_clean_run(tmp_path):
    """After a crash only the missing episodes run, and the summary equals an uninterrupted run's."""
    crashing = ScriptedAgent(misbehave_at=0, in_episode="e3")
    with pytest.raises(RuntimeError, match="boom"):
        # The with-block releases the directory, as the crashed process's exit would.
        with MetricsLogger(tmp_path / "run", CONFIG) as logger:
            run_benchmark(CorridorEnv(GOALS), crashing, logger=logger)
    env = CorridorEnv(GOALS)
    resumed = run_benchmark(env, ScriptedAgent(),
                            logger=MetricsLogger(tmp_path / "run", CONFIG, resume=True))
    assert env.resets == ["e3"]
    assert resumed == run_benchmark(CorridorEnv(GOALS), ScriptedAgent())


def test_a_resume_refuses_a_file_holding_episodes_this_run_does_not_request(tmp_path):
    """A summary of fewer episodes than the file holds would describe a different experiment."""
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(), logger=MetricsLogger(tmp_path / "run", CONFIG))
    env = CorridorEnv(GOALS)
    with pytest.raises(ResultsError, match="does not request"):
        run_benchmark(env, ScriptedAgent(), episode_ids=["e1"],
                      logger=MetricsLogger(tmp_path / "run", CONFIG, resume=True))
    assert env.resets == []


def test_a_resume_under_another_agent_is_refused_before_any_episode_runs(tmp_path):
    """Reused episodes would be summarised under the new agent's name."""
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(), episode_ids=["e1"],
                  logger=MetricsLogger(tmp_path / "run", CONFIG))
    other, env = ScriptedAgent(), CorridorEnv(GOALS)
    other.name = "other"
    with pytest.raises(ResultsError, match="this run's agent is 'other'"):
        run_benchmark(env, other, logger=MetricsLogger(tmp_path / "run", CONFIG, resume=True))
    assert env.resets == []


@pytest.mark.parametrize("ids,message", [
    (["e1", "e9"], "does not serve episodes \\['e9'\\]"), (["e1", "e1"], "more than once"),
    ([], "no episodes to run"), ("e1", "single string")],
    ids=["unknown", "repeated", "empty", "string"])
def test_a_bad_episode_list_is_refused_before_any_episode_runs(ids, message):
    """A typo found after a hundred episodes costs the hundred."""
    env = CorridorEnv(GOALS)
    with pytest.raises(HarnessError, match=message):
        run_benchmark(env, ScriptedAgent(), episode_ids=ids)
    assert env.resets == []


@pytest.mark.parametrize("ids", [None, ["e1", "e2"]], ids=["served_list", "explicit_list"])
def test_an_environment_serving_an_id_twice_is_refused_before_any_episode_runs(ids):
    """With a repeated id no request can say which episode it means; the adapter must scene-qualify its ids."""
    env = DuplicateIds(GOALS)
    with pytest.raises(EnvContractError,
                       match=r"DuplicateIds.episode_ids\(\) serves \['e1'\] more than "
                             r"once.*scene-qualify"):
        run_benchmark(env, ScriptedAgent(), episode_ids=ids)
    assert env.resets == []


def test_a_blank_agent_name_is_refused_before_any_episode_runs(tmp_path):
    """Every row carries the name; a blank one would be refused only by the summary, after the whole run."""
    agent, env = ScriptedAgent(), CorridorEnv(GOALS)
    agent.name = "   "
    with pytest.raises(HarnessError, match="has no name"):
        run_benchmark(env, agent, logger=MetricsLogger(tmp_path / "run", CONFIG))
    assert env.resets == []


def test_an_episode_of_another_split_is_refused_before_the_agent_sees_it():
    """Without a logger, one mismatching episode would surface only in summarise, losing every record."""
    env, agent = SplitOfE2(GOALS), ResetCounting()
    with pytest.raises(EnvContractError,
                       match="episode 'e2' is corridor/train, but this run is corridor/test"):
        run_benchmark(env, agent)
    assert env.resets == ["e1", "e2"] and agent.resets == ["e1"]


def test_a_resumed_run_holds_its_episodes_to_the_logged_split(tmp_path):
    """A resume's first episode must match the logged ones before it runs, not at its log."""
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(), episode_ids=["e1"],
                  logger=MetricsLogger(tmp_path / "run", CONFIG))
    env, agent = SplitOfE2(GOALS), ResetCounting()
    with pytest.raises(EnvContractError, match="this run is corridor/test"):
        run_benchmark(env, agent, logger=MetricsLogger(tmp_path / "run", CONFIG, resume=True))
    assert env.resets == ["e2"] and agent.resets == []


def test_a_bad_confidence_is_refused_before_any_episode_runs():
    """The interval is computed last; a percent typed as 95 must not cost the whole run."""
    env = CorridorEnv(GOALS)
    with pytest.raises(HarnessError, match="confidence"):
        run_benchmark(env, ScriptedAgent(), confidence=95)
    assert env.resets == []


def test_a_recorded_agent_error_is_counted_in_the_summary_and_the_run_goes_on():
    """One crashing episode must not cost the sweep, and must show in the table."""
    summary = run_benchmark(CorridorEnv(GOALS), ScriptedAgent(misbehave_at=1, in_episode="e2"),
                            on_agent_error=ON_AGENT_ERROR_RECORD)
    assert summary.overall.n_episodes == 3 and summary.agent_errors == 1
    assert summary.terminations == {"stop": 2, "step_limit": 0, "agent_error": 1}
