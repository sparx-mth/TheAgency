"""The agent-error policy: a crash is raised by default and, when recorded, is a failure, never a success.

A decision the benchmark cannot execute is the agent's bug, never the
environment's, so it is an ``AgentContractError`` and it never reaches the
environment, whichever the policy. An invariant of our own code breaking is
never the agent's, so it stops the run whichever the policy.
"""
from __future__ import annotations

import collections.abc

import pytest

from sparx_agency.core.planning.objnav.errors import (
    AgentContractError,
    ObjNavInternalError,
    UnknownCategoryError,
)
from sparx_agency.core.planning.objnav.types.actions import NAVIGATION_ACTIONS
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.measurement import TERMINATION_STOP
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import (
    ON_AGENT_ERROR_RECORD,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    TERMINATION_AGENT_ERROR,
)
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import (
    CONFIG,
    GOALS,
    A,
    CorridorEnv,
    ScriptedAgent,
    one_episode,
)


def internal_error():
    raise ObjNavInternalError("converter invariant broke")


def harness_error():
    raise HarnessError("harness invariant broke")


class NanInE2(ScriptedAgent):
    """Reports a mean over no samples -- a NaN -- in e2 alone."""

    def episode_info(self):
        return {"mean_cross_track_m": float("nan") if self.episode_id == "e2" else 0.0}


class InternalInfo(ScriptedAgent):
    """Its diagnostics break one of our invariants."""

    def episode_info(self):
        raise ObjNavInternalError("episode log invariant broke")


class RaisingMapping(collections.abc.Mapping):
    """Diagnostics read lazily off a policy's state, which raise when read."""

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        raise RuntimeError("lazy diagnostic broke")

    def __len__(self):
        return 1


def nested(depth):
    """A list nested ``depth`` deep: deeper than the JSON encoder recurses."""
    value = []
    for _ in range(depth):
        value = [value]
    return value


def test_an_agent_error_is_raised_by_default():
    """A crash is a bug to look at, not a failure to average."""
    with pytest.raises(RuntimeError, match="boom"):
        one_episode(agent=ScriptedAgent(misbehave_at=2))


def test_a_recorded_agent_error_ends_on_a_forced_stop_and_is_never_credited():
    """Even a forced STOP that lands on the goal must not turn a crash into a success."""
    env = CorridorEnv({"e1": 1.0})
    record = one_episode(env, ScriptedAgent(misbehave_at=4), on_agent_error=ON_AGENT_ERROR_RECORD)
    assert env.stopped and env.measure().success
    assert (record.termination, record.agent_error) == (TERMINATION_AGENT_ERROR, "RuntimeError: boom")
    assert (record.success, record.spl, record.stop_called, record.soft_spl) == (False, 0.0, True, 1.0)
    assert (record.steps, record.action_counts["STOP"]) == (5, 1)


BAD_DECISIONS = [(lambda: AgentDecision(A.LOOK_DOWN), "does not accept"),
                 (lambda: A.STOP, "must return an AgentDecision")]


@pytest.mark.parametrize("misbehaviour,message", BAD_DECISIONS,
                         ids=["disallowed_action", "bare_action"])
def test_a_decision_the_benchmark_cannot_execute_is_an_agent_error(misbehaviour, message):
    """A LOOK on a benchmark without tilt is the agent's bug; the environment must never receive it."""
    def attempt(**options):
        env = CorridorEnv({"e1": 1.0}, actions=NAVIGATION_ACTIONS)
        agent = ScriptedAgent(misbehave_at=0, misbehaviour=misbehaviour)
        return one_episode(env, agent, **options)
    with pytest.raises(AgentContractError, match=message):
        attempt()
    record = attempt(on_agent_error=ON_AGENT_ERROR_RECORD)
    assert record.agent_error.startswith("AgentContractError: ")
    assert (record.steps, record.action_counts["STOP"]) == (1, 1)


@pytest.mark.parametrize("misbehaviour,kind", [(internal_error, ObjNavInternalError),
                                               (harness_error, HarnessError)],
                         ids=["objnav_internal", "harness"])
def test_our_own_invariant_breaking_in_act_stops_the_run_even_when_recording(misbehaviour, kind):
    """A bug in our converter or harness, recorded as an agent error, would be charged to the method."""
    env = CorridorEnv({"e1": 1.0})
    with pytest.raises(kind, match="invariant broke"):
        one_episode(env, ScriptedAgent(misbehave_at=1, misbehaviour=misbehaviour),
                    on_agent_error=ON_AGENT_ERROR_RECORD)
    assert env.steps == 1 and not env.stopped


def test_our_own_invariant_breaking_in_episode_info_stops_the_run_even_when_recording():
    """Record mode absorbs the method's diagnostics failing, never ours."""
    with pytest.raises(ObjNavInternalError, match="invariant broke"):
        one_episode(agent=InternalInfo(), on_agent_error=ON_AGENT_ERROR_RECORD)


def test_an_agent_reset_error_propagates_even_when_recording():
    """A category the agent cannot search for is a configuration error, not an episode failure."""
    with pytest.raises(UnknownCategoryError):
        one_episode(agent=ScriptedAgent(reset_error=UnknownCategoryError("no chair")),
                    on_agent_error=ON_AGENT_ERROR_RECORD)


def test_diagnostics_that_are_not_strict_json_are_refused():
    """A NaN in agent_info would make the row unwritable; better said here, naming the agent."""
    with pytest.raises(ResultsError, match=r"ScriptedAgent.episode_info\(\) is not strict JSON"):
        one_episode(agent=ScriptedAgent(info={"heading": float("nan")}))


@pytest.mark.parametrize("info,problem", [({"heading": float("nan")}, "is not strict JSON"),
                                          ([("heading", 1.0)], "must return a dict")],
                         ids=["nan", "not_a_mapping"])
def test_diagnostics_that_cannot_be_stored_are_recorded_beside_a_score_that_stands(info, problem):
    """Under record they are the agent's data failing our check, like a raising episode_info."""
    record = one_episode(agent=ScriptedAgent(info=info), on_agent_error=ON_AGENT_ERROR_RECORD)
    note = record.agent_info["episode_info_error"]
    assert note.startswith("ResultsError: ") and problem in note
    assert (record.success, record.termination, record.agent_error) == (True, TERMINATION_STOP, None)


@pytest.mark.parametrize("info,problem", [(RaisingMapping(), "RuntimeError: lazy diagnostic broke"),
                                          ({"tree": nested(100000)}, "RecursionError: ")],
                         ids=["raises_when_read", "too_deep_to_encode"])
def test_diagnostics_that_fail_as_they_are_stored_are_recorded_beside_a_score_that_stands(info, problem):
    """A report that raises as it is read is an episode_info that raised late; it used to abort the record-mode sweep."""
    record = one_episode(agent=ScriptedAgent(info=info), on_agent_error=ON_AGENT_ERROR_RECORD)
    assert record.agent_info["episode_info_error"].startswith(problem)
    assert (record.success, record.termination, record.agent_error) == (True, TERMINATION_STOP, None)


def test_a_record_mode_sweep_with_a_nan_in_one_episodes_diagnostics_finishes(tmp_path):
    """It used to abort at that episode on every resume, so the directory could never be summarised."""
    summary = run_benchmark(CorridorEnv(GOALS), NanInE2(), on_agent_error=ON_AGENT_ERROR_RECORD,
                            logger=MetricsLogger(tmp_path / "run", CONFIG))
    assert summary.overall.n_episodes == 3 and summary.agent_errors == 0
    assert (tmp_path / "run" / "summary.json").is_file()


def test_failing_diagnostics_are_raised_by_default_and_recorded_when_asked():
    """The episode itself is sound; under record the score stands and the failure is kept beside it."""
    with pytest.raises(ValueError, match="no diagnostics"):
        one_episode(agent=ScriptedAgent(info_error=True))
    record = one_episode(agent=ScriptedAgent(info_error=True), on_agent_error=ON_AGENT_ERROR_RECORD)
    assert record.agent_info == {"episode_info_error": "ValueError: no diagnostics"}
    assert (record.success, record.termination) == (True, TERMINATION_STOP)


def test_an_unknown_on_agent_error_mode_is_refused_before_the_episode_starts():
    """``"ignore"`` must not quietly behave like either real mode."""
    env = CorridorEnv({"e1": 1.0})
    with pytest.raises(HarnessError, match="on_agent_error"):
        one_episode(env, on_agent_error="ignore")
    assert env.resets == []
