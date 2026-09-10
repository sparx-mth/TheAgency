"""Drive an agent through an environment's episodes, score each one, and summarise the run.

The runner is the only code between an agent and a simulator, so it holds
both to their contracts at every step: the environment to the
:class:`ObjNavEnv` contract (``env_contract.py``) -- down to whether each
action moved the agent as the episode's action spec says (``kinematics.py``)
and whether each episode is of the run's benchmark split -- so an adapter's
bug raises :class:`EnvContractError` before it can become a number; the agent
to its own (``agent_contract.py``), so an action the benchmark cannot execute
never reaches the environment. What the runner itself saw -- every action it
sent, every pose it observed -- is the second opinion :func:`score_episode`
checks the environment's measurement against. Everything that can be checked
before the first episode is (``run_plan.py``).

An agent that crashes is different: with ``on_agent_error="record"`` the
episode ends on a forced STOP and is kept as an agent error -- a failure,
never credited with success or SPL even when the forced STOP lands on the
goal -- so one bad episode does not cost a night's sweep. An error in
``agent.reset`` is a configuration error and always propagates, as does every
error the environment raises and every invariant of our own code that breaks
(``agent_contract.INFRASTRUCTURE_ERRORS``).

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    EpisodeMeasurement,
)
from sparx_agency.core.planning.objnav.types.observation import (
    ObjNavObservation,
)
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import (
    ON_AGENT_ERROR_RAISE,
    agent_info,
    decide,
)
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.checks import is_real
from sparx_agency.tasks.planning.objnav_benchmark.env_contract import (
    check_run_identity,
    check_step,
    checked_measurement,
    checked_reset,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import (
    KinematicTolerance,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    ACTION_NAMES,
    TERMINATION_AGENT_ERROR,
    EpisodeRecord,
    EpisodeScore,
)
from sparx_agency.tasks.planning.objnav_benchmark.run_plan import (
    check_options,
    check_participants,
    requested_episode_ids,
    reusable_records,
)
from sparx_agency.tasks.planning.objnav_benchmark.scoring import (
    path_length_3d,
    score_episode,
)
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
)

_STOP = DiscreteAction.STOP


@dataclass(frozen=True)
class _Trace:
    """What the runner itself saw of one episode. Private: built only by :func:`_drive`.

    Attributes:
        positions: Base positions, one per observation, the reset one first.
        action_counts: Actions sent, by :class:`DiscreteAction` name.
        actions_sent: Actions sent, STOP included.
        last_action: The last action sent.
        agent_error: ``"Type: message"`` when the agent raised, else None.
    """

    positions: Tuple[Tuple[float, float, float], ...]
    action_counts: Dict[str, int]
    actions_sent: int
    last_action: DiscreteAction
    agent_error: Optional[str]


def _drive(env: ObjNavEnv, agent: ObjNavAgent, episode: ObjNavEpisode,
           observation: ObjNavObservation, mode: str,
           kinematics: Optional[KinematicTolerance]) -> _Trace:
    """Step the episode to its end, checking the environment after every action."""
    positions = [observation.pose.position()]
    counts = dict.fromkeys(ACTION_NAMES, 0)
    sent = 0
    action = _STOP
    agent_error = None
    # checked_reset refused an episode already over, so this runs at least
    # once; a forced STOP ends it, since check_step demands the episode be over.
    while not env.episode_over:
        action, agent_error = decide(agent, observation, episode.action_spec,
                                     mode)
        before = observation.pose
        observation = env.step(action)
        sent += 1
        counts[action.name] += 1
        check_step(env, episode, observation, action, sent, before=before,
                   kinematics=kinematics)
        positions.append(observation.pose.position())
    return _Trace(tuple(positions), counts, sent, action, agent_error)


def _native_metrics(measurement: EpisodeMeasurement) -> Dict[str, Optional[float]]:
    """The simulator's own values; an infinite distance is stored as None, as ``distance_to_goal_m`` is."""
    return {key: float(value) if math.isfinite(value) else None
            for key, value in measurement.native_metrics.items()}


def _record(episode: ObjNavEpisode, agent_name: str,
            measurement: EpisodeMeasurement, score: EpisodeScore,
            trace: _Trace, info: Dict[str, Any], wall_s: float) -> EpisodeRecord:
    """The episode as a results row."""
    errored = trace.agent_error is not None
    distance = score.distance_to_goal_m
    return EpisodeRecord(
        benchmark=episode.benchmark,
        split=episode.split,
        episode_id=episode.episode_id,
        scene_id=episode.scene_id,
        target_category=episode.target_category,
        agent=agent_name,
        # A crashed agent is never credited, even if its forced STOP landed on
        # the goal; its progress (SoftSPL) still stands as measured.
        success=score.success and not errored,
        spl=0.0 if errored else score.spl,
        soft_spl=score.soft_spl,
        distance_to_goal_m=None if math.isinf(distance) else distance,
        path_length_m=score.path_length_m,
        observed_path_length_m=score.observed_path_length_m,
        shortest_path_m=score.shortest_path_m,
        start_distance_to_goal_m=float(measurement.start_distance_to_goal_m),
        steps=trace.actions_sent,
        stop_called=measurement.stop_called,
        termination=(TERMINATION_AGENT_ERROR if errored
                     else measurement.termination),
        action_counts=dict(trace.action_counts),
        wall_s=float(wall_s),
        agent_error=trace.agent_error,
        native_metrics=_native_metrics(measurement),
        agent_info=info,
    )


def run_episode(env: ObjNavEnv, agent: ObjNavAgent, episode_id: str, *,
                on_agent_error: str = ON_AGENT_ERROR_RAISE,
                require_stop_for_success: bool = True,
                path_tolerance_m: Optional[float] = 1e-3,
                kinematics: Optional[KinematicTolerance] = KinematicTolerance(),
                expected: Optional[Tuple[str, str]] = None,
                clock: Callable[[], float] = time.monotonic) -> EpisodeRecord:
    """Run one episode from reset to measurement and return its scored record.

    Args:
        env: The environment; its episode is reset here.
        agent: The agent; reset here, then asked for one action per step.
        episode_id: One of ``env.episode_ids()``.
        on_agent_error: ``"raise"`` (``ON_AGENT_ERROR_RAISE``) to re-raise an
            exception from ``agent.act``, or the :class:`AgentContractError`
            of a decision the benchmark does not accept; ``"record"``
            (``ON_AGENT_ERROR_RECORD``) to end the episode on a forced STOP
            and record it as :data:`TERMINATION_AGENT_ERROR`: success False,
            SPL 0, SoftSPL as measured. Under ``"record"``, diagnostics that
            ``agent.episode_info`` fails to produce as a dict of strict JSON
            are stored as ``{"episode_info_error": ...}`` beside a score that
            stands. ``agent_contract.INFRASTRUCTURE_ERRORS`` are raised in
            either mode.
        require_stop_for_success: Passed to :func:`score_episode`.
        path_tolerance_m: Passed to :func:`score_episode`; the observed path
            is traced from every observation's pose, the reset one included.
        kinematics: How far each action's realised motion may stray from the
            episode's action spec. None disables the check -- only for a
            simulator whose motion has been verified otherwise.
        expected: The run's ``(benchmark, split)``: an episode of another is
            refused after its reset, before the agent sees it. None accepts
            any.
        clock: Seconds, for ``wall_s``; read once before the reset and once
            after the episode.

    Returns:
        The record. ``distance_to_goal_m`` is None when no goal is reachable
        from where the agent ended, and so is an infinite native distance.

    Raises:
        TypeError: If ``env``, ``agent`` or ``kinematics`` has the wrong type,
            or ``clock`` is not callable.
        HarnessError: On a malformed option or an agent without a name.
        AgentContractError: Under ``"raise"``, when the agent returns
            something that is not an ``AgentDecision``, or an action the
            benchmark does not accept.
        EnvContractError: When the environment breaks its contract.
        ScoringError: When the measurement fails a scoring check.
        ResultsError: Under ``"raise"``, if ``agent.episode_info()`` is not a
            strict-JSON dict.
    """
    check_options(on_agent_error, require_stop_for_success, path_tolerance_m,
                  kinematics, expected)
    check_participants(env, agent)
    if not callable(clock):
        raise TypeError("clock must be callable, got %r" % (clock,))
    started = clock()
    episode, observation = checked_reset(env, episode_id)
    check_run_identity(episode, expected)
    agent.reset(episode)  # a configuration error: always propagates
    trace = _drive(env, agent, episode, observation, on_agent_error,
                   kinematics)
    measurement = checked_measurement(env, episode, trace.actions_sent,
                                      trace.last_action)
    score = score_episode(measurement, path_length_3d(trace.positions),
                          require_stop_for_success=require_stop_for_success,
                          path_tolerance_m=path_tolerance_m)
    info = agent_info(agent, on_agent_error)
    return _record(episode, agent.name, measurement, score, trace, info,
                   clock() - started)


def run_benchmark(env: ObjNavEnv, agent: ObjNavAgent, *,
                  logger: Optional[MetricsLogger] = None,
                  episode_ids: Optional[Sequence[str]] = None,
                  on_agent_error: str = ON_AGENT_ERROR_RAISE,
                  require_stop_for_success: bool = True,
                  path_tolerance_m: Optional[float] = 1e-3,
                  kinematics: Optional[KinematicTolerance] = KinematicTolerance(),
                  progress: Optional[Callable[[int, int, EpisodeRecord], None]] = None,
                  confidence: float = 0.95) -> BenchmarkSummary:
    """Run every requested episode, log each as it ends, and summarise the run.

    Every argument is checked before the first episode -- the episode list
    against the environment, the options, and a resumed logger's episodes
    against this run -- so a typo costs seconds, not a night. A resumed
    logger's completed episodes are skipped and their logged records reused;
    it may hold no episode outside ``episode_ids``. Every episode run must be
    of the run's benchmark and split: a resumed logger's, or else the first
    episode's. The environment is not closed: the caller owns it.

    Args:
        env: The environment.
        agent: The agent.
        logger: Where each record is logged and the summary written; None to
            keep the run in memory only.
        episode_ids: The episodes to run, in order; defaults to
            ``env.episode_ids()``.
        on_agent_error: See :func:`run_episode`.
        require_stop_for_success: See :func:`run_episode`.
        path_tolerance_m: See :func:`run_episode`.
        kinematics: See :func:`run_episode`.
        progress: Called as ``progress(position, total, record)`` after each
            episode this call runs, ``position`` being its 1-based place in
            the requested order (a resumed run's first call may be 51 of 100).
        confidence: Coverage of the summary's intervals, in ``(0, 1)``.

    Returns:
        The summary of every requested episode, reused ones included.

    Raises:
        TypeError: On a wrong type for ``env``, ``agent``, ``kinematics``,
            ``logger`` or ``progress``.
        HarnessError: On a malformed option, an empty, repeated or unknown
            episode id, or a confidence outside ``(0, 1)``.
        ResultsError: If the resumed logger holds episodes this run does not
            request, or another agent's.
        EnvContractError: If the environment serves an episode id twice, or
            an episode of another benchmark or split than the run's; see
            :func:`run_episode` for everything else it raises.
    """
    check_options(on_agent_error, require_stop_for_success, path_tolerance_m,
                  kinematics)
    # stats.py checks this too, but only once every episode has run.
    if not (is_real(confidence) and 0.0 < confidence < 1.0):
        raise HarnessError("confidence must lie strictly between 0 and 1 (0.95 "
                           "for 95%% intervals), got %r" % (confidence,))
    check_participants(env, agent)
    if logger is not None and not isinstance(logger, MetricsLogger):
        raise TypeError("logger must be a MetricsLogger or None, got %s"
                        % type(logger).__name__)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None, got %r"
                        % (progress,))
    ids = requested_episode_ids(env, episode_ids)
    reusable = reusable_records(logger, ids, agent.name)
    # A resumed run's split is the logged one; a fresh run's, its first
    # episode's.
    expected = next(((r.benchmark, r.split) for r in reusable.values()), None)
    records: List[EpisodeRecord] = []
    for position, episode_id in enumerate(ids, start=1):
        record = reusable.get(episode_id)
        if record is None:
            record = run_episode(
                env, agent, episode_id, on_agent_error=on_agent_error,
                require_stop_for_success=require_stop_for_success,
                path_tolerance_m=path_tolerance_m, kinematics=kinematics,
                expected=expected)
            if logger is not None:
                logger.log(record)
            if progress is not None:
                progress(position, len(ids), record)
        if expected is None:
            expected = (record.benchmark, record.split)
        records.append(record)
    summary = summarise(records, confidence=confidence)
    if logger is not None:
        logger.finish(summary)
    return summary
