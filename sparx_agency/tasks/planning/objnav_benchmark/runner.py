"""Drive an agent through an environment's episodes, score each one, and summarise the run.

The runner holds environment and agent contracts at every step. The simulator
is independently checked against every observed pose and executed action.
With on_agent_error='record', ordinary agent failures force STOP and score as
failures; infrastructure errors always propagate. Python 3.8, stdlib only.
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
from sparx_agency.core.planning.objnav.types.measurement import EpisodeMeasurement
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import (
    ON_AGENT_ERROR_RAISE, agent_info, decide)
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.checks import is_real
from sparx_agency.tasks.planning.objnav_benchmark.env_contract import (
    check_run_identity, check_step, checked_measurement, checked_reset)
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import KinematicTolerance
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    ACTION_NAMES, TERMINATION_AGENT_ERROR, EpisodeRecord, EpisodeScore)
from sparx_agency.tasks.planning.objnav_benchmark.run_plan import (
    check_options, check_participants, requested_episode_ids, reusable_records)
from sparx_agency.tasks.planning.objnav_benchmark.scoring import path_length_3d, score_episode
from sparx_agency.tasks.planning.objnav_benchmark.summaries import BenchmarkSummary

_STOP = DiscreteAction.STOP


@dataclass(frozen=True)
class _Trace:
    """What the runner saw, including the reset position and every action."""

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
    # checked_reset refused an episode already over. Forced STOP ends it.
    while not env.episode_over:
        action, agent_error = decide(agent, observation, episode.action_spec, mode)
        before = observation.pose
        observation = env.step(action)
        sent += 1
        counts[action.name] += 1
        check_step(env, episode, observation, action, sent, before=before,
                   kinematics=kinematics)
        positions.append(observation.pose.position())
    return _Trace(tuple(positions), counts, sent, action, agent_error)


def _native_metrics(measurement: EpisodeMeasurement) -> Dict[str, Optional[float]]:
    """Simulator values; an infinite distance is stored as None."""
    return {key: float(value) if math.isfinite(value) else None
            for key, value in measurement.native_metrics.items()}


def _record(episode: ObjNavEpisode, agent_name: str,
            measurement: EpisodeMeasurement, score: EpisodeScore,
            trace: _Trace, info: Dict[str, Any], wall_s: float) -> EpisodeRecord:
    """The episode as a results row; never credit a crashed agent with success."""
    errored = trace.agent_error is not None
    distance = score.distance_to_goal_m
    return EpisodeRecord(
        benchmark=episode.benchmark,
        split=episode.split,
        episode_id=episode.episode_id,
        scene_id=episode.scene_id,
        target_category=episode.target_category,
        agent=agent_name,
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
        termination=(TERMINATION_AGENT_ERROR if errored else measurement.termination),
        action_counts=dict(trace.action_counts),
        wall_s=float(wall_s),
        agent_error=trace.agent_error,
        native_metrics=_native_metrics(measurement),
        agent_info=info,
    )


def _check_path_protocol(dimension: str, epsilon: float) -> None:
    """Refuse ambiguous path accounting before starting any episode."""
    if dimension not in ("3d", "planar"):
        raise HarnessError("path_length_dimension must be '3d' or 'planar'")
    if not (is_real(epsilon) and math.isfinite(epsilon) and epsilon >= 0):
        raise HarnessError("path_length_epsilon_m must be finite and non-negative")


def run_episode(env: ObjNavEnv, agent: ObjNavAgent, episode_id: str, *,
                on_agent_error: str = ON_AGENT_ERROR_RAISE,
                require_stop_for_success: bool = True,
                path_tolerance_m: Optional[float] = 1e-3,
                kinematics: Optional[KinematicTolerance] = KinematicTolerance(),
                expected: Optional[Tuple[str, str]] = None,
                clock: Callable[[], float] = time.monotonic,
                path_length_dimension: str = "3d",
                path_length_epsilon_m: float = 0.0) -> EpisodeRecord:
    """Run one episode from reset to measurement and return its scored record.

    Args:
        env: Environment, reset here. The caller owns closing it.
        agent: Agent, reset here, then asked for one action per observation.
        episode_id: One of env.episode_ids().
        on_agent_error: 'raise' or 'record'. Under 'record', ordinary agent
            errors force STOP and score as failures. Infrastructure errors
            and errors in agent.reset always propagate. Unserializable agent
            diagnostics become episode_info_error under 'record'.
        require_stop_for_success: Passed to score_episode.
        path_tolerance_m: Independent observed-path cross-check tolerance.
        kinematics: Realised-action tolerance; None disables only motion checks.
        expected: Required (benchmark, split), or None for any identity.
        clock: Seconds for wall_s, read before reset and after the episode.
        path_length_dimension: '3d' by default; 'planar' only when required by
            the adapter's declared protocol. Record this in the run config.
        path_length_epsilon_m: Explicit initial accumulator, 0 by default.
            Record a protocol-specific override in the run config too.

    Returns:
        A scored record. Infinite final/native distances are stored as None.

    Raises:
        TypeError: Wrong env/agent/kinematics/clock type.
        HarnessError: Invalid options or a missing agent name.
        AgentContractError: Invalid decision under on_agent_error='raise'.
        EnvContractError: Environment violates reset, step or measurement rules.
        ScoringError: Measurement disagrees with scoring cross-checks.
        ResultsError: Invalid diagnostics under on_agent_error='raise'.
    """
    check_options(on_agent_error, require_stop_for_success, path_tolerance_m,
                  kinematics, expected)
    _check_path_protocol(path_length_dimension, path_length_epsilon_m)
    check_participants(env, agent)
    if not callable(clock):
        raise TypeError("clock must be callable, got %r" % (clock,))
    started = clock()
    episode, observation = checked_reset(env, episode_id)
    check_run_identity(episode, expected)
    agent.reset(episode)  # a configuration error: always propagates
    trace = _drive(env, agent, episode, observation, on_agent_error, kinematics)
    measurement = checked_measurement(env, episode, trace.actions_sent, trace.last_action)
    positions = trace.positions
    if path_length_dimension == "planar":
        positions = tuple((x, y, 0.0) for x, y, _ in positions)
    observed_length = path_length_3d(positions) + path_length_epsilon_m
    score = score_episode(measurement, observed_length,
                          require_stop_for_success=require_stop_for_success,
                          path_tolerance_m=path_tolerance_m)
    info = agent_info(agent, on_agent_error)
    return _record(episode, agent.name, measurement, score, trace, info, clock() - started)


def run_benchmark(env: ObjNavEnv, agent: ObjNavAgent, *,
                  logger: Optional[MetricsLogger] = None,
                  episode_ids: Optional[Sequence[str]] = None,
                  on_agent_error: str = ON_AGENT_ERROR_RAISE,
                  require_stop_for_success: bool = True,
                  path_tolerance_m: Optional[float] = 1e-3,
                  kinematics: Optional[KinematicTolerance] = KinematicTolerance(),
                  progress: Optional[Callable[[int, int, EpisodeRecord], None]] = None,
                  confidence: float = 0.95,
                  path_length_dimension: str = "3d",
                  path_length_epsilon_m: float = 0.0) -> BenchmarkSummary:
    """Run requested episodes, log each as it ends, and summarise the run.

    Options, episode identities and resume compatibility are checked before
    running anything. Completed records are reused, not repeated. An unknown,
    duplicate or empty episode id is refused. A resumed logger may hold no
    episode outside the requested selection, or from another agent/split.

    Args:
        env: Environment; the caller owns closing it.
        agent: Agent under test.
        logger: MetricsLogger or None to keep results in memory only.
        episode_ids: Requested episodes in order; defaults to all served ids.
        on_agent_error: See run_episode.
        require_stop_for_success: See run_episode.
        path_tolerance_m: See run_episode.
        kinematics: See run_episode.
        progress: Called as (position, total, record) after a new episode.
        confidence: Confidence-interval coverage, strictly between 0 and 1.
        path_length_dimension: See run_episode; defaults to 3-D travel.
        path_length_epsilon_m: See run_episode; defaults to zero.

    Returns:
        Summary of every requested episode, including reused records.

    Raises:
        TypeError: Wrong participant, logger, progress or kinematics type.
        HarnessError: Invalid option or episode selection.
        ResultsError: Incompatible resumed records or corrupt results.
        EnvContractError: An environment contract was broken.
    """
    check_options(on_agent_error, require_stop_for_success, path_tolerance_m, kinematics)
    _check_path_protocol(path_length_dimension, path_length_epsilon_m)
    if not (is_real(confidence) and 0.0 < confidence < 1.0):
        raise HarnessError("confidence must lie strictly between 0 and 1 (0.95 "
                           "for 95%% intervals), got %r" % (confidence,))
    check_participants(env, agent)
    if logger is not None and not isinstance(logger, MetricsLogger):
        raise TypeError("logger must be a MetricsLogger or None, got %r" % type(logger).__name__)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None, got %r" % (progress,))
    ids = requested_episode_ids(env, episode_ids)
    reusable = reusable_records(logger, ids, agent.name)
    expected = next(((r.benchmark, r.split) for r in reusable.values()), None)
    records: List[EpisodeRecord] = []
    for position, episode_id in enumerate(ids, start=1):
        record = reusable.get(episode_id)
        if record is None:
            record = run_episode(
                env, agent, episode_id, on_agent_error=on_agent_error,
                require_stop_for_success=require_stop_for_success,
                path_tolerance_m=path_tolerance_m, kinematics=kinematics,
                expected=expected, path_length_dimension=path_length_dimension,
                path_length_epsilon_m=path_length_epsilon_m)
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

