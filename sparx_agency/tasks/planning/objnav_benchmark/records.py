"""One episode as the harness scores and stores it: its score, and its results row.

Rows carry their units in their names and are written as JSON lines, one per
episode, appended as the run goes -- a crash loses one episode, not the sweep.
:meth:`EpisodeRecord.from_row` refuses a row from another schema version
instead of guessing what an old field meant. A record refuses fields that
contradict each other (an SPL on a failure, action counts that do not add up
to the steps), because a row like that averages into a plausible number.

Scores are fractions in ``[0, 1]``; the presentation layer alone prints
percent. The run-level types built from these rows are in ``summaries.py``.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

from sparx_agency.core.planning.objnav.types import (
    NATIVE_KEYS,
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    DiscreteAction,
)
from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_count,
    is_fraction,
    is_length,
    is_name,
    is_real,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import ResultsError

#: The episode-row format. Bump it when a field changes meaning.
RECORD_SCHEMA = "objnav_episode/1"

#: The runner ended the episode because the agent raised and the run was told
#: to record rather than stop. Harness-only: an environment never reports it.
TERMINATION_AGENT_ERROR = "agent_error"

#: Every termination a record can carry.
RECORD_TERMINATIONS = (TERMINATION_STOP, TERMINATION_STEP_LIMIT,
                       TERMINATION_AGENT_ERROR)

#: One action-count key per action, by name.
ACTION_NAMES = tuple(action.name for action in DiscreteAction)


@dataclass(frozen=True)
class EpisodeScore:
    """The standard metrics of one episode, as the harness scored them.

    Attributes:
        success: The benchmark's success judgement, after the protocol checks.
        spl: ``S * l / max(l, p)``.
        soft_spl: ``max(0, 1 - dT / d0) * l / max(l, p)`` -- Habitat's
            definition, gated on neither success nor STOP.
        distance_to_goal_m: ``dT``; ``inf`` when no goal is reachable from
            where the agent ended.
        path_length_m: ``p`` as the environment accounts it, which SPL uses.
        shortest_path_m: ``l``.
        observed_path_length_m: ``p`` recomputed from the observed poses, or
            None when the caller had none.
        native_checked: The ``NATIVE_*`` keys the environment reported that
            were cross-checked and agreed.

    Raises:
        ResultsError: On a score outside ``[0, 1]``, an SPL on a failure, or a
            negative or non-finite length.
    """

    success: bool
    spl: float
    soft_spl: float
    distance_to_goal_m: float
    path_length_m: float
    shortest_path_m: float
    observed_path_length_m: Optional[float] = None
    native_checked: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ResultsError(
                "EpisodeScore.success must be a bool, got %r" % (self.success,))
        for name in ("spl", "soft_spl"):
            if not is_fraction(getattr(self, name)):
                raise ResultsError(
                    "EpisodeScore.%s must lie in [0, 1], got %r"
                    % (name, getattr(self, name)))
        if self.spl > 0.0 and not self.success:
            raise ResultsError(
                "EpisodeScore.spl=%r on a failed episode; SPL is zero unless "
                "the episode succeeded" % (self.spl,))
        for name in ("path_length_m", "shortest_path_m"):
            if not is_length(getattr(self, name)):
                raise ResultsError(
                    "EpisodeScore.%s must be finite and non-negative, got %r"
                    % (name, getattr(self, name)))
        dtg = self.distance_to_goal_m
        if not (is_real(dtg) and not math.isnan(dtg) and dtg >= 0.0):
            raise ResultsError(
                "EpisodeScore.distance_to_goal_m must be non-negative (inf "
                "allowed), got %r" % (dtg,))
        if (self.observed_path_length_m is not None
                and not is_length(self.observed_path_length_m)):
            raise ResultsError(
                "EpisodeScore.observed_path_length_m must be None or finite "
                "and non-negative, got %r" % (self.observed_path_length_m,))
        object.__setattr__(self, "native_checked", tuple(self.native_checked))


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode as a results row: who, what, how it scored, how it behaved.

    Attributes:
        benchmark: Benchmark key (``"hm3d_v2"``).
        split: Dataset split.
        episode_id: Unique within the run.
        scene_id: The scene.
        target_category: The goal category, in the dataset's vocabulary.
        agent: The agent's name.
        success: Scored success. False for every agent-error episode.
        spl: Scored SPL. 0 for every agent-error episode.
        soft_spl: Scored SoftSPL.
        distance_to_goal_m: ``dT``, or None when no goal is reachable from
            where the agent ended (``inf`` does not survive strict JSON).
        path_length_m: ``p``, the environment's accounting.
        observed_path_length_m: ``p`` recomputed by the runner from the
            observed poses.
        shortest_path_m: ``l``.
        start_distance_to_goal_m: ``d0``.
        steps: Actions executed, STOP included.
        stop_called: Whether the episode ended on STOP.
        termination: One of :data:`RECORD_TERMINATIONS`.
        action_counts: Actions executed, by :class:`DiscreteAction` name; every
            name present, summing to ``steps``.
        wall_s: Wall-clock seconds the episode took.
        agent_error: ``"ExceptionType: message"`` when the agent raised, else
            None.
        native_metrics: The environment's own metric values, as reported,
            under the ``NATIVE_*`` keys. None stands for ``inf`` -- a native
            distance to goal from where no goal is reachable -- because strict
            JSON cannot hold it, the same convention as
            ``distance_to_goal_m``.
        agent_info: The agent's per-episode diagnostics.
        schema: :data:`RECORD_SCHEMA`.

    Raises:
        ResultsError: When any field is malformed or two fields contradict
            each other.
    """

    benchmark: str
    split: str
    episode_id: str
    scene_id: str
    target_category: str
    agent: str
    success: bool
    spl: float
    soft_spl: float
    distance_to_goal_m: Optional[float]
    path_length_m: float
    observed_path_length_m: float
    shortest_path_m: float
    start_distance_to_goal_m: float
    steps: int
    stop_called: bool
    termination: str
    action_counts: Dict[str, int]
    wall_s: float
    agent_error: Optional[str] = None
    native_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    agent_info: Dict[str, Any] = field(default_factory=dict)
    schema: str = RECORD_SCHEMA

    def __post_init__(self) -> None:
        for name in ("benchmark", "split", "episode_id", "scene_id",
                     "target_category", "agent"):
            value = getattr(self, name)
            if not is_name(value):
                raise ResultsError(
                    "EpisodeRecord.%s must be a non-blank string, got %r"
                    % (name, value))
        if self.schema != RECORD_SCHEMA:
            raise ResultsError(
                "EpisodeRecord.schema must be %r, got %r"
                % (RECORD_SCHEMA, self.schema))
        self._check_scores()
        self._check_lengths()
        self._check_termination()
        self._check_counts()
        self._check_native()

    def _check_scores(self) -> None:
        for name in ("success", "stop_called"):
            if not isinstance(getattr(self, name), bool):
                raise ResultsError(
                    "EpisodeRecord.%s must be a bool, got %r"
                    % (name, getattr(self, name)))
        for name in ("spl", "soft_spl"):
            if not is_fraction(getattr(self, name)):
                raise ResultsError(
                    "EpisodeRecord.%s must lie in [0, 1], got %r"
                    % (name, getattr(self, name)))
        if self.spl > 0.0 and not self.success:
            raise ResultsError(
                "EpisodeRecord %r has spl=%r but did not succeed"
                % (self.episode_id, self.spl))

    def _check_lengths(self) -> None:
        for name in ("path_length_m", "observed_path_length_m",
                     "shortest_path_m", "start_distance_to_goal_m", "wall_s"):
            if not is_length(getattr(self, name)):
                raise ResultsError(
                    "EpisodeRecord.%s must be finite and non-negative, got %r"
                    % (name, getattr(self, name)))
        if (self.distance_to_goal_m is not None
                and not is_length(self.distance_to_goal_m)):
            raise ResultsError(
                "EpisodeRecord.distance_to_goal_m must be None or finite and "
                "non-negative, got %r" % (self.distance_to_goal_m,))

    def _check_termination(self) -> None:
        if self.termination not in RECORD_TERMINATIONS:
            raise ResultsError(
                "EpisodeRecord.termination must be one of %r, got %r"
                % (RECORD_TERMINATIONS, self.termination))
        errored = self.termination == TERMINATION_AGENT_ERROR
        if errored != (self.agent_error is not None):
            raise ResultsError(
                "EpisodeRecord %r: agent_error must be set exactly when the "
                "termination is %r" % (self.episode_id, TERMINATION_AGENT_ERROR))
        if errored and (self.success or self.spl > 0.0):
            raise ResultsError(
                "EpisodeRecord %r: an agent-error episode cannot be credited "
                "with success or SPL" % (self.episode_id,))
        expected_stop = self.termination != TERMINATION_STEP_LIMIT
        if self.stop_called != expected_stop:
            raise ResultsError(
                "EpisodeRecord %r: stop_called=%r contradicts termination=%r"
                % (self.episode_id, self.stop_called, self.termination))

    def _check_counts(self) -> None:
        if not is_count(self.steps):
            raise ResultsError(
                "EpisodeRecord.steps must be a non-negative integer, got %r"
                % (self.steps,))
        counts = self.action_counts
        if not isinstance(counts, dict) or set(counts) != set(ACTION_NAMES):
            raise ResultsError(
                "EpisodeRecord.action_counts must have exactly the keys %r, "
                "got %r" % (ACTION_NAMES, counts))
        if not all(is_count(value) for value in counts.values()):
            raise ResultsError(
                "EpisodeRecord.action_counts must hold non-negative integers, "
                "got %r" % (counts,))
        if sum(counts.values()) != self.steps:
            raise ResultsError(
                "EpisodeRecord %r: action counts sum to %d but steps=%d"
                % (self.episode_id, sum(counts.values()), self.steps))

    def _check_native(self) -> None:
        if not isinstance(self.native_metrics, dict):
            raise ResultsError(
                "EpisodeRecord.native_metrics must be a dict, got %r"
                % (self.native_metrics,))
        for key, value in self.native_metrics.items():
            if key not in NATIVE_KEYS:
                raise ResultsError(
                    "EpisodeRecord %r: native_metrics has unknown key %r"
                    % (self.episode_id, key))
            if value is not None and not (is_real(value)
                                          and math.isfinite(value)):
                raise ResultsError(
                    "EpisodeRecord %r: native_metrics[%r] must be None (inf) "
                    "or a finite number, got %r"
                    % (self.episode_id, key, value))

    def to_row(self) -> Dict[str, Any]:
        """The record as a plain dict, ready for ``json.dumps(allow_nan=False)``."""
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> EpisodeRecord:
        """Rebuild a record from :meth:`to_row` output, refusing anything else.

        Raises:
            ResultsError: On another schema version, an unknown or missing
                field, or a row that fails validation.
        """
        if not isinstance(row, dict):
            raise ResultsError("an episode row must be a dict, got %r" % (row,))
        if row.get("schema") != RECORD_SCHEMA:
            raise ResultsError(
                "episode row has schema %r, expected %r: it was written by "
                "another version of the harness"
                % (row.get("schema"), RECORD_SCHEMA))
        names = [f.name for f in fields(cls)]
        required = [f.name for f in fields(cls)
                    if f.default is MISSING and f.default_factory is MISSING]
        unknown = sorted(set(row) - set(names))
        missing = sorted(set(required) - set(row))
        if unknown or missing:
            raise ResultsError(
                "episode row %r has unknown fields %r and lacks %r"
                % (row.get("episode_id"), unknown, missing))
        return cls(**row)
