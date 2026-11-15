"""The ground truth an environment reports once an episode is over.

Privileged: this reaches the benchmark harness, never the agent.

It carries **quantities, not scores**. SPL and SoftSPL are computed from these
by one scoring function in the harness, for every benchmark, so a Habitat
number and a RoboTHOR number are produced by the same arithmetic. What each
simulator computes for itself goes into ``native_metrics`` under the canonical
keys below, and the harness cross-checks the two: a disagreement is an adapter
bug (a frame conversion, a unit, a distance measured to the wrong goal), and it
is far cheaper to find as a failed check than as a suspicious SPL.

The definitions, for the standard metrics (Anderson et al. 2018, and the
habitat-lab implementation):

* ``shortest_path_m`` (``l``): the geodesic length of the shortest path from
  the start to the nearest goal, *as the benchmark defines the goal* -- Habitat
  measures to the nearest goal view point, not the object's centre.
* ``start_distance_to_goal_m`` (``d0``) and ``final_distance_to_goal_m``
  (``dT``): the benchmark's distance-to-goal at the first and last step.
  Habitat's ``d0`` equals its ``l``.
* ``path_length_m`` (``p``): what the agent actually travelled -- the sum of
  3-D Euclidean displacements between successive base positions, starting at
  the reset position. Turns, tilts and STOP add nothing; a MOVE_FORWARD that
  hit a wall adds only what it really moved.

Whether a success may be granted without STOP is the benchmark's protocol, so
it is checked by the harness's scoring and not here: Habitat and RoboTHOR
require STOP; SemExp's Gibson evaluator does not.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict

from sparx_agency.core.planning.objnav.errors import EnvContractError

#: The agent called STOP.
TERMINATION_STOP = "stop"
#: The step budget ran out before the agent called STOP.
TERMINATION_STEP_LIMIT = "step_limit"
#: Every way an environment can end an episode.
TERMINATIONS = (TERMINATION_STOP, TERMINATION_STEP_LIMIT)

#: Canonical keys for a simulator's own metric values. An adapter renames its
#: simulator's keys to these (Habitat's ``softspl`` becomes ``soft_spl``).
NATIVE_SUCCESS = "success"
NATIVE_SPL = "spl"
NATIVE_SOFT_SPL = "soft_spl"
NATIVE_DISTANCE_TO_GOAL = "distance_to_goal"
NATIVE_KEYS = (NATIVE_SUCCESS, NATIVE_SPL, NATIVE_SOFT_SPL,
               NATIVE_DISTANCE_TO_GOAL)


def _real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class EpisodeMeasurement:
    """What happened in one episode, measured by the environment.

    Attributes:
        success: The benchmark's own success judgement.
        stop_called: Whether the episode ended on STOP.
        termination: :data:`TERMINATION_STOP` or
            :data:`TERMINATION_STEP_LIMIT`.
        steps: Actions executed, STOP included.
        shortest_path_m: ``l``, the SPL reference, metres.
        start_distance_to_goal_m: ``d0``, the SoftSPL reference, metres.
        final_distance_to_goal_m: ``dT``, the distance-to-goal metric,
            metres. ``inf`` only when the agent ended where no goal is
            reachable.
        path_length_m: ``p``, metres travelled.
        native_metrics: The simulator's own values under :data:`NATIVE_KEYS`,
            for cross-checking. Only values the simulator computes from the
            *same* definitions belong here: a deliberately different
            evaluator's number (allenact's RoboTHOR SPL measures ``p`` in the
            horizontal plane and skips the first step) goes into ``info``, or
            every episode would fail the check. An unknown key is refused, so a
            typo in an adapter cannot silently skip the check.
        info: Free-form extras (goal object ids, the simulator's raw metric
            dict). Never interpreted.

    ``native_metrics`` and ``info`` must be mappings; each is stored as a
    dict copy.

    Raises:
        EnvContractError: On a negative or non-finite quantity, an unknown
            termination, a STOP flag that disagrees with the termination, a
            ``native_metrics`` or ``info`` that is not a mapping, or an
            unknown native key.
    """

    success: bool
    stop_called: bool
    termination: str
    steps: int
    shortest_path_m: float
    start_distance_to_goal_m: float
    final_distance_to_goal_m: float
    path_length_m: float
    native_metrics: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("success", "stop_called"):
            if not isinstance(getattr(self, name), bool):
                raise EnvContractError(
                    "EpisodeMeasurement.%s must be a bool, got %r"
                    % (name, getattr(self, name)))
        if self.termination not in TERMINATIONS:
            raise EnvContractError(
                "EpisodeMeasurement.termination must be one of %r, got %r"
                % (TERMINATIONS, self.termination))
        if self.stop_called != (self.termination == TERMINATION_STOP):
            raise EnvContractError(
                "EpisodeMeasurement.stop_called=%r contradicts termination=%r"
                % (self.stop_called, self.termination))
        if (not isinstance(self.steps, numbers.Integral)
                or isinstance(self.steps, bool) or self.steps < 0):
            raise EnvContractError(
                "EpisodeMeasurement.steps must be a non-negative integer, got "
                "%r" % (self.steps,))
        for name in ("shortest_path_m", "start_distance_to_goal_m",
                     "path_length_m"):
            value = getattr(self, name)
            if not (_real(value) and math.isfinite(value) and value >= 0):
                raise EnvContractError(
                    "EpisodeMeasurement.%s must be finite and non-negative, "
                    "got %r" % (name, value))
        final = self.final_distance_to_goal_m
        if not (_real(final) and not math.isnan(final) and final >= 0):
            raise EnvContractError(
                "EpisodeMeasurement.final_distance_to_goal_m must be "
                "non-negative (inf allowed), got %r" % (final,))
        for name in ("native_metrics", "info"):
            value = getattr(self, name)
            if not isinstance(value, collections.abc.Mapping):
                raise EnvContractError(
                    "EpisodeMeasurement.%s must be a mapping ({} when the "
                    "simulator reports nothing), got %r" % (name, value))
            object.__setattr__(self, name, dict(value))
        for key, value in self.native_metrics.items():
            if key not in NATIVE_KEYS:
                raise EnvContractError(
                    "EpisodeMeasurement.native_metrics has unknown key %r; "
                    "rename the simulator's key to one of %r"
                    % (key, NATIVE_KEYS))
            if not _real(value) or math.isnan(value):
                raise EnvContractError(
                    "EpisodeMeasurement.native_metrics[%r] must be a number, "
                    "got %r" % (key, value))
