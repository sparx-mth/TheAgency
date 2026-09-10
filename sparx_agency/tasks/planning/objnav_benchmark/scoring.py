"""Score one episode the same way for every benchmark: SPL, SoftSPL, and the checks that keep them honest.

One arithmetic for every benchmark is the point. Environments report
*quantities* (:class:`EpisodeMeasurement`); this module turns them into
scores, so a Habitat figure and a RoboTHOR figure come out of the same lines.
The definitions are Anderson et al. 2018 ("On Evaluation of Embodied
Navigation Agents") as habitat-lab implements them in
``habitat/tasks/nav/nav.py`` (classes ``SPL`` and ``SoftSPL``):

* ``SPL = S * l / max(l, p)`` -- ``S`` success, ``l`` the geodesic distance
  from the start to the nearest goal, ``p`` the path travelled: 3-D chords
  between successive base positions from the reset position on, with turns,
  tilts and STOP adding nothing.
* ``SoftSPL = max(0, 1 - dT / d0) * l / max(l, p)`` -- progress toward the
  goal in place of success, gated on neither success nor STOP.

The official code leaves three corners undefined, and a long run meets all of
them: habitat-lab divides by zero when ``l == 0`` and produces NaN when the goal
is unreachable, while RoboTHOR's evaluators take the ratio as 1 when ``l`` and
``p`` are both 0. The conventions here, each applied in exactly one place: the
ratio ``l / max(l, p)`` is 1 when ``max(l, p) == 0``; the soft success is 1 when
``d0 == dT == 0`` and 0 when ``d0 == 0 < dT``; ``dT == inf`` gives soft success 0.

:func:`score_episode` refuses what would otherwise move SPL without failing
anything: a success without STOP where the protocol requires STOP, a path
length the environment accounts differently from the poses the runner saw,
and a simulator's native metric that the recomputation contradicts (the last
two are the cross-checks in ``cross_checks.py``).

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional, Sequence, Tuple

from sparx_agency.core.planning.objnav.types.measurement import (
    EpisodeMeasurement,
)
from sparx_agency.tasks.planning.objnav_benchmark.checks import (
    is_length,
    is_real,
)
from sparx_agency.tasks.planning.objnav_benchmark.cross_checks import (
    check_native_metrics,
    check_path_length,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ScoringError,
)
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeScore


def _metres(name: str, value) -> float:
    """``value`` as a float, once it is known to be a finite non-negative length."""
    if not is_length(value):
        raise ScoringError(
            "%s must be a finite non-negative number of metres, got %r"
            % (name, value))
    return float(value)


def _efficiency(shortest_path_m: float, path_length_m: float) -> float:
    """``l / max(l, p)``, taken as 1 when both are 0 (started on the goal, never moved)."""
    longest = max(shortest_path_m, path_length_m)
    if longest == 0.0:
        return 1.0
    return shortest_path_m / longest


def _soft_success(start_distance_m: float, final_distance_m: float) -> float:
    """``max(0, 1 - dT / d0)``, with this module's conventions for ``d0 == 0`` and ``dT == inf``."""
    if math.isinf(final_distance_m):
        return 0.0
    if start_distance_m == 0.0:
        return 1.0 if final_distance_m == 0.0 else 0.0
    return max(0.0, 1.0 - final_distance_m / start_distance_m)


def spl_score(success: bool, shortest_path_m: float,
              path_length_m: float) -> float:
    """Success weighted by path length for one episode: ``S * l / max(l, p)``.

    habitat-lab's ``SPL.update_metric`` in ``habitat/tasks/nav/nav.py``. A path
    shorter than ``l`` -- impossible against a true geodesic, but a coarse
    grid can overestimate ``l`` -- caps the ratio at 1. When ``l == p == 0``
    the ratio is taken as 1 where habitat-lab raises ``ZeroDivisionError``;
    when ``l == 0 < p`` the formula itself gives 0: the agent wandered off a
    goal it already stood on.

    Args:
        success: The episode's success judgement, as a real ``bool`` (a string
            such as ``"false"`` would be truthy).
        shortest_path_m: ``l``, metres.
        path_length_m: ``p``, metres.

    Returns:
        SPL in ``[0, 1]``; 0 on a failure.

    Raises:
        ScoringError: If ``success`` is not a bool, or a length is negative
            or not finite.
    """
    if not isinstance(success, bool):
        raise ScoringError(
            "spl_score: success must be a bool, got %r" % (success,))
    shortest = _metres("shortest_path_m", shortest_path_m)
    travelled = _metres("path_length_m", path_length_m)
    if not success:
        return 0.0
    return _efficiency(shortest, travelled)


def soft_spl_score(start_distance_m: float, final_distance_m: float,
                   shortest_path_m: float, path_length_m: float) -> float:
    """Habitat's SoftSPL for one episode: ``max(0, 1 - dT / d0) * l / max(l, p)``.

    habitat-lab's ``SoftSPL.update_metric`` in ``habitat/tasks/nav/nav.py``,
    gated on neither success nor STOP. habitat-lab puts ``d0`` in the
    efficiency term as well; there ``d0 == l``, and using ``l`` makes the term
    identical to SPL's on a benchmark where the two differ.

    Where habitat-lab divides by zero or produces NaN, the conventions are:
    ``d0 == 0`` gives soft success 1 if ``dT == 0`` and 0 otherwise; ``dT ==
    inf`` (the agent ended where no goal is reachable) gives soft success 0;
    the efficiency ratio is 1 when ``max(l, p) == 0``.

    Args:
        start_distance_m: ``d0``, the distance to goal at the start, metres.
        final_distance_m: ``dT``, the distance to goal at the end, metres;
            ``inf`` when no goal is reachable from there.
        shortest_path_m: ``l``, metres.
        path_length_m: ``p``, metres.

    Returns:
        SoftSPL in ``[0, 1]``.

    Raises:
        ScoringError: If a value is negative or NaN, or any value but ``dT``
            is infinite.
    """
    start = _metres("start_distance_m", start_distance_m)
    final = final_distance_m
    if not (is_real(final) and not math.isnan(final) and final >= 0.0):
        raise ScoringError(
            "final_distance_m must be non-negative (inf when no goal is "
            "reachable), got %r" % (final,))
    shortest = _metres("shortest_path_m", shortest_path_m)
    travelled = _metres("path_length_m", path_length_m)
    return _soft_success(start, float(final)) * _efficiency(shortest, travelled)


def _position(index: int, value) -> Tuple[float, float, float]:
    """One ``(x, y, z)`` as floats, refusing anything else."""
    coords = tuple(value) if isinstance(value, Iterable) else ()
    if len(coords) != 3 or not all(
            is_real(c) and math.isfinite(c) for c in coords):
        raise ScoringError(
            "path_length_3d: position %d must be three finite numbers "
            "(x, y, z), got %r" % (index, value))
    return (float(coords[0]), float(coords[1]), float(coords[2]))


def path_length_3d(positions: Sequence[Tuple[float, float, float]]) -> float:
    """``p``: the sum of 3-D straight chords between successive base positions.

    Pass every position the agent occupied, *starting with the reset
    position*: habitat-lab's SPL accumulates from the pose at reset, so a list
    that starts after the first action loses that action's displacement.
    Repeated positions (a turn, a tilt, STOP, a blocked forward step) add 0.

    Args:
        positions: ``(x, y, z)`` base positions, metres, in visiting order.

    Returns:
        Metres travelled; 0 for a single position.

    Raises:
        ScoringError: On no positions at all (the reset position is always
            one), or a position that is not three finite numbers.
    """
    points = [_position(index, value) for index, value in enumerate(positions)]
    if not points:
        raise ScoringError(
            "path_length_3d needs at least the reset position, got none")
    return math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _check_options(require_stop: bool, path_tolerance_m: Optional[float],
                   native_tolerance: float) -> None:
    if not isinstance(require_stop, bool):
        raise HarnessError(
            "require_stop_for_success must be a bool, got %r (a string such "
            "as 'false' would read as True)" % (require_stop,))
    if (path_tolerance_m is not None
            and not is_length(path_tolerance_m)):
        raise HarnessError(
            "path_tolerance_m must be None (no check) or a finite "
            "non-negative number of metres, got %r" % (path_tolerance_m,))
    if not is_length(native_tolerance):
        raise HarnessError(
            "native_tolerance must be a finite non-negative number, got %r"
            % (native_tolerance,))


def score_episode(measurement: EpisodeMeasurement,
                  observed_path_length_m: Optional[float] = None, *,
                  require_stop_for_success: bool = True,
                  path_tolerance_m: Optional[float] = 1e-3,
                  native_tolerance: float = 1e-4) -> EpisodeScore:
    """Score one measured episode, refusing a measurement that would mis-score it.

    Three checks run before the scores are trusted:

    1. **STOP.** A success without STOP is refused when the protocol requires
       STOP (Habitat, RoboTHOR). Pass ``require_stop_for_success=False`` only
       for a protocol that grants success on proximity alone (SemExp's
       Gibson evaluator).
    2. **Path length.** When the caller recomputed ``p`` from the poses it
       observed (:func:`path_length_3d`), it must agree with the
       environment's ``p`` within ``path_tolerance_m``.
    3. **Native metrics.** Every value reported under :data:`NATIVE_KEYS` is
       compared with this module's: success exactly (0 or 1), SPL and
       SoftSPL within ``native_tolerance``, distance to goal within
       ``native_tolerance`` of ``dT`` (``inf`` equals only ``inf``).

    Args:
        measurement: What the environment measured.
        observed_path_length_m: ``p`` recomputed from the observed poses, or
            None when the caller had none.
        require_stop_for_success: Whether the benchmark grants success only
            on STOP.
        path_tolerance_m: Largest allowed disagreement between the observed
            and the reported ``p``, metres. None skips the check; the observed
            value is still recorded.
        native_tolerance: Largest allowed disagreement with a native SPL,
            SoftSPL or distance to goal.

    Returns:
        The score. ``native_checked`` lists the native keys checked, in
        :data:`NATIVE_KEYS` order.

    Raises:
        TypeError: If ``measurement`` is not an :class:`EpisodeMeasurement`.
        HarnessError: On a malformed option.
        ScoringError: When a check fails, or the observed path length is
            negative or not finite.
    """
    if not isinstance(measurement, EpisodeMeasurement):
        raise TypeError(
            "score_episode needs an EpisodeMeasurement, got %s"
            % type(measurement).__name__)
    _check_options(require_stop_for_success, path_tolerance_m,
                   native_tolerance)
    if (measurement.success and not measurement.stop_called
            and require_stop_for_success):
        raise ScoringError(
            "the environment reports success without STOP (termination %r), "
            "but this benchmark grants success only on STOP; fix the "
            "adapter's success, or pass require_stop_for_success=False for a "
            "protocol that does not require STOP" % (measurement.termination,))
    observed = None
    if observed_path_length_m is not None:
        observed = _metres("observed_path_length_m", observed_path_length_m)
        check_path_length(observed, measurement.path_length_m,
                          path_tolerance_m)
    spl = spl_score(measurement.success, measurement.shortest_path_m,
                    measurement.path_length_m)
    soft_spl = soft_spl_score(measurement.start_distance_to_goal_m,
                              measurement.final_distance_to_goal_m,
                              measurement.shortest_path_m,
                              measurement.path_length_m)
    checked = check_native_metrics(measurement, spl, soft_spl,
                                   native_tolerance)
    return EpisodeScore(
        success=measurement.success,
        spl=spl,
        soft_spl=soft_spl,
        distance_to_goal_m=float(measurement.final_distance_to_goal_m),
        path_length_m=float(measurement.path_length_m),
        shortest_path_m=float(measurement.shortest_path_m),
        observed_path_length_m=observed,
        native_checked=checked,
    )
