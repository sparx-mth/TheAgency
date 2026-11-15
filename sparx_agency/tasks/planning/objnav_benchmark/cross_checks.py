"""Check a measurement against what the harness saw and computed, before its scores are trusted.

An adapter bug rarely crashes a run; it moves SPL. Two such bugs show only
when the measurement is set beside a second opinion. A path length the
environment accounts differently from the poses the runner observed is a unit
or accounting bug: positions in another unit than metres, a dropped or
flattened axis, ``p`` accounted in 2-D or skipping a step, or a first pose
that is not the reset pose. A sum of chord lengths cannot catch an axis
permutation or a mirror -- a rotated or mirrored frame keeps every chord --
so an unconverted Y-up frame or a mirrored handedness is the kinematic
check's to catch (``kinematics.py``), not this one's. A simulator's own metric
that the harness's recomputation contradicts is a frame, a unit, or a
distance measured to the wrong goal. :func:`score_episode` in ``scoring.py``
runs both checks and raises instead of scoring.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_DISTANCE_TO_GOAL,
    NATIVE_KEYS,
    NATIVE_SOFT_SPL,
    NATIVE_SPL,
    NATIVE_SUCCESS,
    EpisodeMeasurement,
)
from sparx_agency.tasks.planning.objnav_benchmark.errors import ScoringError


def check_path_length(observed: float, reported: float,
                      tolerance: Optional[float]) -> None:
    """Refuse an environment's path length that the observed poses disagree with.

    Args:
        observed: ``p`` recomputed from the observed poses, metres.
        reported: ``p`` as the environment accounts it, metres.
        tolerance: Largest allowed disagreement, metres; None skips the check.

    Raises:
        ScoringError: When the two differ by more than ``tolerance``.
    """
    if tolerance is None or abs(observed - reported) <= tolerance:
        return
    raise ScoringError(
        "the environment reports a path length of %.6f m but the observed "
        "poses trace %.6f m (tolerance %g m); this is almost always an "
        "adapter unit or accounting bug -- positions in another unit than "
        "metres, a dropped or flattened axis, p accounted in 2-D or skipping "
        "a step, or a first pose that is not the reset pose"
        % (reported, observed, tolerance))


def check_native_metrics(measurement: EpisodeMeasurement, spl: float,
                         soft_spl: float, tolerance: float) -> Tuple[str, ...]:
    """Cross-check every native metric the environment reported against the harness's value.

    Success must agree exactly and be exactly 0 or 1; SPL, SoftSPL and the
    distance to goal must agree within ``tolerance``. An infinite distance
    agrees only with another infinite one: ``inf - inf`` is NaN, which a plain
    tolerance check would pass for the wrong reason.

    Args:
        measurement: The measurement whose ``native_metrics`` are checked.
        spl: The harness's SPL for it.
        soft_spl: The harness's SoftSPL for it.
        tolerance: Largest allowed disagreement for SPL, SoftSPL and distance.

    Returns:
        The keys checked, in :data:`NATIVE_KEYS` order; empty when the
        environment reported none.

    Raises:
        ScoringError: On a native success other than 0 or 1, or any native
            value that disagrees with the harness's.
    """
    ours: Dict[str, float] = {
        NATIVE_SUCCESS: 1.0 if measurement.success else 0.0,
        NATIVE_SPL: spl,
        NATIVE_SOFT_SPL: soft_spl,
        NATIVE_DISTANCE_TO_GOAL: float(measurement.final_distance_to_goal_m),
    }
    checked: List[str] = []
    for key in NATIVE_KEYS:
        if key not in measurement.native_metrics:
            continue
        theirs = float(measurement.native_metrics[key])
        if key == NATIVE_SUCCESS and theirs not in (0.0, 1.0):
            raise ScoringError(
                "native metric %r must be exactly 0 or 1, got %r"
                % (key, theirs))
        if math.isinf(theirs) or math.isinf(ours[key]) or key == NATIVE_SUCCESS:
            agrees = theirs == ours[key]
        else:
            agrees = abs(theirs - ours[key]) <= tolerance
        if not agrees:
            raise ScoringError(
                "native metric %r disagrees: the simulator reports %r, the "
                "harness computes %r (tolerance %g); an adapter bug -- a "
                "frame, a unit, or a distance measured to the wrong goal"
                % (key, theirs, ours[key], tolerance))
        checked.append(key)
    return tuple(checked)
