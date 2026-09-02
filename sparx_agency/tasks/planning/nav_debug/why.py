"""Compose the one-line "why" a frame shows under its gauges.

Two controllers can be flying, and each explains itself differently. The XTEND
drift-PID reports an authority and an escape state; FALCON's exploration loop
explains itself through the tracker (is it holding, is it diverged), the
reference (is the setpoint still moving, or parked at a trajectory's end) and
the altitude guard. Whichever lanes a run recorded contribute their part, so one
sentence covers a run from either stack.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

_DRIFT_ROLL_MIN_CM_S = 1.0       # below this the drift hold is not worth saying
_CLIMB_MIN_MS = 0.02


def why(row: dict, our_cmd: Optional[Tuple[float, float, float, float]],
        lanes: dict) -> str:
    """The frame's explanation line.

    Args:
        row: The spine row for this frame.
        our_cmd: The commanded ``(vx, vy, vz, wz)``, or None if not recorded.
        lanes: The resolved lane objects for this instant, as built by
            :meth:`~.session.NavSession._lanes_at`.

    Returns:
        A ``'; '``-joined sentence, empty when no lane had anything to say.
    """
    parts = _drift(row.get("drift")) + _exploration(lanes)
    if our_cmd is not None and our_cmd[2] > _CLIMB_MIN_MS:
        parts.append("climbing")
    return "; ".join(parts)


def _drift(drift) -> List[str]:
    """The XTEND drift-PID's own account of the tick."""
    if drift is None:
        return []
    parts = []
    if drift.escape_state and drift.escape_state != "IDLE":
        parts.append("escaping (%s)" % drift.escape_state)
    elif drift.authority:
        parts.append(drift.authority)
    roll_cm_s = abs(drift.drift_vy) * 100.0
    if roll_cm_s >= _DRIFT_ROLL_MIN_CM_S:
        parts.append("holding %.0fcm/s %s roll vs drift"
                     % (roll_cm_s, "left" if drift.drift_vy > 0 else "right"))
    return parts


def _exploration(lanes: dict) -> List[str]:
    """The exploration loop's account: tracker, reference, altitude guard."""
    parts = []
    tracking, reference = lanes.get("tracking"), lanes.get("reference")
    if tracking is not None and tracking.holding:
        parts.append("holding (reference %.1fs old)" % tracking.reference_age_s)
    elif tracking is not None and tracking.diverged:
        parts.append("diverged %.2fm off reference" % tracking.position_error_m)
    if reference is not None and not reference.moving:
        parts.append("reference parked at trajectory end")
    altitude = lanes.get("altitude")
    if altitude is not None and altitude.guard_rejected:
        parts.append("rangefinder guard rejecting")
    return parts
