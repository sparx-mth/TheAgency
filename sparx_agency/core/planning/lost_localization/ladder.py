"""The recovery ladder as pure data (ROS-free, 3.8-safe).

The escalation is a *table*, not control flow: eleven hand-written states would
be eleven near-identical methods, and the one thing that actually varies between
them -- which axis to drive, how fast, for how long -- is data. Keeping it a
tuple of :class:`Rung` means :mod:`.state_machine` runs ONE rung-executor for
every step of the ladder, and a rung can be turned off in config (the climb rungs
are, on platforms whose bridge refuses a vertical velocity) instead of deleted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .params import LostLocalizationParams

#: Rung kinds. The state machine maps each to exactly one driven axis, because
#: the XTEND bridge is a single-action hold protocol: a command carrying two
#: axes has the lower-priority one silently dropped (yaw beats x beats z).
BACK = "back"    # -x body velocity  (retreat the way we came)
STOP = "stop"    # zero              (stand still and look)
CLIMB = "climb"  # +z body velocity  (see over whatever is occluding the tag)
TURN = "turn"    # +/-yaw rate       (sweep for a tag in any direction)


@dataclass(frozen=True)
class Rung:
    """One step of the ladder.

    Attributes:
        kind: One of :data:`BACK`, :data:`STOP`, :data:`CLIMB`, :data:`TURN`.
        duration_s: How long the rung drives. For a :data:`TURN` that carries a
            ``target_rad`` this is the timeout, not the target -- that sweep
            normally ends on angle.
        label: Human name for logs and the status topic (e.g. ``"back#2"``).
        rate: Magnitude to drive this rung's axis at, overriding the tuning's
            default for that stage. SIGNED for :data:`TURN` (+ = left/CCW);
            unsigned for :data:`BACK`, which always retreats. None => use the
            tuning. Exists so :mod:`.persist` can hand back the rate the
            navigator was already flying rather than the recovery's own.
        target_rad: Rotation (rad) that ends a :data:`TURN` early. None => the
            rung runs its full ``duration_s``. This is what makes the ladder's
            360 sweep a search for a tag rather than a fixed-length spin, and
            what keeps a short, deliberate turn (a persist) from inheriting it.
    """

    kind: str
    duration_s: float
    label: str
    rate: Optional[float] = None
    target_rad: Optional[float] = None


def build_ladder(p: LostLocalizationParams) -> Tuple[Rung, ...]:
    """Assemble the ladder from tuning.

    Every motion rung is followed by its own settle, so disabling a stage (e.g.
    ``climb_enabled=False``) removes the settle with it and can never leave two
    settles back to back.

    Args:
        p: The tuning to build from.

    Returns:
        The rungs in execution order. May be empty if every stage is disabled --
        the caller then gives up immediately rather than flying blind.
    """
    rungs = []
    for i in range(p.back_repeats):
        rungs.append(Rung(BACK, p.back_duration_s, "back#%d" % (i + 1)))
        rungs.append(Rung(STOP, p.dwell_s, "settle-after-back#%d" % (i + 1)))
    if p.climb_enabled:
        for i in range(p.climb_repeats):
            rungs.append(Rung(CLIMB, p.climb_duration_s, "climb#%d" % (i + 1)))
            rungs.append(Rung(STOP, p.dwell_s, "settle-after-climb#%d" % (i + 1)))
    if p.turn_enabled:
        rungs.append(Rung(TURN, p.turn_timeout_s, "sweep360",
                          target_rad=p.turn_target_rad))
    return tuple(rungs)
