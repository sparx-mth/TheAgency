"""Finish the move that was already flying when the pose went cold.

The recovery ladder (:mod:`.ladder`) assumes the drone is standing still and has
no idea where it is. That is the right assumption a second into a dropout, but it
is the WRONG one in the instant the tag leaves the frame, because at that instant
we still know something the ladder does not: what the navigator was in the middle
of doing, and why.

Two cases, and they want opposite things:

* **Mid-turn.** The tag left the frame *because we rotated it out of view*. The
  drone is part-way from one heading to another and the fastest way back to a
  localized state is usually to KEEP GOING -- the next tag is very often already
  swinging into frame. Stopping dead mid-turn strands the camera pointed exactly
  where nothing is visible, which is the one heading we know does not work.
  This includes the stationary settle *between* yaw bursts: the drone is not
  moving, but it is still in a turn, and the last thing it wanted was more of it.
* **Mid-advance.** Flying forward and losing the tag usually means we flew too
  close to it -- the drone has driven up to a wall and the marker has gone out of
  the camera's field of view, or out of focus. Continuing is the one thing that
  cannot help and might hit the wall. Undo it instead: give the metres back, then
  stop and look from where we could see.

So the persist stage is not a fifth rung on the ladder, it is a short, bounded
*prelude* chosen by context, and every branch ends stationary -- a still camera is
what actually re-acquires a tag. If it works, recovery never happens at all. If it
does not, the ladder runs exactly as before, from a standstill, as it assumes.

The context is a fact about the past, so it is read ONCE, on the transition out of
NOMINAL, and never re-read: the moment recovery takes over, the last command it
can see is its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .ladder import BACK, STOP, TURN, Rung
from .params import LostLocalizationParams

#: What the navigator was doing when localization died.
TURNING = "turning"    # rotating in place (or settling between yaw bursts)
FORWARD = "forward"    # translating -- advancing along the route
UNKNOWN = "unknown"    # nobody knows: no recent command, or none ever seen

_KINDS = (TURNING, FORWARD, UNKNOWN)


@dataclass(frozen=True)
class MotionContext:
    """The move that was in flight when the pose went cold.

    Attributes:
        kind: One of :data:`TURNING`, :data:`FORWARD`, :data:`UNKNOWN`.
        rate: The last rate commanded on that move's axis -- a SIGNED yaw rate
            (rad/s) for :data:`TURNING`, a forward speed (m/s) for
            :data:`FORWARD`. Unused (and must be 0) for :data:`UNKNOWN`.

    Raises:
        ValueError: On an unknown kind, or a known kind with a zero rate -- a
            stop carries no intent, so it can never be the thing we continue.
    """

    kind: str = UNKNOWN
    rate: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError("kind must be one of %r, got %r"
                             % (_KINDS, self.kind))
        if self.kind != UNKNOWN and self.rate == 0.0:
            raise ValueError(
                "a %s context needs a non-zero rate: a zero command is a stop, "
                "which expresses no intent to continue" % (self.kind,))

    @classmethod
    def turning(cls, yaw_rate: float) -> "MotionContext":
        """Rotating at ``yaw_rate`` rad/s (signed: + = left/CCW)."""
        return cls(TURNING, float(yaw_rate))

    @classmethod
    def forward(cls, speed: float) -> "MotionContext":
        """Translating at ``speed`` m/s along body x (sign is not used)."""
        return cls(FORWARD, float(speed))

    @classmethod
    def unknown(cls) -> "MotionContext":
        """Nothing to finish -- recovery starts at the plain stop."""
        return cls(UNKNOWN, 0.0)


def build_persist(p: LostLocalizationParams,
                  ctx: MotionContext) -> Tuple[Rung, ...]:
    """Assemble the prelude for the move that was in flight.

    Both motion branches are followed by their own settle, for the same reason
    every ladder rung is: the drone has to hold still for the camera to have a
    real chance at a tag, and without it a turn would run straight into the
    ladder's first back-up without ever having looked.

    Args:
        p: The tuning to build from.
        ctx: What the navigator was doing when the pose went cold.

    Returns:
        The rungs in execution order, or empty when there is nothing worth
        finishing (``persist_enabled`` false, or an :data:`UNKNOWN` context).
        The caller then goes straight to the plain stop, exactly as it did
        before this stage existed.
    """
    if not p.persist_enabled or ctx.kind == UNKNOWN:
        return ()
    settle = Rung(STOP, p.persist_settle_s, "persist-look")
    if ctx.kind == TURNING:
        # The rate is carried through verbatim, NOT re-derived from the sweep
        # tuning. Re-sending the value already flying means the platform's
        # hold-style bridge sees no change at all and the rotation simply
        # continues, where a different rate would step the drone's yaw mid-turn.
        # It is also uncapped, unlike the retreat below: a rotation in place goes
        # nowhere, so a rate the follower was already flying is one we know is safe.
        return (Rung(TURN, p.persist_turn_s, "persist-turn", rate=ctx.rate),
                settle)
    # Retreat at the speed we advanced at, but never faster than the ladder's own
    # blind-retreat speed: this is reversing into space no sensor is looking at
    # (the map is forward-facing and starved of poses while lost), so it is capped
    # to a speed that has been flown backwards before.
    return (Rung(BACK, p.persist_back_s, "persist-back",
                 rate=min(abs(ctx.rate), p.back_speed)),
            settle)