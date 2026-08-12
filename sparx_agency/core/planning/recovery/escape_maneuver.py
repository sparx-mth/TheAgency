"""Scripted reflexes for backing out of a wall, and nothing more.

When :mod:`.stuck_detector` confirms that a command is not reaching the world, the
supervisor has seconds to spend, not a planning problem to solve. These are the
reflexes -- short, open-loop, bounded manoeuvres that try to break contact so
normal navigation can resume:

  * **Forward blocked.** The drone was told to fly and did not, and the map shows
    nothing -- an obstacle the camera cannot see (a door frame, glass, a chair
    leg). Charging harder is the one thing that must not happen. The escape is:
    brake, back off a little to break contact ("exit the wall in the other
    direction"), optionally roll a short step to one side to find out whether that
    side is open, settle, release. If the next attempt is blocked again, the roll
    goes the other way.
  * **Yaw blocked.** The drone was told to turn and did not: a surface is holding
    it. Brake, back off, roll clear (if the platform can strafe), settle, release.

The escape is **capability-aware**. A holonomic platform (``allow_lateral``) can
add the sideways probe; a one-axis follower that cannot command ``linear.y`` skips
it and relies on the back-off alone, which is the part every platform can fly.

The reflex deliberately stops there. It never edits the route, never remembers
where the obstacle was, and never decides to go somewhere else -- that is the
planner's job, and the supervisor signals it by running out of attempts
(:attr:`EscapeManeuver.exhausted`) rather than by planning anything itself.

Durations here are wall-clock seconds, which is correct for an open-loop script:
the manoeuvre's whole purpose is to move a fixed distance regardless of what the
sensors currently believe. That is the opposite of the frame-bounded confirmation
the *detector* uses, and the difference is deliberate.

This is the controller-agnostic sibling of
:class:`sparx_agency.core.planning.trackers.drift_pid.escape.EscapeManeuver`.

Python 3.8 compatible: no PEP 604 unions, no ``match``/``case``; stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .stuck_detector import StuckVerdict


class EscapeState(str):
    """Phases of an escape manoeuvre (a plain string enum, py3.8-safe)."""

    IDLE = "IDLE"
    BRAKE = "BRAKE"
    BACK = "BACK"
    PROBE = "PROBE"
    SETTLE = "SETTLE"


@dataclass(frozen=True)
class EscapeParams:
    """Tuning for :class:`EscapeManeuver` (SI units, body frame REP-103).

    Attributes:
        brake_s: Time held at zero before doing anything (s). The drone is probably
            still moving into whatever it hit; stopping first means the back-off
            starts from rest instead of fighting inertia.
        back_s: Time spent reversing to break contact (s).
        back_speed: Reverse speed during that phase (m/s). Small -- this is breaking
            contact, not retreating.
        probe_s: Time spent translating sideways to test whether a side is open (s).
        probe_speed: Lateral speed during the probe (m/s).
        settle_s: Time held still after the manoeuvre (s), so the depth model gets a
            stationary frame and the pose settles before navigation resumes.
        allow_lateral: Whether the platform can command ``linear.y``. False (a
            one-axis follower) drops the sideways probe and relies on the back-off.
        max_attempts: Escapes allowed for one blockage episode before the supervisor
            gives up and reports it to the planner. Two is usually right: one probe
            each way (or two back-offs on a non-holonomic platform).
    """

    brake_s: float = 0.4
    back_s: float = 0.7
    back_speed: float = 0.10
    probe_s: float = 0.8
    probe_speed: float = 0.10
    settle_s: float = 0.5
    allow_lateral: bool = True
    max_attempts: int = 2

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the manoeuvre relies on."""
        for name in ("brake_s", "back_s", "probe_s", "settle_s"):
            if getattr(self, name) < 0.0:
                raise ValueError("EscapeParams." + name + " must be >= 0")
        for name in ("back_speed", "probe_speed"):
            if getattr(self, name) <= 0.0:
                raise ValueError("EscapeParams." + name + " must be > 0")
        if self.max_attempts < 1:
            raise ValueError("EscapeParams.max_attempts must be >= 1")


@dataclass(frozen=True)
class EscapeCommand:
    """One tick of an escape manoeuvre (body frame REP-103).

    Attributes:
        vx: Forward speed to command (m/s; negative while backing out).
        vy: Lateral speed to command (m/s, + left; 0 unless probing).
        wz: Yaw rate to command (rad/s). Always 0 -- an escape never rotates; the
            axis it is escaping from may be the yaw axis itself.
        state: Phase the manoeuvre is in.
        active: True while the manoeuvre owns the command.
        reason: Short human-readable description, for narration.
    """

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    state: str = EscapeState.IDLE
    active: bool = False
    reason: str = ""


class EscapeManeuver:
    """The stateful escape FSM. One instance owns one blockage episode at a time."""

    def __init__(self, params=None):
        # type: (Optional[EscapeParams]) -> None
        self.params = params or EscapeParams()
        self.reset()

    def reset(self):
        # type: () -> None
        """Abandon any manoeuvre and forget the episode's attempt count."""
        self._state = EscapeState.IDLE
        self._t = 0.0
        self._axis = ""
        self._probe_sign = 1
        self._attempts = 0

    @property
    def active(self):
        # type: () -> bool
        """True while the manoeuvre owns the command."""
        return self._state != EscapeState.IDLE

    @property
    def attempts(self):
        # type: () -> int
        """Escapes run so far in the current blockage episode."""
        return self._attempts

    @property
    def exhausted(self):
        # type: () -> bool
        """True once the reflexes have had their turn and it is the planner's problem."""
        return self._attempts >= self.params.max_attempts

    def episode_over(self):
        # type: () -> None
        """Declare the blockage resolved, so the next one starts fresh.

        Call this when the drone has genuinely made progress again. Without it the
        attempt counter would carry over and the second obstacle of a flight would
        get fewer tries than the first.
        """
        self._attempts = 0

    def trigger(self, verdict, prefer_left=True):
        # type: (StuckVerdict, bool) -> bool
        """Begin an escape for ``verdict`` if one is warranted.

        Args:
            verdict: The confirmed blockage to escape from.
            prefer_left: Which way to probe first -- normally toward whichever side
                the route continues on, so a successful probe also makes progress.

        Returns:
            True if a manoeuvre started, False if one is already running, the
            verdict is empty, or the attempts for this episode are spent.
        """
        if self.active or not verdict.stuck or self.exhausted:
            return False
        self._axis = verdict.axis
        self._state = EscapeState.BRAKE
        self._t = 0.0
        self._attempts += 1
        if self._attempts > 1:
            # Second try at the same obstacle: the first side did not work.
            self._probe_sign = -self._probe_sign
        else:
            self._probe_sign = 1 if prefer_left else -1
        return True

    def abort(self):
        # type: () -> None
        """Stop the manoeuvre immediately, keeping the attempt count.

        Used when localization degrades mid-escape: an open-loop manoeuvre flown on
        a pose that has gone cold is exactly the way to end up somewhere nobody
        expected.
        """
        self._state = EscapeState.IDLE
        self._t = 0.0

    def step(self, dt):
        # type: (float) -> EscapeCommand
        """Advance the manoeuvre by ``dt`` and return this tick's command."""
        if dt <= 0.0:
            raise ValueError("EscapeManeuver.step: dt must be > 0")
        if not self.active:
            return EscapeCommand()
        p = self.params
        self._t += dt
        side = "left" if self._probe_sign > 0 else "right"

        if self._state == EscapeState.BRAKE:
            if self._t >= p.brake_s:
                self._advance(EscapeState.BACK)
            return EscapeCommand(state=EscapeState.BRAKE, active=True,
                                 reason="Blocked on the %s axis -- stopping"
                                        % self._axis)

        if self._state == EscapeState.BACK:
            if self._t >= p.back_s:
                # Only holonomic platforms can probe sideways; the rest go straight
                # to settle and rely on the back-off having broken contact.
                self._advance(EscapeState.PROBE if p.allow_lateral
                              else EscapeState.SETTLE)
            return EscapeCommand(vx=-p.back_speed, state=EscapeState.BACK,
                                 active=True,
                                 reason="Something I cannot see is in the way -- "
                                        "backing out to break contact")

        if self._state == EscapeState.PROBE:
            if self._t >= p.probe_s:
                self._advance(EscapeState.SETTLE)
            return EscapeCommand(vy=p.probe_speed * self._probe_sign,
                                 state=EscapeState.PROBE, active=True,
                                 reason="Trying a short step to the %s to get "
                                        "round it" % side)

        if self._t >= p.settle_s:
            self._state = EscapeState.IDLE
            self._t = 0.0
            return EscapeCommand(reason="Escape finished -- back to the route")
        return EscapeCommand(state=EscapeState.SETTLE, active=True,
                             reason="Holding still to see where I ended up")

    def _advance(self, state):
        # type: (str) -> None
        """Enter ``state`` and restart its phase clock."""
        self._state = state
        self._t = 0.0

