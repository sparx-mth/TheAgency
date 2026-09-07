"""Fly a policy's discrete turn as a real rotation: turn, stop, look again.

A VLN policy's alphabet is ``STOP / FORWARD / TURN_LEFT / TURN_RIGHT``, and the
turn in it is a **pure rotation**: upstream's
``trajectory_to_discrete_actions_close_to_goal`` advances the position only on a
forward action, and a turn changes the heading alone by a fixed angle. That is
also how the policy was evaluated -- each observation is taken from a standstill,
after the previous action finished.

Flying it as a *path* loses exactly that. Rendering a turn as a short step bent
by the turn angle (which is what
:func:`~sparx_agency.core.planning.vlas.internvla_n1.geometry.trajectory_from_action`
does, because everything downstream consumes polylines) gives a holonomic
tracker a waypoint it can reach by crabbing sideways -- so the aircraft
translates 0.25 m and barely changes where it is looking, and a model asking to
look somewhere else never gets the view it asked for. The turn was the *whole*
message and only the least important part of it was flown.

This is the other half: given a target heading, produce the yaw rate to get
there and say when the rotation is **finished** -- rotated, stopped and settled
-- so the caller can take the next observation from a standstill. Deliberately
slow by default: the point of the manoeuvre is the frame at the end of it, and a
fast rotation that overshoots and rings costs more time in settling than it
saves in turning.

Two consumers, one implementation, on purpose. A follower calls
:meth:`TurnInPlace.update` for :attr:`TurnCommand.yaw_rate`; a runner that has to
know when it may next ask the policy calls the same method and reads
:attr:`TurnCommand.done`. Both are fed the same measured yaw, so neither has to
re-derive the other's state machine.

Sibling of :mod:`~sparx_agency.core.planning.vlas.common.yaw_search`, which
decides *where* to look when the policy will not move; this decides how to get
there and when you have arrived.

Clock-free (every method takes ``now_s``), ROS-free, and **numpy-free** -- stdlib
``math`` only -- so the FALCON Noetic container can import it. Python 3.8 idioms
throughout: no dataclass ``slots``, no PEP 604 unions, no ``match``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees, radians
from typing import Optional

from sparx_agency.core.common.types.geometry import normalize_angle

#: Manoeuvre states, also returned on the command for logging.
IDLE = "IDLE"
TURNING = "TURNING"
SETTLING = "SETTLING"
DONE = "DONE"


@dataclass(frozen=True)
class TurnSpec:
    """How a rotation in place is flown, and when it counts as finished.

    Attributes:
        yaw_rate: Cruise rate of the rotation, rad/s. Small on purpose. The
            manoeuvre exists to produce one good observation at the end of it,
            and this airframe translates by tilting -- a fast yaw drags the
            camera through motion blur and leaves the aircraft ringing, which
            the settle below then has to wait out anyway.
        min_yaw_rate: Floor under the proportional taper, rad/s. Without it the
            last few degrees are commanded at a rate the plugin's yaw PID cannot
            act on, and the rotation never closes -- it simply times out a
            degree short, every time.
        slow_down_rad: Heading error inside which the rate tapers linearly to
            ``min_yaw_rate``. Above it the rotation runs at ``yaw_rate``.
        tolerance_rad: Heading error at or below which the rotation has arrived.
            Must be comfortably larger than the platform's yaw noise or the
            manoeuvre oscillates between TURNING and SETTLING.
        settle_rate_rad_s: Measured yaw rate below which the aircraft counts as
            stopped. Arriving is not enough: the observation at the end has to
            be taken from a standstill, and a drone that is still coasting
            through the target heading is not one.
        settle_s: How long the aircraft must be inside ``tolerance_rad`` *and*
            below ``settle_rate_rad_s`` before the turn is DONE.
        timeout_s: Give up and report DONE with ``timed_out`` set. A rotation
            that cannot complete -- a capsized airframe, a follower that is not
            listening, a plugin refusing yaw while landed -- must never wedge
            the flight it is a step of.
    """

    yaw_rate: float = 0.35
    min_yaw_rate: float = 0.10
    slow_down_rad: float = 0.20
    tolerance_rad: float = 0.035
    settle_rate_rad_s: float = 0.05
    settle_s: float = 0.4
    timeout_s: float = 8.0

    def __post_init__(self):
        # type: () -> None
        """Reject a spec that cannot converge.

        Every one of these is a configuration that *runs* and simply never
        finishes a turn, which in the air is indistinguishable from a dead
        follower.
        """
        for name in ("yaw_rate", "min_yaw_rate", "slow_down_rad",
                     "tolerance_rad", "settle_rate_rad_s", "timeout_s"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("%s must be positive; got %r"
                                 % (name, getattr(self, name)))
        if float(self.settle_s) < 0.0:
            raise ValueError("settle_s cannot be negative; got %r" % (self.settle_s,))
        if float(self.min_yaw_rate) > float(self.yaw_rate):
            raise ValueError(
                "min_yaw_rate (%r) cannot exceed yaw_rate (%r): the floor would "
                "become the cruise rate and the taper would speed the rotation "
                "up as it closes" % (self.min_yaw_rate, self.yaw_rate))


@dataclass(frozen=True)
class TurnCommand:
    """One control step of a rotation in place.

    Attributes:
        yaw_rate: Rate to command, rad/s, CCW positive. Zero while settling and
            when idle.
        active: A rotation is in progress -- the caller must not translate.
        done: The rotation finished on this step (arrived and settled, or timed
            out). True for exactly one step per manoeuvre.
        timed_out: ``done`` was reached by running out of time rather than by
            arriving. Worth logging: it means the aircraft did not turn.
        remaining_rad: Signed heading error still to fly, CCW positive.
        state: One of :data:`IDLE` / :data:`TURNING` / :data:`SETTLING` /
            :data:`DONE`, for diagnostics.
    """

    yaw_rate: float
    active: bool
    done: bool
    timed_out: bool
    remaining_rad: float
    state: str


_IDLE_COMMAND = TurnCommand(yaw_rate=0.0, active=False, done=False,
                            timed_out=False, remaining_rad=0.0, state=IDLE)


class TurnInPlace(object):
    """Rotate to a heading, stop, and say when the next observation may be taken.

    Typical use in a follower, once per control step::

        cmd = turn.update(yaw, now)
        if cmd.active:
            publish(vx=0.0, vy=0.0, yaw_rate=cmd.yaw_rate)   # altitude hold only
    """

    def __init__(self, spec=None):
        # type: (Optional[TurnSpec]) -> None
        self.spec = spec or TurnSpec()
        self._target = None      # type: Optional[float]
        self._state = IDLE
        self._started_s = 0.0
        self._settled_s = 0.0
        self._prev_yaw = None    # type: Optional[float]
        self._prev_s = None      # type: Optional[float]
        self.turns = 0

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self, current_yaw, delta_rad, now_s):
        # type: (float, float, float) -> float
        """Begin a rotation of ``delta_rad`` from where the aircraft is looking.

        Args:
            current_yaw: Measured heading now, world radians.
            delta_rad: How far to turn, radians, CCW positive.
            now_s: Clock, the same timebase every later call uses.

        Returns:
            The absolute target heading, world radians in ``(-pi, pi]``.
        """
        return self.start_to(normalize_angle(float(current_yaw) + float(delta_rad)),
                             now_s)

    def start_to(self, target_yaw, now_s):
        # type: (float, float) -> float
        """Begin a rotation to an absolute heading.

        Restarting an active manoeuvre is legal and re-aims it -- a fresh
        decision supersedes the one being flown -- but it also restarts the
        timeout, which is why a caller that merely re-publishes the same goal
        should check :attr:`target` first.

        Args:
            target_yaw: Absolute heading to reach, world radians.
            now_s: Clock.

        Returns:
            The normalised target heading.
        """
        self._target = normalize_angle(float(target_yaw))
        self._state = TURNING
        self._started_s = float(now_s)
        self._settled_s = 0.0
        self._prev_yaw = None
        self._prev_s = None
        self.turns += 1
        return self._target

    def cancel(self):
        # type: () -> None
        """Abandon the rotation without reporting ``done``."""
        self._target = None
        self._state = IDLE
        self._settled_s = 0.0
        self._prev_yaw = None
        self._prev_s = None

    # ── state ────────────────────────────────────────────────────────
    @property
    def active(self):
        # type: () -> bool
        """A rotation is in progress."""
        return self._state in (TURNING, SETTLING)

    @property
    def state(self):
        # type: () -> str
        """Current manoeuvre state."""
        return self._state

    @property
    def target(self):
        # type: () -> Optional[float]
        """The heading being turned to, or ``None`` when idle."""
        return self._target

    # ── per control step ─────────────────────────────────────────────
    def update(self, yaw, now_s, measured_yaw_rate=None):
        # type: (float, float, Optional[float]) -> TurnCommand
        """Advance one step and return the rate to command.

        Args:
            yaw: Measured heading, world radians.
            now_s: Clock.
            measured_yaw_rate: Measured yaw rate, rad/s, when the caller has one
                (odometry does). Left ``None`` it is differentiated from ``yaw``,
                which is good enough for the settle test and means a caller with
                only a heading still works.

        Returns:
            The :class:`TurnCommand` for this step. ``done`` is True on exactly
            one step per manoeuvre; the state falls back to :data:`IDLE`
            afterwards so a caller that misses it does not stall.
        """
        if self._target is None:
            return _IDLE_COMMAND

        now = float(now_s)
        dt = 0.0 if self._prev_s is None else max(0.0, now - self._prev_s)
        rate = self._yaw_rate_measurement(yaw, dt, measured_yaw_rate)
        self._prev_yaw = float(yaw)
        self._prev_s = now
        remaining = normalize_angle(self._target - float(yaw))

        if now - self._started_s > self.spec.timeout_s:
            return self._finish(remaining, timed_out=True)

        if abs(remaining) > self.spec.tolerance_rad and self._state == TURNING:
            return TurnCommand(yaw_rate=self._command(remaining), active=True,
                               done=False, timed_out=False,
                               remaining_rad=remaining, state=TURNING)

        # Inside tolerance (or already settling). Command nothing and wait for
        # the aircraft to actually stop: the frame this manoeuvre exists to
        # produce is worthless taken mid-coast, and this airframe coasts.
        if abs(remaining) > self.spec.tolerance_rad:
            # Drifted back out while settling -- re-aim rather than sit at zero
            # waiting for a heading that is getting worse.
            self._state = TURNING
            self._settled_s = 0.0
            return TurnCommand(yaw_rate=self._command(remaining), active=True,
                               done=False, timed_out=False,
                               remaining_rad=remaining, state=TURNING)
        self._state = SETTLING
        if abs(rate) > self.spec.settle_rate_rad_s:
            self._settled_s = 0.0
        else:
            self._settled_s += dt
            if self._settled_s >= self.spec.settle_s:
                return self._finish(remaining, timed_out=False)
        return TurnCommand(yaw_rate=0.0, active=True, done=False, timed_out=False,
                           remaining_rad=remaining, state=SETTLING)

    # ── helpers ──────────────────────────────────────────────────────
    def _command(self, remaining):
        # type: (float) -> float
        """Rate for this error: taper inside ``slow_down_rad``, floored."""
        magnitude = min(self.spec.yaw_rate,
                        max(self.spec.min_yaw_rate,
                            self.spec.yaw_rate * abs(remaining) / self.spec.slow_down_rad))
        return magnitude if remaining >= 0.0 else -magnitude

    def _yaw_rate_measurement(self, yaw, dt, measured):
        # type: (float, float, Optional[float]) -> float
        """The measured yaw rate: the caller's, or differentiated from the heading.

        A caller with odometry has the real thing and should pass it. The
        fallback exists so a heading is enough to use this class at all, and it
        is deliberately zero on the first step of a manoeuvre -- one step is not
        a rate, and guessing one there would let a turn declare itself settled
        before it had moved.
        """
        if measured is not None:
            return float(measured)
        if self._prev_yaw is None or dt <= 1e-6:
            return 0.0
        return normalize_angle(float(yaw) - self._prev_yaw) / dt

    def _finish(self, remaining, timed_out):
        # type: (float, bool) -> TurnCommand
        """Report DONE once, then go idle."""
        self._target = None
        self._state = IDLE
        self._settled_s = 0.0
        self._prev_yaw = None
        self._prev_s = None
        return TurnCommand(yaw_rate=0.0, active=False, done=True,
                           timed_out=bool(timed_out), remaining_rad=remaining,
                           state=DONE)


def turn_spec_from_config(config, prefix=""):
    # type: (dict, str) -> TurnSpec
    """Build a :class:`TurnSpec` from a plain config mapping.

    Degrees where a human writes degrees (``*_deg``), radians where the rest of
    the stack speaks radians, so a YAML author is never asked to convert.

    Args:
        config: mapping of knob name to value; unknown keys are ignored.
        prefix: optional key prefix, e.g. ``"turn_"``.

    Returns:
        The spec, with any absent knob left at its default.
    """
    def _get(name, default):
        return config.get(prefix + name, default)

    rate_deg = _get("yaw_rate_deg_s", None)
    min_deg = _get("min_yaw_rate_deg_s", None)
    slow_deg = _get("slow_down_deg", None)
    tol_deg = _get("tolerance_deg", None)
    settle_deg = _get("settle_rate_deg_s", None)
    defaults = TurnSpec()
    return TurnSpec(
        yaw_rate=radians(float(rate_deg)) if rate_deg is not None else defaults.yaw_rate,
        min_yaw_rate=(radians(float(min_deg)) if min_deg is not None
                      else defaults.min_yaw_rate),
        slow_down_rad=(radians(float(slow_deg)) if slow_deg is not None
                       else defaults.slow_down_rad),
        tolerance_rad=(radians(float(tol_deg)) if tol_deg is not None
                       else defaults.tolerance_rad),
        settle_rate_rad_s=(radians(float(settle_deg)) if settle_deg is not None
                           else defaults.settle_rate_rad_s),
        settle_s=float(_get("settle_s", defaults.settle_s)),
        timeout_s=float(_get("timeout_s", defaults.timeout_s)),
    )


def describe(spec):
    # type: (TurnSpec) -> str
    """One line naming what a spec will actually fly, in degrees."""
    return ("turn at %.0f deg/s (floor %.0f), taper inside %.0f deg, arrive "
            "within %.1f deg, settle %.1f s, timeout %.0f s"
            % (degrees(spec.yaw_rate), degrees(spec.min_yaw_rate),
               degrees(spec.slow_down_rad), degrees(spec.tolerance_rad),
               spec.settle_s, spec.timeout_s))
