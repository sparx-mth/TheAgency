"""Turn-to-a-bearing "aim and look" policy (ROS-free).

The staged object approach flies to a **vantage point** (the room centre) rather
than to the object's catalogued coordinate, because that coordinate is only as
accurate as the room map that produced it: fly onto it and a few tens of
centimetres of error can leave the object behind or beside the drone, out of
frame, with nothing left to servo onto. Standing off and *looking* is the robust
move -- the camera resolves a bearing far better than the map resolves a position.

This policy is the "look" half. Given the signed heading error to the object
(``bearing_to_object - current_yaw``, normalized), it produces the body-frame
command that swings the nose onto that bearing and then **holds still** long
enough for the detector to score a few clean frames down it. The cycle is::

    SETTLE -> TURN -> SETTLE -> TURN -> ... -> LOOK -> DONE

It starts in SETTLE (a stop) on purpose: the drone has just flown in and is still
coasting, so the first heading measurement would otherwise be taken while moving.

Turning is **pulsed, not continuous**, for the same reason the route follower's
turns are: the platform's yaw is discrete and inertial. A burst is commanded
``yaw_coast_rad`` SHORT of the measured error and the coast fills the rest, then
the drone stops (SETTLE) and the heading is re-measured before the next burst.
A continuous proportional yaw would sail past the bearing and hunt.

``DONE`` is the terminal phase, reached either by looking for ``look_s`` without
the mission confirming the object, or by ``timeout_s`` elapsing with the heading
never converging (a platform that will not turn, or a stale pose). Both mean the
same thing to the caller -- aiming did not find it -- so the mission can fall back
to flying at the object's catalogued coordinate. The policy never blocks forever.

Clock-free and I/O-free: it is fed the heading error and ``dt`` and returns a
command; the caller owns the pose, the bearing arithmetic and the publishing.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.common.types import ControlCommand

#: Aim phases (also reported on the decision, for logging / the HUD).
SETTLE = "settle"   # stopped: bleed off the coast, then re-measure the heading
TURN = "turn"       # firing one yaw burst toward the bearing
LOOK = "look"       # on the bearing, holding still so the detector gets clean frames
DONE = "done"       # terminal: looked (or timed out) without the object being confirmed


@dataclass(frozen=True)
class AimBearingConfig:
    """Tuning for :class:`AimBearingPolicy` (SI units, body frame REP-103).

    Attributes:
        yaw_rate: Rotation speed commanded during a TURN burst (rad/s). Keep it at
            the platform's proven turn rate -- this is a magnitude the yaw deadband
            actually moves on, not a proportional gain.
        yaw_coast_rad: How far the platform keeps rotating after a burst stops
            (rad). Each burst is aimed this much SHORT of the measured error, so
            the coast lands the nose on the bearing instead of past it.
        tolerance_rad: Heading error at or below which the nose counts as being on
            the bearing (-> LOOK). It cannot usefully be tighter than roughly half
            the smallest achievable turn (``min_burst_s * yaw_rate + yaw_coast_rad``),
            since a smaller residual can only be made worse by another burst. Being
            a few degrees off is harmless anyway: the camera's field of view is far
            wider than this, so the object is in frame well before the nose is exact.
        min_burst_s: Shortest TURN burst (s). Below this the platform's yaw deadband
            swallows the command and the drone does not turn at all.
        max_burst_s: Longest TURN burst (s). Caps how much angle is swept open-loop
            before the heading is re-measured.
        settle_s: Stop between bursts (s) -- long enough for the yaw coast to finish
            so the next measurement is of a stationary drone.
        look_s: How long to hold still on the bearing (s) before giving up on
            seeing the object. Size it against the detector's frame rate and the
            confirmation gate's N-consecutive-frames requirement, with margin.
        timeout_s: Hard cap on the whole episode (s). Guarantees termination when
            the heading never converges -- a platform that will not yaw, or a pose
            that never updates -- rather than turning and re-measuring forever.
    """

    yaw_rate: float = 0.7
    yaw_coast_rad: float = 0.26
    tolerance_rad: float = 0.20
    min_burst_s: float = 0.2
    max_burst_s: float = 0.6
    settle_s: float = 1.0
    look_s: float = 4.0
    timeout_s: float = 25.0

    def __post_init__(self) -> None:
        if self.yaw_rate <= 0.0:
            raise ValueError("yaw_rate must be > 0")
        if self.yaw_coast_rad < 0.0:
            raise ValueError("yaw_coast_rad must be >= 0")
        if self.tolerance_rad <= 0.0:
            raise ValueError("tolerance_rad must be > 0")
        if self.min_burst_s <= 0.0:
            raise ValueError("min_burst_s must be > 0")
        if self.max_burst_s < self.min_burst_s:
            raise ValueError("max_burst_s must be >= min_burst_s")
        if self.settle_s < 0.0:
            raise ValueError("settle_s must be >= 0")
        if self.look_s < 0.0:
            raise ValueError("look_s must be >= 0")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0")


@dataclass(frozen=True)
class AimDecision:
    """One tick of the aim manoeuvre.

    Attributes:
        command: Body-frame velocity to publish (a pure yaw, or a stop).
        phase: Current phase (SETTLE / TURN / LOOK / DONE).
        on_bearing: True once the nose is within ``tolerance_rad`` of the bearing
            (i.e. from LOOK onward) -- the drone is pointing at where the object
            should be, whether or not the detector has actually seen it.
        finished: True in the terminal DONE phase: aiming is over and the object
            was not confirmed while it ran. The caller falls back from here.
        timed_out: True when DONE was reached via ``timeout_s`` (the heading never
            converged) rather than by looking. Distinguishes "looked and saw
            nothing" from "could not even turn", which are different faults.
    """

    command: ControlCommand
    phase: str
    on_bearing: bool
    finished: bool
    timed_out: bool = False


class AimBearingPolicy:
    """Swing the nose onto a bearing in pulsed bursts, then hold still and look."""

    def __init__(self, config=None) -> None:
        self.cfg = config or AimBearingConfig()
        self.reset()

    def reset(self) -> None:
        """Restart the manoeuvre at a SETTLE (arrest the arrival motion first)."""
        self._phase = SETTLE
        self._t = 0.0            # time in the current phase
        self._elapsed = 0.0      # time in the whole episode (for timeout_s)
        self._burst_s = 0.0      # commanded duration of the burst in flight
        self._burst_sign = 0.0   # its direction (+1 CCW / -1 CW)
        self._timed_out = False

    @property
    def phase(self) -> str:
        return self._phase

    def update(self, heading_error: float, dt: float) -> AimDecision:
        """Advance the manoeuvre by ``dt`` seconds.

        Args:
            heading_error: Signed, normalized angle from the drone's current
                heading to the object's bearing (rad); ``+`` means the object is to
                the left (CCW). Only read when the policy re-measures -- at the end
                of a SETTLE -- so a noisy value mid-burst cannot disturb the turn.
            dt: Seconds since the previous update.

        Returns:
            The tick's :class:`AimDecision`.
        """
        dt = max(0.0, float(dt))
        if self._phase == DONE:                    # terminal: hold the stop
            return self._emit()

        self._t += dt
        self._elapsed += dt

        # The episode cap is checked before the phase logic so it fires even while a
        # burst is mid-flight: a drone that is not actually turning would otherwise
        # cycle TURN/SETTLE forever, each measurement as far off as the last.
        if self._elapsed >= self.cfg.timeout_s and self._phase != LOOK:
            self._timed_out = True
            self._enter(DONE)
            return self._emit()

        if self._phase == SETTLE and self._t >= self.cfg.settle_s:
            # Stopped and stationary: this is the one place the heading is read.
            if abs(float(heading_error)) <= self.cfg.tolerance_rad:
                self._enter(LOOK)
            else:
                self._start_burst(float(heading_error))
        elif self._phase == TURN and self._t >= self._burst_s:
            self._enter(SETTLE)
        elif self._phase == LOOK and self._t >= self.cfg.look_s:
            self._enter(DONE)
        return self._emit()

    # ── helpers ───────────────────────────────────────────────────────
    def _start_burst(self, heading_error: float) -> None:
        """Size and arm one yaw burst, aimed ``yaw_coast_rad`` short of the error."""
        cfg = self.cfg
        commanded = max(0.0, abs(heading_error) - cfg.yaw_coast_rad)
        burst_s = commanded / cfg.yaw_rate
        self._burst_s = min(max(burst_s, cfg.min_burst_s), cfg.max_burst_s)
        self._burst_sign = 1.0 if heading_error >= 0.0 else -1.0
        self._enter(TURN)

    def _enter(self, phase: str) -> None:
        self._phase = phase
        self._t = 0.0

    def _emit(self) -> AimDecision:
        if self._phase == TURN:
            cmd = ControlCommand.velocity(0.0, 0.0, 0.0,
                                          self._burst_sign * self.cfg.yaw_rate,
                                          source="aim", phase=self._phase)
        else:
            cmd = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0,
                                          source="aim", phase=self._phase)
        # A timeout means the heading never converged, so the nose is NOT on the
        # bearing even though the episode is over -- only a LOOK earns on_bearing.
        return AimDecision(command=cmd, phase=self._phase,
                           on_bearing=(self._phase == LOOK
                                       or (self._phase == DONE and not self._timed_out)),
                           finished=self._phase == DONE,
                           timed_out=self._timed_out)
