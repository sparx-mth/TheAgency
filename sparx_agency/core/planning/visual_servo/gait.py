"""Pulse-and-settle gait for closure commands (pure, ROS-free).

A post-filter on the already force-shaped closure command (see
:class:`~sparx_agency.core.planning.visual_servo.pulse_shaper.PulseShaper`). Where
the shaper decides the fixed per-axis *magnitude* of each pulse, this gait decides
the *cadence*: it lets a short burst of motion through and then forces a brief
STOP so the drone coasts to rest and the camera gets a fresh look at the object
(and its bbox) before the next burst. It turns the servo's otherwise continuous,
"too strong" stream into a **move-a-little / stop-and-look / move-a-little** rhythm.

Two behaviours, both tuned in control ticks (sized for ``~ctrl_hz``):

  * **Duty cycle.** At most ``move_ticks`` consecutive motion ticks are emitted;
    then ``settle_ticks`` ticks are forced to zero (a real stop). So a turn is not
    one strong continuous sweep but a series of small turns, each followed by a
    pause to re-observe — the same discipline the room-scan sweep and the one-axis
    waypoint follower use, applied to the visual servo.
  * **Transition settle.** When the motion *category* changes — e.g. a yaw turn
    giving way to a forward advance — a settle is inserted first, so the platform
    is never asked to swap a turn straight into forward flight without stopping.

The motion "category" is the sign triple of ``(vx, vy, wz)``; an all-zero command
is a natural stop and passes through untouched (it does not consume a burst). The
filter is stateful (one instance per mission, stepped every control tick) and is
deliberately independent of the shaper: it only masks the shaped command to zero
while settling, so the shaper's own state keeps running underneath and motion
resumes cleanly when the settle ends. ROS-free, Python-3.8-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sparx_agency.core.common.types import ControlCommand

MOVING = "MOVING"
SETTLING = "SETTLING"

#: The sign triple of a stopped command.
_STOP_CAT = (0, 0, 0)


@dataclass(frozen=True)
class ClosureGaitConfig:
    """Tuning for :class:`ClosureGait` (all durations in control ticks).

    Attributes:
        move_ticks: Maximum consecutive motion ticks per burst before a forced
            settle (>= 1). Keep it >= the shaper's ``min_burst_ticks`` so each burst
            is still a real, registerable move. At ~10 Hz, 2-3 ticks is a small,
            deliberate increment.
        settle_ticks: Ticks held at zero (a real stop) after a burst, and when the
            motion category changes. ``0`` disables the whole gait (motion passes
            through continuously). Size it long enough for the drone to coast to
            rest and the detector to publish a fresh bbox (~0.3-0.6 s).
        settle_on_axis_change: Insert a settle when the motion category changes
            (e.g. turn -> forward), so a turn never swaps straight into forward
            flight without a stop.
        zero_eps: Magnitude below which an axis counts as stopped (dust guard).
        enabled: Master switch; ``False`` passes every command through untouched.
    """

    move_ticks: int = 2
    settle_ticks: int = 4
    settle_on_axis_change: bool = True
    zero_eps: float = 1e-3
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.move_ticks < 1:
            raise ValueError("move_ticks must be >= 1.")
        if self.settle_ticks < 0:
            raise ValueError("settle_ticks must be >= 0.")
        if self.zero_eps < 0.0:
            raise ValueError("zero_eps must be >= 0.")

    @property
    def active(self) -> bool:
        """True when the gait actually gates (enabled and a non-zero settle)."""
        return self.enabled and self.settle_ticks > 0


class ClosureGait:
    """Move-a-little / stop-and-look cadence filter for closure commands."""

    def __init__(self, config: Optional[ClosureGaitConfig] = None) -> None:
        self.cfg = config or ClosureGaitConfig()
        self.reset()

    def reset(self) -> None:
        """Forget all cadence state (e.g. on a fresh approach / mission restart)."""
        self._state = MOVING
        self._move = 0            # consecutive motion ticks emitted in this burst
        self._settle = 0          # ticks elapsed in the current settle
        self._cat = _STOP_CAT     # last emitted motion category

    @property
    def state(self) -> str:
        return self._state

    def step(self, command: ControlCommand) -> ControlCommand:
        """Return the gated command: the input while MOVING, a zero stop while
        SETTLING. Call once per control tick with the shaped closure command."""
        if not self.cfg.active:
            return command

        cat = self._category(command)

        if self._state == SETTLING:
            if self._settle < self.cfg.settle_ticks:
                self._settle += 1
                return self._stop(command)
            # Settle finished: resume moving THIS tick (fall through to MOVING).
            self._state = MOVING
            self._move = 0
            self._cat = _STOP_CAT

        # MOVING. A natural stop passes through and does not consume a burst.
        if cat == _STOP_CAT:
            self._move = 0
            self._cat = _STOP_CAT
            return command

        # A change of motion (turn <-> forward, or a reversal) stops first. This
        # stop is the first tick of the settle, so it is counted as such.
        if (self.cfg.settle_on_axis_change and self._cat != _STOP_CAT
                and cat != self._cat):
            self._enter_settle(counted=True)
            return self._stop(command)

        # Emit this motion tick; end the burst once it reaches move_ticks.
        self._move += 1
        self._cat = cat
        if self._move >= self.cfg.move_ticks:
            self._enter_settle()
        return command

    # ── helpers ──────────────────────────────────────────────────────
    def _enter_settle(self, counted: bool = False) -> None:
        """Begin a settle. ``counted`` marks this tick as the first zero already
        emitted (the transition path returns a stop on the entering tick, so it
        counts; the duty-cycle path emits the last MOTION tick, so it does not)."""
        self._state = SETTLING
        self._settle = 1 if counted else 0

    def _category(self, command: ControlCommand) -> Tuple[int, int, int]:
        eps = self.cfg.zero_eps
        return (_sign(command.x, eps), _sign(command.y, eps),
                _sign(command.yaw_rate, eps))

    @staticmethod
    def _stop(command: ControlCommand) -> ControlCommand:
        """A zero-velocity command that preserves the source metadata."""
        return ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, **dict(command.metadata))


def _sign(value: float, eps: float) -> int:
    if abs(value) < eps:
        return 0
    return 1 if value > 0.0 else -1
