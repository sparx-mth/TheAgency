"""Stateful minimum-burst + coast-brake shaper for closure commands.

Models the real platform's *discrete, inertial* actuation for the visual-servo /
recovery / scan commands, the same reality the one-axis waypoint follower already
handles for the A*/NavDP route: the drone moves at a **fixed speed per axis**, a
**lone control tick cannot overcome its deadband/inertia** (so any motion must last
at least ``min_burst_ticks`` consecutive ticks to actually register — ~2 at the
route follower's 10 Hz), and it **coasts** after a command stops.

The memoryless :class:`~...force_shaping.CommandForceShaper` already gives the
fixed-speed-per-command part (its ``"fixed"`` mode emits 0 or ±level). This shaper
adds the two things that need *state* across ticks:

  * **Minimum burst.** When an axis starts moving it is LATCHED for at least
    ``min_burst_ticks`` ticks. A small correction the servo would otherwise dribble
    for one tick — which the platform ignores entirely — becomes a real ≥2-tick
    burst that moves it. (Big corrections already last many ticks, so the latch is
    inert there.)
  * **Coast brake (optional).** When a *sustained* burst ends, a brief opposite
    pulse (``brake_ticks``) can be emitted to bleed off the inertial coast instead
    of overshooting. Off by default (``brake_ticks=0``): the primary coast handling
    is the servo stopping *early* (its coast-aware yaw deadband), and an opposite
    pulse on a coarse-yaw platform is only worth it when tuned per airframe.

One instance per mission, stepped every control tick via :meth:`shape`. The fixed
level and deadband per axis come from the same :class:`~...force_shaping.AxisForceProfile`
the force shaper uses, so magnitudes stay consistent. ROS-free, Python-3.8-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import copysign
from typing import Optional

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.planning.visual_servo.force_shaping import (
    AxisForceProfile,
    shape_axis_force,
)


def _sign(v: float) -> float:
    return 0.0 if v == 0.0 else copysign(1.0, v)


class _AxisPulseState:
    """Per-axis burst/brake counters (mutable)."""

    __slots__ = ("active", "sign", "brake")

    def __init__(self) -> None:
        self.active = 0        # consecutive ticks moving in the current burst
        self.sign = 0.0        # current burst direction (+1/-1), 0 when stopped
        self.brake = 0         # remaining brake ticks

    def clear(self) -> None:
        self.active = 0
        self.sign = 0.0
        self.brake = 0


class PulseShaper:
    """Fixed-speed pulses with a minimum burst and an optional coast brake.

    Args:
        vx / vy / wz: Per-axis force profiles (fixed level + deadband); reuse the
            same profiles as :class:`CommandForceShaper`.
        min_burst_ticks: Consecutive ticks any axis motion is held for (>= 1). 2
            mirrors the route follower's ``min_motion_ticks`` at 10 Hz.
        brake_ticks: Opposite-pulse ticks emitted after a sustained burst stops
            (>= 0). 0 disables braking.
        vz: Optional vertical profile; ``None`` passes ``vz`` through untouched.
    """

    def __init__(self, vx: AxisForceProfile, vy: AxisForceProfile,
                 wz: AxisForceProfile, min_burst_ticks: int = 2,
                 brake_ticks: int = 0, vz: Optional[AxisForceProfile] = None) -> None:
        if min_burst_ticks < 1:
            raise ValueError("min_burst_ticks must be >= 1.")
        if brake_ticks < 0:
            raise ValueError("brake_ticks must be >= 0.")
        self.vx, self.vy, self.wz, self.vz = vx, vy, wz, vz
        self.min_burst_ticks = int(min_burst_ticks)
        self.brake_ticks = int(brake_ticks)
        self.reset()

    def reset(self) -> None:
        """Forget all burst/brake state (e.g. on a mission restart)."""
        self._sx = _AxisPulseState()
        self._sy = _AxisPulseState()
        self._sw = _AxisPulseState()

    def shape(self, command: ControlCommand) -> ControlCommand:
        """Return a new VELOCITY command with each axis min-burst/brake shaped."""
        vx = self._axis(command.x, self.vx, self._sx)
        vy = self._axis(command.y, self.vy, self._sy)
        wz = self._axis(command.yaw_rate, self.wz, self._sw)
        z = command.z if self.vz is None else shape_axis_force(command.z, self.vz)
        return ControlCommand.velocity(vx, vy, z, wz, **dict(command.metadata))

    # ── per-axis discrete-pulse law ──────────────────────────────────
    def _axis(self, desired: float, profile: AxisForceProfile,
              st: _AxisPulseState) -> float:
        pulse = shape_axis_force(desired, profile)   # 0 or ±fixed level (deadbanded)
        level = abs(pulse) if pulse != 0.0 else profile.fixed_level
        if profile.max_magnitude is not None:
            level = min(level, profile.max_magnitude)

        if pulse != 0.0:                              # servo wants to move this axis
            s = _sign(pulse)
            if st.sign != s:                          # new direction -> fresh burst
                st.active = 0
            st.sign = s
            st.active += 1
            st.brake = 0
            return copysign(level, s)

        # servo wants to stop this axis
        if 0 < st.active < self.min_burst_ticks:      # finish the minimum burst
            st.active += 1
            return copysign(level, st.sign)
        if st.active >= self.min_burst_ticks:         # a real burst just ended
            st.brake = self.brake_ticks
            st.active = 0
        if st.brake > 0:                              # brake pulse (opposite), then stop
            st.brake -= 1
            out = -copysign(level, st.sign)
            if st.brake == 0:
                st.sign = 0.0
            return out
        st.sign = 0.0
        return 0.0
