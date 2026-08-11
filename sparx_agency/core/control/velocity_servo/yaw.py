"""Servo the heading, rather than hoping it arrives.

The acceleration backend passes the plan's heading straight through, rate
limited, and never closes a loop on it. On an airframe that takes an *attitude*
that is correct: the attitude command contains the heading, and the autopilot's
own attitude loop is what makes it true.

An airframe that takes a **yaw rate** has no such loop above it. Commanding the
plan's ``yaw_rate`` alone is open-loop integration: every millisecond of jitter,
every saturated turn and every dropped message is an error that is never
recovered, and the heading walks away from the plan over a flight. On an
exploration aircraft that matters more than it sounds, because FALCON chooses
heading to aim the depth camera at the frontier it means to observe next -- a
heading that has quietly drifted 20 degrees is a map built of the wrong wall.

So: feedforward the plan's yaw rate, and correct the accumulated heading error
with a proportional term.

.. code-block:: text

    yaw_rate = yaw_rate_plan  +  Kp * wrap(yaw_plan - yaw_measured)
               \\___________/     \\_________________________________/
               turn at the        undo whatever the turn did not
               planned rate       actually achieve

The gain is bounded by the same delay that bounds every other loop here; see
:attr:`~sparx_agency.core.control.velocity_servo.plant.AxisPlant.stable_position_gain`.
"""
from __future__ import annotations

from typing import Optional

from sparx_agency.core.common.types import normalize_angle


class YawServo:
    """Turns a planned heading and a measured one into a yaw-rate command.

    Stateful only in the rate it last commanded, which is what the slew limit is
    measured against. :meth:`reset` when the aircraft stops flying the plan.

    Args:
        gain: Proportional gain on heading error, rad/s per rad.
        max_rate: Ceiling on the commanded rate, rad/s.
        max_accel: Ceiling on how fast that command may change, rad/s^2.
        deadband_rad: Heading error below which the proportional term is not
            applied. Stops the aircraft dithering about a heading it has already
            reached, which on a camera-carrying airframe is visible as a shimmer
            in the map.
    """

    def __init__(self, gain=1.5, max_rate=1.5, max_accel=3.0, deadband_rad=0.01):
        # type: (float, float, float, float) -> None
        if gain < 0.0:
            raise ValueError("gain must be >= 0, got %r" % (gain,))
        for name, value in (("max_rate", max_rate), ("max_accel", max_accel)):
            if value <= 0.0:
                raise ValueError("%s must be > 0, got %r" % (name, value))
        if deadband_rad < 0.0:
            raise ValueError("deadband_rad must be >= 0, got %r" % (deadband_rad,))
        self.gain = float(gain)
        self.max_rate = float(max_rate)
        self.max_accel = float(max_accel)
        self.deadband_rad = float(deadband_rad)
        self._last_rate = 0.0
        self._error = 0.0

    def reset(self):
        # type: () -> None
        """Forget the last commanded rate, so the next tick is unconstrained."""
        self._last_rate = 0.0
        self._error = 0.0

    @property
    def error_rad(self):
        # type: () -> float
        """Heading error the last call acted on, wrapped to (-pi, pi]."""
        return self._error

    @property
    def commanded_rate(self):
        # type: () -> float
        """The rate last put on the wire, rad/s."""
        return self._last_rate

    def update(self, reference_yaw, reference_yaw_rate, measured_yaw, dt):
        # type: (Optional[float], float, float, float) -> float
        """Advance one tick.

        Args:
            reference_yaw: Heading the plan asks for, radians. None means the
                plan expresses no opinion, in which case only the feedforward is
                used and no error is accumulated.
            reference_yaw_rate: The plan's yaw rate, rad/s. Taken from the yaw
                curve's own derivative rather than a sampled ``yaw_dot``, so it
                is exact and carries no transport lag.
            measured_yaw: Measured heading, radians CCW from world +x.
            dt: Seconds since the previous call. Must be > 0.

        Returns:
            The yaw rate to command, rad/s, clamped and slew limited.

        Raises:
            ValueError: If ``dt`` is not positive.
        """
        if dt <= 0.0:
            raise ValueError("YawServo.update: dt must be > 0, got %r" % (dt,))
        wanted = float(reference_yaw_rate)
        if reference_yaw is None:
            self._error = 0.0
        else:
            # Wrapped, so a heading 179 degrees to the left is answered by
            # turning left rather than 181 degrees to the right.
            self._error = normalize_angle(float(reference_yaw) - float(measured_yaw))
            if abs(self._error) > self.deadband_rad:
                wanted += self.gain * self._error

        clamped = max(-self.max_rate, min(self.max_rate, wanted))
        step = self.max_accel * float(dt)
        slewed = max(self._last_rate - step, min(self._last_rate + step, clamped))
        self._last_rate = slewed
        return slewed
