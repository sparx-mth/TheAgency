"""The autopilot underneath, described well enough to be inverted.

An airframe that accepts a velocity setpoint already contains a controller. It
is not a free integrator and it is not instantaneous: it takes a while to notice
the command, and then a while to reach it. Ignoring that is the single largest
source of tracking error on this class of vehicle, and it does not look like
lag -- it looks like a controller that is mysteriously always behind, which
invites raising the position gain until the loop rings.

Every axis is modelled as a first-order lag behind a pure transport delay:

.. code-block:: text

                  -Ls
    v_actual     e
    --------  =  ---------
    v_command    tau s + 1

Three numbers per axis, all of them **measurable in twenty seconds** with a step
command and a log of the odometry: the DC gain is the steady-state ratio, the
delay is how long nothing happens, and the time constant is how long the
remainder takes to reach 63% of its final value. Measure them; do not assume
them. The defaults below are a representative indoor quadrotor and are wrong for
any specific one.

**Why this model and not a better one.** The true response is second order and
mildly oscillatory. A first-order-plus-delay fit is nevertheless the right
choice here, because the only use made of it is a lead term that cancels the
dominant pole, and cancelling a pole you have measured to 10% is worth far more
than modelling a second pole you would then have to differentiate the reference
twice to use. The delay is not cancellable at all -- no causal controller can --
so it survives as the thing that sets the loop's bandwidth ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AxisPlant:
    """First-order-plus-delay response of one velocity axis.

    Attributes:
        dc_gain: Steady-state ``measured / commanded`` velocity. Near 1.0 for an
            autopilot that closes its own velocity loop; well below it for one
            that merely maps the setpoint to a tilt.
        time_constant_s: Seconds to reach 63% of the final value, once moving.
            This is the number the lead term cancels.
        delay_s: Seconds between the command being sent and anything happening.
            Transport, message queueing and the attitude loop's own rise, lumped
            together. Sets the bandwidth ceiling: a position loop crossing over
            much above ``1 / (3 * delay_s)`` will ring.
    """

    dc_gain: float = 1.0
    time_constant_s: float = 0.5
    delay_s: float = 0.15

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the inversion relies on."""
        if self.dc_gain <= 0.0:
            raise ValueError("dc_gain must be > 0, got %r" % (self.dc_gain,))
        if self.time_constant_s < 0.0:
            raise ValueError("time_constant_s must be >= 0, got %r" % (self.time_constant_s,))
        if self.delay_s < 0.0:
            raise ValueError("delay_s must be >= 0, got %r" % (self.delay_s,))

    @property
    def stable_position_gain(self):
        # type: () -> float
        """The largest position gain this axis can carry without ringing, 1/s.

        A position loop closed around a velocity servo crosses over at roughly
        its own gain. The delay contributes ``omega * delay`` of phase lag, so
        holding that below about 20 degrees at crossover -- the usual margin for
        a loop that must not overshoot into a wall -- gives
        ``omega_c <= 1 / (3 * delay)``.

        **The delay alone sets this, and the time constant deliberately does
        not appear.** A first-order lag closed inside a position loop is a
        second-order system with no phase crossover: it never reaches -180
        degrees, so it is stable at any gain and merely becomes progressively
        less damped. Only the transport delay can actually turn the loop
        unstable, which is why it is the one term here. The ``1e-3`` floor
        exists so a plant declared with no delay returns a large finite number
        rather than dividing by zero; it is not a physical bound, and a
        near-zero delay should be read as "this model does not constrain the
        gain" rather than as permission to use the value it returns.

        Advisory: nothing enforces it. It is here so a tuning decision can be
        checked against the plant rather than against taste.
        """
        budget = max(self.delay_s, 1e-3)
        return 1.0 / (3.0 * budget)


def _horizontal_default():
    # type: () -> AxisPlant
    """Horizontal translation: the slowest axis, because it goes through tilt."""
    return AxisPlant(dc_gain=1.0, time_constant_s=0.5, delay_s=0.18)


def _vertical_default():
    # type: () -> AxisPlant
    """Climb and descent: faster, because thrust changes without rotating."""
    return AxisPlant(dc_gain=1.0, time_constant_s=0.4, delay_s=0.05)


def _yaw_default():
    # type: () -> AxisPlant
    """Heading rate: its own loop, and usually the best damped of the four."""
    return AxisPlant(dc_gain=1.0, time_constant_s=0.5, delay_s=0.06)


@dataclass(frozen=True)
class VelocityPlant:
    """The velocity response of the autopilot this controller sits on top of.

    Horizontal is one entry rather than two because the airframe is symmetric in
    its own horizontal plane, and a controller that believed otherwise would
    steer differently depending on which way the aircraft happened to be facing.

    Attributes:
        horizontal: Response of both body-horizontal axes.
        vertical: Response of the climb axis.
        yaw: Response of the heading-rate axis.
    """

    horizontal: AxisPlant = field(default_factory=_horizontal_default)
    vertical: AxisPlant = field(default_factory=_vertical_default)
    yaw: AxisPlant = field(default_factory=_yaw_default)

    @property
    def feedforward_lead_s(self):
        # type: () -> float
        """How far ahead the plan must be read for the aircraft to arrive on time.

        The horizontal delay, because it is the one that dominates and because
        the feedforward is sampled from a single point on the curve rather than
        per axis -- reading position and altitude off two different instants of
        the same trajectory would bend it.
        """
        return self.horizontal.delay_s
