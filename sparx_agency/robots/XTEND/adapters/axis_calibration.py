"""XTEND axis calibration: SI velocities <-> virtual-controller axis counts.

The XTEND is flown by a **virtual controller**, not by a velocity interface. The
WebSocket protocol carries a 5-element ``axes`` array of integers, re-sent at
~30 Hz, and each command is *held* until it is changed:

======  ==================  =========================================
index   protocol name       what it does
======  ==================  =========================================
0       Joystick Horizontal lateral / ROLL   (+ = right)
1       Joystick Vertical   vertical         (+ = up)
2       Trigger             forward / PITCH  (+ = forward)
3       Marker Horizontal   yaw              (+ = turn right)
4       Marker Vertical     unused for navigation
======  ==================  =========================================

Every axis is hard-clamped to ``+-1000`` by the bridge, so **1000 counts is the
absolute physical maximum force on any axis** — there is nothing above it.

Turning a velocity into counts needs a calibration, and the calibration is a
single measured reference point per axis: "``ref_velocity`` came out of
``ref_counts``". Two facts are worth stating plainly because they are easy to
misread:

* **The forward axis is capped at 600, not 1000, by convention** — that ceiling
  is a policy choice about how fast this drone should ever be asked to fly
  indoors, not a protocol limit. Raising ``max_counts`` to 1000 raises the top
  speed by two thirds.
* **The lateral and vertical axes have never been calibrated separately.** They
  historically borrowed the forward axis's numbers, which assumes ROLL and PITCH
  authority are equal. On a drone with a forward-facing payload they usually are
  not. The defaults below preserve the old behaviour exactly, but the fields are
  now separate so a measured lateral calibration can be dropped in without
  touching the conversion code. If sideways moves feel weaker or stronger than
  forward moves at the same commanded m/s, this is the reason and this is the fix.

Nothing in this module talks to ROS or to the network; it is the arithmetic only.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Hard protocol clamp applied to every axis by the bridge.
AXIS_ABS_MAX = 1000


@dataclass(frozen=True)
class AxisCalibration:
    """Calibration for one axis: one reference point plus its limits.

    Attributes:
        ref_velocity: Velocity the reference point was measured at (m/s, or rad/s
            for yaw). Must be > 0.
        ref_counts: Axis counts that produced ``ref_velocity``.
        max_counts: Largest count this axis may ever be commanded. Policy, not
            protocol — the protocol allows up to ``AXIS_ABS_MAX``.
        min_counts: Smallest count that actually moves the platform. Below this
            the motors ignore the command, so a controller should either commit to
            this floor or command zero. 0 disables the floor.
    """

    ref_velocity: float
    ref_counts: int
    max_counts: int
    min_counts: int = 0

    def __post_init__(self):
        # type: () -> None
        """Validate the calibration is usable."""
        if self.ref_velocity <= 0.0:
            raise ValueError("AxisCalibration.ref_velocity must be > 0")
        if not 0 < self.ref_counts <= AXIS_ABS_MAX:
            raise ValueError("AxisCalibration.ref_counts must be in (0, %d]"
                             % AXIS_ABS_MAX)
        if not 0 < self.max_counts <= AXIS_ABS_MAX:
            raise ValueError("AxisCalibration.max_counts must be in (0, %d]"
                             % AXIS_ABS_MAX)
        if not 0 <= self.min_counts < self.max_counts:
            raise ValueError("AxisCalibration.min_counts must be in "
                             "[0, max_counts)")

    @property
    def max_velocity(self):
        # type: () -> float
        """Top velocity this axis can be commanded at, given ``max_counts``."""
        return self.ref_velocity * self.max_counts / float(self.ref_counts)

    @property
    def min_velocity(self):
        # type: () -> float
        """Smallest velocity that actually moves the platform (0 if no floor)."""
        return self.ref_velocity * self.min_counts / float(self.ref_counts)

    def to_counts(self, velocity):
        # type: (float) -> int
        """Convert a signed velocity to signed axis counts.

        Applies the linear calibration, the minimum-force floor (a non-zero
        command below the floor is snapped up to it rather than silently ignored
        by the motors) and the maximum clamp.
        """
        value = float(velocity)
        magnitude = abs(value) / self.ref_velocity * self.ref_counts
        if magnitude <= 0.0:
            return 0
        if self.min_counts and magnitude < self.min_counts:
            magnitude = self.min_counts
        counts = int(round(min(magnitude, self.max_counts)))
        return counts if value > 0.0 else -counts

    def to_velocity(self, counts):
        # type: (float) -> float
        """Convert signed axis counts back to a signed velocity."""
        return float(counts) / self.ref_counts * self.ref_velocity


@dataclass(frozen=True)
class XtendAxisCalibration:
    """The whole platform's axis calibration.

    Defaults reproduce the constants the converter has always used, so adopting
    this class changes no behaviour: forward 0.3 m/s at 400 counts capped at 600,
    yaw 0.65 rad/s at 1000 counts capped at 1000, and lateral/vertical inheriting
    the forward numbers.
    """

    forward: AxisCalibration = AxisCalibration(0.3, 400, 600)
    lateral: AxisCalibration = AxisCalibration(0.3, 400, 600)
    vertical: AxisCalibration = AxisCalibration(0.3, 400, 600)
    yaw: AxisCalibration = AxisCalibration(0.65, 1000, 1000)

    def describe(self):
        # type: () -> str
        """One line per axis: the envelope, in the units an operator thinks in."""
        rows = (("forward", self.forward, "m/s"), ("lateral", self.lateral, "m/s"),
                ("vertical", self.vertical, "m/s"), ("yaw", self.yaw, "rad/s"))
        return "\n".join(
            "%-8s %.3f..%.3f %-5s  (counts %d..%d of %d)"
            % (name, cal.min_velocity, cal.max_velocity, unit,
               cal.min_counts, cal.max_counts, AXIS_ABS_MAX)
            for name, cal, unit in rows)


#: The calibration in force today. Import this rather than hardcoding constants.
XTEND_CALIBRATION = XtendAxisCalibration()
