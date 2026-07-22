"""Per-axis minimum/maximum force shaping for closure (visual-servo) commands.

The platform has a **minimum effective command per axis**: below it the motors do
not move the drone at all, so a sub-threshold velocity is wasted. The visual servo
(:class:`~sparx_agency.core.planning.visual_servo.controller.VisualServoController`)
emits an analog body velocity capped only at the top, so on its own it can dribble
commands too weak to move the drone. This module applies the SAME per-axis force
discipline the multi-axis waypoint follower uses (it reuses
:func:`...trackers.multi_axis_follower.allocation.shape_axis`) so a closure command
either moves the drone or is exactly zero.

Three modes (``AxisForceProfile.mode``):

  * ``"none"``  — only clamp to ``max_magnitude`` (analog passthrough, no floor).
  * ``"snap"``  — deadband-with-snap: a command below ``release_frac*min_magnitude``
    is dropped to zero; between there and ``min_magnitude`` it is snapped UP to
    ``min_magnitude``; above it passes through, clamped to ``max_magnitude``. This
    matches the multi-axis follower exactly (respects both the min AND the max).
  * ``"fixed"`` — bang-bang: a command below the release threshold is dropped to
    zero; anything above is emitted at exactly ``±fixed_magnitude`` (which defaults
    to ``min_magnitude``). The only way to move the axis is one fixed-force pulse.

The controller is responsible for commanding exactly zero on an axis it does not
want to move (its own centring deadbands do this), so shaping never injects motion
the servo did not intend.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import copysign
from typing import Optional

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.planning.trackers.multi_axis_follower.allocation import (
    saturate,
    shape_axis,
)

#: The recognised shaping modes, in order of increasing coarseness.
FORCE_MODES = ("none", "snap", "fixed")


@dataclass(frozen=True)
class AxisForceProfile:
    """Force-shaping profile for a single body axis (SI units).

    Attributes:
        min_magnitude: Smallest command that actually moves this axis ("minimum
            force"). ``0`` disables the floor (equivalent to ``"none"``).
        max_magnitude: Upper clamp on ``|command|``; ``None`` leaves it uncapped
            (the caller has usually already applied a kinematic limit).
        release_frac: A command below ``release_frac * min_magnitude`` is dropped to
            zero (deadband). In ``[0, 1]``.
        zero_eps: Floor on the drop threshold, so numerical dust never triggers a
            spurious minimum-force pulse.
        mode: One of :data:`FORCE_MODES`.
        fixed_magnitude: Bang-bang level for ``"fixed"`` mode; defaults to
            ``min_magnitude`` when ``None``.
    """

    min_magnitude: float
    max_magnitude: Optional[float] = None
    release_frac: float = 0.5
    zero_eps: float = 1e-3
    mode: str = "fixed"
    fixed_magnitude: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in FORCE_MODES:
            raise ValueError("mode must be one of %r, got %r" % (FORCE_MODES, self.mode))
        if self.min_magnitude < 0.0:
            raise ValueError("min_magnitude must be >= 0")
        if not 0.0 <= self.release_frac <= 1.0:
            raise ValueError("release_frac must be in [0, 1]")
        if self.max_magnitude is not None and self.max_magnitude <= 0.0:
            raise ValueError("max_magnitude must be > 0 when given")
        if self.fixed_magnitude is not None and self.fixed_magnitude < 0.0:
            raise ValueError("fixed_magnitude must be >= 0 when given")

    @property
    def fixed_level(self) -> float:
        """The bang-bang command magnitude (``fixed_magnitude`` or ``min_magnitude``)."""
        return self.min_magnitude if self.fixed_magnitude is None else self.fixed_magnitude


def shape_axis_force(cmd: float, profile: AxisForceProfile) -> float:
    """Apply ``profile`` to one axis command and return the shaped value.

    See the module docstring for the three modes. In every mode a command whose
    magnitude is below the release threshold becomes exactly ``0.0``.
    """
    value = float(cmd)
    if profile.mode == "none":
        return value if profile.max_magnitude is None else saturate(value, profile.max_magnitude)
    if profile.mode == "snap":
        shaped = shape_axis(value, profile.min_magnitude, profile.release_frac,
                            profile.zero_eps)
        return shaped if profile.max_magnitude is None else saturate(shaped, profile.max_magnitude)
    # "fixed" — bang-bang: zero below the deadband, else a single fixed-force pulse.
    drop = max(profile.zero_eps, profile.release_frac * profile.min_magnitude)
    if abs(value) <= drop:
        return 0.0
    level = profile.fixed_level
    if profile.max_magnitude is not None:
        level = min(level, profile.max_magnitude)
    return copysign(level, value)


@dataclass(frozen=True)
class CommandForceShaper:
    """Apply per-axis :class:`AxisForceProfile` shaping to a body-velocity command.

    The forward (``vx``), lateral (``vy``) and yaw (``wz``) axes are always shaped;
    the vertical axis (``vz``) is shaped only when ``vz`` is given, otherwise it is
    passed through untouched (closure holds altitude by default).
    """

    vx: AxisForceProfile
    vy: AxisForceProfile
    wz: AxisForceProfile
    vz: Optional[AxisForceProfile] = None

    def shape(self, command: ControlCommand) -> ControlCommand:
        """Return a new VELOCITY command with each axis force-shaped.

        The original command's ``metadata`` is preserved so downstream diagnostics
        (e.g. ``source``/``mode``) survive the shaping stage.
        """
        z = command.z if self.vz is None else shape_axis_force(command.z, self.vz)
        return ControlCommand.velocity(
            shape_axis_force(command.x, self.vx),
            shape_axis_force(command.y, self.vy),
            z,
            shape_axis_force(command.yaw_rate, self.wz),
            **dict(command.metadata),
        )
