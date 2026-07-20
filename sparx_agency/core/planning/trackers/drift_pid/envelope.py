"""The force envelope: what this airframe is allowed to be asked for.

Everything the controller computes passes through here last. The envelope owns
four separate jobs that are easy to confuse and must happen in this order:

  1. **Per-axis maximum** — no single axis may exceed its own cap.
  2. **Combined budget** — the platform is markedly faster when several axes are
     driven at once (the XTEND controller is a free-floating wand; any wrist
     angle already blends pitch/roll/yaw, and the airframe adds those demands
     rather than sharing them out). So the *sum* of normalized per-axis demand is
     capped too, and everything is scaled down together when it is exceeded. A
     forward+lateral+yaw command that each look legal on their own can be a much
     faster drone than any of them alone; this is the dial that stops that.
  3. **Slew** — no axis may change faster than its acceleration limit. This is
     what makes a correction "prolonged" rather than a jab, and it is what keeps
     the depth model usable: DA3 runs at ~2.5 Hz and degrades in fast motion and
     sharp turns, so the controller must not produce either.
  4. **Minimum force** — below a floor the motors simply ignore the command, so a
     sub-floor demand is either snapped up to the floor or dropped to zero.

The ordering is load-bearing and mirrors ``multi_axis_follower.follower._finalize``:
saturate, then slew (remembering the *continuous, unshaped* value so the ramp
stays smooth), then shape. Shaping last means the published command is never a
dribble the motors ignore; keeping the pre-shape value in the slew memory means
the snap-to-floor does not become the starting point of the next ramp.

Body frame is REP-103: ``+vx`` forward, ``+vy`` left, ``+wz`` counter-clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import copysign
from typing import Optional, Tuple

from sparx_agency.core.planning.trackers.multi_axis_follower.allocation import (
    saturate,
    saturate_translation,
    shape_axis,
    slew,
)


@dataclass(frozen=True)
class EnvelopeParams:
    """Per-axis and combined limits for :class:`ForceEnvelope` (SI units).

    Attributes:
        max_vx: Cap on forward speed (m/s).
        max_vx_back: Cap on reverse speed (m/s). Deliberately harder than
            forward: the drone has no rear camera, so every backward metre is
            flown blind. Reverse is only ever an escape move — breaking contact —
            never a way to travel.
        max_vy: Cap on lateral (ROLL / crab) speed (m/s). Usually well below
            ``max_vx``: lateral authority is weaker and it moves the drone
            sideways into space the forward-facing camera has not seen.
        max_wz: Cap on yaw rate (rad/s).
        max_translation: Cap on the magnitude of the ``(vx, vy)`` vector (m/s).
            Stops a diagonal command from being faster than either axis alone.
        combined_effort: Cap on ``|vx|/max_vx + |vy|/max_vy + |wz|/max_wz``. 1.0
            means "one axis at full, or two at half each" — a strict budget. Above
            ~2.0 it stops binding for two axes. This is the multi-axis speed-up
            dial: lower it if the drone feels fast when it turns while flying.
        min_vx: Smallest forward speed that actually moves the platform (m/s).
        min_vy: Smallest lateral speed that actually moves the platform (m/s).
        min_wz: Smallest yaw rate that actually rotates the platform (rad/s).
        release_frac: A per-axis command below ``release_frac * min_*`` is dropped
            to zero; between there and ``min_*`` it is snapped up. 0..1.
        cmd_zero_eps: Magnitude treated as exactly zero (numerical dust guard).
        accel_xy: Slew limit on the translation axes while the demand GROWS
            (m/s^2). Low on purpose: a heavy drone ramping gently is what keeps
            the ~2.5 Hz depth model fed with usable frames.
        decel_xy: Slew limit while the demand SHRINKS or reverses (m/s^2). Must
            be >= ``accel_xy``: the one asymmetry a heavy drone in a closed room
            always wants is to be able to take thrust off faster than it puts
            thrust on — braking for an obstacle must not be rate-limited by the
            same gentleness that shapes the ramp-up.
        accel_wz: Slew limit on the yaw axis while the demand grows (rad/s^2).
        decel_wz: Slew limit on the yaw axis while the demand shrinks (rad/s^2).
    """

    max_vx: float = 0.25
    max_vx_back: float = 0.12
    max_vy: float = 0.12
    max_wz: float = 0.40
    max_translation: float = 0.25
    combined_effort: float = 1.4

    min_vx: float = 0.06
    min_vy: float = 0.06
    min_wz: float = 0.175
    release_frac: float = 0.5
    cmd_zero_eps: float = 1e-3

    accel_xy: float = 0.35
    decel_xy: float = 0.60
    accel_wz: float = 1.2
    decel_wz: float = 2.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the envelope relies on."""
        for name in ("max_vx", "max_vx_back", "max_vy", "max_wz",
                     "max_translation", "combined_effort", "min_vx", "min_vy",
                     "min_wz", "accel_xy", "decel_xy", "accel_wz", "decel_wz"):
            if getattr(self, name) <= 0.0:
                raise ValueError("EnvelopeParams." + name + " must be > 0")
        if not 0.0 <= self.release_frac <= 1.0:
            raise ValueError("EnvelopeParams.release_frac must be in [0, 1]")
        if self.max_translation < max(self.max_vx, self.max_vy):
            raise ValueError(
                "EnvelopeParams.max_translation (%.2f) is below max_vx (%.2f) / "
                "max_vy (%.2f) -- the per-axis cap would be unreachable, so the "
                "dial would silently do nothing"
                % (self.max_translation, self.max_vx, self.max_vy))
        if self.max_vx_back > self.max_vx:
            raise ValueError(
                "EnvelopeParams.max_vx_back (%.2f) exceeds max_vx (%.2f) -- "
                "reverse is flown blind and must never be the faster direction"
                % (self.max_vx_back, self.max_vx))
        if self.min_vx >= self.max_vx_back:
            raise ValueError(
                "EnvelopeParams.min_vx (%.3f) must be below max_vx_back (%.3f) "
                "-- otherwise every reverse command snaps to the floor and the "
                "escape's back-off has exactly one speed" % (self.min_vx,
                                                             self.max_vx_back))
        if self.decel_xy < self.accel_xy or self.decel_wz < self.accel_wz:
            raise ValueError(
                "EnvelopeParams.decel_* must be >= accel_* -- braking may never "
                "be slower than accelerating")
        for axis, floor, ceiling in (("vx", self.min_vx, self.max_vx),
                                     ("vy", self.min_vy, self.max_vy),
                                     ("wz", self.min_wz, self.max_wz)):
            if floor >= ceiling:
                raise ValueError(
                    "EnvelopeParams.min_%s (%.3f) must be below max_%s (%.3f) -- "
                    "otherwise every command on the axis snaps to the floor and "
                    "the axis has exactly one speed" % (axis, floor, axis, ceiling))


class ForceEnvelope:
    """Stateful limiter: turns a desired body velocity into a flyable command."""

    def __init__(self, params=None):
        # type: (Optional[EnvelopeParams]) -> None
        self.params = params or EnvelopeParams()
        self.reset()

    def reset(self):
        # type: () -> None
        """Clear the slew memory (call on a fresh path or after a stop)."""
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0

    def effort(self, vx, vy, wz):
        # type: (float, float, float) -> float
        """Normalized total demand: 1.0 is one axis at its own maximum.

        Exposed so a caller can log or publish how much of the airframe's budget
        the current command is spending.
        """
        p = self.params
        return (abs(vx) / p.max_vx + abs(vy) / p.max_vy + abs(wz) / p.max_wz)

    def apply(self, vx, vy, wz, dt, speed_scale=1.0):
        # type: (float, float, float, float, float) -> Tuple[float, float, float]
        """Clamp, budget, slew and shape one desired body velocity.

        Args:
            vx: Desired forward speed (m/s).
            vy: Desired lateral speed (m/s, + left).
            wz: Desired yaw rate (rad/s, + CCW).
            dt: Seconds since the previous call (drives the slew limit).
            speed_scale: Multiplier applied to every per-axis cap before
                clamping (0..1), so a low-confidence pose flies the same shape of
                command, slower. The minimum-force floors are NOT scaled — they
                are a property of the motors, not of the plan.

        Returns:
            ``(vx, vy, wz)`` ready to publish.
        """
        if dt <= 0.0:
            raise ValueError("ForceEnvelope.apply: dt must be > 0")
        p = self.params
        s = min(1.0, max(0.0, float(speed_scale)))

        # 1. per-axis caps, scaled by confidence. Forward and reverse are
        #    asymmetric on purpose: reverse is flown blind.
        vx = float(vx)
        if vx >= 0.0:
            vx = min(vx, p.max_vx * s)
        else:
            vx = max(vx, -p.max_vx_back * s)
        vy = saturate(float(vy), p.max_vy * s)
        wz = saturate(float(wz), p.max_wz * s)
        vx, vy = saturate_translation(vx, vy, p.max_translation * s)

        # 2. combined budget -- scale ALL axes together so the command keeps its
        #    direction and only loses magnitude.
        demand = self.effort(vx, vy, wz)
        if demand > p.combined_effort and demand > 0.0:
            k = p.combined_effort / demand
            vx, vy, wz = vx * k, vy * k, wz * k

        # 3. slew, remembering the continuous (pre-shape) value. Two rates per
        #    axis: a demand moving AWAY from zero ramps at accel (gentle, keeps
        #    the depth model fed); a demand shrinking or reversing ramps at decel
        #    (braking is never rate-limited by the ramp-up's gentleness).
        vx = slew(vx, self._vx, self._step(vx, self._vx, p.accel_xy, p.decel_xy, dt))
        vy = slew(vy, self._vy, self._step(vy, self._vy, p.accel_xy, p.decel_xy, dt))
        wz = slew(wz, self._wz, self._step(wz, self._wz, p.accel_wz, p.decel_wz, dt))
        self._vx, self._vy, self._wz = vx, vy, wz

        # 4. minimum force, last
        vx = shape_axis(vx, p.min_vx, p.release_frac, p.cmd_zero_eps)
        vy = shape_axis(vy, p.min_vy, p.release_frac, p.cmd_zero_eps)
        wz = shape_axis(wz, p.min_wz, p.release_frac, p.cmd_zero_eps)
        return vx, vy, wz

    @staticmethod
    def _step(target, current, accel, decel, dt):
        # type: (float, float, float, float, float) -> float
        """Slew step for this axis and tick: accel growing, decel shrinking.

        "Shrinking" is any demand whose magnitude drops or whose sign flips —
        both are the drone taking thrust off this axis.
        """
        braking = abs(target) < abs(current) or target * current < 0.0
        return (decel if braking else accel) * dt

    def brake(self, dt):
        # type: (float) -> Tuple[float, float, float]
        """Ramp every axis toward zero at the DECEL slew limit.

        A stop is a command like any other: snapping to zero throws the drone
        forward on its own inertia, which is exactly the motion that ruins the
        next depth frame. Use this instead of publishing a bare zero whenever the
        controller decides to stop but is not in an emergency.
        """
        return self.apply(0.0, 0.0, 0.0, dt)


def counts_for(velocity, ref_velocity, ref_counts, max_counts):
    # type: (float, float, float, float) -> float
    """Convert an axis velocity to the platform's raw axis counts.

    The XTEND virtual controller takes integer axis values, not SI velocities,
    and its calibration is a single reference point per axis (``ref_velocity``
    maps to ``ref_counts``) with a hard ceiling. This helper exists so a
    controller can report, in the operator's units, how much of the real stick
    travel a command is spending — the number that decides whether "0.25 m/s"
    is a gentle nudge or already at the stop.

    Args:
        velocity: Commanded velocity on the axis (m/s or rad/s).
        ref_velocity: Velocity the calibration point was taken at.
        ref_counts: Axis counts that produced ``ref_velocity``.
        max_counts: Hard ceiling on the axis.

    Returns:
        Signed axis counts, clamped to ``+-max_counts``.
    """
    if ref_velocity <= 0.0:
        raise ValueError("counts_for: ref_velocity must be > 0")
    counts = abs(float(velocity)) / ref_velocity * ref_counts
    if counts > max_counts:
        counts = max_counts
    return copysign(counts, velocity) if velocity else 0.0
