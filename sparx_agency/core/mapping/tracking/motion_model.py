"""Constant-velocity motion model for a tracked box centre (alpha-beta filter).

Fuses noisy per-frame box measurements into a smooth centre estimate **and** an
image-plane velocity. Two jobs:

  1. **Predict through brief dropouts.** When the LK tracker loses lock for a few
     frames (motion blur, a fast target, a thin occluder), :meth:`predict` carries
     the box forward along its last velocity so the servo keeps a sensible target
     for a short window instead of snapping to a hover.
  2. **Point the re-search.** The velocity it estimates is exactly the "which way
     did the target go" signal the recovery policy needs to yaw back toward a lost
     target rather than guessing.

Alpha-beta on the centre (position + velocity); a slower EMA on the box size,
whose frame-to-frame change is noisier and carries no useful velocity. Pure numpy
scalar math — ROS-free, cv2-free, Python-3.8-safe, and cheap enough to run every
frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

CxCyWh = Tuple[float, float, float, float]


@dataclass(frozen=True)
class MotionModelConfig:
    """Tuning for :class:`ConstantVelocityBoxModel`.

    Attributes:
        alpha: Position blend for the centre (0..1); higher trusts measurements.
        beta: Velocity blend for the centre (0..1); higher reacts faster but noisier.
        size_alpha: EMA blend for the box width/height (0..1).
        max_speed_px: Clamp on the estimated centre speed (px/s), rejects flow spikes.
        max_dt: Any ``dt`` above this (a long stall) is treated as a fresh anchor
            rather than integrated, so velocity does not explode after a gap.
    """

    alpha: float = 0.5
    beta: float = 0.2
    size_alpha: float = 0.4
    max_speed_px: float = 1500.0
    max_dt: float = 0.5

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "size_alpha"):
            v = getattr(self, name)
            if not (0.0 < v <= 1.0):
                raise ValueError("%s must be in (0, 1], got %r" % (name, v))
        if self.max_speed_px <= 0:
            raise ValueError("max_speed_px must be > 0.")


class ConstantVelocityBoxModel:
    """Alpha-beta constant-velocity estimator over a box centre (+ EMA size)."""

    def __init__(self, config: Optional[MotionModelConfig] = None) -> None:
        self.cfg = config or MotionModelConfig()
        self.reset()

    def reset(self) -> None:
        """Drop all state (no estimate until the next :meth:`update`)."""
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._w: float = 0.0
        self._h: float = 0.0
        self._vx: float = 0.0
        self._vy: float = 0.0

    @property
    def has_state(self) -> bool:
        """True once at least one measurement has been absorbed."""
        return self._cx is not None

    @property
    def velocity_px(self) -> Tuple[float, float]:
        """Estimated centre velocity ``(vx, vy)`` in px/s (image frame)."""
        return (self._vx, self._vy)

    def state_cxcywh(self) -> Optional[CxCyWh]:
        """Current filtered box as ``(cx, cy, w, h)``, or None if unset."""
        if self._cx is None:
            return None
        return (self._cx, self._cy, self._w, self._h)

    def update(self, cxcywh: CxCyWh, dt: float) -> CxCyWh:
        """Absorb a fresh measurement and return the filtered box.

        Args:
            cxcywh: Measured ``(cx, cy, w, h)`` this frame.
            dt: Seconds since the previous update (ignored on the first call).

        Returns:
            The filtered ``(cx, cy, w, h)``.
        """
        mcx, mcy, mw, mh = (float(v) for v in cxcywh)
        dt = float(dt)

        if self._cx is None or dt > self.cfg.max_dt:
            # First measurement, or a gap too long to integrate: re-anchor.
            self._cx, self._cy = mcx, mcy
            self._w, self._h = mw, mh
            self._vx, self._vy = 0.0, 0.0
            return (self._cx, self._cy, self._w, self._h)

        if dt <= 0.0:
            # Duplicate/equal timestamp on an established track: absorb the
            # measured position (no dt to estimate velocity with) but KEEP the
            # velocity estimate rather than wiping it to zero.
            self._cx += self.cfg.alpha * (mcx - self._cx)
            self._cy += self.cfg.alpha * (mcy - self._cy)
            a = self.cfg.size_alpha
            self._w = (1.0 - a) * self._w + a * mw
            self._h = (1.0 - a) * self._h + a * mh
            return (self._cx, self._cy, self._w, self._h)

        # Predict then correct (alpha-beta) on the centre.
        px = self._cx + self._vx * dt
        py = self._cy + self._vy * dt
        rx, ry = mcx - px, mcy - py
        self._cx = px + self.cfg.alpha * rx
        self._cy = py + self.cfg.alpha * ry
        self._vx += (self.cfg.beta / dt) * rx
        self._vy += (self.cfg.beta / dt) * ry
        self._clamp_speed()

        # Low-pass the size.
        a = self.cfg.size_alpha
        self._w = (1.0 - a) * self._w + a * mw
        self._h = (1.0 - a) * self._h + a * mh
        return (self._cx, self._cy, self._w, self._h)

    def predict(self, dt: float) -> Optional[CxCyWh]:
        """Advance the estimate by ``dt`` with no measurement (dead reckoning).

        Integrates the centre along its velocity, holds the size, and returns the
        predicted ``(cx, cy, w, h)`` — or None if there is no state yet. Also
        mutates the internal state so repeated calls chain.
        """
        if self._cx is None:
            return None
        dt = max(0.0, float(dt))
        self._cx += self._vx * dt
        self._cy += self._vy * dt
        return (self._cx, self._cy, self._w, self._h)

    # ── internals ────────────────────────────────────────────────────
    def _clamp_speed(self) -> None:
        speed = (self._vx * self._vx + self._vy * self._vy) ** 0.5
        if speed > self.cfg.max_speed_px:
            scale = self.cfg.max_speed_px / speed
            self._vx *= scale
            self._vy *= scale
