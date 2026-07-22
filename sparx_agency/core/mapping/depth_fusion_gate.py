"""Rotation-aware depth-fusion gate (pure stdlib, ROS-free).

The *mapping* algorithm that decides, per depth frame, whether the frame may be
fused into the map. It exists because an in-place rotation corrupts mapping in
two distinct ways, and a plain freeze boolean only covers the first:

  1. **During the turn** depth and localization are unreliable, so NOTHING
     captured while turning may be fused (a hard freeze). The authoritative
     "are we turning?" signal is the system demo mode, decided by
     :class:`sparx_agency.core.mapping.sensor_freeze_policy.SensorFreezePolicy`.

  2. **On resume** the very first live frame the gate sees may be a frame that
     was *captured during the rotation* and merely delivered late (the mode
     topic and the depth topic have independent latencies, so the
     "stop turning" signal can overtake the last in-flight turn frames). Fusing
     it would smear the map with depth taken at a yaw the map no longer
     believes in. So after a turn the gate rejects every frame whose CAPTURE
     time is not strictly newer than the instant the turn ended, until a
     genuinely fresh frame arrives.

The handshake the gate enforces, end to end:

    fly straight  -> fuse every frame
    turn requested + confirmed (mode == turning) -> freeze: drop every frame
    turn finished (mode -> fly straight) -> arm a capture-time watermark
    resume: drop frames captured at/before the watermark (the stale in-flight
            turn frames), pass the first genuinely fresh frame, then fuse normally

The gate owns no I/O and no clock. ``should_fuse`` is fed each frame's capture
timestamp; ``note_mode`` / ``note_explicit`` are fed the freeze inputs together
with the current time (same clock as the capture stamps). Keeping it ROS-free
lets it sit unchanged whether the map is FALCON's external voxel grid (today,
via the sensor gate) or a map we fuse ourselves later — only the caller changes.

Clock contract: ``should_fuse``'s ``capture_stamp`` and the ``now`` passed to
``note_*`` MUST share one clock (in the FALCON pipeline both are the wall clock,
because depth is wall-stamped at the sensor adapter). The watermark is compared
against capture stamps, so a mismatch would reject or admit the wrong frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Optional, Tuple

from sparx_agency.core.mapping.sensor_freeze_policy import SensorFreezePolicy

#: ``should_fuse`` verdict labels (returned for diagnostics/logging).
FUSE = "fuse"
FROZEN_ROTATING = "frozen_rotating"
STALE_AFTER_ROTATION = "stale_after_rotation"


@dataclass
class DepthFusionGate:
    """Per-frame fuse/drop decision wrapping a freeze policy with a resume guard.

    Attributes:
        policy: The mode-authoritative freeze decision (turning => freeze).
        resume_settle_sec: Extra margin (seconds) added to the resume watermark.
            ``0`` admits a frame as soon as it is captured after the turn ends;
            a small positive value also rejects frames captured during the brief
            physical settle right after the turn is confirmed.
    """

    policy: SensorFreezePolicy = field(default_factory=SensorFreezePolicy)
    resume_settle_sec: float = 0.0

    # ── runtime state (not constructor args) ──
    _frozen: bool = field(init=False, default=False)
    _source: str = field(init=False, default="")
    # Newest capture stamp the gate has observed (tracked even while frozen, so
    # the resume watermark knows how far the turn's frames reached).
    _max_seen_stamp: float = field(init=False, default=-inf)
    # When set, only frames captured strictly after this stamp may be fused
    # (the stale-in-flight guard armed at the end of a turn). None once cleared.
    _resume_watermark: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._frozen, self._source = self.policy.decide()

    # ── freeze inputs (delegate to the policy, then re-evaluate the edge) ──
    def note_explicit(self, frozen: bool, now: Optional[float] = None) -> None:
        """Record an explicit freeze request and re-evaluate the freeze edge."""
        self.policy.note_explicit(frozen)
        self._reevaluate(now)

    def note_mode(self, is_turning_mode: bool, now: Optional[float] = None) -> None:
        """Record a demo-mode message and re-evaluate the freeze edge.

        Args:
            is_turning_mode: Whether the new mode equals the turning mode.
            now: Current time in the capture clock. Used to arm the resume
                watermark on the freeze->resume edge; pass it whenever the
                caller has a clock so a mode signal that overtakes late
                in-flight turn frames cannot let them through.
        """
        self.policy.note_mode(is_turning_mode)
        self._reevaluate(now)

    def reset_freeze(self, now: Optional[float] = None) -> None:
        """Manually clear the mode-based freeze (stuck-mode recovery)."""
        self.policy.reset_mode_freeze()
        self._reevaluate(now)

    # ── per-frame decision ──
    def should_fuse(self, capture_stamp: float) -> Tuple[bool, str]:
        """Return ``(fuse, reason)`` for a frame with this capture timestamp.

        Call this for EVERY incoming frame, even while frozen: it advances the
        newest-seen-stamp the resume watermark is built from. Fuse only when the
        returned bool is True.
        """
        if capture_stamp > self._max_seen_stamp:
            self._max_seen_stamp = capture_stamp
        if self._frozen:
            return False, FROZEN_ROTATING
        if self._resume_watermark is not None:
            if capture_stamp <= self._resume_watermark:
                return False, STALE_AFTER_ROTATION
            # First genuinely-fresh frame after the turn; guard satisfied.
            self._resume_watermark = None
        return True, FUSE

    # ── state queries ──
    @property
    def frozen(self) -> bool:
        """True while the turn freeze is active (mode == turning)."""
        return self._frozen

    @property
    def awaiting_fresh_frame(self) -> bool:
        """True after a turn, until the first fresh frame clears the guard."""
        return self._resume_watermark is not None

    @property
    def source(self) -> str:
        """Label of the freeze decision source (see ``sensor_freeze_policy``)."""
        return self._source

    def is_passing(self) -> bool:
        """True only when live frames flow — neither frozen nor awaiting a fresh
        frame. Callers gate non-depth streams (e.g. pose) on this so the whole
        sensor snapshot stays consistent across the turn and its resume."""
        return not self._frozen and self._resume_watermark is None

    # ── internals ──
    def _reevaluate(self, now: Optional[float]) -> None:
        frozen, source = self.policy.decide()
        if self._frozen and not frozen:
            # freeze -> resume edge: reject everything captured up to now. Use
            # the later of the newest frame seen and the wall clock, so a mode
            # signal that overtook late turn frames still excludes them.
            base = self._max_seen_stamp
            if now is not None and now > base:
                base = now
            self._resume_watermark = base + self.resume_settle_sec
        elif frozen:
            # Re-entering (or staying) frozen drops any pending resume guard;
            # it re-arms when this freeze ends.
            self._resume_watermark = None
        self._frozen = frozen
        self._source = source
