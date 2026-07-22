"""Rotate-with-stops scan search for the "arrived, still looking" phase (ROS-free).

When the coordinate route (A*/NavDP) reaches its goal but the target object was
never confirmed on the way, the mission does not stop: it sweeps the room looking
for the object. This policy produces that sweep as a body-frame velocity command,
frame by frame -- a slow yaw rotation broken by **stops**, so the detector gets a
clear, motion-blur-free look down each new bearing before the drone turns again
(the detector runs at a few Hz; rotating continuously would smear every frame).

The cycle is ``PAUSE -> ROTATE -> PAUSE -> ROTATE -> ...``. It begins with a pause
so the detector gets a clean look straight ahead the instant the drone arrives.
Optionally (``forward_speed > 0``) a small forward **relocate** step is inserted
after ``bursts_before_move`` rotate bursts, to search from a fresh vantage rather
than spinning forever on one spot; left at ``0`` the sweep is purely in place.

There is no terminal state: the sweep runs until the mission FSM leaves the scan
(the target is confirmed, or the goal changes). The command still passes through
the node's per-axis force shaper before publication, so the sweep respects the
platform's minimum/maximum force just like the closure does.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.common.types import ControlCommand

PAUSE = "pause"
ROTATE = "rotate"
RELOCATE = "relocate"


@dataclass(frozen=True)
class ScanSearchConfig:
    """Tuning for :class:`ScanSearchPolicy` (SI units, body frame REP-103).

    Attributes:
        yaw_rate: Rotation speed during a ROTATE burst (rad/s, ``+`` is CCW).
        rotate_s: Duration of one ROTATE burst (s).
        pause_s: Duration of one PAUSE / stop (s) -- long enough for the detector
            to score a few frames down the new bearing.
        direction: Sweep direction, ``+1`` (CCW, turn left) or ``-1`` (CW).
        forward_speed: Speed of an optional relocate step (m/s); ``0`` disables it
            so the sweep stays purely in place.
        forward_s: Duration of a relocate step (s).
        bursts_before_move: Number of ROTATE bursts between relocate steps.
    """

    yaw_rate: float = 0.4
    rotate_s: float = 1.2
    pause_s: float = 1.2
    direction: float = 1.0
    forward_speed: float = 0.0
    forward_s: float = 0.0
    bursts_before_move: int = 8

    def __post_init__(self) -> None:
        if self.yaw_rate <= 0.0:
            raise ValueError("yaw_rate must be > 0")
        if self.rotate_s <= 0.0:
            raise ValueError("rotate_s must be > 0")
        if self.pause_s < 0.0:
            raise ValueError("pause_s must be >= 0")
        if self.direction not in (1.0, -1.0):
            raise ValueError("direction must be +1.0 or -1.0")
        if self.forward_speed < 0.0:
            raise ValueError("forward_speed must be >= 0")
        if self.forward_s < 0.0:
            raise ValueError("forward_s must be >= 0")
        if self.bursts_before_move < 1:
            raise ValueError("bursts_before_move must be >= 1")


class ScanSearchPolicy:
    """Alternate rotate/stop bursts (with optional forward relocation), forever."""

    def __init__(self, config=None) -> None:
        self.cfg = config or ScanSearchConfig()
        self.reset()

    def reset(self) -> None:
        """Restart the sweep at a PAUSE (clean look before the first turn)."""
        self._phase = PAUSE
        self._t = 0.0
        self._bursts = 0

    @property
    def phase(self) -> str:
        return self._phase

    def command(self, dt: float) -> ControlCommand:
        """Advance the sweep by ``dt`` seconds and return this tick's command."""
        cfg = self.cfg
        self._t += max(0.0, float(dt))

        if self._phase == PAUSE and self._t >= cfg.pause_s:
            self._enter(ROTATE)
        elif self._phase == ROTATE and self._t >= cfg.rotate_s:
            self._bursts += 1
            if (cfg.forward_speed > 0.0 and cfg.forward_s > 0.0
                    and self._bursts >= cfg.bursts_before_move):
                self._enter(RELOCATE)
            else:
                self._enter(PAUSE)
        elif self._phase == RELOCATE and self._t >= cfg.forward_s:
            self._bursts = 0
            self._enter(PAUSE)
        return self._emit()

    # ── helpers ───────────────────────────────────────────────────────
    def _enter(self, phase: str) -> None:
        self._phase = phase
        self._t = 0.0

    def _emit(self) -> ControlCommand:
        cfg = self.cfg
        if self._phase == ROTATE:
            return ControlCommand.velocity(0.0, 0.0, 0.0, cfg.direction * cfg.yaw_rate,
                                           source="scan", phase=self._phase)
        if self._phase == RELOCATE:
            return ControlCommand.velocity(cfg.forward_speed, 0.0, 0.0, 0.0,
                                           source="scan", phase=self._phase)
        return ControlCommand.velocity(0.0, 0.0, 0.0, 0.0,
                                       source="scan", phase=self._phase)
