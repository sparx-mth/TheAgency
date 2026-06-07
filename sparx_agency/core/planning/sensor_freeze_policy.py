"""Mode-authoritative sensor-freeze policy (pure stdlib, ROS-free).

A *planning* decision: while the platform is rotating in place, both depth and
localization are unreliable, so the planner chooses NOT to fuse them into the
voxel map (it freezes the sensor stream — holds it on its last value — until
the turn finishes, so the map does not smear). This module decides whether to
freeze, from two inputs:

  1. The system-wide demo mode (e.g. ``"turning"``). This is the
     AUTHORITATIVE signal for whether the platform is really turning.
  2. An explicit per-state freeze request from the navigation controller.
     Tracked for diagnostics and used only as a FALLBACK before the mode
     signal has spoken (or when mode-based freezing is disabled).

Why mode-authoritative instead of OR-ing the two inputs? An ``OR`` gate can
stick frozen: the controller publishes "freeze", the system later leaves
turning mode (timeout / external command) *before* the controller publishes
"unfreeze", and the stale "freeze" keeps the map frozen forever. The system
mode is downstream of everyone's requests and is the only signal that knows
what was actually granted, so trusting it removes that whole class of stuck
state.

The policy is pure decision logic — the ROS node owns the topics, the replay
timer and the heartbeat, and calls :meth:`decide` to act.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

#: The mode topic is present and authoritative.
SRC_MODE_AUTH = "mode_auth"
#: Mode-based freezing is enabled but no mode message has arrived yet, so the
#: explicit request is used as a fallback.
SRC_EXPLICIT_FALLBACK = "explicit_fallback"
#: Mode-based freezing is disabled; only the explicit request controls freeze.
SRC_EXPLICIT_ONLY = "explicit_only"


@dataclass
class SensorFreezePolicy:
    """Stateful freeze decision from a demo mode + an explicit request.

    Attributes:
        freeze_on_turning_mode: When True (default), the demo mode is the
            authoritative source once it has spoken. When False, only the
            explicit request matters (original "explicit-only" semantics).
        explicit_freeze: Last explicit freeze request seen.
        mode_says_freeze: Whether the last demo mode equalled the turning mode.
        n_mode_msgs: How many demo-mode messages have been observed.
    """

    freeze_on_turning_mode: bool = True
    explicit_freeze: bool = False
    mode_says_freeze: bool = False
    n_mode_msgs: int = 0

    def note_explicit(self, frozen: bool) -> None:
        """Record an explicit freeze request from the controller."""
        self.explicit_freeze = bool(frozen)

    def note_mode(self, is_turning_mode: bool) -> None:
        """Record a demo-mode message; ``is_turning_mode`` is the comparison
        result against the configured turning mode name."""
        self.mode_says_freeze = bool(is_turning_mode)
        self.n_mode_msgs += 1

    def reset_mode_freeze(self) -> None:
        """Force-clear the mode-based freeze (manual stuck-mode recovery).

        Non-destructive: the next genuine turning-mode message re-sets it.
        """
        self.mode_says_freeze = False

    def decide(self) -> Tuple[bool, str]:
        """Return ``(frozen, source_label)`` for the current inputs."""
        if not self.freeze_on_turning_mode:
            return self.explicit_freeze, SRC_EXPLICIT_ONLY
        if self.n_mode_msgs == 0:
            # Mode-based freezing is enabled but the mode topic has not spoken
            # yet; stay compatible with explicit-only during startup so a
            # missing mode topic does not silently disable freezing entirely.
            return self.explicit_freeze, SRC_EXPLICIT_FALLBACK
        return self.mode_says_freeze, SRC_MODE_AUTH
