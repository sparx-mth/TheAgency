"""Data vocabulary for the drift-PID controller: state, command and telemetry.

The public attribute surface of :class:`DriftPidCommand` intentionally mirrors
the one-axis ``FollowerCommand`` and ``MultiAxisCommand`` (``state`` / ``done`` /
``required_axis`` / ``freeze`` / ``wp_idx`` / ``num_waypoints`` plus ``vx`` /
``vy`` / ``wz``) so the existing ROS adapter can drive this tracker with no
special-casing, and :class:`DriftTelemetry` rides along as the one genuinely new
field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sparx_agency.core.common.types import ControlCommand


class DriftPidState(str):
    """Navigation states for the drift-PID controller (a py3.8-safe string enum).

    ``IDLE``    — no path yet; holds zero.
    ``TURN``    — heading error is large; rotating to face the leg, translation
                  suppressed, position held by the station-keeping loops.
    ``TRACK``   — flying the leg: feed-forward cruise plus cross-track and
                  heading corrections.
    ``HOLD``    — station-keeping on a fixed anchor (goal reached, or told to
                  hold). This is where fore/aft drift is actively cancelled.
    ``ESCAPE``  — a blockage reflex owns the command.
    """

    IDLE = "IDLE"
    TURN = "TURN"
    TRACK = "TRACK"
    HOLD = "HOLD"
    ESCAPE = "ESCAPE"


@dataclass(frozen=True)
class DriftTelemetry:
    """What the controller has learned and what it is currently fighting.

    This exists to be published: the whole point of an integral term on a drifting
    airframe is that somebody can look at it and see how hard the room is pushing
    the drone around.

    Attributes:
        drift_vy: Standing lateral command the controller holds to stay on
            track (m/s). Positive means it must continuously push left, i.e. the
            drone is being carried to the right.
        drift_vx: Standing fore/aft command held while station-keeping (m/s).
        drift_wz: Standing yaw-rate command held to keep its heading (rad/s).
        cross_track_m: Signed cross-track error this tick (m, + = trajectory is
            to the drone's left).
        along_track_m: Signed along-track error to the station-keeping anchor (m).
        heading_err_rad: Signed heading error this tick (rad).
        effort: Fraction of the airframe's combined force budget the published
            command is spending (1.0 = one axis at its own maximum).
        speed_scale: Confidence-derived speed multiplier in force this tick.
        lead_s: Latency compensation applied to the steering pose this tick (s).
            0 while coasting or while the commands are unproven.
        deadband_extra_m: How far the tracking deadbands were widened this tick
            by the pose's own reported error (m). When this is large, a small
            cross-track number is noise, not drift.
        authority: Why the controller has the authority it has (human-readable).
        blocked_axis: ``""``, ``"forward"`` or ``"yaw"``.
        escape_state: Phase of the escape reflex, or ``"IDLE"``.
    """

    drift_vy: float = 0.0
    drift_vx: float = 0.0
    drift_wz: float = 0.0
    cross_track_m: float = 0.0
    along_track_m: float = 0.0
    heading_err_rad: float = 0.0
    effort: float = 0.0
    speed_scale: float = 1.0
    lead_s: float = 0.0
    deadband_extra_m: float = 0.0
    authority: str = ""
    blocked_axis: str = ""
    escape_state: str = "IDLE"


@dataclass(frozen=True)
class DriftPidCommand:
    """Output of one :meth:`DriftPidFollower.step`.

    Attributes:
        command: Velocity command (``z`` always 0 — the platform holds altitude).
        state: State the controller is in after this step.
        required_axis: Always ``None``: this controller drives every axis at once
            and needs no per-axis mode handshake. Kept for interface parity.
        freeze: Always ``None``. The rotation freeze for continuous trackers is
            owned by ``RotationReobserveSupervisor`` in the adapter, not here —
            one authority per decision. Kept for interface parity.
        done: True once the final goal has been reached.
        wp_idx: Index of the waypoint currently being pursued.
        num_waypoints: Length of the active (re-anchored) path.
        telemetry: What the controller has learned; see :class:`DriftTelemetry`.
        report_blocked: True on the tick the reflexes give up, so the adapter can
            tell the planner there is something here it cannot see. Edge-triggered:
            true once per exhausted episode, never latched.
    """

    command: ControlCommand
    state: str
    required_axis: Optional[str]
    freeze: Optional[bool]
    done: bool
    wp_idx: int
    num_waypoints: int
    telemetry: DriftTelemetry = field(default_factory=DriftTelemetry)
    report_blocked: bool = False

    @property
    def vx(self):
        # type: () -> float
        """Forward velocity command (m/s)."""
        return self.command.x

    @property
    def vy(self):
        # type: () -> float
        """Lateral velocity command (m/s, + left)."""
        return self.command.y

    @property
    def wz(self):
        # type: () -> float
        """Yaw-rate command (rad/s, + CCW)."""
        return self.command.yaw_rate
