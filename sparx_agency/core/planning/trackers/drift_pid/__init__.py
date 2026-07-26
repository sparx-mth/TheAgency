"""Drift-PID path follower: fly the leg, hold the line, get unstuck.

A continuous multi-axis tracker for a heavy indoor drone flown on an AprilTag
pose that degrades rather than fails. Three PID loops (cross-track, along-track,
heading) whose integral terms *are* the per-axis drift estimate, behind a force
envelope that caps each axis and the combined multi-axis demand, scheduled by
localization confidence, with reflexes for walls the camera cannot see.

See ``README.md`` for the design and the tuning order.
"""
from .blockage import AXIS_FORWARD, AXIS_YAW, Blockage, BlockageMonitor, BlockageParams
from .confidence import (
    ConfidenceParams,
    ConfidenceScheduler,
    ControlAuthority,
    LocalizationQuality,
)
from .envelope import EnvelopeParams, ForceEnvelope, counts_for
from .escape import EscapeCommand, EscapeManeuver, EscapeParams, EscapeState
from .follower import DriftPidFollower
from .params import DriftPidParams
from .pid import AxisPid, PidGains
from .types import DriftPidCommand, DriftPidState, DriftTelemetry

__all__ = [
    "AXIS_FORWARD",
    "AXIS_YAW",
    "AxisPid",
    "Blockage",
    "BlockageMonitor",
    "BlockageParams",
    "ConfidenceParams",
    "ConfidenceScheduler",
    "ControlAuthority",
    "DriftPidCommand",
    "DriftPidFollower",
    "DriftPidParams",
    "DriftPidState",
    "DriftTelemetry",
    "EnvelopeParams",
    "EscapeCommand",
    "EscapeManeuver",
    "EscapeParams",
    "EscapeState",
    "ForceEnvelope",
    "LocalizationQuality",
    "PidGains",
    "counts_for",
]
