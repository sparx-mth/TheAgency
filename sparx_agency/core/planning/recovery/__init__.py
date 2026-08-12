"""Controller-agnostic stuck detection and recovery.

The drone gets stuck -- most often clipping a doorway edge or pinned against an
obstacle the camera cannot see -- and the controller flying it has no idea,
because from the controller's side the command went out fine. This package is the
missing feedback loop, wrapped around *any* follower:

  * :class:`~.stuck_detector.StuckDetector` -- notices, from commanded-vs-achieved
    motion over a trailing window, that the drone is not going where it is told.
  * :class:`~.escape_maneuver.EscapeManeuver` -- backs it out with a short,
    open-loop reflex ("exit the wall in the other direction").
  * :class:`~.recovery_supervisor.RecoverySupervisor` -- ties the two together and,
    once the back-out is done, asks for a replan **from the real, recovered pose**.

See the package README for how it wires into the FALCON follower node and how it
relates to ``drift_pid``'s own built-in reflex.

Pure, ROS-free and Python 3.8 compatible (the Noetic FALCON adapter imports
``core`` under 3.8): stdlib + ``Pose2D`` only, no numpy, no scipy.
"""
from .escape_maneuver import (
    EscapeCommand,
    EscapeManeuver,
    EscapeParams,
    EscapeState,
)
from .recovery_supervisor import (
    RECOVERY_ESCAPE,
    RECOVERY_MONITOR,
    RECOVERY_NOMINAL,
    RecoveryDecision,
    RecoveryParams,
    RecoverySupervisor,
)
from .stuck_detector import (
    AXIS_FORWARD,
    AXIS_NONE,
    AXIS_YAW,
    StuckDetector,
    StuckParams,
    StuckVerdict,
)

__all__ = [
    "AXIS_NONE",
    "AXIS_FORWARD",
    "AXIS_YAW",
    "StuckDetector",
    "StuckParams",
    "StuckVerdict",
    "EscapeManeuver",
    "EscapeParams",
    "EscapeCommand",
    "EscapeState",
    "RecoverySupervisor",
    "RecoveryParams",
    "RecoveryDecision",
    "RECOVERY_NOMINAL",
    "RECOVERY_MONITOR",
    "RECOVERY_ESCAPE",
]

