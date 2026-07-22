"""Data vocabulary for the waypoint follower: control axes, states, command.

These are the contract the follower exposes to a caller (e.g. a ROS adapter).
They carry no logic — the state machine lives in ``follower.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sparx_agency.core.common.types import ControlCommand


class ControlAxis(str, Enum):
    """The single axis the follower needs control of in its current state.

    The adapter maps these to whatever handshake the platform requires before
    the corresponding motion is safe.
    """

    YAW = "yaw"          # rotate in place (YAW_ALIGN)
    FORWARD = "forward"  # advance straight (ADVANCE)


class FollowerState(str, Enum):
    """Navigation states. Platform bring-up states live in the ROS adapter.

    ``YAW_SETTLE`` is the pause that follows every rotation burst: the platform
    coasts to a stop (yaw inertia) and then dwells in place so the AprilTag
    localization — which jumps while rotating but is accurate when still —
    re-converges before the next heading is measured.
    """

    YAW_ALIGN = "YAW_ALIGN"
    YAW_SETTLE = "YAW_SETTLE"
    ADVANCE = "ADVANCE"
    BRAKE = "BRAKE"
    DONE = "DONE"


@dataclass(frozen=True)
class FollowerCommand:
    """Output of one :meth:`WaypointFollower.step`.

    Attributes:
        command: Planar velocity command (``vy``/``vz`` always 0).
        state: State the follower is in after this step.
        required_axis: Axis whose confirmation is needed to keep moving, or
            ``None`` when the current state needs no handshake (BRAKE/DONE).
        freeze: Desired sensor-freeze state, or ``None`` to leave it unchanged.
        done: True once the goal has been reached.
        wp_idx: Index of the waypoint currently being pursued.
        num_waypoints: Length of the active (re-anchored) path.
    """

    command: ControlCommand
    state: FollowerState
    required_axis: Optional[ControlAxis]
    freeze: Optional[bool]
    done: bool
    wp_idx: int
    num_waypoints: int

    @property
    def vx(self) -> float:
        return self.command.x

    @property
    def vy(self) -> float:
        """Lateral velocity command (m/s, +left). Always 0 for the one-axis
        follower; carries the ROLL correction when a roll-assist layer wraps it."""
        return self.command.y

    @property
    def wz(self) -> float:
        return self.command.yaw_rate
