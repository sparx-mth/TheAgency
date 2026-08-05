"""Data vocabulary for the multi-axis follower: state and command.

These are the contract the follower exposes to a caller (e.g. a ROS adapter).
They carry no logic — the state machine lives in ``follower.py``. The public
attribute surface intentionally mirrors the one-axis follower's
``FollowerCommand`` (``state`` / ``done`` / ``required_axis`` / ``freeze`` /
``wp_idx`` / ``num_waypoints`` plus ``vx`` / ``wz``) so an adapter can drive
either tracker, with the added ``vy`` lateral channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sparx_agency.core.common.types import ControlCommand


class MultiAxisState(str, Enum):
    """Navigation states for the continuous multi-axis tracker.

    There is no stop-and-spin phase: the drone pursues the path continuously and
    only stops to station-keep once the final goal is captured.

    ``IDLE``  — no path yet; holds zero.
    ``RUN``   — pursuing the active waypoint (forward + lateral + yaw together).
    ``HOLD``  — final goal reached; station-keeps with a decisive deadband.
    """

    IDLE = "IDLE"
    RUN = "RUN"
    HOLD = "HOLD"


@dataclass(frozen=True)
class MultiAxisCommand:
    """Output of one :meth:`MultiAxisFollower.step`.

    Attributes:
        command: Velocity command (``vz`` always 0 — fixed altitude).
        state: State the follower is in after this step.
        required_axis: Always ``None`` (the multi-axis tracker needs no per-axis
            handshake); kept for interface parity with the one-axis follower.
        freeze: Always ``None`` (sensors stay live; the tracker never freezes to
            re-measure); kept for interface parity.
        done: True once the final goal has been reached (state is HOLD).
        wp_idx: Index of the waypoint currently being pursued.
        num_waypoints: Length of the active (re-anchored) path.
        yaw_engaged: Whether yaw is actively past the engage/release hysteresis
            deadband this tick (the follower's own authoritative "really turning,
            not just trimming" signal) -- this is the regime signal
            ``waypoint_follower_node.py``'s ``_supervisor_cmd_wz()`` was missing
            for this tracker (it already has one for ``drift_pid`` via
            ``follower.state``). ``wz`` can be nonzero from slew/shape-axis
            residue even with yaw released, so callers that need to distinguish
            "genuinely turning" from "an ordinary command in flight" should key
            on this, not on ``wz != 0``.
    """

    command: ControlCommand
    state: MultiAxisState
    required_axis: Optional[str]
    freeze: Optional[bool]
    done: bool
    wp_idx: int
    num_waypoints: int
    yaw_engaged: bool = False

    @property
    def vx(self) -> float:
        """Forward velocity command (m/s)."""
        return self.command.x

    @property
    def vy(self) -> float:
        """Lateral velocity command (m/s, +left)."""
        return self.command.y

    @property
    def wz(self) -> float:
        """Yaw-rate command (rad/s, +CCW)."""
        return self.command.yaw_rate
