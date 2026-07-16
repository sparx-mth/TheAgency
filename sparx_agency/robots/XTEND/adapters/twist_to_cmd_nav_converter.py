from __future__ import annotations

import time
from dataclasses import dataclass

FORWARD_REF_VEL = 0.3
FORWARD_REF_VALUE = 400
FORWARD_MAX_VALUE = 600

TURN_REF_ANGULAR = 0.65
TURN_REF_VALUE = 1000
TURN_MAX_VALUE = 1000


def scale_translation_axis(value: float) -> int:
    axis = int(round((abs(float(value)) / FORWARD_REF_VEL) * FORWARD_REF_VALUE))
    return max(0, min(FORWARD_MAX_VALUE, axis))


def scale_yaw_axis(angular_z: float) -> int:
    axis = int(round((abs(float(angular_z)) / TURN_REF_ANGULAR) * TURN_REF_VALUE))
    return max(0, min(TURN_MAX_VALUE, axis))


def _signed_translation(value: float, delta: float) -> int:
    """Scaled translation axis, sign-preserved, 0 inside the deadzone."""
    if value > delta:
        return scale_translation_axis(value)
    if value < -delta:
        return -scale_translation_axis(value)
    return 0


def _signed_yaw(angular_z: float, delta: float) -> int:
    """Scaled yaw axis. XTEND's yaw axis is inverted relative to Twist's angular.z
    (positive angular.z = turn_left = negative yaw axis), matching the sign
    hold_turn_left()/hold_turn_right() already use in xtend_online_bridge_base.py."""
    if angular_z > delta:
        return -scale_yaw_axis(angular_z)
    if angular_z < -delta:
        return scale_yaw_axis(angular_z)
    return 0


@dataclass(frozen=True)
class AxesCommand:
    """One XTEND axes-array command: all four channels are independent and can
    be nonzero simultaneously (the XTEND controller is a free-floating handheld
    wand — any wrist angle already blends pitch/roll/yaw, so combined motion is
    the platform's normal operating mode, not an edge case)."""
    forward: int = 0
    lateral: int = 0
    vertical: int = 0
    yaw: int = 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.forward, self.lateral, self.vertical, self.yaw)

    def is_zero(self) -> bool:
        return self.as_tuple() == (0, 0, 0, 0)

    def describe(self) -> str:
        """Human-readable summary for logging, e.g. 'forward + left + turn_right'.
        Not used for control — only to make log lines readable at a glance."""
        if self.is_zero():
            return "stop"
        parts = []
        if self.forward > 0:
            parts.append("forward")
        elif self.forward < 0:
            parts.append("backward")
        if self.lateral > 0:
            parts.append("right")
        elif self.lateral < 0:
            parts.append("left")
        if self.vertical > 0:
            parts.append("up")
        elif self.vertical < 0:
            parts.append("down")
        if self.yaw > 0:
            parts.append("turn_right")
        elif self.yaw < 0:
            parts.append("turn_left")
        return " + ".join(parts)


class TwistToCmdNavConverter:
    """
    Stateful, ROS-free converter from Twist fields to a combined AxesCommand.

    All four axes (forward/back, lateral, vertical, yaw) are computed
    independently per Twist and can be nonzero at the same time — e.g. flying
    forward while turning is a normal combined command, not two competing ones.

    Call process() on every incoming Twist message.
    Call check_timeout() periodically to detect a stale stream and emit stop.
    Both return an AxesCommand or None when the output should be suppressed.
    """

    def __init__(
        self,
        angular_delta: float = 0.05,
        linear_delta: float = 0.05,
        timeout_sec: float = 1.5,
        publish_stop_on_timeout: bool = True,
        zero_stop_required_count: int = 2,
    ):
        self.angular_delta = float(angular_delta)
        self.linear_delta = float(linear_delta)
        self.timeout_sec = float(timeout_sec)
        self.publish_stop_on_timeout = bool(publish_stop_on_timeout)
        self.zero_stop_required_count = int(zero_stop_required_count)

        self.last_cmd: AxesCommand | None = None
        self.last_twist_time: float = 0.0
        self.zero_stop_count: int = 0

    def _compute_axes(self, lx: float, ly: float, lz: float, az: float) -> AxesCommand:
        return AxesCommand(
            forward=_signed_translation(lx, self.linear_delta),
            # ly>0 ("left" in Twist convention) maps to a negative lateral axis,
            # matching hold_lateral_left()'s sign in xtend_online_bridge_base.py.
            lateral=-_signed_translation(ly, self.linear_delta),
            vertical=_signed_translation(lz, self.linear_delta),
            yaw=_signed_yaw(az, self.angular_delta),
        )

    def process(self, lx: float, ly: float, lz: float, az: float) -> AxesCommand | None:
        """
        Process one Twist. Returns the AxesCommand to emit, or None to suppress.

        Applies two filters:
        - Deduplication: repeated hold commands are swallowed (bridge holds until changed).
        - Stop debounce: transient zero-Twist blips are ignored until
          zero_stop_required_count consecutive zero commands arrive.
        """
        self.last_twist_time = time.time()
        cmd = self._compute_axes(lx, ly, lz, az)

        if cmd.is_zero():
            self.zero_stop_count += 1
            if (
                self.last_cmd is not None
                and not self.last_cmd.is_zero()
                and self.zero_stop_count < self.zero_stop_required_count
            ):
                return None
        else:
            self.zero_stop_count = 0

        if cmd == self.last_cmd:
            return None

        self.last_cmd = cmd
        return cmd

    def check_timeout(self) -> AxesCommand | None:
        """
        Returns a zero AxesCommand once when the Twist stream goes silent past
        timeout_sec. Deduplicated: returns None if stop was already last emitted.
        """
        if not self.publish_stop_on_timeout or self.last_twist_time <= 0.0:
            return None
        if time.time() - self.last_twist_time <= self.timeout_sec:
            return None

        zero = AxesCommand()
        if self.last_cmd is not None and self.last_cmd == zero:
            return None

        self.last_cmd = zero
        return zero
