from __future__ import annotations

import time

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


class TwistToCmdNavConverter:
    """
    Stateful, ROS-free converter from Twist fields to (action, value) cmd_nav pairs.

    Call process() on every incoming Twist message.
    Call check_timeout() periodically to detect a stale stream and emit stop.
    Both return (action, value) or None when the output should be suppressed.
    """

    def __init__(
        self,
        angular_delta: float = 0.05,
        linear_delta: float = 0.05,
        timeout_sec: float = 1.5,
        publish_stop_on_timeout: bool = True,
        zero_stop_required_count: int = 2,
        allow_multi_axes: bool = True,
    ):
        self.angular_delta = float(angular_delta)
        self.linear_delta = float(linear_delta)
        self.timeout_sec = float(timeout_sec)
        self.publish_stop_on_timeout = bool(publish_stop_on_timeout)
        self.zero_stop_required_count = int(zero_stop_required_count)
        self.allow_multi_axes = bool(allow_multi_axes)

        self.last_action: tuple[str, int] | None = None
        self.last_twist_time: float = 0.0
        self.zero_stop_count: int = 0

    def _choose_cmd(self, lx: float, ly: float, lz: float, az: float) -> tuple[str, int]:
        if az > self.angular_delta:
            return "turn_left", scale_yaw_axis(az)
        if az < -self.angular_delta:
            return "turn_right", scale_yaw_axis(az)
        if lx > self.linear_delta:
            return "forward", scale_translation_axis(lx)
        if lx < -self.linear_delta:
            return "backward", scale_translation_axis(lx)
        if self.allow_multi_axes:
            if lz > self.linear_delta:
                return "up", scale_translation_axis(lz)
            if lz < -self.linear_delta:
                return "down", scale_translation_axis(lz)
            if ly > self.linear_delta:
                return "left", scale_translation_axis(ly)
            if ly < -self.linear_delta:
                return "right", scale_translation_axis(ly)
        return "stop", 0

    def process(self, lx: float, ly: float, lz: float, az: float) -> tuple[str, int] | None:
        """
        Process one Twist. Returns (action, value) to emit, or None to suppress.

        Applies two filters:
        - Deduplication: repeated hold commands are swallowed (bridge holds until changed).
        - Stop debounce: transient zero-Twist blips between yaw commands are ignored
          until zero_stop_required_count consecutive stops arrive.
        """
        self.last_twist_time = time.time()
        action, value = self._choose_cmd(lx, ly, lz, az)

        if action == "stop":
            self.zero_stop_count += 1
            if (
                self.last_action is not None
                and self.last_action[0] != "stop"
                and self.zero_stop_count < self.zero_stop_required_count
            ):
                return None
        else:
            self.zero_stop_count = 0

        key = (action, value)
        if key == self.last_action:
            return None

        self.last_action = key
        return action, value

    def check_timeout(self) -> tuple[str, int] | None:
        """
        Returns ("stop", 0) once when the Twist stream goes silent past timeout_sec.
        Deduplicated: returns None if stop was already the last emitted action.
        """
        if not self.publish_stop_on_timeout or self.last_twist_time <= 0.0:
            return None
        if time.time() - self.last_twist_time <= self.timeout_sec:
            return None

        key = ("stop", 0)
        if key == self.last_action:
            return None

        self.last_action = key
        return key