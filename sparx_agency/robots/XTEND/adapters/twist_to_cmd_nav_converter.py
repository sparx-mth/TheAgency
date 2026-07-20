from __future__ import annotations

import time
from dataclasses import dataclass

from sparx_agency.robots.XTEND.adapters.axis_calibration import (
    XTEND_CALIBRATION,
    XtendAxisCalibration,
)

# Kept as module constants because they are the numbers people quote when talking
# about this platform ("0.65 is the yaw calibration point"). They are now DERIVED
# from the single source of truth in axis_calibration.py rather than duplicated,
# so a re-calibration cannot leave two disagreeing copies behind.
FORWARD_REF_VEL = XTEND_CALIBRATION.forward.ref_velocity
FORWARD_REF_VALUE = XTEND_CALIBRATION.forward.ref_counts
FORWARD_MAX_VALUE = XTEND_CALIBRATION.forward.max_counts

TURN_REF_ANGULAR = XTEND_CALIBRATION.yaw.ref_velocity
TURN_REF_VALUE = XTEND_CALIBRATION.yaw.ref_counts
TURN_MAX_VALUE = XTEND_CALIBRATION.yaw.max_counts


def scale_translation_axis(value: float, calibration=None) -> int:
    """Magnitude of a translation velocity in axis counts (unsigned)."""
    cal = (calibration or XTEND_CALIBRATION).forward
    return abs(cal.to_counts(value))


def scale_yaw_axis(angular_z: float, calibration=None) -> int:
    """Magnitude of a yaw rate in axis counts (unsigned)."""
    cal = (calibration or XTEND_CALIBRATION).yaw
    return abs(cal.to_counts(angular_z))


def _signed_axis(value: float, delta: float, cal) -> int:
    """Calibrated axis counts, sign-preserved, 0 inside the deadzone."""
    if -delta <= value <= delta:
        return 0
    return cal.to_counts(value)


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
        calibration: XtendAxisCalibration = None,
    ):
        self.angular_delta = float(angular_delta)
        self.linear_delta = float(linear_delta)
        self.timeout_sec = float(timeout_sec)
        self.publish_stop_on_timeout = bool(publish_stop_on_timeout)
        self.zero_stop_required_count = int(zero_stop_required_count)
        # Per-axis SI <-> counts calibration. Defaults to the platform's measured
        # one; pass a different XtendAxisCalibration to re-calibrate an axis (in
        # particular lateral, which has only ever inherited forward's numbers).
        self.calibration = calibration or XTEND_CALIBRATION

        self.last_cmd: AxesCommand | None = None
        self.last_twist_time: float = 0.0
        self.zero_stop_count: int = 0

    def _compute_axes(self, lx: float, ly: float, lz: float, az: float) -> AxesCommand:
        c = self.calibration
        return AxesCommand(
            forward=_signed_axis(lx, self.linear_delta, c.forward),
            # ly>0 ("left" in Twist convention) maps to a negative lateral axis,
            # matching hold_lateral_left()'s sign in xtend_online_bridge_base.py.
            lateral=-_signed_axis(ly, self.linear_delta, c.lateral),
            vertical=_signed_axis(lz, self.linear_delta, c.vertical),
            # XTEND's yaw axis is inverted relative to Twist's angular.z (positive
            # angular.z = turn_left = negative yaw axis), matching the sign
            # hold_turn_left()/hold_turn_right() use in xtend_online_bridge_base.py.
            yaw=-_signed_axis(az, self.angular_delta, c.yaw),
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
