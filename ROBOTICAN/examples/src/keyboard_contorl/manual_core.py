# manual_core.py

import csv
import os
from dataclasses import dataclass
from typing import Optional

from rooster_manager_interfaces.msg import RoosterState
from fcu_driver_interfaces.msg import ManualControl
from rclpy.node import Node


@dataclass
class AxisState:
    """Holds the current manual control axes.

    NOTE (ROLL mode):
    - x: forward/backward rolling
    - y: roll left/right
    - z: up/down (throttle)
    - r: yaw
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0


class ManualCommandModel:
    """
    Pure logic for manual command:

    - Maintains AxisState.
    - Applies key presses or direct assignments.
    - Supports turtle (slow) mode scale.
    """

    def __init__(self, step: float = 10.0, turtle_scale: float = 0.5):
        self.step = float(step)
        self.axes = AxisState()
        self.turtle_scale = float(turtle_scale)
        self.turtle_mode = False

    @staticmethod
    def _clamp(v: float) -> float:
        return max(-1000.0, min(1000.0, v))

    def set_axes(self, x: float, y: float, z: float, r: float):
        self.axes = AxisState(
            x=self._clamp(x),
            y=self._clamp(y),
            z=self._clamp(z),
            r=self._clamp(r),
        )

    def reset_axes(self):
        self.axes = AxisState()

    def toggle_turtle(self) -> bool:
        self.turtle_mode = not self.turtle_mode
        return self.turtle_mode

    def get_scaled_axes(self) -> AxisState:
        scale = self.turtle_scale if self.turtle_mode else 1.0
        return AxisState(
            x=self.axes.x * scale,
            y=self.axes.y * scale,
            z=self.axes.z * scale,
            r=self.axes.r * scale,
        )

    def apply_increment(self, axis: str, sign: int):
        """Increment/decrement a single axis by step, with clamping."""
        s = self.step * sign
        if axis == "x":
            self.axes.x = self._clamp(self.axes.x + s)
        elif axis == "y":
            self.axes.y = self._clamp(self.axes.y + s)
        elif axis == "z":
            self.axes.z = self._clamp(self.axes.z + s)
        elif axis == "r":
            self.axes.r = self._clamp(self.axes.r + s)


class CsvLogger:
    """Responsible for logging commands + state to CSV."""

    def __init__(self, node: Node, path: str):
        self._node = node
        self._path = os.path.abspath(path)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)

        self._writer.writerow(
            [
                "t", "src",
                "x", "y", "z", "r",
                "roll", "pitch", "azimuth",
                "flight_mode", "armed", "airborne",
            ]
        )
        self._file.flush()
        self._node.get_logger().info(f"Logging to {self._path}")

    def log_command(
        self,
        src: str,
        manual_msg: ManualControl,
        state: Optional[RoosterState],
        now_sec: float,
    ):
        if state is None:
            roll = pitch = azimuth = 0.0
            flight_mode = -1
            armed = False
            airborne = False
        else:
            roll = state.roll
            pitch = state.pitch
            azimuth = state.azimuth
            flight_mode = state.flight_mode
            armed = state.armed
            airborne = state.airborne

        self._writer.writerow(
            [
                f"{now_sec:.6f}", src,
                f"{manual_msg.x:.1f}", f"{manual_msg.y:.1f}",
                f"{manual_msg.z:.1f}", f"{manual_msg.r:.1f}",
                f"{roll:.4f}", f"{pitch:.4f}", f"{azimuth:.4f}",
                flight_mode, int(armed), int(airborne),
            ]
        )
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
