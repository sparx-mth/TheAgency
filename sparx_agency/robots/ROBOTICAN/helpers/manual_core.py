# manual_core.py

import csv
import os
from dataclasses import dataclass
from typing import Optional

from rooster_manager_interfaces.msg import RoosterState
from fcu_driver_interfaces.msg import ManualControl, UAVState
from rclpy.node import Node
from datetime import datetime


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


class RLTransitionLogger:
    """
    Logs RL-style transitions:

    One row per action (pulse / path segment):

    S:  position, angles, velocity, altitude_relative (if available)
    A:  x, y, z, r, duration
    S': same fields as S after action

    File name is timestamped so each run gets a new CSV.
    """

    def __init__(self, node: Node, base_path: str):
        self._node = node

        # Build timestamped file name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(base_path)
        if ext == "":
            ext = ".csv"
        filename = f"csv{os.sep}{root}_{ts}{ext}"
        self._path = os.path.abspath(filename)

        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)

        # Episode is the whole GUI run, step is per action
        self.episode_id = 1
        self.step_id = 0

        self._writer.writerow(
            [
                "episode",
                "step",
                "t_start",
                "t_end",

                # S (before action)
                "S_pos_x",
                "S_pos_y",
                "S_pos_z",
                "S_vel_x",
                "S_vel_y",
                "S_vel_z",
                "S_roll",
                "S_pitch",
                "S_azimuth",
                "S_alt_rel",

                # A (action)
                "A_x",
                "A_y",
                "A_z",
                "A_r",
                "A_duration",

                # S' (after action)
                "Sp_pos_x",
                "Sp_pos_y",
                "Sp_pos_z",
                "Sp_vel_x",
                "Sp_vel_y",
                "Sp_vel_z",
                "Sp_roll",
                "Sp_pitch",
                "Sp_azimuth",
                "Sp_alt_rel",
            ]
        )
        self._file.flush()
        self._node.get_logger().info(f"RL transitions will be logged to {self._path}")

    @staticmethod
    def _extract_state_fields(st: Optional[RoosterState]):
        """
        Returns (pos_x, pos_y, pos_z,
                 vel_x, vel_y, vel_z,
                 roll, pitch, azimuth, alt_rel)
        or empty strings if missing.
        """
        if st is None:
            return ("", "", "", "", "", "", "", "", "", "")

        # --- Position ---
        try:
            pos_x = st.position.x
            pos_y = st.position.y
            pos_z = st.position.z
        except Exception as exp:
            print(f"Failed to extract position from state: {exp}")
            pos_x = pos_y = pos_z = ""

        # --- Velocity ---
        try:
            vel_x = st.velocity.x
            vel_y = st.velocity.y
            vel_z = st.velocity.z
        except Exception as exp:
            print(f"Failed to extract velocity from state: {exp}")
            vel_x = vel_y = vel_z = ""

        # --- Angles / azimuth / altitude_relative ---
        roll = getattr(st, "roll", "")
        pitch = getattr(st, "pitch", "")
        azimuth = getattr(st, "azimuth", "")
        alt_rel = getattr(st, "altitude_relative", "")

        return (
            pos_x,
            pos_y,
            pos_z,
            vel_x,
            vel_y,
            vel_z,
            roll,
            pitch,
            azimuth,
            alt_rel,
        )

    def log_transition(
        self,
        s_before: Optional[RoosterState],
        action_x: float,
        action_y: float,
        action_z: float,
        action_r: float,
        duration: float,
        t_start: float,
        t_end: float,
        s_after: Optional[RoosterState],
    ):
        """Write one (S, A, S') row."""
        self.step_id += 1

        s_vec = self._extract_state_fields(s_before)
        sp_vec = self._extract_state_fields(s_after)

        row = [
            self.episode_id,
            self.step_id,
            f"{t_start:.6f}",
            f"{t_end:.6f}",
            *s_vec,
            action_x,
            action_y,
            action_z,
            action_r,
            duration,
            *sp_vec,
        ]
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
