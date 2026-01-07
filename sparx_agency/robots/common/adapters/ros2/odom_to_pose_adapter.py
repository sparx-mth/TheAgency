from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from nav_msgs.msg import Odometry

from sparx_agency.core.common.types.perception import PoseSE3


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Standard yaw (Z-axis) from quaternion.
    """
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_rot(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass
class OdomPose:
    stamp_sec: float
    x: float
    y: float
    z: float
    yaw: float


class OdomToPoseAdapter:
    """
    Keeps latest odom pose and exposes it as PoseSE3.
    Typically you want yaw-only (ignore roll/pitch) for a 2D costmap.
    """

    def __init__(self, yaw_only: bool = True, zero_z: bool = True):
        self.yaw_only = bool(yaw_only)
        self.zero_z = bool(zero_z)
        self._latest: Optional[OdomPose] = None

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def update_from_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        stamp_sec = self._stamp_to_sec(msg.header.stamp)
        yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))

        z = float(p.z)
        if self.zero_z:
            z = 0.0

        self._latest = OdomPose(
            stamp_sec=stamp_sec,
            x=float(p.x),
            y=float(p.y),
            z=z,
            yaw=yaw,
        )

    def has_pose(self) -> bool:
        return self._latest is not None

    def get_pose_se3(self) -> Optional[PoseSE3]:
        if self._latest is None:
            return None

        if self.yaw_only:
            R = yaw_to_rot(self._latest.yaw)
        else:
            # If you ever want full rotation, you can extend this to use full quaternion->R.
            R = yaw_to_rot(self._latest.yaw)

        t = np.array([self._latest.x, self._latest.y, self._latest.z], dtype=np.float32)
        return PoseSE3(R=R, t=t)

    def get_yaw(self) -> Optional[float]:
        return None if self._latest is None else float(self._latest.yaw)

    def get_xy(self) -> Optional[tuple[float, float]]:
        if self._latest is None:
            return None
        return float(self._latest.x), float(self._latest.y)
