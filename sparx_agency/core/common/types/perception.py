from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for a monocular camera."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class PoseSE3:
    """
    Pose of a frame in a map/world coordinate system.
    R: 3x3 rotation matrix
    t: 3 translation vector
    Represents: p_world = R @ p_local + t
    """
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> "PoseSE3":
        return PoseSE3(R=np.eye(3, dtype=np.float32), t=np.zeros(3, dtype=np.float32))

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.R.astype(np.float32)
        T[:3, 3] = self.t.astype(np.float32)
        return T

    def transform_points(self, pts_local: np.ndarray) -> np.ndarray:
        """pts_local: (N,3) -> (N,3) in world."""
        return (pts_local @ self.R.T) + self.t


@dataclass(frozen=True)
class RGBFrame:
    image: np.ndarray  # HxWx3 uint8 or float32
    stamp_sec: float
    frame_id: str = ""


@dataclass(frozen=True)
class DepthFrame:
    depth_m: np.ndarray  # HxW float32
    stamp_sec: float
    frame_id: str = ""


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray      # Nx3 float32
    stamp_sec: float
    frame_id: str = ""   # coordinate frame of the points


@dataclass(frozen=True)
class Observation:
    """
    Minimal thing mapping needs, no ROS.

    Provide any subset:
      - rgb + intrinsics (+ pose) -> pipeline can run depth->cloud->costmap
      - depth + intrinsics (+ pose) -> pipeline can run cloud->costmap
      - cloud (+ pose) -> pipeline can run costmap
    """
    intrinsics: Optional[Intrinsics] = None
    pose_map_base: Optional[PoseSE3] = None  # base in map/world

    rgb: Optional[RGBFrame] = None
    depth: Optional[DepthFrame] = None
    cloud: Optional[PointCloud] = None

@dataclass
class Detection2D:
    label: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2
    frame_w: int
    frame_h: int

@dataclass
class Detection3D:
    label: str
    score: float
    xyz_world: Tuple[float, float, float]
    xyz_cam: Optional[Tuple[float, float, float]] = None

