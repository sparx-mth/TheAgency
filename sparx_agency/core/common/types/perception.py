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


@dataclass(frozen=True)
class Track2D:
    """One frame of a tracked target's 2D image state.

    Produced by a visual tracker (e.g. the Lucas-Kanade box tracker) and consumed
    by the visual-servo control law. Unlike :class:`Detection2D` (a single
    detector output) a track is *persistent* across frames and carries validity,
    image-plane velocity, and a ``predicted`` flag so a controller can tell a
    measured box from one carried by a motion model through a brief dropout.

    The box is stored subpixel (float) ``x1,y1,x2,y2`` in image pixels, origin
    top-left, ``+x`` right, ``+y`` down.

    Attributes:
        label: Class label of the tracked object.
        bbox_xyxy: ``(x1, y1, x2, y2)`` in pixels (float, subpixel).
        frame_w: Image width (px) the box lives in.
        frame_h: Image height (px) the box lives in.
        valid: False when the tracker has lost lock this frame (box is stale).
        n_matches: Surviving tracked features (LK) that produced the box; 0 when
            the box is predicted or freshly seeded.
        score: Detector confidence of the detection that last (re)seeded the track.
        velocity_px: Bounding-box centre velocity ``(vx, vy)`` in px/s, image frame.
        predicted: True when ``bbox_xyxy`` came from the motion model rather than
            a measurement (i.e. the object was not directly observed this frame).
        age_s: Seconds since the track was first seeded.
    """

    label: str
    bbox_xyxy: Tuple[float, float, float, float]
    frame_w: int
    frame_h: int
    valid: bool = True
    n_matches: int = 0
    score: float = 0.0
    velocity_px: Tuple[float, float] = (0.0, 0.0)
    predicted: bool = False
    age_s: float = 0.0

    @property
    def cx(self) -> float:
        """Box centre x (px)."""
        return 0.5 * (self.bbox_xyxy[0] + self.bbox_xyxy[2])

    @property
    def cy(self) -> float:
        """Box centre y (px)."""
        return 0.5 * (self.bbox_xyxy[1] + self.bbox_xyxy[3])

    @property
    def w(self) -> float:
        """Box width (px), >= 0."""
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def h(self) -> float:
        """Box height (px), >= 0."""
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        """Box area (px^2)."""
        return self.w * self.h

    @property
    def area_frac(self) -> float:
        """Box area as a fraction of the whole image (a monotone proximity proxy)."""
        denom = float(max(1, self.frame_w) * max(1, self.frame_h))
        return self.area / denom

