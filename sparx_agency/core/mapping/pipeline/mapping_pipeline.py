from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from sparx_agency.core.common.types import Observation, RGBFrame, PoseSE3, Intrinsics
from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.core.mapping.interfaces.cloud_generator import CloudGenerator
from sparx_agency.core.mapping.interfaces.costmap import Costmap


@dataclass
class MappingPipelineConfig:
    # Filtering in BASE coordinates after conversion (meters)
    z_min: float = -10.0
    z_max: float = 30.0  # forward?
    range_min: float = 0.3
    range_max: float = 50.0

    # Depth->cloud downsample
    stride: int = 2

    # Camera convention from pinhole projection:
    # optical: x right, y down, z forward
    cloud_is_optical: bool = True

    # Optional camera offset in base frame (meters)
    # (If you don't know, keep zeros; for local obstacle avoidance it's usually fine.)
    cam_in_base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


class PinholeCloudGenerator(CloudGenerator):
    def __init__(self, stride: int = 2):
        self.stride = max(1, int(stride))

    def depth_to_cloud(self, depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
        H, W = depth_m.shape[:2]
        s = self.stride

        ys = np.arange(0, H, s, dtype=np.int32)
        xs = np.arange(0, W, s, dtype=np.int32)
        xv, yv = np.meshgrid(xs, ys)

        d = depth_m[yv, xv].astype(np.float32)
        m = np.isfinite(d) & (d > 0.0)
        if not np.any(m):
            return np.zeros((0, 3), dtype=np.float32)

        xv = xv[m].astype(np.float32)
        yv = yv[m].astype(np.float32)
        d = d[m]

        x = (xv - intr.cx) * d / intr.fx
        y = (yv - intr.cy) * d / intr.fy
        z = d
        return np.stack([x, y, z], axis=1).astype(np.float32)


def optical_to_base(pts_optical: np.ndarray) -> np.ndarray:
    """
    Optical (x right, y down, z forward) -> Base (x forward, y left, z up)
      x_base =  z_opt
      y_base = -x_opt
      z_base = -y_opt
    """
    x = pts_optical[:, 0]
    y = pts_optical[:, 1]
    z = pts_optical[:, 2]
    return np.stack([z, -x, -y], axis=1).astype(np.float32)


class MappingPipeline:
    """
    ROS-free orchestrator:
      Observation -> (depth)->cloud -> filter -> costmap update
    """

    def __init__(
        self,
        costmap: Costmap,
        depth_model: Optional[DepthModel] = None,
        cloud_generator: Optional[CloudGenerator] = None,
        cfg: Optional[MappingPipelineConfig] = None,
    ):
        self.costmap = costmap
        self.depth_model = depth_model
        self.cfg = cfg or MappingPipelineConfig()
        self.cloud_generator = cloud_generator or PinholeCloudGenerator(stride=self.cfg.stride)

        # For debugging / adapters
        self.last_depth: Optional[np.ndarray] = None
        self.last_cloud_optical: Optional[np.ndarray] = None
        self.last_cloud_base: Optional[np.ndarray] = None

    def step(self, obs: Observation) -> None:
        intr = obs.intrinsics
        stamp_sec = None

        # 1) Obtain a cloud in OPTICAL camera coords
        cloud_opt = None
        if obs.cloud is not None:
            cloud_opt = obs.cloud.xyz
            stamp_sec = obs.cloud.stamp_sec

        elif obs.depth is not None:
            if intr is None:
                raise ValueError("Observation.depth provided but intrinsics is None")
            cloud_opt = self.cloud_generator.depth_to_cloud(obs.depth.depth_m, intr)
            stamp_sec = obs.depth.stamp_sec
            self.last_depth = obs.depth.depth_m

        elif obs.rgb is not None:
            if self.depth_model is None:
                raise ValueError("Observation.rgb provided but depth_model is None")
            if intr is None:
                raise ValueError("Observation.rgb provided but intrinsics is None")
            depth = self.depth_model.infer_depth(obs.rgb.image)
            cloud_opt = self.cloud_generator.depth_to_cloud(depth, intr)
            stamp_sec = obs.rgb.stamp_sec
            self.last_depth = depth

        else:
            return

        if cloud_opt is None or cloud_opt.shape[0] == 0:
            self.last_cloud_optical = cloud_opt
            self.last_cloud_base = None
            return

        self.last_cloud_optical = cloud_opt

        # 2) Convert OPTICAL -> BASE
        if self.cfg.cloud_is_optical:
            cloud_base = optical_to_base(cloud_opt)
        else:
            cloud_base = cloud_opt.astype(np.float32, copy=False)

        # apply optional camera offset in base
        ox, oy, oz = self.cfg.cam_in_base_xyz
        if (ox != 0.0) or (oy != 0.0) or (oz != 0.0):
            cloud_base = cloud_base + np.array([ox, oy, oz], dtype=np.float32)

        # 3) Filter in BASE coordinates
        x_r, y_d, z_f = cloud_base[:, 0], cloud_base[:, 1], cloud_base[:, 2]
        # Convert optical -> base-like axes before filtering.
        # optical:  x right, y down, z forward
        # base:     x forward, y left, z up
        cloud_base_updated = np.stack([z_f, -x_r, -y_d], axis=1).astype(np.float32)

        fwd = cloud_base_updated[:, 0]  # forward meters
        left = cloud_base_updated[:, 1]  # left meters
        up = cloud_base_updated[:, 2]  # up meters

        # Use forward range (or planar range) – not sqrt(x^2+y^2) from optical
        planar = np.hypot(fwd, left)

        m = (
                (fwd >= self.cfg.range_min) & (fwd <= self.cfg.range_max) &
                (up >= self.cfg.z_min) & (up <= self.cfg.z_max)
        )
        cloud_base_updated = cloud_base_updated[m]
        self.last_cloud_base = cloud_base_updated

        if cloud_base_updated.shape[0] == 0:
            return

        # 4) Update costmap:
        #    Prefer raycast update if the costmap supports it and we have pose.
        if obs.pose_map_base is not None and hasattr(self.costmap, "update_from_cloud_base_raycast"):
            R = obs.pose_map_base.R
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            rx = float(obs.pose_map_base.t[0])
            ry = float(obs.pose_map_base.t[1])
            self.costmap.update_from_cloud_base_raycast(
                cloud_base=cloud_base,
                robot_xy_yaw=(rx, ry, yaw),
                stamp_sec=stamp_sec,
            )
            return

        # Fallback: transform endpoints to map and mark occupied only (Option A).
        if obs.pose_map_base is None:
            pts_map = cloud_base
        else:
            pts_map = obs.pose_map_base.transform_points(cloud_base)

        self.costmap.update_from_points_xy(pts_map[:, :2].astype(np.float32), stamp_sec=stamp_sec)
