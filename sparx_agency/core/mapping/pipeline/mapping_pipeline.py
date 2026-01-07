from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from sparx_agency.core.common.types import Observation, Intrinsics
from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.core.mapping.interfaces.cloud_generator import CloudGenerator
from sparx_agency.core.mapping.interfaces.costmap import Costmap, GridSpec


@dataclass
class MappingPipelineConfig:
    # Filtering in "base-like" frame (x forward, y left, z up)
    z_min_m: float = -20.0
    z_max_m: float =  20.0
    range_min_m: float = 0.3
    range_max_m: float = 50.0     # forward 5-10m (as you asked)

    # Downsample for speed (set 1 to match depth_image_proc density)
    stride: int = 1

    # Input cloud convention
    cloud_is_optical: bool = True  # depth->cloud outputs x right, y down, z forward


class PinholeCloudGenerator(CloudGenerator):
    """
    Default Depth->Cloud implementation (no ROS).
    Assumes depth is meters.
    """

    def __init__(self, stride: int = 1, half_pixel: bool = False):
        self.stride = max(1, int(stride))
        self.half_pixel = bool(half_pixel)

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

        # depth_image_proc effectively uses the camera_info model;
        # keeping half_pixel=False is usually the correct match for ROS camera_info.
        if self.half_pixel:
            xv = xv + 0.5
            yv = yv + 0.5

        x = (xv - intr.cx) * d / intr.fx
        y = (yv - intr.cy) * d / intr.fy
        z = d  # forward

        return np.stack([x, y, z], axis=1).astype(np.float32)


class MappingPipeline:
    """
    ROS-free orchestrator:
      - takes Observation
      - updates costmap
      - returns (spec, grid) for adapters to publish
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
        self.cloud_generator = cloud_generator or PinholeCloudGenerator(stride=self.cfg.stride, half_pixel=False)

        self.last_depth: Optional[np.ndarray] = None
        self.last_cloud_optical: Optional[np.ndarray] = None
        self.last_cloud_base: Optional[np.ndarray] = None

    @staticmethod
    def _optical_to_base(cloud_xyz: np.ndarray) -> np.ndarray:
        """
        optical: x right, y down, z forward
        base-like: x forward, y left, z up
        """
        x_r = cloud_xyz[:, 0]
        y_d = cloud_xyz[:, 1]
        z_f = cloud_xyz[:, 2]

        x_fwd = z_f
        y_left = -x_r
        z_up = -y_d
        return np.stack([x_fwd, y_left, z_up], axis=1).astype(np.float32)

    def _filter_base(self, cloud_base: np.ndarray) -> np.ndarray:
        x = cloud_base[:, 0]  # forward
        y = cloud_base[:, 1]  # left
        z = cloud_base[:, 2]  # up

        rng = np.sqrt(x * x + y * y)

        m = (
            (z >= self.cfg.z_min_m) & (z <= self.cfg.z_max_m) &
            (rng >= self.cfg.range_min_m) & (rng <= self.cfg.range_max_m) &
            (x >= 0.0) & (x <= self.cfg.range_max_m)  # only forward points
        )
        return cloud_base[m]

    def step(self, obs: Observation) -> Optional[Tuple[GridSpec, np.ndarray]]:
        intr = obs.intrinsics
        stamp_sec = None

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
            self.last_depth = depth
            cloud_opt = self.cloud_generator.depth_to_cloud(depth, intr)
            stamp_sec = obs.rgb.stamp_sec

        else:
            return None

        if cloud_opt is None or cloud_opt.shape[0] == 0:
            self.last_cloud_optical = cloud_opt
            return None

        self.last_cloud_optical = cloud_opt

        # Convert to base-like coordinates before filtering/costmap
        cloud_base = self._optical_to_base(cloud_opt) if self.cfg.cloud_is_optical else cloud_opt
        cloud_base = self._filter_base(cloud_base)
        self.last_cloud_base = cloud_base

        if cloud_base.shape[0] == 0:
            return None

        # raycast using robot pose (x,y,yaw) if available
        if obs.pose_map_base is not None:
            R = obs.pose_map_base.R
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            rx = float(obs.pose_map_base.t[0])
            ry = float(obs.pose_map_base.t[1])

            if hasattr(self.costmap, "update_from_cloud_cam_raycast"):
                # costmap expects cloud in base-like frame and robot pose in map
                self.costmap.update_from_cloud_cam_raycast(
                    cloud_base, (rx, ry, yaw), stamp_sec=stamp_sec
                )
            else:
                # Fallback: just mark endpoints occupied (not great)
                pts_xy = cloud_base[:, :2].astype(np.float32)
                pts_xy[:, 0] += rx
                pts_xy[:, 1] += ry
                self.costmap.update_from_points_xy(pts_xy, stamp_sec=stamp_sec)
        else:
            # Demo fallback: no pose, no raycast
            pts_xy = cloud_base[:, :2].astype(np.float32)
            self.costmap.update_from_points_xy(pts_xy, stamp_sec=stamp_sec)

        return self.costmap.get_grid()
