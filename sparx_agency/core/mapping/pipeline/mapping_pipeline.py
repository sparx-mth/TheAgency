# sparx_agency/core/mapping/pipeline/mapping_pipeline.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from sparx_agency.core.common.types import Observation, RGBFrame, Intrinsics
from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.core.mapping.interfaces.cloud_generator import CloudGenerator
from sparx_agency.core.mapping.interfaces.costmap import Costmap


@dataclass
class MappingPipelineConfig:
    # Filtering in BASE coordinates (after optical->base conversion)
    z_min: float = -1.5
    z_max: float = 1.0  # Ignore "lines" high in the sky (e.g., above 3m)
    range_min: float = 0.5
    range_max: float = 15.0  # Depth Anything V2 accuracy drops after 15m

    # Downsample pixels for speed
    stride: int = 2

    # If your depth->cloud is optical (x right, y down, z forward)
    cloud_is_optical: bool = True


class PinholeCloudGenerator(CloudGenerator):
    def __init__(self, stride: int = 2):
        self.stride = max(1, int(stride))

    def depth_to_cloud(self, depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
        if depth_m is None:
            return np.zeros((0, 3), dtype=np.float32)

        depth_m = np.asarray(depth_m)
        if depth_m.ndim != 2:
            raise ValueError(f"depth_m must be HxW, got shape={depth_m.shape}")

        H, W = depth_m.shape
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
        z = d  # forward in optical

        return np.stack([x, y, z], axis=1).astype(np.float32)


def optical_xyz_to_base_xyz(pts_optical: np.ndarray) -> np.ndarray:
    """
    Converts Camera Optical (X-Right, Y-Down, Z-Forward)
    to Robot Base (X-Forward, Y-Left, Z-Up)
    """
    p = np.asarray(pts_optical, dtype=np.float32)
    x_opt = pts_optical[:, 0]
    y_opt = pts_optical[:, 1]
    z_opt = pts_optical[:, 2]

    # RE-MAP THE AXES:
    base_x = z_opt  # Forward is Depth
    base_y = -x_opt  # Left is -Right
    base_z = -y_opt  # Up is -Down

    return np.stack([base_x, base_y, base_z], axis=1).astype(np.float32)


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
        self.cloud_generator = cloud_generator or PinholeCloudGenerator()
        self.cfg = cfg or MappingPipelineConfig()

        self.last_depth: Optional[np.ndarray] = None
        self.last_cloud_cam: np.ndarray = np.zeros((0, 3), dtype=np.float32)  # NEVER None
        self.last_cloud_base: np.ndarray = np.zeros((0, 3), dtype=np.float32)

    def step(self, obs: Observation) -> None:
        # Reset debug outputs each step (still not None)
        self.last_cloud_cam = np.zeros((0, 3), dtype=np.float32)
        self.last_cloud_base = np.zeros((0, 3), dtype=np.float32)

        intr = obs.intrinsics
        stamp_sec = None

        # 1) Get depth
        if obs.cloud is not None:
            cloud_cam = np.asarray(obs.cloud.xyz, dtype=np.float32).reshape((-1, 3))
            stamp_sec = obs.cloud.stamp_sec
            self.last_cloud_cam = cloud_cam
        elif obs.depth is not None:
            if intr is None:
                raise ValueError("Observation.depth provided but intrinsics is None")
            depth = np.asarray(obs.depth.depth_m, dtype=np.float32)
            d_min = depth.min()
            d_max = depth.max()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-7)
            max_depth = 20.0
            depth_meters = max_depth * (1.0 - depth_norm)

            self.last_depth = depth_meters.astype(np.float32)
            cloud_cam = self.cloud_generator.depth_to_cloud(self.last_depth, intr)
            # self.last_depth = depth
            # cloud_cam = self.cloud_generator.depth_to_cloud(depth, intr)
            self.last_cloud_cam = cloud_cam
        elif obs.rgb is not None:
            if self.depth_model is None:
                raise ValueError("Observation.rgb provided but depth_model is None")
            if intr is None:
                raise ValueError("Observation.rgb provided but intrinsics is None")

            stamp_sec = obs.rgb.stamp_sec
            # Get raw 0-255 disparity from model
            depth_m = self.depth_model.infer_depth(obs.rgb.image)

            expected_h, expected_w = intr.height, intr.width
            if depth_m.shape[:2] != (expected_h, expected_w):
                depth_m = cv2.resize(depth_m, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

            # 2. DO NOT apply the 20.0 * (1 - depth/255) formula here again.
            # Just use the result from the model.
            self.last_depth = depth_m.astype(np.float32)

            # 3. Generate Cloud
            cloud_cam = self.cloud_generator.depth_to_cloud(self.last_depth, intr)
            self.last_cloud_cam = cloud_cam
        else:
            return

        if self.last_cloud_cam.shape[0] == 0:
            return

        # 1. Convert Optical (Cam) -> Base (Robot FLU)
        # Using your mapping for the Gazebo drone link
        cloud_base = optical_xyz_to_base_xyz(self.last_cloud_cam)
        self.last_cloud_base = cloud_base

        # 2. Extract Pose and Transform to World
        if obs.pose_map_base is not None:
            # It rotates and translates the points into the 'odom' frame.
            cloud_odom = obs.pose_map_base.transform_points(cloud_base)

            # Update costmap with the WORLD points
            self.costmap.update_from_cloud(
                cloud_xyz=cloud_odom,
                sensor_origin=obs.pose_map_base.t
            )
        else:
            # Fallback: if no pose, we can only update relative to the drone
            # Use identity pose (0,0,0) as sensor origin
            self.costmap.update_from_cloud(
                cloud_xyz=cloud_base,
                sensor_origin=np.zeros(3, dtype=np.float32),
            )
