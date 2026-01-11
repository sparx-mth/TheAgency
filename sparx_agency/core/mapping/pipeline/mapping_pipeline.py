# sparx_agency/core/mapping/pipeline/mapping_pipeline.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot
from matplotlib import pyplot as plt

from sparx_agency.core.common.types import Observation, RGBFrame, Intrinsics
from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.core.mapping.interfaces.cloud_generator import CloudGenerator
from sparx_agency.core.mapping.interfaces.costmap import Costmap
from sparx_agency.core.mapping.costmap.sanity_check_costmap import SanityCheckCostmap


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

        # Try the standard projective form used in the Depth Anything example:
        x = (xv - intr.cx) / intr.fx  # Normalized X
        y = (yv - intr.cy) / intr.fy  # Normalized Y
        points = np.stack([x * d, y * d, d], axis=1).astype(np.float32)  # Standard projection Z is forward in Optical

        return points # np.stack([x, y, z], axis=1).astype(np.float32)


def optical_xyz_to_base_xyz(pts_optical: np.ndarray) -> np.ndarray:
    """
    Converts Camera Optical (X-Right, Y-Down, Z-Forward)
    to Robot Base (X-Forward, Y-Left, Z-Up)
    """
    # pts_optical[:, 0] = X (Right), [:, 1] = Y (Down), [:, 2] = Z (Depth)
    x_opt = pts_optical[:, 0]
    y_opt = pts_optical[:, 1]
    z_opt = pts_optical[:, 2]

    # STANDARD FLU CONVENTION:
    base_x = z_opt  # Depth is Forward
    base_y = -x_opt  # Right is Left
    base_z = -y_opt  # Down is Up

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

        self.frame_count = 0
        self.save_interval = 50  # Adjust X here


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
            # depth = np.asarray(obs.depth.depth_m, dtype=np.float32)
            # d_min = depth.min()
            # d_max = depth.max()
            # depth_norm = (depth - d_min) / (d_max - d_min + 1e-7)
            # max_depth = 20.0
            # depth_meters = max_depth / (depth_norm + 0.01)
            raw_depth = np.asarray(obs.depth.depth_m, dtype=np.float32)
            self.last_depth = raw_depth * 20.0
            # self.last_depth = depth_meters.astype(np.float32)
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
        if self.last_depth is not None:
            raw_depth = np.asarray(self.last_depth, dtype=np.float32)
            self.visualize_depth_errors(raw_depth, cloud_base)
        else:
            print("No depth data available for visualization")


        # 2. Extract Pose and Transform to World
        if obs.pose_map_base is not None:
            # print(f"")
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

    def visualize_depth_errors(self, raw_depth, cloud_base):
        self.frame_count += 1
        if self.frame_count % self.save_interval != 0:
            return

        # Use a clear figure size
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Depth Geometry Analysis - Frame {self.frame_count}")

        # 1. Image Depth (The 'D' values)
        # This helps see if your depth model is saturated
        axs[0, 0].hist(raw_depth.ravel(), bins=100, color='gray')
        axs[0, 0].set_title("Raw Model Output (D values)")
        axs[0, 0].set_ylabel("Frequency")

        # 2. Forward Distance (Robot X)
        # Should show clusters where walls/objects are
        axs[0, 1].hist(cloud_base[:, 0], bins=100, color='green', alpha=0.7)
        axs[0, 1].set_title("Base X: Forward (Depth)")
        axs[0, 1].set_xlabel("Meters")

        # 3. Lateral Distance (Robot Y)
        # Should look roughly centered around 0
        axs[1, 0].hist(cloud_base[:, 1], bins=100, color='red', alpha=0.7)
        axs[1, 0].set_title("Base Y: Left/Right (Lateral)")
        axs[1, 0].set_xlabel("Meters")

        # 4. Vertical Distance (Robot Z)
        # THE SMOKING GUN: If the cone is diving, you will see a massive
        # spread into negative numbers here.
        axs[1, 1].hist(cloud_base[:, 2], bins=100, color='blue', alpha=0.7)
        axs[1, 1].set_title("Base Z: Up/Down (Height)")
        axs[1, 1].set_xlabel("Meters")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save as PNG - check your terminal's current directory for these files
        filename = f"depth_diagnostic_{self.frame_count:04d}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Diagnostic saved to {filename}")

