# sparx_agency/core/mapping/pipeline/mapping_pipeline.py

from __future__ import annotations

import datetime
import time
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
    debug: bool = False




class PinholeCloudGenerator(CloudGenerator):
    def __init__(self, stride: int = 2):
        self.stride = max(1, int(stride))

    def depth_to_cloud_to_base_xyz(self, depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
        if depth_m is None:
            return np.zeros((0, 3), dtype=np.float32)

        depth_m = np.asarray(depth_m)
        H, W = depth_m.shape
        s = self.stride

        # 1. Create integer-based meshgrids
        ys = np.arange(0, H, s, dtype=np.int32)
        xs = np.arange(0, W, s, dtype=np.int32)
        xv, yv = np.meshgrid(xs, ys)

        xv = xv.astype(np.int32)
        yv = yv.astype(np.int32)

        # 2. Sample depth using integer indices
        d = depth_m[yv, xv].astype(np.float32)

        # 3. Create a mask and handle the empty-case immediately
        # This stops the 'x referenced before assignment' error
        m = np.isfinite(d) & (d > 0.0)
        if not np.any(m):
            return np.zeros((0, 3), dtype=np.float32)
        try:
            xv = xv[m].astype(np.float32)
            yv = yv[m].astype(np.float32)
            d = d[m]
        except Exception as e:
            print(f"Error: xv, yv, d is {xv}, {yv}, {d} with error: {e}")



        y =  -(xv - intr.cx) * d / intr.fx
        z =  -(yv - intr.cy) * d / intr.fy
        x = d  # forward in optical

        return np.stack([x, y, z], axis=1).astype(np.float32)


def optical_xyz_to_base_xyz(pts_optical: np.ndarray) -> np.ndarray:

    # Merged with depth_to_cloud_to_base_xyz() in MappingPipeline.step()

    # pts_optical[:, 0] is X (Right)
    # pts_optical[:, 1] is Y (Down)
    # pts_optical[:, 2] is Z (Forward/Depth)

    # 1. FORWARD must be Positive X (Red in Rviz)
    # We take the Depth (Z_opt) and put it in X_base
    base_x = pts_optical[:, 0] # Z

    # 2. LEFT must be Positive Y (Green in Rviz)
    # In optical, X is Right. So Left is -X.
    base_y = pts_optical[:, 1] # -X

    # 3. UP must be Positive Z (Blue in Rviz)
    # In optical, Y is Down. So Up is -Y.

    base_z = pts_optical[:, 2] # -Y

    # LOGGING: If this is working, Max Base X should be ~10.0 to 15.0
    # and Max Base Y should be ~half of that.
    print(f"DEBUG: Max X (Fwd): {np.max(base_x):.2f}, Max Y (Side): {np.max(base_y):.2f}")

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
        self.last_cloud_local: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.last_cloud_global: np.ndarray = np.zeros((0, 3), dtype=np.float32)


        self.frame_count = 0
        self.save_interval = 100  # Adjust X here
        self._last_indices = None

    def step(self, obs: Observation) -> None:
        # 1. Initialization and Reset
        self.last_cloud_local = np.zeros((0, 3), dtype=np.float32)
        self.last_cloud_global = np.zeros((0, 3), dtype=np.float32)

        intr = obs.intrinsics

        # 2. Source Selection (Cloud vs Depth vs RGB)
        cloud_local_corr = None

        if obs.cloud is not None:
            cloud_local_corr = np.asarray(obs.cloud.xyz, dtype=np.float32).reshape((-1, 3))

        elif obs.depth is not None:
            if intr is None:
                raise ValueError("Observation.depth provided but intrinsics is None")
            self.last_depth = np.asarray(obs.depth.depth_m, dtype=np.float32)
            cloud_local_corr = self.cloud_generator.depth_to_cloud_to_base_xyz(self.last_depth, intr)

        elif obs.rgb is not None:
            if self.depth_model is None or intr is None:
                raise ValueError("Observation.rgb provided but depth_model or intrinsics is None")

            # Infer depth from RGB and generate cloud
            t0 = time.perf_counter()
            self.last_depth = self.depth_model.infer_depth(obs.rgb.image).astype(np.float32)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Depth inference took {dt_ms:.2f} ms")

            t0 = time.perf_counter()
            cloud_local_corr = self.cloud_generator.depth_to_cloud_to_base_xyz(self.last_depth, intr)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Cloud generation took {dt_ms:.2f} ms")
        self.last_cloud_local = cloud_local_corr


        if self.last_depth is not None and self.cfg.debug:
            self.visualize_depth_errors(self.last_depth, cloud_local_corr)

        # 4. Costmap Update (World vs Local)
        if obs.pose_map_base is not None:
            # Transform to 'odom/map' frame and update
            t0 = time.perf_counter()
            cloud_odom = obs.pose_map_base.transform_points(cloud_local_corr)
            self.costmap.update_from_cloud(cloud_odom, obs.pose_map_base.t)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Costmap update took {dt_ms:.2f} ms")
            self.last_cloud_global = cloud_odom

        else:
            # Fallback: update relative to sensor origin
            self.costmap.update_from_cloud(
                cloud_xyz=cloud_local_corr,
                sensor_origin=np.zeros(3, dtype=np.float32),
            )
            self.last_cloud_local = cloud_local_corr

        # 2. Extract Pose and Transform to World
        if obs.pose_map_base is not None:
            # It rotates and translates the points into the 'odom' frame.
            cloud_odom = obs.pose_map_base.transform_points(cloud_local_corr)
            # Update costmap with the WORLD points
            self.costmap.update_from_cloud(cloud_odom, obs.pose_map_base.t)
            self.last_cloud_global = cloud_odom
        else:
            # Fallback: if no pose, we can only update relative to the drone
            # Use identity pose (0,0,0) as sensor origin
            try:
                self.costmap.update_from_cloud(
                    cloud_xyz=cloud_local_corr,
                    sensor_origin=np.zeros(3, dtype=np.float32),
                )
                self.last_cloud_global = cloud_local_corr
            except Exception as e:
                print(f"Error: costmap.update_from_cloud is {e}")

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
        time_str = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
        filename = f"depth_axes_diagnostic_{self.frame_count:04d}_{time_str}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Diagnostic saved to {filename}")


