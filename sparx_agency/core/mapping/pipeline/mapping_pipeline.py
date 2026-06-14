# sparx_agency/core/mapping/pipeline/mapping_pipeline.py

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sparx_agency.core.common.spatial_math import rpy_deg_to_R_base
# matplotlib imports moved inside methods to avoid global initialization issues
# in environments with numpy/system library mismatches.

from sparx_agency.core.common.types import Observation, Intrinsics
from sparx_agency.core.mapping.interfaces.depth_model import DepthModel
from sparx_agency.core.mapping.interfaces.cloud_generator import CloudGenerator
from sparx_agency.core.mapping.interfaces.costmap import Costmap


@dataclass
class MappingPipelineConfig:
    # Filtering in BASE coordinates (after optical->base conversion if needed)
    z_min: float = -1.5
    z_max: float = 1.0
    range_min: float = 0.5
    range_max: float = 15.0  # DA depth gets less reliable far away

    # Downsample pixels for depth->cloud (legacy path)
    stride: int = 2

    # If obs.cloud is in camera optical frame: X right, Y down, Z forward
    cloud_is_optical: bool = True

    # If True, resets tmp costmap each step and updates it
    use_tmp_costmap: bool = True

    # Apply base-frame filters to cloud (recommended)
    filter_cloud: bool = True

    debug: bool = False


class PinholeCloudGenerator(CloudGenerator):
    """
    Legacy depth->cloud path.
    Returns points in BASE-ish convention:
      X forward, Y left, Z up
    """

    def __init__(
        self,
        stride: int = 2,
        cam_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        t_base: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.stride = max(1, int(stride))
        self.cam_rpy_deg = cam_rpy_deg
        self.t_base = np.array(t_base, dtype=np.float32)

        r, p, y = cam_rpy_deg
        self.R_base = rpy_deg_to_R_base(r, p, y)  # 3x3

    def depth_to_cloud_to_base_xyz(self, depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
        if depth_m is None:
            return np.zeros((0, 3), dtype=np.float32)

        depth_m = np.asarray(depth_m)
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

        # Base convention:
        # X forward = depth
        # Y left    = -(x_right)
        # Z up      = -(y_down)
        y_left = -(xv - intr.cx) * d / intr.fx
        z_up = -(yv - intr.cy) * d / intr.fy
        x_fwd = d

        pts = np.stack([x_fwd, y_left, z_up], axis=1).astype(np.float32)
        return pts

    def angle_correction_translation(self, pts: np.ndarray) -> np.ndarray:
        # Rotate points in base frame
        pts = (self.R_base @ pts.T).T
        # Translate (camera position in base)
        pts += self.t_base[None, :]
        return pts

    def transform_points(self, pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (pts @ R.T) + t[None, :]


def optical_xyz_to_base_xyz(pts_optical: np.ndarray) -> np.ndarray:
    """
    ROS camera optical frame:
      X right, Y down, Z forward
    Base frame we use in mapping:
      X forward, Y left, Z up
    """
    if pts_optical.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # optical: [x_right, y_down, z_forward]
    x_fwd = pts_optical[:, 2]
    y_left = -pts_optical[:, 0]
    z_up = pts_optical[:, 1]
    return np.stack([x_fwd, y_left, z_up], axis=1).astype(np.float32)


class MappingPipeline:
    """
    ROS-free orchestrator:
      Observation -> (cloud/depth/rgb)->cloud -> filter -> costmap update

    Supports:
      - accumulated costmap (always updated)
      - optional tmp costmap (reset every step, then updated)
    """

    def __init__(
        self,
        costmap: Costmap,
        depth_model: Optional[DepthModel] = None,
        cloud_generator: Optional[CloudGenerator] = None,
        cfg: Optional[MappingPipelineConfig] = None,
        costmap_tmp: Optional[Costmap] = None,
    ):
        # Backward compatible name: "costmap" is the accumulated map
        self.costmap_accum = costmap
        self.costmap_tmp = costmap_tmp

        self.depth_model = depth_model
        self.cloud_generator = cloud_generator or PinholeCloudGenerator()
        self.cfg = cfg or MappingPipelineConfig()

        self.last_depth: Optional[np.ndarray] = None
        self.last_cloud_local: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.last_cloud_global: np.ndarray = np.zeros((0, 3), dtype=np.float32)

        self.frame_count = 0
        self.save_interval = 100

    def _filter_cloud_base(self, cloud_base: np.ndarray) -> np.ndarray:
        if cloud_base is None or cloud_base.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        x = cloud_base[:, 0]
        y = cloud_base[:, 1]
        z = cloud_base[:, 2]
        r = np.sqrt(x * x + y * y)

        m = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(z)
            & (r >= self.cfg.range_min)
            & (r <= self.cfg.range_max)
            & (z >= self.cfg.z_min)
            & (z <= self.cfg.z_max)
        )
        out = cloud_base[m]
        if out.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return out.astype(np.float32)

    def step(self, obs: Observation) -> None:
        self.last_cloud_local = np.zeros((0, 3), dtype=np.float32)
        self.last_cloud_global = np.zeros((0, 3), dtype=np.float32)

        intr = obs.intrinsics
        cloud_base = None

        # 1) Preferred path: v3 TRT PointCloud2 already exists -> obs.cloud.xyz
        if obs.cloud is not None:
            pts = np.asarray(obs.cloud.xyz, dtype=np.float32).reshape((-1, 3))
            if self.cfg.cloud_is_optical:
                cloud_base = optical_xyz_to_base_xyz(pts)
            else:
                cloud_base = pts

        # 2) Depth provided -> backproject
        elif obs.depth is not None:
            if intr is None:
                raise ValueError("Observation.depth provided but intrinsics is None")
            self.last_depth = np.asarray(obs.depth.depth_m, dtype=np.float32)
            cloud_base = self.cloud_generator.depth_to_cloud_to_base_xyz(self.last_depth, intr)

        # 3) RGB provided -> infer depth -> backproject (legacy)
        elif obs.rgb is not None:
            if self.depth_model is None or intr is None:
                raise ValueError("Observation.rgb provided but depth_model or intrinsics is None")
            t0 = time.perf_counter()
            self.last_depth = self.depth_model.infer_depth(obs.rgb.image).astype(np.float32)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Depth inference took {dt_ms:.2f} ms")

            t0 = time.perf_counter()
            cloud_base = self.cloud_generator.depth_to_cloud_to_base_xyz(self.last_depth, intr)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Cloud generation took {dt_ms:.2f} ms")

        else:
            return

        if cloud_base is None or cloud_base.shape[0] == 0:
            return

        if self.cfg.filter_cloud:
            cloud_base = self._filter_cloud_base(cloud_base)
            if cloud_base.shape[0] == 0:
                return

        self.last_cloud_local = cloud_base

        # Transform to global if pose exists
        if obs.pose_map_base is not None:
            cloud_global = obs.pose_map_base.transform_points(cloud_base)
            sensor_origin_global = obs.pose_map_base.t
        else:
            cloud_global = cloud_base
            sensor_origin_global = np.zeros(3, dtype=np.float32)

        self.last_cloud_global = cloud_global

        # TMP costmap: reset each step, then update
        if self.costmap_tmp is not None and self.cfg.use_tmp_costmap:
            self.costmap_tmp.reset()
            self.costmap_tmp.update_from_cloud(cloud_global, sensor_origin_global)

        # ACCUM costmap: always update
        self.costmap_accum.update_from_cloud(cloud_global, sensor_origin_global)

        if self.last_depth is not None and self.cfg.debug:
            self.visualize_depth_errors(self.last_depth, cloud_base)

    def visualize_depth_errors(self, raw_depth: np.ndarray, cloud_base: np.ndarray) -> None:
        self.frame_count += 1
        if self.frame_count % self.save_interval != 0:
            return

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Depth Geometry Analysis - Frame {self.frame_count}")

        axs[0, 0].hist(raw_depth.ravel(), bins=100, color="gray")
        axs[0, 0].set_title("Raw Depth Values")
        axs[0, 0].set_ylabel("Frequency")

        axs[0, 1].hist(cloud_base[:, 0], bins=100, color="green", alpha=0.7)
        axs[0, 1].set_title("Base X: Forward")
        axs[0, 1].set_xlabel("Meters")

        axs[1, 0].hist(cloud_base[:, 1], bins=100, color="red", alpha=0.7)
        axs[1, 0].set_title("Base Y: Left/Right")
        axs[1, 0].set_xlabel("Meters")

        axs[1, 1].hist(cloud_base[:, 2], bins=100, color="blue", alpha=0.7)
        axs[1, 1].set_title("Base Z: Up/Down")
        axs[1, 1].set_xlabel("Meters")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save as PNG - check your terminal's current directory for these files
        time_str = datetime.datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
        filename = f"depth_axes_diagnostic_{self.frame_count:04d}_{time_str}.png"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Diagnostic saved to {filename}")