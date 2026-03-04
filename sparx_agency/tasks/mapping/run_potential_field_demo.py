#!/usr/bin/env python3
"""
ROS2 test node:
- Subscribes to DepthAnything v3 TRT PointCloud2 output
- Builds:
  - tmp occupancy/prob map (reset each frame)
  - accumulated occupancy/prob map
  - repulsive potential field (derived from accumulated prob map)

Core mapping stays ROS-free; this file is the ROS2 adapter / test harness.

Expected in your repo:
- LogOddsGridCostmap (your existing one)  OR ProbabilisticGridCostmap
- PotentialFieldLayer (the ROS-free layer we added)

Adjust imports below to match your exact paths.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridCostmap, LogOddsGridConfig  # <-- adjust if needed
from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer  # <-- adjust if needed
from sparx_agency.robots.common.state_converter import costmap_to_occupancygrid


# ---------------- PointCloud2 -> Nx3 float32 ----------------
def pointcloud2_to_xyz_array(msg: PointCloud2, skip_nans: bool = True) -> np.ndarray:
    """
    Convert ROS2 sensor_msgs/PointCloud2 to (N,3) float32 array for x,y,z.
    Assumes x/y/z are float32 (common).
    """
    offsets = {f.name: f.offset for f in msg.fields}
    if "x" not in offsets or "y" not in offsets or "z" not in offsets:
        return np.zeros((0, 3), dtype=np.float32)

    def _dtype_ok(name: str) -> bool:
        for f in msg.fields:
            if f.name == name:
                return f.datatype == PointField.FLOAT32
        return False

    if not (_dtype_ok("x") and _dtype_ok("y") and _dtype_ok("z")):
        return np.zeros((0, 3), dtype=np.float32)

    x_off, y_off, z_off = offsets["x"], offsets["y"], offsets["z"]
    step = msg.point_step
    data = msg.data

    if step <= 0 or len(data) < step:
        return np.zeros((0, 3), dtype=np.float32)

    n = len(data) // step
    buf = np.frombuffer(data, dtype=np.uint8, count=n * step).reshape(n, step)

    x = buf[:, x_off:x_off + 4].view(np.float32).reshape(-1)
    y = buf[:, y_off:y_off + 4].view(np.float32).reshape(-1)
    z = buf[:, z_off:z_off + 4].view(np.float32).reshape(-1)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    if skip_nans:
        m = np.isfinite(xyz).all(axis=1)
        xyz = xyz[m]
    return xyz


# ---------------- TF transform points ----------------

def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Quaternion -> rotation matrix.
    """
    # Normalize to be safe
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=np.float32)
    return R


def transform_points(pts: np.ndarray, tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    R = quat_to_R(q.x, q.y, q.z, q.w)
    p = (pts @ R.T) + np.array([t.x, t.y, t.z], dtype=np.float32)[None, :]
    return p.astype(np.float32, copy=False)


# ---------------- Grid publishing helpers ----------------

def grid_to_occupancy_msg(
    grid_spec,
    grid_data_h_w: np.ndarray,
    stamp,
    frame_id: str,
) -> OccupancyGrid:
    """
    grid_data_h_w: int8 (H,W), values in [-1,100] (or 120 debug etc.)
    """
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.info.resolution = float(grid_spec.resolution_m)
    msg.info.width = int(grid_spec.width)
    msg.info.height = int(grid_spec.height)
    msg.info.origin.position.x = float(grid_spec.origin_x)
    msg.info.origin.position.y = float(grid_spec.origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    # OccupancyGrid expects row-major flattening of (H,W)
    msg.data = grid_data_h_w.reshape(-1).astype(np.int8).tolist()
    return msg


def potential_to_occupancy_like(U: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray:
    """
    U: float32 (H,W) in [0,1]
    unknown_mask: bool (H,W)
    Returns int8 (H,W): -1 unknown, else 0..100
    """
    out = np.clip(U * 100.0, 0.0, 100.0).astype(np.int16)
    out = out.astype(np.int8, copy=False)
    out = out.copy()
    out[unknown_mask] = -1
    return out


# ---------------- Main node ----------------

class DepthAnythingV3OccPotentialNode(Node):
    def __init__(self):
        super().__init__("depth_anything_v3_occ_potential")

        # ---- Parameters ----
        self.declare_parameter("cloud_topic", "/depth_anything/points")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("cloud_is_optical", True)  # if no TF, we can still remap optical->base-ish if you want later
        self.declare_parameter("publish_hz", 2.0)

        # Map configs
        self.declare_parameter("map_size_m", 60.0)
        self.declare_parameter("resolution_m", 0.1)

        # Z band used to decide obstacle vs non-obstacle (2D map from 3D points)
        self.declare_parameter("occ_z_min", -0.2)
        self.declare_parameter("occ_z_max", 1.5)

        # Potential field params
        self.declare_parameter("occ_thresh", 0.65)
        self.declare_parameter("sigma_m", 0.6)
        self.declare_parameter("inflation_radius_m", 0.35)
        self.declare_parameter("unknown_as_obstacle", False)

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.use_tf = bool(self.get_parameter("use_tf").value)

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self) if self.use_tf else None

        # ---- Costmaps (tmp + accum) ----
        size_m = float(self.get_parameter("map_size_m").value)
        res_m = float(self.get_parameter("resolution_m").value)

        # Adjust if your LogOddsGridConfig uses different names
        cfg_accum = LogOddsGridConfig(size_m=size_m, resolution_m=res_m, frame_id=self.target_frame)
        cfg_tmp = LogOddsGridConfig(size_m=size_m, resolution_m=res_m, frame_id=self.target_frame)

        self.costmap_accum = LogOddsGridCostmap(cfg_accum)
        self.costmap_tmp = LogOddsGridCostmap(cfg_tmp)

        # ---- Potential layer ----
        self.potential = PotentialFieldLayer(
            occ_thresh=float(self.get_parameter("occ_thresh").value),
            sigma_m=float(self.get_parameter("sigma_m").value),
            k_rep=1.0,
            inflation_radius_m=float(self.get_parameter("inflation_radius_m").value),
            u_max=1.0,
            unknown_as_obstacle=bool(self.get_parameter("unknown_as_obstacle").value),
        )

        self.occ_z_min = float(self.get_parameter("occ_z_min").value)
        self.occ_z_max = float(self.get_parameter("occ_z_max").value)

        # ---- Publishers ----
        self.pub_tmp = self.create_publisher(OccupancyGrid, "occ_tmp", 1)
        self.pub_accum = self.create_publisher(OccupancyGrid, "occ_accum", 1)
        self.pub_potential = self.create_publisher(OccupancyGrid, "potential_repulsive", 1)

        # ---- Subscriber ----
        self.sub_cloud = self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, 10)

        # ---- State ----
        self._last_stamp = None

        # publish timer
        publish_hz = float(self.get_parameter("publish_hz").value)
        period = 1.0 / max(0.1, publish_hz)
        self.timer = self.create_timer(period, self.on_publish)

        self.get_logger().info(f"Subscribed to: {self.cloud_topic}")
        self.get_logger().info(f"Target frame: {self.target_frame} (use_tf={self.use_tf})")
        self.get_logger().info("Publishing: /occ_tmp, /occ_accum, /potential_repulsive")

    def _lookup_tf(self, source_frame: str, stamp_ros) -> Optional[TransformStamped]:
        try:
            # Use message time for TF lookup
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                stamp_ros,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return None

    def on_cloud(self, msg: PointCloud2):
        xyz = pointcloud2_to_xyz_array(msg, skip_nans=True)
        if xyz.shape[0] == 0:
            return

        pts = xyz  # in msg.header.frame_id

        # Transform to target frame if enabled
        if self.use_tf:
            tf = self._lookup_tf(msg.header.frame_id, msg.header.stamp)
            if tf is None:
                # If TF is missing, do nothing for now (avoids integrating in wrong frame)
                self.get_logger().warn_throttle(2.0, f"No TF {self.target_frame} <- {msg.header.frame_id}")
                return
            pts = transform_points(pts, tf)

        # 2D mapping: decide which points are "occupied"
        z = pts[:, 2]
        is_occ = (z >= self.occ_z_min) & (z <= self.occ_z_max)

        # Update tmp (reset each frame)
        self.costmap_tmp.reset()
        self.costmap_tmp.update_from_points_xy(pts[:, 0], pts[:, 1], is_occ, stamp_sec=stamp_to_sec(msg.header.stamp))

        # Update accum (integrate)
        self.costmap_accum.update_from_points_xy(pts[:, 0], pts[:, 1], is_occ, stamp_sec=stamp_to_sec(msg.header.stamp))

        self._last_stamp = msg.header.stamp

    def on_publish(self):
        if self._last_stamp is None:
            return

        # Publish tmp occupancy
        spec_tmp, grid_tmp = self.costmap_tmp.get_grid()
        msg_tmp = costmap_to_occupancygrid(self.costmap_tmp, self._last_stamp, self._target_frame)
        self.pub_tmp.publish(msg_tmp)

        # Publish accum occupancy
        spec_acc, grid_acc = self.costmap_accum.get_grid()
        msg_acc = costmap_to_occupancygrid(self.costmap_accum, self._last_stamp, self.target_frame)
        self.pub_accum.publish(msg_acc)

        # Potential from accumulated probability grid
        p = self.costmap_accum.get_prob(unknown=np.nan)  # or your get_prob_grid_float / get_prob(unknown=...)
        unknown_mask = ~np.isfinite(p)
        U, _D = self.potential.compute_from_prob_grid(p, spec_acc.resolution_m)
        pot_grid = potential_to_occupancy_like(U, unknown_mask)

        msg_pot = costmap_to_occupancygrid(self.costmap_accum, self._last_stamp, self.target_frame, pot_grid=pot_grid)
        self.pub_potential.publish(msg_pot)


def main():
    rclpy.init()
    node = DepthAnythingV3OccPotentialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()