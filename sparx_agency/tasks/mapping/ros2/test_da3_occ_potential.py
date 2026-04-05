#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridCostmap, LogOddsGridConfig
from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer
from sparx_agency.core.mapping.pipeline.mapping_pipeline import optical_xyz_to_base_xyz
from sparx_agency.robots.common import stamp_to_sec
from sparx_agency.robots.common.spatial_math import quat_to_rot
from sparx_agency.robots.common.state_converter import costmap_to_occupancygrid


def pointcloud2_to_xyz_array(msg: PointCloud2, skip_nans: bool = True) -> np.ndarray:
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


def transform_points(pts: np.ndarray, tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    q = tf.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    return (pts @ R.T) + np.array([t.x, t.y, t.z], dtype=np.float32)[None, :]

def potential_to_grid(U: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray:
    out = np.clip(U * 100.0, 0.0, 100.0).astype(np.int16).astype(np.int8)
    out = out.copy()
    out[unknown_mask] = -1
    return out


class DA3OccPotentialTest(Node):
    def __init__(self):
        super().__init__("da3_occ_potential_test")

        self.declare_parameter("cloud_topic", "/camera/point_cloud")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("publish_hz", 2.0)

        self.cloud_topic = self.get_parameter("cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.use_tf = bool(self.get_parameter("use_tf").value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self) if self.use_tf else None

        # Costmaps (tmp + accum)
        cfg_a = LogOddsGridConfig()
        cfg_a.frame_id = self.target_frame
        self.accum = LogOddsGridCostmap(cfg_a)

        cfg_t = LogOddsGridConfig()
        cfg_t.frame_id = self.target_frame
        self.tmp = LogOddsGridCostmap(cfg_t)

        # Potential
        self.potential = PotentialFieldLayer(
            occ_thresh=0.65,
            sigma_m=0.6,
            k_rep=1.0,
            inflation_radius_m=0.35,
            u_max=1.0,
            unknown_as_obstacle=False,
        )
        # QoS (sensor data)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # Publishers
        self.pub_tmp = self.create_publisher(OccupancyGrid, "occ_tmp", qos)
        self.pub_acc = self.create_publisher(OccupancyGrid, "occ_accum", qos)
        self.pub_pot = self.create_publisher(OccupancyGrid, "potential_repulsive", qos)

        # Subscriber
        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, qos)

        # Timer
        hz = float(self.get_parameter("publish_hz").value)
        self.timer = self.create_timer(1.0 / max(0.1, hz), self.on_publish)

        self._last_stamp = None
        self.get_logger().info(f"Subscribed: {self.cloud_topic}")
        self.get_logger().info("Publishing: occ_tmp, occ_accum, potential_repulsive")

    def _lookup_tf(self, source_frame: str, stamp_ros) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                stamp_ros,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception:
            return None

    def on_cloud(self, msg: PointCloud2):
        pts = pointcloud2_to_xyz_array(msg, skip_nans=True)
        if pts.shape[0] == 0:
            return

        if self.use_tf:
            tf = self._lookup_tf(msg.header.frame_id, msg.header.stamp)
            if tf is None:
                self.get_logger().warn(f"No TF {self.target_frame} <- {msg.header.frame_id}")
                return
            pts = transform_points(pts, tf)
            # pts are now in target_frame
            pts_xy = pts[:, :2].astype(np.float32, copy=False)
        else:
            # No TF: treat incoming cloud as optical and project onto a stable XY plane
            pts_base = optical_xyz_to_base_xyz(pts)
            x = pts_base[:, 0]  # forward (depth)
            y = pts_base[:, 1]  # left
            z = pts_base[:, 2]  # up

            r = np.sqrt(x * x + y * y)

            m = (
                    np.isfinite(pts_base).all(axis=1)
                    & (x > 0.4) & (x < 6.0)  # depth range
                    & (r < 6.0)  # radial clamp
                    & (z > -0.2) & (z < 1.2)  # height band (tune!)
            )

            pts_xy = pts_base[m, :2].astype(np.float32, copy=False)

            self.tmp.reset()
            self.tmp.update_from_points_xy(pts_xy, stamp_sec=stamp_to_sec(msg.header.stamp))
            self.accum.update_from_points_xy(pts_xy, stamp_sec=stamp_to_sec(msg.header.stamp))

            # IMPORTANT: publish in the same frame you’re actually using
            # otherwise RViz fixed frame 'map' will not match anything
            self.target_frame = msg.header.frame_id  # or "camera_link" if that's the frame_id
            self.accum.cfg.frame_id = self.target_frame
            self.tmp.cfg.frame_id = self.target_frame

            self.get_logger().info(f"raw min/max: x {pts[:, 0].min():.2f}/{pts[:, 0].max():.2f} "
                                   f"y {pts[:, 1].min():.2f}/{pts[:, 1].max():.2f} "
                                   f"z {pts[:, 2].min():.2f}/{pts[:, 2].max():.2f}", throttle_duration_sec=2.0)

        # tmp reset per frame
        self.tmp.reset()
        self.tmp.update_from_points_xy(pts_xy, stamp_sec=stamp_to_sec(msg.header.stamp))

        # accum integrate
        self.accum.update_from_points_xy(pts_xy, stamp_sec=stamp_to_sec(msg.header.stamp))

        self._last_stamp = msg.header.stamp

    def on_publish(self):
        if self._last_stamp is None:
            return

        self.pub_tmp.publish(costmap_to_occupancygrid(self.tmp, self._last_stamp, self.target_frame))

        spec_a, grid_a = self.accum.get_grid()
        self.pub_acc.publish(costmap_to_occupancygrid(self.accum, self._last_stamp, self.target_frame))

        # Potential from accumulated probability
        p = self.accum.get_prob(unknown=np.nan)  # or get_prob_grid_float(unknown=np.nan)
        unknown_mask = ~np.isfinite(p)
        U, _D = self.potential.compute_from_prob_grid(p, spec_a.resolution_m)
        pot_grid = potential_to_grid(U, unknown_mask)

        self.pub_pot.publish(costmap_to_occupancygrid(self.accum, self._last_stamp, self.target_frame, pot_grid=pot_grid))


def main():
    rclpy.init()
    node = DA3OccPotentialTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()