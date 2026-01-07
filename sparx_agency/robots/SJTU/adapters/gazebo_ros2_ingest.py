#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
import math
import os
from dataclasses import dataclass
from typing import Optional, Any

import cv2
import numpy as np

import rclpy
from depth_anything_v2.dpt import DepthAnythingV2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs_py import point_cloud2

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid

# --- core types (ROS-free) ---
from sparx_agency.core.common.types.perception import (
    Intrinsics,
    PoseSE3,
    RGBFrame,
    Observation,
)
from sparx_agency.core.mapping.costmap import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridConfig
from sparx_agency.core.mapping.interfaces import DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config
from sparx_agency.robots.SJTU.helpers.helpers import make_depth_grid_vis, depth_to_vis_u8
# ROS-free mapping pipeline
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, PinholeCloudGenerator, \
    MappingPipelineConfig


def strip_leading_slash(s: str) -> str:
    if not s:
        return s
    return s[1:] if s.startswith("/") else s


def stamp_to_sec(stamp) -> float:
    # builtin_interfaces/msg/Time
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Returns 3x3 rotation matrix
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=np.float32)


def image_msg_to_rgb_numpy(msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs/Image to HxWx3 uint8 RGB WITHOUT cv_bridge.

    Supports: rgb8, bgr8
    """
    enc = (msg.encoding or "").lower()
    if enc not in ("rgb8", "bgr8"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding} (expected rgb8 or bgr8)")

    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)

    # raw bytes -> (h, step) -> slice first w*3 bytes -> (h, w, 3)
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        raise ValueError(f"Image buffer too small: {buf.size} < {h*step}")

    row = buf.reshape((h, step))
    rgb = row[:, :w*3].reshape((h, w, 3))

    if enc == "bgr8":
        rgb = rgb[..., ::-1].copy()

    return rgb


@dataclass
class LatestState:
    odom: Optional[Odometry] = None
    cam_info: Optional[CameraInfo] = None


def cam_info_to_intrinsics(ci: CameraInfo) -> Intrinsics:
    fx = float(ci.k[0])
    fy = float(ci.k[4])
    cx = float(ci.k[2])
    cy = float(ci.k[5])
    return Intrinsics(
        width=int(ci.width),
        height=int(ci.height),
        fx=fx, fy=fy, cx=cx, cy=cy
    )


def odom_to_pose_se3(odom: Odometry) -> PoseSE3:
    p = odom.pose.pose.position
    o = odom.pose.pose.orientation
    R = quat_to_rot(float(o.x), float(o.y), float(o.z), float(o.w))
    t = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)
    return PoseSE3(R=R, t=t)


def pipeline_result_to_occupancygrid(result: Any, stamp, frame_id: str) -> OccupancyGrid:
    """
    Try to convert whatever MappingPipeline returns into nav_msgs/OccupancyGrid.
    Supports:
      - direct OccupancyGrid
      - result.to_occupancy_grid(stamp, frame_id)
      - result.to_ros_msg(stamp, frame_id)
      - result.to_msg(stamp, frame_id)
    """
    if isinstance(result, OccupancyGrid):
        result.header.stamp = stamp
        result.header.frame_id = frame_id
        return result

    for fn_name in ("to_occupancy_grid", "to_ros_msg", "to_msg"):
        if hasattr(result, fn_name):
            fn = getattr(result, fn_name)
            msg = fn(stamp=stamp, frame_id=frame_id)  # prefer kwargs
            if isinstance(msg, OccupancyGrid):
                return msg

    raise TypeError(
        "MappingPipeline.step(obs) must return an OccupancyGrid or an object with "
        "to_occupancy_grid()/to_ros_msg()/to_msg() returning OccupancyGrid"
    )


def costmap_to_occupancygrid(costmap, stamp, frame_id: str) -> OccupancyGrid:
    """
    Convert ROS-free costmap (ProbabilisticGridCostmap) into nav_msgs/OccupancyGrid.
    """
    spec, grid = costmap.get_grid()  # GridSpec + (H,W) int8
    msg = OccupancyGrid()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.info.resolution = float(spec.resolution_m)
    msg.info.width = int(spec.width)
    msg.info.height = int(spec.height)

    msg.info.origin.position.x = float(spec.origin_x)
    msg.info.origin.position.y = float(spec.origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    msg.data = grid.reshape(-1, order="C").tolist()
    return msg



class GazeboRos2Ingest(Node):
    """
    Gazebo adapter:
      - ROS IN:  image + camera_info + odom
      - CORE:    Observation -> MappingPipeline
      - ROS OUT: /occupancy_grid (nav_msgs/OccupancyGrid)

    Notes:
      - No cv_bridge
      - By default uses frames from messages (after stripping leading '/')
    """

    def __init__(self):
        super().__init__("gazebo_ros2_ingest")

        # --- params ---
        self.declare_parameter("rgb_topic", "/simple_drone/front/image_raw")
        self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info")
        self.declare_parameter("odom_topic", "/simple_drone/odom")
        self.declare_parameter("occupancy_topic", "/occupancy_grid")

        self.declare_parameter("use_msg_frames", True)
        self.declare_parameter("strip_leading_slash", True)

        # fallback frames if use_msg_frames:=false
        self.declare_parameter("map_frame", "simple_drone/odom")
        self.declare_parameter("base_frame", "simple_drone/base_footprint")
        self.declare_parameter("camera_frame", "simple_drone/front_cam_link")

        # processing throttle
        self.declare_parameter("process_every_n_images", 1)

        # mapping / pipeline params (keep defaults OK)
        self.declare_parameter("resolution_m", 0.30)
        self.declare_parameter("size_x_m", 40.0)
        self.declare_parameter("size_y_m", 40.0)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.occupancy_topic = str(self.get_parameter("occupancy_topic").value)

        self.use_msg_frames = bool(self.get_parameter("use_msg_frames").value)
        self.strip_slash = bool(self.get_parameter("strip_leading_slash").value)
        self.process_every_n = int(self.get_parameter("process_every_n_images").value)

        # Debug
        self.declare_parameter("debug_publish_depth", True)
        self.declare_parameter("debug_publish_cloud", True)
        self.declare_parameter("debug_publish_depth_grid", True)

        self.declare_parameter("debug_save_dir", "")
        self.declare_parameter("debug_save_every_n", 30)  # every Nth frame
        self.declare_parameter("depth_grid_w", 50)
        self.declare_parameter("depth_grid_h", 50)

        self._dbg_count = 0

        # Debug publishers
        self.pub_depth_raw = self.create_publisher(Image, "/debug/depth_raw", 1)  # 32FC1
        self.pub_depth_vis = self.create_publisher(Image, "/debug/depth_vis", 1)  # mono8
        self.pub_cloud = self.create_publisher(PointCloud2, "/debug/cloud_cam", 1)  # PointCloud2
        self.pub_depth_grid = self.create_publisher(Image, "/debug/depth_grid_vis", 1)

        self._img_count = 0
        self._latest = LatestState()

        self.costmap = ProbabilisticGridCostmap(ProbabilisticGridConfig())
        self.depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())

        self.cloud_generator = PinholeCloudGenerator()
        self.pipeline_cfg = MappingPipelineConfig()
        # --- pipeline ---
        self.pipeline = MappingPipeline(
            costmap=self.costmap,
            depth_model=self.depth_model,
            cloud_generator=self.cloud_generator,
            cfg=self.pipeline_cfg,
        )

        # --- QoS ---
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub_rgb = self.create_subscription(Image, self.rgb_topic, self.cb_rgb, sensor_qos)
        self.sub_ci = self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_ci, sensor_qos)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.cb_odom, sensor_qos)

        self.pub_occ = self.create_publisher(OccupancyGrid, self.occupancy_topic, map_pub_qos)

        self.get_logger().info(
            f"Gazebo ingest started.\n"
            f"  rgb_topic={self.rgb_topic}\n"
            f"  camera_info_topic={self.camera_info_topic}\n"
            f"  odom_topic={self.odom_topic}\n"
            f"  occupancy_topic={self.occupancy_topic}\n"
            f"  use_msg_frames={self.use_msg_frames}, strip_leading_slash={self.strip_slash}\n"
            f"  process_every_n_images={self.process_every_n}"
        )

    def norm_frame(self, s: str) -> str:
        return strip_leading_slash(s) if self.strip_slash else s

    def cb_ci(self, msg: CameraInfo):
        self._latest.cam_info = msg

    def cb_odom(self, msg: Odometry):
        self._latest.odom = msg

    def cb_rgb(self, msg: Image):
        self._img_count += 1
        if self.process_every_n > 1 and (self._img_count % self.process_every_n) != 0:
            return

        if self._latest.odom is None:
            self.get_logger().warn("No odom yet; skipping frame.")
            return

        if self._latest.cam_info is None:
            self.get_logger().warn("No camera_info yet; skipping frame.")
            return

        try:
            rgb = image_msg_to_rgb_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")
            return

        odom = self._latest.odom
        cam_info = self._latest.cam_info

        # Frames
        if self.use_msg_frames:
            map_frame = self.norm_frame(odom.header.frame_id)
            base_frame = self.norm_frame(odom.child_frame_id)
            cam_frame = self.norm_frame(msg.header.frame_id if msg.header.frame_id else cam_info.header.frame_id)
        else:
            map_frame = self.norm_frame(str(self.get_parameter("map_frame").value))
            base_frame = self.norm_frame(str(self.get_parameter("base_frame").value))
            cam_frame = self.norm_frame(str(self.get_parameter("camera_frame").value))

        stamp_sec = stamp_to_sec(msg.header.stamp)
        intr = cam_info_to_intrinsics(cam_info)
        pose = odom_to_pose_se3(odom)

        obs = Observation(
            intrinsics=intr,
            pose_map_base=pose,
            rgb=RGBFrame(image=rgb, stamp_sec=stamp_sec, frame_id=cam_frame),
        )

        # Run core pipeline (updates self.costmap internally)
        try:
            self.pipeline.step(obs)
        except Exception as e:
            self.get_logger().error(f"MappingPipeline.step() failed: {e}")
            return
        # Debug: publish depth / cloud / grid
        depth_m = getattr(self.pipeline, "last_depth", None)
        cloud_cam = getattr(self.pipeline, "last_cloud_cam", None)

        debug_depth = bool(self.get_parameter("debug_publish_depth").value)
        debug_cloud = bool(self.get_parameter("debug_publish_cloud").value)
        debug_grid = bool(self.get_parameter("debug_publish_depth_grid").value)

        grid_w = int(self.get_parameter("depth_grid_w").value)
        grid_h = int(self.get_parameter("depth_grid_h").value)

        save_dir = str(self.get_parameter("debug_save_dir").value).strip()
        save_every_n = int(self.get_parameter("debug_save_every_n").value)

        self._dbg_count += 1
        if depth_m is not None and debug_depth:
            # raw 32FC1
            depth_msg = self.numpy_to_image_msg(depth_m.astype(np.float32), frame_id=msg.header.frame_id, stamp=msg.header.stamp, encoding="32FC1")
            depth_msg.header = msg.header
            depth_msg.header.frame_id = cam_frame
            self.pub_depth_raw.publish(depth_msg)

            # vis mono8
            vis_u8 = depth_to_vis_u8(depth_m, clip_min=self.pipeline.cfg.range_min,
                                     clip_max=self.pipeline.cfg.range_max)
            vis_msg = self.numpy_to_image_msg(vis_u8, frame_id=msg.header.frame_id, stamp=msg.header.stamp, encoding="mono8")
            vis_msg.header = msg.header
            vis_msg.header.frame_id = cam_frame
            self.pub_depth_vis.publish(vis_msg)

            # grid vis
            if debug_grid:
                grid_vis = make_depth_grid_vis(depth_m, grid_w, grid_h,
                                               clip_min=self.pipeline.cfg.range_min,
                                               clip_max=self.pipeline.cfg.range_max)
                grid_msg = self.numpy_to_image_msg(grid_vis, frame_id=msg.header.frame_id, stamp=msg.header.stamp, encoding="bgr8")
                grid_msg.header = msg.header
                grid_msg.header = msg.header
                grid_msg.header.frame_id = cam_frame
                self.pub_depth_grid.publish(grid_msg)

            # optional save
            if save_dir and (self._dbg_count % max(1, save_every_n) == 0):
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(os.path.join(save_dir, f"depth_vis_{ts}.png"), vis_u8)
                if debug_grid:
                    cv2.imwrite(os.path.join(save_dir, f"depth_grid_{ts}.png"), grid_vis)

        if cloud_cam is not None and debug_cloud:
            # publish PointCloud2 in camera frame
            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = cam_frame

            pts = cloud_cam.astype(np.float32)
            cloud_msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
            self.pub_cloud.publish(cloud_msg)

            if save_dir and (self._dbg_count % max(1, save_every_n) == 0):
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                np.save(os.path.join(save_dir, f"cloud_cam_{ts}.npy"), pts)

        # Publish occupancy grid from costmap
        try:
            occ_msg = costmap_to_occupancygrid(self.costmap, stamp=msg.header.stamp, frame_id=map_frame)
        except Exception as e:
            self.get_logger().error(f"Cannot convert costmap to OccupancyGrid: {e}")
            return

        self.pub_occ.publish(occ_msg)
        self.get_logger().info(
            f"Published occupancy grid {occ_msg.info.width}x{occ_msg.info.height} "
            f"res={occ_msg.info.resolution:.3f} frame={occ_msg.header.frame_id}"
        )

    def image_msg_to_numpy(self, msg: Image) -> np.ndarray:
        """
        Supports: rgb8, bgr8, mono8, 32FC1
        Returns a numpy view/copy shaped (H,W,C) or (H,W).
        """
        h = msg.height
        w = msg.width
        enc = msg.encoding.lower()

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            arr = arr.reshape((h, w, 3))
            return arr

        if enc in ("mono8",):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            return arr.reshape((h, w))

        if enc in ("32fc1",):
            arr = np.frombuffer(msg.data, dtype=np.float32)
            return arr.reshape((h, w))

        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    def numpy_to_image_msg(self, arr: np.ndarray, *, frame_id: str, stamp, encoding: str) -> Image:
        """
        Create sensor_msgs/Image from numpy without cv_bridge.
        encoding examples: 'rgb8', 'bgr8', 'mono8', '32FC1'
        """
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        if encoding.lower() in ("rgb8", "bgr8"):
            assert arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8
            msg.height, msg.width = arr.shape[0], arr.shape[1]
            msg.encoding = encoding
            msg.is_bigendian = False
            msg.step = msg.width * 3
            msg.data = arr.tobytes()
            return msg

        if encoding.lower() == "mono8":
            assert arr.ndim == 2 and arr.dtype == np.uint8
            msg.height, msg.width = arr.shape
            msg.encoding = "mono8"
            msg.is_bigendian = False
            msg.step = msg.width
            msg.data = arr.tobytes()
            return msg

        if encoding.lower() == "32fc1":
            assert arr.ndim == 2 and arr.dtype == np.float32
            msg.height, msg.width = arr.shape
            msg.encoding = "32FC1"
            msg.is_bigendian = False
            msg.step = msg.width * 4
            msg.data = arr.tobytes()
            return msg

        raise ValueError(f"Unsupported encoding: {encoding}")


def main():
    rclpy.init()
    node = GazeboRos2Ingest()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
