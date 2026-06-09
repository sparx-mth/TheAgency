#!/usr/bin/env python3

import math
import struct
from typing import Optional

import cv2
import message_filters
import numpy as np
import open3d as o3d
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float64, Header

from visualization_msgs.msg import Marker

from sparx_agency.core.common.spatial_math import open3d_pose_to_ros_pose


def load_intrinsics_from_yaml(camera_yaml: str, image_width: int, image_height: int):
    with open(camera_yaml, "r") as f:
        data = yaml.safe_load(f)

    yaml_w = int(data["image_width"])
    yaml_h = int(data["image_height"])

    if "projection_matrix" in data:
        P = np.array(data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        fx = float(P[0, 0])
        fy = float(P[1, 1])
        cx = float(P[0, 2])
        cy = float(P[1, 2])
    elif "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
    else:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])

    sx = float(image_width) / float(yaml_w)
    sy = float(image_height) / float(yaml_h)

    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy

    return o3d.camera.PinholeCameraIntrinsic(
        image_width,
        image_height,
        fx,
        fy,
        cx,
        cy,
    )


def make_rgbd_image(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_trunc_m: float,
) -> o3d.geometry.RGBDImage:
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]

    if depth_m.shape[:2] != bgr.shape[:2]:
        depth_m = cv2.resize(
            depth_m,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    depth_m = depth_m.astype(np.float32)
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m < 0.0] = 0.0

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth_m)

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=depth_trunc_m,
        convert_rgb_to_intensity=False,
    )


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return (b_deg - a_deg + 180.0) % 360.0 - 180.0


def yaw_delta_init(
    prev_yaw_deg: Optional[float],
    curr_yaw_deg: Optional[float],
    yaw_sign: float,
    yaw_axis: str,
) -> Optional[np.ndarray]:
    if prev_yaw_deg is None or curr_yaw_deg is None:
        return None

    if not np.isfinite(prev_yaw_deg) or not np.isfinite(curr_yaw_deg):
        return None

    delta_deg = yaw_sign * angle_diff_deg(prev_yaw_deg, curr_yaw_deg)
    delta_rad = math.radians(delta_deg)

    if yaw_axis == "camera_y_up":
        axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    elif yaw_axis == "camera_y_down":
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif yaw_axis == "camera_z":
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown yaw_axis: {yaw_axis}")

    rotvec = axis * delta_rad
    R = Rot.from_rotvec(rotvec).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def pair_metrics(success: bool, trans: np.ndarray, info: np.ndarray) -> dict:
    rot = Rot.from_matrix(trans[:3, :3])
    rot_deg = np.linalg.norm(rot.as_rotvec()) * 180.0 / np.pi
    t = trans[:3, 3]
    t_norm = np.linalg.norm(t)
    score = float(np.mean(np.diag(info)))

    return {
        "success": bool(success),
        "rot_deg": float(rot_deg),
        "t_norm": float(t_norm),
        "score": score,
    }


def accept_pair(metrics: dict, min_score: float, max_rot_deg: float, max_t_norm: float) -> bool:
    if not metrics["success"]:
        return False
    if metrics["score"] < min_score:
        return False
    if metrics["rot_deg"] < 1e-4 and metrics["t_norm"] < 1e-4:
        return False
    if metrics["rot_deg"] > max_rot_deg:
        return False
    if metrics["t_norm"] > max_t_norm:
        return False
    return True


def transform_to_pose_msg(
    T_world_cam: np.ndarray,
    stamp,
    frame_id: str,
) -> PoseStamped:
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame_id

    t = T_world_cam[:3, 3]
    q = Rot.from_matrix(T_world_cam[:3, :3]).as_quat()

    pose.pose.position.x = float(t[0])
    pose.pose.position.y = float(t[1])
    pose.pose.position.z = float(t[2])

    pose.pose.orientation.x = float(q[0])
    pose.pose.orientation.y = float(q[1])
    pose.pose.orientation.z = float(q[2])
    pose.pose.orientation.w = float(q[3])

    return pose


class OnlineRgbdLocalizationNode(Node):
    def __init__(self):
        super().__init__("online_rgbd_localization_node")

        self.rgb_topic = self.declare_parameter("rgb_topic", "/xtend/rgb").value
        self.depth_topic = self.declare_parameter("depth_topic", "/xtend/depth_m").value
        self.bearing_topic = self.declare_parameter("bearing_topic", "/xtend/bearing").value

        self.camera_yaml = self.declare_parameter("camera_yaml", "/home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml").value
        self.world_frame = self.declare_parameter("world_frame", "xtend_odom").value

        self.depth_trunc_m = float(self.declare_parameter("depth_trunc_m", 7.0).value)
        self.depth_min_m = float(self.declare_parameter("depth_min_m", 0.3).value)
        self.depth_max_m = float(self.declare_parameter("depth_max_m", 5.0).value)
        self.depth_diff_max = float(self.declare_parameter("depth_diff_max", 0.05).value)

        self.min_score = float(self.declare_parameter("min_score", 5e4).value)
        self.max_rot_deg = float(self.declare_parameter("max_rot_deg", 5.0).value)
        self.max_t_norm = float(self.declare_parameter("max_t_norm", 0.6).value)

        self.use_yaw_init = bool(self.declare_parameter("use_yaw_init", False).value)
        self.yaw_sign = float(self.declare_parameter("yaw_sign", 1.0).value)
        self.yaw_axis = self.declare_parameter("yaw_axis", "camera_y_up").value
        self.init_mode = self.declare_parameter("init_mode", "yaw_then_last").value
        self.initial_camera_height_m = float(self.declare_parameter("initial_camera_height_m", 0.0).value)

        self.publish_marker = bool(self.declare_parameter("publish_marker", True).value)

        if not self.camera_yaml:
            raise RuntimeError("camera_yaml parameter is required")

        self.bridge = CvBridge()

        self.prev_rgbd = None
        self.prev_bearing = None
        self.latest_bearing = None
        self.pinhole = None

        self.last_transformation = np.eye(4, dtype=np.float64)
        self.cam_to_world = np.eye(4, dtype=np.float64)

        self.path_msg = RosPath()
        self.path_msg.header.frame_id = self.world_frame

        self.pose_pub = self.create_publisher(PoseStamped, "pose", 10)
        self.path_pub = self.create_publisher(RosPath, "path", 10)
        self.marker_pub = self.create_publisher(Marker, "trajectory_marker", 10)

        self.bearing_sub = self.create_subscription(
            Float64,
            self.bearing_topic,
            self.bearing_cb,
            10,
        )

        rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=10,
            slop=0.08,
        )
        self.sync.registerCallback(self.rgbd_cb)

        option = o3d.pipelines.odometry.OdometryOption()
        option.iteration_number_per_pyramid_level = o3d.utility.IntVector([200, 100, 50, 20])
        option.depth_diff_max = self.depth_diff_max
        option.depth_min = self.depth_min_m
        option.depth_max = self.depth_max_m
        self.odo_option = option

        self.frame_idx = 0

        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Camera YAML: {self.camera_yaml}")
        self.get_logger().info(f"Use yaw init: {self.use_yaw_init}")

    def bearing_cb(self, msg: Float64):
        self.latest_bearing = float(msg.data)

    def rgbd_cb(self, rgb_msg: Image, depth_msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        if depth_m.dtype != np.float32:
            depth_m = depth_m.astype(np.float32)

        h, w = depth_m.shape[:2]

        if self.pinhole is None:
            self.pinhole = load_intrinsics_from_yaml(self.camera_yaml, w, h)
            self.get_logger().info(f"Loaded intrinsics for depth/RGB size: {w}x{h}")

        rgbd_curr = make_rgbd_image(
            bgr=bgr,
            depth_m=depth_m,
            depth_trunc_m=self.depth_trunc_m,
        )

        stamp = rgb_msg.header.stamp

        if self.prev_rgbd is None:
            self.prev_rgbd = rgbd_curr
            self.prev_bearing = self.latest_bearing
            self.publish_pose_and_path(stamp)
            self.get_logger().info("Stored first RGB-D frame")
            return

        yaw_init_T = None
        if self.use_yaw_init:
            yaw_init_T = yaw_delta_init(
                prev_yaw_deg=self.prev_bearing,
                curr_yaw_deg=self.latest_bearing,
                yaw_sign=self.yaw_sign,
                yaw_axis=self.yaw_axis,
            )

        if self.init_mode == "identity":
            odo_init = np.eye(4, dtype=np.float64)
        elif self.init_mode == "last":
            odo_init = self.last_transformation
        elif self.init_mode == "yaw":
            odo_init = yaw_init_T if yaw_init_T is not None else np.eye(4, dtype=np.float64)
        elif self.init_mode == "yaw_then_last":
            odo_init = yaw_init_T if yaw_init_T is not None else self.last_transformation
        else:
            self.get_logger().warn(f"Unknown init_mode={self.init_mode}, using identity")
            odo_init = np.eye(4, dtype=np.float64)

        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            self.prev_rgbd,
            rgbd_curr,
            self.pinhole,
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            self.odo_option,
        )

        metrics = pair_metrics(success, trans, info)
        accepted = accept_pair(
            metrics,
            min_score=self.min_score,
            max_rot_deg=self.max_rot_deg,
            max_t_norm=self.max_t_norm,
        )

        if accepted:
            self.last_transformation = trans
            self.cam_to_world = self.cam_to_world @ np.linalg.inv(trans)
            self.publish_pose_and_path(stamp)

        else:
            self.last_transformation = np.eye(4, dtype=np.float64)

        self.get_logger().info(
            f"frame={self.frame_idx} accepted={accepted} "
            f"score={metrics['score']:.1f} "
            f"rot={metrics['rot_deg']:.3f}deg "
            f"t={metrics['t_norm']:.3f}m "
            f"bearing={self.latest_bearing}"
        )

        self.prev_rgbd = rgbd_curr
        self.prev_bearing = self.latest_bearing
        self.frame_idx += 1

    def make_pointcloud2_xyzrgb(
            points_xyz: np.ndarray,
            colors_rgb: np.ndarray,
            stamp,
            frame_id: str,
    ) -> PointCloud2:
        """
        Create PointCloud2 with fields x, y, z, rgb.
        points_xyz: Nx3 float
        colors_rgb: Nx3 float in [0,1] or uint8 in [0,255]
        """
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.height = 1
        msg.width = int(points_xyz.shape[0])

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False

        if colors_rgb.dtype != np.uint8:
            colors_u8 = np.clip(colors_rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            colors_u8 = colors_rgb

        buffer = bytearray()

        for p, c in zip(points_xyz, colors_u8):
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            rgb_uint32 = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack("f", struct.pack("I", rgb_uint32))[0]

            buffer.extend(struct.pack("ffff", float(p[0]), float(p[1]), float(p[2]), rgb_float))

        msg.data = bytes(buffer)
        return msg

    def publish_pose_and_path(self, stamp):
        T_ros = open3d_pose_to_ros_pose(self.cam_to_world)
        T_ros[2, 3] += self.initial_camera_height_m
        pose_msg = transform_to_pose_msg(
            T_ros,
            stamp=stamp,
            frame_id=self.world_frame,
        )

        self.pose_pub.publish(pose_msg)

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

        if self.publish_marker:
            self.publish_trajectory_marker(stamp)

    def publish_trajectory_marker(self, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.world_frame
        marker.ns = "rgbd_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.03

        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 1.0

        for pose in self.path_msg.poses:
            marker.points.append(pose.pose.position)

        self.marker_pub.publish(marker)


def main():
    rclpy.init()
    node = OnlineRgbdLocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()