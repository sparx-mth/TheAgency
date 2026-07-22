#!/usr/bin/env python3
import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose

import tf2_ros
from tf2_ros import TransformException

from cv_bridge import CvBridge

from sparx_agency.core.mapping.depth.depth_bbox_fusion import (
    bbox_to_xyz_cam_from_depth,
    transform_point,
)
from sparx_agency.core.mapping.vlm_semantic.nanoowl_parser import (
    parse_nanoowl_json_detections,
)
from sparx_agency.core.common.spatial_math import quat_to_rot


def camera_info_to_intrinsics(msg: CameraInfo):
    fx = float(msg.k[0])
    fy = float(msg.k[4])
    cx = float(msg.k[2])
    cy = float(msg.k[5])
    return fx, fy, cx, cy


def transform_to_matrix(t):
    """
    geometry_msgs/TransformStamped -> 4x4 numpy
    """
    tx = t.transform.translation.x
    ty = t.transform.translation.y
    tz = t.transform.translation.z
    qx = t.transform.rotation.x
    qy = t.transform.rotation.y
    qz = t.transform.rotation.z
    qw = t.transform.rotation.w

    # quaternion to rotation matrix

    R = quat_to_rot(qx, qy, qz, qw)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


class NanoOwlDepthFusionNode(Node):
    def __init__(self):
        super().__init__("nanoowl_depth_fusion")

        # Topics (match your setup)
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("camera_info_topic", "/video/camera_info")
        self.declare_parameter("nanoowl_json_topic", "/nanoowl/json")

        # Frames
        self.declare_parameter("world_frame", "map")     # map/odom/world
        self.declare_parameter("camera_frame", "camera_link")  # from CameraInfo

        # Depth sampling params
        self.declare_parameter("sample_grid", 9)
        self.declare_parameter("low_quantile", 0.2)
        self.declare_parameter("min_depth", 0.15)
        self.declare_parameter("max_depth", 20.0)

        # TF timeout
        self.declare_parameter("tf_timeout_sec", 0.05)

        self.bridge = CvBridge()

        self._depth_m = None
        self._depth_header = None
        self._cam_info = None
        self._last_json_str = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subs
        self.create_subscription(Image, self.get_parameter("depth_topic").value, self.on_depth, 10)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self.on_caminfo, 10)
        self.create_subscription(String, self.get_parameter("nanoowl_json_topic").value, self.on_json, 10)

        # Pubs
        self.pub_pose_world = self.create_publisher(PoseArray, "/nanoowl/detections_world", 10)
        self.pub_json_out = self.create_publisher(String, "/nanoowl/detections_world_json", 10)

        self.get_logger().info("NanoOwlDepthFusionNode started")

    def on_caminfo(self, msg: CameraInfo):
        self._cam_info = msg

    def on_depth(self, msg: Image):
        # expects 32FC1 in meters
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {e}")
            return

        depth_m = np.asarray(depth, dtype=np.float32)
        if depth_m.ndim != 2:
            self.get_logger().warn(f"Depth not single channel: {depth_m.shape}")
            return

        self._depth_m = depth_m
        self._depth_header = msg.header
        self.try_fuse()

    def on_json(self, msg: String):
        self._last_json_str = msg.data
        self.try_fuse()

    def _lookup_T_world_cam(self, stamp):
        world_frame = self.get_parameter("world_frame").value
        camera_frame = self.get_parameter("camera_frame").value
        timeout = float(self.get_parameter("tf_timeout_sec").value)

        try:
            tf = self.tf_buffer.lookup_transform(
                world_frame, camera_frame, stamp, timeout=Duration(seconds=timeout)
            )
            return transform_to_matrix(tf)
        except TransformException as e:
            self.get_logger().warn(f"TF missing {world_frame} <- {camera_frame}: {e}")
            return None

    def try_fuse(self):
        if self._depth_m is None or self._cam_info is None or not self._last_json_str:
            return

        # intrinsics
        fx, fy, cx, cy = camera_info_to_intrinsics(self._cam_info)

        # detections
        try:
            dets, Wj, Hj = parse_nanoowl_json_detections(self._last_json_str)
        except Exception as e:
            self.get_logger().error(f"Bad nanoowl json: {e}")
            return

        if not dets:
            return

        H, W = self._depth_m.shape
        # If NanoOWL image size differs, scale bbox
        sx = (W / float(Wj)) if Wj else 1.0
        sy = (H / float(Hj)) if Hj else 1.0

        stamp = self._depth_header.stamp if self._depth_header is not None else self._cam_info.header.stamp

        # TF world<-cam
        T_world_cam = self._lookup_T_world_cam(stamp)
        if T_world_cam is None:
            return  # no TF => cannot output world XYZ

        sample_grid = int(self.get_parameter("sample_grid").value)
        low_quantile = float(self.get_parameter("low_quantile").value)
        min_depth = float(self.get_parameter("min_depth").value)
        max_depth = float(self.get_parameter("max_depth").value)

        # publish PoseArray in world
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self.get_parameter("world_frame").value

        # enrich JSON output
        out = json.loads(self._last_json_str)
        out.setdefault("nanoowl_world", {})
        out["nanoowl_world"]["frame_id"] = pa.header.frame_id
        out["nanoowl_world"]["detections"] = []

        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            # scale if needed
            bbox = (int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)))

            xyz_cam = bbox_to_xyz_cam_from_depth(
                self._depth_m, bbox, fx, fy, cx, cy,
                sample_grid=sample_grid,
                low_quantile=low_quantile,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            if xyz_cam is None:
                out["nanoowl_world"]["detections"].append({
                    "label": d["label"], "score": d["score"], "bbox": list(bbox),
                    "ok": False, "reason": "no_valid_depth"
                })
                continue

            xyz_world = transform_point(T_world_cam, xyz_cam)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = xyz_world
            pose.orientation.w = 1.0
            pa.poses.append(pose)

            out["nanoowl_world"]["detections"].append({
                "label": d["label"],
                "score": d["score"],
                "bbox": list(bbox),
                "ok": True,
                "xyz_world": list(xyz_world),
                "xyz_cam": list(xyz_cam),  # keep debug
            })

        if pa.poses:
            self.pub_pose_world.publish(pa)
        self.pub_json_out.publish(String(data=json.dumps(out)))

        # Optional: if you want “process each JSON once”
        # self._last_json_str = None


def main():
    rclpy.init()
    node = NanoOwlDepthFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
