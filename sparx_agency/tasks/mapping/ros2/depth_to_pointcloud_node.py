#!/usr/bin/env python3
"""
depth_to_pointcloud_node.py

Backprojects a 32FC1 depth image into a PointCloud2 in the camera optical frame.
Intended for bag replay — avoids re-running TRT inference when /xtend/depth_m is
already recorded.

Subscriptions:
  depth_topic        sensor_msgs/Image       (32FC1, metric metres)
  camera_info_topic  sensor_msgs/CameraInfo  (cached on first message)

Publications:
  pointcloud_topic   sensor_msgs/PointCloud2 (camera optical frame)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge


_BE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class DepthToPointcloudNode(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud_node")

        self.declare_parameter("depth_topic",       "/xtend/depth_m")
        self.declare_parameter("camera_info_topic", "/xtend/camera_info")
        self.declare_parameter("pointcloud_topic",  "/xtend/pointcloud")
        self.declare_parameter("clip_min_m",        0.05)
        self.declare_parameter("clip_max_m",        10.0)

        depth_topic = str(self.get_parameter("depth_topic").value)
        info_topic  = str(self.get_parameter("camera_info_topic").value)
        cloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self._z_min = float(self.get_parameter("clip_min_m").value)
        self._z_max = float(self.get_parameter("clip_max_m").value)

        self._bridge   = CvBridge()
        self._cam_info = None  # cached on first camera_info message

        self._pub = self.create_publisher(PointCloud2, cloud_topic, _BE_QOS)
        self.create_subscription(CameraInfo, info_topic,  self._info_cb,  _BE_QOS)
        self.create_subscription(Image,      depth_topic, self._depth_cb, _BE_QOS)

        self.get_logger().info(
            f"depth_to_pointcloud: {depth_topic} + {info_topic} → {cloud_topic}"
        )

    def _info_cb(self, msg: CameraInfo) -> None:
        self._cam_info = msg  # just cache it; intrinsics are constant throughout the bag

    def _depth_cb(self, msg: Image) -> None:
        if self._cam_info is None:
            return  # wait for camera_info

        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        h, w  = depth.shape

        # Scale K to actual depth resolution (depth size may differ from camera_info size)
        sx = w / self._cam_info.width  if self._cam_info.width  > 0 else 1.0
        sy = h / self._cam_info.height if self._cam_info.height > 0 else 1.0
        fx = float(self._cam_info.k[0]) * sx
        fy = float(self._cam_info.k[4]) * sy
        cx = float(self._cam_info.k[2]) * sx
        cy = float(self._cam_info.k[5]) * sy

        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        z    = depth.flatten().astype(np.float32)
        mask = np.isfinite(z) & (z > self._z_min) & (z < self._z_max)

        pts = np.stack([
            (u.flatten()[mask] - cx) * z[mask] / fx,
            (v.flatten()[mask] - cy) * z[mask] / fy,
            z[mask],
        ], axis=1).astype(np.float32)

        out             = PointCloud2()
        out.header      = msg.header
        out.height      = 1
        out.width       = len(pts)
        out.fields      = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step   = 12
        out.row_step     = 12 * len(pts)
        out.is_dense     = True
        out.data         = pts.tobytes()
        self._pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(DepthToPointcloudNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()