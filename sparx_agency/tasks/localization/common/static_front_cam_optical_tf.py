#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math

class StaticFrontCamOpticalTF(Node):
    def __init__(self):
        super().__init__("static_front_cam_optical_tf")

        broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "simple_drone/front_cam_link"
        t.child_frame_id = "simple_drone/front_cam_optical_frame"

        # No translation
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Rotation: camera_link -> optical
        # roll=-90°, yaw=-90°
        roll = -math.pi / 2
        pitch = 0.0
        yaw = -math.pi / 2

        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        broadcaster.sendTransform(t)
        self.get_logger().info("Published static TF front_cam_link → front_cam_optical_frame")

    def rpy_to_quat(self, r, p, y):
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw


def main():
    rclpy.init()
    node = StaticFrontCamOpticalTF()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
