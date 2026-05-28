#!/usr/bin/env python3
"""
pose_to_tf_node.py

Bridges an april_tag_pose (PoseStamped) → dynamic TF broadcast.
Publishes: map → xtend_camera (optical frame: X=right, Y=down, Z=forward)

The apriltag node publishes world_T_ros_body (ROS body convention: X=forward,
Y=left, Z=up).  xtend_camera is the optical frame, so we compose:
    q_optical = q_body  ⊗  q_body_T_optical
where q_body_T_optical = (-0.5, 0.5, -0.5, 0.5) corresponds to the rotation
matrix [[0,0,1],[-1,0,0],[0,-1,0]] (optical X→body -Y, Y→body -Z, Z→body X).

ROS parameters:
  pose_topic  (str)  topic carrying PoseStamped from apriltag node
                     default: /xtend/april_tag_pose
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

# Fixed quaternion (x,y,z,w): rotates from optical frame to ROS body frame.
# Verified: Rz(-90°)·Rx(-90°) = [[0,0,1],[-1,0,0],[0,-1,0]]
_BODY_T_OPT = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, xyzw convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


class PoseToTFNode(Node):
    def __init__(self):
        super().__init__("pose_to_tf_node")
        self._broadcaster = TransformBroadcaster(self)

        self.declare_parameter("pose_topic", "/xtend/april_tag_pose")
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value

        self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        self.get_logger().info(f"pose_to_tf_node ready — {pose_topic} → map→xtend_camera")

    def _on_pose(self, msg: PoseStamped) -> None:
        q_body = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])
        q_opt = _quat_mul(q_body, _BODY_T_OPT)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "xtend_camera"
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation.x = float(q_opt[0])
        t.transform.rotation.y = float(q_opt[1])
        t.transform.rotation.z = float(q_opt[2])
        t.transform.rotation.w = float(q_opt[3])
        self._broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = PoseToTFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()