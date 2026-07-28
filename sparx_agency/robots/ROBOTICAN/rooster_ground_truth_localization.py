#!/usr/bin/env python3
"""rooster_ground_truth_localization.py

Sphera-simulator-only localization source: republishes the simulator's own
ground-truth pawn pose (sphera_common_interfaces/msg/SpheraPawnState, only
importable where the Sphera ROS2 interfaces are built - inside the `it`
container, not on the host) as a plain PoseStamped, in the same format
tasks/localization/ros2/localization_node.py's real (AprilTag/optical-flow)
providers produce.

This exists so the ROBOTICAN pipeline (rooster_dome_main.py --pose-topic,
DA3/room_mapper consumers) can be exercised end-to-end against the
simulator without needing a physically-placed AprilTag - it is not a
localization *algorithm*, just a passthrough of what Sphera already knows.
Never applicable to a real drone.

Yaw is encoded the same way xtend_dome_main.py's _LocalizationListener
expects: z=sin(yaw/2), w=cos(yaw/2), x=y=0 (planar rotation only).
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sphera_common_interfaces.msg import SpheraPawnState


class RoosterGroundTruthLocalization(Node):
    def __init__(self):
        super().__init__("rooster_ground_truth_localization")

        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("pose_topic", "")
        self.declare_parameter("source_topic", "")
        rooster_id = self.get_parameter("rooster_id").value
        pose_topic = self.get_parameter("pose_topic").value or f"/{rooster_id}/localization"
        source_topic = self.get_parameter("source_topic").value or f"/{rooster_id}/localization_source"

        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.source_pub = self.create_publisher(String, source_topic, 10)
        self.create_subscription(
            SpheraPawnState, f"/{rooster_id}/sphera/state", self._on_state, 10)

        self.get_logger().info(
            f"rooster_ground_truth_localization ready for {rooster_id}\n"
            f"  sphera state in: /{rooster_id}/sphera/state\n"
            f"  pose out:        {pose_topic}"
        )

    def _on_state(self, msg: SpheraPawnState):
        # Sphera/Unreal is left-handed (clockwise-positive yaw); ROS/FLU is
        # right-handed (counter-clockwise-positive about +Z). Confirmed live
        # 2026-07-28, with paired raw (/R1/sphera/state) vs. transformed
        # (/R1/localization) readings around a single, isolated pure-right
        # manual move (no yaw change), cross-checked against the user's own
        # direct visual confirmation that the drone moved right in Sphera:
        # raw yaw ~ -95deg, raw (dx, dy) ~= (-0.556, ~0) -- solving
        # world_ros = R(-raw_yaw) . local_ros for the local "right" command
        # shows BOTH raw_x and raw_y must be negated (not just raw_y) for
        # the transformed pose to place "right" on the correct side once
        # yaw is also negated. With only position.y negated (the prior
        # fix), FALCON's map/BEV showed the drone moving left for a real
        # right move -- this is what that looked like from the outside.
        # See LESSONS.md for the full derivation and the two earlier,
        # wrong "no bug exists" conclusions that preceded this fix.
        yaw = -float(msg.rotation.yaw)
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = -float(msg.location.x)
        pose.pose.position.y = -float(msg.location.y)
        pose.pose.position.z = float(msg.location.z)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(pose)

        source = String()
        source.data = "sphera_ground_truth"
        self.source_pub.publish(source)


def main(args=None):
    rclpy.init(args=args)
    node = RoosterGroundTruthLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
