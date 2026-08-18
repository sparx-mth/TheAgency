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
expects: z=sin(yaw/2), w=cos(yaw/2), x=y=0 (planar rotation only) -- this is
a deliberate, shared contract with other consumers (DA3/room_mapper), not a
bug, and must not change.

**2026-08-17 addition**: the pose above being yaw-only means NOTHING
downstream of it can ever see the aircraft's real roll/pitch -- confirmed
live as the root cause of a follower's tilt-cutoff reflex never firing
through an actual capsize (see LESSONS.md). Sphera's own SpheraPawnState
carries real roll/pitch (``msg.rotation.roll/pitch``), so this node now also
republishes them, raw and unmodified, on a separate topic
(``attitude_topic``, default ``/{rooster_id}/attitude_rpy``, a plain
``geometry_msgs/Vector3`` so it bridges to ROS1 without a custom message
type). Sign convention for roll/pitch is UNVERIFIED -- consumers should use
magnitude only (e.g. ``abs(roll) > limit``), not signed comparisons, until
someone measures it the way x/y/yaw were measured (see this file's own
comment on those).
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import String
from sphera_common_interfaces.msg import SpheraPawnState


class RoosterGroundTruthLocalization(Node):
    def __init__(self):
        super().__init__("rooster_ground_truth_localization")

        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("pose_topic", "")
        self.declare_parameter("source_topic", "")
        self.declare_parameter("attitude_topic", "")
        # 2026-08-17: /<id>/sphera/state has TWO publishers -- the live pawn
        # plus a second one emitting a pose ~677m from origin, BOTH stamped
        # frame_id "Rooster_1", and it survives a full Sphera restart. Left
        # unfiltered, this node republished them alternately, so
        # /<id>/localization flip-flopped between the real pose and a bogus
        # one every sample -- which FALCON fuses straight into its voxel map.
        # Two independent guards, either can be disabled with <=0.0:
        #   reject_radius_m: drop anything farther than this from origin
        #                    (measured: real pawn ~62m, bogus ~677m).
        #   max_jump_m:      after the first accepted pose, drop samples that
        #                    teleport further than this between messages.
        self.declare_parameter("reject_radius_m", 300.0)
        self.declare_parameter("max_jump_m", 5.0)
        # The jump guard MUST be able to re-latch: the drone legitimately
        # teleports when Sphera respawns R1, and a latch held from before the
        # respawn otherwise rejects every real sample forever (measured: 13k
        # consecutive drops and a totally silent /localization). After this
        # many consecutive rejections, accept the new position as truth.
        self.declare_parameter("relatch_after_rejects", 25)
        rooster_id = self.get_parameter("rooster_id").value
        pose_topic = self.get_parameter("pose_topic").value or f"/{rooster_id}/localization"
        source_topic = self.get_parameter("source_topic").value or f"/{rooster_id}/localization_source"
        attitude_topic = self.get_parameter("attitude_topic").value or f"/{rooster_id}/attitude_rpy"
        self.reject_radius_m = float(self.get_parameter("reject_radius_m").value)
        self.max_jump_m = float(self.get_parameter("max_jump_m").value)
        self.relatch_after_rejects = int(self.get_parameter("relatch_after_rejects").value)
        self._last_xy = None
        self._rejected = 0
        self._consecutive_rejects = 0

        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.source_pub = self.create_publisher(String, source_topic, 10)
        self.attitude_pub = self.create_publisher(Vector3, attitude_topic, 10)
        # Sphera publishes this at BEST_EFFORT; must match explicitly or we
        # silently receive nothing (a plain int here defaults to RELIABLE).
        self.create_subscription(
            SpheraPawnState, f"/{rooster_id}/sphera/state", self._on_state,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.get_logger().info(
            f"rooster_ground_truth_localization ready for {rooster_id}\n"
            f"  sphera state in: /{rooster_id}/sphera/state\n"
            f"  pose out:        {pose_topic} (yaw-only, planar contract)\n"
            f"  attitude out:    {attitude_topic} (raw roll/pitch/yaw, rad)"
        )

    def _keep(self, x: float, y: float) -> bool:
        """Reject the second, bogus /sphera/state publisher's poses.

        See the reject_radius_m/max_jump_m parameter comments. Rejections are
        logged (throttled) rather than silently dropped, because a storm of
        them means the guard is mis-tuned for this map, not that the defect
        went away.
        """
        if self.reject_radius_m > 0.0 and math.hypot(x, y) > self.reject_radius_m:
            self._note_reject("outside reject_radius_m", x, y)
            return False
        if (self.max_jump_m > 0.0 and self._last_xy is not None
                and math.hypot(x - self._last_xy[0], y - self._last_xy[1]) > self.max_jump_m):
            self._note_reject("jump exceeds max_jump_m", x, y)
            self._consecutive_rejects += 1
            if self._consecutive_rejects < self.relatch_after_rejects:
                return False
            # Sustained divergence is a real teleport (R1 respawn), not a
            # glitch -- adopt it rather than stay latched on a dead position.
            self.get_logger().warn(
                f"re-latching pose to ({x:.1f}, {y:.1f}) after "
                f"{self._consecutive_rejects} consecutive rejects (respawn?)")
        self._consecutive_rejects = 0
        self._last_xy = (x, y)
        return True

    def _note_reject(self, why: str, x: float, y: float) -> None:
        self._rejected += 1
        if self._rejected % 200 == 1:
            self.get_logger().warn(
                f"dropped implausible sphera pose ({why}): ({x:.1f}, {y:.1f}) "
                f"-- {self._rejected} so far (known dual-publisher defect)")

    def _on_state(self, msg: SpheraPawnState):
        # x/y negated for handedness; yaw needs no sense negation but does
        # need a +pi reference offset (yaw=0 means facing world +X here, not
        # the raw feed's own zero). See LESSONS.md for the derivation of both.
        if not self._keep(-float(msg.location.x), -float(msg.location.y)):
            return
        yaw = math.atan2(math.sin(float(msg.rotation.yaw) + math.pi),
                          math.cos(float(msg.rotation.yaw) + math.pi))
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = -float(msg.location.x)
        pose.pose.position.y = -float(msg.location.y)
        pose.pose.position.z = float(msg.location.z)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(pose)

        attitude = Vector3()
        attitude.x = float(msg.rotation.roll)
        attitude.y = float(msg.rotation.pitch)
        attitude.z = float(msg.rotation.yaw)
        self.attitude_pub.publish(attitude)

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
