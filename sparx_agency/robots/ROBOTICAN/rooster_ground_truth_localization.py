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
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
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

        # Velocity is differentiated here rather than in each consumer: this is
        # the only node holding the raw truth stream, and a controller closing a
        # loop on velocity must not close it on the autopilot's own estimate
        # (PX4's drifted convincingly while the aircraft sat still -- LESSONS.md).
        # World-frame linear + yaw rate; consumers rotate into body frame using
        # the pose published alongside. tau=0 disables the filter.
        self.declare_parameter("velocity_topic", "")
        # 0.25s, raised from 0.15 after the first closed-loop flight: this is a
        # differentiated position, so the noise it carries lands straight on the
        # controller's proportional term. Costs a little lag, buys a usable signal.
        self.declare_parameter("velocity_filter_tau_s", 0.25)
        velocity_topic = (self.get_parameter("velocity_topic").value
                          or f"/{rooster_id}/velocity_truth")
        self.velocity_filter_tau_s = float(
            self.get_parameter("velocity_filter_tau_s").value)
        # Prefer the physics engine's own velocity over differentiating the
        # pose -- see _publish_velocity. False restores the pre-2026-08-18
        # differentiated path.
        self.declare_parameter("use_sphera_velocity", True)
        self.declare_parameter("dead_field_samples", 25)
        self.declare_parameter("dead_field_speed_mps", 0.05)
        self.use_sphera_velocity = bool(
            self.get_parameter("use_sphera_velocity").value)
        self.dead_field_samples = int(
            self.get_parameter("dead_field_samples").value)
        self.dead_field_speed_mps = float(
            self.get_parameter("dead_field_speed_mps").value)
        self._dead_velocity_field = 0
        self._prev_sample = None      # (t_sec, x, y, z, yaw)
        self._filtered = [0.0, 0.0, 0.0, 0.0]   # vx, vy, vz, yaw_rate

        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.source_pub = self.create_publisher(String, source_topic, 10)
        self.attitude_pub = self.create_publisher(Vector3, attitude_topic, 10)
        self.velocity_pub = self.create_publisher(TwistStamped, velocity_topic, 10)
        # Sphera publishes this at BEST_EFFORT; must match explicitly or we
        # silently receive nothing (a plain int here defaults to RELIABLE).
        self.create_subscription(
            SpheraPawnState, f"/{rooster_id}/sphera/state", self._on_state,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.get_logger().info(
            f"rooster_ground_truth_localization ready for {rooster_id}\n"
            f"  sphera state in: /{rooster_id}/sphera/state\n"
            f"  pose out:        {pose_topic} (yaw-only, planar contract)\n"
            f"  attitude out:    {attitude_topic} (raw roll/pitch/yaw, rad)\n"
            f"  velocity out:    {velocity_topic} (world frame, truth-derived)"
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

        self._publish_velocity(pose, yaw, msg.velocity)

        source = String()
        source.data = "sphera_ground_truth"
        self.source_pub.publish(source)

    def _publish_velocity(self, pose: PoseStamped, yaw: float,
                          sphera_velocity=None) -> None:
        """Publish a world-frame velocity for the controllers to close on.

        Linear velocity comes from Sphera's own physics engine when it is
        available (``SpheraPawnState.velocity``, m/s) rather than from
        differentiating position. That matters more than it sounds: the
        differentiated path needed a 0.25 s low-pass to be usable at all, and
        that lag landed straight on the velocity servo's proportional term --
        it is why ``servo_kp`` had to be cut from 220 to 90 after a ~1.15 Hz
        limit cycle. Physics velocity carries neither the differentiation noise
        nor the filter lag.

        Yaw rate has no equivalent field (``Rotator`` carries angles only), so
        it stays differentiated and filtered.

        Args:
            pose: The pose just published (already sign-corrected).
            yaw: Planar yaw of that pose, radians.
            sphera_velocity: Raw ``SpheraPawnState.velocity``, in Sphera's own
                frame. ``None`` forces the differentiated path.
        """
        stamp = pose.header.stamp
        now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        x, y, z = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        prev, self._prev_sample = self._prev_sample, (now, x, y, z, yaw)
        if prev is None:
            return
        dt = now - prev[0]
        if dt <= 0.0:
            return

        yaw_delta = math.atan2(math.sin(yaw - prev[4]), math.cos(yaw - prev[4]))
        derived = [(x - prev[1]) / dt, (y - prev[2]) / dt, (z - prev[3]) / dt]
        tau = self.velocity_filter_tau_s
        alpha = 1.0 if tau <= 0.0 else dt / (tau + dt)

        physics = self._physics_velocity(sphera_velocity, derived)
        if physics is None:
            for i in range(3):
                self._filtered[i] += alpha * (derived[i] - self._filtered[i])
            linear = list(self._filtered[:3])
        else:
            # Unfiltered on purpose -- this is a measurement, not a difference.
            linear = physics
            self._filtered[:3] = physics
        self._filtered[3] += alpha * (yaw_delta / dt - self._filtered[3])

        twist = TwistStamped()
        twist.header = pose.header
        twist.twist.linear.x = linear[0]
        twist.twist.linear.y = linear[1]
        twist.twist.linear.z = linear[2]
        twist.twist.angular.z = self._filtered[3]
        self.velocity_pub.publish(twist)

    def _physics_velocity(self, raw, derived):
        """Sphera's physics velocity in ROS world frame, or None to fall back.

        Same handedness correction as the pose (x and y negated, z untouched) --
        velocity is the derivative of a position we already negate, so it must
        carry the identical sign flip.

        A vendor build that leaves the field unpopulated would otherwise pin the
        servo's feedback at zero and let its integrator wind to full deflection,
        so an all-zero field seen repeatedly while the position is demonstrably
        moving disables this path for the rest of the run, loudly.

        Args:
            raw: ``SpheraPawnState.velocity``, or None.
            derived: The differentiated ``[vx, vy, vz]``, used only to decide
                whether an all-zero field is a real standstill or a dead field.

        Returns:
            ``[vx, vy, vz]`` in the ROS world frame, or ``None`` to differentiate.
        """
        if raw is None or not self.use_sphera_velocity:
            return None
        vx, vy, vz = -float(raw.x), -float(raw.y), float(raw.z)
        if vx or vy or vz:
            self._dead_velocity_field = 0
            return [vx, vy, vz]
        if math.sqrt(sum(c * c for c in derived)) < self.dead_field_speed_mps:
            return [0.0, 0.0, 0.0]          # genuinely stationary
        self._dead_velocity_field += 1
        if self._dead_velocity_field < self.dead_field_samples:
            return [0.0, 0.0, 0.0]
        self.use_sphera_velocity = False
        self.get_logger().error(
            f"SpheraPawnState.velocity stayed all-zero for "
            f"{self._dead_velocity_field} samples while the pose moved -- this "
            f"vendor build does not populate it. Falling back to a "
            f"differentiated position for the rest of this run; expect the "
            f"~0.25 s of filter lag the servo gains were cut for.")
        return None


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
