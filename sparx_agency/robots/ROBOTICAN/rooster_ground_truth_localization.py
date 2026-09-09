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

import time
from collections import deque

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
        # Metres of motion that separate the real aircraft from Sphera's
        # stationary impostor pawn. 0 disables the cluster test below.
        #
        # Sphera intermittently publishes a second, stationary pawn on this
        # topic. Its share of the messages can be the large majority (83% of one
        # flight), so a streak of `relatch_after_rejects` from it is routine and
        # the relatch ratchets onto it -- after which the REAL aircraft's poses
        # are the ones rejected, its streaks are the rare ones, and the latch
        # never comes back. Measured 2026-09-02: a whole 453 s flight published
        # a pose frozen at the impostor's position while the aircraft flew, and
        # the follower drove the axes to their 900-count ceiling against an
        # error that could not close.
        #
        # The consecutive-reject relatch cannot help: the two publishers
        # INTERLEAVE, so every impostor message is accepted (it matches the
        # latch) and resets the streak, and 25 consecutive rejects never
        # accumulate. Replaying a real contaminated flight confirmed it -- the
        # latch stayed on the impostor for all 8749 poses either way.
        #
        # What does separate them is motion, over a window rather than a streak:
        # across that flight the impostor's position spanned 2.9e-05 m and the
        # real aircraft's spanned 44 m. So when the ACCEPTED cluster is static
        # and the REJECTED one is moving, the latch is on the wrong pawn and is
        # handed over. 0.01 m is ~300x the impostor's noise and well below a
        # hovering drone's own centimetre jitter.
        #
        # Costs nothing on a healthy stack: with one publisher there are no
        # rejects, so this code never runs.
        self.declare_parameter("relatch_min_span_m", 0.01)
        # Seconds of history the cluster test compares over.
        self.declare_parameter("relatch_window_s", 3.0)
        rooster_id = self.get_parameter("rooster_id").value
        pose_topic = self.get_parameter("pose_topic").value or f"/{rooster_id}/localization"
        source_topic = self.get_parameter("source_topic").value or f"/{rooster_id}/localization_source"
        attitude_topic = self.get_parameter("attitude_topic").value or f"/{rooster_id}/attitude_rpy"
        self.reject_radius_m = float(self.get_parameter("reject_radius_m").value)
        self.max_jump_m = float(self.get_parameter("max_jump_m").value)
        self.relatch_after_rejects = int(self.get_parameter("relatch_after_rejects").value)
        self.relatch_min_span_m = float(
            self.get_parameter("relatch_min_span_m").value)
        self.relatch_window_s = float(
            self.get_parameter("relatch_window_s").value)
        #: Recent (t, x, y) for each cluster, for the motion comparison.
        self._accepted_recent = deque()
        self._rejected_recent = deque()
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
        # 0.05s. Was 0.25 while velocity was differenced between CONSECUTIVE
        # samples; with the fixed window below most of the noise is gone before
        # the filter sees it, so the lag no longer has to be paid. See
        # velocity_window_s.
        self.declare_parameter("velocity_filter_tau_s", 0.05)
        # Seconds spanned by the position difference. 0 restores the old
        # consecutive-sample behaviour.
        #
        # Sphera stamps its state at ~129 Hz with 2.3x dt jitter (p90 18.1 ms
        # against a 7.9 ms median, measured live 2026-09-02 off the message
        # HEADER, not arrival), and a stamp that disagrees with the physics tick
        # by a few ms turns into metres per second once divided by 8 ms: the raw
        # consecutive derivative runs p90 1.0 m/s of noise on a 0.5 m/s signal.
        # Differencing over a fixed 60 ms window divides the same timing error
        # by 8x instead. Measured over 2958 live samples, the pair below is
        # strictly better than the old one on BOTH axes -- 0.9x the tick-to-tick
        # noise and 80 ms of total lag against 250 ms.
        self.declare_parameter("velocity_window_s", 0.06)
        velocity_topic = (self.get_parameter("velocity_topic").value
                          or f"/{rooster_id}/velocity_truth")
        self.velocity_filter_tau_s = float(
            self.get_parameter("velocity_filter_tau_s").value)
        self.velocity_window_s = float(
            self.get_parameter("velocity_window_s").value)
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
        #: Recent samples, newest last, trimmed to just span velocity_window_s.
        self._window = deque()        # of (t_sec, x, y, z, yaw)
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
            self._remember(self._rejected_recent, x, y)
            if self._latch_is_on_a_static_pawn():
                self.get_logger().warn(
                    f"re-latching pose to ({x:.1f}, {y:.1f}): the latched pose "
                    f"has not moved while this one has -- the latch was on "
                    f"Sphera's stationary impostor pawn (known dual-publisher "
                    f"defect)", throttle_duration_sec=10.0)
                self._consecutive_rejects = 0
                self._accepted_recent.clear()
                self._rejected_recent.clear()
                self._last_xy = (x, y)
                self._remember(self._accepted_recent, x, y)
                return True
            self._consecutive_rejects += 1
            if self._consecutive_rejects < self.relatch_after_rejects:
                return False
            # Sustained divergence is a real teleport (R1 respawn), not a
            # glitch -- adopt it rather than stay latched on a dead position.
            #
            # UNLESS the candidate has not moved. Observed live 2026-09-02: the
            # cluster test above correctly moved the latch onto the real
            # aircraft, and 2.5 s later THIS path dragged it back to the
            # impostor, because the impostor's ~83 % share reaches 25
            # consecutive rejects within seconds while the aircraft's does not.
            # A respawned aircraft still drifts by centimetres; the impostor
            # spans tens of micrometres.
            span = self._span(self._rejected_recent)
            if (self.relatch_min_span_m > 0.0 and len(self._rejected_recent) >= 5
                    and span < self.relatch_min_span_m):
                self.get_logger().warn(
                    f"refusing the streak re-latch to ({x:.1f}, {y:.1f}): "
                    f"{self._consecutive_rejects} consecutive rejects but the "
                    f"candidate moved only {span * 1000.0:.2f} mm -- stationary "
                    f"pawn, not a respawn", throttle_duration_sec=10.0)
                self._consecutive_rejects = 0
                return False
            self.get_logger().warn(
                f"re-latching pose to ({x:.1f}, {y:.1f}) after "
                f"{self._consecutive_rejects} consecutive rejects (respawn?)")
        self._consecutive_rejects = 0
        self._last_xy = (x, y)
        self._remember(self._accepted_recent, x, y)
        return True

    def _remember(self, window, x: float, y: float) -> None:
        """Append a pose to a cluster's history and drop what has aged out."""
        now = time.monotonic()
        window.append((now, x, y))
        while window and now - window[0][0] > self.relatch_window_s:
            window.popleft()

    @staticmethod
    def _span(window) -> float:
        """Widest separation among a cluster's recent poses, metres."""
        if len(window) < 2:
            return 0.0
        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def _latch_is_on_a_static_pawn(self) -> bool:
        """Whether the pose we are latched to is the impostor and this one is not.

        Symmetric and conservative: it fires only when BOTH halves are true over
        a full window -- the accepted cluster has not moved, and the rejected one
        has. A real aircraft flying while a second static pawn interleaves gives
        exactly that; nothing else does. On a healthy stack there are no rejects,
        so this never runs.
        """
        if self.relatch_min_span_m <= 0.0:
            return False
        if len(self._accepted_recent) < 5 or len(self._rejected_recent) < 5:
            return False
        oldest = min(self._accepted_recent[0][0], self._rejected_recent[0][0])
        if time.monotonic() - oldest < self.relatch_window_s:
            return False                      # not a full window yet
        return (self._span(self._accepted_recent) < self.relatch_min_span_m
                and self._span(self._rejected_recent) >= self.relatch_min_span_m)

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
        step = now - prev[0]
        if step <= 0.0:
            return

        base = self._window_base(now, x, y, z, yaw)
        dt = now - base[0]
        if dt <= 0.0:
            return

        yaw_delta = math.atan2(math.sin(yaw - base[4]), math.cos(yaw - base[4]))
        derived = [(x - base[1]) / dt, (y - base[2]) / dt, (z - base[3]) / dt]
        tau = self.velocity_filter_tau_s
        # The filter steps on the SAMPLE interval, not on the window it differences
        # over -- those are different times and using dt here would make the
        # filter's own time constant depend on the window length.
        alpha = 1.0 if tau <= 0.0 else step / (tau + step)

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

    def _window_base(self, now, x, y, z, yaw):
        """The sample to difference against: the oldest still inside the window.

        Returns the immediately previous sample when ``velocity_window_s`` is 0,
        which is the pre-2026-09-02 behaviour.

        Args:
            now: Stamp of the sample just received, seconds.
            x, y, z: Its sign-corrected world position, metres.
            yaw: Its planar yaw, radians.

        Returns:
            ``(t, x, y, z, yaw)`` to difference against.
        """
        self._window.append((now, x, y, z, yaw))
        window = self.velocity_window_s
        if window <= 0.0:
            while len(self._window) > 2:
                self._window.popleft()
            return self._window[0]
        # Keep exactly one sample older than the window, so the span is >= window
        # rather than the first one that happens to fall short of it.
        while len(self._window) > 2 and now - self._window[1][0] >= window:
            self._window.popleft()
        return self._window[0]

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
