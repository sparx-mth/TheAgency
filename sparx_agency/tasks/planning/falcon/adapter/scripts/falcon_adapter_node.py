#!/usr/bin/env python3
"""
falcon_adapter_node.py -- ROS1 adapter: drone telemetry -> FALCON topics.

Thin glue that bridges a drone's pose + depth to the topics FALCON's mapping
and exploration expect. The ROS-free algorithms live in core and are unit
tested without ROS:

  - the dead-reckoning localization noise model
    (``core.localization.dead_reckoning_noise``), and
  - the depth sensor noise model (``core.mapping.depth_noise``).

This node owns ONLY ROS concerns: rosparams -> core params, the body<->camera
extrinsic, message (de)serialization, throttling, depth-stamp recovery, the
world-frame velocity estimate, and publishing pose/odom/depth + TF.

You fly the drone (manually or via the waypoint follower); FALCON builds the
map and plans from the pose + depth it receives here.

  in   ~drone_ns + /gt_pose (Pose)
  in   ~drone_ns + /front_depth/depth/image_raw (Image)
  out  /odom_world      (Odometry)    -- FALCON's pose belief
  out  /map_ros/pose    (PoseStamped) -- camera-in-world mapping reference
  out  /map_ros/depth   (Image)       -- depth forwarded to voxel mapping
  out  TF: world->body (always GT) and body->camera

LOCALISATION NOISE (all params default to 0 -> a clean GT pass-through)
  The belief self-propagates from a noisy body-frame increment and is never
  re-anchored to GT, emulating IMU/VIO dead-reckoning without loop closure.
  See the core module docstring for the full model; the rosparams that drive
  it are listed in the footer.

See the file footer for the full rosparam list.
"""
import tf
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from sparx_agency.core.localization import se3
from sparx_agency.core.localization.dead_reckoning_noise import (
    AXES, DeadReckoningNoiseModel, DeadReckoningNoiseParams)
from sparx_agency.core.mapping.depth_noise import DepthNoiseParams, add_depth_noise

POS_AXES = ("x", "y", "z")


class FalconAdapterNode:
    def __init__(self):
        rospy.init_node("falcon_adapter")
        G = rospy.get_param

        # ── Frames / rates ──
        self.drone_ns = G("~drone_ns", "/simple_drone")
        self.world_frame = G("~world_frame", "world")
        self.body_frame = G("~body_frame", "body")
        self.cam_frame = G("~cam_frame", "camera")
        self.odom_min_dt = float(G("~odom_min_dt", 0.04))
        self.depth_min_dt = float(G("~depth_min_dt", 0.04))
        self.publish_mapping_inputs = bool(G("~publish_mapping_inputs", True))

        # ── Body->camera extrinsic (FLU body -> optical camera; mirrors
        #    mapping_sync_node so both stamp the same camera-in-world pose). ──
        cam = (float(G("~cam_offset_x", 0.2)), float(G("~cam_offset_y", 0.0)),
               float(G("~cam_offset_z", 0.0)))
        self.T_b_c = np.array([[0.0, 0.0, 1.0, cam[0]],
                               [-1.0, 0.0, 0.0, cam[1]],
                               [0.0, -1.0, 0.0, cam[2]],
                               [0.0, 0.0, 0.0, 1.0]])
        self.T_b_c_quat = se3.quaternion_from_matrix(self.T_b_c)
        self.T_b_c_trans = cam

        # ── Noise models (core). rng seeded from rosparam (-1 = nondeterministic). ──
        seed = int(G("~noise_seed", -1))
        rng = np.random.RandomState(seed if seed >= 0 else None)
        self.pose_noise = DeadReckoningNoiseModel(self._read_pose_noise(G), rng)
        self.depth_noise = self._read_depth_noise(G)
        self.pose_noise_enabled = self.pose_noise.params.enabled()

        # ── Runtime state ──
        self.cur_pose = None          # last geometry_msgs/Pose seen
        self.vel = np.zeros(3)        # world-frame velocity for odom.twist
        self.prev_time = None         # odom throttle clock
        self.prev_depth_time = None   # depth throttle clock
        self._last_vel_t = None       # velocity finite-difference clock
        self._noise_t_prev = None     # belief-integration clock
        # The localization is depth-derived, so the pose carries the depth
        # frame's capture stamp. pose_adapter publishes a bare Pose with no
        # header, so we recover the stamp here and re-apply it to the pose/odom
        # forwarded to FALCON, keeping depth and pose on one clock. None until
        # the first depth frame arrives.
        self.last_depth_stamp = None

        self.tf_br = tf.TransformBroadcaster()

        # ── Publishers (to FALCON) ──
        self.odom_pub = rospy.Publisher("/odom_world", Odometry, queue_size=10)
        self.pose_pub = None
        self.depth_pub = None
        if self.publish_mapping_inputs:
            self.pose_pub = rospy.Publisher("/map_ros/pose", PoseStamped, queue_size=10)
            self.depth_pub = rospy.Publisher("/map_ros/depth", Image, queue_size=2)

        # ── Subscribers (from drone) ──
        rospy.Subscriber(self.drone_ns + "/gt_pose", Pose, self._gt_pose_cb)
        rospy.Subscriber(self.drone_ns + "/front_depth/depth/image_raw",
                         Image, self._depth_cb)

        # Give the bridged subscribers a moment to receive their first message
        # before we start forwarding to FALCON.
        rospy.sleep(float(G("~startup_delay_sec", 1.0)))
        self._banner()

    # ─── rosparam -> core params ──────────────────────────────────
    def _read_pose_noise(self, G):
        p = DeadReckoningNoiseParams()
        for a in AXES:
            p.jitter_mean[a] = float(G("~jitter_%s_mean" % a, 0.0))
            p.jitter_std[a] = float(G("~jitter_%s_std" % a, 0.0))
            p.bias_per_s_mean[a] = float(G("~bias_%s_per_s_mean" % a, 0.0))
            p.bias_per_s_std[a] = float(G("~bias_%s_per_s_std" % a, 0.0))
        for a in POS_AXES:
            p.drift_mean_per_motion[a] = float(G("~drift_%s_mean_per_m" % a, 0.0))
            p.drift_std_per_motion[a] = float(G("~drift_%s_std_per_m" % a, 0.0))
        # Yaw drift is per RADIAN of yaw rotation (so a yaw-in-place still drifts).
        p.drift_mean_per_motion["yaw"] = float(G("~drift_yaw_mean_per_rad", 0.0))
        p.drift_std_per_motion["yaw"] = float(G("~drift_yaw_std_per_rad", 0.0))
        p.outlier_rate_hz = float(G("~outlier_rate_hz", 0.0))
        p.outlier_pos_std = float(G("~outlier_pos_std", 0.0))
        p.outlier_yaw_std = float(G("~outlier_yaw_std", 0.0))
        return p

    def _read_depth_noise(self, G):
        return DepthNoiseParams(
            std=float(G("~noise_depth_std", 0.0)),
            proportional=float(G("~noise_depth_proportional", 0.0)))

    # ─── Pose callback (drone -> FALCON) ──────────────────────────
    def _gt_pose_cb(self, msg):
        now = rospy.Time.now()
        if self.prev_time is not None and (now - self.prev_time).to_sec() < self.odom_min_dt:
            return
        self.prev_time = now

        # World-frame velocity for odom.twist (finite difference).
        if self.cur_pose is not None and self._last_vel_t is not None:
            dt_vel = max((now - self._last_vel_t).to_sec(), 1e-3)
            self.vel = np.array([
                (msg.position.x - self.cur_pose.position.x) / dt_vel,
                (msg.position.y - self.cur_pose.position.y) / dt_vel,
                (msg.position.z - self.cur_pose.position.z) / dt_vel,
            ])
        self._last_vel_t = now
        self.cur_pose = msg

        # Stamp published pose/odom with the depth frame's capture time (the
        # localization is depth-derived); fall back to wall-clock until the
        # first depth frame is seen. The dt math above uses real wall-clock.
        stamp = self.last_depth_stamp if self.last_depth_stamp is not None else now

        falcon_pose = self._belief_pose(msg, now) if self.pose_noise_enabled else msg
        fp, fo = falcon_pose.position, falcon_pose.orientation

        # 1. Odometry — FALCON's pose belief.
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.body_frame
        odom.pose.pose = falcon_pose
        odom.twist.twist.linear.x = self.vel[0]
        odom.twist.twist.linear.y = self.vel[1]
        odom.twist.twist.linear.z = self.vel[2]
        self.odom_pub.publish(odom)

        # 2. Camera-in-world sensor pose — FALCON's mapping reference.
        if self.pose_pub is not None:
            T_w_b = se3.make_transform((fp.x, fp.y, fp.z), (fo.x, fo.y, fo.z, fo.w))
            T_w_c = T_w_b @ self.T_b_c
            cam_pos = T_w_c[:3, 3]
            cam_quat = se3.quaternion_from_matrix(T_w_c)
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.world_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
            ps.pose.orientation.x, ps.pose.orientation.y = float(cam_quat[0]), float(cam_quat[1])
            ps.pose.orientation.z, ps.pose.orientation.w = float(cam_quat[2]), float(cam_quat[3])
            self.pose_pub.publish(ps)

        # 3. TF — always GT, so RViz stays a fair witness of the true pose.
        gt_p, gt_o = msg.position, msg.orientation
        self.tf_br.sendTransform((gt_p.x, gt_p.y, gt_p.z),
                                 (gt_o.x, gt_o.y, gt_o.z, gt_o.w),
                                 stamp, self.body_frame, self.world_frame)
        self.tf_br.sendTransform(self.T_b_c_trans, self.T_b_c_quat,
                                 stamp, self.cam_frame, self.body_frame)

    def _belief_pose(self, gt_msg, now):
        """Advance the dead-reckoning belief one tick and return its Pose."""
        q = gt_msg.orientation
        T_gt = se3.make_transform(
            (gt_msg.position.x, gt_msg.position.y, gt_msg.position.z),
            (q.x, q.y, q.z, q.w))
        dt = (now - self._noise_t_prev).to_sec() if self._noise_t_prev else 0.0
        self._noise_t_prev = now
        T_pub = self.pose_noise.step(T_gt, dt)
        out = Pose()
        out.position.x, out.position.y, out.position.z = (
            float(T_pub[0, 3]), float(T_pub[1, 3]), float(T_pub[2, 3]))
        qx, qy, qz, qw = se3.quaternion_from_matrix(T_pub)
        out.orientation.x, out.orientation.y = float(qx), float(qy)
        out.orientation.z, out.orientation.w = float(qz), float(qw)
        return out

    # ─── Depth callback ───────────────────────────────────────────
    def _depth_cb(self, msg):
        now = rospy.Time.now()
        if self.prev_depth_time is not None and (now - self.prev_depth_time).to_sec() < self.depth_min_dt:
            return
        self.prev_depth_time = now

        # Remember the REAL capture stamp so _gt_pose_cb can stamp the matching
        # pose with it. Do NOT restamp the depth: it already carries the capture
        # time, and the forwarded pose is stamped with that same value, so depth
        # and pose stay aligned on one clock. FALCON's voxel mapping reads the
        # native encoding (16UC1 mm or 32FC1 m) itself, so we forward unchanged
        # except for optional 32FC1 noise injection.
        self.last_depth_stamp = msg.header.stamp
        if self.depth_noise.enabled() and msg.encoding == "32FC1":
            msg = self._noisy_depth(msg)
        if self.depth_pub is not None:
            self.depth_pub.publish(msg)

    def _noisy_depth(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        noisy = add_depth_noise(arr, self.depth_noise, self.pose_noise.rng)
        out = Image()
        out.header = msg.header
        out.height, out.width = msg.height, msg.width
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = msg.step
        out.data = noisy.tobytes()
        return out

    # ─── Banner ───────────────────────────────────────────────────
    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("falcon_adapter (core dead-reckoning + depth noise)")
        L("  drone_ns = %s", self.drone_ns)
        L("  pose noise = %s   depth noise = %s",
          "on" if self.pose_noise_enabled else "off",
          "on" if self.depth_noise.enabled() else "off")
        L("  publish_mapping_inputs = %s  (pose+depth to FALCON)",
          self.publish_mapping_inputs)
        L("  cam_offset = (%.3f, %.3f, %.3f)", *self.T_b_c_trans)
        L("=" * 64)


def main():
    try:
        FalconAdapterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The noise maths live in
# core.localization.dead_reckoning_noise and core.mapping.depth_noise; this node
# maps rosparams -> those params and owns ROS I/O, the camera extrinsic,
# throttling, depth-stamp recovery and TF.
#
#   frames/io: ~drone_ns (/simple_drone) [+/gt_pose +/front_depth/depth/image_raw]
#       ~world_frame (world) ~body_frame (body) ~cam_frame (camera)
#       ~cam_offset_x/y/z (0.2/0.0/0.0) ~publish_mapping_inputs (true)
#       ~odom_min_dt (0.04) ~depth_min_dt (0.04) ~startup_delay_sec (1.0)
#   pose noise (all 0 = clean GT pass-through):
#       ~jitter_{x,y,z,yaw}_mean/_std        per-tick jitter (published only)
#       ~drift_{x,y,z}_mean_per_m /_std_per_m  scale-factor drift per metre
#       ~drift_yaw_mean_per_rad /_std_per_rad  yaw drift per radian turned
#       ~bias_{x,y,z,yaw}_per_s_mean/_std    always-on bias rate (per second)
#       ~outlier_rate_hz (0) ~outlier_pos_std (0) ~outlier_yaw_std (0)
#       ~noise_seed (-1 = nondeterministic)
#   depth noise: ~noise_depth_std (0) ~noise_depth_proportional (0)  [32FC1 only]
# ============================================================================
