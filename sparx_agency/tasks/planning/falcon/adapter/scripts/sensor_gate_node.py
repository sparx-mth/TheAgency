#!/usr/bin/env python3
"""
sensor_gate_node.py -- ROS1 adapter: pose+depth pass-through that can FREEZE.

Sits between the drone (Gazebo, the ros1_bridge, or a real drone) and
falcon_adapter. While the platform rotates in place it stops republishing live
pose/depth and instead replays the last frame seen before the freeze, so
FALCON's map does not smear during the turn. When the turn ends it does NOT
immediately trust the next live frame: a frame captured *during* the rotation
can be delivered late (the mode topic and the depth topic have independent
latencies), so the gate keeps replaying until a frame captured strictly after
the turn ended arrives, and only then resumes the live stream. That is exactly
"start updating from the forward-flight state, never re-link an old turn frame".

The DECISION is the ROS-free mapping algorithm in
``core.mapping.depth_fusion_gate`` (mode-authoritative freeze +
capture-time staleness guard; unit tested without ROS). This node owns ONLY ROS
concerns: the topics, the replay timer, the diagnostic heartbeat and the manual
recovery hook. It feeds the gate each depth frame's capture stamp and the wall
clock, and gates the non-depth streams on the gate's "is the live stream
flowing?" query so the pose and depth a CO-LOCATED consumer pairs stay a
consistent snapshot across the turn.

Scope: that pose/depth consistency only holds for a consumer that reads BOTH the
gated pose (``~out_ns + /gt_pose``) and the gated depth from this node — i.e.
``falcon_adapter`` in the pure-Gazebo nav stack. The real-drone voxel path
(``mapping_sync``) pairs this node's gated depth with an UN-gated localization
PoseStamped, so freezing the gated pose here does not freeze it; that path runs
its OWN authoritative freeze (the same ``DepthFusionGate``, on the raw depth, in
the capture clock). Clock note: this node arms its resume watermark from the
newest depth CAPTURE stamp it has seen (not the host wall clock), so it can
never deadlock under a host/sensor clock offset; it just lacks the pose-clock
"now" that lets ``mapping_sync`` also reject a turn frame that overtakes the
mode signal — acceptable here because this gate is co-located with its sensor.

  in   ~pose_topic        (Pose)        default ~in_ns + /gt_pose
  in   ~depth_topic       (Image)       default ~in_ns + /front_depth/depth/image_raw
  in   ~camera_info_topic (CameraInfo)  only when ~bridge_camera_info=true
  in   /sensor_gate/freeze (Bool)            explicit per-state request
  in   ~demo_mode_topic (String)             system mode (authoritative)
  in   /sensor_gate/reset_mode_freeze (Bool) manual stuck-mode recovery
  out  ~out_ns + /gt_pose                       (Pose)
  out  ~out_ns + /front_depth/depth/image_raw   (Image)
  out  ~out_ns + /front_depth/depth/camera_info (CameraInfo, optional)

Point falcon_adapter's ~drone_ns at this gate's ~out_ns.

See the file footer for the full rosparam list.
"""
import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo

from sparx_agency.core.mapping.depth_fusion_gate import DepthFusionGate
from sparx_agency.core.mapping.sensor_freeze_policy import SensorFreezePolicy


class SensorGateNode:
    def __init__(self):
        rospy.init_node("sensor_gate")
        G = rospy.get_param

        self.in_ns = G("~in_ns", "/simple_drone")
        self.out_ns = G("~out_ns", "/gated_drone")
        self.replay_hz = float(G("~replay_hz", 30.0))
        self.bridge_caminfo = bool(G("~bridge_camera_info", False))

        # Rotation-aware depth-fusion gate (mode-authoritative when enabled).
        # turning_mode_name is the demo-mode string that means "freeze"; matched
        # case-insensitively. resume_settle_sec adds margin to the post-turn
        # staleness watermark (0 = admit as soon as a frame is captured after the
        # turn ends; bump it to also skip the brief physical settle).
        self.demo_mode_topic = G("~demo_mode_topic", "/xtend/demo_mode")
        self.turning_mode_name = str(G("~turning_mode_name", "turning")).strip().lower()
        self.hb_period = float(G("~heartbeat_period_sec", 2.0))
        self.gate = DepthFusionGate(
            policy=SensorFreezePolicy(
                freeze_on_turning_mode=bool(G("~freeze_on_turning_mode", True))),
            resume_settle_sec=float(G("~resume_settle_sec", 0.0)))

        # Topic resolution: explicit private param wins, else the sjtu_drone default.
        self.in_pose_t = G("~pose_topic", self.in_ns + "/gt_pose")
        self.in_depth_t = G("~depth_topic", self.in_ns + "/front_depth/depth/image_raw")
        self.in_caminfo_t = G("~camera_info_topic",
                              self.in_ns + "/front_depth/depth/camera_info")

        # ── Caches (the snapshot replayed while not passing) ──
        self.last_pose = None
        self.last_depth = None
        self.last_caminfo = None

        # ── Diagnostics ──
        self._n_explicit_msgs = 0
        self._n_drop_frozen = 0
        self._n_drop_stale = 0
        self._n_passed_depth = 0
        self._last_mode_str = "(no msgs yet)"
        self._last_explicit_str = "(no msgs yet)"
        self._last_state = self._state_label()
        self._last_state_change_t = rospy.Time.now()

        # ── Publishers ──
        self.pub_pose = rospy.Publisher(self.out_ns + "/gt_pose", Pose, queue_size=1)
        self.pub_depth = rospy.Publisher(
            self.out_ns + "/front_depth/depth/image_raw", Image, queue_size=2)
        self.pub_caminfo = None
        if self.bridge_caminfo:
            self.pub_caminfo = rospy.Publisher(
                self.out_ns + "/front_depth/depth/camera_info", CameraInfo, queue_size=2)

        # ── Subscribers ──
        rospy.Subscriber(self.in_pose_t, Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.in_depth_t, Image, self._depth_cb, queue_size=2)
        if self.bridge_caminfo:
            rospy.Subscriber(self.in_caminfo_t, CameraInfo, self._caminfo_cb, queue_size=2)
        rospy.Subscriber("/sensor_gate/freeze", Bool, self._freeze_cb, queue_size=1)
        if self.gate.policy.freeze_on_turning_mode:
            rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)
        rospy.Subscriber("/sensor_gate/reset_mode_freeze", Bool,
                         self._reset_mode_freeze_cb, queue_size=1)

        # ── Timers ──
        rospy.Timer(rospy.Duration(1.0 / self.replay_hz), self._replay)
        if self.hb_period > 0.0:
            rospy.Timer(rospy.Duration(self.hb_period), self._heartbeat)

        self._banner()

    # ── Derived state ──
    def _state_label(self):
        if self.gate.frozen:
            return "FROZEN"
        if self.gate.awaiting_fresh_frame:
            return "RESUME_WAIT"
        return "FUSING"

    def _log_transition(self, trigger):
        """Log when the gate moves between FUSING / FROZEN / RESUME_WAIT."""
        new = self._state_label()
        if new != self._last_state:
            self._last_state_change_t = rospy.Time.now()
            rospy.loginfo(
                "sensor_gate: %s -> %s  src=%s  (explicit=%s mode_turning=%s "
                "last_mode=%r) trigger=%s",
                self._last_state, new, self.gate.source,
                self.gate.policy.explicit_freeze, self.gate.policy.mode_says_freeze,
                self._last_mode_str, trigger)
            self._last_state = new

    # ── Subscribers ──
    def _freeze_cb(self, msg):
        self._n_explicit_msgs += 1
        new = bool(msg.data)
        self._last_explicit_str = "True" if new else "False"
        if new != self.gate.policy.explicit_freeze:
            # No wall-clock `now`: this node has no reliable capture-clock source
            # (the pose is unstamped), so the resume watermark falls back to the
            # newest depth capture stamp seen. That keeps it in the capture clock
            # and cannot deadlock, regardless of any host/sensor clock offset.
            self.gate.note_explicit(new)
            self._log_transition("/sensor_gate/freeze=%s" % new)

    def _demo_mode_cb(self, msg):
        mode = (msg.data or "").strip().lower()
        self._last_mode_str = mode if mode else "(empty)"
        self.gate.note_mode(mode == self.turning_mode_name)
        self._log_transition("demo_mode=%r" % mode)

    def _reset_mode_freeze_cb(self, msg):
        if self.gate.policy.mode_says_freeze:
            rospy.logwarn("sensor_gate: MANUAL reset of mode-based freeze "
                          "(last_mode=%r, msgs=%d)",
                          self._last_mode_str, self.gate.policy.n_mode_msgs)
            self.gate.reset_freeze()
            self._log_transition("manual /sensor_gate/reset_mode_freeze")

    # ── Pass-through ──
    def _pose_cb(self, msg):
        # Pose/camera_info follow the live-stream state: held (replayed) while
        # frozen OR while awaiting the first fresh depth, so the snapshot fed to
        # the mapper stays a consistent (pose, depth) pair across the resume.
        if not self.gate.is_passing():
            return
        self.last_pose = msg
        self.pub_pose.publish(msg)

    def _depth_cb(self, msg):
        # Drive the gate with this frame's capture stamp on EVERY frame (even
        # while frozen) so the resume watermark tracks how far the turn reached.
        fuse, reason = self.gate.should_fuse(msg.header.stamp.to_sec())
        if not fuse:
            if reason == "stale_after_rotation":
                self._n_drop_stale += 1
            else:
                self._n_drop_frozen += 1
            return
        self._n_passed_depth += 1
        # A fresh frame can also retire RESUME_WAIT; surface that transition.
        self._log_transition("fresh depth resumed live stream")
        self.last_depth = msg
        self.pub_depth.publish(msg)

    def _caminfo_cb(self, msg):
        if not self.gate.is_passing():
            return
        self.last_caminfo = msg
        self.pub_caminfo.publish(msg)

    # ── Replay while not passing (refresh stamps so downstream age checks pass) ──
    def _replay(self, _evt):
        if self.gate.is_passing():
            return
        now = rospy.Time.now()
        if self.last_pose is not None:
            self.pub_pose.publish(self.last_pose)
        if self.last_depth is not None:
            self.last_depth.header.stamp = now
            self.pub_depth.publish(self.last_depth)
        if self.pub_caminfo is not None and self.last_caminfo is not None:
            self.last_caminfo.header.stamp = now
            self.pub_caminfo.publish(self.last_caminfo)

    # ── Diagnostics ──
    def _heartbeat(self, _evt):
        age = (rospy.Time.now() - self._last_state_change_t).to_sec()
        rospy.loginfo(
            "sensor_gate hb: state=%s src=%s  [explicit=%s mode_turning=%s]  "
            "last_mode=%r last_explicit=%s  msgs: mode=%d explicit=%d  "
            "depth: passed=%d drop[frozen=%d stale=%d]  age=%.1fs",
            self._state_label(), self.gate.source,
            self.gate.policy.explicit_freeze, self.gate.policy.mode_says_freeze,
            self._last_mode_str, self._last_explicit_str,
            self.gate.policy.n_mode_msgs, self._n_explicit_msgs,
            self._n_passed_depth, self._n_drop_frozen, self._n_drop_stale, age)

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("sensor_gate  in_ns=%s  out_ns=%s  replay=%.0fHz",
          self.in_ns, self.out_ns, self.replay_hz)
        L("  in pose  = %s", self.in_pose_t)
        L("  in depth = %s", self.in_depth_t)
        L("  camera_info = %s",
          self.in_caminfo_t if self.bridge_caminfo
          else "DISABLED (intrinsics from rosparam)")
        if self.gate.policy.freeze_on_turning_mode:
            L("  freeze logic = MODE-AUTHORITATIVE  frozen <- (%s == %r)",
              self.demo_mode_topic, self.turning_mode_name)
            L("                 /sensor_gate/freeze is fallback before first mode msg")
        else:
            L("  freeze logic = EXPLICIT-ONLY  frozen <- /sensor_gate/freeze")
        L("  resume guard = drop frames captured <= turn-end + %.2fs (stale in-flight)",
          self.gate.resume_settle_sec)
        L("  manual override: rostopic pub --once "
          "/sensor_gate/reset_mode_freeze std_msgs/Bool 'data: true'")
        L("=" * 64)


def main():
    try:
        SensorGateNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The rotation-aware
# fuse/drop decision lives in core.mapping.depth_fusion_gate (mode-authoritative
# freeze + post-turn capture-time staleness guard); this node owns the ROS I/O,
# the replay timer and the heartbeat.
#
#   io: ~in_ns (/simple_drone) ~out_ns (/gated_drone) ~replay_hz (30.0)
#       ~pose_topic (in_ns/gt_pose) ~depth_topic (in_ns/front_depth/depth/image_raw)
#       ~camera_info_topic (in_ns/front_depth/depth/camera_info)
#       ~bridge_camera_info (false)
#   freeze: ~freeze_on_turning_mode (true) ~demo_mode_topic (/xtend/demo_mode)
#       ~turning_mode_name (turning) ~resume_settle_sec (0.0)
#       ~heartbeat_period_sec (2.0; 0 disables)
# ============================================================================
