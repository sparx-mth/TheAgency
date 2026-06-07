#!/usr/bin/env python3
"""
sensor_gate_node.py -- ROS1 adapter: pose+depth pass-through that can FREEZE.

Sits between the drone (Gazebo, the ros1_bridge, or a real drone) and
falcon_adapter. While frozen it stops republishing live pose/depth and instead
replays the last frame seen before the freeze, so FALCON's map does not smear
during an in-place rotation.

The freeze DECISION is the ROS-free, mode-authoritative policy in
``core.planning.sensor_freeze_policy`` (the planner decides not to fuse the
voxel map while rotating; unit tested without ROS). This node owns ONLY ROS
concerns: the topics, the replay timer, the diagnostic heartbeat and the manual
recovery hook.

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

from sparx_agency.core.planning.sensor_freeze_policy import SensorFreezePolicy


class SensorGateNode:
    def __init__(self):
        rospy.init_node("sensor_gate")
        G = rospy.get_param

        self.in_ns = G("~in_ns", "/simple_drone")
        self.out_ns = G("~out_ns", "/gated_drone")
        self.replay_hz = float(G("~replay_hz", 30.0))
        self.bridge_caminfo = bool(G("~bridge_camera_info", False))

        # Freeze policy (mode-authoritative when enabled). turning_mode_name is
        # the demo-mode string that means "freeze"; matched case-insensitively.
        self.policy = SensorFreezePolicy(
            freeze_on_turning_mode=bool(G("~freeze_on_turning_mode", True)))
        self.demo_mode_topic = G("~demo_mode_topic", "/xtend/demo_mode")
        self.turning_mode_name = str(G("~turning_mode_name", "turning")).strip().lower()
        self.hb_period = float(G("~heartbeat_period_sec", 2.0))

        # Topic resolution: explicit private param wins, else the sjtu_drone default.
        self.in_pose_t = G("~pose_topic", self.in_ns + "/gt_pose")
        self.in_depth_t = G("~depth_topic", self.in_ns + "/front_depth/depth/image_raw")
        self.in_caminfo_t = G("~camera_info_topic",
                              self.in_ns + "/front_depth/depth/camera_info")

        # ── Effective state + caches ──
        self.frozen = False
        self.freeze_source = None
        self.last_pose = None
        self.last_depth = None
        self.last_caminfo = None

        # ── Diagnostics ──
        self._n_explicit_msgs = 0
        self._last_mode_str = "(no msgs yet)"
        self._last_explicit_str = "(no msgs yet)"
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
        if self.policy.freeze_on_turning_mode:
            rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)
        rospy.Subscriber("/sensor_gate/reset_mode_freeze", Bool,
                         self._reset_mode_freeze_cb, queue_size=1)

        # ── Timers ──
        rospy.Timer(rospy.Duration(1.0 / self.replay_hz), self._replay)
        if self.hb_period > 0.0:
            rospy.Timer(rospy.Duration(self.hb_period), self._heartbeat)

        self._recompute("startup")
        self._banner()

    # ── Freeze recompute (delegates the decision to the core policy) ──
    def _recompute(self, trigger):
        new, source = self.policy.decide()
        if new != self.frozen or source != self.freeze_source:
            if new != self.frozen:
                self._last_state_change_t = rospy.Time.now()
            rospy.loginfo(
                "sensor_gate: %s  src=%s  (explicit=%s mode_turning=%s "
                "last_mode=%r) trigger=%s",
                "FREEZE" if new else "UNFREEZE", source,
                self.policy.explicit_freeze, self.policy.mode_says_freeze,
                self._last_mode_str, trigger)
        self.frozen = new
        self.freeze_source = source

    # ── Subscribers ──
    def _freeze_cb(self, msg):
        self._n_explicit_msgs += 1
        new = bool(msg.data)
        self._last_explicit_str = "True" if new else "False"
        if new != self.policy.explicit_freeze:
            self.policy.note_explicit(new)
            self._recompute("/sensor_gate/freeze=%s" % new)

    def _demo_mode_cb(self, msg):
        mode = (msg.data or "").strip().lower()
        self._last_mode_str = mode if mode else "(empty)"
        was_turning = self.policy.mode_says_freeze
        first = self.policy.n_mode_msgs == 0
        self.policy.note_mode(mode == self.turning_mode_name)
        # Recompute on the first message (it may promote us from
        # explicit_fallback to mode_auth even if the bool value is unchanged).
        if self.policy.mode_says_freeze != was_turning or first:
            self._recompute("demo_mode=%r" % mode)

    def _reset_mode_freeze_cb(self, msg):
        if self.policy.mode_says_freeze:
            rospy.logwarn("sensor_gate: MANUAL reset of mode-based freeze "
                          "(last_mode=%r, msgs=%d)",
                          self._last_mode_str, self.policy.n_mode_msgs)
            self.policy.reset_mode_freeze()
            self._recompute("manual /sensor_gate/reset_mode_freeze")

    # ── Pass-through (drop live updates while frozen so the cache holds the
    #    snapshot from the moment freezing began) ──
    def _pose_cb(self, msg):
        if self.frozen:
            return
        self.last_pose = msg
        self.pub_pose.publish(msg)

    def _depth_cb(self, msg):
        if self.frozen:
            return
        self.last_depth = msg
        self.pub_depth.publish(msg)

    def _caminfo_cb(self, msg):
        if self.frozen:
            return
        self.last_caminfo = msg
        self.pub_caminfo.publish(msg)

    # ── Replay during freeze (refresh stamps so downstream age checks pass) ──
    def _replay(self, _evt):
        if not self.frozen:
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
            "sensor_gate hb: frozen=%s src=%s  [explicit=%s mode_turning=%s]  "
            "last_mode=%r last_explicit=%s  msgs: mode=%d explicit=%d  age=%.1fs",
            self.frozen, self.freeze_source,
            self.policy.explicit_freeze, self.policy.mode_says_freeze,
            self._last_mode_str, self._last_explicit_str,
            self.policy.n_mode_msgs, self._n_explicit_msgs, age)

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
        if self.policy.freeze_on_turning_mode:
            L("  freeze logic = MODE-AUTHORITATIVE  frozen <- (%s == %r)",
              self.demo_mode_topic, self.turning_mode_name)
            L("                 /sensor_gate/freeze is fallback before first mode msg")
        else:
            L("  freeze logic = EXPLICIT-ONLY  frozen <- /sensor_gate/freeze")
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
# ROSPARAMS (all private ~; defaults in parentheses). The mode-authoritative
# freeze decision lives in core.planning.sensor_freeze_policy; this node owns the
# ROS I/O, the replay timer and the heartbeat.
#
#   io: ~in_ns (/simple_drone) ~out_ns (/gated_drone) ~replay_hz (30.0)
#       ~pose_topic (in_ns/gt_pose) ~depth_topic (in_ns/front_depth/depth/image_raw)
#       ~camera_info_topic (in_ns/front_depth/depth/camera_info)
#       ~bridge_camera_info (false)
#   freeze: ~freeze_on_turning_mode (true) ~demo_mode_topic (/xtend/demo_mode)
#       ~turning_mode_name (turning) ~heartbeat_period_sec (2.0; 0 disables)
# ============================================================================
