#!/usr/bin/env python3
"""
mapping_sync_node.py -- timestamp-exact depth <-> localization pairing + a hard
localization gate for FALCON's voxel mapping.

Thin ROS1 glue around the ROS-free matcher in
``sparx_agency.core.localization.temporal_transform_buffer``. Every depth frame
FALCON fuses is paired with the pose carrying the SAME header.stamp (the capture
time); a depth frame with no co-temporal pose is DROPPED ("no location => no
voxels"). Emitted pose and depth share an identical stamp, so FALCON's
Transformer resolves an exact match.

Ordering-agnostic: both streams are buffered and a pair is emitted when its
second half arrives (pose-leads OR pose-trails); a held depth with no pose ages
out after ~match_hold_sec and is dropped.

Multi-source: ~pose_stamped_topics may list several topics (comma string or YAML
list); order is PRIORITY (first with a co-temporal pose wins). All sources MUST
share one world frame -- the node measures cross-source position disagreement on
co-temporal frames and warns ("sources disagree ...") if not.

Rotation freeze: because this node forms the (depth, pose) PAIR that FALCON
fuses, it is the authoritative place to freeze the voxel map while the platform
rotates in place -- where depth and localization are unreliable -- and to reject
the stale in-flight turn frame on resume. The pure decision is the ROS-free
``core.mapping.depth_fusion_gate.DepthFusionGate`` (driven by ~demo_mode_topic);
this node applies it in the CAPTURE clock (depth/pose stamps), so while turning
NO pair is emitted (no smear), and on resume only a frame captured strictly
after the turn ends is paired. Freezing here -- not only at the upstream sensor
gate -- is what makes the freeze hold: the gate's depth is gated but its pose
source (e.g. /xtend/april_tag_pose) is not, so a held depth would otherwise pair
with the live rotating pose.

  in   ~pose_stamped_topics (PoseStamped, 1+ topics)  default /flow_depth/pose_est
  in   ~depth_topic         (Image, RAW capture stamp) default /xtend/depth_m
  in   ~demo_mode_topic     (String, system mode)      default /xtend/demo_mode
  out  ~out_pose_topic      (PoseStamped, camera-in-world) default /map_ros/pose
  out  ~out_depth_topic     (Image, forwarded unchanged)   default /map_ros/depth

See the file footer for the full rosparam list.
"""
import threading
from bisect import bisect_left

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sparx_agency.core.common.math import se3
from sparx_agency.core.localization.temporal_transform_buffer import (
    MultiSourceTemporalMatcher, EXACT, NEAREST)
from sparx_agency.core.mapping.depth_fusion_gate import DepthFusionGate
from sparx_agency.core.mapping.sensor_freeze_policy import SensorFreezePolicy


def _parse_topics(raw, fallback):
    """One topic (string), or several (comma string / YAML list). Order = priority."""
    if raw is None:
        raw = fallback
    items = ([str(s).strip() for s in raw] if isinstance(raw, (list, tuple))
             else [s.strip() for s in str(raw).split(",")])
    items = [s for s in items if s]
    return items or [fallback]


class _PendingDepth:
    __slots__ = ("stamp", "msg", "wall")

    def __init__(self, stamp, msg, wall):
        self.stamp = stamp      # capture time (s) from the depth header
        self.msg = msg          # original Image, forwarded unchanged
        self.wall = wall        # rospy.Time received (for the hold)


class MappingSyncNode:
    def __init__(self):
        rospy.init_node("mapping_sync")
        G = rospy.get_param

        self.topics = _parse_topics(G("~pose_stamped_topics", None),
                                    G("~pose_stamped_topic", "/flow_depth/pose_est"))
        self.nsrc = len(self.topics)
        self.depth_topic = G("~depth_topic", "/xtend/depth_m")
        self.out_pose_topic = G("~out_pose_topic", "/map_ros/pose")
        self.out_depth_topic = G("~out_depth_topic", "/map_ros/depth")
        self.world_frame = G("~world_frame", "world")

        self.sync_tol = float(G("~sync_tolerance", 0.05))
        self.max_interp_gap = float(G("~max_interp_gap", 0.12))
        self.match_hold_sec = float(G("~match_hold_sec", 0.5))
        self.buffer_sec = float(G("~pose_buffer_sec", 5.0))
        self.depth_min_dt = float(G("~depth_min_dt", 0.0))
        self.clock_warn_sec = float(G("~clock_warn_sec", 0.5))
        self.disagree_warn_m = float(G("~disagree_warn_m", 0.30))

        # Rotation freeze (authoritative voxel-pair gate). Decision is the
        # ROS-free core.mapping.depth_fusion_gate; applied here in the CAPTURE
        # clock. turning_mode_name is the demo-mode string that means "freeze".
        self.demo_mode_topic = G("~demo_mode_topic", "/xtend/demo_mode")
        self.turning_mode_name = str(G("~turning_mode_name", "turning")).strip().lower()
        self.gate = DepthFusionGate(
            policy=SensorFreezePolicy(
                freeze_on_turning_mode=bool(G("~freeze_on_turning_mode", True))),
            resume_settle_sec=float(G("~resume_settle_sec", 0.0)))
        self._newest_pose_stamp = None      # capture-clock "now" for the watermark
        self._last_mode_str = "(no msgs yet)"

        # Frame: incoming pose is body(FLU) and right-multiplied by T_b_c
        # (+cam offset) unless it is already the camera-in-world pose.
        self.pose_is_camera_frame = bool(G("~pose_is_camera_frame", False))
        cam = (float(G("~cam_offset_x", 0.2)), float(G("~cam_offset_y", 0.0)),
               float(G("~cam_offset_z", 0.0)))
        self.T_b_c = np.array([[0.0, 0.0, 1.0, cam[0]],
                               [-1.0, 0.0, 0.0, cam[1]],
                               [0.0, -1.0, 0.0, cam[2]],
                               [0.0, 0.0, 0.0, 1.0]])

        self._lock = threading.Lock()
        self._matcher = MultiSourceTemporalMatcher(self.nsrc, self.buffer_sec)
        self._pending = []                  # _PendingDepth, stamp-sorted
        self._prev_intake_t = None
        self._last_pose_wall = None

        # diagnostics
        self._n_pose = [0] * self.nsrc
        self._n_emit_src = [0] * self.nsrc
        self._n_depth = self._n_exact = self._n_nearest = self._n_interp = 0
        self._n_drop_throttle = self._n_drop_empty = self._n_drop_far = 0
        self._n_drop_frozen = self._n_drop_stale = 0
        self._n_disagree = 0
        self._last_far = self._last_disagree = 0.0

        self.pub_pose = rospy.Publisher(self.out_pose_topic, PoseStamped, queue_size=10)
        self.pub_depth = rospy.Publisher(self.out_depth_topic, Image, queue_size=8)
        for idx, tp in enumerate(self.topics):
            rospy.Subscriber(tp, PoseStamped, self._pose_cb,
                             callback_args=idx, queue_size=200)
        rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=16)
        if self.gate.policy.freeze_on_turning_mode:
            rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)
        rospy.Timer(rospy.Duration(0.05), self._sweep_timer)
        rospy.Timer(rospy.Duration(2.0), self._heartbeat)
        self._banner(cam)

    # -- localization in (one callback per source via callback_args) ----------
    def _pose_cb(self, msg, src):
        t = msg.header.stamp.to_sec()
        if t <= 0.0:
            rospy.logwarn_throttle(5.0, "mapping_sync: pose on %s has stamp %.6f "
                                   "<= 0 -- is it the capture time?", self.topics[src], t)
        p, o = msg.pose.position, msg.pose.orientation
        T = se3.make_transform((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        with self._lock:
            self._n_pose[src] += 1
            self._last_pose_wall = rospy.Time.now()
            # Newest pose stamp is the capture-clock "now" the resume watermark
            # is armed from (poses keep flowing through a turn, so it tracks the
            # true turn-end even if the mode signal overtakes late depth frames).
            if self._newest_pose_stamp is None or t > self._newest_pose_stamp:
                self._newest_pose_stamp = t
            self._matcher.insert(src, t, T)
            if self.nsrc > 1:
                d = self._matcher.disagreement(src, t, T, self.sync_tol)
                if d is not None and d > self.disagree_warn_m:
                    self._n_disagree += 1
                    self._last_disagree = d
            emits = self._match_pending_locked()
        self._emit(emits)

    # -- demo mode in (drives the rotation freeze) ----------------------------
    def _demo_mode_cb(self, msg):
        mode = (msg.data or "").strip().lower()
        with self._lock:
            self._last_mode_str = mode if mode else "(empty)"
            # Arm/clear the freeze in the capture clock: at the turn->resume edge
            # the watermark becomes the newest pose stamp, so every frame
            # captured during the turn (stamp <= now) is rejected on resume.
            self.gate.note_mode(mode == self.turning_mode_name,
                                now=self._newest_pose_stamp)

    # -- depth in -------------------------------------------------------------
    def _depth_cb(self, msg):
        td = msg.header.stamp.to_sec()
        self._n_depth += 1
        # Rotation freeze (authoritative for the voxel pair). While turning, emit
        # NOTHING (no smear). On resume, drop the stale in-flight turn frame until
        # one captured strictly after the turn ends arrives. Skip should_fuse while
        # frozen so the watermark is not polluted by upstream replay stamps.
        with self._lock:
            if self.gate.frozen:
                self._n_drop_frozen += 1
                return
            fuse, _ = self.gate.should_fuse(td)
            if not fuse:
                self._n_drop_stale += 1
                return
        if self.depth_min_dt > 0.0 and self._prev_intake_t is not None:
            if (td - self._prev_intake_t) < self.depth_min_dt:
                self._n_drop_throttle += 1
                return
        self._prev_intake_t = td

        emits = []
        with self._lock:
            T, kind, src = self._matcher.lookup(td, self.sync_tol, self.max_interp_gap)
            if T is not None:
                emits.append(self._build_emit_locked(msg, T, kind, src))
            else:
                p = _PendingDepth(td, msg, rospy.Time.now())
                i = bisect_left([q.stamp for q in self._pending], td)
                self._pending.insert(i, p)
            self._sweep_locked(rospy.Time.now())
        self._emit(emits)

    # -- matching (lock held) -------------------------------------------------
    def _match_pending_locked(self):
        if not self._pending:
            return []
        emits, keep = [], []
        for p in self._pending:
            T, kind, src = self._matcher.lookup(p.stamp, self.sync_tol, self.max_interp_gap)
            if T is not None:
                emits.append(self._build_emit_locked(p.msg, T, kind, src))
            else:
                keep.append(p)
        self._pending = keep
        return emits

    def _sweep_locked(self, now_wall):
        if not self._pending:
            return
        keep = []
        for p in self._pending:
            if (now_wall - p.wall).to_sec() > self.match_hold_sec:
                self._record_drop_locked(p.stamp)      # gate: no location => dropped
            else:
                keep.append(p)
        self._pending = keep

    def _record_drop_locked(self, td):
        dt = self._matcher.nearest_dt(td)              # classify the drop
        if dt is not None and dt > self.clock_warn_sec:
            self._n_drop_far += 1                       # poses exist but FAR -> clocks
            self._last_far = dt
        else:
            self._n_drop_empty += 1                     # nothing near -> real dropout

    def _build_emit_locked(self, depth_msg, T_body, kind, src):
        if kind == EXACT:
            self._n_exact += 1
        elif kind == NEAREST:
            self._n_nearest += 1
        else:
            self._n_interp += 1
        if 0 <= src < self.nsrc:
            self._n_emit_src[src] += 1
        T_w_c = T_body if self.pose_is_camera_frame else (T_body @ self.T_b_c)
        ps = PoseStamped()
        ps.header.stamp = depth_msg.header.stamp        # IDENTICAL to the depth stamp
        ps.header.frame_id = self.world_frame
        ps.pose.position.x = float(T_w_c[0, 3])
        ps.pose.position.y = float(T_w_c[1, 3])
        ps.pose.position.z = float(T_w_c[2, 3])
        qx, qy, qz, qw = se3.quaternion_from_matrix(T_w_c)
        ps.pose.orientation.x, ps.pose.orientation.y = float(qx), float(qy)
        ps.pose.orientation.z, ps.pose.orientation.w = float(qz), float(qw)
        return (ps, depth_msg)

    # -- emit (lock NOT held) -------------------------------------------------
    def _emit(self, emits):
        for pose_msg, depth_msg in emits:
            self.pub_pose.publish(pose_msg)             # pose first
            self.pub_depth.publish(depth_msg)

    # -- timers ---------------------------------------------------------------
    def _sweep_timer(self, _evt):
        with self._lock:
            self._sweep_locked(rospy.Time.now())

    def _heartbeat(self, _evt):
        with self._lock:
            bufs = self._matcher.sizes()
            pend = len(self._pending)
            n_pose = list(self._n_pose)
            emit_src = list(self._n_emit_src)
            n_depth = self._n_depth
            ex, ne, it = self._n_exact, self._n_nearest, self._n_interp
            d_empty, d_far, d_thr = self._n_drop_empty, self._n_drop_far, self._n_drop_throttle
            d_frozen, d_stale = self._n_drop_frozen, self._n_drop_stale
            ndis, last_dis = self._n_disagree, self._last_disagree
            if self.gate.frozen:
                gate_state = "FROZEN"
            elif self.gate.awaiting_fresh_frame:
                gate_state = "RESUME_WAIT"
            else:
                gate_state = "FUSING"
        emit = ex + ne + it
        pose_age = ((rospy.Time.now() - self._last_pose_wall).to_sec()
                    if self._last_pose_wall else -1.0)
        src_str = "  ".join("[%d]%s pose=%d buf=%d emit=%d"
                            % (i, self.topics[i].split("/")[-1], n_pose[i], bufs[i], emit_src[i])
                            for i in range(self.nsrc))
        rospy.loginfo("mapping_sync hb | %s | depth=%d -> emit=%d [exact=%d near=%d "
                      "interp=%d] held=%d drop[dropout=%d clockmismatch=%d throttle=%d "
                      "frozen=%d stale=%d] | gate=%s | last_pose=%.1fs ago",
                      src_str, n_depth, emit, ex, ne, it,
                      pend, d_empty, d_far, d_thr, d_frozen, d_stale, gate_state, pose_age)
        if self.nsrc > 1 and ndis > 0:
            rospy.logwarn_throttle(5.0, "mapping_sync: %d co-temporal frames where "
                                   "sources disagree by up to %.2fm -- they may be in "
                                   "DIFFERENT world frames. If so, use ONE source.",
                                   ndis, last_dis)
        if n_depth > 10 and emit == 0:
            if d_far > 0:
                rospy.logwarn_throttle(5.0, "mapping_sync: NOT pairing; nearest pose to "
                                       "dropped depth ~%.2fs away -- depth and pose look "
                                       "like DIFFERENT clocks. Feed ~depth_topic the RAW "
                                       "depth the localization used.", self._last_far)
            else:
                rospy.logwarn_throttle(5.0, "mapping_sync: depth flowing but ZERO paired "
                                       "and no poses near them -- is any localization "
                                       "publishing a PoseStamped on %s?", self.topics)

    def _banner(self, cam):
        L = rospy.loginfo
        L("=" * 72)
        L("mapping_sync -- multi-source pairing + localization gate")
        for i, tp in enumerate(self.topics):
            L("  pose  in[%d] = %s   (priority %d%s)", i, tp, i,
              " = HIGHEST" if i == 0 else "")
        if self.nsrc > 1:
            L("  >>> %d sources; first co-temporal wins; ALL must share one world "
              "frame (disagree_warn=%.2fm)", self.nsrc, self.disagree_warn_m)
        L("  depth in  = %s   (stamp == capture time; use the RAW depth)", self.depth_topic)
        if self.gate.policy.freeze_on_turning_mode:
            L("  rotation freeze = ON  (drop pairs while %s == %r; resume after fresh "
              "frame, settle=%.2fs)", self.demo_mode_topic, self.turning_mode_name,
              self.gate.resume_settle_sec)
        else:
            L("  rotation freeze = OFF (freeze_on_turning_mode=false)")
        L("  pose  out = %s   depth out = %s", self.out_pose_topic, self.out_depth_topic)
        L("  sync_tol=%.3fs interp_gap=%.3fs match_hold=%.3fs buffer=%.1fs",
          self.sync_tol, self.max_interp_gap, self.match_hold_sec, self.buffer_sec)
        L("  pose_is_camera_frame=%s cam_offset=(%.3f,%.3f,%.3f)",
          self.pose_is_camera_frame, *cam)
        L("=" * 72)


def main():
    try:
        MappingSyncNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The timestamp-matching
# maths live in core.localization.temporal_transform_buffer and the rotation
# freeze in core.mapping.depth_fusion_gate; this node maps rosparams -> matcher
# + gate and owns the ROS I/O, threading, the late-pose gate and the freeze.
#
#   sources: ~pose_stamped_topics (/flow_depth/pose_est)  comma string or list,
#            order = priority; ~pose_stamped_topic (singular) still honored
#   io: ~depth_topic (/xtend/depth_m)  ~out_pose_topic (/map_ros/pose)
#       ~out_depth_topic (/map_ros/depth)  ~world_frame (world)
#   timing: ~sync_tolerance (0.05) ~max_interp_gap (0.12, 0=off)
#       ~match_hold_sec (0.5) ~pose_buffer_sec (5.0) ~depth_min_dt (0.0, off)
#   freeze: ~demo_mode_topic (/xtend/demo_mode) ~turning_mode_name (turning)
#       ~freeze_on_turning_mode (true) ~resume_settle_sec (0.0)
#   diagnostics: ~clock_warn_sec (0.5) ~disagree_warn_m (0.30)
#   frame (mirror falcon_adapter): ~pose_is_camera_frame (false)
#       ~cam_offset_x/y/z (0.2/0.0/0.0)
# ============================================================================
