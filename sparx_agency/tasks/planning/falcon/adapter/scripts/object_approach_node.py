#!/usr/bin/env python3
"""object_approach_node.py -- ROS1 adapter: lock onto a named object and fly to it.

The "hover / visual approach" mission on top of the FALCON nav stack. While the
target is not yet confirmed the node stays PASSIVE and the existing A*/NavDP
follower flies the coordinate route; the detector (yolo_detector_node) scans in
parallel. Once the target is confirmed for N consecutive detector frames, the node
takes over ``/cmd_vel`` -- via the ``visual_servoing`` demo-mode hand-off, which
makes the follower go passive so there is exactly one publisher -- and visually
servos onto the object at camera rate until it is centred and very close (a stable
hover-lock directly in front of it). If the track is lost it actively re-searches
in the direction the target left; if it cannot re-acquire it hands control back and
returns to SEARCH.

All the maths is ROS-free and unit-tested:
  * detect-once/track-many bbox tracking   core.planning.visual_tracking.TargetTracker
  * bbox (+depth range) -> body velocity    core.planning.visual_servo.VisualServoController
  * N-consecutive-frame acquisition         core.planning.visual_servo.TargetConfirmationGate
  * where to look when lost                  core.planning.visual_servo.ReSearchPolicy
  * SEARCH/APPROACH/HOVER_LOCK/RECOVER       core.planning.visual_servo.VisualApproachStateMachine
  * bbox + depth -> metric range            core.mapping.depth.bbox_to_xyz_cam_from_depth
This node owns ONLY ROS concerns: sensor I/O, the demo-mode hand-off, /cmd_vel,
and feeding the pure state machine. NO localization is used for the approach.

Inputs  (mirrors navdp_click / combination transports):
  ~rgb_topic     frame-path String or raw Image   (tracked every frame)
  ~depth_topic   frame-path String or raw Image   (optional; metric range)
  ~detections_topic  std_msgs/String JSON         (from yolo_detector_node)
  ~target_topic  std_msgs/String                  (the mission "goal", e.g. "hat")
  ~enable_topic  std_msgs/Bool                     (mode switch; ~start_enabled)
  ~demo_mode_topic  std_msgs/String                (to know we hold visual_servoing)
Outputs:
  <drone_ns>/cmd_vel  geometry_msgs/Twist          (holonomic vx, vy, wz)
  ~demo_mode_request_topic  std_msgs/String        (visual_servoing <-> fly_straight)
  ~status_topic  std_msgs/String                    (diagnostics, optional)
See the file footer for the full rosparam list.
"""
import json
import threading
from collections import deque

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.types import Intrinsics, KinematicLimits
from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.depth.depth_bbox_fusion import bbox_to_xyz_cam_from_depth
from sparx_agency.core.planning.visual_tracking import TargetTracker, TargetTrackerConfig
from sparx_agency.core.planning.visual_servo import (
    VisualServoController, VisualServoParams, VisualServoRequest,
    TargetConfirmationGate, ConfirmationGateConfig,
    ReSearchPolicy, ReSearchConfig,
    VisualApproachStateMachine, ApproachFSMConfig, SEARCH,
)

MODE_VISUAL_SERVOING = "visual_servoing"   # follower goes passive on this
MODE_RELEASE = "fly_straight"              # hand control back to the follower


def _param_bool(name, default):
    """Read a boolean rosparam, failing loud on a non-boolean string (bool-param trap)."""
    v = rospy.get_param(name, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError("%s must be a boolean (true/false), got %r" % (name, v))


class ObjectApproachNode(object):
    def __init__(self):
        rospy.init_node("object_approach")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "")
        self.image_transport = str(G("~image_transport", "frame_path")).strip().lower()
        if self.image_transport not in ("frame_path", "topic"):
            raise ValueError("~image_transport must be 'frame_path' or 'topic', "
                             "got %r" % self.image_transport)
        _fp = self.image_transport == "frame_path"
        self.rgb_topic = G("~rgb_topic",
                           "/xtend/rgb_frame_path" if _fp else "/xtend/rgb")
        self.depth_topic = G("~depth_topic",
                             "/xtend/depth_frame_path" if _fp else "/xtend/depth_m")
        self.pose_topic = G("~pose_topic", "/xtend/localization")   # unused by servo; diag only
        self.pose_type = str(G("~pose_type", "pose_stamped")).strip().lower()
        self.camera_info_topic = G("~camera_info_topic", "")

        self.detections_topic = G("~detections_topic", "/object_approach/detections")
        self.target_topic = G("~target_topic", "/object_approach/goal")
        self.enable_topic = G("~enable_topic", "/object_approach/enable")
        self.status_topic = G("~status_topic", "/object_approach/status")
        self.demo_mode_topic = G("~demo_mode_topic", "/xtend/demo_mode")
        self.demo_mode_request_topic = G("~demo_mode_request_topic",
                                         "/xtend/demo_mode_request")

        # Intrinsics matching the live RGB/depth stream (see navdp_click for the
        # K-vs-P note). Depth range uses these; the servo only needs width/height.
        self.intr = Intrinsics(
            width=int(G("~img_width", 504)), height=int(G("~img_height", 294)),
            fx=float(G("~fx", 322.6351083474948)), fy=float(G("~fy", 323.3893307141174)),
            cx=float(G("~cx", 242.06479658679714)), cy=float(G("~cy", 90.03019076680604)))

        self.ctrl_hz = float(G("~ctrl_hz", 15.0))
        self.reseed_on_detection = _param_bool("~reseed_on_detection", True)
        self.frame_buffer_len = int(G("~frame_buffer_len", 30))
        self.enabled = _param_bool("~start_enabled", True)
        self.target = str(G("~target_object", "refrigerator")).strip().lower()

        # ── Core objects (ROS-free) ──────────────────────────────────
        self.limits = KinematicLimits(
            max_speed_xy=float(G("~max_speed_xy", 0.4)),
            max_speed_z=float(G("~max_speed_z", 0.3)),
            max_yaw_rate=float(G("~max_yaw_rate", 0.6)))
        self.tracker = TargetTracker(TargetTrackerConfig(
            input_is_bgr=True,
            max_predict_s=float(G("~max_predict_s", 0.4))))
        self.servo = VisualServoController(VisualServoParams(
            mode=str(G("~servo_mode", "holonomic")).strip().lower(),
            kp_yaw=float(G("~kp_yaw", 1.2)),
            max_yaw_rate=float(G("~max_yaw_rate", 0.6)),
            use_lateral=_param_bool("~use_lateral", True),
            use_vertical=_param_bool("~use_vertical", False),
            vx_max=float(G("~vx_max", 0.35)),
            use_depth=_param_bool("~use_depth", True),
            target_range_m=float(G("~target_range_m", 0.8)),
            slowdown_range_m=float(G("~slowdown_range_m", 2.0)),
            target_area_frac=float(G("~target_area_frac", 0.12)),
            center_tol=float(G("~center_tol", 0.15))),
            default_limits=self.limits)
        self.gate = TargetConfirmationGate(self.target, ConfirmationGateConfig(
            n_confirm=int(G("~n_confirm", 3)),
            min_score=float(G("~min_score", 0.30))))
        # The FSM's recover_timeout_s is the single source of truth for how long
        # we re-search before handing control back; the recovery policy's own
        # give-up mirrors it (the node acts on the FSM, not on rec.give_up).
        recover_timeout_s = float(G("~recover_timeout_s", 6.0))
        self.recovery = ReSearchPolicy(ReSearchConfig(
            search_yaw_rate=float(G("~search_yaw_rate", 0.5)),
            max_search_s=recover_timeout_s))
        self.fsm = VisualApproachStateMachine(ApproachFSMConfig(
            recover_timeout_s=recover_timeout_s))

        # ── Shared state ──────────────────────────────────────────────
        # _lock guards tracker/gate/frame-buffer; _mode_lock guards the
        # enabled + demo-mode-request handshake (mutated by the enable callback
        # thread and the control-timer thread).
        self._lock = threading.Lock()
        self._mode_lock = threading.Lock()
        self.rgb = None
        self.rgb_stamp = 0.0
        self.depth = None
        self._frame_buf = deque(maxlen=max(2, self.frame_buffer_len))  # (stamp, bgr)
        self._confirmed = False
        self.current_demo_mode = None
        self._requested_mode = None
        self._last_dec = None
        self._last_res = None
        self._prev_tick_t = None

        # ── ROS I/O (publishers before subscribers) ──────────────────
        self.cmd_pub = rospy.Publisher(self.drone_ns + "/cmd_vel", Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1)

        if _fp:
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, String, self._depth_path_cb, queue_size=2)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=2)
        if self.camera_info_topic:
            rospy.Subscriber(self.camera_info_topic, CameraInfo, self._cam_info_cb,
                             queue_size=1)
        rospy.Subscriber(self.detections_topic, String, self._det_cb, queue_size=5)
        rospy.Subscriber(self.target_topic, String, self._target_cb, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self._enable_cb, queue_size=1)
        rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)

        self._banner()

    # ─── Sensor callbacks ────────────────────────────────────────────
    def _rgb_path_cb(self, msg):
        try:
            parsed = parse_frame_path_message(msg.data)
            bgr = cv2.imread(parsed.path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None for %s" % parsed.path)
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "object_approach: dropping RGB frame-path (%s)", e)
            return
        self._push_rgb(bgr, parsed.stamp_seconds)

    def _depth_path_cb(self, msg):
        try:
            parsed = parse_frame_path_message(msg.data)
            arr = np.squeeze(np.load(parsed.path))
            if arr.ndim != 2:
                raise ValueError("depth has shape %r; expected HxW" % (arr.shape,))
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "object_approach: dropping depth frame-path (%s)", e)
            return
        self.depth = np.ascontiguousarray(arr, dtype=np.float32)

    def _rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        bgr = arr if msg.encoding == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self._push_rgb(bgr.copy(), msg.header.stamp.to_sec())

    def _depth_cb(self, msg):
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(msg.data, np.float32).reshape(
                msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            self.depth = (np.frombuffer(msg.data, np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) / 1000.0)
        else:
            rospy.logwarn_throttle(5.0, "object_approach: unsupported depth encoding %r",
                                   msg.encoding)

    def _push_rgb(self, bgr, stamp):
        with self._lock:
            self.rgb = bgr
            self.rgb_stamp = float(stamp)
            self._frame_buf.append((float(stamp), bgr))

    def _cam_info_cb(self, msg):
        if any(msg.K):
            fx, fy, cx, cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
        elif any(msg.P):
            fx, fy, cx, cy = msg.P[0], msg.P[5], msg.P[2], msg.P[6]
        else:
            return
        self.intr = Intrinsics(width=int(msg.width), height=int(msg.height),
                               fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

    def _target_cb(self, msg):
        target = str(msg.data).strip().lower()
        if not target or target == self.target:
            return
        rospy.loginfo("object_approach: goal %r -> %r (re-acquiring)", self.target, target)
        with self._lock:
            self.target = target
            self.gate.set_target(target)
            self.tracker.reset()
            self.fsm.reset()

    def _enable_cb(self, msg):
        want = bool(msg.data)
        with self._mode_lock:
            if want == self.enabled:
                return
            self.enabled = want
        rospy.loginfo("object_approach: %s", "ENABLED" if want else "DISABLED")
        if not want:
            self._release()

    def _demo_mode_cb(self, msg):
        self.current_demo_mode = str(msg.data).strip().lower()

    # ─── Detections: confirm + (re)seed the tracker ──────────────────
    def _det_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            dets = [Detection2D(label=str(d["label"]).strip().lower(),
                                score=float(d["score"]),
                                bbox_xyxy=tuple(int(v) for v in d["bbox"]),
                                frame_w=int(payload.get("w", self.intr.width)),
                                frame_h=int(payload.get("h", self.intr.height)))
                    for d in payload.get("detections", [])]
            stamp = float(payload.get("stamp", self.rgb_stamp))
        except (ValueError, KeyError, TypeError) as e:
            rospy.logwarn_throttle(5.0, "object_approach: bad detections msg (%s)", e)
            return

        with self._lock:
            state = self.gate.update(dets)
            self._confirmed = state.confirmed
            if state.best is None:
                return
            # Seed on acquisition; re-seed later to bound drift / regain a lost lock.
            need_seed = (not self.tracker.has_target and state.confirmed) or \
                        (self.tracker.has_target and self.reseed_on_detection)
            if need_seed:
                frame = self._closest_frame(stamp)
                if frame is not None:
                    self.tracker.on_detection(frame, state.best, stamp)

    def _closest_frame(self, stamp):
        if not self._frame_buf:
            return None
        return min(self._frame_buf, key=lambda kv: abs(kv[0] - stamp))[1]

    # ─── Control loop ────────────────────────────────────────────────
    def start(self):
        rospy.Timer(rospy.Duration(1.0 / max(self.ctrl_hz, 1.0)), self._tick)
        rospy.Timer(rospy.Duration(2.0), self._hb)
        rospy.spin()

    def _tick(self, _evt):
        # A bad frame must never kill the timer thread (rospy quirk); catch, hold.
        try:
            self._step()
        except Exception as e:                    # noqa: BLE001 -- resilience is the point
            rospy.logwarn_throttle(2.0, "object_approach: tick error (%s: %s) -- holding",
                                   type(e).__name__, e)
            if self._driving():
                self._publish_cmd(0.0, 0.0, 0.0)

    def _step(self):
        now = rospy.Time.now().to_sec()
        dt = 0.0 if self._prev_tick_t is None else max(0.0, now - self._prev_tick_t)
        self._prev_tick_t = now

        with self._mode_lock:
            enabled = self.enabled
        if not enabled:
            # Release EVERY disabled tick (idempotent): closes the race where a
            # disable that lands mid-drive-tick could otherwise leave the follower
            # latched-passive with no active /cmd_vel publisher.
            self._release()
            return

        with self._lock:
            rgb, stamp, depth = self.rgb, self.rgb_stamp, self.depth
            has_target = self.tracker.has_target
            confirmed = self._confirmed
            track = self.tracker.on_frame(rgb, stamp) if (has_target and rgb is not None) else None
            last_track = self.tracker.last_track   # snapshot for recovery under the lock

        track_valid = bool(track is not None and track.valid)

        # Servo result (needs a valid track); also yields at_target for the FSM.
        res = None
        if track_valid:
            rng = self._range_to(track, depth)
            res = self.servo.step(VisualServoRequest(
                track=track, intrinsics=self.intr, range_m=rng, dt=dt))
        at_target = bool(res is not None and res.at_target)

        dec = self.fsm.update(confirmed=confirmed, track_valid=track_valid,
                              at_target=at_target, dt=dt)
        self._last_dec, self._last_res = dec, res

        if dec.reset_acquisition:
            with self._lock:
                self.gate.reset()
                self.tracker.reset()
                self._confirmed = False

        if not dec.drive_cmd_vel:                 # SEARCH: hand /cmd_vel back
            self._release()
            return

        # We own /cmd_vel: make the follower passive first. Do NOT publish until
        # the hand-off is confirmed (demo_mode == visual_servoing), else we and the
        # still-active follower would both drive /cmd_vel for the round-trip.
        self._request_mode(MODE_VISUAL_SERVOING)
        if self.current_demo_mode != MODE_VISUAL_SERVOING:
            return
        if dec.mode == "RECOVER" or res is None:
            self._drive_recovery(last_track, dec.lost_for_s)
        else:
            c = res.command
            self._publish_cmd(c.x, c.y, c.yaw_rate)

    def _drive_recovery(self, last_track, lost_for_s):
        rec = self.recovery.command(last_track, lost_for_s,
                                    self.intr.width, self.intr.height)
        c = rec.command
        self._publish_cmd(c.x, c.y, c.yaw_rate)

    def _range_to(self, track, depth):
        """Metric range (m) to the tracked box from depth, or None."""
        if depth is None:
            return None
        if depth.shape[:2] != (self.intr.height, self.intr.width):
            # Depth not aligned to the RGB/intrinsics geometry we index it with;
            # fall back to the area proxy rather than sample the wrong pixels.
            rospy.logwarn_throttle(
                10.0, "object_approach: depth %r != intrinsics %dx%d; using area proxy",
                depth.shape[:2], self.intr.height, self.intr.width)
            return None
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        xyz = bbox_to_xyz_cam_from_depth(depth, (x1, y1, x2, y2),
                                         self.intr.fx, self.intr.fy,
                                         self.intr.cx, self.intr.cy)
        return None if xyz is None else float(xyz[2])

    # ─── Demo-mode hand-off + cmd_vel ────────────────────────────────
    def _driving(self):
        return self._requested_mode == MODE_VISUAL_SERVOING

    def _request_mode(self, mode):
        with self._mode_lock:
            self._request_mode_locked(mode)

    def _request_mode_locked(self, mode):
        """Publish a demo-mode request on change. Caller must hold _mode_lock."""
        if self._requested_mode == mode:
            return
        self._requested_mode = mode
        self.demo_req_pub.publish(String(data=mode))
        rospy.loginfo("object_approach: request demo_mode=%s (current=%s)",
                      mode, self.current_demo_mode)

    def _release(self):
        """Hand /cmd_vel back to the follower (once).

        Only acts if we actually held control: if we never requested
        visual_servoing we leave demo_mode entirely to the nav stack's follower,
        so an idle/searching object_approach never fights the follower's own mode
        management. Releasing is mandatory once we DID take over -- the follower
        stays passive as long as the latched demo_mode reads visual_servoing, so we
        must actively request a non-servoing mode to hand control back.
        """
        with self._mode_lock:
            if self._requested_mode != MODE_VISUAL_SERVOING:
                return
            self._publish_cmd(0.0, 0.0, 0.0)          # one brake before releasing
            self._request_mode_locked(MODE_RELEASE)

    def _publish_cmd(self, vx, vy, wz):
        m = Twist()
        m.linear.x = float(vx)
        m.linear.y = float(vy)          # holonomic crab (XTEND accepts it)
        m.linear.z = 0.0                # fixed altitude (platform holds it)
        m.angular.z = float(wz)
        self.cmd_pub.publish(m)

    # ─── Heartbeat ───────────────────────────────────────────────────
    def _hb(self, _evt):
        dec, res = self._last_dec, self._last_res
        mode = dec.mode if dec is not None else "?"
        bits = ["state=%s" % mode, "enabled=%s" % self.enabled,
                "target=%r" % self.target, "confirmed=%s" % self._confirmed,
                "demo=%s" % (self.current_demo_mode or "none")]
        if res is not None:
            bits.append("xoff=%+.2f area=%.3f rng=%s at=%s"
                        % (res.x_offset, res.area_frac,
                           "%.2f" % res.range_m if res.range_m is not None else "-",
                           res.at_target))
        line = "object_approach hb  " + "  ".join(bits)
        rospy.loginfo(line)
        self.status_pub.publish(String(data=line))

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("object_approach (lock onto a named object -> visual approach)")
        L("  rgb   in  = %s  (%s)", self.rgb_topic, self.image_transport)
        L("  depth in  = %s", self.depth_topic)
        L("  dets  in  = %s", self.detections_topic)
        L("  goal  in  = %s   (start target=%r)", self.target_topic, self.target)
        L("  enable    = %s   (start_enabled=%s)", self.enable_topic, self.enabled)
        L("  cmd_vel out = %s  (holonomic, via %s hand-off)",
          self.drone_ns + "/cmd_vel", MODE_VISUAL_SERVOING)
        L("  intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
          self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
          self.intr.width, self.intr.height)
        L("  success = target centred & within target_range/area (hover-lock, no land)")
        L("=" * 64)


def main():
    try:
        ObjectApproachNode().start()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). All servo/track/FSM maths
# is ROS-free in core.planning.visual_servo / visual_tracking / mapping.detection;
# this node owns ROS I/O, the demo-mode hand-off and the control-loop plumbing.
#
#   IO: ~image_transport (frame_path | topic)
#       ~rgb_topic (/xtend/rgb_frame_path) ~depth_topic (/xtend/depth_frame_path)
#       ~detections_topic (/object_approach/detections)  ~target_topic (/object_approach/goal)
#       ~enable_topic (/object_approach/enable)  ~status_topic (/object_approach/status)
#       ~pose_topic (/xtend/localization) ~pose_type (pose_stamped)  [diagnostic only]
#       ~drone_ns ('')  [-> <drone_ns>/cmd_vel]
#       ~demo_mode_topic (/xtend/demo_mode)  ~demo_mode_request_topic (/xtend/demo_mode_request)
#   camera (MUST match the live stream; K over P): ~fx ~fy ~cx ~cy ~img_width (504)
#       ~img_height (294)  ~camera_info_topic ('' = use params)
#   mission: ~target_object (refrigerator) ~start_enabled (true) ~ctrl_hz (15.0)
#   acquisition: ~n_confirm (3) ~min_score (0.30)
#   tracking: ~reseed_on_detection (true) ~frame_buffer_len (30) ~max_predict_s (0.4)
#   servo: ~servo_mode (holonomic | yaw_forward_xor) ~kp_yaw (1.2) ~vx_max (0.35)
#       ~use_lateral (true) ~use_vertical (false) ~center_tol (0.15)
#       ~use_depth (true) ~target_range_m (0.8) ~slowdown_range_m (2.0)
#       ~target_area_frac (0.12)   [used when depth absent]
#   limits: ~max_speed_xy (0.4) ~max_speed_z (0.3) ~max_yaw_rate (0.6)
#   recovery: ~search_yaw_rate (0.5) ~recover_timeout_s (6.0)  [governs re-search
#       duration; the recovery policy's give-up mirrors it]
# ============================================================================
