#!/usr/bin/env python3
"""object_approach_node.py -- ROS1 adapter: lock onto a named object and fly to it.

The "hover / visual approach" mission on top of the FALCON nav stack. While the
target is not yet confirmed the node stays PASSIVE and the existing A*/NavDP
follower flies the coordinate route; the detector (yolo_detector_ros2_node) scans in
parallel. Once the target is confirmed for N consecutive detector frames, the node
takes over ``/cmd_vel`` -- via the ``visual_servoing`` demo-mode hand-off, which
makes the follower go passive so there is exactly one publisher -- first holds a
brief stop-in-place (ACQUIRE_STOP: an actively-published zero-velocity stop for
``~acquire_stop_s`` that brakes off any inherited route motion so no stale A*/NavDP
command carries into closure), then visually servos onto the object at camera rate
until it is centred and very close (a stable hover-lock directly in front of it). There is NO terminal stop: HOVER_LOCK keeps
tracking, and if the object moves it re-enters APPROACH. If the track is lost it
actively re-searches in the direction the target left; if it cannot re-acquire it
hands control back, re-asserts the goal, and returns to SEARCH.

Beyond the base search->approach loop this node adds four mission behaviours:
  1. DISCRETE/INERTIAL FLIGHT-COMMAND SHAPING of every published command (the
     platform yaws/advances at a fixed speed, ignores a lone control tick, and
     coasts). A stateful PulseShaper latches any motion for >= min_burst_ticks and
     can brake off the coast; the servo uses a coarse, closeness-growing yaw
     deadband so fine centring is done by crab, not yaw. Runs at ~10 Hz (the route
     follower's calibration). ~closure_mode picks multi_axis (holonomic)/waypoint.
  2. GOAL RE-INJECT: on a lost-track give-up, re-publish the last/initial goal so
     the planner flies back to it instead of stalling.
  3. SCAN-AT-GOAL: once the route reaches its goal still unconfirmed, sweep the
     room (slow rotate with stops) looking for the object.
  4. LIVE HUD: publish the target-lock overlay (detections + tracked box + the
     exact shaped command) as an Image for target_lock_viewer_node.

All the maths is ROS-free and unit-tested:
  * detect-once/track-many bbox tracking   core.mapping.tracking.TargetTracker
  * bbox (+depth range) -> body velocity    core.planning.visual_servo.VisualServoController
  * N-consecutive-frame acquisition         core.planning.visual_servo.TargetConfirmationGate
  * where to look when lost                  core.planning.visual_servo.ReSearchPolicy
  * SEARCH/SCAN/ACQUIRE_STOP/APPROACH/HOVER_LOCK/RECOVER  core.planning.visual_servo.VisualApproachStateMachine
  * min-burst + coast flight-command shaping core.planning.visual_servo.PulseShaper
  * rotate-with-stops room sweep             core.planning.visual_servo.ScanSearchPolicy
  * bbox + depth -> metric range             core.mapping.depth.bbox_to_xyz_cam_from_depth
This node owns ONLY ROS concerns: sensor I/O, the demo-mode hand-off, /cmd_vel,
and feeding the pure state machines. Pose is used ONLY for arrival detection.

Inputs  (mirrors navdp_click / combination transports):
  ~rgb_topic     frame-path String or raw Image   (tracked every frame)
  ~depth_topic   frame-path String or raw Image   (optional; metric range)
  ~detections_topic  std_msgs/String JSON         (from yolo_detector_ros2_node)
  ~target_topic  std_msgs/String                  (the mission "goal", e.g. "hat")
  ~enable_topic  std_msgs/Bool                     (mode switch; ~start_enabled)
  ~demo_mode_topic  std_msgs/String                (to know we hold visual_servoing)
  ~pose_topic    PoseStamped/Pose                  (arrival detection only)
  ~goal_in_topic geometry_msgs/Point               (the coordinate route's goal)
Outputs:
  <drone_ns>/cmd_vel  geometry_msgs/Twist          (force-shaped vx, vy, wz)
  ~demo_mode_request_topic  std_msgs/String        (visual_servoing <-> fly_straight)
  ~goal_out_topic  geometry_msgs/Point             (goal re-inject on give-up)
  ~overlay_topic   sensor_msgs/Image               (live target-lock HUD)
  ~status_topic  std_msgs/String                    (diagnostics, optional)
See the file footer for the full rosparam list.
"""
import math
import threading
from collections import deque

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from sparx_agency.core.common.detection_message import parse_detections_message
from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.types import ControlCommand, Intrinsics, KinematicLimits
from sparx_agency.core.mapping.depth.depth_bbox_fusion import bbox_to_xyz_cam_from_depth
from sparx_agency.core.mapping.tracking import (
    make_lock_tracker, TargetTrackerConfig, DetectionOnlyConfig,
    DETECTOR_TRACKER, LOCK_MODES,
)
from sparx_agency.core.planning.visual_servo import (
    VisualServoController, VisualServoParams, VisualServoRequest,
    TargetConfirmationGate, ConfirmationGateConfig, select_overlapping_target_detection,
    ReSearchPolicy, ReSearchConfig,
    VisualApproachStateMachine, ApproachFSMConfig, SEARCH, SCAN, ACQUIRE_STOP,
    AxisForceProfile, PulseShaper,
    ClosureGait, ClosureGaitConfig,
    ScanSearchConfig, ScanSearchPolicy,
)

# The live target-lock HUD (same renderer as the offline tool). Optional: if the
# offline package is not importable in this runtime the mission still runs, only
# the overlay is disabled.
try:
    from sparx_agency.tasks.planning.object_approach_offline import overlay as _overlay
    from sparx_agency.tasks.planning.object_approach_offline.pipeline import FrameResult
    _HAVE_OVERLAY = True
except Exception as _e:                       # noqa: BLE001 -- viz is non-critical
    _overlay = None
    FrameResult = None
    _HAVE_OVERLAY = False
    _OVERLAY_IMPORT_ERR = _e

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
        # Re-assert the demo-mode request at this period until the arbiter echoes
        # it back on demo_mode. Mirrors the follower's ~request_repeat_sec so a
        # last-write-wins arbiter cannot strand our single take-over request and
        # stall the swap into visual servoing.
        self.request_repeat_sec = float(G("~request_repeat_sec", 0.5))

        # Intrinsics matching the live RGB/depth stream (see navdp_click for the
        # K-vs-P note). Depth range uses these; the servo only needs width/height.
        self.intr = Intrinsics(
            width=int(G("~img_width", 504)), height=int(G("~img_height", 294)),
            fx=float(G("~fx", 322.6351083474948)), fy=float(G("~fy", 323.3893307141174)),
            cx=float(G("~cx", 242.06479658679714)), cy=float(G("~cy", 90.03019076680604)))

        # 10 Hz matches the route follower's calibrated fixed-speed/tick numbers
        # (~4 deg/tick at 0.7 rad/s), so the minimum-burst + coast model transfers.
        self.ctrl_hz = float(G("~ctrl_hz", 10.0))
        self.reseed_on_detection = _param_bool("~reseed_on_detection", True)
        self.frame_buffer_len = int(G("~frame_buffer_len", 30))
        self.enabled = _param_bool("~start_enabled", True)
        self.target = str(G("~target_object", "refrigerator")).strip().lower()

        # ── Closure "version" ────────────────────────────────────────
        # multi_axis (holonomic vx+vy+yaw; the default) or waypoint (yaw XOR
        # forward, matching the one-axis follower). It picks the servo mode and
        # whether lateral crab is used; ~servo_mode / ~use_lateral still override.
        self.closure_mode = str(G("~closure_mode", "multi_axis")).strip().lower()
        if self.closure_mode not in ("multi_axis", "waypoint"):
            raise ValueError("~closure_mode must be 'multi_axis' or 'waypoint', "
                             "got %r" % self.closure_mode)
        _is_mx = self.closure_mode == "multi_axis"
        servo_mode = str(G("~servo_mode",
                           "holonomic" if _is_mx else "yaw_forward_xor")).strip().lower()
        use_lateral = _param_bool("~use_lateral", _is_mx)

        # ── Core objects (ROS-free) ──────────────────────────────────
        self.limits = KinematicLimits(
            max_speed_xy=float(G("~max_speed_xy", 0.4)),
            max_speed_z=float(G("~max_speed_z", 0.3)),
            # 0.7 rad/s matches the waypoint follower's proven ~yaw_rate. It is the
            # yaw-pulse magnitude the platform actually turns on, and it caps the
            # shaper's wz axis -- keep it >= the fixed yaw pulse below or the pulse
            # would be clamped back down and the drone would not turn.
            max_yaw_rate=float(G("~max_yaw_rate", 0.7)))
        # ── Closure strategy: detector_tracker (default) or detector-only ──
        # detector_tracker: the detector seeds an optical-flow tracker propagated
        # every frame between detections. detector: the detector's box alone drives
        # closure (no tracking) -- for when the detector already keeps up with the
        # RGB stream, so tracking only adds a way to drift onto the background.
        self.lock_mode = str(G("~lock_mode", DETECTOR_TRACKER)).strip().lower()
        if self.lock_mode not in LOCK_MODES:
            raise ValueError("~lock_mode must be one of %s, got %r"
                             % (list(LOCK_MODES), self.lock_mode))
        # How long a detection stays "fresh": the detector-only closure window AND
        # the HUD's "detector sees it now" (green) staleness gate share this knob.
        self.det_fresh_s = float(G("~max_det_age_s", 0.5))
        self.tracker = make_lock_tracker(
            self.lock_mode,
            tracker_config=TargetTrackerConfig(
                input_is_bgr=True,
                max_predict_s=float(G("~max_predict_s", 0.4)),
                max_unconfirmed_s=float(G("~max_unconfirmed_s", 2.0))),
            detection_config=DetectionOnlyConfig(max_det_age_s=self.det_fresh_s))
        # Soft re-confirmation while tracking: a weak detection ON the tracked box
        # (>= ~soft_confirm_min_score, IoU >= ~confirm_iou) keeps the lock alive and
        # resets the unconfirmed timer, so a genuinely-tracked object is not dropped
        # while pure background drift (no overlapping detection) still times out.
        self.confirm_iou = float(G("~confirm_iou", 0.4))
        self.soft_confirm_min_score = float(G("~soft_confirm_min_score", 0.05))
        self.servo = VisualServoController(VisualServoParams(
            mode=servo_mode,
            kp_yaw=float(G("~kp_yaw", 1.2)),
            max_yaw_rate=float(G("~max_yaw_rate", 0.7)),
            use_lateral=use_lateral,
            use_vertical=_param_bool("~use_vertical", False),
            vx_max=float(G("~vx_max", 0.35)),
            use_depth=_param_bool("~use_depth", True),
            target_range_m=float(G("~target_range_m", 1.0)),   # stop ~0.5 m from the object
            slowdown_range_m=float(G("~slowdown_range_m", 2.0)),
            target_area_frac=float(G("~target_area_frac", 0.12)),
            # ~center_tol is the ACQUISITION ANGLE: the allowed centring deviation
            # for hover-lock. On a pulsed platform we cannot centre to a degree, so
            # "centred" is a small deviation, not exact zero.
            center_tol=float(G("~center_tol", 0.15)),
            # Coarse-yaw platform: a large yaw deadband (min-burst + coast, the yaw
            # acquisition angle) that grows as we close so a yaw does not sweep the
            # object out of frame; a coast-aware crab deadband does the fine centring.
            yaw_deadband=float(G("~yaw_deadband", 0.35)),
            yaw_close_deadband=float(G("~yaw_close_deadband", 0.15)),
            lateral_deadband=float(G("~lateral_deadband", 0.10))),
            default_limits=self.limits)
        self.servo_vx_max = float(G("~vx_max", 0.35))
        self.servo_vy_max = float(G("~max_lateral_speed", 0.25))

        # ── Per-axis force shaping ("minimum force") ─────────────────
        # The servo emits an analog velocity capped only at the top; the platform
        # needs a minimum per-axis force to move at all, so shape every published
        # command (servo / recovery / scan) through the same discipline the
        # multi-axis follower uses. Default mode is "fixed" (bang-bang: 0 or a
        # single fixed pulse per axis). Max = the kinematic limits.
        #
        # The fixed pulse magnitudes default to the SAME proven values the working
        # waypoint follower publishes -- ~fixed_vx=0.3 m/s forward and ~fixed_wz=0.7
        # rad/s yaw (its ~vel_x / ~yaw_rate). Those are the magnitudes the platform's
        # deadband actually moves on; the earlier default (fall back to the tiny
        # ~min_* floor: 0.06 m/s, 8 deg/s) was too weak, so a yaw pulse fired but the
        # drone did not turn. The pulse is held for ~min_burst_ticks (2) consecutive
        # ticks, mirroring the follower's cmd_commit_ticks=2 "at least two commands".
        #
        # It is a stateful PulseShaper (not the memoryless CommandForceShaper): the
        # platform's yaw/forward actuation is discrete + inertial, so a lone control
        # tick does not overcome its deadband. The shaper LATCHES any motion for at
        # least ~min_burst_ticks ticks (a real >=2-tick burst that moves it) -- the
        # count-based-speed model -- and can emit a brief opposite BRAKE pulse to
        # bleed off the coast. Sized for ~ctrl_hz=10 (the follower's calibration).
        self.force_mode = str(G("~force_mode", "fixed")).strip().lower()
        release_frac = float(G("~force_release_frac", 0.5))
        zero_eps = float(G("~force_zero_eps", 1e-3))

        def _axis(min_mag, max_mag, fixed_mag):
            return AxisForceProfile(
                min_magnitude=float(min_mag), max_magnitude=float(max_mag),
                release_frac=release_frac, zero_eps=zero_eps, mode=self.force_mode,
                fixed_magnitude=(None if fixed_mag is None or float(fixed_mag) <= 0.0
                                 else float(fixed_mag)))

        min_vx = float(G("~min_vx", 0.06))
        min_vy = float(G("~min_vy", 0.06))
        min_wz = math.radians(float(G("~min_wz_deg", 8.0)))
        # Proven pulse magnitudes: forward (vx) and yaw (wz) use the waypoint
        # follower's ~vel_x=0.3 m/s / ~yaw_rate=0.7 rad/s. Lateral ROLL (vy) is
        # assumed to move on the same 0.3 m/s forward-force pulse.
        self.shaper = PulseShaper(
            vx=_axis(min_vx, self.limits.max_speed_xy, G("~fixed_vx", 0.3)),
            vy=_axis(min_vy, self.limits.max_speed_xy, G("~fixed_vy", 0.3)),
            wz=_axis(min_wz, self.limits.max_yaw_rate,
                     math.radians(float(G("~fixed_wz_deg", math.degrees(0.7))))),
            min_burst_ticks=int(G("~min_burst_ticks", 2)),
            brake_ticks=int(G("~brake_ticks", 1)))

        # ── Pulse-and-settle closure gait ────────────────────────────
        # A cadence post-filter on the shaped SERVO command: the shaper sets the
        # pulse magnitude, this gait sets the rhythm. It lets a short burst through
        # (~gait_move_ticks) then forces a brief stop (~gait_settle_ticks) so the
        # drone coasts to rest and the camera gets a fresh bbox before the next
        # burst -- turning a "too strong / too continuous" servo stream into a
        # move-a-little / stop-and-look rhythm, and stopping when a turn gives way
        # to forward flight (~gait_settle_on_transition). Sized for ~ctrl_hz=10.
        self.gait = ClosureGait(ClosureGaitConfig(
            move_ticks=int(G("~gait_move_ticks", 2)),
            settle_ticks=int(G("~gait_settle_ticks", 4)),
            settle_on_axis_change=_param_bool("~gait_settle_on_transition", True),
            enabled=_param_bool("~gait_enabled", True)))

        # ── Scan-at-goal sweep (arrived, still looking) ──────────────
        self.arrive_radius_m = float(G("~arrive_radius_m", 0.6))
        self.scan = ScanSearchPolicy(ScanSearchConfig(
            yaw_rate=float(G("~scan_yaw_rate", 0.4)),
            rotate_s=float(G("~scan_rotate_s", 1.2)),
            pause_s=float(G("~scan_pause_s", 1.2)),
            direction=(1.0 if float(G("~scan_direction", 1.0)) >= 0.0 else -1.0),
            forward_speed=float(G("~scan_forward_speed", 0.0)),
            forward_s=float(G("~scan_forward_s", 0.0)),
            bursts_before_move=int(G("~scan_bursts_before_move", 8))))
        self.gate = TargetConfirmationGate(self.target, ConfirmationGateConfig(
            n_confirm=int(G("~n_confirm", 3)),
            min_score=float(G("~min_score", 0.30))))
        # The FSM's recover_timeout_s is the single source of truth for how long
        # we re-search before handing control back; the recovery policy's own
        # give-up mirrors it (the node acts on the FSM, not on rec.give_up).
        recover_timeout_s = float(G("~recover_timeout_s", 6.0))
        # RECOVER manoeuvre: chase the bearing it left on (directional), or peek
        # around a central occluder (oscillating sidestep+yaw). Speeds are kept
        # small and the peek oscillates so the drone stays near the loss position
        # (wall safety); tune the *_speed knobs down further near tight spaces.
        self.recovery = ReSearchPolicy(ReSearchConfig(
            search_yaw_rate=float(G("~search_yaw_rate", 0.5)),
            max_search_s=recover_timeout_s,
            hold_before_search_s=float(G("~recover_hold_s", 0.3)),
            center_exit_frac=float(G("~recover_center_exit_frac", 0.25)),
            directional_roll_speed=float(G("~recover_directional_roll", 0.05)),
            peek_forward_speed=float(G("~recover_peek_forward", 0.06)),
            peek_forward_s=float(G("~recover_peek_forward_s", 0.6)),
            peek_roll_speed=float(G("~recover_peek_roll", 0.10)),
            peek_orbit=_param_bool("~recover_peek_orbit", True)))
        # On acquisition (target confirmed + tracker locked) hold a brief
        # stop-in-place before the visual approach begins: the node takes over
        # /cmd_vel and actively publishes a zero-velocity stop for ~acquire_stop_s
        # so the drone brakes off any inherited A*/NavDP route motion and the
        # follower's last command lapses -- only then does it start flying to the
        # object. 0 disables the settle (approach immediately, the legacy path).
        self.fsm = VisualApproachStateMachine(ApproachFSMConfig(
            recover_timeout_s=recover_timeout_s,
            acquire_stop_s=float(G("~acquire_stop_s", 1.5))))

        # ── Goal memory + re-inject ───────────────────────────────────
        # We remember the coordinate route's goal (from ~goal_x/y and any live
        # /waypoint_nav/goal click) so that (a) we can tell when the drone has
        # arrived (-> scan), and (b) on a lost-track give-up we re-assert it, so the
        # planner resumes flying to the last/initial goal instead of stalling.
        self.goal_in_topic = G("~goal_in_topic", "/waypoint_nav/goal")
        self.goal_out_topic = G("~goal_out_topic", "/waypoint_nav/goal")
        gx, gy = G("~goal_x", None), G("~goal_y", None)
        self._goal_xy = None if gx is None or gy is None else (float(gx), float(gy))
        self._pose_xy = None

        # ── Live HUD (overlay Image) ──────────────────────────────────
        self.publish_overlay = _param_bool("~publish_overlay", True) and _HAVE_OVERLAY
        self.overlay_topic = G("~overlay_topic", "/object_approach/overlay")
        self.viz_hz = float(G("~viz_hz", 10.0))
        # Full-scale gauge references (max the servo can command, floored by any
        # kinematic limit) -- for a HUD gauge, not a command.
        self.gauge_max_vx = min(self.servo_vx_max, self.limits.max_speed_xy)
        self.gauge_max_vy = min(self.servo_vy_max, self.limits.max_speed_xy)
        self.gauge_max_yaw_rate = min(float(G("~max_yaw_rate", 0.7)),
                                      self.limits.max_yaw_rate)

        # ── Shared state ──────────────────────────────────────────────
        # _lock guards tracker/gate/frame-buffer/viz snapshot; _mode_lock guards
        # the enabled + demo-mode-request handshake (mutated by the enable callback
        # thread and the control-timer thread).
        self._lock = threading.Lock()
        self._mode_lock = threading.Lock()
        self.rgb = None
        self.rgb_stamp = 0.0
        self.depth = None
        self._frame_buf = deque(maxlen=max(2, self.frame_buffer_len))  # (stamp, bgr)
        self._confirmed = False
        self._streak = 0
        self._last_dets = []
        self._last_target_det = None     # matching target detection (HUD green box)
        self._last_target_det_t = 0.0    # its stamp, for the HUD freshness gate
        self.current_demo_mode = None
        self._requested_mode = None
        self._last_request_pub_t = rospy.Time(0)
        self._last_dec = None
        self._last_res = None
        self._last_track = None
        self._last_cmd = None            # last SHAPED ControlCommand (None in SEARCH)
        self._last_cmd_source = "planner (visual approach passive)"
        self._prev_tick_t = None

        # ── ROS I/O (publishers before subscribers) ──────────────────
        self.cmd_pub = rospy.Publisher(self.drone_ns + "/cmd_vel", Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1)
        self.goal_pub = rospy.Publisher(self.goal_out_topic, Point, queue_size=1, latch=True)
        self.overlay_pub = (rospy.Publisher(self.overlay_topic, Image, queue_size=1)
                            if self.publish_overlay else None)

        if _fp:
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, String, self._depth_path_cb, queue_size=2)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=2)
        if self.camera_info_topic:
            rospy.Subscriber(self.camera_info_topic, CameraInfo, self._cam_info_cb,
                             queue_size=1)
        # Pose (for arrival detection): PoseStamped on /xtend/localization by default.
        if self.pose_type == "pose_stamped":
            rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_stamped_cb, queue_size=5)
        else:
            rospy.Subscriber(self.pose_topic, Pose, self._pose_cb, queue_size=5)
        rospy.Subscriber(self.goal_in_topic, Point, self._goal_cb, queue_size=1)
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
            # Drop the old target's detection so the HUD does not draw a green box
            # (labelled with the NEW target) over the OLD object until the detector
            # re-fires on the new prompt.
            self._last_target_det = None

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

    def _pose_stamped_cb(self, msg):
        self._pose_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _pose_cb(self, msg):
        self._pose_xy = (float(msg.position.x), float(msg.position.y))

    def _goal_cb(self, msg):
        # Remember the coordinate route's live goal (a bev click). Ignore our own
        # latched re-inject (same value) so we never fight the planner over it.
        self._goal_xy = (float(msg.x), float(msg.y))

    def _arrived_at_goal(self):
        """True once the drone is within ``arrive_radius_m`` of the known goal --
        the proxy for "the A*/NavDP route reached its goal" (there is no done
        topic). None goal or pose -> False (never scans without a known goal)."""
        if self._goal_xy is None or self._pose_xy is None:
            return False
        dx = self._pose_xy[0] - self._goal_xy[0]
        dy = self._pose_xy[1] - self._goal_xy[1]
        return math.hypot(dx, dy) <= self.arrive_radius_m

    # ─── Detections: confirm + (re)seed the tracker ──────────────────
    def _det_cb(self, msg):
        # Detections arrive from the host-side ROS2 sidecar over the bridge; the
        # wire format is core.common.detection_message.
        try:
            parsed = parse_detections_message(
                msg.data, default_width=self.intr.width,
                default_height=self.intr.height, default_stamp=self.rgb_stamp)
        except ValueError as e:
            rospy.logwarn_throttle(5.0, "object_approach: bad detections msg (%s)", e)
            return
        dets, stamp = parsed.detections, parsed.stamp

        with self._lock:
            state = self.gate.update(dets)
            self._confirmed = state.confirmed
            self._streak = state.streak
            self._last_dets = dets          # for the HUD overlay
            self._last_target_det = state.best   # HARD match -> HUD green box
            self._last_target_det_t = stamp      # stamped for the freshness gate
            # Hard match acquires/greens; while tracking, a WEAK detection on the
            # tracked box also re-confirms (keeps the lock alive + resets the
            # unconfirmed timer). Background drift has no such overlap and times out.
            confirm_det = state.best
            if (confirm_det is None and self.tracker.has_target
                    and self.confirm_iou > 0.0 and self.tracker.last_track is not None):
                confirm_det = select_overlapping_target_detection(
                    dets, self.target, self.tracker.last_track.bbox_xyxy,
                    self.confirm_iou, self.soft_confirm_min_score)
            if confirm_det is None:
                return
            # Seed on acquisition; re-seed later to bound drift / regain a lost lock.
            # A non-propagating (detector-only) tracker has no state to coast on, so
            # ALWAYS re-feed it the latest detection -- "seed once" would freeze the
            # lock and abandon a still-visible target.
            reseed = self.reseed_on_detection or not self.tracker.propagates
            need_seed = (not self.tracker.has_target and state.confirmed) or \
                        (self.tracker.has_target and reseed)
            if need_seed:
                frame = self._closest_frame(stamp)
                if frame is not None:
                    self.tracker.on_detection(frame, confirm_det, stamp)

    def _closest_frame(self, stamp):
        if not self._frame_buf:
            return None
        return min(self._frame_buf, key=lambda kv: abs(kv[0] - stamp))[1]

    # ─── Control loop ────────────────────────────────────────────────
    def start(self):
        rospy.Timer(rospy.Duration(1.0 / max(self.ctrl_hz, 1.0)), self._tick)
        rospy.Timer(rospy.Duration(2.0), self._hb)
        if self.overlay_pub is not None and self.viz_hz > 0.0:
            rospy.Timer(rospy.Duration(1.0 / self.viz_hz), self._publish_overlay)
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

        # Arrival at the coordinate goal (no done topic exists) lets the FSM switch
        # from passive SEARCH to an active room SCAN when still unconfirmed.
        arrived = self._arrived_at_goal()
        dec = self.fsm.update(confirmed=confirmed, track_valid=track_valid,
                              at_target=at_target, dt=dt, arrived_at_goal=arrived)
        self._last_dec, self._last_res, self._last_track = dec, res, track

        if dec.reset_acquisition:
            with self._lock:
                self.gate.reset()
                self.tracker.reset()
                self._confirmed = False
            # Lost the object for good: re-assert the goal so the planner flies us
            # back to the last/initial goal rather than stalling where we gave up.
            self._reinject_goal()

        # The sweep is stateful: keep it reset unless we are actively scanning, so
        # each SCAN episode starts from a clean look straight ahead.
        if dec.mode != SCAN:
            self.scan.reset()

        if not dec.drive_cmd_vel:                 # SEARCH: hand /cmd_vel back
            self._release()
            self._last_cmd = None
            self._last_cmd_source = "planner (visual approach passive)"
            return

        # We own /cmd_vel: make the follower passive first. Do NOT publish until
        # the hand-off is confirmed (demo_mode == visual_servoing), else we and the
        # still-active follower would both drive /cmd_vel for the round-trip.
        self._request_mode(MODE_VISUAL_SERVOING)
        if self.current_demo_mode != MODE_VISUAL_SERVOING:
            self._last_cmd = None
            self._last_cmd_source = "awaiting %s hand-off" % MODE_VISUAL_SERVOING
            return
        if dec.mode == ACQUIRE_STOP:               # just locked: settle in place
            # Now that we own /cmd_vel (the follower is passive), actively publish a
            # zero-velocity stop so the drone brakes off any inherited route motion
            # and no stale A*/NavDP command lingers before we start flying in. Keep
            # the closure gait reset so the approach opens with a fresh burst.
            self.gait.reset()
            self._publish_cmd(0.0, 0.0, 0.0, "acquire_stop")
        elif dec.mode == SCAN:                      # arrived, sweeping the room
            c = self.scan.command(dt)
            self._publish_cmd(c.x, c.y, c.yaw_rate, "scan:%s" % self.scan.phase)
        elif dec.mode == "RECOVER" or res is None:
            self._drive_recovery(last_track, dec.lost_for_s)
        else:
            # Servo closure (APPROACH / HOVER_LOCK): gate the shaped command through
            # the pulse-and-settle gait so closing is a move-a-little / stop-and-look
            # rhythm rather than one strong continuous run.
            c = res.command
            self._publish_cmd(c.x, c.y, c.yaw_rate, "servo:%s" % res.mode, gated=True)

    def _drive_recovery(self, last_track, lost_for_s):
        # Recovery has its own bounded sweep cadence, so it is not gated; keep the
        # closure gait reset so a regained track re-opens APPROACH with a fresh burst.
        self.gait.reset()
        rec = self.recovery.command(last_track, lost_for_s,
                                    self.intr.width, self.intr.height)
        c = rec.command
        self._publish_cmd(c.x, c.y, c.yaw_rate, "recovery:%s" % rec.phase)

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
        # True only once the hand-off is GRANTED (the arbiter echoed
        # visual_servoing), i.e. we are actually the sole cmd_vel owner -- not
        # merely while we are still requesting it. Gating on the echoed mode (not
        # the pending request) means a mid-hand-off error tick does not inject a
        # stop onto cmd_vel while the follower is still the active publisher.
        return self.current_demo_mode == MODE_VISUAL_SERVOING

    def _request_mode(self, mode):
        with self._mode_lock:
            self._request_mode_locked(mode)

    def _request_mode_locked(self, mode):
        """Publish a demo-mode request, re-asserting until granted. Hold _mode_lock.

        Publishes on change and then KEEPS re-publishing every
        ``request_repeat_sec`` until the arbiter echoes the mode back on
        ``demo_mode`` (``current_demo_mode == mode``). Publishing once is not
        enough: the follower re-requests its own mode on the same request topic
        every ``request_repeat_sec`` (waypoint_follower ``_request_demo_mode``),
        so under a last-write-wins arbiter a single take-over request can be
        overwritten and the swap into visual servoing would stall forever.
        """
        if self._requested_mode != mode:
            self._requested_mode = mode
            self._last_request_pub_t = rospy.Time(0)   # force an immediate publish
            rospy.loginfo("object_approach: request demo_mode=%s (current=%s)",
                          mode, self.current_demo_mode)
        if self.current_demo_mode == mode:             # already granted; nothing to do
            return
        now = rospy.Time.now()
        if (now - self._last_request_pub_t).to_sec() >= self.request_repeat_sec:
            self.demo_req_pub.publish(String(data=mode))
            self._last_request_pub_t = now

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
            # Clear the pulse-burst + gait state FIRST so the release "stop" is a
            # clean zero (not a min-burst that would finish as one more motion pulse),
            # and so the next closing episode starts fresh.
            self.shaper.reset()
            self.gait.reset()
            self._publish_cmd(0.0, 0.0, 0.0, "stop")
            self._request_mode_locked(MODE_RELEASE)

    def _publish_cmd(self, vx, vy, wz, source="servo", gated=False):
        """Force-shape (per-axis min/max) then publish the body-velocity Twist.

        Shaping is the FINAL stage before the wire, so every published command --
        servo, recovery sweep, room scan, or brake -- respects the platform's
        minimum/maximum per-axis force. Shaping ``0 -> 0``, so a stop stays a stop.

        ``gated`` additionally runs the shaped command through the pulse-and-settle
        closure gait (the move-a-little / stop-and-look cadence). Only the servo
        closure path sets it; scan/recovery/brake keep their own cadence and pass
        through un-gated.
        """
        shaped = self.shaper.shape(
            ControlCommand.velocity(float(vx), float(vy), 0.0, float(wz)))
        if gated:
            shaped = self.gait.step(shaped)
        m = Twist()
        m.linear.x = float(shaped.x)
        m.linear.y = float(shaped.y)    # holonomic crab (0 in waypoint closure)
        m.linear.z = 0.0                # fixed altitude (platform holds it)
        m.angular.z = float(shaped.yaw_rate)
        self.cmd_pub.publish(m)
        self._last_cmd = shaped
        self._last_cmd_source = source

    def _reinject_goal(self):
        """Re-publish the last/initial goal so the planner resumes to it (R2)."""
        if self._goal_xy is None:
            return
        p = Point(x=float(self._goal_xy[0]), y=float(self._goal_xy[1]), z=0.0)
        self.goal_pub.publish(p)
        rospy.loginfo("object_approach: re-inject goal (%.2f, %.2f) -> %s",
                      p.x, p.y, self.goal_out_topic)

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

    # ─── Live HUD overlay ────────────────────────────────────────────
    def _publish_overlay(self, _evt):
        """Render the live target-lock HUD (the same renderer the offline tool uses)
        from the ACTUAL mission state -- detections, tracked box, and the exact
        SHAPED command being published -- and publish it as a bgr8 Image."""
        if self.overlay_pub is None:
            return
        with self._lock:
            bgr = self.rgb                     # node stores BGR (see _push_rgb)
            dets = list(self._last_dets)
            # Only treat the target as "detected now" (HUD green) while the last
            # matching detection is fresh -- a stalled/crashed detector must not
            # leave a confident green box frozen on a target we no longer see.
            target_det = self._last_target_det
            if target_det is not None and \
                    (self.rgb_stamp - self._last_target_det_t) > self.det_fresh_s:
                target_det = None
            track = self._last_track
            stamp = self.rgb_stamp
        if bgr is None:
            return
        res, dec, cmd = self._last_res, self._last_dec, self._last_cmd
        fr = FrameResult(
            stamp_s=stamp, dt=0.0, target=self.target,
            detections=dets, target_detection=target_det,
            confirmed=self._confirmed, streak=self._streak,
            track=track, fsm_mode=(dec.mode if dec is not None else SEARCH),
            at_target=bool(res is not None and res.at_target),
            x_offset=None if res is None else res.x_offset,
            y_offset=None if res is None else res.y_offset,
            area_frac=None if res is None else res.area_frac,
            range_m=None if res is None else res.range_m,
            command=cmd, cmd_source=self._last_cmd_source,
            gauge_max_vx=self.gauge_max_vx, gauge_max_vy=self.gauge_max_vy,
            gauge_max_yaw_rate=self.gauge_max_yaw_rate)
        try:
            img = _overlay.render(bgr, fr)
        except Exception as e:                    # noqa: BLE001 -- viz must not kill the node
            rospy.logwarn_throttle(5.0, "object_approach: overlay render failed (%s)", e)
            return
        img = np.ascontiguousarray(img)
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(img.shape[1] * 3)
        msg.data = img.tobytes()
        self.overlay_pub.publish(msg)

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("object_approach (lock onto a named object -> visual approach)")
        L("  rgb   in  = %s  (%s)", self.rgb_topic, self.image_transport)
        L("  depth in  = %s", self.depth_topic)
        L("  dets  in  = %s", self.detections_topic)
        L("  goal  in  = %s   (start target=%r)", self.target_topic, self.target)
        L("  enable    = %s   (start_enabled=%s)", self.enable_topic, self.enabled)
        L("  cmd_vel out = %s  (via %s hand-off)",
          self.drone_ns + "/cmd_vel", MODE_VISUAL_SERVOING)
        L("  lock_mode = %s   (%s)", self.lock_mode,
          "detector seeds an optical-flow tracker"
          if self.lock_mode == DETECTOR_TRACKER else "detector box only, no tracking")
        L("  closure   = %s   force=%s (min vx=%.3f vy=%.3f wz=%.0f deg/s)",
          self.closure_mode, self.force_mode, self.shaper.vx.min_magnitude,
          self.shaper.vy.min_magnitude, math.degrees(self.shaper.wz.min_magnitude))
        L("  gait      = %s  (move %d / settle %d ticks, transition-stop=%s)",
          "on" if self.gait.cfg.active else "off", self.gait.cfg.move_ticks,
          self.gait.cfg.settle_ticks, self.gait.cfg.settle_on_axis_change)
        L("  nav goal  = %s  (in=%s out=%s, arrive<%.2fm -> SCAN)",
          self._goal_xy, self.goal_in_topic, self.goal_out_topic, self.arrive_radius_m)
        L("  HUD out   = %s  @ %.1f Hz (%s)", self.overlay_topic, self.viz_hz,
          "on" if self.overlay_pub is not None else
          ("off" if _HAVE_OVERLAY else "overlay import failed: %s" % _OVERLAY_IMPORT_ERR))
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
# is ROS-free in core.planning.visual_servo / core.mapping.tracking / mapping.detection;
# this node owns ROS I/O, the demo-mode hand-off and the control-loop plumbing.
#
#   IO: ~image_transport (frame_path | topic)
#       ~rgb_topic (/xtend/rgb_frame_path) ~depth_topic (/xtend/depth_frame_path)
#       ~detections_topic (/object_approach/detections)  ~target_topic (/object_approach/goal)
#       ~enable_topic (/object_approach/enable)  ~status_topic (/object_approach/status)
#       ~pose_topic (/xtend/localization) ~pose_type (pose_stamped | pose)  [arrival test]
#       ~drone_ns ('')  [-> <drone_ns>/cmd_vel]
#       ~demo_mode_topic (/xtend/demo_mode)  ~demo_mode_request_topic (/xtend/demo_mode_request)
#   camera (MUST match the live stream; K over P): ~fx ~fy ~cx ~cy ~img_width (504)
#       ~img_height (294)  ~camera_info_topic ('' = use params)
#   mission: ~target_object (refrigerator) ~start_enabled (true) ~ctrl_hz (10.0,
#       matches the route follower's fixed-speed/tick calibration)
#   acquisition: ~n_confirm (3) ~min_score (0.30) ~acquire_stop_s (1.5, stop-in-place
#       hold right after confirm+lock -- the node publishes a zero-velocity stop that
#       brakes off inherited A*/NavDP route motion before the approach; 0 = off)
#   closure strategy: ~lock_mode (detector_tracker | detector). detector_tracker
#       (default): detector seeds an optical-flow tracker propagated every frame.
#       detector: the detector's box alone drives closure (no tracking), holding
#       the last box for ~max_det_age_s (0.5) -- use when the detector keeps up
#       with the RGB stream.
#   tracking: ~reseed_on_detection (true) ~frame_buffer_len (30) ~max_predict_s (0.4)
#       ~max_unconfirmed_s (2.0, drop the lock if the detector hasn't re-confirmed
#       the target for this long -> stops tracking the background). A weak detection
#       ON the tracked box re-confirms: ~confirm_iou (0.4) ~soft_confirm_min_score
#       (0.05, below min_score; the detector's conf_thresh must reach this low)
#   closure version: ~closure_mode (multi_axis | waypoint). multi_axis -> holonomic
#       servo (vx+vy+yaw); waypoint -> yaw_forward_xor (yaw OR forward, no crab).
#       ~servo_mode / ~use_lateral still override the derived defaults.
#   servo: ~kp_yaw (1.2) ~vx_max (0.35) ~max_lateral_speed (0.25) ~use_vertical (false)
#       ~center_tol (0.15, the ACQUISITION ANGLE = allowed centring deviation for
#       hover-lock; not exact centre) ~use_depth (true) ~target_range_m (0.5, STOP
#       distance: hold this far from the object) ~slowdown_range_m (2.0)
#       ~target_area_frac (0.12) [area used when depth absent] ~lateral_deadband
#       (0.10, coast-aware crab deadband for the fine centring)
#       coarse-yaw platform: ~yaw_deadband (0.22, larger than the crab deadband so
#       fine centring is done by crab, not yaw) ~yaw_close_deadband (0.15, extra yaw
#       deadband as we close so a yaw does not sweep the object out of frame)
#   flight-command shaping (discrete + inertial, on EVERY published command):
#       ~force_mode (fixed | snap | none; default fixed = bang-bang 0/±pulse)
#       ~min_vx (0.06) ~min_vy (0.06) ~min_wz_deg (8.0) ~force_release_frac (0.5)
#       ~force_zero_eps (1e-3). Fixed pulse magnitudes = the waypoint follower's
#       proven ~vel_x/~yaw_rate: ~fixed_vx (0.3 m/s) ~fixed_wz_deg (~40.1 = 0.7 rad/s)
#       ~fixed_vy (0.3 m/s lateral ROLL pulse; assumed same as the forward force)
#       ~min_burst_ticks (2, hold any motion >=N ticks so a lone tick that the
#       platform ignores becomes a real burst) ~brake_ticks (0, opposite pulses to
#       bleed off the coast after a burst; 0 = rely on the yaw deadband stopping early)
#   closure gait (move-a-little / stop-and-look cadence on the SERVO command only):
#       ~gait_enabled (true) ~gait_move_ticks (2, max motion ticks per burst)
#       ~gait_settle_ticks (4, forced-stop ticks after a burst / on a turn<->forward
#       change; 0 = gait off) ~gait_settle_on_transition (true, stop when the motion
#       axis changes). Sized for ~ctrl_hz=10.
#   limits: ~max_speed_xy (0.4) ~max_speed_z (0.3) ~max_yaw_rate (0.7, >= the fixed
#       yaw pulse so the shaper does not clamp it below the turn magnitude)
#   recovery (RECOVER manoeuvre to re-see a lost target -- all small/bounded for
#       wall safety, time-bounded by recover_timeout_s):
#       ~search_yaw_rate (0.5) ~recover_timeout_s (6.0) ~recover_hold_s (0.3, hover
#       first to let a re-detection recover) ~recover_center_exit_frac (0.25, below
#       this exit strength the target is "vanished centre" -> peek, else directional)
#       directional (ran to a side): ~recover_directional_roll (0.05, crab toward it)
#       peek (occluded ahead): ~recover_peek_forward (0.06) ~recover_peek_forward_s
#       (0.6, one bounded forward nudge) ~recover_peek_roll (0.10, sidestep held to
#       one side) ~recover_peek_orbit (true, yaw opposite the sidestep to keep
#       looking around the occluder)
#   goal (arrival -> SCAN, re-inject on give-up): ~goal_in_topic (/waypoint_nav/goal)
#       ~goal_out_topic (/waypoint_nav/goal) ~goal_x/~goal_y (initial goal, unset =
#       none) ~arrive_radius_m (0.6)
#   scan-at-goal sweep: ~scan_yaw_rate (0.4) ~scan_rotate_s (1.2) ~scan_pause_s (1.2)
#       ~scan_direction (+1 CCW / -1 CW) ~scan_forward_speed (0.0 = in place)
#       ~scan_forward_s (0.0) ~scan_bursts_before_move (8)
#   HUD overlay: ~publish_overlay (true) ~overlay_topic (/object_approach/overlay)
#       ~viz_hz (10.0)  [sensor_msgs/Image bgr8; view with target_lock_viewer_node]
# ============================================================================
