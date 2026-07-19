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
until it is centred and very close. On reaching the object (the depth range holds
``<= ~land_range_m`` for ``~land_confirm_ticks`` ticks) the mission TERMINATES by
LANDING: the node stops driving ``/cmd_vel`` (so the platform's coasting cannot drift
it into the object) and requests the ``finish`` demo mode, whose ROS2 manager sends
stop -> land -> disarm on ``/xtend/cmd_nav`` (the only land path from this ROS1
container -- cmd_nav is not bridged; demo_mode_request is). Setting ``~land_range_m``
<= 0 restores the legacy hover-lock terminal (HOVER_LOCK keeps tracking with no land,
re-entering APPROACH if the object moves). While closing, if the track is lost the
node actively re-searches in the direction the target left; if it cannot re-acquire
it hands control back, re-asserts the goal, and returns to SEARCH.

STAGED APPROACH. The coordinate goal the mission flies to is normally NOT the
object: the director sends the drone to a vantage point (the room centre) and
publishes the object's catalogued position on ``~object_position_topic``. That
position is only as accurate as the room map that produced it, so flying onto it
risks ending up beside or past the object with nothing in frame. Instead, on
arriving at the vantage point still unconfirmed, the node enters AIM: it turns the
nose onto the object's bearing in pulsed bursts and holds still, looking -- the
camera resolves a bearing far better than the map resolves a position, so this is
the best shot at handing the servo a lock. Only if the look fails does it ESCALATE,
re-publishing the goal at the object's own (x, y) so the planner flies the last leg
after all; from there the mission is the unstaged one. With no object position
published (or ``~aim_before_direct`` false) none of this arms.

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
  * turn onto a bearing and hold, looking     core.planning.visual_servo.AimBearingPolicy
  * SEARCH/AIM/SCAN/ACQUIRE_STOP/APPROACH/HOVER_LOCK/RECOVER  core.planning.visual_servo.VisualApproachStateMachine
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
  ~pose_topic    PoseStamped/Pose                  (arrival detection + aim heading)
  ~goal_in_topic geometry_msgs/Point               (the coordinate route's goal)
  ~object_position_topic geometry_msgs/Point       (the object's catalogued x,y)
Outputs:
  <drone_ns>/cmd_vel  geometry_msgs/Twist          (force-shaped vx, vy, wz)
  ~demo_mode_request_topic  std_msgs/String        (visual_servoing / fly_straight / finish)
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
from sparx_agency.core.common.spatial_math import quat_to_yaw
from sparx_agency.core.common.types import (
    ControlCommand, Intrinsics, KinematicLimits, normalize_angle,
)
from sparx_agency.core.mapping.depth.depth_bbox_fusion import bbox_to_xyz_cam_from_depth
from sparx_agency.core.mapping.tracking import (
    make_lock_tracker, TargetTrackerConfig, DetectionOnlyConfig,
    DETECTOR_TRACKER, LOCK_MODES,
)
from sparx_agency.core.planning.visual_servo import (
    VisualServoController, VisualServoParams, VisualServoRequest,
    TargetConfirmationGate, ConfirmationGateConfig, select_overlapping_target_detection,
    ReSearchPolicy, ReSearchConfig,
    VisualApproachStateMachine, ApproachFSMConfig, SEARCH, SCAN, ACQUIRE_STOP, LAND,
    APPROACH, HOVER_LOCK, RECOVER, AIM,
    AxisForceProfile, PulseShaper,
    ClosureGait, ClosureGaitConfig,
    ScanSearchConfig, ScanSearchPolicy,
    AimBearingConfig, AimBearingPolicy,
)

from thinking import Thinker

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
MODE_FINISH = "finish"                     # ROS2 demo manager: stop -> land -> disarm
MODE_RECOVERY = "recovery"                 # lost_localization owns /cmd_vel; WE go passive

#: What each FSM state means for the operator, as one first-person (template,
#: level) line about the target. Keyed by mode so the narration rides the state
#: machine's own transitions -- the Thinker's gate turns a per-tick say() into one
#: line per transition, so no edge bookkeeping is needed here. States the operator
#: does not need narrated are simply absent: SEARCH (passive; the route follower
#: narrates the flying) and LAND (the mission end, narrated by _begin_land).
MODE_THOUGHTS = {
    ACQUIRE_STOP: ("Locked onto the %s -- stopping to settle before I close in", "info"),
    APPROACH: ("Homing on the %s", "info"),
    HOVER_LOCK: ("Reached the %s -- holding position and keeping it in frame", "info"),
    RECOVER: ("Lost the %s from frame -- searching", "warn"),
    SCAN: ("Arrived at the goal without seeing the %s -- sweeping the room for it",
           "info"),
    AIM: ("Reached my vantage point -- turning to look at where the %s should be",
          "info"),
}


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
        # Stored so the LAND trigger (which needs a metric range) can warn at startup
        # if it was enabled without depth -- otherwise it would silently never land.
        self.use_depth = _param_bool("~use_depth", True)
        self.servo = VisualServoController(VisualServoParams(
            mode=servo_mode,
            kp_yaw=float(G("~kp_yaw", 1.2)),
            max_yaw_rate=float(G("~max_yaw_rate", 0.7)),
            use_lateral=use_lateral,
            use_vertical=_param_bool("~use_vertical", False),
            vx_max=float(G("~vx_max", 0.35)),
            use_depth=self.use_depth,
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
        # ── Terminal LAND on closure ─────────────────────────────────
        # Reaching the object ENDS the mission by landing, not by an endless
        # hover-lock: once the depth range to the locked target holds <=
        # ~land_range_m for ~land_confirm_ticks consecutive ticks, the node stops
        # driving /cmd_vel and requests the FINISH demo mode. /xtend/cmd_nav is NOT
        # bridged into this ROS1 container (see falcon/bridge/bridge.yaml), so the
        # demo-mode handshake is the ONLY land path reachable from here -- the ROS2
        # xtend_drone_demo_manager turns FINISH into stop -> land -> disarm on
        # /xtend/cmd_nav. Needs depth (~use_depth true); ~land_range_m <= 0 disables
        # it (keep the legacy hover-lock-forever behaviour).
        _land_range = float(G("~land_range_m", 1.0))
        self.land_range_m = _land_range if _land_range > 0.0 else None
        self.land_confirm_ticks = int(G("~land_confirm_ticks", 3))
        # ── Terminal LAND on coordinate arrival (reached "by A* alone") ──
        # When ~land_at_goal is true, arriving at the coordinate goal still
        # unconfirmed is itself a terminal LAND: the goal IS the object's location,
        # so having flown there with the planner (the detector never saw it) is a
        # success -- land there instead of sweeping the room. This is POSE-based (it
        # keys off _arrived_at_goal, ~arrive_radius_m) and needs NO depth, so it works
        # even with ~use_depth false or ~land_range_m<=0. Off by default (arrival ->
        # scan-at-goal, the legacy behaviour). Independent of the depth land above:
        # either, both, or neither may be enabled.
        self.land_at_goal = _param_bool("~land_at_goal", False)
        self.arrive_land_confirm_ticks = int(G("~arrive_land_confirm_ticks", 5))
        if self.land_range_m is not None and not self.use_depth:
            # The depth LAND trigger keys off the metric depth range, which is None
            # whenever ~use_depth is false -- so it would never fire. (The pose-based
            # ~land_at_goal trigger, if enabled, still lands on arrival.) Fail loud.
            rospy.logwarn(
                "object_approach: ~land_range_m=%.2f set but ~use_depth is false -- "
                "the DEPTH land will NEVER trigger; %s Enable ~use_depth, set "
                "~land_range_m<=0, or rely on ~land_at_goal.", self.land_range_m,
                "arrival-land (~land_at_goal) still applies." if self.land_at_goal
                else "the drone will hover-lock, not land.")

        # On acquisition (target confirmed + tracker locked) hold a brief
        # stop-in-place before the visual approach begins: the node takes over
        # /cmd_vel and actively publishes a zero-velocity stop for ~acquire_stop_s
        # so the drone brakes off any inherited A*/NavDP route motion and the
        # follower's last command lapses -- only then does it start flying to the
        # object. 0 disables the settle (approach immediately, the legacy path).
        self.fsm = VisualApproachStateMachine(ApproachFSMConfig(
            recover_timeout_s=recover_timeout_s,
            acquire_stop_s=float(G("~acquire_stop_s", 1.5)),
            land_range_m=self.land_range_m,
            land_confirm_ticks=self.land_confirm_ticks,
            land_at_goal=self.land_at_goal,
            arrive_land_confirm_ticks=self.arrive_land_confirm_ticks))

        # ── Goal memory + re-inject ───────────────────────────────────
        # We remember the coordinate route's goal (from ~goal_x/y and any live
        # /waypoint_nav/goal click) so that (a) we can tell when the drone has
        # arrived (-> aim/scan), and (b) on a lost-track give-up we re-assert it, so
        # the planner resumes flying to the last/initial goal instead of stalling.
        self.goal_in_topic = G("~goal_in_topic", "/waypoint_nav/goal")
        self.goal_out_topic = G("~goal_out_topic", "/waypoint_nav/goal")
        gx, gy = G("~goal_x", None), G("~goal_y", None)
        self._goal_xy = None if gx is None or gy is None else (float(gx), float(gy))
        self._pose_xy = None
        self._pose_yaw = None

        # ── Staged approach: stand off, AIM, then close ───────────────
        # The object's catalogued (x, y) is only as good as the room map that made
        # it -- fly onto it and a few tens of cm of error can leave the object
        # behind or beside us, out of frame, with nothing to servo onto. So the
        # mission director sends us to a STAGING vantage point instead (the room
        # centre) and publishes the object's own position here. On arriving at the
        # staging point still unconfirmed we turn the nose onto the object's bearing
        # and look: the camera resolves a bearing far better than the map resolves a
        # position, so this is the shot most likely to hand the servo a lock.
        # Only if that look fails do we ESCALATE -- re-target the coordinate goal at
        # the object's own (x, y) and let the planner fly the last leg after all.
        # Without an object position (no director, or ~aim_before_direct false) none
        # of this arms and the node behaves exactly as before.
        self.object_position_topic = G("~object_position_topic",
                                       "/object_approach/object_position")
        self.aim_enabled = _param_bool("~aim_before_direct", True)
        self.aim = AimBearingPolicy(AimBearingConfig(
            yaw_rate=float(G("~aim_yaw_rate", 0.7)),
            # How far the platform keeps turning after a burst stops. The follower's
            # calibration puts it at ~15 deg; each burst is aimed that much short.
            yaw_coast_rad=math.radians(float(G("~aim_yaw_coast_deg", 15.0))),
            # Being a few degrees off costs nothing: the camera's horizontal field of
            # view is ~76 deg, so the object is well in frame long before the nose is
            # exact -- and this is roughly the tightest a pulsed yaw can hold anyway.
            tolerance_rad=math.radians(float(G("~aim_tolerance_deg", 12.0))),
            min_burst_s=float(G("~aim_min_burst_s", 0.2)),
            max_burst_s=float(G("~aim_max_burst_s", 0.6)),
            settle_s=float(G("~aim_settle_s", 1.0)),
            # Long enough for the detector (a few Hz) to produce the ~n_confirm
            # consecutive hits the gate needs, with margin for a slow frame.
            look_s=float(G("~aim_look_s", 4.0)),
            timeout_s=float(G("~aim_timeout_s", 25.0))))
        self._object_xy = None
        # Latched once we have given up on aiming and re-targeted the goal at the
        # object itself: from then on this IS an object goal, so arriving at it means
        # what it always did (land there / sweep), and we must never aim again.
        self._escalated = False

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
        # Terminal land latch: once the target is reached we stop driving /cmd_vel and
        # request FINISH, and never servo again (a late detection cannot restart us).
        self._landing = False
        self._land_stop_published = False
        # True once we ever entered a visual approach (APPROACH/HOVER_LOCK). Lets
        # _begin_land label the terminal LAND accurately: arrival-land fires only from
        # SEARCH, so if this is still False the reach was "by A* alone", not a depth
        # reach -- even if a stale tracker box happened to supply a range this tick.
        self._entered_visual_approach = False

        # ── ROS I/O (publishers before subscribers) ──────────────────
        # Default is the drone's own topic (unchanged); object_approach.launch points it
        # at the cmd_vel_gate's input so the GO gate governs the visual servo too.
        self.cmd_vel_topic = str(G("~cmd_vel_topic", self.drone_ns + "/cmd_vel"))
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1)
        self.goal_pub = rospy.Publisher(self.goal_out_topic, Point, queue_size=1, latch=True)
        self.overlay_pub = (rospy.Publisher(self.overlay_topic, Image, queue_size=1)
                            if self.publish_overlay else None)
        # Narrates the target story (lock -> home -> lost -> give up) and the
        # mission end onto the shared thinking log the BEV viewer draws.
        self.thinker = Thinker("object_approach")

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
        rospy.Subscriber(self.object_position_topic, Point, self._object_position_cb,
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
        self._pose_yaw = self._yaw_of(msg.pose.orientation)

    def _pose_cb(self, msg):
        self._pose_xy = (float(msg.position.x), float(msg.position.y))
        self._pose_yaw = self._yaw_of(msg.orientation)

    @staticmethod
    def _yaw_of(q):
        """Heading (rad) from a pose quaternion, or None if it carries no rotation.

        A degenerate (all-zero) quaternion is not "yaw 0", it is a localization
        source that never filled the field in -- and silently reading it as 0 would
        aim the drone down whatever bearing that happens to be. Report it as absent
        instead so the aim is skipped rather than flown blind.
        """
        n = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if n < 1e-6:
            return None
        return quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))

    def _goal_cb(self, msg):
        # Remember the coordinate route's live goal (a bev click). Ignore our own
        # latched re-inject (same value) so we never fight the planner over it.
        self._goal_xy = (float(msg.x), float(msg.y))

    def _object_position_cb(self, msg):
        """The mission director's catalogued position for the hunted object.

        This is what makes the goal we are flying to a *staging* point rather than
        the object: knowing both, we can tell them apart, aim from one at the other,
        and fall back to the other if aiming fails. A new object (a live retarget)
        clears the escalation latch so the fresh target gets its own aim.
        """
        xy = (float(msg.x), float(msg.y))
        if xy == self._object_xy:
            return
        self._object_xy = xy
        self._escalated = False
        self.aim.reset()
        rospy.loginfo("object_approach: object position (%.2f, %.2f) -- will aim at it "
                      "from the goal before flying onto it", xy[0], xy[1])

    def _arrived_at_goal(self):
        """True once the drone is within ``arrive_radius_m`` of the known goal --
        the proxy for "the A*/NavDP route reached its goal" (there is no done
        topic). None goal or pose -> False (never scans without a known goal)."""
        if self._goal_xy is None or self._pose_xy is None:
            return False
        dx = self._pose_xy[0] - self._goal_xy[0]
        dy = self._pose_xy[1] - self._goal_xy[1]
        return math.hypot(dx, dy) <= self.arrive_radius_m

    def _aim_ready(self):
        """True when arriving at the current goal should AIM rather than end there.

        All of: aiming is enabled, we know where the object is, we have a heading to
        turn from, we have not already escalated, and the goal really is a DIFFERENT
        place from the object (a goal already on the object is not a staging point,
        so there would be nothing to aim at or escalate to).
        """
        if not self.aim_enabled or self._escalated:
            return False
        if self._object_xy is None or self._goal_xy is None:
            return False
        if self._pose_yaw is None:
            rospy.logwarn_throttle(
                10.0, "object_approach: cannot aim at the %s -- %s carries no "
                      "orientation, so there is no heading to turn from; falling back "
                      "to flying at its coordinate", self.target, self.pose_topic)
            return False
        dx = self._goal_xy[0] - self._object_xy[0]
        dy = self._goal_xy[1] - self._object_xy[1]
        return math.hypot(dx, dy) > self.arrive_radius_m

    def _heading_error_to_object(self):
        """Signed angle (rad) from the drone's heading to the object's bearing."""
        if self._object_xy is None or self._pose_xy is None or self._pose_yaw is None:
            return None
        bearing = math.atan2(self._object_xy[1] - self._pose_xy[1],
                             self._object_xy[0] - self._pose_xy[0])
        return normalize_angle(bearing - self._pose_yaw)

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

        # Terminal: we reached the target and are landing. Never servo again -- keep
        # driving the land sequence regardless of enable/track/detection state.
        if self._landing:
            self._drive_land()
            return

        # The drone does not know where it is: lost_localization owns /cmd_vel and
        # is manoeuvring blind to re-acquire an AprilTag. Go fully passive, exactly
        # as the follower does (waypoint_follower_node's MODE_RECOVERY check) --
        # publish nothing and, crucially, request nothing: our usual
        # visual_servoing claim would be re-asserted every 0.5s against the
        # recovery's own, and a last-write-wins arbiter would flip modes at ~2 Hz
        # with the drone alternating "back up" and "servo forward" while lost.
        # We keep _requested_mode, so when the recovery hands back we simply
        # re-claim visual_servoing on the next tick and the approach carries on.
        if self.current_demo_mode == MODE_RECOVERY:
            rospy.logwarn_throttle(
                5.0, "object_approach: passive -- lost_localization is recovering "
                     "the drone's position; the approach resumes when it releases")
            return

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

        # Arrival at the coordinate goal (no done topic exists) lets the FSM, when
        # still unconfirmed, AIM at the object from this vantage point (staged
        # approach), land there (~land_at_goal: reached "by A* alone"), or switch
        # from passive SEARCH to an active room SCAN (the legacy behaviour).
        arrived = self._arrived_at_goal()
        # Step the aim manoeuvre while it owns the mission, exactly as the servo is
        # stepped above: the FSM consumes its "finished" flag, we publish its command.
        # Only once the hand-off is GRANTED, though (_driving): its settle/look timers
        # measure real manoeuvring, and running them while the follower still owns
        # /cmd_vel would burn the look window without the drone having turned at all --
        # we would "look" down the bearing we arrived on and escalate having seen
        # nothing. A cold pose (no heading error) pauses it for the same reason.
        aim_dec = None
        if self.fsm.state == AIM and self._driving():
            err = self._heading_error_to_object()
            if err is not None:
                aim_dec = self.aim.update(err, dt)
        # Feed the metric range so the FSM can commit to the terminal LAND at
        # land_range_m. res.range_m already respects ~use_depth (None when depth is
        # off) and is None without a valid track, so LAND needs working depth.
        fsm_range = res.range_m if res is not None else None
        dec = self.fsm.update(confirmed=confirmed, track_valid=track_valid,
                              at_target=at_target, dt=dt, arrived_at_goal=arrived,
                              range_m=fsm_range, aim_ready=self._aim_ready(),
                              aim_done=bool(aim_dec is not None and aim_dec.finished))
        self._last_dec, self._last_res, self._last_track = dec, res, track
        self._narrate_mode(dec.mode)
        if dec.mode in ("APPROACH", "HOVER_LOCK"):
            self._entered_visual_approach = True

        if dec.mode == LAND:
            # Reached the object: end the mission by landing. Stop closing (so the
            # platform's inertial actuation cannot drift us into it) and fire the
            # FINISH land sequence. Latched terminal from here on.
            self._begin_land()
            return

        if dec.escalate_goal:
            # Aimed at where the object should be and still did not see it. Give up
            # on the standoff and re-target the goal at the object's own coordinate,
            # so the planner flies the last leg it had deliberately been held back
            # from -- and arriving THERE means what it always did (land / sweep).
            self._escalate_to_object()

        if dec.reset_acquisition:
            with self._lock:
                self.gate.reset()
                self.tracker.reset()
                self._confirmed = False
            self.thinker.say("Could not find the %s, returning to A* navigation"
                             % self.target, category="object", level="warn")
            # Lost the object for good: re-assert the goal so the planner flies us
            # back to the last/initial goal rather than stalling where we gave up.
            self._reinject_goal()

        # The sweep and the aim are stateful: keep each reset unless it is the one
        # running, so every episode starts clean (the sweep from a look straight
        # ahead, the aim from a stop that arrests the arrival motion).
        if dec.mode != SCAN:
            self.scan.reset()
        if dec.mode != AIM:
            self.aim.reset()

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
        elif dec.mode == AIM:                       # arrived, turning to look at it
            # aim_dec is None on the tick the FSM ENTERS aim (the policy had not run
            # yet) and whenever the pose goes cold mid-turn; a stop is the right
            # command for both -- it is also how the manoeuvre opens.
            c = None if aim_dec is None else aim_dec.command
            phase = "start" if aim_dec is None else aim_dec.phase
            self._publish_cmd(0.0 if c is None else c.x, 0.0 if c is None else c.y,
                              0.0 if c is None else c.yaw_rate, "aim:%s" % phase)
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

    def _narrate_mode(self, mode):
        """Narrate what the FSM just decided about the target (see MODE_THOUGHTS).

        Called every tick from the one decision funnel: the Thinker's gate drops the
        unchanged line, so this emits once per transition without any edge state of
        its own, and re-emits when the target itself changes (~target_topic).
        """
        thought = MODE_THOUGHTS.get(mode)
        if thought is not None:
            self.thinker.say(thought[0] % self.target, category="object",
                             level=thought[1])

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

    def _begin_land(self):
        """Commit to the terminal land sequence (idempotent).

        The drone reached the object -- either the depth range to the locked target
        fell within ``land_range_m`` (visual reach), or the coordinate route arrived
        at the goal still unconfirmed with ``land_at_goal`` on (reached "by A* alone").
        Two things then happen, in order and permanently:
          1. STOP sending motion. The servo streams ``/cmd_vel`` pulses and the
             platform coasts on them (discrete + inertial actuation), so going quiet
             is not enough -- we publish one clean zero-stop, then fall silent so the
             twist bridge's stale-twist watchdog holds the drone. The co-running route
             follower also goes passive on the ``finish`` demo mode (it treats FINISH
             like ``visual_servoing``), so no node keeps ``/cmd_vel`` alive.
          2. LAND. ``/xtend/cmd_nav`` is not bridged into this ROS1 container, so the
             only land path from here is the demo-mode handshake: requesting FINISH
             makes the ROS2 demo manager send stop -> land -> disarm on
             ``/xtend/cmd_nav`` (see xtend_drone_demo_manager).
        Once ``_landing`` is latched the node never drives ``/cmd_vel`` again, so a
        late detection cannot restart the approach after we decided to land.
        """
        if not self._landing:
            self._landing = True
            rng = None if self._last_res is None else self._last_res.range_m
            # Report the trigger that fired. A visual reach passes through
            # APPROACH/HOVER_LOCK and has a depth range; arrival-land fires straight
            # from SEARCH. Key off _entered_visual_approach (not merely "is a range
            # present"), so a stale tracker box that supplies a range on the arrival
            # tick cannot mislabel a coordinate reach as a depth reach.
            if self._entered_visual_approach and self.land_range_m is not None \
                    and rng is not None:
                reason = "range=%.2f m <= land_range=%.2f m" % (rng, self.land_range_m)
                thought = ("Reached the %s -- stopping and landing to finish the "
                           "mission" % self.target)
            else:
                reason = "arrived at the coordinate goal (reached by A* alone)"
                thought = ("Arrived at the goal without ever seeing the %s -- landing "
                           "here to finish the mission" % self.target)
            rospy.logwarn(
                "object_approach: REACHED TARGET (%s) -- stopping commands and "
                "requesting FINISH (land)", reason)
            # The node falls silent from here (it only re-requests FINISH), so this is
            # the mission's last word in the operator's log.
            self.thinker.say(thought, category="mission")
        self._drive_land()

    def _drive_land(self):
        """Run the latched land sequence each tick: brake ``/cmd_vel`` once, then keep
        requesting FINISH until the demo manager grants it. Never publishes motion."""
        if not self._land_stop_published:
            # Clear the pulse/gait state so the final /cmd_vel value is a true zero
            # (not a lingering min-burst pulse), then fall silent for the watchdog.
            self.shaper.reset()
            self.gait.reset()
            self._publish_cmd(0.0, 0.0, 0.0, "land_stop")
            self._land_stop_published = True
        self._request_mode(MODE_FINISH)

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

    def _escalate_to_object(self):
        """Re-target the coordinate goal at the object itself (idempotent).

        The staged approach's fallback. We stood off at the vantage point, turned to
        face where the catalogue says the object is, looked for ``~aim_look_s`` -- and
        the detector still never confirmed it. So the standoff has bought us all it
        can: publish the object's own (x, y) as the coordinate goal and let the
        planner fly the last leg. From here the mission is exactly the unstaged one
        (arrive there -> land with ~land_at_goal, or sweep), and the latch makes sure
        we never aim again for this target -- the goal now IS the object.
        """
        if self._object_xy is None or self._escalated:
            return
        self._escalated = True
        self._goal_xy = (self._object_xy[0], self._object_xy[1])
        self.goal_pub.publish(Point(x=float(self._object_xy[0]),
                                    y=float(self._object_xy[1]), z=0.0))
        rospy.logwarn("object_approach: aimed at the %s from the goal and never saw "
                      "it -- re-targeting the goal at its own position (%.2f, %.2f)",
                      self.target, self._object_xy[0], self._object_xy[1])
        self.thinker.say(
            "Looked where the %s should be and could not see it -- flying to its "
            "recorded position (%.2f, %.2f) instead"
            % (self.target, self._object_xy[0], self._object_xy[1]),
            category="object", level="warn")

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
        L("  nav goal  = %s  (in=%s out=%s, arrive<%.2fm)",
          self._goal_xy, self.goal_in_topic, self.goal_out_topic, self.arrive_radius_m)
        if self.aim_enabled:
            L("  staged    = on: arriving at the goal with a known object position "
              "-> AIM at its bearing (+-%.0f deg), look %.1fs, then fly to it",
              math.degrees(self.aim.cfg.tolerance_rad), self.aim.cfg.look_s)
            L("              object position in = %s  (%s)", self.object_position_topic,
              "unset -- no staging until the director publishes it"
              if self._object_xy is None else str(self._object_xy))
        else:
            L("  staged    = off (~aim_before_direct false): the goal is flown "
              "straight through, arrival -> land/scan as configured")
        L("  HUD out   = %s  @ %.1f Hz (%s)", self.overlay_topic, self.viz_hz,
          "on" if self.overlay_pub is not None else
          ("off" if _HAVE_OVERLAY else "overlay import failed: %s" % _OVERLAY_IMPORT_ERR))
        L("  intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
          self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
          self.intr.width, self.intr.height)
        if self.land_range_m is not None:
            L("  land(depth) = reach range <= %.2f m (x%d ticks) -> STOP + FINISH "
              "(stop->land->disarm via demo manager)",
              self.land_range_m, self.land_confirm_ticks)
            L("  success   = target reached within %.2f m -> land the drone",
              self.land_range_m)
        else:
            L("  land(depth) = disabled (~land_range_m <= 0): no visual-reach land")
        if self.land_at_goal:
            L("  land(goal)  = arrive within %.2fm of the goal (x%d ticks), unconfirmed "
              "-> STOP + FINISH (reached by A* alone; pose-based, no depth)",
              self.arrive_radius_m, self.arrive_land_confirm_ticks)
        else:
            L("  land(goal)  = disabled (~land_at_goal false): arrival -> room SCAN")
        if self.land_range_m is None and not self.land_at_goal:
            L("  success   = target centred & within target_range/area (hover-lock)")
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
#   terminal land (reach the object -> stop + land, ends the mission). TWO independent
#       triggers, either/both/neither:
#     VISUAL reach (depth): ~land_range_m (1.0, depth range at/below which the object is
#       "reached"; needs ~use_depth; <= 0 disables) ~land_confirm_ticks (3, consecutive
#       in-range ticks so one depth glitch cannot land). Fires during APPROACH/HOVER_LOCK.
#     COORDINATE reach (pose): ~land_at_goal (false; true = arriving within ~arrive_radius_m
#       of the goal still unconfirmed for ~arrive_land_confirm_ticks (5) ticks is itself a
#       LAND -- the object was reached "by A* alone", so the goal is its location; land
#       there instead of the arrival->SCAN room sweep). POSE-based, needs NO depth, so it
#       works with ~use_depth false / ~land_range_m<=0. Fires from SEARCH only.
#     On EITHER trigger the node stops driving /cmd_vel (the twist bridge's watchdog then
#       holds the drone) and requests the ~demo_mode_request_topic 'finish' mode; the ROS2
#       xtend_drone_demo_manager turns FINISH into stop -> land -> disarm on /xtend/cmd_nav
#       (cmd_nav is NOT bridged into this container, so demo_mode_request is the only land
#       path from here).
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
#   goal (arrival -> AIM/SCAN, re-inject on give-up): ~goal_in_topic (/waypoint_nav/goal)
#       ~goal_out_topic (/waypoint_nav/goal) ~goal_x/~goal_y (initial goal, unset =
#       none) ~arrive_radius_m (0.6)
#   staged approach (stand off at the goal, AIM at the object, escalate only if that
#       fails). Arms itself the moment an object position arrives; without one the
#       node behaves exactly as it did before:
#       ~object_position_topic (/object_approach/object_position, geometry_msgs/Point
#         -- the object's catalogued world (x,y), published by mission_director)
#       ~aim_before_direct (true; false = fly the goal straight through, no aiming)
#       ~aim_yaw_rate (0.7 rad/s, the burst magnitude the platform turns on)
#       ~aim_yaw_coast_deg (15.0, how far it keeps turning after a burst -- each
#         burst is aimed this much SHORT and the coast lands it)
#       ~aim_tolerance_deg (12.0, "on the bearing"; the camera's ~76 deg horizontal
#         FOV means a few degrees off still has the object well in frame)
#       ~aim_min_burst_s (0.2, below this the yaw deadband eats the command)
#       ~aim_max_burst_s (0.6, angle swept open-loop before re-measuring)
#       ~aim_settle_s (1.0, stop between bursts so the coast finishes first)
#       ~aim_look_s (4.0, hold still on the bearing -- size against the detector rate
#         and ~n_confirm) ~aim_timeout_s (25.0, hard cap so a platform that will not
#         turn still escalates instead of aiming forever)
#       On the aim failing, the node publishes the object's (x,y) on ~goal_out_topic
#       and the mission continues exactly as the unstaged one (arrive -> land/scan).
#       NEEDS a pose with orientation: a localization source that publishes no
#       quaternion cannot be aimed from, and the node says so and skips to the
#       coordinate rather than turning down an unknown bearing.
#   scan-at-goal sweep: ~scan_yaw_rate (0.4) ~scan_rotate_s (1.2) ~scan_pause_s (1.2)
#       ~scan_direction (+1 CCW / -1 CW) ~scan_forward_speed (0.0 = in place)
#       ~scan_forward_s (0.0) ~scan_bursts_before_move (8)
#   HUD overlay: ~publish_overlay (true) ~overlay_topic (/object_approach/overlay)
#       ~viz_hz (10.0)  [sensor_msgs/Image bgr8; view with target_lock_viewer_node]
#   thinking log (see thinking.py): ~thinking (true, false silences this node's
#       narration) ~thinking_topic (/nav/thinking) ~thinking_echo (true, also mirror
#       each thought to rosout)
# ============================================================================
