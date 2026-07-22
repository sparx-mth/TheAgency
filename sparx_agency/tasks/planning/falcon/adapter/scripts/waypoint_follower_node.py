#!/usr/bin/env python3
"""
waypoint_follower_node.py -- ROS1 adapter: nav_msgs/Path -> /cmd_vel (X+YAW only).

Thin glue around the ROS-free follower in
``sparx_agency.core.planning.trackers.waypoint_follower.WaypointFollower``. All
of the navigation logic -- the YAW_ALIGN/ADVANCE/BRAKE/DONE state machine, path
re-anchoring, yaw-lead inertia compensation, the platform invariant (vy=vz=0,
vx=0 OR wz=0), slew and saturation -- lives in core and is unit tested without
ROS. This node owns ONLY ROS / platform concerns:

  - rosparams -> WaypointFollowerParams + topics,
  - platform bring-up: WAIT_POSE -> TAKING_OFF -> HOVER_SETTLE -> WAIT_PATH,
  - the DemoMode handshake (the core asks for a YAW or FORWARD axis; this node
    requests the matching ROS DemoMode and only lets the follower move once the
    flight controller confirms it),
  - VISUAL_SERVOING passivity (another node owns /cmd_vel then),
  - sensor-gate freeze, the startup hold, the optional cmd_vel JSONL log,
  - assembling geometry_msgs/Twist with linear.y = linear.z = 0 hardwired.

Drop-in replacement for the legacy falcon_adapter ``waypoint_follower.py``:
identical topics, message types and rosparam names.

  in   ~drone_ns + /gt_pose (Pose)
  in   ~drone_ns + /state   (Int8)
  in   ~path_topic (Path)            /path/waypoints
  in   ~demo_mode_topic (String)     /xtend/demo_mode
  out  ~drone_ns + /cmd_vel (Twist)
  out  ~drone_ns + /takeoff (Empty)
  out  /sensor_gate/freeze (Bool)
  out  ~demo_mode_request_topic (String)  /xtend/demo_mode_request
  out  ~thinking_topic (String)      /nav/thinking (narration; see thinking.py)

See the file footer for the full rosparam list.
"""
import datetime
import json
import math
import os

import rospy
import tf.transformations as tft
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, Empty, Float32, Int8, String

from sparx_agency.core.common.types import KinematicLimits, Pose2D
from sparx_agency.core.localization.pose_estimator import (
    PoseEstimatorParams,
    WindowedPoseEstimator,
)
from sparx_agency.core.planning.trackers.waypoint_follower import (
    ControlAxis,
    FollowerState,
    MotionModelParams,
    WaypointFollower,
    WaypointFollowerParams,
    predict_trajectory,
    prediction_score,
)
from sparx_agency.core.planning.trackers.multi_axis_follower import (
    MultiAxisFollower,
    MultiAxisFollowerParams,
)
from sparx_agency.core.planning.trackers.multi_axis_follower import (
    predict_trajectory as mx_predict_trajectory,
)
from sparx_agency.core.planning.trackers.roll_assist_follower import (
    CrossTrackRollCorrector,
    CrossTrackRollParams,
    RollAssistFollower,
)
from sparx_agency.core.planning.trackers.rotation_supervisor import (
    RotationReobserveSupervisor,
    RotationSupervisorParams,
)
from thinking import Thinker
# Global (not ~) param naming the shared per-tick certainty CSV; '' disables it.
# Global for the same reason /thinking/log_path is: only drift_pid narrates here
# today, but a global name leaves room for another controller to log to it too.
CERTAINTY_LOG_PATH_PARAM = "/certainty/log_path"
# NOTE: the pure_pursuit imports (core HermiteSmoother / PurePursuitTracker and the
# sibling pure_pursuit_follower.py) are deliberately deferred into
# _build_pure_pursuit() so they are loaded ONLY when ~controller:=pure_pursuit.
# The waypoint and multi_axis controllers must keep starting even if the
# pure-pursuit helper/module is not deployed in this container.

# DemoMode payloads bridged over /xtend/demo_mode(_request). The core control
# axis maps onto the platform's flight modes; visual_servoing is platform-only.
AXIS_TO_MODE = {ControlAxis.YAW: "turning", ControlAxis.FORWARD: "fly_straight"}
MODE_VISUAL_SERVOING = "visual_servoing"
# Terminal land mode: the demo manager sends stop -> land -> disarm on /xtend/cmd_nav
# and object_approach has stopped driving /cmd_vel. The follower must stay passive
# here too (like visual_servoing) so it does not re-drive /cmd_vel toward its route
# goal while the drone is stopping/landing (which would defeat the land's stop and
# fight the mode arbiter). Once the mission is landing there is nothing left to fly.
MODE_FINISH = "finish"
# Lost-localization recovery: our pose has gone cold (no AprilTag in view), so
# lost_localization_node has taken the drone off us and is backing it up / climbing
# / sweeping to re-acquire one. Stay passive (like visual_servoing) for two
# reasons: there must be exactly one publisher on cmd_vel, and our own navigation
# is meaningless right now anyway -- every pose we hold is stale, so any command we
# computed would be aimed at where the drone was, not where it is. The recovery
# actively requests fly_straight back when it releases; we resume from there.
MODE_RECOVERY = "recovery"
# "Forward flight" mode, requested during every stop (and at bring-up): it tells
# the FC to hold heading (stop rotating) while we command zero velocity, so the
# platform is stable for a clean voxel update -- "forward mode but not flying".
MODE_FORWARD = AXIS_TO_MODE[ControlAxis.FORWARD]
# "Turning" mode, held through the physical yaw coast at the end of a turn so the
# mode-authoritative depth gate stays frozen until the drone has actually stopped
# (the inertia frames right after a turn are the worst to fuse). See _ctrl_loop.
MODE_TURNING = AXIS_TO_MODE[ControlAxis.YAW]


#: Every controller ~controller accepts. An unknown value is an error, not a
#: silent fallback to the one-axis follower.
_CONTROLLERS = frozenset(
    {"waypoint", "multi_axis", "pure_pursuit", "roll_assist", "drift_pid"})


class _Bringup:
    """Node-side platform bring-up states (the core owns navigation states)."""
    WAIT_POSE = "WAIT_POSE"
    TAKING_OFF = "TAKING_OFF"
    HOVER_SETTLE = "HOVER_SETTLE"
    MAP_SETTLE = "MAP_SETTLE"      # forward mode + stopped + first voxel update
    WAIT_PATH = "WAIT_PATH"
    RUNNING = "RUNNING"


# What the operator hears as bring-up advances; narrated from the single _enter()
# funnel. MAP_SETTLE is absent on purpose: that state IS a stop for the map, so it
# is narrated as a map thought by _begin_map_wait (the same line every later stop
# uses) rather than twice in two voices.
_BRINGUP_THOUGHTS = {
    _Bringup.WAIT_POSE: "Waiting for localization before I move",
    _Bringup.TAKING_OFF: "Taking off",
    _Bringup.HOVER_SETTLE: "Hovering to settle before I start navigating",
    _Bringup.WAIT_PATH: "Waiting for a route from the planner",
    _Bringup.RUNNING: "Flying the route",
}
# The stop-for-the-map line. One text for both the bring-up warm-up and every
# mid-flight frozen-turn re-observation: to an operator it is the same decision.
_THOUGHT_MAP_WAIT = "Stopping for a voxel map update"


class WaypointFollowerNode:
    def __init__(self):
        rospy.init_node("waypoint_follower")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "/simple_drone")

        self.follower = WaypointFollower(WaypointFollowerParams(
            vel_x=float(G("~vel_x", 0.25)),
            yaw_rate=float(G("~yaw_rate", 0.7)),
            pos_radius=float(G("~pos_acquisition_radius", 0.35)),
            yaw_settle=float(G("~yaw_settle_thresh", 0.05)),
            yaw_drift_thresh=float(G("~yaw_drift_thresh", 0.40)),
            skip_yaw_thresh=float(G("~skip_yaw_thresh", 0.25)),
            vx_brake_thresh=float(G("~vx_brake_thresh", 0.05)),
            brake_timeout_s=float(G("~brake_timeout_s", 2.0)),
            passed_bearing_rad=math.radians(float(G("~passed_bearing_deg", 100.0))),
            # Pulse -> settle -> re-measure yaw (slow turn, inertia, jumpy yaw).
            yaw_settle_dwell_s=float(G("~yaw_settle_dwell_s", 0.8)),
            yaw_settle_eps=float(G("~yaw_settle_eps", 0.05)),
            min_motion_ticks=int(G("~min_motion_ticks", 2)),
            yaw_coast_rad=math.radians(float(G("~yaw_coast_deg", 15.0))),
            # Per-burst increment: split a big turn into ~this-size chunks, each
            # followed by a stop + voxel update + re-measure (not one big sweep).
            yaw_burst_max_rad=math.radians(float(G("~yaw_burst_max_deg", 25.0))),
            yaw_burst_max_ticks=int(G("~yaw_burst_max_ticks", 30)),
            # Master on/off for freeze-during-rotation (shared with the holonomic
            # supervisor via ~freeze_on_rotation). False keeps the map live in turns.
            freeze_on_rotation=bool(G("~freeze_on_rotation", True)),
            # Angle-gated map freeze: a turn LARGER than this freezes the voxel map
            # (and forces the post-turn re-observation below); a smaller correction
            # stays live so the map keeps updating through a gentle nudge.
            freeze_yaw_thresh_rad=math.radians(float(G("~freeze_yaw_thresh_deg", 20.0))),
            # Fresh voxel updates a frozen turn re-observes while stopped before it
            # advances/turns again (>=2: never act on a map built before the turn).
            settle_map_updates=int(G("~settle_map_updates", 2)),
            # Graded-pulse / mid-burst-feedback / anti-deadlock yaw upgrades. All
            # default OFF (inert); the launch enables them (needs ctrl_rate_hz:=10
            # for the 4 deg/tick -> 8 deg-min / 24 deg-cap numbers).
            yaw_graded_pulses=bool(G("~yaw_graded_pulses", False)),
            yaw_burst_grade_max_ticks=int(G("~yaw_burst_grade_max_ticks", 6)),
            yaw_settle_dwell_per_tick=float(G("~yaw_settle_dwell_per_tick", 0.0)),
            yaw_burst_live_feedback=bool(G("~yaw_burst_live_feedback", False)),
            yaw_fb_reach_rad=math.radians(float(G("~yaw_fb_reach_deg", 0.0))),
            yaw_fb_confirm_ticks=int(G("~yaw_fb_confirm_ticks", 2)),
            yaw_max_reversals=int(G("~yaw_max_reversals", 0)),
            yaw_accept_growth_rad=math.radians(float(G("~yaw_accept_growth_deg", 0.0))),
            # Gentle predictive ADVANCE gate. The legacy ~yaw_acquisition_radius
            # (previously read by nobody) now feeds the cross-track tolerance.
            yaw_capture_tol_m=float(G("~yaw_acquisition_radius", 0.20)),
            yaw_acquire_max=math.radians(float(G("~yaw_acquire_max_deg", 35.0))),
            yaw_lead_pct=float(G("~yaw_lead_pct", 10.0)),
            vel_xy_sat=float(G("~vel_xy_sat", 1.25)),
            yaw_rate_sat=float(G("~yaw_rate_sat", 2.4)),
            accel_limit=float(G("~accel_limit", 1.5)),
            yaw_accel_limit=float(G("~yaw_accel_limit", 3.5)),
            forward_only=bool(G("~forward_only", False)),
        ))

        # ── Controller selection ─────────────────────────────────────
        # ~controller picks the path tracker; falling back is a one-line param.
        #   "waypoint"     (default) one-axis follower built above (X advance OR
        #                  yaw, never both -- the "stupid" controller),
        #   "multi_axis"   combined forward + lateral + yaw, crabbing (ROLL) for
        #                  small offsets and yawing only past a deadband,
        #   "pure_pursuit" splines the path (Hermite) and tracks it with Pure
        #                  Pursuit on a moving lookahead (holonomic),
        #   "roll_assist"  the one-axis waypoint follower UNCHANGED (same align ->
        #                  advance nav, discrete yaw, freeze/map gates, handshake),
        #                  with a cross-track ROLL (linear.y) correction layered on
        #                  top that only pulls the drone back onto its trajectory
        #                  when it drifts sideways -- full while advancing, weak
        #                  while turning, small while holding.
        #   "drift_pid"    continuous multi-axis tracker whose three PID loops
        #                  LEARN the standing per-axis drift (their integral IS the
        #                  drift estimate) instead of only pushing back against it,
        #                  behind a per-axis + combined force envelope, scheduled by
        #                  localization confidence, with reflexes for walls the
        #                  camera cannot see. See core/.../trackers/drift_pid.
        # Altitude is never commanded by any of them (vz = 0). The one-axis
        # follower is always constructed above; multi_axis/pure_pursuit replace the
        # handle, roll_assist WRAPS it. ``_holonomic`` groups the two continuous
        # trackers that use the rotation supervisor + no per-axis handshake;
        # ``_lateral`` groups everyone that drives linear.y (adds roll_assist,
        # which still uses the one-axis handshake) so the loop branches once.
        self.controller_kind = str(G("~controller", "waypoint")).strip().lower()
        if self.controller_kind not in _CONTROLLERS:
            # Previously an unknown string fell silently through to the one-axis
            # follower, so a typo flew a controller nobody selected.
            raise ValueError(
                "~controller=%r is not a known controller. Choose one of: %s"
                % (self.controller_kind, ", ".join(sorted(_CONTROLLERS))))
        self._holonomic = self.controller_kind in (
            "multi_axis", "pure_pursuit", "drift_pid")
        self._lateral = self.controller_kind in (
            "multi_axis", "pure_pursuit", "roll_assist", "drift_pid")
        # drift_pid consumes the localization confidence signals; nothing else does.
        self.quality = None
        self.drift_telemetry = None
        self.certainty_log = None
        self._last_quality = None
        # Altitude hold: defaults BEFORE the controller dispatch below, because
        # _build_drift_pid assigns the real AltitudeHold -- a default set after
        # it would silently overwrite the built hold back to None.
        self.alt_hold = None
        self._alt_vz = 0.0
        # DemoMode the holonomic controllers request (best effort) and, if
        # ~mx_require_mode, gate on. The platform now accepts multi-axis commands,
        # so the per-axis handshake of the legacy path does not apply to them.
        self.multi_axis_demo_mode = str(
            G("~mx_demo_mode", MODE_FORWARD)).strip().lower()
        self.multi_axis_require_mode = bool(G("~mx_require_mode", False))
        if self.controller_kind == "multi_axis":
            self.follower = self._build_multi_axis(G)
        elif self.controller_kind == "pure_pursuit":
            self.follower = self._build_pure_pursuit(G)
        elif self.controller_kind == "roll_assist":
            self.follower = self._build_roll_assist(G)
        elif self.controller_kind == "drift_pid":
            self.follower = self._build_drift_pid(G)

        # drift_pid coordinates its own turn pitch inside the control law
        # (~dp_turn_pitch_bias, a floor while the yaw axis is active), so the
        # publish-time ~yaw_pitch_bias must stay out of its way: a second,
        # invisible forward injection would corrupt the envelope's slew memory,
        # the effectiveness estimate and the certainty log's mirror of what
        # actually flew.
        self._core_owns_pitch_bias = (self.controller_kind == "drift_pid")

        # ── Rotation supervisor (holonomic controllers only) ─────────
        # The one-axis follower freezes + re-observes inside its own state machine.
        # The CONTINUOUS holonomic trackers (multi_axis, pure_pursuit) have no
        # discrete turn, so this supervisor imposes the same discipline on them:
        # freeze the map throughout any turn (rate-based) and stop every
        # ~rot_reobserve_every_deg of rotation to re-observe >=settle_map_updates
        # voxels while stopped. ~freeze_on_rotation is the shared master switch.
        self.rot_sup = RotationReobserveSupervisor(RotationSupervisorParams(
            enabled=bool(G("~freeze_on_rotation", True)),
            wz_turn_on=float(G("~rot_wz_turn_on", 0.20)),
            wz_turn_off=float(G("~rot_wz_turn_off", 0.10)),
            turn_off_ticks=int(G("~rot_turn_off_ticks", 3)),
            reobserve_every_rad=math.radians(float(G("~rot_reobserve_every_deg", 25.0))),
            settle_eps=float(G("~yaw_settle_eps", 0.05)),
            settle_dwell_s=float(G("~yaw_settle_dwell_s", 0.8)),
            settle_map_updates=int(G("~settle_map_updates", 2)),
            max_coast_s=float(G("~rot_max_coast_s", 2.0)),
            map_wait_timeout_s=float(G("~map_wait_timeout_s", 3.0))))

        # ── Pose estimator ───────────────────────────────────────────
        # Fuses the ~10 Hz noisy /gt_pose stream with the command being executed,
        # so the follower (run at ctrl_rate_hz) sees a DENOISED pose + yaw-rate:
        # drift-rejected when stopped, true-rate while turning. ~use_pose_estimator
        # false feeds the raw pose (today's behaviour). Ingest stays at the full
        # /gt_pose rate (decoupled from the control loop).
        self.use_pose_estimator = bool(G("~use_pose_estimator", False))
        self.estimator = WindowedPoseEstimator(PoseEstimatorParams(
            window_s=float(G("~est_window_s", 0.6)),
            min_samples=int(G("~est_min_samples", 2)),
            max_buffer_s=float(G("~est_max_buffer_s", 1.5)),
            wz_cmd_eps=float(G("~est_wz_cmd_eps", 0.05)),
            vx_cmd_eps=float(G("~est_vx_cmd_eps", 0.03)),
            settle_wz_eps=float(G("~yaw_settle_eps", 0.05)),
            wz_ff_ref=float(G("~yaw_rate", 0.7)),
            vx_ff_ref=float(G("~vel_x", 0.3)),
            ff_blend_min=float(G("~est_ff_blend_min", 0.15)),
            ff_blend_max=float(G("~est_ff_blend_max", 0.6)),
            dropout_s=float(G("~est_dropout_s", 0.25)),
            max_coast_s=float(G("~est_max_coast_s", 1.0)),
            fresh_tau_s=float(G("~est_fresh_tau_s", 0.3)),
        ))
        self._last_pub_vx = 0.0      # cmd executed last tick -> estimator feed-forward
        self._last_pub_wz = 0.0
        self._last_pub_vy = 0.0      # lateral feed-forward (multi-axis only; 0 otherwise)

        # Takeoff / settle (platform bring-up).
        self.auto_takeoff = bool(G("~auto_takeoff", True))
        self.takeoff_z = float(G("~takeoff_z", 1.0))
        self.takeoff_z_thresh = float(G("~takeoff_z_thresh", 0.5))
        self.takeoff_timeout = float(G("~takeoff_timeout", 30.0))
        self.takeoff_retry_sec = float(G("~takeoff_retry_sec", 1.0))
        self.hover_settle_sec = float(G("~hover_settle_sec", 2.5))

        # Loop rates + gating.
        self.ctrl_rate_hz = float(G("~ctrl_rate_hz", 5.0))
        self.status_hz = float(G("~status_hz", 1.0))
        self.freeze_during_yaw = bool(G("~freeze_during_yaw", True))
        # Hold 'turning' demo-mode through the physical yaw coast (until wz~0) so
        # the mode-authoritative voxel gate stays frozen over the post-turn inertia
        # settle, not just the commanded burst. Off = request forward the instant
        # YAW_SETTLE begins (legacy; relies on ~resume_settle_sec on the gate node).
        self.freeze_through_coast = bool(G("~freeze_through_coast", True))
        # Default 0: bring-up motion is gated event-driven by the MAP_SETTLE voxel
        # warm-up, not by a fixed time hold. Set >0 only to force an extra blanket
        # hold after node start.
        self.startup_hold_sec = float(G("~startup_hold_sec", 0.0))
        self.frame_id = G("~frame_id", "world")

        # Trajectory prediction (rollout) -> /path/predicted for the BEV viewer
        # and the planner's dynamics-aware collision check.
        self.predict_hz = float(G("~predict_hz", 2.0))
        self.predict_horizon_s = float(G("~predict_horizon_s", 30.0))
        self.predicted_path_topic = G("~predicted_path_topic", "/path/predicted")
        self.predicted_score_topic = G("~predicted_score_topic",
                                       "/path/predicted_score")
        # Pure-pursuit visualization: the smooth (splined) trajectory and the
        # current lookahead point the tracker is aiming at, for the BEV viewer.
        self.smooth_path_topic = G("~smooth_path_topic", "/path/smooth")
        self.lookahead_topic = G("~lookahead_topic", "/path/lookahead")
        self._motion = MotionModelParams(
            yaw_tau_s=float(G("~predict_yaw_tau_s", 0.5)),
            vx_tau_s=float(G("~predict_vx_tau_s", 0.3)))
        self._path_pts = []

        # Map-settle gate: at bring-up and during every stop, hold position in
        # forward mode and wait for a fresh BEV (voxel) update before moving on,
        # so the map reflects post-stop data (never the turn itself). Proceeds
        # after a timeout if none arrives, so a mapping stall cannot hang flight.
        self.require_map_update = bool(G("~require_map_update", True))
        self.map_update_topic = G("~map_update_topic", "/falcon/bev_2d")
        self.map_wait_timeout_s = float(G("~map_wait_timeout_s", 3.0))
        self.mapsettle_min_s = float(G("~mapsettle_min_s", 0.5))
        # Bring-up requires this many fresh voxel updates before the FIRST move,
        # so the map is genuinely seeded (not a single frame) even if the goal is
        # behind the drone. Running re-stops still need only one (see _map_ready).
        self.mapsettle_min_updates = int(G("~mapsettle_min_updates", 2))
        self._bev_count = 0           # ++ on every BEV (voxel) update received
        self._await_map = False       # in a stop, waiting for a fresh update
        self._await_baseline = 0      # _bev_count when the current stop unfroze
        self._await_t0 = rospy.Time(0)

        # DemoMode handshake.
        self.demo_mode_topic = G("~demo_mode_topic", "/xtend/demo_mode")
        self.demo_mode_request_topic = G("~demo_mode_request_topic",
                                         "/xtend/demo_mode_request")
        self.request_repeat_sec = float(G("~request_repeat_sec", 0.5))
        self.request_timeout_sec = float(G("~request_timeout_sec", 5.0))

        # Optional cmd_vel JSONL logger ({ts} expands at startup).
        self._log_file = self._open_log(G("~cmd_log_path",
                                           "/home/falcon/runs/cmd_log_{ts}.jsonl"))

        # ── Runtime state ──
        self.state = _Bringup.WAIT_POSE
        self.t_state = rospy.Time.now()
        self.cur_pose = None          # geometry_msgs/Pose
        self.drone_state = None
        self.have_path = False
        self.last_freeze = None
        self.last_takeoff = rospy.Time(0)
        self.takeoff_count = 0
        self.current_demo_mode = None
        self.requested_demo_mode = None
        self._last_request_pub_t = rospy.Time(0)
        self._request_entered_t = rospy.Time(0)
        self._node_start_t = rospy.Time.now()
        self.dt = 1.0 / self.ctrl_rate_hz

        # Command-commitment on the twist sent to the drone: each motion command
        # (a turn, or forward) must be emitted at least ~cmd_commit_ticks times in
        # a row before switching to a different command -- a single-tick command
        # can't overcome the motor deadband, so the drone would ignore a lone turn
        # then stop. A premature switch re-emits the under-committed motion until
        # it reaches the floor, then the new command takes effect. The command
        # "category" is the sign of (vx, wz); (0,0) is a stop and is never held.
        # Applies to both FALCON and NavDP (both flow through this follower).
        self.cmd_commit_ticks = int(G("~cmd_commit_ticks", 2))
        self.cmd_stop_eps = float(G("~cmd_stop_eps", 1e-3))
        self._cmd_cat = None          # last emitted command category (sx, sz)
        self._cmd_run = 0             # consecutive emits of that category
        self._cmd_vx = self._cmd_wz = 0.0   # last motion twist, repeated if held

        # Turn-in-place PITCH bias: a pure-yaw twist (wz only, vx=0) is a weak
        # command on this platform -- it yaws without ever tilting, so the turn
        # starts late and coasts sloppily. Riding a small forward PITCH along with
        # the yaw gives the turn something to bite on. Applied only when the twist
        # is otherwise a pure yaw, so it never adds to a commanded forward run;
        # set 0.0 to restore the old pure-yaw behaviour.
        self.yaw_pitch_bias = float(G("~yaw_pitch_bias", 0.05))

        # Narrate what this node DECIDES onto /nav/thinking: which waypoint it is
        # aligning to or flying at, why it stopped, what it is waiting for.
        self.thinker = Thinker("waypoint_follower")
        # Waypoint hand-off is the one narration with no state of its own to
        # describe -- it exists only on the wp_idx edge -- so it needs this.
        self._prev_wp_idx = None

        # ── Topics ──
        # Default is the drone's own topic (unchanged). nav_stack points it at the
        # cmd_vel_gate's input instead, so the GO gate decides what actually reaches
        # the drone -- this node keeps publishing exactly as before either way.
        self.t_cmd_vel = G("~cmd_vel_topic", self.drone_ns + "/cmd_vel")
        self.t_takeoff = self.drone_ns + "/takeoff"
        self.t_pose = self.drone_ns + "/gt_pose"
        self.t_dstate = self.drone_ns + "/state"
        self.t_path = G("~path_topic", "/path/waypoints")

        self.cmd_vel_pub = rospy.Publisher(self.t_cmd_vel, Twist, queue_size=1)
        self.takeoff_pub = rospy.Publisher(self.t_takeoff, Empty, queue_size=1, latch=True)
        self.freeze_pub = rospy.Publisher("/sensor_gate/freeze", Bool, queue_size=1, latch=True)
        self.demo_req_pub = rospy.Publisher(self.demo_mode_request_topic, String,
                                            queue_size=1, latch=True)
        self.pred_path_pub = rospy.Publisher(self.predicted_path_topic, Path,
                                             queue_size=1, latch=True)
        self.pred_score_pub = rospy.Publisher(self.predicted_score_topic, Float32,
                                              queue_size=1, latch=True)
        # Pure-pursuit-only viz publishers (latched). Harmless for the other
        # controllers (never published to), so created unconditionally.
        self.smooth_path_pub = rospy.Publisher(self.smooth_path_topic, Path,
                                               queue_size=1, latch=True)
        self.lookahead_pub = rospy.Publisher(self.lookahead_topic, PointStamped,
                                             queue_size=1, latch=True)

        # Lateral axis sign, applied at publish (holonomic path). +1 = REP-103
        # as-is; -1 corrects the inversion MEASURED on this airframe (five
        # flights: commanded left -> moved right at ~full magnitude, commanded
        # right -> nothing). One reversible dial, so the next flight is a clean
        # A/B rather than a blind platform-wide sign flip.
        self.cmd_vy_sign = float(G("~cmd_vy_sign", 1.0))
        if self.cmd_vy_sign not in (-1.0, 1.0):
            raise ValueError("~cmd_vy_sign must be exactly 1 or -1, got %r"
                             % self.cmd_vy_sign)
        # GO gate awareness. cmd_vel_gate is the single choke point for VELOCITY,
        # but this node's blockage detection, escape reflexes and blockage reports
        # to the planner have side effects the velocity gate cannot stop: a held
        # drone that keeps "trying" to fly measures no progress, invents an unseen
        # obstacle and makes A* box itself in -- all before GO is ever pressed.
        # So mirror lost_localization_node: read the gate's latched status string
        # and HOLD the whole follower (not just mute its output) while it says
        # HELD. No gate => the topic never arrives => _go_allowed stays True and
        # nothing changes for existing launches. '' disables the coupling.
        self.go_status_topic = str(G("~go_status_topic", "/mission/go_status")).strip()
        self._go_allowed = True
        if self.go_status_topic:
            rospy.Subscriber(self.go_status_topic, String, self._go_status_cb,
                             queue_size=1)

        rospy.Subscriber(self.t_pose, Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.t_dstate, Int8, self._dstate_cb, queue_size=10)
        rospy.Subscriber(self.t_path, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)
        if self.require_map_update:
            rospy.Subscriber(self.map_update_topic, OccupancyGrid, self._bev_cb,
                             queue_size=1)

        rospy.sleep(float(G("~startup_delay_sec", 0.0)))  # 0: no fixed bring-up wait
        rospy.on_shutdown(self._on_shutdown)
        rospy.Timer(rospy.Duration(self.dt), self._ctrl_loop)
        rospy.Timer(rospy.Duration(1.0 / self.status_hz), self._status)
        if self.predict_hz > 0:
            rospy.Timer(rospy.Duration(1.0 / self.predict_hz),
                        lambda _e: self._publish_prediction())
        self._banner()

    # ─── Controller factory ──────────────────────────────────────
    def _build_multi_axis(self, G):
        """Construct the multi-axis (vx + vy + yaw) follower from rosparams.

        Shares ~vel_x / ~yaw_rate / ~pos_acquisition_radius with the legacy
        follower; everything else is namespaced ~mx_* so the two tunings never
        collide. ``deg`` params are converted to radians here."""
        return MultiAxisFollower(MultiAxisFollowerParams(
            cruise_speed=float(G("~vel_x", 0.3)),
            lateral_speed_max=float(G("~mx_lateral_speed_max", 0.25)),
            yaw_rate=float(G("~mx_yaw_rate", float(G("~yaw_rate", 0.6)))),
            pos_radius=float(G("~pos_acquisition_radius", 0.35)),
            slow_radius=float(G("~mx_slow_radius", 0.8)),
            arrive_speed_min=float(G("~mx_arrive_speed_min", 0.08)),
            yaw_engage_rad=math.radians(float(G("~mx_yaw_engage_deg", 25.0))),
            yaw_release_rad=math.radians(float(G("~mx_yaw_release_deg", 10.0))),
            yaw_kp=float(G("~mx_yaw_kp", 1.2)),
            travel_cone_rad=math.radians(float(G("~mx_travel_cone_deg", 80.0))),
            translate_suppress_rad=math.radians(
                float(G("~mx_translate_suppress_deg", 120.0))),
            translate_suppress_floor=float(G("~mx_translate_suppress_floor", 0.2)),
            min_vx=float(G("~mx_min_vx", 0.06)),
            min_vy=float(G("~mx_min_vy", 0.06)),
            min_wz=math.radians(float(G("~mx_min_wz_deg", 8.0))),
            release_frac=float(G("~mx_min_release_frac", 0.5)),
            cmd_zero_eps=float(G("~mx_cmd_zero_eps", 1e-3)),
            hold_deadband=float(G("~mx_hold_deadband", 0.18)),
            hold_kp=float(G("~mx_hold_kp", 0.8)),
            hold_speed_max=float(G("~mx_hold_speed_max", 0.2)),
            hold_reacquire_margin=float(G("~mx_hold_reacquire_margin", 0.15)),
            passed_bearing_rad=math.radians(float(G("~mx_passed_bearing_deg", 110.0))),
            vel_xy_sat=float(G("~mx_vel_xy_sat", float(G("~vel_xy_sat", 1.0)))),
            yaw_rate_sat=float(G("~mx_yaw_rate_sat", float(G("~yaw_rate_sat", 1.5)))),
            accel_limit=float(G("~mx_accel_limit", float(G("~accel_limit", 1.0)))),
            yaw_accel_limit=float(G("~mx_yaw_accel_limit",
                                    float(G("~yaw_accel_limit", 2.5)))),
        ))

    def _build_roll_assist(self, G):
        """Wrap the one-axis follower with a cross-track ROLL corrector.

        The base ``WaypointFollower`` built in ``__init__`` (from ~vel_x / ~yaw_rate
        / ~pos_acquisition_radius / the whole yaw-burst tuning) keeps FULL charge of
        navigation -- align to the next point, advance, the discrete yaw pulse/settle
        loop, the map-freeze gate and the per-axis handshake are all unchanged. This
        only adds a lateral (ROLL, +linear.y = left) velocity that pulls the drone
        back onto its trajectory when it drifts sideways, scaled by what the base is
        doing this tick (full while advancing, weak while turning, small while
        holding). Tuning is namespaced ~ra_* so it never collides with the base
        follower's params; ``deg`` params are converted to radians in core."""
        corrector = CrossTrackRollCorrector(CrossTrackRollParams(
            kp_lat=float(G("~ra_kp_lat", 0.8)),
            lateral_speed_max=float(G("~ra_lateral_speed_max",
                                      float(G("~max_lateral_speed", 0.25)))),
            deadband_m=float(G("~ra_deadband_m", 0.05)),
            advance_frac=float(G("~ra_advance_frac", 1.0)),
            turn_frac=float(G("~ra_turn_frac", 0.35)),
            hold_frac=float(G("~ra_hold_frac", 0.25)),
            kp_fwd=float(G("~ra_kp_fwd", 0.6)),
            forward_speed_max=float(G("~ra_forward_speed_max", 0.15)),
            forward_deadband_m=float(G("~ra_forward_deadband_m", 0.08)),
            turn_fwd_frac=float(G("~ra_turn_fwd_frac", 0.35)),
            hold_fwd_frac=float(G("~ra_hold_fwd_frac", 0.25)),
            min_vy=float(G("~ra_min_vy", 0.06)),
            min_vx=float(G("~ra_min_vx", 0.06)),
            release_frac=float(G("~ra_release_frac", 0.5)),
            cmd_zero_eps=float(G("~ra_cmd_zero_eps", 1e-3)),
            accel_limit=float(G("~ra_accel_limit", 1.0)),
            yaw_active_eps=float(G("~ra_yaw_active_eps",
                                   float(G("~yaw_settle_eps", 0.05)))),
        ))
        return RollAssistFollower(self.follower, corrector)

    def _build_drift_pid(self, G):
        """Construct the drift-cancelling PID tracker and its ROS plumbing.

        Three PID loops (cross-track, along-track, heading) whose INTEGRAL terms
        are the learned per-axis drift, behind a force envelope that caps each
        axis and the combined multi-axis demand. Tuning is namespaced ~dp_*.

        Unlike the other controllers this one consumes localization QUALITY, so
        it also brings up the monitor for the four confidence topics and the
        publishers that expose what it has learned (/falcon/drift) and where it
        could not get through (/falcon/blockage).

        Imported HERE rather than at module load so the other controllers keep
        starting even if this module is not deployed in the container."""
        from drift_pid_follower import (          # sibling in scripts/
            DriftTelemetryPublisher, build_drift_pid, param_bool)
        from localization_quality import LocalizationQualityMonitor
        from certainty_log import CertaintyLog
        # Mirror the SI->axis-counts translation the XTEND bridge applies AFTER the
        # GO gate, so each certainty row can also carry the command the DRONE
        # actually receives (forward/lateral/vertical/yaw of -1000..1000) next to
        # the Twist that produced it. Uses the real converter as the single source
        # of truth. Defensive: a missing XTEND package logs the Twist alone rather
        # than take the flight node down over a diagnostic.
        self._twist_to_axes = None
        try:
            from sparx_agency.robots.XTEND.adapters.twist_to_cmd_nav_converter \
                import twist_to_axes
            self._twist_to_axes = twist_to_axes
        except Exception as exc:   # pragma: no cover - import guard
            rospy.logwarn("drift_pid: XTEND axis translation unavailable (%s) -- "
                          "certainty log will carry the Twist but not drone counts",
                          exc)
        follower = build_drift_pid(G)
        # Altitude hold: keep the drone at the tag-plane height. The target
        # follows ~cruise_z (the altitude the whole stack already assumes)
        # unless ~dp_alt_z overrides it. Climbing is confidence-gated and
        # ceiling-capped inside the helper -- see altitude_hold.py.
        if param_bool("~dp_alt_hold", False):
            from altitude_hold import AltitudeHold, AltitudeHoldParams
            self.alt_hold = AltitudeHold(AltitudeHoldParams(
                target_z=float(G("~dp_alt_z", float(G("~cruise_z", 1.0)))),
                deadband_m=float(G("~dp_alt_deadband_m", 0.10)),
                kp=float(G("~dp_alt_kp", 0.5)),
                climb_max=float(G("~dp_alt_climb_max", 0.15)),
                descend_max=float(G("~dp_alt_descend_max", 0.10)),
                ceiling_m=float(G("~dp_alt_ceiling_m", 1.2)),
                conf_min_climb=float(G("~dp_alt_conf_min", 0.35)),
                conf_min_descend=float(G("~dp_alt_conf_descend", 0.10)),
                min_z_m=float(G("~dp_alt_min_z_m", 0.2)),
                pulse_trigger_m=float(G("~dp_alt_pulse_trigger_m", 0.20)),
                pulse_translation_scale=float(G("~dp_alt_pulse_tscale", 0.2))))
            rospy.loginfo(
                "drift_pid: altitude hold ON -- target %.2fm, ceiling %.2fm, "
                "climb <= %.2fm/s (conf >= %.2f), pulse below %.2fm (translation "
                "x%.2f), descend <= %.2fm/s",
                self.alt_hold.params.target_z, self.alt_hold.params.ceiling_m,
                self.alt_hold.params.climb_max,
                self.alt_hold.params.conf_min_climb,
                self.alt_hold.params.target_z
                - self.alt_hold.params.pulse_trigger_m,
                self.alt_hold.params.pulse_translation_scale,
                self.alt_hold.params.descend_max)
        self.quality = LocalizationQualityMonitor(
            conf_topic=G("~dp_conf_topic", "/xtend/localization_confidence"),
            std_topic=G("~dp_std_topic", "/xtend/localization_pos_std"),
            eff_topic=G("~dp_eff_topic", "/xtend/localization_cmd_effectiveness"),
            source_topic=G("~dp_source_topic", "/xtend/localization_source"),
            require=param_bool("~dp_require_quality", False))
        self.drift_telemetry = DriftTelemetryPublisher(
            drift_topic=G("~dp_drift_topic", "/falcon/drift"),
            blockage_topic=G("~dp_blockage_topic", "/falcon/blockage"),
            rate_hz=float(G("~dp_telemetry_hz", 2.0)))
        # Per-tick certainty CSV: AprilTag confidence, drift corrections and the
        # command sent, all on one row, so a confidence dip can be matched
        # against what the drone actually did about it. Opt-in, like the thought
        # journal: '' (the default) disables it. A journal that cannot be opened
        # must not take a flight node down over a diagnostic.
        cert_path = str(rospy.get_param(CERTAINTY_LOG_PATH_PARAM, "") or "").strip()
        if cert_path:
            try:
                self.certainty_log = CertaintyLog(cert_path)
                rospy.loginfo("drift_pid: logging certainty + commands to %s",
                              cert_path)
            except (IOError, OSError) as e:
                rospy.logerr("drift_pid: cannot open the certainty log %s (%s); "
                             "flying without it", cert_path, e)
        return follower

    def _build_pure_pursuit(self, G):
        """Construct the spline-then-Pure-Pursuit follower from rosparams.

        Composes the core HermiteSmoother (~pp_smooth_*) and PurePursuitTracker
        (~pp_*) behind the follower interface (see pure_pursuit_follower.py). The
        spline timing + the tracker yaw-rate cap share one KinematicLimits; an
        optional per-axis minimum-force snap (~pp_min_*) mirrors the platform
        deadband. Altitude is never commanded (the 2D tracker is planar).

        The pure-pursuit dependencies are imported HERE (not at module load) so
        the other controllers start even when this controller is not deployed."""
        from sparx_agency.core.planning.smoothers.hermite import (
            HermiteParams, HermiteSmoother)
        from sparx_agency.core.planning.trackers.pure_pursuit import (
            PurePursuitParams, PurePursuitTracker)
        from pure_pursuit_follower import PurePursuitFollower  # sibling in scripts/
        limits = KinematicLimits(
            max_speed_xy=float(G("~pp_max_speed", 0.5)),
            max_yaw_rate=float(G("~pp_max_yaw_rate", 0.6)),
        )
        tracker = PurePursuitTracker(
            params=PurePursuitParams(
                holonomic=bool(G("~pp_holonomic", True)),
                base_lookahead=float(G("~pp_base_lookahead", 0.6)),
                min_lookahead=float(G("~pp_min_lookahead", 0.3)),
                max_lookahead=float(G("~pp_max_lookahead", 1.5)),
                lookahead_speed_gain=float(G("~pp_lookahead_speed_gain", 0.5)),
                cruise_speed=float(G("~pp_cruise_speed", float(G("~vel_x", 0.4)))),
                min_speed=float(G("~pp_min_speed", 0.1)),
                max_speed=float(G("~pp_max_speed", 0.5)),
                curvature_speed_factor=float(G("~pp_curvature_speed_factor", 0.5)),
                curvature_lookahead_factor=float(G("~pp_curvature_lookahead_factor", 0.8)),
                slow_down_distance=float(G("~pp_slow_down_distance", 1.0)),
                goal_tolerance=float(G("~pp_goal_tolerance",
                                       float(G("~pos_acquisition_radius", 0.35)))),
                path_tolerance=float(G("~pp_path_tolerance", 0.8)),
                max_yaw_rate=float(G("~pp_max_yaw_rate", 0.6)),
                speed_smoothing=float(G("~pp_speed_smoothing", 0.3)),
                yaw_rate_smoothing=float(G("~pp_yaw_rate_smoothing", 0.3)),
                sample_dt=float(G("~pp_sample_dt", 0.05)),
                closest_search_back=int(G("~pp_closest_search_back", 10)),
                closest_search_forward=int(G("~pp_closest_search_forward", 120)),
            ),
            default_limits=limits,
        )
        smoother = HermiteSmoother(HermiteParams(
            dt=float(G("~pp_smooth_dt", 0.02)),
            min_point_spacing=float(G("~pp_smooth_min_point_spacing", 0.05)),
            tangent_scale=float(G("~pp_smooth_tangent_scale", 0.5)),
            nominal_speed_xy=float(G("~pp_smooth_nominal_speed", 0.4)),
            arc_lut_samples=int(G("~pp_smooth_arc_lut_samples", 600)),
            zero_endpoint_velocity=bool(G("~pp_smooth_zero_endpoint_velocity", False)),
        ))
        return PurePursuitFollower(
            tracker, smoother, limits=limits,
            fixed_z=float(G("~takeoff_z", 1.0)),
            smooth_sample_dt=float(G("~pp_viz_sample_dt", 0.1)),
            min_vx=float(G("~pp_min_vx", 0.06)),
            min_vy=float(G("~pp_min_vy", 0.06)),
            min_wz=math.radians(float(G("~pp_min_wz_deg", 8.0))),
            min_release_frac=float(G("~pp_min_release_frac", 0.5)),
            cmd_zero_eps=float(G("~pp_cmd_zero_eps", 1e-3)),
        )

    # ─── Callbacks ───────────────────────────────────────────────
    def _pose_cb(self, msg):
        if self.cur_pose is None:
            rospy.loginfo("waypoint_follower: first /gt_pose pose=(%.2f,%.2f,%.2f)",
                          msg.position.x, msg.position.y, msg.position.z)
        self.cur_pose = msg
        # Feed the estimator at the FULL /gt_pose rate (decoupled from the control
        # loop). /gt_pose is a bare Pose (no stamp), so use the receive time.
        q = msg.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.estimator.add_measurement(msg.position.x, msg.position.y, yaw,
                                       rospy.Time.now().to_sec())

    def _dstate_cb(self, msg):
        self.drone_state = msg.data

    def _demo_mode_cb(self, msg):
        new_mode = (msg.data or "").strip().lower()
        if new_mode != self.current_demo_mode:
            rospy.loginfo("[MODE] DemoMode  %s -> %s", self.current_demo_mode, new_mode)
            self.current_demo_mode = new_mode

    def _bev_cb(self, _msg):
        # Each BEV (voxel) update is one "the map advanced" tick; the map-settle
        # gate counts these to confirm a fresh post-stop update before moving on.
        self._bev_count += 1

    def _path_cb(self, msg):
        pose2d = self._pose2d()
        pts = [Pose2D(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) < 2:
            rospy.logwarn("waypoint_follower: ignoring path with %d waypoints", len(pts))
            self.thinker.say("Ignoring a route of %d waypoints -- too short to fly"
                             % len(pts), category="plan", level="warn")
            return
        changed = not self._same_route(pts, self._path_pts)
        self.follower.set_path(pts, pose2d)
        self._path_pts = pts
        self.have_path = True
        rospy.loginfo("[PATH] NEW PATH %d waypoints  start=(%.2f,%.2f) end=(%.2f,%.2f)",
                      len(pts), pts[0].x, pts[0].y, pts[-1].x, pts[-1].y)
        # set_path re-anchors, restarting the index space, so the previous index is
        # meaningless now whether or not the route itself changed.
        self._prev_wp_idx = None
        if changed:
            # A fresh route restarts every story this node tells: "Aligning to
            # waypoint 1" is news again even when the previous route ended on the
            # same words. Only for a route that really IS fresh -- see _same_route.
            self.thinker.forget()
            length = sum(math.hypot(b.x - a.x, b.y - a.y)
                         for a, b in zip(pts, pts[1:]))
            self.thinker.say("New route: %d waypoints, %.1fm to the goal at "
                             "(x=%.2f, y=%.2f)" % (len(pts), length, pts[-1].x,
                                                   pts[-1].y), category="plan")
        self._publish_prediction()   # refresh the predicted trajectory at once
        if self.controller_kind == "pure_pursuit":
            self._publish_smooth_path()   # refresh the splined trajectory viz
        if self.state == _Bringup.WAIT_PATH:
            self._enter(_Bringup.RUNNING)

    # ─── Control loop ────────────────────────────────────────────
    def _ctrl_loop(self, _):
        if self.state == _Bringup.WAIT_POSE:
            if self.cur_pose is not None:
                self._enter(_Bringup.TAKING_OFF if self.auto_takeoff
                            else _Bringup.HOVER_SETTLE)
            else:
                # WAIT_POSE is assigned at construction, so it is the one bring-up
                # state that never passes through the _enter() funnel.
                self.thinker.say(_BRINGUP_THOUGHTS[_Bringup.WAIT_POSE],
                                 category="mission")
            return

        if self.state == _Bringup.TAKING_OFF:
            self._do_takeoff()
            return

        if self.state == _Bringup.HOVER_SETTLE:
            if self._t_in() > self.hover_settle_sec:
                self._begin_map_wait()           # unfreeze + snapshot BEV count
                self._enter(_Bringup.MAP_SETTLE)
            return

        if self.state == _Bringup.MAP_SETTLE:
            self._step_map_settle()
            return

        if self.state == _Bringup.WAIT_PATH:
            if self.have_path:
                self._enter(_Bringup.RUNNING)
            return

        # ── RUNNING: delegate navigation to the core follower ──
        # While the external state machine holds VISUAL_SERVOING, another node owns
        # /cmd_vel; go fully passive so there is exactly one publisher. FINISH is the
        # terminal land: also stay passive (no node should drive /cmd_vel while the
        # drone is stopping/landing) so we neither re-drive the route nor fight the
        # object-approach land over the demo-mode arbiter. RECOVERY is the same deal
        # for a cold pose: lost_localization_node owns /cmd_vel, and our pose is
        # stale anyway, so anything we computed would aim at a stale position.
        if self.current_demo_mode in (MODE_VISUAL_SERVOING, MODE_FINISH,
                                      MODE_RECOVERY):
            self._narrate_passive()
            return

        raw2d = self._pose2d()
        if raw2d is None:
            return

        # Pose fed to the follower: the estimator's denoised pose (command
        # feed-forward removes per-frame noise / hover drift), or the raw pose when
        # ~use_pose_estimator is off. The estimator is told the command executed
        # LAST tick (what the platform is actually doing now).
        est_hold = False
        if self.use_pose_estimator:
            # Feed the commanded lateral too so a crab is propagated, not dropped
            # as drift (vy is 0 for the one-axis follower, so its behaviour is
            # unchanged). See WindowedPoseEstimator.set_command.
            self.estimator.set_command(self._last_pub_vx, self._last_pub_wz,
                                       vy=self._last_pub_vy)
            est = self.estimator.estimate(rospy.Time.now().to_sec())
            pose2d = est.as_pose2d() if est.mode != "invalid" else raw2d
            # Stale/no localization: hold rather than fly open-loop on a dead pose.
            est_hold = est.mode in ("invalid", "hold")
        else:
            pose2d = raw2d

        if est_hold:
            self.thinker.say("No localization -- holding", category="sensor",
                             level="warn", repeat_after_s=5.0)
        else:
            # The hold is over. The sensor slot tells only this one story, so
            # forget it now or the NEXT hold would be swallowed as a repeat.
            self.thinker.forget("sensor")

        # Drive the DemoMode handshake for the axis the follower needs, and
        # tell it whether that axis is confirmed (until then it holds zero).
        axis = self.follower.required_axis()
        confirmed = True
        defer_stop_mode = False
        sup = None                         # holonomic rotation-supervisor decision
        if axis is not None:
            mode = AXIS_TO_MODE[axis]
            confirmed = (self.current_demo_mode == mode)
            self._request_demo_mode(mode)
        elif self._holonomic:
            # The holonomic controllers (multi_axis, pure_pursuit) drive all axes
            # at once, so a supervisor (not the tracker) owns the rotation freeze:
            # from last tick's commanded yaw rate + the heading + the voxel count it
            # decides freeze (-> request 'turning', which the mode-authoritative gate
            # freezes on) and hold (-> stop the tracker for a stationary re-observe).
            sup = self.rot_sup.update(pose2d.yaw, self._supervisor_cmd_wz(),
                                      self.dt, self._bev_count)
            want_mode = MODE_TURNING if sup.freeze else self.multi_axis_demo_mode
            self._request_demo_mode(want_mode)
            # Do not gate the tracker on 'turning' confirmation (that would deadlock
            # the freeze); proceed unless the operator explicitly requires the mode.
            confirmed = (not self.multi_axis_require_mode
                         or self.current_demo_mode == want_mode)
        elif self.freeze_through_coast:
            # In a stop, but pick the mode AFTER stepping so it can follow
            # cmd.freeze: while still coasting out of a frozen turn we hold
            # 'turning' (the mode-authoritative voxel gate stays frozen through the
            # inertia settle); once actually stopped we request forward mode.
            defer_stop_mode = True
        else:
            # Legacy: request forward mode the instant the stop begins (relies on
            # the gate node's ~resume_settle_sec to skip the physical coast).
            self._request_demo_mode(MODE_FORWARD)

        # Fresh voxel updates a frozen-turn stop must re-observe (stopped, sensors
        # live) before the follower may move on. 0 when advancing or in a small
        # live correction -> no stationary re-observation is forced.
        map_need = getattr(self.follower, "settle_map_updates_required", 0)
        hold = (est_hold
                or not self._go_allowed                # GO gate says HELD -> freeze
                or (rospy.Time.now() - self._node_start_t).to_sec() < self.startup_hold_sec
                or (sup is not None and sup.hold))   # supervisor mid-turn stop
        if self.quality is not None:
            # drift_pid decides its own speed, gains and whether to learn drift
            # from how much the pose can be trusted THIS tick. Fed before step so
            # the controller and the estimator see the same instant. Cached so
            # the certainty log below reports the SAME snapshot the controller
            # actually acted on, not a slightly later one.
            self._last_quality = self.quality.snapshot()
            self.follower.set_quality(self._last_quality)
        # Altitude hold (drift_pid only): a cautious vz on the solved pose z,
        # decided BEFORE the follower steps so a climb pulse's translation
        # yield is folded into the control law itself -- the envelope's slew
        # memory, the effort/effectiveness accounting, the blockage monitor and
        # the certainty log then all see the command that actually flies, and
        # the publish path below stays a dumb pass-through. Gated hard inside
        # the helper: never while held, never on a coasted or vague pose, never
        # above the ceiling. 0.0 whenever any gate says no.
        self._alt_vz = 0.0
        step_kwargs = {}
        if self.alt_hold is not None:
            q = self._last_quality
            alt = self.alt_hold.update(
                z=(self.cur_pose.position.z if self.cur_pose is not None else None),
                confidence=(q.confidence if q is not None else 0.0),
                coasting=(q.coasting if q is not None else True),
                pose_valid=(q is not None and q.valid),
                flying=not hold)
            self._alt_vz = alt.vz
            step_kwargs["translation_scale"] = alt.translation_scale
            if alt.translation_scale < 1.0:
                self.thinker.say(alt.reason, category="sensor",
                                 repeat_after_s=2.0)
        cmd = self.follower.step(pose2d, self.dt, axis_confirmed=confirmed,
                                 hold=hold, map_ready=self._map_ready(map_need),
                                 **step_kwargs)
        if self.drift_telemetry is not None:
            self._publish_drift(cmd, pose2d)
        self._last_pub_vx, self._last_pub_wz = cmd.vx, cmd.wz   # for next tick's feed-forward
        self._last_pub_vy = (cmd.vy if self._lateral else 0.0)
        self._narrate_nav(cmd)

        if defer_stop_mode:
            # Coasting out of a frozen turn (cmd.freeze True) -> hold 'turning' so
            # the gate stays frozen over the inertia; stopped/dwelling/braking
            # (cmd.freeze False/None) -> forward mode for the clean stationary
            # voxel update. cmd.freeze is None only under an external hold, which
            # already means "don't move", so forward is the safe choice there.
            self._request_demo_mode(MODE_TURNING if cmd.freeze else MODE_FORWARD)

        if sup is not None:
            # Holonomic: the supervisor owns the freeze (the 'turning' mode was
            # already requested above); mirror it onto the /sensor_gate/freeze
            # fallback bool so both freeze signals agree.
            self._set_freeze(sup.freeze and self.freeze_during_yaw)
        elif cmd.freeze is not None:
            # The core asks to freeze only while rotating; ~freeze_during_yaw
            # lets an operator disable that without touching the algorithm.
            self._set_freeze(cmd.freeze and self.freeze_during_yaw)
        self._update_map_wait(cmd, map_need)
        if self._holonomic:
            self._publish_twist_multi(cmd.vx, cmd.vy, cmd.wz, vz=self._alt_vz)
        elif self._lateral:
            # roll_assist: the base (vx, wz) still flow through the command-commitment
            # gate (the one-axis follower emits discrete pulses that need it); the
            # cross-track ROLL correction rides on linear.y, continuous and un-gated.
            self._publish_twist(cmd.vx, cmd.wz, vy=cmd.vy)
        else:
            self._publish_twist(cmd.vx, cmd.wz)
        if self.controller_kind == "pure_pursuit":
            self._publish_lookahead()   # smooth path is published on each new path

    def _supervisor_cmd_wz(self):
        """Yaw command fed to the rotation supervisor: real turns only.

        drift_pid trims its heading MID-LEG with yaw rates that can cross the
        supervisor's ``wz_turn_on`` (2026-07-21 flight: a -0.23 rad/s TRACK trim
        armed it, and the moment the trim quietened the supervisor stopped a
        cleanly tracking drone mid-corridor for a stationary re-observe; the
        drone coasted, drifted sideways and had to re-acquire the leg it was
        already on). The follower knows the difference between trimming and
        turning, so let it speak: only its TURN and ESCAPE regimes count as
        rotation worth the freeze + stop-and-re-observe discipline. TRACK and
        HOLD yaw is a trim -- the map stays live and the flight is never
        interrupted for it. The other holonomic trackers have no regime signal,
        so their commanded rate passes through unchanged."""
        if (self.controller_kind == "drift_pid"
                and self.follower.state not in ("TURN", "ESCAPE")):
            return 0.0
        return self._last_pub_wz

    def _narrate_drift(self, cmd):
        """Narrate the drift-PID controller's decision for this tick.

        Says the things an operator cannot see from the outside: which waypoint
        it is flying at, that it is escaping, that it has slowed down because the
        pose got vague, and -- the one worth watching -- how much standing drift
        it has learned it must fight."""
        t = cmd.telemetry
        if t.escape_state != "IDLE":
            self.thinker.say(t.authority or "Working my way free",
                             category="plan", level="warn")
            return
        if cmd.state == "HOLD" and not cmd.done:
            self.thinker.say(t.authority, category="sensor", level="warn",
                             repeat_after_s=5.0)
            return
        wp = self._waypoint_xy(cmd.wp_idx)
        if wp is None:
            return
        drift_cms = abs(t.drift_vy) * 100.0
        drift_note = ""
        if drift_cms >= 1.0:
            drift_note = " (holding %.0f cm/s of %s roll against the drift)" % (
                drift_cms, "left" if t.drift_vy > 0.0 else "right")
        self.thinker.say(
            "Flying to waypoint %d/%d (x=%.2f, y=%.2f), %.0f cm off the line%s"
            % (cmd.wp_idx + 1, cmd.num_waypoints, wp[0], wp[1],
               abs(t.cross_track_m) * 100.0, drift_note))

    def _publish_drift(self, cmd, pose2d):
        """Expose what the drift-PID controller has learned, and where it stuck.

        Also appends one row to the certainty log (if enabled): position/yaw,
        AprilTag confidence, the drift corrections, the target waypoint and the
        command sent, all from the SAME tick.

        ``report_blocked`` is edge-triggered by the core (true once per exhausted
        blockage episode), so forwarding it straight through cannot spam the
        planner with the same obstacle."""
        telemetry = getattr(cmd, "telemetry", None)
        if telemetry is None:
            return
        self.drift_telemetry.publish_drift(telemetry)
        if self.certainty_log is not None and self._last_quality is not None:
            wp_idx = getattr(cmd, "wp_idx", None)
            # The command the DRONE receives: the published twist put through the
            # same SI->counts translation the XTEND bridge runs after the gate.
            # The core command already carries the turn pitch and any climb
            # yield (they live inside the control law now), so the only
            # publish-side differences left are the ~cmd_vy_sign lateral flip
            # and the altitude-hold vz -- both applied here so the log mirrors
            # what actually flies.
            axes = (self._twist_to_axes(cmd.vx,
                                        cmd.vy * self.cmd_vy_sign,
                                        self._alt_vz, cmd.wz)
                    if self._twist_to_axes is not None else None)
            self.certainty_log.write(
                ros_stamp=rospy.Time.now().to_sec(),
                pose2d=pose2d,
                quality=self._last_quality,
                telemetry=telemetry,
                target_xy=self._waypoint_xy(wp_idx) if wp_idx is not None else None,
                wp_idx=wp_idx,
                num_waypoints=getattr(cmd, "num_waypoints", None),
                cmd_vx=cmd.vx, cmd_vy=cmd.vy, cmd_wz=cmd.wz,
                # Altitude + regime: this platform has no altitude hold, so a
                # falling pos_z with linear.z pinned at 0 is the ROLL/PITCH tilt
                # bleeding height; state tells forward-flight (TRACK) apart from an
                # arrived/held tick, which is why vx can be 0 with a live route.
                pos_z=(self.cur_pose.position.z
                       if self.cur_pose is not None else None),
                state=getattr(cmd, "state", ""),
                axes=axes)
        if getattr(cmd, "report_blocked", False):
            self.drift_telemetry.publish_blockage(pose2d, self.frame_id,
                                                  telemetry.blocked_axis)
            self.thinker.say(
                "Something I cannot see is blocking me and backing off did not "
                "clear it -- asking the planner for another way round",
                category="plan", level="warn")

    def _go_status_cb(self, msg):
        """Track the GO gate: HELD means freeze the follower, not just its output.

        Mirrors ``lost_localization_node`` -- the gate publishes a latched status
        string and a leading ``GO`` is the only thing that clears this node to fly.
        While HELD the control loop forces ``hold`` (see ``_on_timer``), so the
        drift_pid follower ramps to zero and freezes its drift learning AND -- the
        reason this is not left to the velocity gate alone -- runs no blockage
        detection, escape or blockage report, so a drone waiting for GO cannot
        invent an obstacle and box itself in before it has ever been allowed to
        move."""
        allowed = (msg.data or "").strip().upper().startswith("GO")
        if allowed != self._go_allowed:
            self._go_allowed = allowed
            self.thinker.say(
                "GO given -- clear to fly the route" if allowed else
                "No GO yet -- holding still and not trying to force a way through",
                category="mission", level="info" if allowed else "warn")

    # ─── Narration ───────────────────────────────────────────────
    def _narrate_passive(self):
        """Narrate why this node has stopped driving /cmd_vel.

        Going silent is itself a decision an operator needs explained: the drone
        keeps flying, but somebody else is flying it."""
        if self.current_demo_mode == MODE_VISUAL_SERVOING:
            self.thinker.say("Handing control to visual servoing for the final "
                             "approach")
        elif self.current_demo_mode == MODE_FINISH:
            self.thinker.say("Mission finished -- standing down while the drone "
                             "lands", category="mission")
        else:
            self.thinker.say("No localization -- standing by while the recovery "
                             "re-acquires it", category="sensor", level="warn",
                             repeat_after_s=5.0)

    def _waypoint_xy(self, idx):
        """Position of the waypoint ``idx`` names, or None if out of range.

        ``wp_idx`` indexes the follower's ACTIVE (re-anchored) path, not the path
        this node received: set_path drops the waypoints already passed, so
        ``self._path_pts[idx]`` would name a different point."""
        path = self.follower.active_path
        return path[idx] if 0 <= idx < len(path) else None

    @staticmethod
    def _same_route(pts, prev):
        """True when a path is the route we are already flying (within float noise).

        ``/path/waypoints`` is a STREAM, not a one-shot: a planner may re-publish
        the route the drone is already on (an unchanged A* commit, a NavDP leg
        echoed as it is re-flown). Those republishes must not count as a new
        route, because a new route calls ``thinker.forget()`` -- which would wipe
        every narration slot and replay the drone's whole train of thought from
        the top, on repeat, burying the reasoning the log exists to show.
        """
        if not prev or len(pts) != len(prev):
            return False
        return all(abs(a.x - b.x) <= 1e-3 and abs(a.y - b.y) <= 1e-3
                   for a, b in zip(pts, prev))

    def _narrate_nav(self, cmd):
        """Narrate the follower's decision for this tick.

        Safe to call every tick: ``say`` is edge-triggered, so each line reaches
        the log once -- when the decision actually changes -- and again when the
        follower moves to the next waypoint."""
        if cmd.done:
            self.thinker.say("Reached the goal -- route complete")
            return
        # Pure pursuit's wp_idx walks the spline samples it flies rather than the
        # route's corners, so per-waypoint lines would be a per-tick counter.
        if self.controller_kind == "pure_pursuit":
            return
        if self.controller_kind == "drift_pid":
            self._narrate_drift(cmd)
            return
        total = cmd.num_waypoints
        if (self._prev_wp_idx is not None and cmd.wp_idx > self._prev_wp_idx
                and cmd.wp_idx < total):
            self.thinker.say("Reached waypoint %d, heading for waypoint %d"
                             % (self._prev_wp_idx + 1, cmd.wp_idx + 1))
        self._prev_wp_idx = cmd.wp_idx
        wp = self._waypoint_xy(cmd.wp_idx)
        if wp is None:      # past the last waypoint: braking into the goal
            return
        n = cmd.wp_idx + 1
        if cmd.required_axis == ControlAxis.YAW:
            self.thinker.say("Aligning to waypoint %d/%d (x=%.2f, y=%.2f)"
                             % (n, total, wp[0], wp[1]))
        elif cmd.required_axis == ControlAxis.FORWARD:
            self.thinker.say("Flying forward to waypoint %d/%d (x=%.2f, y=%.2f)"
                             % (n, total, wp[0], wp[1]))
        elif cmd.state == FollowerState.BRAKE:
            self.thinker.say("Stopping to turn")

    # ─── ROS helpers ─────────────────────────────────────────────
    def _pose2d(self):
        if self.cur_pose is None:
            return None
        q = self.cur_pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return Pose2D(self.cur_pose.position.x, self.cur_pose.position.y, yaw)

    def _publish_smooth_path(self):
        """Publish the pure-pursuit follower's smooth (splined) trajectory for the
        BEV viewer. Called once per new path (the spline does not change tick to
        tick); latched so the viewer sees it even between updates."""
        smooth = getattr(self.follower, "smooth_xy", None)
        if not smooth or len(smooth) < 2:
            return
        m = Path()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        for x, y in smooth:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        self.smooth_path_pub.publish(m)

    def _publish_lookahead(self):
        """Publish the pure-pursuit lookahead point (the aim point) for the BEV
        viewer. Called every control tick (it moves along the path)."""
        la = getattr(self.follower, "lookahead", None)
        if la is None:
            return
        p = PointStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = self.frame_id
        p.point.x, p.point.y, p.point.z = float(la[0]), float(la[1]), 0.0
        self.lookahead_pub.publish(p)

    def _publish_prediction(self):
        """Roll the follower forward from the current pose and publish the
        predicted trajectory (+ a 0..1 quality score). Best-effort: the BEV
        viewer draws it and the planner uses it for a dynamics-aware collision
        check. No map here, so the score is dynamics-only (no clearance)."""
        # Pure pursuit has no stop-and-turn rollout; its smooth path + lookahead
        # are the visualization instead, so skip the predicted-trajectory rollout.
        # drift_pid is skipped for a harder reason: the rollout below is called
        # with self.follower.params, and both predictors require the one-axis /
        # multi-axis param dataclasses. Handing either a DriftPidParams raises
        # AttributeError, which the except clause does NOT catch -- so this guard
        # is load-bearing, not cosmetic. The planner treats /path/predicted as
        # optional, so the only cost is that its dynamics-aware collision check
        # falls back to the geometric one.
        if self.controller_kind in ("pure_pursuit", "drift_pid"):
            return
        pose2d = self._pose2d()
        if pose2d is None or len(self._path_pts) < 2:
            return
        try:
            predictor = (mx_predict_trajectory
                         if self.controller_kind == "multi_axis"
                         else predict_trajectory)
            res = predictor(self.follower.params, pose2d, self._path_pts,
                            self.dt, self.predict_horizon_s, motion=self._motion)
        except ValueError:
            return
        if len(res.poses) < 2:
            return
        m = Path()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        for p in res.poses:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = float(p.x)
            ps.pose.position.y = float(p.y)
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        self.pred_path_pub.publish(m)
        self.pred_score_pub.publish(Float32(data=float(prediction_score(res))))

    def _do_takeoff(self):
        now = rospy.Time.now()
        if now - self.last_takeoff > rospy.Duration(self.takeoff_retry_sec):
            self.takeoff_pub.publish(Empty())
            self.last_takeoff = now
            self.takeoff_count += 1
        if self.cur_pose is not None and self.cur_pose.position.z > self.takeoff_z_thresh:
            rospy.loginfo("[TAKEOFF] reached z=%.2f > %.2f -- settling",
                          self.cur_pose.position.z, self.takeoff_z_thresh)
            self._enter(_Bringup.HOVER_SETTLE)
        elif (self.takeoff_count > 0
              and now - self.last_takeoff > rospy.Duration(self.takeoff_timeout)):
            rospy.logwarn("[TAKEOFF] timeout after %d attempts -- settling", self.takeoff_count)
            self._enter(_Bringup.HOVER_SETTLE)

    # ─── Map-settle gate ─────────────────────────────────────────
    def _begin_map_wait(self):
        """Unfreeze and snapshot the BEV count so a fresh *post-stop* voxel
        update can be detected. Called when a stop begins (bring-up, or the
        unfrozen dwell of a RUNNING YAW_SETTLE)."""
        self.thinker.say(_THOUGHT_MAP_WAIT, category="map")
        self._set_freeze(False)
        self._await_map = self.require_map_update
        self._await_baseline = self._bev_count
        self._await_t0 = rospy.Time.now()

    def _map_ready(self, min_updates=1):
        """True once ``min_updates`` fresh BEV (voxel) updates have landed since
        the stop unfroze, or the wait timed out (so a mapping stall never hangs
        the drone). ``min_updates <= 0`` (advancing / small live correction) or
        map gating disabled -> always ready. Bring-up passes ~mapsettle_min_updates;
        a frozen-turn stop passes the follower's ``settle_map_updates_required``
        (>=2)."""
        if min_updates <= 0 or not self.require_map_update:
            return True
        if not self._await_map:
            # Gating is ON but this stop's wait is not armed yet: the coast->dwell
            # edge arms it (in _update_map_wait) AFTER this is consulted each tick.
            # Treat un-armed as NOT ready so a dwell that would otherwise finish in
            # its very first tick (dt >= yaw_settle_dwell_s) can never skip the
            # >=2-update re-observation; arming lands the same tick.
            return False
        if self._bev_count - self._await_baseline >= min_updates:
            return True
        if (rospy.Time.now() - self._await_t0).to_sec() > self.map_wait_timeout_s:
            rospy.logwarn_throttle(2.0, "[MAP] fewer than %d voxel updates within "
                                   "%.1fs -- proceeding", min_updates,
                                   self.map_wait_timeout_s)
            return True
        return False

    def _step_map_settle(self):
        """Bring-up: hold in forward mode, stopped, until the first voxel update
        (or timeout) -- the drone sends no motion before the map is current."""
        self._request_demo_mode(MODE_FORWARD)
        self._publish_twist(0.0, 0.0)
        if self._t_in() >= self.mapsettle_min_s and self._map_ready(self.mapsettle_min_updates):
            self._await_map = False
            self.thinker.forget("map")   # so the next stop narrates the same line
            self._enter(_Bringup.WAIT_PATH if not self.have_path else _Bringup.RUNNING)

    def _update_map_wait(self, cmd, min_updates):
        """Track the RUNNING stop: start a fresh map wait when a *frozen* turn's
        YAW_SETTLE unfreezes (begins its dwell, ``min_updates > 0``), and end it
        once the settle is left. A small live correction (``min_updates == 0``)
        never starts a wait, so the map is only re-observed when it was frozen."""
        in_dwell = (cmd.state == FollowerState.YAW_SETTLE and cmd.freeze is False
                    and min_updates > 0)
        if in_dwell and not self._await_map and self.require_map_update:
            self._await_map = True
            self._await_baseline = self._bev_count
            self._await_t0 = rospy.Time.now()
            self.thinker.say(_THOUGHT_MAP_WAIT, category="map")
        elif self._await_map and cmd.state != FollowerState.YAW_SETTLE:
            self._await_map = False
            self.thinker.forget("map")   # so the next stop narrates the same line

    def _request_demo_mode(self, mode):
        """Rate-limited, latched DemoMode request with a soft timeout warning."""
        if self.requested_demo_mode != mode:
            self.requested_demo_mode = mode
            self._request_entered_t = rospy.Time.now()
            self._last_request_pub_t = rospy.Time(0)
            rospy.loginfo("[MODE] requesting %s (current=%s)", mode, self.current_demo_mode)
        if self.current_demo_mode == mode:
            return
        now = rospy.Time.now()
        if (now - self._last_request_pub_t).to_sec() >= self.request_repeat_sec:
            self.demo_req_pub.publish(String(data=mode))
            self._last_request_pub_t = now
        if (self.request_timeout_sec > 0.0
                and (now - self._request_entered_t).to_sec() > self.request_timeout_sec):
            rospy.logwarn_throttle(2.0, "[MODE] '%s' not confirmed after %.1fs (current=%s)",
                                   mode, (now - self._request_entered_t).to_sec(),
                                   self.current_demo_mode)

    def _set_freeze(self, want):
        if self.last_freeze is want:
            return
        self.freeze_pub.publish(Bool(data=bool(want)))
        self.last_freeze = want

    def _with_yaw_pitch_bias(self, vx, wz):
        """Ride a small forward PITCH on a turn-in-place command.

        A twist that is pure yaw (``wz`` alive, ``vx`` at rest) leaves the
        platform flat, so the turn bites late and coasts. ``~yaw_pitch_bias``
        (m/s) is added as forward speed for exactly those twists; a twist that
        already commands forward motion, or one that commands no yaw at all, is
        returned untouched.

        Args:
            vx: Forward speed about to be published, in m/s.
            wz: Yaw rate about to be published, in rad/s.

        Returns:
            The forward speed to publish, in m/s.
        """
        if self.yaw_pitch_bias <= 0.0:
            return vx
        eps = self.cmd_stop_eps
        if abs(wz) < eps or abs(vx) >= eps:
            return vx
        return self.yaw_pitch_bias

    def _publish_twist(self, vx, wz, vy=0.0):
        """Assemble the Twist. linear.z = 0 is hardwired; the core has already
        enforced the invariant, saturated and slew-limited.

        ``vy`` is 0 for the one-axis waypoint controller (linear.y stays off, the
        proven behaviour) and carries the roll_assist cross-track ROLL correction
        otherwise. The command-commitment gate below applies ONLY to (vx, wz) --
        the discrete forward/yaw pulses that need the lone-pulse protection -- so a
        held motion still re-emits the base (vx, wz) while this tick's continuous
        ROLL correction (vy) passes straight through."""
        # Command-commitment (see __init__): emit each motion >=cmd_commit_ticks
        # times before switching, repeating an under-committed motion so a lone
        # 1-tick turn/forward can't be followed straight into a stop.
        eps = self.cmd_stop_eps
        cat = (0 if abs(vx) < eps else (1 if vx > 0 else -1),
               0 if abs(wz) < eps else (1 if wz > 0 else -1))
        if cat == self._cmd_cat:
            self._cmd_run += 1
        elif (self._cmd_cat not in (None, (0, 0))
              and self._cmd_run < self.cmd_commit_ticks):
            vx, wz, cat = self._cmd_vx, self._cmd_wz, self._cmd_cat  # hold motion
            self._cmd_run += 1
        else:
            self._cmd_cat, self._cmd_run = cat, 1
        if cat != (0, 0):
            self._cmd_vx, self._cmd_wz = vx, wz   # remember motion twist to repeat

        # After the gate, so the commitment category stays that of the pure yaw.
        vx = self._with_yaw_pitch_bias(vx, wz)

        m = Twist()
        m.linear.x = vx
        m.linear.y = vy   # 0 for the one-axis controller; ROLL correction for roll_assist
        m.linear.z = 0.0  # HARDWIRED -- fixed altitude (platform holds it)
        m.angular.z = wz
        self.cmd_vel_pub.publish(m)
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps({
                    "t": rospy.Time.now().to_sec(),
                    "linear": {"x": float(vx), "y": float(vy), "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": float(wz)},
                }) + "\n")
                self._log_file.flush()
            except Exception as e:
                rospy.logwarn_throttle(10.0, "waypoint_follower: log write failed: %s", e)

    def _publish_twist_multi(self, vx, vy, wz, vz=0.0):
        """Assemble a multi-axis Twist: linear.x=vx, linear.y=vy, angular.z=wz.

        This path is deliberately DUMB: every piece of flight wisdom (turn
        pitch, climb yield, saturation, slew) has already happened inside the
        controller, so what arrives here is what flies. Only two platform
        conventions are applied, both simple sign/route facts:

        ``linear.y`` is multiplied by ``~cmd_vy_sign`` on the way out: the flight
        logs measured the LATERAL axis inverted on this airframe (commanded left
        -> moved right at ~full magnitude; commanded right -> nothing), so the
        sign is correctable here, at the single point every holonomic command
        passes through -- the core keeps computing in clean REP-103 and the flip
        is one reversible dial away from an A/B test.

        ``linear.z`` carries the altitude-hold correction (0 unless the hold is
        enabled and decides to act) -- no longer hardwired: the platform's own
        height drifts, and the tags all sit at one altitude.

        The publish-time ``~yaw_pitch_bias`` only applies to trackers whose core
        does not coordinate the turn itself; drift_pid owns that in-law
        (``~dp_turn_pitch_bias``), so for it the bias here is OFF -- a second,
        invisible forward injection would desynchronize the envelope, the
        effectiveness estimate and the certainty log from what actually flew.

        No command-commitment gate -- the multi-axis controllers emit
        continuous, minimum-force-shaped commands, so they never need the
        lone-pulse protection the one-axis path uses."""
        if not getattr(self, "_core_owns_pitch_bias", False):
            vx = self._with_yaw_pitch_bias(vx, wz)
        vy = vy * self.cmd_vy_sign
        m = Twist()
        m.linear.x = vx
        m.linear.y = vy  # lateral (crab) -- enabled for the multi-axis controller
        m.linear.z = float(vz)  # altitude-hold correction (0 = platform's own hold)
        m.angular.z = wz
        self.cmd_vel_pub.publish(m)
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps({
                    "t": rospy.Time.now().to_sec(),
                    "linear": {"x": float(vx), "y": float(vy), "z": float(vz)},
                    "angular": {"x": 0.0, "y": 0.0, "z": float(wz)},
                }) + "\n")
                self._log_file.flush()
            except Exception as e:
                rospy.logwarn_throttle(10.0, "waypoint_follower: log write failed: %s", e)

    def _publish_twist_multi(self, vx, vy, wz):
        """Assemble a multi-axis Twist: linear.x=vx, linear.y=vy, angular.z=wz;
        linear.z=0 hardwired (fixed altitude). No command-commitment gate -- the
        multi-axis controller emits continuous, minimum-force-shaped commands, so
        it never needs the lone-pulse protection the one-axis path uses."""
        m = Twist()
        m.linear.x = vx
        m.linear.y = vy  # lateral (crab) -- enabled for the multi-axis controller
        m.linear.z = 0.0  # HARDWIRED -- fixed altitude (platform holds it)
        m.angular.z = wz
        self.cmd_vel_pub.publish(m)
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps({
                    "t": rospy.Time.now().to_sec(),
                    "linear": {"x": float(vx), "y": float(vy), "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": float(wz)},
                }) + "\n")
                self._log_file.flush()
            except Exception as e:
                rospy.logwarn_throttle(10.0, "waypoint_follower: log write failed: %s", e)

    def _open_log(self, path):
        if path and "{ts}" in path:
            path = path.replace("{ts}", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not path:
            return None
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            return open(path, "w")
        except Exception as e:
            rospy.logwarn("waypoint_follower: failed to open log %s: %s", path, e)
            return None

    def _enter(self, new):
        if new != self.state:
            rospy.loginfo("[STATE] %s -> %s", self.state, new)
            self.state = new
            self.t_state = rospy.Time.now()
            thought = _BRINGUP_THOUGHTS.get(new)
            if thought is not None:
                self.thinker.say(thought, category="mission")

    def _t_in(self):
        return (rospy.Time.now() - self.t_state).to_sec()

    def _status(self, _):
        if self.cur_pose is None:
            rospy.loginfo("[%-12s] no /gt_pose yet (subscribed to %s)", self.state, self.t_pose)
            return
        p = self.cur_pose.position
        pose2d = self._pose2d()
        f = self.follower
        rospy.loginfo("[%-12s] pose=(%.2f,%.2f,%.2f) yaw=%5.1f° | nav=%s done=%s mode=%s",
                      self.state, p.x, p.y, p.z, math.degrees(pose2d.yaw),
                      f.state.value, f.done, self.current_demo_mode or "none")

    def _on_shutdown(self):
        try:
            for _ in range(5):
                self._publish_twist(0.0, 0.0)
                rospy.sleep(0.02)
        except Exception:
            pass
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass

    def _banner(self):
        L = rospy.loginfo
        p = self.follower.params
        L("=" * 72)
        if self.controller_kind == "multi_axis":
            L("waypoint_follower (core MultiAxisFollower)  X+Y+YAW, fixed altitude")
            L("  drone_ns = %s   ctrl=%dHz", self.drone_ns, int(self.ctrl_rate_hz))
            L("  cruise=%.2f m/s  lat_max=%.2f m/s  yaw_rate=%.2f rad/s",
              p.cruise_speed, p.lateral_speed_max, p.yaw_rate)
            L("  pos_radius=%.2f  yaw_engage=%.0f deg  release=%.0f deg  cone=%.0f deg",
              p.pos_radius, math.degrees(p.yaw_engage_rad),
              math.degrees(p.yaw_release_rad), math.degrees(p.travel_cone_rad))
            L("  min force: vx=%.3f vy=%.3f wz=%.0f deg/s  hold_deadband=%.2f m",
              p.min_vx, p.min_vy, math.degrees(p.min_wz), p.hold_deadband)
            L("  demo_mode=%s (require=%s)", self.multi_axis_demo_mode,
              self.multi_axis_require_mode)
            L("  PUBLISHED Twist invariants:  vz=0  (vx, vy, wz combined)")
            L("=" * 72)
            return
        if self.controller_kind == "pure_pursuit":
            L("waypoint_follower (Hermite spline + core PurePursuitTracker)  fixed alt")
            L("  drone_ns = %s   ctrl=%dHz", self.drone_ns, int(self.ctrl_rate_hz))
            L("  cruise=%.2f m/s  lookahead=%.2f..%.2f m (base %.2f)  holonomic=%s",
              p.cruise_speed, p.min_lookahead, p.max_lookahead, p.base_lookahead,
              p.holonomic)
            L("  goal_tol=%.2f m  path_tol=%.2f m  max_yaw_rate=%.2f rad/s",
              p.goal_tolerance, p.path_tolerance, p.max_yaw_rate)
            L("  smooth -> %s   lookahead -> %s", self.smooth_path_topic,
              self.lookahead_topic)
            L("  demo_mode=%s (require=%s)", self.multi_axis_demo_mode,
              self.multi_axis_require_mode)
            L("  PUBLISHED Twist invariants:  vz=0  (vx, vy, wz combined)")
            L("=" * 72)
            return
        if self.controller_kind == "drift_pid":
            e, c = p.envelope, p.confidence
            L("waypoint_follower (core DriftPidFollower)  X+Y+YAW, drift-cancelling")
            L("  drone_ns = %s   ctrl=%dHz", self.drone_ns, int(self.ctrl_rate_hz))
            L("  cruise=%.2f m/s  lookahead=%.2f m  pos_radius=%.2f m",
              p.cruise_speed, p.lookahead_m, p.pos_radius)
            L("  ENVELOPE  vx<=%.2f  vy<=%.2f m/s  wz<=%.2f rad/s  |v|<=%.2f m/s",
              e.max_vx, e.max_vy, e.max_wz, e.max_translation)
            L("            combined effort <= %.2f (1.0 = one axis at its max)",
              e.combined_effort)
            L("            min force: vx=%.3f vy=%.3f m/s  wz=%.0f deg/s",
              e.min_vx, e.min_vy, math.degrees(e.min_wz))
            L("            accel: xy=%.2f m/s2  yaw=%.2f rad/s2", e.accel_xy,
              e.accel_wz)
            L("  DRIFT PID (kp/ki/kd -> max correction)")
            L("    cross-track %.2f/%.2f/%.2f -> %.2f m/s   deadband %.2f m",
              p.lateral_pid.kp, p.lateral_pid.ki, p.lateral_pid.kd,
              p.lateral_pid.out_limit, p.lateral_pid.deadband)
            L("    along-track %.2f/%.2f/%.2f -> %.2f m/s   deadband %.2f m",
              p.forward_pid.kp, p.forward_pid.ki, p.forward_pid.kd,
              p.forward_pid.out_limit, p.forward_pid.deadband)
            L("    heading    %.2f/%.2f/%.2f -> %.2f rad/s  deadband %.1f deg",
              p.yaw_pid.kp, p.yaw_pid.ki, p.yaw_pid.kd, p.yaw_pid.out_limit,
              math.degrees(p.yaw_pid.deadband))
            L("  CONFIDENCE  full>=%.2f  floor<=%.2f (speed x%.2f)  learn>=%.2f  "
              "hold<%.2f  age<=%.2fs", c.conf_full, c.conf_min, c.speed_floor,
              c.conf_integrate, c.conf_hold, c.max_age_s)
            L("  QUALITY   latency lead=%.2fs (earned, off while coasting)  "
              "earned-speed floor=%.2f", c.latency_s, c.eff_speed_floor)
            L("            std->deadband +%.2f m/m above %.2f m (cap +%.2f m)  "
              "brake %.2f/%.2f m/s2", c.std_deadband_gain, c.std_ref_m,
              c.deadband_extra_max_m, e.decel_xy, e.accel_xy)
            L("  BLOCKAGE  %.1fs window, confirm %d ticks, progress<%.0f%%  "
              "-> escape x%d", p.blockage.window_s, p.blockage.confirm_ticks,
              p.blockage.progress_frac * 100.0, p.escape.max_attempts)
            L("  telemetry -> /falcon/drift    blockage -> /falcon/blockage")
            L("  PUBLISHED Twist invariants:  vz=0  (vx, vy, wz combined)")
            L("=" * 72)
            return
        if self.controller_kind == "roll_assist":
            rp = self.follower.roll_params
            L("waypoint_follower (core RollAssistFollower)  X+YAW nav + cross-track ROLL")
            L("  drone_ns = %s   ctrl=%dHz", self.drone_ns, int(self.ctrl_rate_hz))
            L("  base: vel_x=%.2f m/s  yaw_rate=%.2f rad/s  pos_radius=%.2f  forward_only=%s",
              p.vel_x, p.yaw_rate, p.pos_radius, p.forward_only)
            L("  ROLL: kp_lat=%.2f  lat_max=%.2f m/s  deadband=%.2f m",
              rp.kp_lat, rp.lateral_speed_max, rp.deadband_m)
            L("  lateral gain frac: advance=%.2f  turn=%.2f  hold=%.2f",
              rp.advance_frac, rp.turn_frac, rp.hold_frac)
            L("  along-track: kp_fwd=%.2f  fwd_max=%.2f m/s (turn=%.2f hold=%.2f)",
              rp.kp_fwd, rp.forward_speed_max, rp.turn_fwd_frac, rp.hold_fwd_frac)
            L("  PUBLISHED Twist invariants:  vz=0  (base vx, wz + ROLL vy)")
            L("=" * 72)
            return
        L("waypoint_follower (core WaypointFollower)  X+YAW only, fixed altitude")
        L("  drone_ns = %s   ctrl=%dHz", self.drone_ns, int(self.ctrl_rate_hz))
        L("  vel_x=%.2f m/s  yaw_rate=%.2f rad/s  forward_only=%s",
          p.vel_x, p.yaw_rate, p.forward_only)
        L("  pos_radius=%.2f  min_motion_ticks=%d  settle_dwell=%.2fs",
          p.pos_radius, p.min_motion_ticks, p.yaw_settle_dwell_s)
        L("  map-settle: require=%s topic=%s timeout=%.1fs", self.require_map_update,
          self.map_update_topic, self.map_wait_timeout_s)
        L("  takeoff_z=%.2f m (Empty -> %s; no vz commands)", self.takeoff_z, self.t_takeoff)
        L("  PUBLISHED Twist invariants:  vy=0  vz=0  (vx=0 OR wz=0)")
        L("=" * 72)


def main():
    try:
        WaypointFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The navigation maths live
# in core.planning.trackers.waypoint_follower; this node maps rosparams ->
# WaypointFollowerParams and owns ROS I/O, bring-up, the DemoMode handshake,
# the freeze gate, the startup hold and the cmd_vel log.
#
#   IO: ~drone_ns (/simple_drone) [+/gt_pose +/state +/cmd_vel +/takeoff]
#       ~path_topic (/path/waypoints) ~frame_id (world)
#       ~predicted_path_topic (/path/predicted) ~predicted_score_topic (/path/predicted_score)
#       ~demo_mode_topic (/xtend/demo_mode) ~demo_mode_request_topic (/xtend/demo_mode_request)
#   follower: ~vel_x (0.3) ~yaw_rate (0.7) ~pos_acquisition_radius (0.35)
#       ~yaw_settle_thresh (0.05) ~yaw_drift_thresh (0.40) ~skip_yaw_thresh (0.25)
#       ~vx_brake_thresh (0.05) ~brake_timeout_s (2.0) ~passed_bearing_deg (100)
#       ~vel_xy_sat (1.25) ~yaw_rate_sat (2.4) ~accel_limit (1.5)
#       ~yaw_accel_limit (3.5) ~forward_only (false) ~yaw_lead_pct (10, deprecated)
#   yaw pulse->settle: ~yaw_settle_dwell_s (0.8) ~yaw_settle_eps (0.05)
#       ~min_motion_ticks (2; min consecutive ticks for forward OR yaw) ~yaw_coast_deg (15)
#       ~yaw_burst_max_deg (25; per-burst increment -- big turns split into chunks,
#         each followed by a stop + voxel update + re-measure)
#       ~yaw_acquisition_radius (0.20; cross-track ADVANCE tol, m) ~yaw_acquire_max_deg (35)
#   rotation freeze: ~freeze_on_rotation (true; MASTER on/off -- false keeps the map
#       live through every turn, for both controller families)
#     waypoint controller (angle-gated): ~freeze_yaw_thresh_deg (20; a turn LARGER than
#       this freezes the voxel map and forces the post-turn re-observation -- a smaller
#       correction skips both; 0 = freeze every turn) ~settle_map_updates (2; fresh
#       voxel updates a frozen turn re-observes, stopped, before it moves on)
#       ~freeze_through_coast (true; hold 'turning' until wz~0 so the gate stays frozen
#       over the end-of-turn inertia, not just the commanded burst)
#     holonomic supervisor (pure_pursuit/multi_axis; rate-based freeze-throughout +
#       stop every N deg to re-observe): ~rot_wz_turn_on (0.20 rad/s) ~rot_wz_turn_off
#       (0.10) ~rot_turn_off_ticks (3) ~rot_reobserve_every_deg (25) ~rot_max_coast_s
#       (2.0); reuses ~settle_map_updates ~yaw_settle_dwell_s ~yaw_settle_eps
#       ~map_wait_timeout_s
#   prediction: ~predict_hz (2.0) ~predict_horizon_s (30) ~predict_yaw_tau_s (0.5)
#       ~predict_vx_tau_s (0.3)
#   map-settle (forward mode + fresh voxel updates before any motion / before the
#     next turn): ~require_map_update (true) ~map_update_topic (/falcon/bev_2d)
#     ~map_wait_timeout_s (3.0; proceed anyway after this) ~mapsettle_min_s (0.5)
#     ~mapsettle_min_updates (2; fresh updates before the FIRST move at bring-up)
#   takeoff: ~auto_takeoff (true) ~takeoff_z (1.0) ~takeoff_z_thresh (0.5)
#       ~takeoff_timeout (30) ~takeoff_retry_sec (1.0) ~hover_settle_sec (2.5)
#   controller: ~controller (waypoint | multi_axis | pure_pursuit | roll_assist |
#       drift_pid). An unrecognised value now RAISES at startup rather than falling
#       silently through to the one-axis follower.
#     drift_pid is the continuous multi-axis tracker that LEARNS the drift: three
#       PID loops (cross-track / along-track / heading) whose integral term IS the
#       per-axis drift estimate, so a steady sideways push stops leaving a standing
#       offset the way a P-only law does. On top of that: a force envelope capping
#       each axis AND the combined multi-axis demand (this platform is markedly
#       faster when several axes are driven together), speed/gain scheduling from
#       localization confidence with the integrators FROZEN while the pose is
#       coasted, and reflexes for walls the camera cannot see (brake -> back off ->
#       probe sideways). When the reflexes are spent it reports the spot once on
#       /falcon/blockage and the PLANNER reroutes -- the controller never edits the
#       route. Publishes /falcon/drift (what it has learned, m/s per axis).
#       Consumes the four ROS2 localization quality topics; they must be present in
#       bridge.yaml or it flies without confidence gating (and warns loudly).
#       All ~78 dials are namespaced ~dp_* and exposed in mission.yaml under
#       "CONTROLLER 5"; see core/planning/trackers/drift_pid/README.md for the
#       design and the tuning order. Extra topics/behaviour params:
#       ~dp_require_quality (false) ~dp_telemetry_hz (2.0) ~dp_drift_topic
#       (/falcon/drift) ~dp_blockage_topic (/falcon/blockage) ~dp_conf_topic
#       ~dp_std_topic ~dp_eff_topic ~dp_source_topic.
#     roll_assist keeps the one-axis waypoint follower UNCHANGED (same align->advance
#       nav, discrete yaw, freeze/map gates, per-axis handshake) and layers a
#       cross-track ROLL (linear.y) correction on top: it only pulls the drone back
#       onto its trajectory when it drifts sideways -- full gain while advancing,
#       weak while turning (ROLL would spoil the rotation), small while holding; a
#       small forward/back nudge corrects along-track drift while turning/holding.
#       No correction while gated (hold / unconfirmed axis / done). Tuning ~ra_*:
#       ~ra_kp_lat (0.8) ~ra_lateral_speed_max (=max_lateral_speed, 0.25)
#       ~ra_deadband_m (0.05) ~ra_advance_frac (1.0) ~ra_turn_frac (0.35)
#       ~ra_hold_frac (0.25) ~ra_kp_fwd (0.6) ~ra_forward_speed_max (0.15)
#       ~ra_forward_deadband_m (0.08) ~ra_turn_fwd_frac (0.35) ~ra_hold_fwd_frac (0.25)
#       ~ra_min_vy (0.06) ~ra_min_vx (0.06) ~ra_release_frac (0.5)
#       ~ra_cmd_zero_eps (1e-3) ~ra_accel_limit (1.0) ~ra_yaw_active_eps (=yaw_settle_eps).
#     multi_axis swaps in the combined forward+lateral+yaw tracker (un-hardwires
#       linear.y; no per-axis handshake). Tuning namespaced ~mx_*:
#       ~mx_lateral_speed_max (0.25) ~mx_yaw_rate (0.6) ~mx_slow_radius (0.8)
#       ~mx_arrive_speed_min (0.08) ~mx_yaw_engage_deg (25) ~mx_yaw_release_deg (10)
#       ~mx_yaw_kp (1.2) ~mx_travel_cone_deg (80) ~mx_translate_suppress_deg (120)
#       ~mx_translate_suppress_floor (0.2) ~mx_min_vx (0.06) ~mx_min_vy (0.06)
#       ~mx_min_wz_deg (8) ~mx_min_release_frac (0.5) ~mx_hold_deadband (0.18)
#       ~mx_hold_kp (0.8) ~mx_hold_speed_max (0.2) ~mx_hold_reacquire_margin (0.15)
#       ~mx_passed_bearing_deg (110) ~mx_demo_mode (fly_straight) ~mx_require_mode (false)
#     pure_pursuit splines the path (core HermiteSmoother) then tracks it with the
#       core PurePursuitTracker on a moving lookahead (holonomic). Tracker ~pp_*:
#       ~pp_holonomic (true) ~pp_cruise_speed (=vel_x) ~pp_max_speed (0.5)
#       ~pp_min_speed (0.1) ~pp_base_lookahead (0.6) ~pp_min/max_lookahead (0.3/1.5)
#       ~pp_lookahead_speed_gain (0.5) ~pp_goal_tolerance (=pos_acquisition_radius)
#       ~pp_path_tolerance (0.8) ~pp_max_yaw_rate (0.6) ~pp_curvature_speed_factor (0.5)
#       ~pp_curvature_lookahead_factor (0.8) ~pp_slow_down_distance (1.0)
#       ~pp_speed_smoothing (0.3) ~pp_yaw_rate_smoothing (0.3) ~pp_sample_dt (0.05)
#       ~pp_closest_search_back/forward (10/120). Spline ~pp_smooth_*:
#       ~pp_smooth_dt (0.02) ~pp_smooth_min_point_spacing (0.05)
#       ~pp_smooth_tangent_scale (0.5) ~pp_smooth_nominal_speed (0.4)
#       ~pp_smooth_arc_lut_samples (600) ~pp_smooth_zero_endpoint_velocity (false).
#       Min-force snap ~pp_min_vx/vy (0.06) ~pp_min_wz_deg (8) ~pp_min_release_frac (0.5).
#       Viz out: ~smooth_path_topic (/path/smooth) ~lookahead_topic (/path/lookahead)
#       ~pp_viz_sample_dt (0.1).
#   loop/gate: ~ctrl_rate_hz (5) ~status_hz (1) ~freeze_during_yaw (true)
#       ~startup_hold_sec (3.0) ~startup_delay_sec (1.0)
#       ~request_repeat_sec (0.5) ~request_timeout_sec (5.0)
#   logging: ~cmd_log_path (/home/falcon/runs/cmd_log_{ts}.jsonl; empty disables)
#   narration (inherited from thinking.Thinker): ~thinking (true; false silences
#       this node's thoughts) ~thinking_topic (/nav/thinking) ~thinking_echo (true)
# ============================================================================
