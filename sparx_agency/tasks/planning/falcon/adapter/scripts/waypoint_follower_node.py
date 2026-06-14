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

See the file footer for the full rosparam list.
"""
import datetime
import json
import math
import os

import rospy
import tf.transformations as tft
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, Empty, Float32, Int8, String

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.waypoint_follower import (
    ControlAxis,
    FollowerState,
    MotionModelParams,
    WaypointFollower,
    WaypointFollowerParams,
    predict_trajectory,
    prediction_score,
)

# DemoMode payloads bridged over /xtend/demo_mode(_request). The core control
# axis maps onto the platform's flight modes; visual_servoing is platform-only.
AXIS_TO_MODE = {ControlAxis.YAW: "turning", ControlAxis.FORWARD: "fly_straight"}
MODE_VISUAL_SERVOING = "visual_servoing"
# "Forward flight" mode, requested during every stop (and at bring-up): it tells
# the FC to hold heading (stop rotating) while we command zero velocity, so the
# platform is stable for a clean voxel update -- "forward mode but not flying".
MODE_FORWARD = AXIS_TO_MODE[ControlAxis.FORWARD]


class _Bringup:
    """Node-side platform bring-up states (the core owns navigation states)."""
    WAIT_POSE = "WAIT_POSE"
    TAKING_OFF = "TAKING_OFF"
    HOVER_SETTLE = "HOVER_SETTLE"
    MAP_SETTLE = "MAP_SETTLE"      # forward mode + stopped + first voxel update
    WAIT_PATH = "WAIT_PATH"
    RUNNING = "RUNNING"


class WaypointFollowerNode:
    def __init__(self):
        rospy.init_node("waypoint_follower")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "/simple_drone")

        self.follower = WaypointFollower(WaypointFollowerParams(
            vel_x=float(G("~vel_x", 0.3)),
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
        self.startup_hold_sec = float(G("~startup_hold_sec", 3.0))
        self.frame_id = G("~frame_id", "world")

        # Trajectory prediction (rollout) -> /path/predicted for the BEV viewer
        # and the planner's dynamics-aware collision check.
        self.predict_hz = float(G("~predict_hz", 2.0))
        self.predict_horizon_s = float(G("~predict_horizon_s", 30.0))
        self.predicted_path_topic = G("~predicted_path_topic", "/path/predicted")
        self.predicted_score_topic = G("~predicted_score_topic",
                                       "/path/predicted_score")
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

        # ── Topics ──
        self.t_cmd_vel = self.drone_ns + "/cmd_vel"
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

        rospy.Subscriber(self.t_pose, Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.t_dstate, Int8, self._dstate_cb, queue_size=10)
        rospy.Subscriber(self.t_path, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.demo_mode_topic, String, self._demo_mode_cb, queue_size=10)
        if self.require_map_update:
            rospy.Subscriber(self.map_update_topic, OccupancyGrid, self._bev_cb,
                             queue_size=1)

        rospy.sleep(float(G("~startup_delay_sec", 1.0)))
        rospy.on_shutdown(self._on_shutdown)
        rospy.Timer(rospy.Duration(self.dt), self._ctrl_loop)
        rospy.Timer(rospy.Duration(1.0 / self.status_hz), self._status)
        if self.predict_hz > 0:
            rospy.Timer(rospy.Duration(1.0 / self.predict_hz),
                        lambda _e: self._publish_prediction())
        self._banner()

    # ─── Callbacks ───────────────────────────────────────────────
    def _pose_cb(self, msg):
        if self.cur_pose is None:
            rospy.loginfo("waypoint_follower: first /gt_pose pose=(%.2f,%.2f,%.2f)",
                          msg.position.x, msg.position.y, msg.position.z)
        self.cur_pose = msg

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
            return
        self.follower.set_path(pts, pose2d)
        self._path_pts = pts
        self.have_path = True
        rospy.loginfo("[PATH] NEW PATH %d waypoints  start=(%.2f,%.2f) end=(%.2f,%.2f)",
                      len(pts), pts[0].x, pts[0].y, pts[-1].x, pts[-1].y)
        self._publish_prediction()   # refresh the predicted trajectory at once
        if self.state == _Bringup.WAIT_PATH:
            self._enter(_Bringup.RUNNING)

    # ─── Control loop ────────────────────────────────────────────
    def _ctrl_loop(self, _):
        if self.state == _Bringup.WAIT_POSE:
            if self.cur_pose is not None:
                self._enter(_Bringup.TAKING_OFF if self.auto_takeoff
                            else _Bringup.HOVER_SETTLE)
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
        # While the external state machine holds VISUAL_SERVOING, another node
        # owns /cmd_vel; go fully passive so there is exactly one publisher.
        if self.current_demo_mode == MODE_VISUAL_SERVOING:
            return

        pose2d = self._pose2d()
        if pose2d is None:
            return

        # Drive the DemoMode handshake for the axis the follower needs, and
        # tell it whether that axis is confirmed (until then it holds zero).
        axis = self.follower.required_axis()
        confirmed = True
        if axis is not None:
            mode = AXIS_TO_MODE[axis]
            confirmed = (self.current_demo_mode == mode)
            self._request_demo_mode(mode)
        else:
            # In a stop (settle/brake/done): request forward mode -- hold heading,
            # do not fly -- so the platform is stable for a clean voxel update.
            self._request_demo_mode(MODE_FORWARD)

        hold = ((rospy.Time.now() - self._node_start_t).to_sec() < self.startup_hold_sec)
        cmd = self.follower.step(pose2d, self.dt, axis_confirmed=confirmed,
                                 hold=hold, map_ready=self._map_ready())

        if cmd.freeze is not None:
            # The core asks to freeze only while rotating; ~freeze_during_yaw
            # lets an operator disable that without touching the algorithm.
            self._set_freeze(cmd.freeze and self.freeze_during_yaw)
        self._update_map_wait(cmd)
        self._publish_twist(cmd.vx, cmd.wz)

    # ─── ROS helpers ─────────────────────────────────────────────
    def _pose2d(self):
        if self.cur_pose is None:
            return None
        q = self.cur_pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        return Pose2D(self.cur_pose.position.x, self.cur_pose.position.y, yaw)

    def _publish_prediction(self):
        """Roll the follower forward from the current pose and publish the
        predicted trajectory (+ a 0..1 quality score). Best-effort: the BEV
        viewer draws it and the planner uses it for a dynamics-aware collision
        check. No map here, so the score is dynamics-only (no clearance)."""
        pose2d = self._pose2d()
        if pose2d is None or len(self._path_pts) < 2:
            return
        try:
            res = predict_trajectory(self.follower.params, pose2d, self._path_pts,
                                     self.dt, self.predict_horizon_s,
                                     motion=self._motion)
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
        self._set_freeze(False)
        self._await_map = self.require_map_update
        self._await_baseline = self._bev_count
        self._await_t0 = rospy.Time.now()

    def _map_ready(self):
        """True once a fresh BEV (voxel) update has landed since the stop unfroze,
        or the wait timed out (so a mapping stall never hangs the drone)."""
        if not self._await_map:
            return True
        if self._bev_count > self._await_baseline:
            return True
        if (rospy.Time.now() - self._await_t0).to_sec() > self.map_wait_timeout_s:
            rospy.logwarn_throttle(2.0, "[MAP] no voxel update within %.1fs -- "
                                   "proceeding", self.map_wait_timeout_s)
            return True
        return False

    def _step_map_settle(self):
        """Bring-up: hold in forward mode, stopped, until the first voxel update
        (or timeout) -- the drone sends no motion before the map is current."""
        self._request_demo_mode(MODE_FORWARD)
        self._publish_twist(0.0, 0.0)
        if self._t_in() >= self.mapsettle_min_s and self._map_ready():
            self._await_map = False
            self._enter(_Bringup.WAIT_PATH if not self.have_path else _Bringup.RUNNING)

    def _update_map_wait(self, cmd):
        """Track the RUNNING stop: start a fresh map wait when a YAW_SETTLE
        unfreezes (begins its dwell), and end it once the settle is left."""
        in_dwell = (cmd.state == FollowerState.YAW_SETTLE and cmd.freeze is False)
        if in_dwell and not self._await_map and self.require_map_update:
            self._await_map = True
            self._await_baseline = self._bev_count
            self._await_t0 = rospy.Time.now()
        elif self._await_map and cmd.state != FollowerState.YAW_SETTLE:
            self._await_map = False

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

    def _publish_twist(self, vx, wz):
        """Assemble the Twist. linear.y = linear.z = 0 are hardwired here; the
        core has already enforced the invariant, saturated and slew-limited."""
        m = Twist()
        m.linear.x = vx
        m.linear.y = 0.0  # HARDWIRED -- no lateral movement, ever
        m.linear.z = 0.0  # HARDWIRED -- fixed altitude (platform holds it)
        m.angular.z = wz
        self.cmd_vel_pub.publish(m)
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps({
                    "t": rospy.Time.now().to_sec(),
                    "linear": {"x": float(vx), "y": 0.0, "z": 0.0},
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
#   prediction: ~predict_hz (2.0) ~predict_horizon_s (30) ~predict_yaw_tau_s (0.5)
#       ~predict_vx_tau_s (0.3)
#   map-settle (forward mode + a fresh voxel update before any motion / before the
#     next turn): ~require_map_update (true) ~map_update_topic (/falcon/bev_2d)
#     ~map_wait_timeout_s (3.0; proceed anyway after this) ~mapsettle_min_s (0.5)
#   takeoff: ~auto_takeoff (true) ~takeoff_z (1.0) ~takeoff_z_thresh (0.5)
#       ~takeoff_timeout (30) ~takeoff_retry_sec (1.0) ~hover_settle_sec (2.5)
#   loop/gate: ~ctrl_rate_hz (5) ~status_hz (1) ~freeze_during_yaw (true)
#       ~startup_hold_sec (3.0) ~startup_delay_sec (1.0)
#       ~request_repeat_sec (0.5) ~request_timeout_sec (5.0)
#   logging: ~cmd_log_path (/home/falcon/runs/cmd_log_{ts}.jsonl; empty disables)
# ============================================================================
