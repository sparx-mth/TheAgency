#!/usr/bin/env python3
"""astar_navdp_fallback_node.py -- FALCON "fallback" mode: A*-primary, NavDP rescue.

A fourth navigation mode built for one real failure: at a hard turn at the end of
a corridor, a narrow door, or a tight passage, A* often finds **no route at all**
-- the voxel map mis-marks the opening as occupied, or the inflation radius plus a
small localization error paints an obstacle exactly where the drone must pass.
When that happens A* silently keeps its last (stale) path and the drone stalls.

This node is the arbiter that drives ``/path/waypoints_fallback`` -- the single raw
path the planner-agnostic ``path_corrector`` -> ``trajectory_simplifier`` ->
``waypoint_follower`` chain flies. It runs a 2-mode HYSTERETIC state machine:

  PRIMARY  -- echo the A* route straight through (plain ``nav_mode:=astar``). Watch
              the A* per-attempt status signal (``/path/astar_status``, published by
              ``astar_planner`` once per real planning attempt). When A* reports NO
              route on ``~astar_fail_confirm`` CONSECUTIVE attempts, STOP the drone
              and hand control to NavDP.
  FALLBACK -- drive NavDP toward the FINAL mission goal expressed in the drone body
              frame (``world_to_body_2d`` of the goal at the current pose -- e.g.
              world goal (0,-3) with the drone at (0,-1) -> body (0,-2)). NavDP is a
              learned local policy that can thread the opening A* rejected. It flies
              each leg to its midpoint (``~leg_execute_fraction``), then re-infers.
              Only after A* reports a route on ``~astar_ok_confirm`` CONSECUTIVE
              attempts do we hand control BACK to A*.

The hysteresis (``fail_confirm`` to leave A*, a larger ``ok_confirm`` to return) is
the point: we never abandon NavDP the instant A* finds one route -- that would
zig-zag right at the hard spot. Unlike ``combination`` mode there is NO visibility
gate: the final goal is usually NOT visible (it is around the corner / past the
door -- exactly why A* failed), so NavDP is handed the bearing to the goal and
left to plan the local avoidance.

All the maths is ROS-free and unit-tested in ``core.planning.navdp`` /
``core.planning.planners.common``:
  * world goal -> body-frame point-goal   (geometry.world_to_body_2d + point_to_pointgoal)
  * NavDP HTTP request/response           (client.NavDPPointgoalClient)
  * body trajectory -> world path         (geometry.anchor_trajectory_to_world)
  * progress along the flown leg          (utils_2d.arclength_fraction_2d)
This node owns ONLY ROS concerns; it runs HEADLESS (the route is seen on the BEV
viewer). RGB/depth/pose ingestion mirrors ``combination_planner`` / ``navdp_click``.

Run:
    rosrun falcon_adapter astar_navdp_fallback_node.py
See the file footer for the full rosparam list.
"""
import math
import time

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.math import se3
from sparx_agency.core.common.types import Intrinsics, Pose2D
from sparx_agency.core.planning.navdp import (
    NavDPError,
    NavDPPointgoalClient,
    anchor_trajectory_to_world,
    point_to_pointgoal,
    world_to_body_2d,
)
from sparx_agency.core.planning.planners.common.utils_2d import arclength_fraction_2d

# Top-level mode.
_PRIMARY = "primary"     # echo A*; count consecutive A* failures
_FALLBACK = "fallback"   # drive NavDP toward the final goal; count consecutive A* successes

# Leg sub-state (only meaningful while FALLBACK).
_HOLD = "hold"           # stopped: settle + infer (also "stop & wait" for a slow re-infer)
_FOLLOW = "follow"       # flying a NavDP leg; re-infer at the midpoint

# _run_inference outcomes.
_LEG = "leg"                 # a new leg was published
_ARRIVED = "arrived"         # within arrival radius of the goal -> hold
_UNAVAILABLE = "unavailable" # fatal stream mismatch -> shutting down
_PENDING = "pending"         # transient/slow failure -> hold and retry
_ABORTED = "aborted"         # left fallback mid-call -> caller does nothing


def _param_bool(name, default):
    """Read a boolean rosparam, failing loud on a non-boolean string.

    roslaunch type-infers a literal ``true``/``false`` to a real bool, but a typo
    (``fales``) would arrive as a string ``bool()`` silently treats as ``True``.
    Parse explicitly and raise instead (CLAUDE.md: prefer errors over fallbacks).
    """
    v = rospy.get_param(name, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError("%s must be a boolean (true/false), got %r" % (name, v))


class AStarNavDPFallbackNode:
    def __init__(self):
        rospy.init_node("astar_navdp_fallback")
        G = rospy.get_param

        self.frame_id = G("~frame_id", "world")

        # ── A* source (status + path) + outputs ──────────────────────
        self.status_topic = G("~astar_status_topic", "/path/astar_status")
        self.astar_path_topic = G("~astar_path_topic", "/path/waypoints_astar")
        self.path_topic = G("~path_topic", "/path/waypoints_fallback")
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_navdp_full")
        # Mission goal (same topic astar_planner replans on; bev_click_goal publishes
        # here). NavDP is aimed at THIS world point, converted to the body frame.
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        gx = G("~goal_x", None)
        gy = G("~goal_y", None)
        self.goal_world = ((float(gx), float(gy))
                           if gx is not None and gy is not None else None)

        # ── Hysteresis thresholds ────────────────────────────────────
        self.tick_hz = float(G("~tick_hz", 5.0))
        # Consecutive A* "no route" reports before stopping and engaging NavDP.
        self.fail_confirm = max(1, int(G("~astar_fail_confirm", 3)))
        # Consecutive A* "route found" reports before handing control back to A*.
        # Keep >= fail_confirm so the return is deliberately sticky (anti-zigzag).
        self.ok_confirm = max(1, int(G("~astar_ok_confirm", 5)))

        # ── NavDP leg behaviour (mirrors combination) ────────────────
        self.engage_settle_s = float(G("~engage_settle_s", 1.0))
        self.leg_fraction = float(G("~leg_execute_fraction", 0.5))
        self.leg_timeout_s = float(G("~leg_timeout_s", 10.0))
        self.leg_endpoint_radius_m = float(G("~leg_endpoint_radius_m", 0.4))
        # Within this range of the goal, NavDP has arrived -> hold (mission done).
        self.arrival_radius_m = float(G("~arrival_radius_m", 0.5))

        # ── Image transport (mirrors navdp_click / combination) ──────
        self.image_transport = str(G("~image_transport", "frame_path")).strip().lower()
        if self.image_transport not in ("frame_path", "topic"):
            raise ValueError("~image_transport must be 'frame_path' or 'topic', "
                             "got %r" % self.image_transport)
        _fp = self.image_transport == "frame_path"
        self.rgb_topic = G("~rgb_topic",
                           "/xtend/rgb_frame_path" if _fp else "/xtend/rgb")
        self.depth_topic = G("~depth_topic",
                             "/xtend/depth_frame_path" if _fp else "/xtend/depth_m")
        self.pose_topic = G("~pose_topic", "/xtend/localization")
        self.pose_type = G("~pose_type", "pose_stamped")
        self.camera_info_topic = G("~camera_info_topic", "")  # "" -> use params

        # ── Camera intrinsics (MUST match the NavDP stream) ──────────
        self.intr = Intrinsics(
            width=int(G("~img_width", 504)),
            height=int(G("~img_height", 294)),
            fx=float(G("~fx", 322.6351083474948)),
            fy=float(G("~fy", 323.3893307141174)),
            cx=float(G("~cx", 242.06479658679714)),
            cy=float(G("~cy", 90.03019076680604)))

        self.depth_max_m = float(G("~depth_max_m", 5.0))
        self.client = NavDPPointgoalClient(
            "http://127.0.0.1:%d" % int(G("~port", 8888)),
            timeout_s=float(G("~timeout_s", 10.0)),
            depth_max_m=self.depth_max_m,
            logger=rospy.logwarn)

        # ── Shared state (callbacks write, tick reads) ───────────────
        self.rgb = None
        self.depth = None
        self.altitude = float(G("~default_altitude", 1.0))   # until a pose arrives
        self.pose_xyyaw = None                               # (x, y, yaw) world
        self.astar_msg = None                                # last A* Path (for echo)
        self._echoed_msg = None                              # last A* Path published (dedup)
        self.leg_pts = None                                  # list[Pose2D] current leg
        self.leg_t = None                                    # rospy.Time the leg was published
        self.hold_t = None                                   # rospy.Time the HOLD began
        self.settle_until = None                             # rospy.Time the brake-settle ends
        self.fail_streak = 0                                 # consecutive A* "no route"
        self.ok_streak = 0                                   # consecutive A* "route found"
        self._arrived_held = False                           # published the arrival hold already
        self.mode = _PRIMARY
        self.leg_state = _HOLD
        self.n_legs = 0
        self._got_cam_info = False
        self._reset_done = False
        self._streams_checked = False

        # ── Publishers (latched; before subscribers so the A* echo never races) ──
        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.pub_full = rospy.Publisher(self.full_path_topic, Path, queue_size=1, latch=True)

        # ── Subscriptions ────────────────────────────────────────────
        if self.image_transport == "frame_path":
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, String, self._depth_path_cb, queue_size=2)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=2)
        if self.pose_type == "pose_stamped":
            rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_stamped_cb, queue_size=10)
        elif self.pose_type == "pose":
            rospy.Subscriber(self.pose_topic, Pose, self._pose_cb, queue_size=10)
        else:
            raise ValueError("~pose_type must be 'pose' or 'pose_stamped', got %r"
                             % self.pose_type)
        if self.camera_info_topic:
            rospy.Subscriber(self.camera_info_topic, CameraInfo, self._cam_info_cb, queue_size=1)
        rospy.Subscriber(self.astar_path_topic, Path, self._astar_path_cb, queue_size=1)
        rospy.Subscriber(self.status_topic, Bool, self._status_cb, queue_size=10)
        rospy.Subscriber(self.goal_topic, Point, self._goal_cb, queue_size=1)

    # ─── Sensor ingestion (mirrors combination_planner_node) ─────────
    def _rgb_path_cb(self, msg):
        try:
            path = parse_frame_path_message(msg.data).path
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None for %s" % path)
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "fallback: dropping RGB frame-path (%s)", e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _depth_path_cb(self, msg):
        try:
            path = parse_frame_path_message(msg.data).path
            arr = np.squeeze(np.load(path))
            if arr.ndim != 2:
                raise ValueError("depth %s has shape %r; expected HxW" % (path, arr.shape))
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "fallback: dropping depth frame-path (%s)", e)
            return
        self.depth = np.ascontiguousarray(arr, dtype=np.float32)

    def _rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.rgb = arr.copy()

    def _depth_cb(self, msg):
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(msg.data, np.float32).reshape(
                msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            self.depth = (np.frombuffer(msg.data, np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) / 1000.0)
        else:
            rospy.logwarn_throttle(5.0, "fallback: unsupported depth encoding %r "
                                   "(need 32FC1 or 16UC1); ignoring frame", msg.encoding)

    def _pose_cb(self, msg):
        yaw = se3.yaw_from_quaternion((msg.orientation.x, msg.orientation.y,
                                       msg.orientation.z, msg.orientation.w))
        # altitude FIRST, then the (x, y, yaw) tuple, so a non-None pose always has
        # its co-temporal altitude already in place.
        self.altitude = float(msg.position.z)
        self.pose_xyyaw = (float(msg.position.x), float(msg.position.y), yaw)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    def _cam_info_cb(self, msg):
        if self._reset_done:
            return
        if any(msg.K):
            fx, fy, cx, cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
        elif any(msg.P):
            fx, fy, cx, cy = msg.P[0], msg.P[5], msg.P[2], msg.P[6]
        else:
            return
        self.intr = Intrinsics(width=int(msg.width), height=int(msg.height),
                               fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
        self._got_cam_info = True

    # ─── A* echo + goal + status ─────────────────────────────────────
    def _echo_astar(self, force=False):
        """Publish the latest A* path on the fallback topic (the primary route).

        Deduplicated by message identity -- A* only republishes on a genuine plan
        change, so this never spams the follower (which resets progress on every
        message). ``force`` re-publishes even an unchanged path (to resume A* after
        a NavDP episode).
        """
        if self.astar_msg is None:
            return False
        if not force and self.astar_msg is self._echoed_msg:
            return False
        self.pub_path.publish(self.astar_msg)
        self._echoed_msg = self.astar_msg
        return True

    def _astar_path_cb(self, msg):
        self.astar_msg = msg
        if self.mode == _PRIMARY:
            self._echo_astar()                # follow A* directly while primary

    def _goal_cb(self, msg):
        self.goal_world = (float(msg.x), float(msg.y))
        self._arrived_held = False            # a new goal is not reached yet

    def _status_cb(self, msg):
        """One A* planning-attempt outcome -> update the two hysteresis streaks."""
        if bool(msg.data):
            self.ok_streak += 1
            self.fail_streak = 0
        else:
            self.fail_streak += 1
            self.ok_streak = 0

    # ─── NavDP server handshake / guards ─────────────────────────────
    def _ensure_reset(self):
        """Reset NavDP with the current intrinsics (lazy: NavDP may be down)."""
        if self._reset_done:
            return True
        if self.client.reset(self.intr):
            self._reset_done = True
            rospy.loginfo("fallback: NavDP reset OK (%s)", self.client.url)
            return True
        return False

    def _streams_ok(self, rgb, depth):
        """Fail loud (once) if RGB and depth are not the same resolution."""
        if self._streams_checked:
            return True
        rgb_hw, depth_hw = rgb.shape[:2], depth.shape[:2]
        if rgb_hw != depth_hw:
            rospy.logfatal("fallback: RGB %dx%d and depth %dx%d differ; they must be "
                           "aligned. Fix the stream.", rgb_hw[1], rgb_hw[0],
                           depth_hw[1], depth_hw[0])
            rospy.signal_shutdown("RGB/depth resolution mismatch")
            return False
        if rgb_hw != (self.intr.height, self.intr.width):
            rospy.logwarn("fallback: stream is %dx%d but intrinsics are %dx%d; goals "
                          "will be geometrically wrong -- pass matching intrinsics.",
                          rgb_hw[1], rgb_hw[0], self.intr.width, self.intr.height)
        self._streams_checked = True
        return True

    def _snapshot(self):
        """One CONSISTENT frame for a decision: capture the sensor refs atomically,
        then validate. Returns ``{pose, alt, rgb, depth, goal}`` or ``None``.
        """
        pose, rgb, depth = self.pose_xyyaw, self.rgb, self.depth
        goal, alt = self.goal_world, self.altitude
        if (rgb is None or depth is None or pose is None or goal is None
                or not np.all(np.isfinite(pose))):
            return None
        return {"pose": pose, "alt": alt, "rgb": rgb.copy(),
                "depth": depth.copy(), "goal": goal}

    # ─── One NavDP inference (BLOCKING) ──────────────────────────────
    def _run_inference(self):
        """Aim NavDP at the final goal (body frame), ask for a leg, publish it.

        Blocks on the HTTP step; the drone keeps flying its latched leg (or holds)
        meanwhile. On success it publishes the leg and updates ``leg_*`` state.
        """
        snap = self._snapshot()
        if snap is None or not self._ensure_reset():
            return _PENDING                       # not ready / NavDP down -> retry/hold
        if not self._streams_ok(snap["rgb"], snap["depth"]):
            return _UNAVAILABLE                    # fatal stream mismatch -> shutting down
        ox, oy, oyaw = snap["pose"]
        gwx, gwy = snap["goal"]
        if math.hypot(gwx - ox, gwy - oy) <= self.arrival_radius_m:
            return _ARRIVED

        # Final mission goal expressed in the drone body frame, then scaled into
        # NavDP's input range (bearing preserved). e.g. world goal (0,-3) at pose
        # (0,-1) -> body (0,-2). Anchoring the reply back with THIS pose round-trips.
        fwd, left = world_to_body_2d(gwx, gwy, ox, oy, oyaw)
        gx, gy = point_to_pointgoal(fwd, left)

        t0 = time.time()
        result = self.client.pointgoal_step(snap["rgb"], snap["depth"], gx, gy,
                                            altitude=snap["alt"])
        dt = time.time() - t0
        if self.mode != _FALLBACK:                # left fallback during the blocking call
            return _ABORTED
        if result is None:
            rospy.logwarn("fallback: NavDP no result (%.1fs) -- holding, will retry", dt)
            return _PENDING
        try:
            traj = self.client.best_trajectory(result)
        except NavDPError as e:
            rospy.logwarn("fallback: %s (%.1fs) -- holding, will retry", e, dt)
            return _PENDING
        leg_world = anchor_trajectory_to_world(traj, ox, oy, oyaw)
        if len(leg_world) < 2:
            rospy.logwarn("fallback: NavDP leg too short (%d) -- holding, will retry",
                          len(leg_world))
            return _PENDING

        stamp = rospy.Time.now()
        self.pub_path.publish(self._make_path(leg_world, stamp))
        self.pub_full.publish(self._make_path(leg_world, stamp))
        self._echoed_msg = None                   # we left A*; a resume force-republishes
        self.leg_pts = [Pose2D(float(x), float(y)) for x, y in leg_world]
        self.leg_t = stamp
        self.n_legs += 1
        rospy.loginfo("fallback: leg #%d -> goal body=(%.2f, %.2f)  %d pts (infer %.1fs)",
                      self.n_legs, fwd, left, len(leg_world), dt)
        return _LEG

    # ─── Leg sub-state handlers (only while FALLBACK) ────────────────
    def _hold(self):
        """Stopped: wait out the brake-settle, then infer until a leg arrives."""
        if self._arrived_held:
            return                                # at the goal: hold, don't keep inferring
        if self.settle_until is not None and rospy.Time.now() < self.settle_until:
            return                                # let the drone brake to a clean stop
        outcome = self._run_inference()
        if outcome == _LEG:
            self.leg_state = _FOLLOW
        elif outcome == _ARRIVED:
            self._hold_on_arrival()
        elif outcome in (_ABORTED, _UNAVAILABLE):
            return
        else:
            rospy.loginfo_throttle(3.0, "fallback: holding for NavDP route ...")

    def _follow(self):
        """Fly the leg; re-infer at the midpoint (the 2nd half is the buffer)."""
        leg, pose = self.leg_pts, self.pose_xyyaw   # capture once (a cb may null leg_pts)
        if pose is None or not leg:
            return
        ox, oy, _ = pose
        pose2d = Pose2D(ox, oy)
        frac = arclength_fraction_2d(leg, pose2d)
        near_end = pose2d.distance_to(leg[-1]) <= self.leg_endpoint_radius_m
        timed_out = (self.leg_t is not None
                     and (rospy.Time.now() - self.leg_t).to_sec() > self.leg_timeout_s)
        if not (frac >= self.leg_fraction or near_end or timed_out):
            return                                # keep flying the leg
        why = "midpoint" if frac >= self.leg_fraction else ("endpoint" if near_end else "watchdog")
        rospy.loginfo("fallback: leg #%d re-infer (%s, frac=%.0f%%)",
                      self.n_legs, why, 100.0 * frac)
        outcome = self._run_inference()           # drone keeps flying the latched leg during this
        if outcome == _LEG:
            self.leg_state = _FOLLOW
        elif outcome == _ARRIVED:
            self._hold_on_arrival()
        elif outcome in (_ABORTED, _UNAVAILABLE):
            return
        else:                                     # _PENDING: leg flies out, then hold
            rospy.loginfo("fallback: NavDP not ready -- flying out the leg, then "
                          "holding for the new route")
            self.settle_until = None              # the leg is decelerating to its end
            self.leg_state = _HOLD                # do NOT publish a hold: let the leg coast

    def _hold_on_arrival(self):
        """Goal reached via NavDP: STOP once and stay put (no re-infer)."""
        if not self._arrived_held:
            rospy.loginfo("fallback: goal reached via NavDP (<= %.2fm) -- holding",
                          self.arrival_radius_m)
            self._publish_hold()
            self._arrived_held = True
        self.leg_state = _HOLD
        self.settle_until = None

    # ─── Mode transitions ────────────────────────────────────────────
    def _enter_fallback(self):
        """A* failed enough: STOP the drone and hand control to NavDP."""
        rospy.logwarn("fallback: A* found NO route on %d consecutive attempts -- "
                      "STOPPING and engaging NavDP", self.fail_streak)
        self.mode = _FALLBACK
        self.fail_streak = 0
        self.ok_streak = 0
        self._arrived_held = False
        self._publish_hold()                      # STOP for a clean, stationary inference
        self.hold_t = rospy.Time.now()
        self.settle_until = self.hold_t + rospy.Duration(self.engage_settle_s)
        self.leg_state = _HOLD

    def _resume_primary(self):
        """A* recovered: hand control back and republish the A* route."""
        rospy.loginfo("fallback: A* found a route on %d consecutive attempts -- "
                      "resuming A*", self.ok_streak)
        self.mode = _PRIMARY
        self.fail_streak = 0
        self.ok_streak = 0
        self.leg_pts = None
        self.leg_state = _HOLD
        self._arrived_held = False
        self._echo_astar(force=True)

    def _publish_hold(self):
        """Command the drone to STOP and hold at its current pose.

        A 2-point path at the current pose collapses (via the follower's reanchor)
        to a single waypoint the drone is already on, so it brakes to DONE. The
        path_corrector passes a coincident-point path through verbatim.
        """
        if self.pose_xyyaw is None:
            return
        ox, oy, _ = self.pose_xyyaw
        self.pub_path.publish(self._make_path([(ox, oy), (ox, oy)], rospy.Time.now()))
        self._echoed_msg = None                   # we left A*; a resume force-republishes
        self.leg_pts = None

    def _make_path(self, world_xy, stamp):
        """Build a latched world-frame ``nav_msgs/Path`` from ``(x, y)`` pairs."""
        m = Path()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        for wx, wy in world_xy:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0           # identity; follower derives heading
            m.poses.append(ps)
        return m

    # ─── Control tick ────────────────────────────────────────────────
    def _tick(self, _evt):
        # A single bad frame must NEVER kill the loop: an uncaught exception in a
        # rospy.Timer callback terminates the timer thread. Catch everything and
        # drop safely back to flying A*.
        try:
            if self.mode == _PRIMARY:
                self._echo_astar()                # keep flying A* (deduped)
                if self.fail_streak >= self.fail_confirm:
                    if self._ensure_reset():      # don't stop into a dead NavDP server
                        self._enter_fallback()
                    else:
                        rospy.logwarn_throttle(2.0, "fallback: A* failing but NavDP "
                                               "unreachable -- staying on A*")
                return
            # FALLBACK: recovery wins over continuing a leg (checked first every tick).
            if self.ok_streak >= self.ok_confirm:
                self._resume_primary()
                return
            if self.leg_state == _HOLD:
                self._hold()
            elif self.leg_state == _FOLLOW:
                self._follow()
        except Exception as e:                    # noqa: BLE001 -- resilience is the point
            rospy.logwarn_throttle(2.0, "fallback: tick error (%s: %s) -- following "
                                   "A*, retrying next frame", type(e).__name__, e)
            self.mode = _PRIMARY
            self.leg_pts = None
            self.fail_streak = 0
            self.ok_streak = 0
            try:
                self._echo_astar(force=True)
            except Exception:                     # noqa: BLE001
                pass

    # ─── Bring-up ────────────────────────────────────────────────────
    def start(self):
        if self.camera_info_topic:
            t0 = time.time()
            while (not rospy.is_shutdown() and not self._got_cam_info
                   and time.time() - t0 < 2.0):
                time.sleep(0.05)
            if not self._got_cam_info:
                rospy.logwarn("fallback: no %s yet -- using param intrinsics",
                              self.camera_info_topic)
        if not self._ensure_reset():
            rospy.logwarn("fallback: NavDP not reachable yet at %s -- will retry when "
                          "a rescue is needed", self.client.url)
        self._banner()
        rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self._tick)
        rospy.spin()

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("astar_navdp_fallback (A*-primary + NavDP rescue, hysteretic)")
        L("  A* status in = %s  (Bool per attempt)", self.status_topic)
        L("  A* path   in = %s", self.astar_path_topic)
        L("  goal      in = %s  init=%s", self.goal_topic,
          "(%.2f,%.2f)" % self.goal_world if self.goal_world else "none")
        L("  rgb       in = %s", self.rgb_topic)
        L("  depth     in = %s", self.depth_topic)
        L("  pose      in = %s  (%s)", self.pose_topic, self.pose_type)
        L("  navdp        = %s  (timeout %.0fs)", self.client.url, self.client.timeout_s)
        L("  path     out = %s  (raw -> path_corrector)", self.path_topic)
        L("  switch       = A* -> NavDP after %d consecutive fails; NavDP -> A* after "
          "%d consecutive successes", self.fail_confirm, self.ok_confirm)
        L("  leg          = re-infer at %.0f%% of each leg; arrive within %.2fm",
          100.0 * self.leg_fraction, self.arrival_radius_m)
        L("  intrinsics   : fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
          self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
          self.intr.width, self.intr.height)
        L("=" * 64)


def main():
    try:
        AStarNavDPFallbackNode().start()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The geometry, HTTP and
# progress maths live in core.planning.navdp / core.planning.planners; this node
# owns ROS I/O and the PRIMARY <-> FALLBACK hysteretic state machine.
#
#   sources/outputs:
#     ~astar_status_topic (/path/astar_status)   A* per-attempt Bool (True=route)
#     ~astar_path_topic (/path/waypoints_astar)  raw A* route, echoed while PRIMARY
#     ~path_topic (/path/waypoints_fallback)     arbitrated raw path -> path_corrector
#     ~full_path_topic (/path/waypoints_navdp_full)  full NavDP leg, display only
#     ~goal_topic (/waypoint_nav/goal)           mission goal (world); NavDP aims here
#     ~goal_x ~goal_y (unset)                     optional initial goal
#     ~frame_id (world)
#   hysteresis:
#     ~astar_fail_confirm (3)   consecutive A* "no route" before stopping for NavDP
#     ~astar_ok_confirm (5)     consecutive A* "route found" before resuming A*
#                               (keep >= fail_confirm: sticky return, anti-zigzag)
#     ~tick_hz (5.0)            state-machine rate
#   NavDP leg:
#     ~engage_settle_s (1.0)    brake-settle before the first inference (clean frame)
#     ~leg_execute_fraction (0.5)  re-infer after this fraction of a leg (rest buffers)
#     ~leg_timeout_s (10.0)     watchdog: re-infer if a leg runs this long
#     ~leg_endpoint_radius_m (0.4)  re-infer once within this of the leg end
#     ~arrival_radius_m (0.5)   within this of the goal -> hold (mission done)
#   image transport (mirrors navdp_click / combination):
#     ~image_transport (frame_path | topic)
#     ~rgb_topic ~depth_topic ~pose_topic (/xtend/localization) ~pose_type (pose_stamped)
#     ~camera_info_topic ('' = use the fx/fy/cx/cy params; K preferred over P)
#   camera (MUST match the live NavDP stream; launch wires these to navdp_*):
#     ~fx ~fy ~cx ~cy ~img_width (504) ~img_height (294)
#   NavDP server: ~port (8888) ~timeout_s (10.0) ~depth_max_m (5.0)
#   misc: ~default_altitude (1.0; used until the first pose arrives)
#
#   Behaviour: PRIMARY echoes A* on ~path_topic and watches ~astar_status_topic;
#   after ~astar_fail_confirm consecutive failures it STOPS the drone and engages
#   NavDP (only if the NavDP server is reachable). FALLBACK drives NavDP toward the
#   mission goal in the body frame, flying each leg to ~leg_execute_fraction then
#   re-inferring, until A* reports ~astar_ok_confirm consecutive successes -> resume
#   A*. NavDP arrival (<= ~arrival_radius_m) holds. There is NO visibility gate: the
#   goal is usually not visible (past the door / around the corner), which is why A*
#   failed -- NavDP is handed the bearing and plans the local avoidance.
# ============================================================================
