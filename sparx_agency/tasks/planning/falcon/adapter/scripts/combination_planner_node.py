#!/usr/bin/env python3
"""combination_planner_node.py -- FALCON "combination" mode: A* global + NavDP local.

A third navigation mode that fuses the two planners FALCON already has:

  * **A\\*** plans a collision-free global route to the mission goal (the
    ``astar_planner`` node keeps it fresh on ``/path/waypoints_astar``).
  * **NavDP** is a learned point-goal policy that produces a smooth, locally
    grounded trajectory toward a goal it can SEE in the current RGB-D frame.

This node is the arbiter that drives ``/path/waypoints_combo`` -- the single raw
path the planner-agnostic ``path_corrector`` -> ``trajectory_simplifier`` ->
``waypoint_follower`` chain flies. It has two regimes:

  1. **Disabled (default):** echo the live A* path straight through, so the drone
     follows A* exactly as in plain ``nav_mode:=astar``. The run can start here.
  2. **Enabled (a ``std_msgs/Bool True`` on ``~enable_topic``, or
     ``~start_enabled:=true`` for the whole run):** run the loop below, where A*
     only *steers* the local goal and NavDP supplies the flown trajectory.

Combination loop (one iteration per NavDP leg)::

    1. Read the latest A* route to the destination (it replans continuously).
    2. Pick the FARTHEST A* waypoint visible in the current camera frame
       (in front, projects in-frame, and -- by default -- not occluded by a
       nearer wall). See core ``select_farthest_visible_waypoint``.
    3. Express that waypoint in NavDP's body frame (drone = origin, +x forward)
       and ask NavDP for a trajectory toward it; anchor the body trajectory back
       to the world at the SAME pose, and publish it on ``/path/waypoints_combo``
       (the corrector recentres + the simplifier cleans it, then it is flown).
    4. Track the leg until the drone passes its midpoint (arclength fraction
       ``~leg_execute_fraction``), then go to 1 -- NavDP is accurate near the
       camera, so re-grounding every half-leg keeps it in its sweet spot while A*
       supplies the global direction.

All the maths is ROS-free and unit-tested in ``core.planning.navdp`` /
``core.planning.planners.common``:
  * world A* waypoint -> body-frame goal     (geometry.world_to_body_2d + point_to_pointgoal)
  * farthest visible waypoint                (local_goal.select_farthest_visible_waypoint)
  * NavDP HTTP request/response              (client.NavDPPointgoalClient)
  * body trajectory -> world path            (geometry.anchor_trajectory_to_world)
  * progress along the flown leg             (utils_2d.arclength_fraction_2d)
This node owns ONLY ROS concerns: subscriptions, the small state machine, the
NavDP server handshake and publishing the world path. It runs HEADLESS (no GUI);
the route is seen on the BEV viewer.

RGB/depth/pose ingestion mirrors ``navdp_click_node`` (the same ``frame_path`` |
``topic`` transport and pose handling), and intrinsics MUST match the stream
NavDP receives -- the launch wires ``~fx ~fy ~cx ~cy`` to the shared ``cam_*``
args, exactly as for ``navdp_click``. The NavDP server (``navdp_trt_server.py``,
default ``127.0.0.1:8888``) must be up before combination is enabled.

Run:
    rosrun falcon_adapter combination_planner_node.py
See the file footer for the full rosparam list.
"""
import time

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.math import se3
from sparx_agency.core.common.types import Intrinsics, Pose2D
from sparx_agency.core.planning.navdp import (
    NAVDP_MAX_FWD_M,
    NAVDP_MAX_LAT_M,
    NavDPError,
    NavDPPointgoalClient,
    anchor_trajectory_to_world,
    select_farthest_visible_waypoint,
)
from sparx_agency.core.planning.planners.common.utils_2d import arclength_fraction_2d

# State machine
_ASTAR = "astar_follow"     # disabled: echo A* through to the follower
_INFER = "combo_infer"      # enabled: pick a local goal + ask NavDP for a leg
_FOLLOW = "combo_follow"    # enabled: fly the leg until its midpoint


def _param_bool(name, default):
    """Read a boolean rosparam, failing loud on a non-boolean string.

    roslaunch type-infers a literal ``true``/``false`` to a real bool, but a
    typo (``fales``) would arrive as a string that ``bool()`` silently treats as
    ``True``. Parse explicitly and raise instead (CLAUDE.md: prefer errors over
    silent fallbacks).
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


class CombinationPlannerNode:
    def __init__(self):
        rospy.init_node("combination_planner")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "")
        self.frame_id = G("~frame_id", "world")

        # ── A* source + outputs ──────────────────────────────────────
        self.astar_path_topic = G("~astar_path_topic", "/path/waypoints_astar")
        # The arbitrated raw path the corrector consumes (A* echo while disabled;
        # the chosen NavDP leg while enabled). Mirrors how A*/NavDP each own a raw
        # planner topic; the corrector + simplifier downstream are unchanged.
        self.path_topic = G("~path_topic", "/path/waypoints_combo")
        # The full NavDP leg, display-only (the BEV viewer draws it).
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_navdp_full")

        # ── Combination behaviour ────────────────────────────────────
        self.enable_topic = G("~enable_topic", "/combination/enable")
        self.combo_enabled = _param_bool("~start_enabled", False)
        self.tick_hz = float(G("~tick_hz", 5.0))
        # Re-infer once the drone passes this fraction of the current NavDP leg
        # (0.5 = midpoint). On the FINAL leg (the local goal IS the mission goal)
        # almost the whole leg is flown before refining -- see _check_progress.
        self.leg_fraction = float(G("~leg_execute_fraction", 0.5))
        # Final-approach hand-off: when the local goal IS the mission goal and is
        # within this range, fly A* straight in (the follower reaches the true
        # goal and stops) instead of re-inferring ever-shorter NavDP legs.
        self.final_handoff_m = float(G("~final_handoff_m", 1.5))
        # Watchdog: re-infer if a leg has been followed this long, or the drone is
        # within this radius of the leg end -- so a leg whose flown (corrected)
        # path is shorter than the raw leg, or that doubles back, cannot deadlock
        # the midpoint trigger.
        self.leg_timeout_s = float(G("~leg_timeout_s", 8.0))
        self.leg_endpoint_radius_m = float(G("~leg_endpoint_radius_m", 0.4))

        # ── Visibility knobs (see core point_visible) ────────────────
        self.require_unoccluded = _param_bool("~require_unoccluded", True)
        self.vis_depth_tol_m = float(G("~visibility_depth_tol_m", 0.5))
        self.vis_patch_half = int(G("~visibility_patch_half_px", 6))
        self.min_goal_fwd_m = float(G("~min_goal_fwd_m", 0.5))

        # ── Image transport (mirrors navdp_click) ────────────────────
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
        # Short timeout: a NavDP step runs inside the control tick, so a hung
        # server must not stall the loop for long (the drone keeps flying the last
        # latched leg meanwhile). Lower than navdp_click's interactive 30 s.
        self.client = NavDPPointgoalClient(
            "http://127.0.0.1:%d" % int(G("~port", 8888)),
            timeout_s=float(G("~timeout_s", 5.0)),
            depth_max_m=self.depth_max_m,
            logger=rospy.logwarn)

        # ── Shared state (callbacks write, tick reads) ───────────────
        self.rgb = None
        self.depth = None
        self.altitude = float(G("~default_altitude", 1.0))   # until a pose arrives
        self.pose_xyyaw = None                               # (x, y, yaw) world
        self.astar_pts = None                                # list[(x, y)] world
        self.astar_msg = None                                # last A* Path (for echo)
        self._echoed_msg = None                              # last A* Path published (dedup)
        self.leg_pts = None                                  # list[Pose2D] flown leg
        self.leg_final = False                               # local goal == mission goal
        self.leg_t = None                                    # rospy.Time the leg was published
        self.state = _ASTAR if not self.combo_enabled else _INFER
        self.n_legs = 0
        self._got_cam_info = False
        self._reset_done = False
        self._streams_checked = False

        # ── Publishers (latched: the follower holds the last leg). Created
        #    BEFORE the subscribers so an immediately-latched A* path can be
        #    echoed from _astar_cb without racing the publisher's creation. ──
        self.pub_path = rospy.Publisher(self.path_topic, Path,
                                        queue_size=1, latch=True)
        self.pub_full = rospy.Publisher(self.full_path_topic, Path,
                                        queue_size=1, latch=True)

        # ── Subscriptions ────────────────────────────────────────────
        if self.image_transport == "frame_path":
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, String, self._depth_path_cb, queue_size=2)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=2)
        if self.pose_type == "pose_stamped":
            rospy.Subscriber(self.pose_topic, PoseStamped,
                             self._pose_stamped_cb, queue_size=10)
        elif self.pose_type == "pose":
            rospy.Subscriber(self.pose_topic, Pose, self._pose_cb, queue_size=10)
        else:
            raise ValueError("~pose_type must be 'pose' or 'pose_stamped', got %r"
                             % self.pose_type)
        if self.camera_info_topic:
            rospy.Subscriber(self.camera_info_topic, CameraInfo,
                             self._cam_info_cb, queue_size=1)
        rospy.Subscriber(self.astar_path_topic, Path, self._astar_cb, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self._enable_cb, queue_size=1)

    # ─── Sensor ingestion (mirrors navdp_click_node) ─────────────────
    def _rgb_path_cb(self, msg):
        try:
            path = parse_frame_path_message(msg.data).path
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None for %s" % path)
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "combination: dropping RGB frame-path (%s)", e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _depth_path_cb(self, msg):
        try:
            path = parse_frame_path_message(msg.data).path
            arr = np.squeeze(np.load(path))
            if arr.ndim != 2:
                raise ValueError("depth %s has shape %r; expected HxW"
                                 % (path, arr.shape))
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "combination: dropping depth frame-path (%s)", e)
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
            rospy.logwarn_throttle(5.0, "combination: unsupported depth encoding "
                                   "%r (need 32FC1 or 16UC1); ignoring frame",
                                   msg.encoding)

    def _pose_cb(self, msg):
        yaw = se3.yaw_from_quaternion((msg.orientation.x, msg.orientation.y,
                                       msg.orientation.z, msg.orientation.w))
        # Set altitude FIRST, then the (x, y, yaw) tuple, so a non-None pose_xyyaw
        # always has its co-temporal altitude already in place.
        self.altitude = float(msg.position.z)
        self.pose_xyyaw = (float(msg.position.x), float(msg.position.y), yaw)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    def _cam_info_cb(self, msg):
        # Prefer the raw 3x3 K (the loaded depth is unrectified); fall back to P.
        # Latched at reset -- a fixed camera's intrinsics are static.
        if self._reset_done:
            return
        if any(msg.K):
            fx, fy, cx, cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
        elif any(msg.P):
            fx, fy, cx, cy = msg.P[0], msg.P[5], msg.P[2], msg.P[6]
        else:
            return
        self.intr = Intrinsics(width=int(msg.width), height=int(msg.height),
                               fx=float(fx), fy=float(fy),
                               cx=float(cx), cy=float(cy))
        self._got_cam_info = True

    # ─── A* echo + enable signal ─────────────────────────────────────
    def _echo_astar(self, force=False):
        """Publish the latest A* path on the combo topic (the always-available
        fallback). Deduplicated by message identity -- A* itself only republishes
        on a genuine plan change, so this never spams the follower (which has no
        path dedup and resets progress on every message). ``force`` re-publishes
        even an unchanged path, to resume A* after a NavDP leg was flown.
        """
        if self.astar_msg is None:
            return False
        if not force and self.astar_msg is self._echoed_msg:
            return False
        self.pub_path.publish(self.astar_msg)
        self._echoed_msg = self.astar_msg
        return True

    def _astar_cb(self, msg):
        self.astar_msg = msg
        self.astar_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not self.combo_enabled:
            self._echo_astar()                # follow A* directly while disabled

    def _enable_cb(self, msg):
        want = bool(msg.data)
        if want == self.combo_enabled:
            return
        self.combo_enabled = want
        if want:
            rospy.loginfo("combination: ENABLED -- NavDP local legs steered by A*")
            self.state = _INFER
            self.leg_pts = None
            self.leg_final = False
        else:
            rospy.loginfo("combination: DISABLED -- following A* directly")
            self.state = _ASTAR
            self._echo_astar(force=True)      # resume A* immediately

    # ─── NavDP server handshake ──────────────────────────────────────
    def _ensure_reset(self):
        """Reset NavDP with the current intrinsics (lazy: NavDP may be down)."""
        if self._reset_done:
            return True
        if self.client.reset(self.intr):
            self._reset_done = True
            rospy.loginfo("combination: NavDP reset OK (%s)", self.client.url)
            return True
        return False

    def _streams_ok(self, rgb, depth):
        """Fail loud (once) if RGB and depth are not the same resolution.

        Both are sent to NavDP and depth is indexed at projected pixels, so they
        must describe one image; a mismatch makes every goal/leg geometrically
        wrong. Warn (non-fatal) if the stream resolution differs from intrinsics.
        """
        if self._streams_checked:
            return True
        rgb_hw, depth_hw = rgb.shape[:2], depth.shape[:2]
        if rgb_hw != depth_hw:
            rospy.logfatal("combination: RGB %dx%d and depth %dx%d differ; they "
                           "must be aligned. Fix the stream.", rgb_hw[1], rgb_hw[0],
                           depth_hw[1], depth_hw[0])
            rospy.signal_shutdown("RGB/depth resolution mismatch")
            return False
        if rgb_hw != (self.intr.height, self.intr.width):
            rospy.logwarn("combination: stream is %dx%d but intrinsics are %dx%d; "
                          "goals will be geometrically wrong -- pass intrinsics "
                          "matching the live stream.", rgb_hw[1], rgb_hw[0],
                          self.intr.width, self.intr.height)
        self._streams_checked = True
        return True

    # ─── Control tick ────────────────────────────────────────────────
    def _tick(self, _evt):
        if not self.combo_enabled:
            return                          # A* echo handled in _astar_cb
        if self.state == _INFER:
            self._do_infer()
        elif self.state == _FOLLOW:
            self._check_progress()

    def _do_infer(self):
        """Pick a local goal, ask NavDP for a leg, publish it.

        Whenever a NavDP leg cannot be produced (sensors/A* not ready, NavDP
        unreachable, no visible waypoint, or the final approach), the node FALLS
        BACK to the A* path so the drone always has a route toward the goal --
        never a dead stop -- and retries the inference on the next tick.
        """
        if (self.rgb is None or self.depth is None or self.pose_xyyaw is None
                or not self.astar_pts or len(self.astar_pts) < 2):
            rospy.logwarn_throttle(2.0, "combination: waiting for RGB/depth/pose/A* "
                                   "-- following A* if available")
            self._echo_astar()
            return
        if not self._ensure_reset():
            rospy.logwarn_throttle(2.0, "combination: NavDP not reachable at %s "
                                   "-- following A*", self.client.url)
            self._echo_astar()
            return

        rgb, depth = self.rgb.copy(), self.depth.copy()
        if not self._streams_ok(rgb, depth):
            return                            # fatal mismatch -> node is shutting down
        ox, oy, oyaw = self.pose_xyyaw
        alt = self.altitude
        waypoints = self.astar_pts

        goal = select_farthest_visible_waypoint(
            waypoints, ox, oy, oyaw, depth, self.intr, cam_height_m=max(alt, 0.1),
            require_unoccluded=self.require_unoccluded,
            depth_tol_m=self.vis_depth_tol_m, depth_patch_half=self.vis_patch_half,
            min_fwd_m=self.min_goal_fwd_m,
            max_fwd_m=NAVDP_MAX_FWD_M, max_lat_m=NAVDP_MAX_LAT_M)
        if goal is None:
            rospy.logwarn_throttle(2.0, "combination: no A* waypoint visible -- "
                                   "following A*")
            self._echo_astar()
            return

        # Final approach: when the local goal IS the mission goal and close, hand
        # the last stretch to A* (the follower drives straight to the true goal and
        # stops) rather than re-inferring ever-shorter NavDP legs that jitter and
        # stall ~min_goal_fwd_m short.
        is_final = goal.index >= len(waypoints) - 1
        if is_final and goal.body[0] <= self.final_handoff_m:
            rospy.loginfo_throttle(2.0, "combination: final approach (%.2fm) -- "
                                   "handing off to A*", goal.body[0])
            # Deduped (NOT force): this branch stays in _INFER and re-runs every
            # tick, so a force-republish would resend the identical A* path at
            # tick_hz and continuously reset the no-dedup follower's progress.
            # Dedup publishes once on handoff (after a leg, _echoed_msg is None)
            # and once per genuine A* replan -- which is exactly what's wanted.
            self._echo_astar()
            return

        gx, gy = goal.goal
        result = self.client.pointgoal_step(rgb, depth, gx, gy, altitude=alt)
        # M3: a disable (or other state change) during the blocking HTTP call must
        # not publish a stale leg. Re-check before mutating state / publishing.
        if not self.combo_enabled or self.state != _INFER:
            return
        if result is None:
            rospy.logwarn("combination: NavDP returned no result -- following A*")
            self._echo_astar()
            return
        try:
            traj = self.client.best_trajectory(result)
        except NavDPError as e:
            rospy.logwarn("combination: %s -- following A*", e)
            self._echo_astar()
            return

        leg_world = anchor_trajectory_to_world(traj, ox, oy, oyaw)
        if len(leg_world) < 2:
            rospy.logwarn("combination: NavDP leg too short (%d) -- following A*",
                          len(leg_world))
            self._echo_astar()
            return

        stamp = rospy.Time.now()
        self.pub_path.publish(self._make_path(leg_world, stamp))
        self.pub_full.publish(self._make_path(leg_world, stamp))
        self._echoed_msg = None               # next fallback force-republishes A*
        self.leg_pts = [Pose2D(float(x), float(y)) for x, y in leg_world]
        self.leg_final = is_final
        self.leg_t = stamp
        self.state = _FOLLOW
        self.n_legs += 1
        rospy.loginfo("combination: leg #%d -> A* wp[%d]=(%.2f, %.2f) fwd=%.2fm  "
                      "%d navdp pts%s", self.n_legs, goal.index, goal.world[0],
                      goal.world[1], goal.body[0], len(leg_world),
                      "  (final)" if is_final else "")

    def _check_progress(self):
        """Re-infer once the drone passes the leg midpoint, reaches its end, or a
        watchdog fires -- so the loop cannot deadlock if the flown (corrected) path
        is shorter than the raw leg or the raw leg doubles back."""
        if self.pose_xyyaw is None or not self.leg_pts:
            return
        ox, oy, _ = self.pose_xyyaw
        pose2d = Pose2D(ox, oy)
        frac = arclength_fraction_2d(self.leg_pts, pose2d)
        # Normal legs hand off at the midpoint (NavDP drifts further out); a final
        # leg flies almost all the way before refining.
        threshold = max(self.leg_fraction, 0.9) if self.leg_final else self.leg_fraction
        near_end = pose2d.distance_to(self.leg_pts[-1]) <= self.leg_endpoint_radius_m
        timed_out = (self.leg_t is not None
                     and (rospy.Time.now() - self.leg_t).to_sec() > self.leg_timeout_s)
        if frac >= threshold or near_end or timed_out:
            why = ("midpoint" if frac >= threshold
                   else ("endpoint" if near_end else "watchdog"))
            rospy.loginfo("combination: leg #%d re-infer (%s, frac=%.0f%%)",
                          self.n_legs, why, 100.0 * frac)
            self.state = _INFER
            self._do_infer()

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
            ps.pose.orientation.w = 1.0       # identity; follower derives heading
            m.poses.append(ps)
        return m

    # ─── Bring-up ────────────────────────────────────────────────────
    def start(self):
        # If intrinsics come from a camera_info topic, wait briefly so NavDP is
        # reset with the stream's true intrinsics before the first inference.
        if self.camera_info_topic:
            t0 = time.time()
            while (not rospy.is_shutdown() and not self._got_cam_info
                   and time.time() - t0 < 2.0):
                time.sleep(0.05)
            if not self._got_cam_info:
                rospy.logwarn("combination: no %s yet -- using param intrinsics",
                              self.camera_info_topic)
        # Attempt an initial reset. NavDP may be down; we retry lazily before the
        # first leg, so A*-following still works until the server comes up.
        if not self._ensure_reset():
            rospy.logwarn("combination: NavDP not reachable yet at %s -- will retry "
                          "when a leg is needed", self.client.url)
        self._banner()
        rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self._tick)
        rospy.spin()

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("combination_planner (A* global route + NavDP local legs)")
        L("  A* path  in = %s", self.astar_path_topic)
        L("  rgb      in = %s", self.rgb_topic)
        L("  depth    in = %s", self.depth_topic)
        L("  pose     in = %s  (%s)", self.pose_topic, self.pose_type)
        L("  enable   in = %s  (start_enabled=%s)", self.enable_topic,
          self.combo_enabled)
        L("  navdp       = %s", self.client.url)
        L("  path    out = %s  (raw -> path_corrector)", self.path_topic)
        L("  leg handoff = %.0f%% of each NavDP leg (re-infer at the midpoint)",
          100.0 * self.leg_fraction)
        L("  visibility  = %s%s",
          "in-frame + unoccluded" if self.require_unoccluded else "in-frame only",
          "" if not self.require_unoccluded
          else " (depth_tol=%.2fm)" % self.vis_depth_tol_m)
        L("  intrinsics  : fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
          self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
          self.intr.width, self.intr.height)
        L("=" * 64)


def main():
    try:
        CombinationPlannerNode().start()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The geometry, selection,
# HTTP and progress maths live in core.planning.navdp / core.planning.planners;
# this node owns ROS I/O and the small A*<->NavDP state machine.
#
#   sources/outputs:
#     ~astar_path_topic (/path/waypoints_astar)  raw A* route (nav_msgs/Path)
#     ~path_topic (/path/waypoints_combo)        arbitrated raw path -> path_corrector
#     ~full_path_topic (/path/waypoints_navdp_full)  full NavDP leg, display only
#     ~frame_id (world)
#   combination:
#     ~enable_topic (/combination/enable)        std_msgs/Bool: True=combine, False=A*
#     ~start_enabled (false)                     start in combination mode (whole run)
#     ~tick_hz (5.0)                             state-machine rate
#     ~leg_execute_fraction (0.5)                re-infer after this fraction of a leg
#     ~final_handoff_m (1.5)                     hand the final approach to A* within this range
#     ~leg_timeout_s (8.0)                       watchdog: re-infer if a leg runs this long
#     ~leg_endpoint_radius_m (0.4)               re-infer once within this of the leg end
#   When a leg cannot be produced (NavDP down / no result / no visible waypoint /
#   final approach) the node FALLS BACK to publishing the A* path, so the drone
#   always has a route; it retries inference each tick.
#   visibility (core point_visible):
#     ~require_unoccluded (true)                 require line-of-sight (depth) not just FOV
#     ~visibility_depth_tol_m (0.5)              slack on the occlusion test
#     ~visibility_patch_half_px (6)              depth patch half-size at the pixel
#     ~min_goal_fwd_m (0.5)                      ignore waypoints nearer than this
#   image transport (mirrors navdp_click):
#     ~image_transport (frame_path | topic)
#     ~rgb_topic ~depth_topic ~pose_topic (/xtend/localization) ~pose_type (pose_stamped)
#     ~camera_info_topic ('' = use the fx/fy/cx/cy params; K preferred over P)
#   camera (MUST match the live NavDP stream; launch wires these to cam_*):
#     ~fx ~fy ~cx ~cy ~img_width (504) ~img_height (294)
#   NavDP server: ~port (8888) ~timeout_s (5.0; short -- runs in the control tick) ~depth_max_m (5.0)
#   misc: ~default_altitude (1.0; used until the first pose arrives) ~drone_ns ('')
# ============================================================================
