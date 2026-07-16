#!/usr/bin/env python3
"""combination_planner_node.py -- FALCON "combination" mode: A* global + NavDP local.

A third navigation mode that fuses the two planners FALCON already has:

  * **A\\*** plans a collision-free global route to the mission goal (the
    ``astar_planner`` node keeps it fresh on ``/path/waypoints_astar``).
  * **NavDP** is a learned point-goal policy that produces a smooth, locally
    grounded trajectory toward a goal it can SEE in the current RGB-D frame.

This node is the arbiter that drives ``/path/waypoints_combo`` -- the single raw
path the planner-agnostic ``path_corrector`` -> ``trajectory_simplifier`` ->
``waypoint_follower`` chain flies. When DISABLED it echoes the A* path straight
through (plain ``nav_mode:=astar``). When ENABLED (a ``std_msgs/Bool True`` on
``~enable_topic``, or ``~start_enabled:=true``) it runs a 3-state loop:

  CRUISE -- follow the A* route while watching for a waypoint that is VISIBLE in
            the current frame AND at least ``~min_engage_fwd_m`` ahead (a goal
            worth a NavDP leg). No such point yet -> keep flying A*.
  HOLD   -- a good point appeared: STOP the drone (clean, stationary frame --
            important on a slow Jetson), let it settle, then ask NavDP for a leg.
            This state is also the "stop and wait": when a re-infer is slow and
            the previous leg has been flown out, the drone holds here until the
            new route arrives (bounded by ``~max_wait_s``, then it resumes A*).
  FOLLOW -- fly the published NavDP leg. At the midpoint (``~leg_execute_fraction``)
            re-infer; the leg's *second half* is a latency buffer -- the drone
            keeps flying it during the (blocking) inference, so a fast reply means
            a seamless switch and a slow one means it coasts to the leg end and
            holds. NavDP is accurate near the camera, so re-grounding every
            half-leg keeps it in its sweet spot while A* supplies the direction.

If no waypoint is visible at a re-infer, or NavDP is unreachable, the node falls
back to flying A* -- the drone always has a route and never dead-stalls.

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

RGB/depth/pose ingestion mirrors ``navdp_click_node`` (same ``frame_path`` |
``topic`` transport and pose handling), and intrinsics MUST match the stream
NavDP receives -- the launch wires ``~fx ~fy ~cx ~cy`` to the shared ``cam_*``
args. The NavDP server (``navdp_trt_server.py``, default ``127.0.0.1:8888``) must
be up before combination is enabled (until then the node just flies A*).

Run:
    rosrun falcon_adapter combination_planner_node.py
See the file footer for the full rosparam list.
"""
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
    NAVDP_MAX_FWD_M,
    NAVDP_MAX_LAT_M,
    NavDPError,
    NavDPPointgoalClient,
    anchor_trajectory_to_world,
    select_farthest_visible_waypoint,
)
from sparx_agency.core.planning.planners.common.utils_2d import arclength_fraction_2d
from thinking import Thinker

# State machine (only meaningful while ENABLED; disabled -> A* echo in _astar_cb).
_CRUISE = "cruise"     # follow A*, watch for an engage-able visible point
_HOLD = "hold"         # stopped: settle + infer (also the "stop & wait" for a slow re-infer)
_FOLLOW = "follow"     # flying a NavDP leg; re-infer at the midpoint

# _run_inference outcomes.
_LEG = "leg"               # a new leg was published
_NO_POINT = "no_point"     # nothing visible -> caller should resume A*
_UNAVAILABLE = "unavailable"  # NavDP unreachable / bad stream -> resume A*
_FINAL = "final"           # the mission goal is the visible goal & close -> hand to A*
_PENDING = "pending"       # transient/slow failure -> hold and retry
_ABORTED = "aborted"       # disabled mid-call -> caller does nothing

# Operator-facing (text, level) for every reason _resume_astar is called with: the
# outcomes above that hand back to A*, plus the hold watchdog's "wait_timeout".
# Kept beside the codes they narrate so a new outcome cannot be added silently.
_RESUME_THOUGHTS = {
    _NO_POINT: ("No waypoint in view for NavDP -- back on the A* route", "info"),
    _UNAVAILABLE: ("NavDP is unavailable -- back on the A* route", "warn"),
    _FINAL: ("The mission goal is close -- flying the last stretch on the A* route",
             "info"),
    "wait_timeout": ("NavDP never sent a leg -- giving up on it and going back to "
                     "the A* route", "warn"),
}


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


class CombinationPlannerNode:
    def __init__(self):
        rospy.init_node("combination_planner")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "")
        self.frame_id = G("~frame_id", "world")

        # ── A* source + outputs ──────────────────────────────────────
        self.astar_path_topic = G("~astar_path_topic", "/path/waypoints_astar")
        self.path_topic = G("~path_topic", "/path/waypoints_combo")
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_navdp_full")
        # Same goal topic astar_planner replans on (bev_click_goal publishes here). A
        # new goal here means the operator clicked: it arms an IMMEDIATE re-inference
        # off the click's fresh A* route, instead of waiting out the current leg or
        # the cruise engage streak. The periodic same-goal A* replan does NOT arm it,
        # so the idle cadence is unchanged. Default matches astar/bev_click_goal.
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")

        # ── Combination behaviour ────────────────────────────────────
        self.enable_topic = G("~enable_topic", "/combination/enable")
        self.combo_enabled = _param_bool("~start_enabled", False)
        self.tick_hz = float(G("~tick_hz", 5.0))
        # Engage NavDP only when the farthest visible waypoint is at least this far
        # ahead -- a closer-only view is not worth a leg, so keep flying A*.
        self.min_engage_fwd_m = float(G("~min_engage_fwd_m", 1.5))
        # Hysteresis: require this many CONSECUTIVE engage-able detections before
        # stopping, so depth/occlusion flicker cannot cause stop/go lurching.
        self.engage_confirm_ticks = max(1, int(G("~engage_confirm_ticks", 2)))
        # After stopping to engage, wait this long for the drone to brake to a clean,
        # stationary frame before the first inference (matters on a slow Jetson).
        self.engage_settle_s = float(G("~engage_settle_s", 1.0))
        # Bounded hold while waiting for a (slow/failed) inference: keep stopped and
        # retrying for up to this long (checked BEFORE each blocking attempt), then
        # give up and resume A* so a dead/hung server never strands the drone. Keep
        # comfortably larger than ~timeout_s so several attempts fit.
        self.max_wait_s = float(G("~max_wait_s", 30.0))
        # Re-infer once the drone passes this fraction of the current NavDP leg
        # (0.5 = midpoint). A FINAL leg (local goal == mission goal) flies further.
        self.leg_fraction = float(G("~leg_execute_fraction", 0.5))
        # Final-approach hand-off: when the local goal IS the mission goal and within
        # this range, fly A* straight in (the follower reaches the true goal & stops).
        self.final_handoff_m = float(G("~final_handoff_m", 1.5))
        # Watchdog: re-infer if a leg has been followed this long, or the drone is
        # within this radius of the leg end -- so a flown (corrected) path shorter
        # than the raw leg, or one that doubles back, cannot deadlock the midpoint.
        self.leg_timeout_s = float(G("~leg_timeout_s", 10.0))
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
        # Generous timeout: a NavDP step is slow on a Jetson. While it blocks, the
        # drone keeps flying the latched leg (the buffer) or holds -- it never moves
        # un-commanded -- so a long timeout is safe and avoids premature give-up.
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
        self.astar_pts = None                                # list[(x, y)] world
        self.astar_msg = None                                # last A* Path (for echo)
        self._echoed_msg = None                              # last A* Path published (dedup)
        self.leg_pts = None                                  # list[Pose2D] current leg
        self.leg_final = False                               # local goal == mission goal
        self.leg_t = None                                    # rospy.Time the leg was published
        self.hold_t = None                                   # rospy.Time the HOLD began
        self.settle_until = None                             # rospy.Time the engage brake-settle ends
        self.engage_confirm = 0                              # consecutive engage-able detections
        self._new_goal_pending = False                       # a goal click arrived; await its A* path
        self._force_engage = False                           # that A* path landed -> re-infer NOW
        self.state = _CRUISE
        self.n_legs = 0
        self._got_cam_info = False
        self._reset_done = False
        self._streams_checked = False

        # Narrates the leg lifecycle (engage / leg / hand back to A*) onto the
        # shared thinking log; the per-tick loginfo below stays for the terminal.
        self.thinker = Thinker("combination_planner")

        # ── Publishers (latched; created BEFORE subscribers so _astar_cb can
        #    echo an immediately-latched A* path without racing the publisher). ──
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
        rospy.Subscriber(self.astar_path_topic, Path, self._astar_cb, queue_size=1)
        rospy.Subscriber(self.enable_topic, Bool, self._enable_cb, queue_size=1)
        rospy.Subscriber(self.goal_topic, Point, self._goal_cb, queue_size=1)

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
                raise ValueError("depth %s has shape %r; expected HxW" % (path, arr.shape))
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
            rospy.logwarn_throttle(5.0, "combination: unsupported depth encoding %r "
                                   "(need 32FC1 or 16UC1); ignoring frame", msg.encoding)

    def _pose_cb(self, msg):
        yaw = se3.yaw_from_quaternion((msg.orientation.x, msg.orientation.y,
                                       msg.orientation.z, msg.orientation.w))
        # altitude FIRST, then the (x, y, yaw) tuple, so a non-None pose_xyyaw always
        # has its co-temporal altitude already in place.
        self.altitude = float(msg.position.z)
        self.pose_xyyaw = (float(msg.position.x), float(msg.position.y), yaw)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    def _cam_info_cb(self, msg):
        # Prefer raw K (unrectified depth); fall back to P. Latched at reset.
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

    # ─── A* echo + enable signal ─────────────────────────────────────
    def _echo_astar(self, force=False):
        """Publish the latest A* path on the combo topic (the always-available
        fallback). Deduplicated by message identity -- A* only republishes on a
        genuine plan change, so this never spams the follower (which has no path
        dedup and resets progress on every message). ``force`` re-publishes even an
        unchanged path, to resume A* after a NavDP leg or a hold.
        """
        if self.astar_msg is None:
            return False
        if not force and self.astar_msg is self._echoed_msg:
            return False
        self.pub_path.publish(self.astar_msg)
        self._echoed_msg = self.astar_msg
        return True

    def _goal_cb(self, _msg):
        """A new goal was clicked. Arm an immediate re-inference, but wait for the
        click's fresh A* route to land in ``_astar_cb`` before firing (so the leg is
        planned against the new route, not the stale one). Every click re-arms, so
        re-clicking the same spot also forces a fresh leg."""
        self._new_goal_pending = True

    def _astar_cb(self, msg):
        self.astar_msg = msg
        self.astar_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self._new_goal_pending:
            # The route for the just-clicked goal has arrived -> re-infer immediately.
            self._new_goal_pending = False
            self._force_engage = True
        if not self.combo_enabled:
            self._echo_astar()                # follow A* directly while disabled

    def _enable_cb(self, msg):
        want = bool(msg.data)
        if want == self.combo_enabled:
            return
        self.combo_enabled = want
        self.state = _CRUISE                  # (re)start by cruising A*
        self.leg_pts = None
        if want:
            rospy.loginfo("combination: ENABLED -- cruising A*; will engage NavDP "
                          "when a point >= %.1fm is visible", self.min_engage_fwd_m)
        else:
            rospy.loginfo("combination: DISABLED -- following A* directly")
            self._echo_astar(force=True)      # resume A* immediately

    # ─── NavDP server handshake / guards ─────────────────────────────
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
            rospy.logfatal("combination: RGB %dx%d and depth %dx%d differ; they must "
                           "be aligned. Fix the stream.", rgb_hw[1], rgb_hw[0],
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

    def _snapshot(self):
        """One CONSISTENT frame for a decision: capture the sensor refs atomically,
        then validate. Returns ``{pose, alt, rgb, depth, waypoints}`` or ``None``.

        Selection, anchoring and the final-goal test all use this single snapshot,
        so a callback firing mid-tick can never mix a goal computed from one frame
        with an anchor pose (or waypoint count) from another.
        """
        pose, rgb, depth = self.pose_xyyaw, self.rgb, self.depth
        wpts, alt = self.astar_pts, self.altitude
        if (rgb is None or depth is None or pose is None or wpts is None
                or len(wpts) < 2 or not np.all(np.isfinite(pose))):
            return None
        return {"pose": pose, "alt": alt, "rgb": rgb.copy(),
                "depth": depth.copy(), "waypoints": list(wpts)}

    def _select(self, snap, engage_gate):
        """Farthest visible A* waypoint (a :class:`LocalGoal`) from ``snap``, or None.

        ``engage_gate`` additionally requires the goal to be >= ``min_engage_fwd_m``
        ahead (used in CRUISE to decide whether a point is worth a NavDP leg).
        """
        ox, oy, oyaw = snap["pose"]
        goal = select_farthest_visible_waypoint(
            snap["waypoints"], ox, oy, oyaw, snap["depth"], self.intr,
            cam_height_m=max(snap["alt"], 0.1),
            require_unoccluded=self.require_unoccluded,
            depth_tol_m=self.vis_depth_tol_m, depth_patch_half=self.vis_patch_half,
            min_fwd_m=self.min_goal_fwd_m, max_fwd_m=NAVDP_MAX_FWD_M,
            max_lat_m=NAVDP_MAX_LAT_M)
        if goal is None:
            return None
        if engage_gate and goal.body[0] < self.min_engage_fwd_m:
            return None
        return goal

    # ─── One NavDP inference (BLOCKING) ──────────────────────────────
    def _run_inference(self):
        """Pick a goal, ask NavDP for a leg, publish it. Returns an outcome code.

        Blocks on the HTTP step; the drone keeps flying its latched leg (or holds)
        meanwhile. On success it publishes the leg and updates ``leg_*`` state.
        """
        snap = self._snapshot()
        if snap is None or not self._ensure_reset():
            return _PENDING                       # not ready / NavDP down -> retry/hold
        if not self._streams_ok(snap["rgb"], snap["depth"]):
            return _UNAVAILABLE                    # fatal stream mismatch -> shutting down
        ox, oy, oyaw = snap["pose"]
        alt, waypoints = snap["alt"], snap["waypoints"]

        # Select, the final-goal test and the anchor below all use THIS snapshot
        # (one frame), so an A* replan or a new frame mid-tick can't desync them.
        goal = self._select(snap, engage_gate=False)
        if goal is None:
            return _NO_POINT
        is_final = goal.index >= len(waypoints) - 1
        if is_final and goal.body[0] <= self.final_handoff_m:
            return _FINAL

        t0 = time.time()
        result = self.client.pointgoal_step(snap["rgb"], snap["depth"], goal.goal[0],
                                            goal.goal[1], altitude=alt)
        dt = time.time() - t0
        if not self.combo_enabled:                # disabled during the blocking call
            return _ABORTED
        if result is None:
            rospy.logwarn("combination: NavDP no result (%.1fs) -- holding, will retry", dt)
            return _PENDING
        try:
            traj = self.client.best_trajectory(result)
        except NavDPError as e:
            rospy.logwarn("combination: %s (%.1fs) -- holding, will retry", e, dt)
            return _PENDING
        leg_world = anchor_trajectory_to_world(traj, ox, oy, oyaw)
        if len(leg_world) < 2:
            rospy.logwarn("combination: NavDP leg too short (%d) -- holding, will retry",
                          len(leg_world))
            return _PENDING

        stamp = rospy.Time.now()
        self.pub_path.publish(self._make_path(leg_world, stamp))
        self.pub_full.publish(self._make_path(leg_world, stamp))
        self._echoed_msg = None                   # we left A*; a later fallback force-republishes
        self.leg_pts = [Pose2D(float(x), float(y)) for x, y in leg_world]
        self.leg_final = is_final
        self.leg_t = stamp
        self.n_legs += 1
        rospy.loginfo("combination: leg #%d -> A* wp[%d]=(%.2f, %.2f) fwd=%.2fm  "
                      "%d pts (infer %.1fs)%s", self.n_legs, goal.index, goal.world[0],
                      goal.world[1], goal.body[0], len(leg_world), dt,
                      "  (final)" if is_final else "")
        self.thinker.say("NavDP leg ready -- flying it toward waypoint %d%s"
                         % (goal.index, " (the mission goal)" if is_final else ""),
                         category="plan")
        return _LEG

    # ─── State handlers ──────────────────────────────────────────────
    def _cruise(self):
        """Follow A*; engage NavDP when a far-enough visible point appears."""
        self._echo_astar()                        # keep flying A* (deduped)
        snap = self._snapshot()
        goal = self._select(snap, engage_gate=True) if snap is not None else None
        if goal is None:
            self.engage_confirm = 0               # reset the hysteresis streak
            rospy.loginfo_throttle(5.0, "combination: cruising A* (no engage-able "
                                   "point visible yet)")
            return
        # Hysteresis: require N consecutive detections so depth/occlusion flicker
        # can't trigger a stop-then-A* lurch.
        self.engage_confirm += 1
        if self.engage_confirm < self.engage_confirm_ticks:
            return
        if not self._ensure_reset():              # don't stop if NavDP is unreachable
            rospy.logwarn_throttle(2.0, "combination: point visible but NavDP "
                                   "unreachable -- staying on A*")
            self.thinker.say("A waypoint is in view but NavDP is unreachable -- "
                             "staying on the A* route", category="plan",
                             level="warn", repeat_after_s=5.0)
            self.engage_confirm = 0
            return
        rospy.loginfo("combination: visible point at %.2fm (x%d) -- stopping to "
                      "engage NavDP", goal.body[0], self.engage_confirm)
        self.thinker.say("Waypoint %d is in clear view ahead -- stopping to hand "
                         "this stretch to NavDP" % goal.index, category="plan")
        self.engage_confirm = 0
        self._publish_hold()                      # STOP for a clean, stationary inference
        self.hold_t = rospy.Time.now()
        self.settle_until = self.hold_t + rospy.Duration(self.engage_settle_s)
        self.state = _HOLD

    def _hold(self):
        """Stopped: wait out the brake-settle, then infer until a leg arrives.

        Also the "stop & wait" landing state for a slow re-infer (entered from
        FOLLOW): the drone holds (its last leg has been flown out) until the new
        route comes back, or ``max_wait_s`` elapses and it resumes A*.
        """
        if self.settle_until is not None and rospy.Time.now() < self.settle_until:
            return                                # let the drone brake to a clean stop
        # Bound the wait BEFORE blocking on another inference: the step can block up
        # to timeout_s, so checking max_wait only afterwards would let it overrun.
        if self.hold_t is not None and (rospy.Time.now() - self.hold_t).to_sec() > self.max_wait_s:
            rospy.logwarn("combination: waited %.0fs for NavDP -- resuming A*", self.max_wait_s)
            self._resume_astar("wait_timeout")
            return
        outcome = self._run_inference()
        if outcome == _LEG:
            self.state = _FOLLOW
        elif outcome == _ABORTED:
            return
        elif outcome in (_NO_POINT, _UNAVAILABLE, _FINAL):
            self._resume_astar(outcome)
        else:
            rospy.loginfo_throttle(3.0, "combination: holding for NavDP route ...")
            # Restated periodically: stopped-and-waiting looks identical to a stalled
            # log, so the operator needs to see it is still the live state.
            self.thinker.say("Stopped and waiting for NavDP to send a leg",
                             category="plan", level="warn", repeat_after_s=5.0)

    def _follow(self):
        """Fly the leg; re-infer at the midpoint (the 2nd half is the buffer)."""
        leg, pose = self.leg_pts, self.pose_xyyaw   # capture once (a callback may null leg_pts)
        if pose is None or not leg:
            return
        ox, oy, _ = pose
        pose2d = Pose2D(ox, oy)
        frac = arclength_fraction_2d(leg, pose2d)
        threshold = max(self.leg_fraction, 0.9) if self.leg_final else self.leg_fraction
        near_end = pose2d.distance_to(leg[-1]) <= self.leg_endpoint_radius_m
        timed_out = (self.leg_t is not None
                     and (rospy.Time.now() - self.leg_t).to_sec() > self.leg_timeout_s)
        if not (frac >= threshold or near_end or timed_out):
            return                                # keep flying the leg
        why = "midpoint" if frac >= threshold else ("endpoint" if near_end else "watchdog")
        rospy.loginfo("combination: leg #%d re-infer (%s, frac=%.0f%%)",
                      self.n_legs, why, 100.0 * frac)
        outcome = self._run_inference()           # drone keeps flying the latched leg during this
        if outcome == _LEG:
            self.state = _FOLLOW
        elif outcome == _ABORTED:
            return
        elif outcome in (_NO_POINT, _UNAVAILABLE, _FINAL):
            self._resume_astar(outcome)
        else:                                     # _PENDING: leg flies out, follower holds -> wait
            rospy.loginfo("combination: NavDP not ready -- flying out the leg, then "
                          "holding for the new route")
            self.thinker.say("NavDP has not sent the next leg -- flying out this one, "
                             "then holding for it", category="plan", level="warn")
            self.hold_t = rospy.Time.now()
            self.settle_until = None              # no extra settle; the leg is decelerating to its end
            self.state = _HOLD                    # do NOT publish a hold: let the latched leg coast to the end

    def _resume_astar(self, reason):
        """Drop back to CRUISE and republish A* so the drone keeps making progress."""
        rospy.loginfo("combination: -> A* (%s)", reason)
        thought = _RESUME_THOUGHTS.get(reason)
        if thought is not None:
            self.thinker.say(thought[0], category="plan", level=thought[1])
        self.state = _CRUISE
        self.leg_pts = None
        self.hold_t = None
        self.settle_until = None
        self._echo_astar(force=True)

    def _publish_hold(self):
        """Command the drone to STOP and hold at its current pose.

        A 2-point path at the current pose collapses (via the follower's reanchor)
        to a single waypoint the drone is already on, so it brakes to DONE and
        holds zero twist. Used to stand still for a clean inference frame. The
        path_corrector passes a coincident-point path through verbatim, so the
        stop survives the correction chain regardless of apf_pin_last.
        """
        if self.pose_xyyaw is None or not self.combo_enabled:
            return
        ox, oy, _ = self.pose_xyyaw
        self.pub_path.publish(self._make_path([(ox, oy), (ox, oy)], rospy.Time.now()))
        self._echoed_msg = None                   # we left A*; a later fallback force-republishes
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

    def _begin_forced_engage(self):
        """New goal: STOP and re-infer now, bypassing the cruise gate/hysteresis.

        Mirrors the CRUISE->HOLD commit (stop for a clean, stationary frame, settle,
        then infer) but without requiring a >= min_engage_fwd_m point or the confirm
        streak -- the operator asked for a leg on the new route explicitly. Returns
        True if it committed to a stop; False (no route/pose yet, or NavDP down) so
        the caller falls through to normal handling and re-engages on the cadence.
        """
        if (self.pose_xyyaw is None or self.astar_pts is None
                or len(self.astar_pts) < 2):
            return False                          # no fresh route/pose to engage on yet
        if not self._ensure_reset():
            rospy.logwarn_throttle(2.0, "combination: new goal but NavDP unreachable "
                                   "-- staying on A*")
            self.thinker.say("New goal, but NavDP is unreachable -- staying on the "
                             "A* route", category="plan", level="warn",
                             repeat_after_s=5.0)
            return False
        rospy.loginfo("combination: NEW GOAL -- stopping to re-infer immediately")
        # A click restarts the leg story, so re-clicking the same spot -- which does
        # force a fresh leg -- must narrate again rather than read as an unchanged repeat.
        self.thinker.forget("plan")
        self.thinker.say("New goal -- stopping to plan a fresh NavDP leg for it",
                         category="plan")
        self.engage_confirm = 0
        self._publish_hold()                      # STOP for a clean, stationary inference
        self.hold_t = rospy.Time.now()
        self.settle_until = self.hold_t + rospy.Duration(self.engage_settle_s)
        self.state = _HOLD
        return True

    # ─── Control tick ────────────────────────────────────────────────
    def _tick(self, _evt):
        if not self.combo_enabled:
            return                                # A* echo handled in _astar_cb
        # A single bad frame (no visible point, a non-finite pose, a transient NavDP/
        # geometry hiccup) must NEVER kill the loop: an uncaught exception in a
        # rospy.Timer callback terminates the timer thread, stopping all ticks.
        # Catch everything, drop safely back to flying A*, and retry next frame.
        try:
            if self._force_engage:
                self._force_engage = False        # one-shot; consumed whether or not it commits
                if self._begin_forced_engage():
                    return                         # HOLD infers over the next ticks
                # not ready / NavDP down -> fall through to the normal state handler
            if self.state == _CRUISE:
                self._cruise()
            elif self.state == _HOLD:
                self._hold()
            elif self.state == _FOLLOW:
                self._follow()
        except Exception as e:                    # noqa: BLE001 -- resilience is the point
            rospy.logwarn_throttle(2.0, "combination: tick error (%s: %s) -- "
                                   "following A*, retrying next frame",
                                   type(e).__name__, e)
            self.state = _CRUISE
            self.leg_pts = None
            try:
                self._echo_astar(force=True)
            except Exception:                     # noqa: BLE001
                pass

    # ─── Bring-up ────────────────────────────────────────────────────
    def start(self):
        # If intrinsics come from a camera_info topic, wait briefly so NavDP is reset
        # with the stream's true intrinsics before the first inference.
        if self.camera_info_topic:
            t0 = time.time()
            while (not rospy.is_shutdown() and not self._got_cam_info
                   and time.time() - t0 < 2.0):
                time.sleep(0.05)
            if not self._got_cam_info:
                rospy.logwarn("combination: no %s yet -- using param intrinsics",
                              self.camera_info_topic)
        # Attempt an initial reset. NavDP may be down; we retry lazily, so A*-cruising
        # still works until the server comes up.
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
        L("  enable   in = %s  (start_enabled=%s)", self.enable_topic, self.combo_enabled)
        L("  navdp       = %s  (timeout %.0fs)", self.client.url, self.client.timeout_s)
        L("  path    out = %s  (raw -> path_corrector)", self.path_topic)
        L("  engage      = stop & infer when a point >= %.1fm is visible "
          "(settle %.1fs)", self.min_engage_fwd_m, self.engage_settle_s)
        L("  leg handoff = re-infer at %.0f%% of each leg; hold & wait up to %.0fs "
          "for a slow route", 100.0 * self.leg_fraction, self.max_wait_s)
        L("  visibility  = %s%s",
          "in-frame + unoccluded" if self.require_unoccluded else "in-frame only",
          "" if not self.require_unoccluded else " (depth_tol=%.2fm)" % self.vis_depth_tol_m)
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
# this node owns ROS I/O and the CRUISE -> HOLD -> FOLLOW state machine.
#
#   sources/outputs:
#     ~astar_path_topic (/path/waypoints_astar)  raw A* route (nav_msgs/Path)
#     ~path_topic (/path/waypoints_combo)        arbitrated raw path -> path_corrector
#     ~full_path_topic (/path/waypoints_navdp_full)  full NavDP leg, display only
#     ~frame_id (world)
#   combination:
#     ~enable_topic (/combination/enable)        std_msgs/Bool: True=combine, False=A*
#     ~start_enabled (false)                     start combined for the whole run
#     ~tick_hz (5.0)                             state-machine rate
#     ~min_engage_fwd_m (1.5)                    engage NavDP only for a point >= this far ahead
#     ~engage_confirm_ticks (2)                  consecutive engage-able detections before stopping (hysteresis)
#     ~engage_settle_s (1.0)                     brake-settle before the first inference (clean frame)
#     ~max_wait_s (30.0)                         hold this long (checked before each attempt) for a slow inference, then A*
#     ~leg_execute_fraction (0.5)               re-infer after this fraction of a leg (the rest is buffer)
#     ~final_handoff_m (1.5)                    hand the final approach to A* within this range
#     ~leg_timeout_s (10.0)                     watchdog: re-infer if a leg runs this long
#     ~leg_endpoint_radius_m (0.4)              re-infer once within this of the leg end
#   Behaviour: CRUISE follows A* until a visible point >= min_engage_fwd_m, then
#   STOPS, settles, and infers; FOLLOW flies the leg and re-infers at the midpoint
#   (the leg's 2nd half buffers the inference latency); if a re-infer is slow the
#   drone flies out the leg and HOLDS until the new route arrives (<= max_wait_s),
#   else it resumes A*. No visible point / NavDP down -> fly A*.
#   visibility (core point_visible):
#     ~require_unoccluded (true)                require line-of-sight (depth) not just FOV
#     ~visibility_depth_tol_m (0.5)             slack on the occlusion test
#     ~visibility_patch_half_px (6)             depth patch half-size at the pixel
#     ~min_goal_fwd_m (0.5)                     ignore waypoints nearer than this
#   image transport (mirrors navdp_click):
#     ~image_transport (frame_path | topic)
#     ~rgb_topic ~depth_topic ~pose_topic (/xtend/localization) ~pose_type (pose_stamped)
#     ~camera_info_topic ('' = use the fx/fy/cx/cy params; K preferred over P)
#   camera (MUST match the live NavDP stream; launch wires these to cam_*):
#     ~fx ~fy ~cx ~cy ~img_width (504) ~img_height (294)
#   NavDP server: ~port (8888) ~timeout_s (10.0; one inference, generous for Jetson) ~depth_max_m (5.0)
#   misc: ~default_altitude (1.0; used until the first pose arrives) ~drone_ns ('')
#   thinking (see thinking.py): ~thinking (true; false silences the leg-lifecycle
#     narration) ~thinking_topic (/nav/thinking) ~thinking_echo (true; mirror to rosout)
# ============================================================================
