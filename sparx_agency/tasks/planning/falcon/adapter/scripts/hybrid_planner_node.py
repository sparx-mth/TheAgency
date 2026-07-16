#!/usr/bin/env python3
"""hybrid_planner_node.py -- FALCON "hybrid" mode: A* on easy legs, NavDP on hard ones.

A fifth navigation mode. The insight: a slow stop-and-turn follower tracks a clean
A* route beautifully on straight, open stretches, but *struggles* exactly where
the flying gets tricky -- a hard corridor turn, an S-bend, or threading a doorway
/ narrow gap. NavDP, a learned local point-goal policy, is best precisely there.
So this mode flies **A* by default and hands control to NavDP only for the
difficult maneuver, then takes it straight back once the hard part is behind**.

It is the arbiter that drives ``/path/waypoints_hybrid`` -- the single raw path the
planner-agnostic ``path_corrector`` -> ``trajectory_simplifier`` ->
``waypoint_follower`` chain flies. Two-mode hysteretic state machine:

  PRIMARY  -- echo the A* route straight through (plain ``nav_mode:=astar``). Each
              tick, look at the route just AHEAD of the drone for a hard turn (a
              corner turning >= ``~turn_thresh_deg`` coming within ``~turn_engage_
              distance_m`` -- "close enough to the turn") or a narrow passage (free
              width measured perpendicular to the route -- a doorway is tight on both
              sides). When difficult on ``~difficulty_confirm`` CONSECUTIVE ticks, STOP
              the drone and hand control to NavDP. Because the turn signal is the
              DISTANCE to the corner it stays asserted for the whole approach, so the
              confirm reliably fires (it never sat on A* through a real turn). (It also
              rescues a boxed-in A*: ``~engage_on_astar_fail`` engages when A* reports
              NO route, like ``nav_mode:=fallback``.)
  ENGAGED  -- drive NavDP toward the farthest A* waypoint VISIBLE in the current
              frame (the operator's "furthest point on A* I can see"), fly the
              returned leg to its midpoint (``~leg_execute_fraction``), then
              re-infer -- keeping NavDP in its accurate near field while A* supplies
              the direction. Hand control BACK to A* once the route ahead is easy
              again AND A* has a route, on ``~recover_confirm`` CONSECUTIVE ticks.

The hysteresis (a small confirm to engage, a larger, sticky one to return) is the
point: the drone never flips A*<->NavDP right at the hard spot. Unlike
``combination`` (which always fuses) this mode stays on pure A* for the easy 90%
of a mission; unlike ``fallback`` (which only rescues when A* fails) it engages on
a *geometric* difficulty A* solves but flies poorly.

All the maths is ROS-free and unit-tested in ``core.planning``:
  * route difficulty (turn + doorway)      (replanning.route_difficulty.assess_route_difficulty)
  * farthest visible A* waypoint           (navdp.select_farthest_visible_waypoint)
  * NavDP HTTP request/response            (navdp.NavDPPointgoalClient)
  * body trajectory -> world path          (navdp.anchor_trajectory_to_world)
  * progress along the flown leg           (planners.common.utils_2d.arclength_fraction_2d)
This node owns ONLY ROS concerns: it subscribes to the A* route, the per-attempt
A* status Bool, the BEV occupancy grid (for the doorway test) and RGB/depth/pose,
runs the small state machine, and publishes the arbitrated world path. It runs
HEADLESS (the route is seen on the BEV viewer). RGB/depth/pose ingestion and the
BEV decode mirror ``combination_planner`` / ``astar_planner``.

Run:
    rosrun falcon_adapter hybrid_planner_node.py
See the file footer for the full rosparam list.
"""
import math
import time

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.math import se3
from sparx_agency.core.common.types import Intrinsics, Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.navdp import (
    NAVDP_MAX_FWD_M,
    NAVDP_MAX_LAT_M,
    NavDPError,
    NavDPPointgoalClient,
    anchor_trajectory_to_world,
    point_to_pointgoal,
    select_farthest_visible_waypoint,
    world_to_body_2d,
)
from sparx_agency.core.planning.planners.common.utils_2d import arclength_fraction_2d
from sparx_agency.core.planning.replanning import assess_route_difficulty
from thinking import Thinker

# nav_msgs/OccupancyGrid int8 convention published by bev_publisher_node.
BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)

# Top-level mode.
_PRIMARY = "primary"     # fly A*; assess the route ahead for a difficult maneuver
_ENGAGED = "engaged"     # fly NavDP through the difficult stretch

# Leg sub-state (only meaningful while ENGAGED).
_HOLD = "hold"           # stopped: settle + infer (also the "stop & wait" for a slow re-infer)
_FOLLOW = "follow"       # flying a NavDP leg; re-infer at the midpoint

# _run_inference outcomes.
_LEG = "leg"                  # a new leg was published
_NO_POINT = "no_point"        # nothing visible -> keep holding (bounded by max_wait)
_UNAVAILABLE = "unavailable"  # fatal stream mismatch -> shutting down
_FINAL = "final"              # the visible goal is the mission goal & close -> hand to A*
_ARRIVED = "arrived"          # NavDP reached the mission goal (rescue) -> hold
_PENDING = "pending"          # transient/slow failure -> hold and retry
_ABORTED = "aborted"          # left ENGAGED mid-call -> caller does nothing

# Plain-English rendering of an engage reason (route_difficulty's verdict, or the
# boxed-A* rescue) for the operator's thinking log: "turn+narrow" is our jargon,
# not something an operator should have to decode mid-flight.
_ENGAGE_WHY = {
    "turn": "Sharp turn ahead",
    "narrow": "Narrow passage ahead",
    "turn+narrow": "Sharp turn through a narrow passage ahead",
    "astar_no_route": "A* has no route",
}

# Likewise for a return to A*: WHY control came back, keyed by _resume_primary's
# reason.
_RESUME_WHY = {
    "hard part cleared": "The hard part is behind me -- taking back control from "
                         "NavDP and flying A* again",
    "final_approach": "NavDP got me close to the goal -- A* flies the final approach",
    "wait_timeout": "NavDP gave me nothing for too long -- dropping it and flying "
                    "A* again",
}


def _fmt_dist(d):
    """Format a distance-to-turn for logs: ``inf`` (no hard turn) -> ``'far'``."""
    return "far" if d == float("inf") else "%.1fm" % d


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


class HybridPlannerNode:
    def __init__(self):
        rospy.init_node("hybrid_planner")
        G = rospy.get_param

        self.frame_id = G("~frame_id", "world")

        # ── A* source (status + path) + BEV + outputs ────────────────
        self.status_topic = G("~astar_status_topic", "/path/astar_status")
        self.astar_path_topic = G("~astar_path_topic", "/path/waypoints_astar")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        self.path_topic = G("~path_topic", "/path/waypoints_hybrid")
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_navdp_full")
        # Human-readable "who is planning now" (std_msgs/String) for the BEV viewer
        # HUD: "A*" while PRIMARY, "NavDP (<reason>)" while ENGAGED.
        self.nav_status_topic = G("~nav_status_topic", "/nav/status")
        # Mission goal (same topic astar_planner replans on). Used ONLY by the
        # boxed-A* rescue: when A* has NO route there is no forward A* waypoint to
        # aim NavDP at, so -- exactly like nav_mode:=fallback -- NavDP is driven
        # toward this world goal expressed in the drone body frame (no visibility
        # gate; the goal is around the corner / past the door A* rejected).
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        gx = G("~goal_x", None)
        gy = G("~goal_y", None)
        self.goal_world = ((float(gx), float(gy))
                           if gx is not None and gy is not None else None)

        # ── Difficulty detector (core route_difficulty) ──────────────
        # How far ahead of the drone to look for a difficult maneuver, and how far
        # to skip past the drone's own footprint (a maneuver just finished must not
        # keep reading as "ahead").
        self.difficulty_lookahead_m = float(G("~difficulty_lookahead_m", 2.0))
        self.difficulty_skip_m = float(G("~difficulty_skip_m", 0.3))
        # The raw A* route is jagged (grid staircase + LOS jogs + sharp un-chamfered
        # corners). Assess difficulty on the SMOOTHED route the drone actually tracks:
        # drop vertices turning less than this before measuring, so staircase noise
        # doesn't sum into a false "hard turn" on a straight corridor. 0 = raw route.
        self.difficulty_merge_collinear_deg = float(G("~difficulty_merge_collinear_deg", 15.0))
        # Hard turn: a route corner whose heading change (deg) reaches this is one
        # the stop-and-turn follower flies poorly -> hand to NavDP. The operator's
        # ">70 deg" corner; only a genuinely sharp corner engages NavDP, not a bend.
        self.turn_thresh_deg = float(G("~turn_thresh_deg", 70.0))
        # ENGAGE RANGE: hand to NavDP once such a hard turn is within this distance
        # ahead of the drone ("from the moment I get close enough to the turn"). A*
        # flies the whole approach up to here; NavDP takes the corner itself. The turn
        # signal is the DISTANCE to the next hard corner, so it stays asserted for the
        # entire approach within this range (reliable confirm), not just one tick.
        self.turn_engage_distance_m = float(G("~turn_engage_distance_m", 2.0))
        # Entry/exit chord (m) for measuring a corner's net turn: robust to a corner
        # the planner split into two nearby vertices, and to single-cell grid jitter.
        self.turn_corner_span_m = float(G("~turn_corner_span_m", 0.7))
        # Narrow passage: free width (m) below which the route is a doorway / tight
        # gap. Measured perpendicular to the route (tight on BOTH sides), so an open
        # corridor whose A* route merely clips a corner is NOT flagged. Strict: only a
        # real door-width gap engages NavDP.
        self.passage_width_m = float(G("~passage_width_m", 0.75))
        self.difficulty_sample_step_m = float(G("~difficulty_sample_step_m", 0.1))
        # A doorway must be narrow over at least this arclength to count -- rejects a
        # single BEV occupancy speckle cell (common on a monocular-depth bag replay)
        # tripping a false doorway on an open corridor.
        self.min_narrow_span_m = float(G("~min_narrow_span_m", 0.3))

        # ── Hysteresis ───────────────────────────────────────────────
        self.tick_hz = float(G("~tick_hz", 5.0))
        # Consecutive "difficult ahead" ticks before stopping and engaging NavDP.
        self.difficulty_confirm = max(1, int(G("~difficulty_confirm", 3)))
        # Consecutive "route ahead easy again (and A* has a route)" ticks before
        # handing control back to A*. Keep >= difficulty_confirm so the return is
        # deliberately sticky (anti-zigzag right at the hard spot).
        self.recover_confirm = max(1, int(G("~recover_confirm", 5)))
        # Do NOT hand back to A* until the drone has flown at least this far (m) from
        # where NavDP engaged -- i.e. actually THROUGH the difficult maneuver, not the
        # moment the route momentarily reads easy. ~= the lookahead so it covers the
        # turn/door. Set 0 to disable (return as soon as the route ahead clears).
        self.min_pass_distance_m = float(G("~min_pass_distance_m", 2.0))
        # Also engage when A* itself is boxed in (no route), like nav_mode:=fallback.
        # DEFAULT OFF: hybrid is purely difficulty-triggered (hard turns / doorways).
        # On a flaky map (e.g. a bag replay) A* status can drop out and this would
        # otherwise drive NavDP to the goal for the WHOLE route; use nav_mode:=fallback
        # if you specifically want boxed-A* rescue.
        self.engage_on_astar_fail = _param_bool("~engage_on_astar_fail", False)
        self.astar_fail_confirm = max(1, int(G("~astar_fail_confirm", 3)))

        # ── NavDP leg behaviour (mirrors combination) ────────────────
        self.engage_settle_s = float(G("~engage_settle_s", 1.0))
        self.leg_fraction = float(G("~leg_execute_fraction", 0.5))
        self.final_handoff_m = float(G("~final_handoff_m", 1.5))
        self.leg_timeout_s = float(G("~leg_timeout_s", 8.0))
        self.leg_endpoint_radius_m = float(G("~leg_endpoint_radius_m", 0.4))
        # Bounded hold while waiting for a slow / blind inference before giving up
        # to A* (checked BEFORE each blocking attempt), so a dead/blind server never
        # strands the drone. Keep comfortably larger than ~timeout_s.
        self.max_wait_s = float(G("~max_wait_s", 30.0))
        # Within this range of the mission goal a rescue leg has arrived -> hold.
        self.arrival_radius_m = float(G("~arrival_radius_m", 0.5))

        # ── Visibility knobs (see core point_visible) ────────────────
        self.require_unoccluded = _param_bool("~require_unoccluded", True)
        self.vis_depth_tol_m = float(G("~visibility_depth_tol_m", 0.5))
        self.vis_patch_half = int(G("~visibility_patch_half_px", 6))
        self.min_goal_fwd_m = float(G("~min_goal_fwd_m", 0.5))
        # At a hard turn the post-turn route swings OUT of the forward camera FOV, so
        # the visibility-gated selection finds nothing. Rather than HOLD at the corner
        # (then revert to A* after ~max_wait_s -- the stop-and-turn NavDP was engaged to
        # avoid), drive NavDP toward the farthest A* waypoint reachable by GEOMETRY
        # around the corner (see _blind_turn_target). Set false to hold instead.
        self.blind_turn_target = _param_bool("~blind_turn_target", True)

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

        # Narrates the arbitration to the operator's BEV thinking log. Two slots:
        # the default "plan" slot tells the A* <-> NavDP switch story; "navdp_health"
        # tells the independent story of whether NavDP can take the hard part at all.
        self.thinker = Thinker("hybrid_planner")

        # ── Shared state (callbacks write, tick reads) ───────────────
        self.rgb = None
        self.depth = None
        self.altitude = float(G("~default_altitude", 1.0))   # until a pose arrives
        self.pose_xyyaw = None                               # (x, y, yaw) world
        self.grid = None                                     # OccupancyGrid2D (BEV)
        self.astar_pts = None                                # list[(x, y)] world
        self.astar_msg = None                                # last A* Path (for echo)
        self._echoed_msg = None                              # last A* Path published (dedup)
        self.astar_ok = True                                 # last A* status (route found?)
        self.fail_streak = 0                                 # consecutive A* "no route"
        self._progress_idx = 0                               # forward-monotone projection hint
        self.leg_pts = None                                  # list[Pose2D] current leg
        self.leg_final = False                               # local goal == mission goal
        self.leg_t = None                                    # rospy.Time the leg was published
        self.hold_t = None                                   # rospy.Time the HOLD began
        self.settle_until = None                             # rospy.Time the brake-settle ends
        self.difficulty_streak = 0                           # consecutive "difficult ahead"
        self.recover_streak = 0                              # consecutive "easy again + A* ok"
        self._engage_pos = None                              # (x, y) where NavDP engaged (travel gate)
        self._rescue = False                                 # engaged via astar_no_route (aim NavDP at the goal)
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
        self.pub_nav_status = rospy.Publisher(self.nav_status_topic, String,
                                              queue_size=1, latch=True)
        self._nav_status = None                              # last published (dedup)

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
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
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
            rospy.logwarn_throttle(5.0, "hybrid: dropping RGB frame-path (%s)", e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _depth_path_cb(self, msg):
        try:
            path = parse_frame_path_message(msg.data).path
            arr = np.squeeze(np.load(path))
            if arr.ndim != 2:
                raise ValueError("depth %s has shape %r; expected HxW" % (path, arr.shape))
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "hybrid: dropping depth frame-path (%s)", e)
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
            rospy.logwarn_throttle(5.0, "hybrid: unsupported depth encoding %r "
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

    def _bev_cb(self, msg):
        """Decode the latest BEV OccupancyGrid into a core OccupancyGrid2D for the
        doorway (perpendicular-width) test. A bad frame must not kill the loop."""
        try:
            i = msg.info
            try:
                data = np.frombuffer(bytes(bytearray(msg.data)), dtype=np.int8)
            except Exception:                     # noqa: BLE001
                data = np.asarray(msg.data, dtype=np.int8)
            data = data.reshape(i.height, i.width).astype(np.int16)
            params = OccupancyGrid2DParams(
                resolution=i.resolution, origin_x=i.origin.position.x,
                origin_y=i.origin.position.y, frame_id=self.frame_id)
            self.grid = OccupancyGrid2D(data, params, values=BEV_VALUES)
        except Exception as e:                    # noqa: BLE001
            rospy.logwarn_throttle(5.0, "hybrid: dropping BEV frame (%s)", e)

    # ─── A* echo + status ────────────────────────────────────────────
    def _echo_astar(self, force=False):
        """Publish the latest A* path on the hybrid topic (the primary route).

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
        self.astar_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._progress_idx = 0                    # new route -> reset the projection hint
        if self.mode == _PRIMARY:
            self._echo_astar()                    # follow A* directly while primary

    def _status_cb(self, msg):
        """One A* planning-attempt outcome -> track 'has route' + the fail streak."""
        ok = bool(msg.data)
        self.astar_ok = ok
        if ok:
            self.fail_streak = 0
        else:
            self.fail_streak += 1

    def _goal_cb(self, msg):
        """New mission goal (world). Used by the boxed-A* rescue to aim NavDP."""
        self.goal_world = (float(msg.x), float(msg.y))
        self._arrived_held = False            # a new goal is not reached yet

    # ─── Difficulty assessment (core route_difficulty) ───────────────
    def _assess_difficulty(self):
        """RouteDifficulty on the A* route ahead of the drone, or None if not ready.

        The perpendicular free-width (doorway) test needs the BEV; until it arrives
        only the turn signal fires (occupied=None). The grid is captured once so a
        mid-assessment BEV callback cannot desync the query.
        """
        pose, pts, grid = self.pose_xyyaw, self.astar_pts, self.grid
        if pose is None or pts is None or len(pts) < 2:
            return None
        poly = [Pose2D(float(x), float(y)) for x, y in pts]
        occ = None
        if grid is not None:
            def occ(x, y, _g=grid):
                gx, gy = _g.world_to_grid(x, y)
                return _g.in_bounds(gx, gy) and _g.is_occupied(gx, gy)
        diff, seg = assess_route_difficulty(
            poly, Pose2D(float(pose[0]), float(pose[1])),
            lookahead_m=self.difficulty_lookahead_m,
            turn_thresh_deg=self.turn_thresh_deg,
            passage_width_thresh_m=self.passage_width_m,
            occupied=occ, min_index=self._progress_idx,
            skip_m=self.difficulty_skip_m,
            sample_step_m=self.difficulty_sample_step_m,
            merge_collinear_deg=self.difficulty_merge_collinear_deg,
            min_narrow_span_m=self.min_narrow_span_m,
            turn_span_m=self.turn_corner_span_m,
            turn_scan_m=self.turn_engage_distance_m)
        self._progress_idx = seg
        return diff

    # ─── NavDP server handshake / guards ─────────────────────────────
    def _ensure_reset(self):
        """Reset NavDP with the current intrinsics (lazy: NavDP may be down)."""
        if self._reset_done:
            return True
        if self.client.reset(self.intr):
            self._reset_done = True
            rospy.loginfo("hybrid: NavDP reset OK (%s)", self.client.url)
            return True
        return False

    def _streams_ok(self, rgb, depth):
        """Fail loud (once) if RGB and depth are not the same resolution."""
        if self._streams_checked:
            return True
        rgb_hw, depth_hw = rgb.shape[:2], depth.shape[:2]
        if rgb_hw != depth_hw:
            rospy.logfatal("hybrid: RGB %dx%d and depth %dx%d differ; they must be "
                           "aligned. Fix the stream.", rgb_hw[1], rgb_hw[0],
                           depth_hw[1], depth_hw[0])
            rospy.signal_shutdown("RGB/depth resolution mismatch")
            return False
        if rgb_hw != (self.intr.height, self.intr.width):
            rospy.logwarn("hybrid: stream is %dx%d but intrinsics are %dx%d; goals "
                          "will be geometrically wrong -- pass matching intrinsics.",
                          rgb_hw[1], rgb_hw[0], self.intr.width, self.intr.height)
        self._streams_checked = True
        return True

    def _snapshot(self):
        """One CONSISTENT frame for a NavDP decision: capture the sensor refs
        atomically, then validate. Returns ``{pose, alt, rgb, depth, waypoints}``
        or ``None``. Selection and anchoring both use this single snapshot.
        """
        pose, rgb, depth = self.pose_xyyaw, self.rgb, self.depth
        wpts, alt = self.astar_pts, self.altitude
        if (rgb is None or depth is None or pose is None or wpts is None
                or len(wpts) < 2 or not np.all(np.isfinite(pose))):
            return None
        return {"pose": pose, "alt": alt, "rgb": rgb.copy(),
                "depth": depth.copy(), "waypoints": list(wpts)}

    def _select(self, snap):
        """Farthest visible A* waypoint (a :class:`LocalGoal`) from ``snap``, or None."""
        ox, oy, oyaw = snap["pose"]
        return select_farthest_visible_waypoint(
            snap["waypoints"], ox, oy, oyaw, snap["depth"], self.intr,
            cam_height_m=max(snap["alt"], 0.1),
            require_unoccluded=self.require_unoccluded,
            depth_tol_m=self.vis_depth_tol_m, depth_patch_half=self.vis_patch_half,
            min_fwd_m=self.min_goal_fwd_m, max_fwd_m=NAVDP_MAX_FWD_M,
            max_lat_m=NAVDP_MAX_LAT_M)

    def _choose_target(self, snap):
        """Pick NavDP's point-goal for this inference; return ``(gx, gy, is_final,
        tag)`` or a terminal outcome code (``_NO_POINT`` / ``_FINAL`` / ``_ARRIVED``).

        Normally the farthest A* waypoint VISIBLE in the frame (the user's "furthest
        point on A* I can see"). Three fallbacks when nothing is visible: a boxed-A*
        rescue (``self._rescue``) aims NavDP at the mission goal in the body frame
        (like ``nav_mode:=fallback``; the route is a coincident STOP so there is no
        forward waypoint); a hard-turn engage aims at the farthest A* waypoint
        reachable by GEOMETRY around the corner (:meth:`_blind_turn_target`) so NavDP
        flies THROUGH the turn rather than stalling as the route leaves the FOV; and
        with neither a target -> ``_NO_POINT``.
        """
        ox, oy, oyaw = snap["pose"]
        goal = self._select(snap)
        if goal is not None:
            is_final = goal.index >= len(snap["waypoints"]) - 1
            if is_final and goal.body[0] <= self.final_handoff_m:
                return _FINAL
            tag = "A* wp[%d]=(%.2f, %.2f) fwd=%.2fm" % (
                goal.index, goal.world[0], goal.world[1], goal.body[0])
            return (goal.goal[0], goal.goal[1], is_final, tag)
        if self._rescue and self.goal_world is not None:
            gwx, gwy = self.goal_world
            if math.hypot(gwx - ox, gwy - oy) <= self.arrival_radius_m:
                return _ARRIVED
            fwd, left = world_to_body_2d(gwx, gwy, ox, oy, oyaw)
            gx, gy = point_to_pointgoal(fwd, left)
            return (gx, gy, False, "goal body=(%.2f, %.2f)" % (fwd, left))
        # Nothing VISIBLE, but this is a hard-turn engage (not a rescue): the post-turn
        # route has swung out of the forward FOV. Drive NavDP THROUGH the turn toward
        # the farthest A* waypoint reachable by geometry (around the corner) instead of
        # holding here and reverting to A* -- the very stop-and-turn we engaged to avoid.
        if self.blind_turn_target and not self._rescue:
            blind = self._blind_turn_target(snap)
            if blind is not None:
                gx, gy, fwd, left, idx = blind
                is_final = idx >= len(snap["waypoints"]) - 1
                return (gx, gy, is_final,
                        "A* wp[%d] blind body=(%.2f, %.2f)" % (idx, fwd, left))
        return _NO_POINT

    def _blind_turn_target(self, snap):
        """Farthest A* waypoint reachable by NavDP by GEOMETRY (no visibility gate).

        At a hard turn the post-turn route leaves the forward camera FOV, so the
        visibility-gated :meth:`_select` finds nothing and NavDP would hold at the
        corner until ``max_wait_s`` and hand the turn back to A*. Instead aim it at
        the farthest A* waypoint that is genuinely AHEAD of the drone (positive
        forward -- a non-positive forward collapses to a straight-ahead goal in
        :func:`point_to_pointgoal`, which would drive into the corner wall) and
        inside NavDP's reachable box, ignoring visibility: A* already vouched the
        route is collision-free and NavDP's local policy avoids the near walls, so
        this yaws/drives the drone through the corner. As it rotates, the route
        swings back into frame and the normal visible selection resumes.

        Returns ``(gx, gy, fwd, left, index)`` for that waypoint, or ``None`` when no
        forward, in-range waypoint exists (e.g. a boxed rescue's coincident stop).
        """
        ox, oy, oyaw = snap["pose"]
        best = None
        for i, (wx, wy) in enumerate(snap["waypoints"]):
            fwd, left = world_to_body_2d(wx, wy, ox, oy, oyaw)
            dist = math.hypot(fwd, left)
            if (fwd >= 0.1 and self.min_goal_fwd_m <= dist <= NAVDP_MAX_FWD_M
                    and abs(left) <= NAVDP_MAX_LAT_M):
                gx, gy = point_to_pointgoal(fwd, left)
                best = (gx, gy, fwd, left, i)      # keep the FARTHEST reachable (last)
        return best

    # ─── One NavDP inference (BLOCKING) ──────────────────────────────
    def _run_inference(self):
        """Choose a target, ask NavDP for a leg, publish it. Returns an outcome code.

        Blocks on the HTTP step; the drone keeps flying its latched leg (or holds)
        meanwhile. On success it publishes the leg and updates ``leg_*`` state.
        """
        snap = self._snapshot()
        if snap is None or not self._ensure_reset():
            return _PENDING                       # not ready / NavDP down -> retry/hold
        if not self._streams_ok(snap["rgb"], snap["depth"]):
            return _UNAVAILABLE                    # fatal stream mismatch -> shutting down
        ox, oy, oyaw = snap["pose"]
        alt = snap["alt"]

        target = self._choose_target(snap)
        if isinstance(target, str):               # a terminal outcome code
            return target
        gx, gy, is_final, tag = target

        t0 = time.time()
        result = self.client.pointgoal_step(snap["rgb"], snap["depth"], gx, gy,
                                            altitude=alt)
        dt = time.time() - t0
        if self.mode != _ENGAGED:                 # left ENGAGED during the blocking call
            return _ABORTED
        if result is None:
            rospy.logwarn("hybrid: NavDP no result (%.1fs) -- holding, will retry", dt)
            return _PENDING
        try:
            traj = self.client.best_trajectory(result)
        except NavDPError as e:
            rospy.logwarn("hybrid: %s (%.1fs) -- holding, will retry", e, dt)
            return _PENDING
        leg_world = anchor_trajectory_to_world(traj, ox, oy, oyaw)
        if len(leg_world) < 2:
            rospy.logwarn("hybrid: NavDP leg too short (%d) -- holding, will retry",
                          len(leg_world))
            return _PENDING

        stamp = rospy.Time.now()
        self.pub_path.publish(self._make_path(leg_world, stamp))
        self.pub_full.publish(self._make_path(leg_world, stamp))
        self._echoed_msg = None                   # we left A*; a resume force-republishes
        self.leg_pts = [Pose2D(float(x), float(y)) for x, y in leg_world]
        self.leg_final = is_final
        self.leg_t = stamp
        self.n_legs += 1
        rospy.loginfo("hybrid: leg #%d -> %s  %d pts (infer %.1fs)%s", self.n_legs,
                      tag, len(leg_world), dt, "  (final)" if is_final else "")
        return _LEG

    # ─── Leg sub-state handlers (only while ENGAGED) ─────────────────
    def _hold(self):
        """Stopped: wait out the brake-settle, then infer until a leg arrives.

        Also the "stop & wait" state for a slow/blind re-infer: the drone holds
        (its last leg has been flown out) until the new route comes back, or
        ``max_wait_s`` elapses and it resumes A* so a dead server never strands it.
        """
        if self._arrived_held:
            return
        if self.settle_until is not None and rospy.Time.now() < self.settle_until:
            return                                # let the drone brake to a clean stop
        # Bound the wait BEFORE blocking on another inference (the step itself can
        # block up to timeout_s, so checking only afterwards would let it overrun).
        if self.hold_t is not None and (rospy.Time.now() - self.hold_t).to_sec() > self.max_wait_s:
            rospy.logwarn("hybrid: waited %.0fs for NavDP -- resuming A*", self.max_wait_s)
            # The rescue engaged BECAUSE A* was boxed in, so "back to A*" is only a
            # recovery if A* found a route meanwhile. If it did not and NavDP never
            # returned a leg either, there is no plan left -- say so rather than let
            # the resume line imply A* has one. Read before _resume_primary clears it.
            stuck = self._rescue and not self.astar_ok
            self._resume_primary("wait_timeout")
            if stuck:
                self.thinker.say("No route from A* and NavDP returned nothing -- "
                                 "I am stuck", category="plan", level="error")
            return
        outcome = self._run_inference()
        if outcome == _LEG:
            self.leg_state = _FOLLOW
        elif outcome == _FINAL:
            self._resume_primary("final_approach")
        elif outcome == _ARRIVED:
            self._hold_on_arrival()
        elif outcome in (_ABORTED, _UNAVAILABLE):
            return
        else:                                     # _NO_POINT / _PENDING -> keep holding
            rospy.loginfo_throttle(3.0, "hybrid: holding for a NavDP leg ...")

    def _hold_on_arrival(self):
        """Mission goal reached via a NavDP rescue leg: STOP once and stay put."""
        if not self._arrived_held:
            rospy.loginfo("hybrid: goal reached via NavDP (<= %.2fm) -- holding",
                          self.arrival_radius_m)
            self._publish_hold()
            self._arrived_held = True
        self.leg_state = _HOLD
        self.settle_until = None
        self.hold_t = None                        # so a later new goal re-infers cleanly

    def _follow(self):
        """Fly the leg; re-infer at the midpoint (the 2nd half is the buffer)."""
        leg, pose = self.leg_pts, self.pose_xyyaw   # capture once (a cb may null leg_pts)
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
        rospy.loginfo("hybrid: leg #%d re-infer (%s, frac=%.0f%%)",
                      self.n_legs, why, 100.0 * frac)
        outcome = self._run_inference()           # drone keeps flying the latched leg during this
        if outcome == _LEG:
            self.leg_state = _FOLLOW
        elif outcome == _FINAL:
            self._resume_primary("final_approach")
        elif outcome == _ARRIVED:
            self._hold_on_arrival()
        elif outcome in (_ABORTED, _UNAVAILABLE):
            return
        else:                                     # _NO_POINT / _PENDING: fly out the leg, then hold
            rospy.loginfo("hybrid: NavDP not ready -- flying out the leg, then holding")
            self.hold_t = rospy.Time.now()
            self.settle_until = None              # the leg is decelerating to its end
            self.leg_state = _HOLD                # do NOT publish a hold: let the leg coast

    # ─── Mode transitions ────────────────────────────────────────────
    def _enter_engaged(self, reason, diff):
        """A difficult maneuver (or a boxed A*) is ahead: STOP and hand to NavDP."""
        detail = ""
        if diff is not None:
            detail = " (turn=%.0fdeg @%s width=%.2fm)" % (
                diff.turn_deg, _fmt_dist(diff.turn_dist_m), diff.passage_width_m)
        rospy.logwarn("hybrid: DIFFICULT maneuver ahead [%s]%s -- STOPPING, "
                      "engaging NavDP", reason, detail)
        self._publish_nav_status("NavDP  (%s)" % reason)
        self.mode = _ENGAGED
        # Narrate only once the hand-over is real. say() raises on a mislabelled
        # thought, and narration must never be able to abort a control transition
        # and strand the arbiter half-engaged.
        self.thinker.say("%s -- stopping and handing control to NavDP"
                         % _ENGAGE_WHY.get(reason, reason), category="plan")
        # NavDP just proved reachable and now has the controls: its health story restarts.
        self.thinker.forget("navdp_health")
        self._rescue = reason == "astar_no_route"  # aim NavDP at the goal, not an A* wp
        self._engage_pos = ((self.pose_xyyaw[0], self.pose_xyyaw[1])
                            if self.pose_xyyaw is not None else None)
        self.difficulty_streak = 0
        self.recover_streak = 0
        # Consume the fail streak that triggered the engage: while ENGAGED, a fresh
        # streak must re-accumulate to matter, and a resume must not instantly re-fire.
        self.fail_streak = 0
        self._arrived_held = False
        self._publish_hold()                      # STOP for a clean, stationary inference
        self.hold_t = rospy.Time.now()
        self.settle_until = self.hold_t + rospy.Duration(self.engage_settle_s)
        self.leg_state = _HOLD

    def _resume_primary(self, reason):
        """The hard part is behind (or a bounded give-up): hand control back to A*."""
        rospy.loginfo("hybrid: -> A* (%s)", reason)
        self.thinker.say(_RESUME_WHY.get(reason, "Handing control back to A* (%s)"
                                         % reason), category="plan")
        self._publish_nav_status("A*")
        self.mode = _PRIMARY
        self._rescue = False
        self._engage_pos = None
        self.difficulty_streak = 0
        self.recover_streak = 0
        # Clear the fail streak too: a stale count must not re-engage the rescue on
        # the very next tick (a still-boxed A* re-accumulates a fresh confirm first).
        self.fail_streak = 0
        self.leg_pts = None
        self.leg_state = _HOLD
        self.hold_t = None
        self.settle_until = None
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

    def _publish_nav_status(self, text):
        """Publish 'who is planning now' (deduped) for the BEV viewer HUD."""
        if text == self._nav_status:
            return
        self._nav_status = text
        self.pub_nav_status.publish(String(data=text))

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
    def _primary(self, diff):
        """Fly A*; engage NavDP when a difficult maneuver is confirmed ahead."""
        self._echo_astar()                        # keep flying A* (deduped)
        hard = diff is not None and diff.is_difficult
        self.difficulty_streak = self.difficulty_streak + 1 if hard else 0
        # Live readout so you can see WHY it does/doesn't engage along the route:
        # the sharpest corner ahead, how far off it is, and the engage countdown.
        if diff is not None:
            rospy.loginfo_throttle(
                2.0, "hybrid PRIMARY: turn=%.0f/%.0fdeg @%s (engage<=%.1fm) "
                "width=%.2f/%.2fm [%s] streak=%d/%d", diff.turn_deg,
                self.turn_thresh_deg, _fmt_dist(diff.turn_dist_m),
                self.turn_engage_distance_m, diff.passage_width_m,
                self.passage_width_m, diff.reason, self.difficulty_streak,
                self.difficulty_confirm)
        reason = None
        if self.difficulty_streak >= self.difficulty_confirm:
            reason = diff.reason                   # difficulty implies pose is known (_assess)
        elif (self.engage_on_astar_fail and self.fail_streak >= self.astar_fail_confirm
              and self.pose_xyyaw is not None):    # need a pose to STOP-hold and to aim
            reason = "astar_no_route"
        if reason is None:
            return
        if not self._ensure_reset():              # don't stop into a dead NavDP server
            rospy.logwarn_throttle(2.0, "hybrid: difficulty ahead (%s) but NavDP "
                                   "unreachable -- staying on A*", reason)
            self.thinker.say("%s, but NavDP is unreachable -- staying on A*"
                             % _ENGAGE_WHY.get(reason, reason), category="plan",
                             level="error", key="navdp_health", repeat_after_s=5.0)
            self.difficulty_streak = 0
            return
        self._enter_engaged(reason, diff)

    def _engaged(self, diff):
        """Fly NavDP through the hard part; return to A* once it is behind.

        Return-to-A* requires ALL of, held for ``recover_confirm`` consecutive ticks:
        (1) at least one NavDP leg has flown (``n_legs >= 1`` -- never bail during the
        brake-settle before the first inference); (2) the drone has flown at least
        ``min_pass_distance_m`` from where it engaged, i.e. actually THROUGH the
        maneuver, not the instant the route momentarily reads easy; (3) the route
        ahead is easy again. A boxed-A* RESCUE additionally waits for A* to report a
        route again; a difficulty engage does not (a flapping per-attempt status must
        not strand it on NavDP).
        """
        traveled = (math.hypot(self.pose_xyyaw[0] - self._engage_pos[0],
                               self.pose_xyyaw[1] - self._engage_pos[1])
                    if (self._engage_pos is not None and self.pose_xyyaw is not None)
                    else 0.0)
        passed = traveled >= self.min_pass_distance_m
        route_ok = self.astar_pts is not None and len(self.astar_pts) >= 2
        astar_recovered = self.astar_ok if self._rescue else True
        easy = (self.n_legs >= 1 and passed and diff is not None
                and not diff.is_difficult and route_ok and astar_recovered)
        self.recover_streak = self.recover_streak + 1 if easy else 0
        if diff is not None:
            rospy.loginfo_throttle(
                2.0, "hybrid ENGAGED[%s]: flown=%.1f/%.1fm ahead turn=%.0fdeg @%s "
                "width=%.2fm easy=%s recover=%d/%d",
                "rescue" if self._rescue else "difficulty", traveled,
                self.min_pass_distance_m, diff.turn_deg, _fmt_dist(diff.turn_dist_m),
                diff.passage_width_m, easy, self.recover_streak, self.recover_confirm)
        if self.recover_streak >= self.recover_confirm:
            self._resume_primary("hard part cleared")
            return
        if self.leg_state == _HOLD:
            self._hold()
        elif self.leg_state == _FOLLOW:
            self._follow()

    def _tick(self, _evt):
        # A single bad frame must NEVER kill the loop: an uncaught exception in a
        # rospy.Timer callback terminates the timer thread. Catch everything and
        # drop safely back to flying A*.
        try:
            diff = self._assess_difficulty()
            if self.mode == _PRIMARY:
                self._primary(diff)
            else:
                self._engaged(diff)
        except Exception as e:                    # noqa: BLE001 -- resilience is the point
            rospy.logwarn_throttle(2.0, "hybrid: tick error (%s: %s) -- following "
                                   "A*, retrying next frame", type(e).__name__, e)
            self.mode = _PRIMARY
            self._rescue = False
            self.leg_pts = None
            self.difficulty_streak = 0
            self.recover_streak = 0
            self.fail_streak = 0                   # don't instantly re-engage on a stale count
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
                rospy.logwarn("hybrid: no %s yet -- using param intrinsics",
                              self.camera_info_topic)
        if not self._ensure_reset():
            rospy.logwarn("hybrid: NavDP not reachable yet at %s -- will retry when a "
                          "difficult maneuver is reached", self.client.url)
        self._banner()
        self._publish_nav_status("A*")            # start on A* until a hard spot
        rospy.Timer(rospy.Duration(1.0 / self.tick_hz), self._tick)
        rospy.spin()

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("hybrid_planner (A* on easy legs + NavDP on hard maneuvers)")
        L("  A* status in = %s  (Bool per attempt)", self.status_topic)
        L("  A* path   in = %s", self.astar_path_topic)
        L("  bev       in = %s  (doorway/narrow-passage test)", self.bev_topic)
        L("  rgb       in = %s", self.rgb_topic)
        L("  depth     in = %s", self.depth_topic)
        L("  pose      in = %s  (%s)", self.pose_topic, self.pose_type)
        L("  navdp        = %s  (timeout %.0fs)", self.client.url, self.client.timeout_s)
        L("  path     out = %s  (raw -> path_corrector)", self.path_topic)
        L("  engage       = hard turn >= %.0fdeg within %.1fm ahead OR passage < "
          "%.2fm within %.1fm, x%d ticks", self.turn_thresh_deg,
          self.turn_engage_distance_m, self.passage_width_m,
          self.difficulty_lookahead_m, self.difficulty_confirm)
        L("  return       = route ahead easy + A* has a route, x%d ticks (sticky)",
          self.recover_confirm)
        L("  astar rescue = %s (after %d A* no-route reports)",
          self.engage_on_astar_fail, self.astar_fail_confirm)
        L("  leg          = re-infer at %.0f%% of each leg; final handoff <= %.2fm",
          100.0 * self.leg_fraction, self.final_handoff_m)
        L("  intrinsics   : fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
          self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
          self.intr.width, self.intr.height)
        L("=" * 64)


def main():
    try:
        HybridPlannerNode().start()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The difficulty detection,
# waypoint selection, geometry, HTTP and progress maths live in core.planning
# (replanning.route_difficulty / navdp / planners.common); this node owns ROS I/O
# and the PRIMARY <-> ENGAGED hysteretic state machine.
#
#   sources/outputs:
#     ~astar_status_topic (/path/astar_status)   A* per-attempt Bool (True=route)
#     ~astar_path_topic (/path/waypoints_astar)  raw A* route, echoed while PRIMARY
#     ~bev_topic (/falcon/bev_2d)                BEV OccupancyGrid (doorway test)
#     ~path_topic (/path/waypoints_hybrid)       arbitrated raw path -> path_corrector
#     ~full_path_topic (/path/waypoints_navdp_full)  full NavDP leg, display only
#     ~nav_status_topic (/nav/status)            std_msgs/String, latched: "who is
#                                                planning now" for the viewer HUD
#                                                ("A*" / "NavDP (<reason>)")
#     ~goal_topic (/waypoint_nav/goal)           mission goal (world); the boxed-A*
#                                                rescue aims NavDP here (body frame)
#     ~goal_x ~goal_y (unset)                    optional initial goal
#     ~arrival_radius_m (0.5)                    within this of the goal a rescue leg holds
#     ~frame_id (world)
#   difficulty detector (core route_difficulty):
#     ~difficulty_lookahead_m (2.0)   forward window (m) for the NARROWNESS (doorway) test
#     ~difficulty_skip_m (0.3)        skip this much past the drone (a just-finished
#                                     maneuver must not keep reading as "ahead")
#     ~difficulty_merge_collinear_deg (15.0)  drop route vertices turning less than this
#                                     before measuring (smooth the jagged raw A*); 0=raw
#     ~turn_thresh_deg (70.0)         a route corner turning >= this (deg) is a hard turn
#                                     (only a genuinely sharp corner engages NavDP)
#     ~turn_engage_distance_m (2.0)   engage NavDP once a hard turn is within this distance
#                                     ahead ("close enough to the turn"); A* flies up to it
#     ~turn_corner_span_m (0.7)       entry/exit chord (m) for a corner's net turn (robust
#                                     to a split corner / grid jitter)
#     ~passage_width_m (0.75)         free width (m) below which the route is a doorway
#                                     (measured perpendicular; tight on BOTH sides)
#     ~min_narrow_span_m (0.3)        a doorway must stay narrow over this arclength to
#                                     count (rejects a single BEV speckle cell)
#     ~difficulty_sample_step_m (0.1) perpendicular-march / sampling step (~one cell)
#   hysteresis:
#     ~difficulty_confirm (3)   consecutive "difficult ahead" ticks before engaging NavDP
#     ~recover_confirm (5)      consecutive "easy again + A* ok" ticks before resuming A*
#                               (keep >= difficulty_confirm: sticky return, anti-zigzag)
#     ~min_pass_distance_m (2.0)  don't return to A* until the drone has flown this far
#                               from where NavDP engaged (through the maneuver); 0 = off
#     ~engage_on_astar_fail (false) also engage when A* is boxed in: aims NavDP at the
#                               mission goal in the body frame (like nav_mode:=fallback).
#                               OFF by default -- hybrid is purely difficulty-triggered;
#                               a flaky map (bag replay) would otherwise drive NavDP the
#                               whole route. Use nav_mode:=fallback for boxed-A* rescue.
#     ~astar_fail_confirm (3)   consecutive A* "no route" reports before that rescue
#     ~tick_hz (5.0)            state-machine rate
#   NavDP leg (mirrors combination):
#     ~engage_settle_s (1.0)    brake-settle before the first inference (clean frame)
#     ~leg_execute_fraction (0.5)  re-infer after this fraction of a leg (rest buffers)
#     ~final_handoff_m (1.5)    hand the final approach to A* within this range
#     ~leg_timeout_s (8.0)      watchdog: re-infer if a leg runs this long
#     ~leg_endpoint_radius_m (0.4)  re-infer once within this of the leg end
#     ~max_wait_s (30.0)        hold this long for a slow/blind inference, then A*
#   visibility (core point_visible):
#     ~require_unoccluded (true) require clear line-of-sight, not merely in-FOV
#     ~visibility_depth_tol_m (0.5) slack on the occlusion test
#     ~visibility_patch_half_px (6) depth patch half-size at the pixel
#     ~min_goal_fwd_m (0.5)     ignore A* waypoints nearer than this
#     ~blind_turn_target (true) at a hard turn the post-turn route leaves the FOV, so
#                               nothing is "visible": aim NavDP at the farthest A*
#                               waypoint reachable by GEOMETRY around the corner so it
#                               flies THROUGH the turn (else it holds + reverts to A*)
#   image transport (mirrors navdp_click / combination):
#     ~image_transport (frame_path | topic)
#     ~rgb_topic ~depth_topic ~pose_topic (/xtend/localization) ~pose_type (pose_stamped)
#     ~camera_info_topic ('' = use the fx/fy/cx/cy params; K preferred over P)
#   camera (MUST match the live NavDP stream; launch wires these to navdp_*):
#     ~fx ~fy ~cx ~cy ~img_width (504) ~img_height (294)
#   NavDP server: ~port (8888) ~timeout_s (10.0) ~depth_max_m (5.0)
#   narration (inherited from thinking.Thinker): ~thinking (true; false silences the
#     A* <-> NavDP arbitration story) ~thinking_topic (/nav/thinking) ~thinking_echo
#     (true; also mirror each thought to rosout)
#   misc: ~default_altitude (1.0; used until the first pose arrives)
#
#   Behaviour: PRIMARY echoes A* on ~path_topic and assesses the route AHEAD each
#   tick; after ~difficulty_confirm consecutive "hard turn or narrow passage"
#   verdicts (or ~astar_fail_confirm A* no-route reports) it STOPS the drone and
#   engages NavDP toward the farthest VISIBLE A* waypoint, flying each leg to
#   ~leg_execute_fraction then re-inferring, until the route ahead is easy again
#   AND A* has a route for ~recover_confirm consecutive ticks -> resume A*. Needs
#   the NavDP server up (127.0.0.1:8888) once a hard maneuver is reached.
# ============================================================================
