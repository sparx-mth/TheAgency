#!/usr/bin/env python3
"""
astar_planner_node.py -- ROS1 adapter: 2D BEV OccupancyGrid -> APF-safe waypoints.

Thin glue around the ROS-free planner in
``sparx_agency.core.planning.planners.astar.WeightedAStarPlanner2D``. All of the
algorithm -- weighted cost build (inflation + UNKNOWN weighting), bounding-box
A* with the octile heuristic, line-of-sight smoothing, corner-preserving
resampling, goal snapping and start-prefix trimming -- lives in core and is unit
tested without ROS. This node owns ONLY ROS concerns:

  - rosparams -> WeightedAStarParams + topics/frame/goal,
  - decoding nav_msgs/OccupancyGrid into a core OccupancyGrid2D,
  - the map-warmup gate (hold until FALCON has integrated real FREE cells),
  - goal-click handling and lazy collision/periodic replanning,
  - APF post-processing: recentring the raw A* path off walls (so a corridor
    turn no longer hugs the inside corner) via the repulsive potential field,
  - publishing nav_msgs/Path and logging.

APF safety recentring (``~safety_correct``, default on)
------------------------------------------------------
Raw A* returns the *shortest* path, which hugs walls and cuts corners. We
post-process it with the repulsive potential field ``U_rep`` built from the SAME
BEV grid: ``PotentialFieldLayer`` (core/mapping) generates ``U_rep`` + a
distance field, and ``TrajectorySafetyCorrector`` (core/planning/safety) nudges
each waypoint down ``-∇U_rep`` (perpendicular to the path, capped) so it drifts
toward the centre of free space. The corrected path is published on
``~path_topic`` (the follower flies it); the un-corrected A* path is published
on ``~raw_path_topic`` purely for visualization. Collision/predicted-collision
replanning runs on the corrected (flown) path. Set ``~safety_correct:=false`` to
fly the raw A* path (both topics then carry it).

Drop-in replacement for the legacy falcon_adapter ``astar_planner.py``:
identical core topics and message types (``~path_topic`` adds the APF recentring;
``~raw_path_topic`` is new and additive).

  in   ~bev_topic  (OccupancyGrid)  /falcon/bev_2d
  in   ~drone_ns + /gt_pose (Pose)
  in   ~goal_topic (Point)          /waypoint_nav/goal
  out  ~path_topic (Path, latched)      /path/waypoints      (APF-safe, flown)
  out  ~raw_path_topic (Path, latched)  /path/waypoints_raw  (raw A*, viz only)

See the file footer for the full rosparam list.
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from sparx_agency.core.common.types import Path2D, Pose2D, PlanStatus
from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar import (
    WeightedAStarPlanner2D, WeightedAStarParams)
from sparx_agency.core.planning.safety import (
    TrajectorySafetyCorrector, TrajectoryCorrectionParams)

# nav_msgs/OccupancyGrid int8 convention published by bev_publisher_node.
BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


class AStarPlannerNode:
    def __init__(self):
        rospy.init_node("astar_planner")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "/simple_drone")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        self.path_topic = G("~path_topic", "/path/waypoints")
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        self.frame_id = G("~frame_id", "world")

        self.params = WeightedAStarParams(
            connectivity=int(G("~connectivity", 8)),
            inflate_radius_m=float(G("~inflate_radius_m", 0.4)),
            unknown_blocked=bool(G("~unknown_blocked", False)),
            unknown_cost=float(G("~unknown_cost", 1.0)),
            search_margin_m=float(G("~search_margin_m", 3.0)),
            turn_penalty=float(G("~turn_penalty", 0.3)),
            los_smoothing=bool(G("~los_smoothing", True)),
            waypoint_spacing_m=float(G("~waypoint_spacing_m", 3.0)),
            goal_snap_radius_m=float(G("~goal_snap_radius_m", 2.0)),
            start_skip_m=float(G("~start_skip_m", 0.4)),
            max_expansions=int(G("~max_expansions", 200000)),
            # Corner rounding -> gentler turns for the stop-and-turn follower.
            corner_round=bool(G("~corner_round", True)),
            corner_merge_rad=math.radians(float(G("~corner_merge_deg", 8.0))),
            corner_max_turn_rad=math.radians(float(G("~corner_max_turn_deg", 14.0))),
            corner_chamfer_max_rad=math.radians(float(G("~corner_chamfer_max_deg", 28.0))),
            corner_chamfer_dist_m=float(G("~corner_chamfer_dist_m", 0.5)),
            corner_min_runup_m=float(G("~corner_min_runup_m", 0.6)),
        )
        self.planner = WeightedAStarPlanner2D(self.params)

        # ── APF safety recentring ────────────────────────────────────
        # Post-process the raw A* path away from walls toward corridor centres
        # using the repulsive field U_rep built from the SAME BEV grid. The
        # field generator (PotentialFieldLayer, cv2-backed) and the recentring
        # algorithm (TrajectorySafetyCorrector, numpy-only) both live in core;
        # this node only feeds the live BEV in and publishes the result.
        self.safety_correct = bool(G("~safety_correct", True))
        self.raw_path_topic = G("~raw_path_topic", "/path/waypoints_raw")
        self._apf_layer = None
        self._corrector = None
        self._pub_forces = None        # repulsive-force MarkerArray publisher (set below)
        if self.safety_correct:
            self._apf_layer = PotentialFieldLayer(
                occ_thresh=float(G("~apf_occ_thresh", 0.65)),
                # sigma_m is the ONLY spatial field knob: it sets the Gaussian
                # repulsion spread (sigma_px = sigma_m / res). PotentialFieldLayer
                # ignores repulse_radius_m, so we do not expose it as a no-op param.
                sigma_m=float(G("~apf_sigma_m", 0.6)),
            )
            self._corrector = TrajectorySafetyCorrector(TrajectoryCorrectionParams(
                # Centering strategy. "line_search" (default) moves each waypoint
                # straight to the lateral potential minimum -- it balances ALL
                # surrounding walls (exact corridor centre), is independent of the
                # field's absolute scale (so it needs NO gain tuning -- the old
                # "scale apf_gain to ~10" hack is gone), and is single-pass (it does
                # not slow down as the range grows). "descent" = legacy iterative
                # gradient descent; gain/max_step below apply only to it.
                centering=str(G("~apf_centering", "line_search")),
                center_step_m=float(G("~apf_center_step_m", 0.05)),
                # Corner aggressiveness: extra lateral search range per 90deg of
                # turn (fraction of max_total_shift) so corners swing WIDE around
                # the inside wall instead of cutting close. 0 disables it.
                corner_swing=float(G("~apf_corner_swing", 1.0)),
                iterations=int(G("~apf_iterations", 5)),         # descent mode only
                gain=float(G("~apf_gain", 1.0)),                 # descent mode only
                max_step_m=float(G("~apf_max_step_m", 0.4)),     # descent mode only
                # max_total_shift_m bounds the displacement off the A* path. For
                # line_search it is the lateral search half-range, so make it wide
                # enough to reach the corridor centre from a wall-hugging path
                # (>= corridor half-width). Raised to 2.0 m for a strong, dominant
                # recentring; the medial-axis search self-limits at the centre and
                # cannot run into a wall, so a large range is safe.
                max_total_shift_m=float(G("~apf_max_total_shift_m", 2.0)),
                # smoothing applies to DESCENT only; line_search lands on the smooth
                # medial axis and skips it (smoothing would drag the path back).
                smoothing_passes=int(G("~apf_smoothing_passes", 2)),
                # >0 also pushes each waypoint to a minimum clearance (needs the
                # distance field, which we always supply); 0 = pure centring.
                min_clearance_m=float(G("~apf_min_clearance_m", 0.0)),
                # Push perpendicular to the path so recentring never slides
                # waypoints fore/aft and distorts the follower's spacing.
                lateral_only=bool(G("~apf_lateral_only", True)),
                # Pin the goal: it is the user's click, not a wall to flee.
                pin_last=bool(G("~apf_pin_last", True)),
            ))
            # TEMP debug: per-waypoint orig/push/new logging each plan (see
            # _log_apf_debug). Disable with ~apf_debug:=false once verified.
            self._apf_debug = bool(G("~apf_debug", True))
            # Safety re-check: revert to the (A*-validated) raw path if the
            # corrected one collides under inflation. If that ever makes red==green
            # in the viewer, set ~apf_collision_recheck:=false to SEE the corrected
            # path (debug only -- it may clip an inflated obstacle).
            self._apf_collision_recheck = bool(G("~apf_collision_recheck", True))

            # Treat UNKNOWN cells as free (matches A*, which plans through unknown
            # as free). When true (default) the corrector may recentre a waypoint
            # that sits over unknown space, pushed only by the KNOWN walls; the
            # repulsive field already ignores unknown cells. False restores the old
            # "see-only-what-you-see" freeze (waypoints over unknown stay put).
            self._apf_unknown_free = bool(G("~apf_treat_unknown_as_free", True))

            # Unknown-area damping. Treating unknown as free means a half-mapped
            # corridor has no opposing wall to balance the push, so the medial-axis
            # correction over-shoots toward the unmapped side. Scale each waypoint's
            # shift by the fraction of KNOWN cells around its corrected position
            # (1.0 = fully mapped, walls both sides -> full correction; lower near
            # unknown -> damped). As the map fills in, damping fades to none.
            self._apf_unknown_damping = bool(G("~apf_unknown_damping", True))
            self._apf_unknown_radius_m = float(G("~apf_unknown_radius_m", 0.75))

            # Repulsive-force arrows: F_rep = -grad U_rep sampled at each waypoint,
            # published as a visualization_msgs/MarkerArray of ARROW markers so RViz
            # (and the BEV viewer) show how the obstacles push the path off the walls.
            self._publish_force_markers = bool(G("~publish_forces", True))
            self.forces_topic = G("~forces_topic", "/path/forces")
            # Arrow length = force_arrow_scale * |F_rep| (proportional, made large
            # for visibility); arrows are lifted to force_arrow_z in the world frame.
            self.force_arrow_scale = float(G("~force_arrow_scale", 1.0))
            self.force_arrow_z = float(G("~force_arrow_z", 0.0))
            # Coarse field of F_rep over the free cells (every force_field_stride
            # cells), so EVERY wall section's push is visible, not just at waypoints.
            self._publish_force_field = bool(G("~publish_force_field", True))
            self.force_field_stride = max(1, int(G("~force_field_stride", 4)))
            if self._publish_force_markers:
                self._pub_forces = rospy.Publisher(self.forces_topic, MarkerArray,
                                                   queue_size=1, latch=True)

        # ── Planning cadence ─────────────────────────────────────────
        # A* -- and the APF post-process that runs once per A* path -- fire on a
        # STRICT periodic timer (plan_period_s). The published trajectory is then
        # frozen until the next tick: the live BEV keeps updating in the
        # background but never triggers a replan, so the follower is not chasing a
        # path (and an APF field) that shifts every map frame. A goal click still
        # plans immediately.
        self.plan_period_s = float(G("~plan_period_s", 3.0))

        # Optional MID-CYCLE replan triggers. OFF by default -- keeping them off is
        # what freezes the trajectory between ticks. Enable any of them only to
        # react to obstacles faster than plan_period_s (the path and its APF field
        # may then change between ticks).
        self.replan_on_collision = bool(G("~replan_on_collision", False))
        self.replan_on_bev = bool(G("~replan_on_bev", False))

        # Dynamics-aware replan: the follower publishes its predicted (stop-and-
        # turn) trajectory; replan if THAT collides even when the geometric path
        # does not (overshoot into a wall the straight path misses). Guarded by a
        # freshness check, the same confirm streak, and a consecutive-replan cap.
        # Off by default with the rest of the mid-cycle replans (keeps the freeze).
        self.replan_on_predicted_collision = bool(
            G("~replan_on_predicted_collision", False))
        self.predicted_path_topic = G("~predicted_path_topic", "/path/predicted")
        self.max_predicted_replans = int(G("~max_predicted_replans", 3))

        # Collision-replan debounce. A single noisy depth frame can paint a
        # spurious occupied cell across the path; reacting on the first frame
        # makes the slow platform veer away and then veer back when the noise
        # clears -- a costly oscillation. So a collision must persist over
        # ~replan_collision_confirm consecutive BEV frames ("wait for more depth
        # to confirm the obstacle") before we replan. A real obstacle persists
        # and clears the gate within a frame or two, while the 0.4m inflation
        # buffer in path_collides keeps that brief, frame-bounded delay safe.
        # The streak resets after each replan, so this also caps the sustained
        # replan rate WITHOUT ever holding a known-colliding path (a time-based
        # cooldown would do the latter and is unsafe here: replanning is the
        # only obstacle avoidance). Set confirm=1 for the legacy behavior.
        self.replan_collision_confirm = max(1, int(G("~replan_collision_confirm", 2)))
        self._collision_streak = 0       # consecutive colliding BEV frames

        # Map-warmup gate: refuse to plan until the BEV holds at least this many
        # genuine FREE cells. FREE comes only from FALCON's real depth fusion, so
        # it is a true "the map has warmed up" signal -- without it an all-UNKNOWN
        # map (with unknown_blocked=False) looks like open space and the drone
        # would cruise blind. A goal click does NOT bypass the gate.
        self.min_free_cells = int(G("~min_free_cells_to_plan", 80))
        self._warmed_up = self.min_free_cells <= 0

        gx = G("~goal_x", None)
        gy = G("~goal_y", None)
        self.goal = (Pose2D(float(gx), float(gy))
                     if gx is not None and gy is not None else None)

        self.pose = None          # Pose2D
        self.grid = None          # OccupancyGrid2D (one per BEV message)
        self.has_plan = False
        self.last_points = ()     # tuple[Pose2D] of the published path
        self.fail_reason = "(not tried yet)"

        self._last_plan_stamp = rospy.Time(0)     # stamp of the published path
        self._predicted_points = ()               # tuple[Pose2D] from the follower
        self._predicted_stamp = rospy.Time(0)
        self._pred_collision_streak = 0
        self._predicted_replans = 0               # consecutive; reset when clear

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        # Raw (un-corrected) A* path -- visualization only; the follower ignores it.
        self.pub_raw_path = rospy.Publisher(self.raw_path_topic, Path,
                                            queue_size=1, latch=True)
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.drone_ns + "/gt_pose", Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.goal_topic, Point, self._goal_cb, queue_size=1)
        if self.replan_on_predicted_collision:
            rospy.Subscriber(self.predicted_path_topic, Path, self._predicted_cb,
                             queue_size=1)

        # Strict A* cadence (always on): every plan_period_s recompute A* + APF
        # once and publish; the trajectory is frozen in between.
        rospy.Timer(rospy.Duration(self.plan_period_s), self._plan_tick)
        rospy.Timer(rospy.Duration(2.0), self._status)
        self._banner()

    # ─── Callbacks ───────────────────────────────────────────────
    def _pose_cb(self, msg):
        first = self.pose is None
        self.pose = Pose2D(float(msg.position.x), float(msg.position.y))
        if first:
            rospy.loginfo("astar_planner: first pose start=(%.2f,%.2f)",
                          self.pose.x, self.pose.y)

    def _goal_cb(self, msg):
        new = Pose2D(float(msg.x), float(msg.y))
        same = (self.goal is not None
                and abs(new.x - self.goal.x) < 1e-3
                and abs(new.y - self.goal.y) < 1e-3)
        rospy.loginfo("astar_planner: GOAL RECEIVED (%.2f, %.2f)%s",
                      new.x, new.y, "  (== current goal)" if same else "")
        # An explicit click always forces a fresh plan: clear state + cost cache.
        self.goal = new
        self.has_plan = False
        self._collision_streak = 0
        self._pred_collision_streak = 0
        self._predicted_replans = 0
        self.planner.invalidate_cache()
        if not self._try_plan():
            rospy.logwarn("astar_planner: click goal (%.2f, %.2f) accepted but no "
                          "path yet -- reason=%s (will retry on next BEV)",
                          new.x, new.y, self.fail_reason)

    def _bev_cb(self, msg):
        """Store the latest BEV. A map update does NOT replan -- planning is driven
        strictly by the periodic timer (and goal clicks), so the APF correction
        runs once per A* path and the published trajectory stays frozen between
        ticks. The only thing a map update triggers is the FIRST plan, so we start
        as soon as the map warms up instead of waiting a full period. The
        replan_on_* hooks below are off by default; enable them to react to
        obstacles mid-cycle (at the cost of a path that can move between ticks).
        """
        first = self.grid is None
        self.grid = self._decode(msg)
        if first:
            i = msg.info
            rospy.loginfo("astar_planner: first BEV W=%d H=%d res=%.2f origin=(%.1f,%.1f)",
                          i.width, i.height, i.resolution,
                          i.origin.position.x, i.origin.position.y)
        if not self.has_plan:
            self._try_plan()                 # one-shot: first path ASAP after warmup
        elif self.replan_on_bev:
            self._try_plan()
        elif self.replan_on_collision:
            self._collision_replan_check()
        if self.has_plan and self.replan_on_predicted_collision:
            self._predicted_collision_check()

    def _predicted_cb(self, msg):
        """Store the follower's predicted trajectory (world Pose2D) + its stamp."""
        self._predicted_points = tuple(
            Pose2D(float(p.pose.position.x), float(p.pose.position.y))
            for p in msg.poses)
        self._predicted_stamp = msg.header.stamp

    # ─── Decode ──────────────────────────────────────────────────
    def _decode(self, msg):
        """nav_msgs/OccupancyGrid -> core OccupancyGrid2D (BEV value convention)."""
        i = msg.info
        try:
            data = np.frombuffer(bytes(bytearray(msg.data)), dtype=np.int8)
        except Exception:
            data = np.asarray(msg.data, dtype=np.int8)
        data = data.reshape(i.height, i.width).astype(np.int16)
        params = OccupancyGrid2DParams(
            resolution=i.resolution,
            origin_x=i.origin.position.x,
            origin_y=i.origin.position.y,
            frame_id=self.frame_id,
        )
        return OccupancyGrid2D(data, params, values=BEV_VALUES)

    # ─── Planning ────────────────────────────────────────────────
    def _collision_replan_check(self):
        """Replan only on a *confirmed* path collision, not single-frame noise.

        Transient depth noise can briefly paint an occupied cell across the
        published path. Replanning on the first such frame makes the slow
        platform turn away, and when the noise clears it turns back -- an
        oscillation. So the collision must hold for ``replan_collision_confirm``
        consecutive BEV frames (the "wait for more depth" gate) before we act.
        A genuine obstacle persists and clears the gate within a frame or two,
        while the 0.4m inflation buffer in ``path_collides`` keeps that brief,
        frame-bounded delay safe. We never *hold* a confirmed colliding path:
        once confirmed we replan immediately (replanning is the only avoidance).
        """
        if not self.planner.path_collides(self.grid, self.last_points):
            if self._collision_streak:
                rospy.loginfo("astar_planner: path clear again after %d colliding "
                              "frame(s) -- treated as noise, no replan",
                              self._collision_streak)
            self._collision_streak = 0
            return

        self._collision_streak += 1
        if self._collision_streak < self.replan_collision_confirm:
            rospy.loginfo("astar_planner: path collision unconfirmed %d/%d -- "
                          "waiting for more depth before replanning",
                          self._collision_streak, self.replan_collision_confirm)
            return

        rospy.logwarn("astar_planner: path blocked on %d consecutive BEV frame(s) "
                      "-- replanning", self._collision_streak)
        self.has_plan = False
        self._try_plan()      # resets _collision_streak on success

    def _predicted_collision_check(self):
        """Replan when the follower's *predicted* trajectory hits an obstacle.

        The geometric path can be clear while the drone's real stop-and-turn
        motion overshoots a corner into a wall. We run the same inflated-obstacle
        line check on the predicted rollout. A freshness guard ignores a
        prediction made against a now-superseded path; the same confirm streak
        debounces depth noise; a consecutive-replan cap (reset once the
        prediction is clear again) prevents a replan oscillation while leaving
        the geometric collision replan fully active.
        """
        if (len(self._predicted_points) < 2
                or self._predicted_stamp < self._last_plan_stamp):
            return
        if not self.planner.path_collides(self.grid, self._predicted_points):
            self._pred_collision_streak = 0
            self._predicted_replans = 0
            return
        self._pred_collision_streak += 1
        if self._pred_collision_streak < self.replan_collision_confirm:
            return
        if self._predicted_replans >= self.max_predicted_replans:
            rospy.logwarn_throttle(
                5.0, "astar_planner: predicted-collision replan cap (%d) reached "
                "-- holding (geometric replan still active)", self.max_predicted_replans)
            return
        rospy.logwarn("astar_planner: PREDICTED trajectory blocked on %d frame(s) "
                      "-- dynamics-aware replan", self._pred_collision_streak)
        self._predicted_replans += 1
        self.has_plan = False
        self._try_plan()

    def _plan_tick(self, _evt):
        """Strict A* cadence: every plan_period_s recompute A* + APF once and
        publish, then freeze until the next tick. Fires regardless of has_plan, so
        each cycle yields a fresh path from the latest map and current pose -- the
        one periodic source of a new trajectory once the first plan exists."""
        self._try_plan()

    def _try_plan(self):
        if self.grid is None:
            self.fail_reason = "no BEV yet"; return False
        if self.goal is None:
            self.fail_reason = "no goal set"; return False
        if self.pose is None:
            self.fail_reason = "no pose yet"; return False
        if not self._warmup_ok():
            return False

        t0 = rospy.Time.now()
        req = PlanRequest(start=self.pose, goal=self.goal, frame_id=self.frame_id)
        res = self.planner.plan(req, self.grid)
        if not res.ok:
            self.fail_reason = res.message
            rospy.logwarn_throttle(
                5.0, "astar_planner: PLAN FAILED start=(%.2f,%.2f) goal=(%.2f,%.2f) "
                "status=%s reason=%s", self.pose.x, self.pose.y, self.goal.x,
                self.goal.y, res.status.value, res.message)
            return False

        anchored = self._anchor_to_drone(res.path)   # path begins at the drone pose
        raw_points = anchored.points
        safe_points = self._apply_safety(anchored)
        # The follower flies -- and collision checks run against -- the SAFE path.
        self.last_points = safe_points
        self._publish(safe_points, (rospy.Time.now() - t0).to_sec())
        self._publish_raw(raw_points)
        self.has_plan = True
        self._collision_streak = 0    # every fresh path starts its debounce clean
        self._pred_collision_streak = 0
        self.fail_reason = "(success)"
        return True

    # ─── APF safety recentring ───────────────────────────────────
    def _anchor_to_drone(self, path):
        """Make the path originate exactly at the drone's current pose.

        The A* planner trims waypoints within ~start_skip_m of the start and never
        re-inserts the start pose, so the published path begins a step ahead of the
        drone (it appears to "start at the second waypoint"). Prepend the live pose
        as waypoint 0 -- or replace a near-coincident first point with it -- so both
        the raw and the corrected path start exactly at the drone. The corrector
        pins waypoint 0, so the origin stays fixed through correction.
        """
        pts = path.points
        if self.pose is None:
            return path
        p0 = pts[0]
        if math.hypot(p0.x - self.pose.x, p0.y - self.pose.y) < 0.15:
            new_pts = (self.pose,) + tuple(pts[1:])   # replace near-coincident start
        else:
            new_pts = (self.pose,) + tuple(pts)       # prepend the true origin
        return Path2D(points=new_pts, frame_id=path.frame_id,
                      metadata=dict(path.metadata))

    def _apply_safety(self, raw_path):
        """Recentre the raw A* path away from walls; return the safe waypoints.

        Builds the repulsive field from the current BEV and runs
        ``TrajectorySafetyCorrector`` over the path. The corrector pushes against
        a *softer* obstacle model (the Gaussian U_rep) than A*'s hard inflation, so
        the recentred path is re-validated against the planner's own collision
        gate. That re-validation is now PER-WAYPOINT (``_clip_to_clear``): only the
        waypoints whose corrected segment would clip an inflated obstacle are
        pulled back toward raw, instead of reverting the whole path. The old
        whole-path revert was goal-sensitive -- one corner-cut chord (whose exact
        position depends on the goal) zeroed out the entire correction, so a tiny
        goal shift could flip correction fully on/off. The flown path stays never
        less safe than plain A*. Any APF *failure* falls back (loudly) to raw.
        """
        if not self.safety_correct:
            return raw_path.points
        try:
            self._build_field()
            if self._pub_forces is not None:
                self._publish_forces(raw_path.points)
            safe = self._corrector.correct_path(raw_path)
            if self._apf_debug:
                self._log_apf_debug(raw_path.points, safe.points)
            safe_pts = safe.points
            if self._apf_unknown_damping:
                # Damp the push where it heads into unknown (unbalanced) BEFORE the
                # collision clip, so both only ever pull the path back toward raw.
                safe_pts = self._dampen_unknown(raw_path.points, safe_pts)
            pts = (self._clip_to_clear(raw_path.points, safe_pts)
                   if self._apf_collision_recheck else safe_pts)
            n_moved = sum(1 for r, s in zip(raw_path.points, pts)
                          if math.hypot(s.x - r.x, s.y - r.y) > 1e-3)
            rospy.loginfo_throttle(
                5.0, "astar_planner: APF recentred %d/%d waypoint(s)",
                n_moved, len(pts))
            return pts
        except Exception as exc:                    # noqa: BLE001 -- stay flying
            rospy.logwarn_throttle(
                5.0, "astar_planner: APF correction failed (%s) -- "
                "publishing raw A* path on %s", exc, self.path_topic)
            return raw_path.points

    def _map_confidence(self, x, y):
        """Fraction of KNOWN cells in a disk of ~apf_unknown_radius_m around (x, y).

        1.0 = fully mapped neighbourhood (walls observed on both sides, so the
        repulsive forces balance); lower when (x, y) is in or near unknown space,
        where the push is unbalanced. Used to scale down the correction there.
        """
        g = self.grid
        rad = max(1, int(round(self._apf_unknown_radius_m / g.resolution)))
        gx, gy = g.world_to_grid(x, y)
        x0, x1 = max(0, gx - rad), min(g.width, gx + rad + 1)
        y0, y1 = max(0, gy - rad), min(g.height, gy + rad + 1)
        win = g.grid[y0:y1, x0:x1]
        if win.size == 0:
            return 1.0
        return float(np.count_nonzero(win != g.values.unknown)) / float(win.size)

    def _dampen_unknown(self, raw_points, corrected_points):
        """Scale each waypoint's correction by the map confidence at its corrected
        position, so a push into/near unknown space (no opposing wall to balance
        it) is damped while a push to the centre of a fully-mapped corridor is kept
        at full strength. ``final = raw + confidence * (corrected - raw)``.
        """
        out = []
        for r, c in zip(raw_points, corrected_points):
            conf = self._map_confidence(c.x, c.y)
            if conf >= 1.0 - 1e-6:
                out.append(c)
            else:
                out.append(Pose2D(r.x + conf * (c.x - r.x),
                                   r.y + conf * (c.y - r.y), c.yaw))
        return tuple(out)

    def _clip_to_clear(self, raw_points, safe_points):
        """Per-waypoint safety clip of the APF-corrected path against inflation.

        Replaces the all-or-nothing revert (which republished the ENTIRE raw path
        whenever a single corrected segment clipped an inflated obstacle, so a tiny
        goal change could zero out every waypoint). Each interior waypoint is
        pulled back toward its raw position only as far as needed to keep BOTH its
        adjacent segments clear (bisection), so a single corner-cut reverts just
        that waypoint while the rest stay centred. Never less safe than plain A*: a
        waypoint that cannot be cleared falls back to its raw position.
        """
        out = list(safe_points)
        n = len(out)
        if n < 3:
            return tuple(out)

        def clear(a, b):
            return not self.planner.path_collides(self.grid, (a, b))

        # Re-evaluate EVERY interior waypoint to its most-centred clear position each
        # sweep (not only colliding ones): pulling one waypoint back can later free a
        # neighbour to return to full correction, so we recompute rather than latch a
        # one-time revert. Converges in a few sweeps; endpoints stay pinned.
        for _sweep in range(3):
            changed = False
            for i in range(1, n - 1):
                full = safe_points[i]                # t=1: full correction
                if clear(out[i - 1], full) and clear(full, out[i + 1]):
                    new = full
                else:
                    rx, ry = raw_points[i].x, raw_points[i].y
                    sx, sy = safe_points[i].x, safe_points[i].y
                    lo, hi, best = 0.0, 1.0, raw_points[i]   # t: 0=raw .. 1=corrected
                    for _ in range(6):           # bisect for the most-centred clear t
                        t = 0.5 * (lo + hi)
                        cand = Pose2D(rx + t * (sx - rx), ry + t * (sy - ry), full.yaw)
                        if clear(out[i - 1], cand) and clear(cand, out[i + 1]):
                            best, lo = cand, t
                        else:
                            hi = t
                    new = best
                if math.hypot(new.x - out[i].x, new.y - out[i].y) > 1e-6:
                    out[i] = new
                    changed = True
            if not changed:
                break
        return tuple(out)

    def _log_apf_debug(self, raw_points, safe_points):
        """TEMP DEBUG (~apf_debug): prove the scaled APF push reaches the waypoints.

        Per plan cycle, logs the most-shifted waypoint's Original [x,y], applied
        repulsive push [dx,dy] (= New - Original: the scaled force actually added
        to the X/Y coordinates) and New [x,y], plus a summary over all waypoints.
        Reading it:
          * every |push| ~ 0  -> field/gain too weak, or waypoints far from walls
            (raise ~apf_gain / ~apf_max_total_shift_m, or there is nothing to push
            off here);
          * pushes non-zero but red==green in the viewer -> the collision re-check
            reverted to raw (see the warn above; ~apf_collision_recheck).
        Disable with ~apf_debug:=false once verified.
        """
        cp = self._corrector.params
        n = min(len(raw_points), len(safe_points))
        if n == 0:
            return
        dxy = [(safe_points[i].x - raw_points[i].x,
                safe_points[i].y - raw_points[i].y) for i in range(n)]
        mags = [math.hypot(dx, dy) for dx, dy in dxy]
        n_moved = sum(1 for m in mags if m > 1e-4)
        i = max(range(n), key=lambda k: mags[k])
        rospy.loginfo(
            "astar_planner APF DEBUG: gain=%.2f max_step=%.2fm max_shift=%.2fm | "
            "%d/%d wp moved, mean|push|=%.3fm, max|push|=%.3fm @wp%d",
            cp.gain, cp.max_step_m, cp.max_total_shift_m, n_moved, n,
            sum(mags) / n, mags[i], i)
        rospy.loginfo(
            "astar_planner APF DEBUG wp%d: orig=[%.3f, %.3f] force=[%.3f, %.3f] "
            "new=[%.3f, %.3f]  |push|=%.3fm",
            i, raw_points[i].x, raw_points[i].y, dxy[i][0], dxy[i][1],
            safe_points[i].x, safe_points[i].y, mags[i])

    def _build_field(self):
        """Install the BEV's repulsive field into the corrector (same world frame).

        The BEV ``OccupancyGrid2D`` in its native (un-flipped) ROS orientation
        has row<->y, col<->x about ``(origin_x, origin_y)`` -- exactly the
        convention ``PotentialFieldSampler`` expects -- so ``U_rep``, the
        distance field and the A* path all share one metric frame. No flip.
        """
        g = self.grid
        raw = g.grid                                       # int16 (H,W) BEV values
        # BEV ints -> probability grid: occupied(100)->1, free(0)->0,
        # unknown(-1)->NaN. PotentialFieldLayer treats NaN as FREE, so U_rep and
        # D_obs already draw their repulsion ONLY from known walls -- unknown space
        # is open (exactly like A* with unknown_blocked=False).
        p_occ = np.where(raw == g.values.occupied, 1.0,
                         np.where(raw == g.values.free, 0.0, np.nan)
                         ).astype(np.float32)
        # known_mask gates which waypoints the corrector may move. Default
        # (apf_treat_unknown_as_free) passes None -> unknown counts as free, so a
        # waypoint over unknown space is still recentred (pushed only by the known
        # walls). Pass the observed mask to restore the old freeze-over-unknown.
        known = None if self._apf_unknown_free else (raw != g.values.unknown)
        u_rep, d_obs = self._apf_layer.compute_from_prob_grid(p_occ, g.resolution)
        # Sample at cell CENTRES: OccupancyGrid2D -- and the A* path it produces
        # via grid_to_world -- places cell (gx,gy) at origin+(gx+0.5)*res, while
        # the sampler indexes from the grid corner. Shift the origin by half a
        # cell so U_rep/D_obs align exactly with the path frame (no ~res/2 bias).
        half = 0.5 * g.resolution
        self._corrector.set_field(
            u_rep, g.resolution, g.origin_x + half, g.origin_y + half,
            d_obs=d_obs, known_mask=known)

    def _warmup_ok(self):
        """Hold until the BEV has enough genuine FREE cells (one-shot latch)."""
        if self._warmed_up:
            return True
        n_free = int((self.grid.grid == self.grid.values.free).sum())
        if n_free < self.min_free_cells:
            self.fail_reason = "map warming up: %d/%d FREE cells" % (n_free, self.min_free_cells)
            rospy.loginfo_throttle(2.0, "astar_planner: %s -- holding", self.fail_reason)
            return False
        self._warmed_up = True
        rospy.loginfo("astar_planner: map warmed up (%d FREE cells >= %d) -- planning enabled",
                      n_free, self.min_free_cells)
        return True

    # ─── Publish / status ────────────────────────────────────────
    def _path_msg(self, points):
        """Build a nav_msgs/Path (identity orientation) from Pose2D points."""
        m = Path()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        for pt in points:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = pt.x
            ps.pose.position.y = pt.y
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        return m

    def _publish(self, points, plan_dt_s):
        """Publish the SAFE (flown) path on ~path_topic and stamp it."""
        m = self._path_msg(points)
        self._last_plan_stamp = m.header.stamp   # freshness ref for predicted check
        self.pub_path.publish(m)
        length = sum(a.distance_to(b) for a, b in zip(points[:-1], points[1:]))
        rospy.loginfo("astar_planner: PATH PUBLISHED %d wp %.2fm plan=%.0fms "
                      "first=(%.2f,%.2f) last=(%.2f,%.2f)", len(points), length,
                      1000.0 * plan_dt_s, points[0].x, points[0].y,
                      points[-1].x, points[-1].y)

    def _publish_raw(self, points):
        """Publish the un-corrected A* path on ~raw_path_topic (red viz overlay)."""
        self.pub_raw_path.publish(self._path_msg(points))

    def _publish_forces(self, points):
        """Publish F_rep = -grad U_rep at each waypoint as bright-yellow ARROW
        markers (visualization_msgs/MarkerArray) for RViz and the BEV viewer.

        ``field.descent(x, y)`` is the gradient of the Gaussian-summed U_rep, i.e.
        the repulsive force from ALL surrounding obstacles, so each arrow points
        away from the nearby walls toward open space -- direct proof that the field
        pushes the trajectory off the walls. Sampled at the raw A* waypoints, where
        the path hugs the walls and the force is strongest and clearest. Arrow
        length = ``force_arrow_scale`` * |F_rep| (proportional to magnitude).
        """
        field = self._corrector.field if self._corrector is not None else None
        if field is None:
            return
        arr = MarkerArray()
        wipe = Marker()                            # clear last plan's arrows first
        wipe.header.frame_id = self.frame_id
        wipe.ns = "apf_forces"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        stamp = rospy.Time.now()
        for idx, p in enumerate(points):
            f = field.descent(p.x, p.y)            # -grad U_rep = F_rep, as [x, y]
            if f is None:
                continue
            fx, fy = float(f[0]), float(f[1])
            if math.hypot(fx, fy) < 1e-6:
                continue                           # ~no force here -> skip zero arrow
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "apf_forces"
            m.id = idx + 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [
                Point(p.x, p.y, self.force_arrow_z),
                Point(p.x + fx * self.force_arrow_scale,
                      p.y + fy * self.force_arrow_scale, self.force_arrow_z),
            ]
            m.scale.x = 0.05                       # shaft diameter
            m.scale.y = 0.12                       # arrowhead diameter
            m.scale.z = 0.0
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.95  # bright yellow
            m.pose.orientation.w = 1.0             # identity (ARROW uses points[])
            arr.markers.append(m)
        if self._publish_force_field:
            self._append_force_field(arr, stamp)
        self._pub_forces.publish(arr)

    def _append_force_field(self, arr, stamp):
        """Append a coarse grid of F_rep arrows over the free cells (ns 'apf_field').

        Sampling -grad U_rep across the free space -- not only at the waypoints --
        makes it visually clear how much EACH wall section pushes: arrows are dense
        and strong next to walls and fade to nothing in open space. Drawn dimmer and
        thinner than the per-waypoint arrows so the trajectory's forces still stand
        out. ``force_field_stride`` controls density; cells with negligible force
        are skipped so only the wall-influenced region is drawn.
        """
        field = self._corrector.field if self._corrector is not None else None
        if field is None or self.grid is None:
            return
        g = self.grid
        cells = g.grid
        wipe = Marker()                            # clear last plan's field arrows
        wipe.header.frame_id = self.frame_id
        wipe.ns = "apf_field"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        stride = self.force_field_stride
        mid = 0
        for gy in range(0, g.height, stride):
            for gx in range(0, g.width, stride):
                if int(cells[gy, gx]) != g.values.free:
                    continue                       # only sample observed FREE cells
                x, y = g.grid_to_world(gx, gy)
                f = field.descent(x, y)
                if f is None:
                    continue
                fx, fy = float(f[0]), float(f[1])
                if math.hypot(fx, fy) < 1e-2:
                    continue                       # negligible push -> skip (open space)
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = stamp
                m.ns = "apf_field"
                m.id = mid
                mid += 1
                m.type = Marker.ARROW
                m.action = Marker.ADD
                m.points = [
                    Point(x, y, self.force_arrow_z),
                    Point(x + fx * self.force_arrow_scale,
                          y + fy * self.force_arrow_scale, self.force_arrow_z),
                ]
                m.scale.x = 0.025                  # thin shaft
                m.scale.y = 0.06                   # small head
                m.scale.z = 0.0
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.75, 0.0, 0.55  # dim gold
                m.pose.orientation.w = 1.0
                arr.markers.append(m)

    def _status(self, _e):
        if self.has_plan:
            return
        rospy.loginfo("astar_planner waiting: bev=%s pose=%s goal=%s last_reason=%s",
                      "yes" if self.grid is not None else "NO",
                      "yes" if self.pose is not None else "NO",
                      ("(%.2f,%.2f)" % (self.goal.x, self.goal.y)) if self.goal else "NO",
                      self.fail_reason)

    def _banner(self):
        p, L = self.params, rospy.loginfo
        L("=" * 64)
        L("astar_planner (core WeightedAStarPlanner2D)")
        L("  bev  in  = %s", self.bev_topic)
        L("  pose in  = %s/gt_pose", self.drone_ns)
        L("  goal in  = %s", self.goal_topic)
        L("  path out = %s   (APF-safe, flown)", self.path_topic)
        L("  raw  out = %s   (raw A*, viz only)", self.raw_path_topic)
        L("  plan period = %.1fs  (strict A*+APF cadence; trajectory frozen between ticks)",
          self.plan_period_s)
        L("  mid-cycle replan: bev=%s collision=%s predicted=%s",
          self.replan_on_bev, self.replan_on_collision, self.replan_on_predicted_collision)
        L("  goal init= %s", "(%.2f,%.2f)" % (self.goal.x, self.goal.y) if self.goal else "none")
        L("  conn=%d inflate=%.2fm unknown=%s max_seg=%.2fm los=%s turn_pen=%.2f",
          p.connectivity, p.inflate_radius_m,
          "blocked" if p.unknown_blocked else "free x%.1f" % p.unknown_cost,
          p.waypoint_spacing_m, p.los_smoothing, p.turn_penalty)
        L("  search_margin=%.1fm start_skip=%.2fm snap=%.1fm collision_replan=%s",
          p.search_margin_m, p.start_skip_m, p.goal_snap_radius_m, self.replan_on_collision)
        L("  collision replan confirm=%d frame(s)", self.replan_collision_confirm)
        L("  corner_round=%s merge=%.0fdeg max_turn=%.0fdeg chamfer<=%.0fdeg "
          "dist=%.2fm runup=%.2fm", p.corner_round, math.degrees(p.corner_merge_rad),
          math.degrees(p.corner_max_turn_rad), math.degrees(p.corner_chamfer_max_rad),
          p.corner_chamfer_dist_m, p.corner_min_runup_m)
        L("  predicted-collision replan=%s cap=%d topic=%s",
          self.replan_on_predicted_collision, self.max_predicted_replans,
          self.predicted_path_topic)
        if self.safety_correct:
            cp = self._corrector.params
            L("  APF recentre=ON  centering=%s  search=%.2fm corner_swing=%.2f "
              "step=%.2fm pin_goal=%s",
              cp.centering, cp.max_total_shift_m, cp.corner_swing,
              cp.center_step_m, cp.pin_last)
            L("  APF descent-only: iters=%d gain=%.2f max_step=%.2fm smooth=%d | "
              "debug=%s recheck=%s",
              cp.iterations, cp.gain, cp.max_step_m, cp.smoothing_passes,
              self._apf_debug, self._apf_collision_recheck)
            L("  APF unknown: treat_free=%s damping=%s radius=%.2fm",
              self._apf_unknown_free, self._apf_unknown_damping,
              self._apf_unknown_radius_m)
            L("  APF F_rep arrows: publish=%s scale=%.2f -> %s",
              self._publish_force_markers, self.force_arrow_scale, self.forces_topic)
        else:
            L("  APF recentre=OFF (raw A* on both path topics)")
        L("=" * 64)


def main():
    try:
        AStarPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The planning maths live in
# core.planning.planners.astar; this node only maps rosparams -> WeightedAStarParams
# and owns ROS I/O, the warmup gate and replan triggering.
#
#   IO: ~bev_topic (/falcon/bev_2d) ~drone_ns (/simple_drone) [+/gt_pose]
#       ~goal_topic (/waypoint_nav/goal) ~path_topic (/path/waypoints; APF-safe)
#       ~raw_path_topic (/path/waypoints_raw; raw A*, viz only)
#       ~frame_id (world) ~goal_x ~goal_y (initial goal; unset = wait for a click)
#   planner: ~connectivity (8) ~inflate_radius_m (0.4) ~unknown_blocked (false)
#       ~unknown_cost (1.0) ~search_margin_m (3.0) ~turn_penalty (0.3)
#       ~los_smoothing (true) ~waypoint_spacing_m (3.0) ~goal_snap_radius_m (2.0)
#       ~start_skip_m (0.4) ~max_expansions (200000)
#   corner rounding (gentler turns): ~corner_round (true) ~corner_merge_deg (8)
#       ~corner_max_turn_deg (14) ~corner_chamfer_max_deg (28)
#       ~corner_chamfer_dist_m (0.5) ~corner_min_runup_m (0.6)
#   planning cadence: ~plan_period_s (3.0) -- A* AND the once-per-path APF
#       post-process run ONLY on this strict timer (and on a goal click); the
#       published trajectory is frozen between ticks so the follower is not
#       chasing a path/field that shifts every BEV frame.
#   OPTIONAL mid-cycle replan (ALL OFF by default -- enabling any lets the path
#     move between ticks):
#       ~replan_on_collision (false) ~replan_on_bev (false)
#       ~replan_collision_confirm (2; consecutive colliding BEV frames required
#         before a collision replan -- debounces single-frame depth noise; only
#         used when ~replan_on_collision is true)
#       ~replan_on_predicted_collision (false; replan when the follower's predicted
#         stop-and-turn trajectory on ~predicted_path_topic (/path/predicted)
#         collides even if the geometric path does not -- dynamics-aware)
#       ~max_predicted_replans (3; consecutive cap, reset once prediction is clear)
#   warmup gate: ~min_free_cells_to_plan (80; 0 disables)
#   APF safety recentring (post-process the raw A* path off walls toward
#     corridor centres via the repulsive field U_rep; the follower flies the
#     corrected path, raw A* is published on ~raw_path_topic for viz):
#       ~safety_correct (true; false = fly raw A*, both topics carry it)
#       ~apf_occ_thresh (0.65) ~apf_sigma_m (0.6; Gaussian repulsion spread --
#         the only spatial PotentialFieldLayer field-generation knob)
#       ~apf_centering (line_search; moves each waypoint to the MAX-clearance
#         medial axis -- the exact centre, farthest from walls, swinging wide at
#         corners. "descent" = legacy gradient descent)
#       ~apf_center_step_m (0.05; line_search normal-sample spacing)
#       ~apf_corner_swing (1.0; line_search: extra lateral search range per 90deg
#         of turn so corners swing wide instead of cutting close; 0 disables)
#       ~apf_max_total_shift_m (2.0; line_search lateral search half-range = the
#         max recentring -- make it >= the corridor half-width)
#       ~apf_iterations (5) ~apf_gain (1.0) ~apf_max_step_m (0.4)
#         ~apf_smoothing_passes (2) ~apf_lateral_only (true) -- DESCENT ONLY
#       ~apf_pin_last (true; keep the goal click fixed)
#       ~apf_min_clearance_m (0.0; >0 also pushes to a min distance-to-wall)
#       ~apf_collision_recheck (true; per-waypoint clip of the corrected path
#         against inflation -- pulls back only the colliding waypoints, never the
#         whole path; false disables the check)
#       ~apf_treat_unknown_as_free (true; recentre waypoints over UNKNOWN cells too,
#         pushed only by known walls -- matches A*. false = freeze them)
#       ~apf_unknown_damping (true; scale each waypoint's correction by the fraction
#         of KNOWN cells around its corrected position, so a push into/near unmapped
#         space -- where no opposing wall balances it -- is damped. Full correction
#         in fully-mapped corridors; fades back in as the map fills.)
#       ~apf_unknown_radius_m (0.75; confidence-window radius in metres)
#       ~apf_debug (true; TEMP per-waypoint orig/push/new logging each plan cycle)
#   force arrows (F_rep = -grad U_rep at each waypoint -> RViz/BEV, proving the
#     obstacle push direction): ~publish_forces (true) ~forces_topic (/path/forces;
#     visualization_msgs/MarkerArray of bright-yellow ARROW markers)
#       ~force_arrow_scale (1.0; arrow length = scale * |F_rep|)
#       ~force_arrow_z (0.0; world-frame height of the arrows)
#       ~publish_force_field (true; also a coarse F_rep field over the free cells --
#         gold ARROW grid -- showing how hard EACH wall section pushes)
#       ~force_field_stride (4; field-arrow spacing in cells)
# ============================================================================
