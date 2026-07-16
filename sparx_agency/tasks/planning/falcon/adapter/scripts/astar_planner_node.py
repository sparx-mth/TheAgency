#!/usr/bin/env python3
"""
astar_planner_node.py -- ROS1 adapter: 2D BEV OccupancyGrid -> A* waypoints.

Thin glue around the ROS-free planner in
``sparx_agency.core.planning.planners.astar.WeightedAStarPlanner2D``. All of the
algorithm -- weighted cost build (inflation + UNKNOWN weighting), bounding-box
A* with the octile heuristic, line-of-sight smoothing, corner-preserving
resampling, goal snapping and start-prefix trimming -- lives in core and is unit
tested without ROS. This node owns ONLY ROS concerns:

  - rosparams -> WeightedAStarParams + topics/frame/goal,
  - decoding nav_msgs/OccupancyGrid into a core OccupancyGrid2D,
  - the map-warmup gate (hold until FALCON has integrated real FREE cells),
  - goal-click handling and SMART, event-driven replanning (``~smart_replan``,
    default on): the committed route is frozen and replanned only on a confirmed
    collision (safety), a large map discovery (in-corridor or map-wide), a large
    yaw rotation, or a periodic catch-all tick -- all but collision are
    opportunistic (length-hysteresis adopt); see the file footer.
    ``~smart_replan:=false`` restores the legacy periodic timer + optional
    mid-cycle hooks,
  - anchoring the path to the live drone pose,
  - publishing the A* path as nav_msgs/Path and logging.

Separation of concerns (the path corrector is its own node)
-----------------------------------------------------------
This node now publishes ONLY the raw A* path, on ``~path_topic``
(``/path/waypoints_astar``). It does NOT recentre the path off walls. A separate,
planner-agnostic ``path_corrector_node`` subscribes to that topic, reshapes the
path against the same BEV (repulsive potential field by default) and republishes
the corrected path on ``/path/waypoints`` -- the topic the follower flies. To
correct a different planner (NavDP, RRT*, ...) instead, point the corrector's
``~input_path_topic`` at that planner's topic; nothing here changes. With the
corrector disabled (``enabled:=false``) it passes the A* path through unchanged.

Collision / predicted-collision replanning here runs on THIS node's published
(raw A*) path. The corrector only ever makes the flown path safer (it clips any
corrected waypoint back to clear inflated obstacles), so the flown path is never
less safe than the A* path this node already validates.

  in   ~bev_topic  (OccupancyGrid)  /falcon/bev_2d
  in   ~drone_ns + /gt_pose (Pose)
  in   ~goal_topic (Point)          /waypoint_nav/goal
  out  ~path_topic (Path, latched)  /path/waypoints_astar  (raw A* -> corrector)

See the file footer for the full rosparam list.
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, String

from sparx_agency.core.common.math import se3
from sparx_agency.core.common.types import Path2D, Pose2D, PlanStatus, normalize_angle
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar import (
    WeightedAStarPlanner2D, WeightedAStarParams)
from sparx_agency.core.planning.replanning import (
    changed_mask, corridor_mask, count_changed_in_corridor,
    polyline_length, remaining_polyline, route_obstacle_confidence)
from thinking import Thinker

# nav_msgs/OccupancyGrid int8 convention published by bev_publisher_node.
BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)

_TRUE_STRINGS = ("true", "1", "yes", "on")
_FALSE_STRINGS = ("false", "0", "no", "off")


def _get_bool_param(name, default):
    """Read a boolean rosparam, raising on a value that is not clearly boolean.

    roslaunch coerces ``value="true"`` / ``value="false"`` to a real Python bool,
    but it leaves an UNRECOGNISED string (e.g. the typo ``"fales"``) as a raw
    string -- and ``bool("fales")`` is ``True``. A plain ``bool(get_param(...))``
    cast therefore *looks* like sanitisation while silently flipping a default-off
    flag ON (exactly the bug that made the planner replan on every BEV frame). Per
    the repo rule "prefer raising errors over silent fallbacks to default values",
    validate explicitly and raise so a typo fails loudly at node start-up.
    """
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    raise ValueError(
        "rosparam %s expected a boolean (true/false), got %r -- check the launch "
        "file / overrides for a typo (e.g. 'fales' instead of 'false')"
        % (name, value))


class AStarPlannerNode:
    # Quiet-map evaluation cadence (s): under smart_replan, re-run the trigger
    # evaluation this often when no BEV frame arrives (the publisher skips
    # unchanged maps and mapping freezes during turns). Fast enough to catch a
    # rotation or the periodic window promptly; evaluation is cheap (A* only
    # runs when a trigger actually fires).
    SMART_TICK_S = 1.0

    def __init__(self):
        rospy.init_node("astar_planner")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "/simple_drone")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        # Raw A* output. A separate path_corrector_node recentres this against the
        # BEV and republishes the corrected path on /path/waypoints (the flown one).
        self.path_topic = G("~path_topic", "/path/waypoints_astar")
        # Per-attempt plan-status signal (std_msgs/Bool): True = a route was found
        # this attempt, False = no route (res.ok == False). Published only when the
        # planner ACTUALLY ran (never for the not-ready gates: no BEV/goal/pose or
        # warmup), so a consumer counting consecutive fails/successes sees one
        # sample per real planning attempt. The fallback arbiter (nav_mode:=fallback)
        # uses this to switch to NavDP after N failures and back after M successes.
        self.status_topic = G("~status_topic", "/path/astar_status")
        # Human-readable replan events (std_msgs/String), for the BEV viewer HUD:
        # first route, obstacle reroute, boxed-in STOP, discovery reroute. Distinct
        # from ~status_topic (a per-attempt Bool the fallback arbiter counts).
        self.event_topic = G("~event_topic", "/path/astar_event")
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        self.frame_id = G("~frame_id", "world")

        self.params = WeightedAStarParams(
            connectivity=int(G("~connectivity", 8)),
            inflate_radius_m=float(G("~inflate_radius_m", 0.4)),
            unknown_blocked=_get_bool_param("~unknown_blocked", False),
            unknown_cost=float(G("~unknown_cost", 1.0)),
            search_margin_m=float(G("~search_margin_m", 3.0)),
            turn_penalty=float(G("~turn_penalty", 0.3)),
            heading_penalty_m=float(G("~heading_penalty_m", 1.5)),
            los_smoothing=_get_bool_param("~los_smoothing", True),
            waypoint_spacing_m=float(G("~waypoint_spacing_m", 3.0)),
            goal_snap_radius_m=float(G("~goal_snap_radius_m", 2.0)),
            start_skip_m=float(G("~start_skip_m", 0.4)),
            max_expansions=int(G("~max_expansions", 200000)),
            # Corner rounding -> gentler turns for the stop-and-turn follower.
            corner_round=_get_bool_param("~corner_round", True),
            corner_merge_rad=math.radians(float(G("~corner_merge_deg", 8.0))),
            corner_max_turn_rad=math.radians(float(G("~corner_max_turn_deg", 14.0))),
            corner_chamfer_max_rad=math.radians(float(G("~corner_chamfer_max_deg", 28.0))),
            corner_chamfer_dist_m=float(G("~corner_chamfer_dist_m", 0.5)),
            corner_min_runup_m=float(G("~corner_min_runup_m", 0.6)),
        )
        self.planner = WeightedAStarPlanner2D(self.params)

        # ── Planning cadence ─────────────────────────────────────────
        # A* fires on a STRICT periodic timer (plan_period_s). The published path
        # is then frozen until the next tick: the live BEV keeps updating in the
        # background but never triggers a replan, so the follower is not chasing a
        # path that shifts every map frame. A goal click still plans immediately.
        # (The downstream path_corrector likewise re-corrects only when a new A*
        # path is published, so the flown trajectory stays frozen between ticks.)
        self.plan_period_s = float(G("~plan_period_s", 3.0))

        # Optional MID-CYCLE replan triggers. OFF by default -- keeping them off is
        # what freezes the trajectory between ticks. Enable any of them only to
        # react to obstacles faster than plan_period_s (the path and its APF field
        # may then change between ticks).
        self.replan_on_collision = _get_bool_param("~replan_on_collision", False)
        self.replan_on_bev = _get_bool_param("~replan_on_bev", False)

        # Dynamics-aware replan: the follower publishes its predicted (stop-and-
        # turn) trajectory; replan if THAT collides even when the geometric path
        # does not (overshoot into a wall the straight path misses). Guarded by a
        # freshness check, the same confirm streak, and a consecutive-replan cap.
        # Off by default with the rest of the mid-cycle replans (keeps the freeze).
        self.replan_on_predicted_collision = _get_bool_param(
            "~replan_on_predicted_collision", False)
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

        # ── Confidence-aware obstacle confirmation ───────────────────────────
        # The frame-count debounce above only rejects a collision that *flickers*.
        # A stably mis-triggered depth speckle holds the binary BEV cell OCCUPIED
        # for many consecutive frames, so the streak still clears and the drone
        # reroutes -- veering off the cell before it can re-observe (and clean) it.
        # The BEV publisher exposes the temporal filter's per-cell confidence
        # (evidence / t_max, 0..100) on ~conf_topic. Here we require the on-route
        # obstacle's confidence to reach ~replan_confirm_conf (fraction of full
        # evidence) before rerouting: a marginal cell is kept on-route so the drone
        # keeps looking (it either firms up -> reroute, or is seen free and decays
        # -> no reroute), while a solid wall firms up fast and reroutes as before.
        # This is a SECOND gate on top of the frame count -- it only ever DELAYS a
        # reroute, never bypasses the frame debounce. Set 0 to disable (frame count
        # only). Without a confidence grid the node falls back to the frame count.
        self.conf_topic = G("~conf_topic", "/falcon/bev_2d_conf")
        self.replan_confirm_conf = float(G("~replan_confirm_conf", 0.75))
        # Safety backstop: reroute anyway once a collision has persisted this many
        # consecutive frames even if the confidence gate is unmet -- bounds the time
        # the drone may fly toward a persistently-flagged but never-confirmed cell
        # (e.g. one just outside the sensor's vertical band). 0 = no ceiling.
        self.replan_collision_confirm_max = max(
            0, int(G("~replan_collision_confirm_max", 30)))
        self.conf_grid = None            # (H,W) float32 [0,1], from ~conf_topic
        self._conf_key = None            # lattice key of the stored conf grid

        # ── Smart replanning (event-driven; supersedes the periodic + mid-cycle
        #    hooks above when ON) ───────────────────────────────────────────────
        # The slow stop-and-turn platform must NOT be whipsawed by a route that
        # A* re-optimises every map frame: re-publishing a path resets the
        # follower (waypoint index + yaw state), so a route that flips between
        # near-equal alternatives makes the drone re-turn and barely advance.
        # With ~smart_replan (default) the planner freezes the committed route and
        # replans ONLY on these events, all evaluated per BEV frame:
        #   * COLLISION  -- the remaining committed route now crosses a confirmed
        #     obstacle. Safety; bypasses the commit window; if A* finds no way
        #     around, STOP the drone (don't fly the wall) and let NavDP take over.
        #   * DISCOVERY  -- a large chunk of new area appeared: either inside the
        #     route corridor (route-relevant, small floor) or ANYWHERE on the map
        #     (higher floor -- a reveal to the side can still open a shortcut).
        #   * ROTATION   -- the drone swung its yaw by a large angle (mid-turn or
        #     just finished): the camera swept new terrain, re-evaluate.
        #   * PERIODIC   -- ~replan_period_s elapsed since the last A* evaluation:
        #     catch-all re-evaluation even with no other trigger.
        # All but COLLISION are opportunistic: they respect the commit window and
        # the new route is ADOPTED only if it is meaningfully shorter than the
        # remaining committed one (length hysteresis), else the current route is
        # kept (no re-publish, no follower reset) -- so a static map re-evaluates
        # but never whipsaws the follower.
        # ~smart_replan:=false restores the exact legacy behaviour (periodic timer
        # + the optional replan_on_* hooks).
        self.smart_replan = _get_bool_param("~smart_replan", True)
        # Corridor half-width for the route-relevance test (m -> cells at plan time).
        self.replan_corridor_radius_m = float(G("~replan_corridor_radius_m", 1.0))
        # "Many, not a few" floor: CHANGED corridor cells (newly observed, or
        # flipped free<->occupied in already-mapped space) needed to consider an
        # opportunistic replan (at 0.15 m cells, 60 ~= 1.35 m^2 of changed area).
        self.replan_min_new_cells = int(G("~replan_min_new_cells", 60))
        # Commit window: after (re)committing a route, keep flying it for at least
        # this long before an opportunistic replan may reconsider it -- so the slow
        # drone gets time to make progress. Collisions bypass this window.
        self.replan_commit_min_s = float(G("~replan_commit_min_s", 4.0))
        # Length hysteresis: adopt a discovery-triggered candidate only if it is at
        # least this fraction shorter than the remaining committed route. This is
        # what kills the flip-flop between two near-equal L/R routes -- neither is
        # 15% better than the other, so the committed one is kept.
        self.replan_improve_frac = float(G("~replan_improve_frac", 0.15))
        # Rotation trigger: a yaw swing this large since the last commit/evaluation
        # has swept the camera across new terrain (typically revealed OFF-corridor,
        # to the side the drone now faces), so re-evaluate the route. Fires both
        # mid-rotation and right after it completes. <=0 disables.
        self.replan_rotation_min_deg = float(G("~replan_rotation_min_deg", 30.0))
        # Map-wide discovery floor: CHANGED cells ANYWHERE on the BEV needed to
        # re-evaluate. An off-corridor change -- a region opening up to the left,
        # or new obstacles landing in mapped free space -- can still change the
        # best route, so it counts too, just against a higher bar than the
        # route-corridor floor above (at 0.15 m cells, 240 ~= 5.4 m^2). <=0 disables.
        self.replan_min_new_cells_any = int(G("~replan_min_new_cells_any", 240))
        # Periodic catch-all: re-evaluate once this long has passed since the last
        # A* evaluation, even with no other trigger, so a stale route is never kept
        # forever. Adoption still needs the length hysteresis, so a static map
        # re-evaluates but never re-publishes. <=0 disables. (Distinct from the
        # LEGACY ~plan_period_s strict re-publish timer.)
        self.replan_period_s = float(G("~replan_period_s", 7.0))
        # On a confirmed obstacle with NO A* route around it, STOP the drone (2-pt
        # hold) instead of leaving the colliding route latched. status=False still
        # drives the NavDP fallback arbiter.
        self.stop_on_blocked_plan_fail = _get_bool_param(
            "~stop_on_blocked_plan_fail", True)

        # Smart-replan committed-route bookkeeping (all reset together on commit).
        self._map_at_commit = None       # int8 HxW: raw grid values at last commit
        self._grid_key = None            # lattice identity guarding the index diff
        self._commit_time = rospy.Time(0)
        self._progress_idx = 0           # forward-monotone projection hint
        self._blocked_hold = False       # published a STOP hold while boxed in
        # High-water mark of changed corridor cells already evaluated: a
        # suppressed discovery (kept the route) raises this so the SAME reveal does
        # not re-fire A* every commit window; only a FURTHER reveal (n_new above the
        # mark by another min_new_cells) triggers again. Reset on every real commit.
        self._discovery_baseline = 0
        self._discovery_baseline_any = 0     # same, for the map-wide count
        # Yaw at the last commit/evaluation: the rotation trigger measures the
        # drone's swing from here.
        self._yaw_baseline = 0.0

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
        self._last_bev_time = rospy.Time(0)   # when the last BEV frame arrived
        self.has_plan = False
        self.last_points = ()     # tuple[Pose2D] of the published path
        self.fail_reason = "(not tried yet)"

        self._last_plan_stamp = rospy.Time(0)     # stamp of the published path
        self._predicted_points = ()               # tuple[Pose2D] from the follower
        self._predicted_stamp = rospy.Time(0)
        self._pred_collision_streak = 0
        self._predicted_replans = 0               # consecutive; reset when clear

        # Narrates the same discrete planning events as ~event_topic, but in the
        # operator's own words, onto the shared thinking log under the BEV map.
        self.thinker = Thinker("astar_planner")

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher(self.status_topic, Bool, queue_size=1, latch=True)
        self.pub_event = rospy.Publisher(self.event_topic, String, queue_size=5, latch=True)
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.conf_topic, OccupancyGrid, self._conf_cb, queue_size=1)
        rospy.Subscriber(self.drone_ns + "/gt_pose", Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.goal_topic, Point, self._goal_cb, queue_size=1)
        if self.replan_on_predicted_collision:
            rospy.Subscriber(self.predicted_path_topic, Path, self._predicted_cb,
                             queue_size=1)

        # Legacy cadence: the strict periodic replan timer runs ONLY when smart
        # replanning is off. Under smart_replan evaluation is event-driven off
        # _bev_cb, PLUS a 1 Hz quiet-map tick: the BEV publisher skips unchanged
        # frames and mapping freezes while the drone turns, so without the tick a
        # silent BEV would starve the rotation and periodic triggers entirely.
        # (The tick only EVALUATES -- adoption still needs the length hysteresis,
        # so this never resurrects the periodic re-publish oscillation.)
        if not self.smart_replan:
            rospy.Timer(rospy.Duration(self.plan_period_s), self._plan_tick)
        else:
            rospy.Timer(rospy.Duration(self.SMART_TICK_S), self._smart_tick)
        rospy.Timer(rospy.Duration(2.0), self._status)
        self._banner()

    # ─── Callbacks ───────────────────────────────────────────────
    def _pose_cb(self, msg):
        first = self.pose is None
        # Keep the drone's YAW: heading-aware A* (heading_penalty_m) reads
        # request.start.yaw to prefer flying the way the drone already looks.
        yaw = se3.yaw_from_quaternion((msg.orientation.x, msg.orientation.y,
                                       msg.orientation.z, msg.orientation.w))
        self.pose = Pose2D(float(msg.position.x), float(msg.position.y), float(yaw))
        if first:
            rospy.loginfo("astar_planner: first pose start=(%.2f,%.2f, yaw=%.0fdeg)",
                          self.pose.x, self.pose.y, math.degrees(yaw))

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
        if self.smart_replan:
            self._reset_smart_state()
            cand = (self._plan_candidate()
                    if self.grid is not None and self.pose is not None else None)
            if cand is not None:
                self._commit(cand)
            else:
                self._publish_status(False)   # no route yet -> NavDP fallback
                rospy.logwarn("astar_planner: click goal (%.2f, %.2f) accepted but no "
                              "path yet -- reason=%s (will retry on next BEV)",
                              new.x, new.y, self.fail_reason)
            return
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
        self._last_bev_time = rospy.Time.now()
        if first:
            i = msg.info
            rospy.loginfo("astar_planner: first BEV W=%d H=%d res=%.2f origin=(%.1f,%.1f)",
                          i.width, i.height, i.resolution,
                          i.origin.position.x, i.origin.position.y)
        if self.smart_replan:
            self._smart_evaluate()           # event-driven collision + discovery
            return
        if not self.has_plan:
            self._try_plan()                 # one-shot: first path ASAP after warmup
        elif self.replan_on_bev:
            self._try_plan(allow_suppress=True)
        elif self.replan_on_collision:
            self._collision_replan_check()
        if self.has_plan and self.replan_on_predicted_collision:
            self._predicted_collision_check()

    def _conf_cb(self, msg):
        """Store the BEV per-cell OCCUPIED confidence (int8 0..100 -> [0,1]).

        Kept with its lattice key so the collision gate uses it only when it is
        co-registered with the current BEV (same origin/resolution/shape); a stale
        or mismatched grid is ignored and the gate falls back to the frame count.
        """
        i = msg.info
        try:
            data = np.frombuffer(bytes(bytearray(msg.data)), dtype=np.int8)
        except Exception:
            data = np.asarray(msg.data, dtype=np.int8)
        self.conf_grid = (data.reshape(i.height, i.width).astype(np.float32) / 100.0)
        self._conf_key = (i.height, i.width, round(i.origin.position.x, 6),
                          round(i.origin.position.y, 6), round(i.resolution, 6))

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
        if not self._obstacle_confirmed(self.last_points, self._collision_streak):
            return            # obstacle not yet confident -- keep looking

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
        if not self._obstacle_confirmed(self._predicted_points,
                                        self._pred_collision_streak):
            return            # predicted obstacle not yet confident -- keep looking
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

    # ─── Confidence-aware obstacle confirmation ──────────────────────
    def _obstacle_confirmed(self, points, streak):
        """Second confirmation stage (after the frame-count debounce): is the map
        actually CONFIDENT the on-route obstacle is real, or is it a low-confidence
        speckle that another look would clear?

        Reads the BEV per-cell confidence (evidence / t_max) at the occupied cells
        ``points`` crosses and reroutes only once it reaches ~replan_confirm_conf.
        A marginal cell is kept on-route so the drone keeps observing it (it either
        firms up -> reroute, or is seen free and decays -> no reroute), which is
        exactly what stops the noisy-map veer-away/veer-back churn. A generous
        frame ceiling (~replan_collision_confirm_max) still forces a reroute so the
        drone never flies indefinitely toward an unconfirmed cell. Degrades to the
        pure frame-count gate (returns True) when the gate is disabled
        (~replan_confirm_conf<=0) or no co-registered confidence grid is available.

        Args:
            points: World waypoints whose crossed obstacle is being confirmed.
            streak: The caller's current consecutive-collision frame count (used
                for the safety ceiling; geometric and predicted checks pass their
                own streak).

        Returns:
            True  -> confirmed: reroute now.
            False -> not yet confident: keep the committed route and keep looking.
        """
        if self.replan_confirm_conf <= 0.0:
            return True                       # confidence gate disabled
        conf = self._route_obstacle_confidence(points)
        if conf is None:
            return True                       # no confidence grid -> frame count only
        if conf >= self.replan_confirm_conf:
            return True
        if (self.replan_collision_confirm_max > 0
                and streak >= self.replan_collision_confirm_max):
            rospy.logwarn("astar_planner: on-route obstacle still unconfirmed by "
                          "confidence (%.2f < %.2f) after %d frames (ceiling %d) -- "
                          "rerouting", conf, self.replan_confirm_conf, streak,
                          self.replan_collision_confirm_max)
            return True
        rospy.loginfo_throttle(
            1.0, "astar_planner: on-route obstacle confidence %.2f < %.2f -- keeping "
            "route, re-observing (frame %d/%s)", conf, self.replan_confirm_conf,
            streak, self.replan_collision_confirm_max or "inf")
        return False

    def _route_obstacle_confidence(self, points):
        """Peak BEV confidence of the occupied cells ``points`` crosses, or None
        when no confidence grid co-registered with the current BEV is available (the
        caller then confirms on the frame count alone)."""
        if (self.conf_grid is None or self.grid is None
                or self._conf_key != self._grid_key_of(self.grid)):
            return None
        inflate_cells = int(round(self.params.inflate_radius_m / self.grid.resolution))
        return route_obstacle_confidence(
            self.grid, self.conf_grid, points, inflate_cells)

    # ─── Smart replanning (event-driven; ~smart_replan) ──────────────
    def _smart_tick(self, _evt):
        """Quiet-map heartbeat: run the trigger evaluation on the LAST grid when
        the BEV has gone silent (unchanged map / mapping frozen during a turn).
        Without this, the rotation and periodic triggers -- which do not depend on
        map changes at all -- would only ever be checked when a map change happens
        to arrive. A frame that arrived within the last tick already evaluated."""
        if self.grid is None:
            return
        if (rospy.Time.now() - self._last_bev_time).to_sec() < self.SMART_TICK_S:
            return
        self._smart_evaluate()

    def _smart_evaluate(self):
        """Decide, on one BEV frame, whether to replan the committed route.

        Order matters: (1) get the first route / recover from a boxed-in stop,
        (2) COLLISION (safety, every frame, bypasses the commit window),
        (3) lattice guard for the cross-frame diff, (4) commit window, then
        (5) the opportunistic triggers -- ROTATION, DISCOVERY (corridor or
        map-wide), PERIODIC -- all length-hysteresis adopt. Between events the
        committed route is frozen -- never re-published -- so the follower keeps
        its progress and the drone actually flies.
        """
        if self.grid is None or self.goal is None or self.pose is None:
            return
        if not self._warmup_ok():
            return
        # (1) No committed route yet, or recovering from a boxed-in STOP hold:
        #     keep planning every frame until A* succeeds, then commit + resume.
        if not self.has_plan or self._blocked_hold:
            first, recovering = not self.has_plan, self._blocked_hold
            cand = self._plan_candidate()
            if cand is not None:
                self._commit(cand)            # clears has_plan/_blocked_hold
                if first:
                    self._publish_event(
                        "PLAN: first route to goal",
                        "I have a route to the goal -- setting off")
                else:
                    self._publish_event(
                        "REPLAN: route reopened -> resuming A*",
                        "The way is open again -- resuming my own route")
            else:
                self._publish_status(False)   # A* cannot plan -> NavDP fallback
            return
        pose = self.pose
        remaining, self._progress_idx = remaining_polyline(
            self.last_points, pose, self._progress_idx)
        # (2) Collision: the remaining committed route now crosses a confirmed
        #     obstacle. Safety -- bypasses the commit window.
        if self._collision_confirmed(remaining, pose):
            self._forced_replan()
            return
        # The committed route is valid this frame. Emit one status=True so a
        # fallback arbiter counting consecutive successes (to resume A* from NavDP)
        # is fed on every frame, not only when the route changes.
        self._publish_status(True)
        # (3) Lattice guard: an index-wise map diff is only meaningful on a
        #     world-fixed grid. On any origin/resolution/shape change, reseed the
        #     snapshot and skip discovery this frame (never diff a shifted grid).
        key = self._grid_key_of(self.grid)
        if self._map_at_commit is None or key != self._grid_key:
            self._reseed_known(key)
            return
        # (4) Commit window: give the slow platform time to fly the route.
        if (rospy.Time.now() - self._commit_time).to_sec() < self.replan_commit_min_s:
            return
        # (5) Opportunistic triggers: rotation / discovery / periodic. One A*
        #     evaluation consumes every pending trigger (the high-water marks stop
        #     a kept reveal or turn from re-running A* every commit window).
        reason = self._opportunistic_trigger(pose)
        if reason is not None:
            self._opportunistic_replan(remaining, reason)

    def _collision_confirmed(self, remaining, pose):
        """True once the remaining route has collided on ``replan_collision_confirm``
        consecutive frames (debounces single-frame depth noise). The drone's own
        cell is exempted (it sits in its inflation skirt), so a route that merely
        starts inside the skirt is not a false collision."""
        start = remaining[0] if remaining else pose
        collides = self.planner.path_collides(
            self.grid, remaining, passable_start=start)
        if not collides:
            if self._collision_streak:
                rospy.loginfo("astar_planner: route clear again after %d colliding "
                              "frame(s) -- treated as noise, no replan",
                              self._collision_streak)
            self._collision_streak = 0
            return False
        self._collision_streak += 1
        if self._collision_streak < self.replan_collision_confirm:
            rospy.loginfo_throttle(
                1.0, "astar_planner: route collision unconfirmed %d/%d -- waiting "
                "for more depth", self._collision_streak, self.replan_collision_confirm)
            return False
        return self._obstacle_confirmed(remaining, self._collision_streak)

    def _forced_replan(self):
        """Confirmed obstacle on the route: replan and adopt unconditionally (the
        old route is invalid). If A* finds no way around, STOP the drone rather
        than leave the colliding route latched."""
        rospy.logwarn("astar_planner: committed route blocked on %d consecutive "
                      "frame(s) -- replanning", self._collision_streak)
        cand = self._plan_candidate()
        if cand is not None:
            self._publish_event(
                "REPLAN: obstacle on route -> new route",
                "Replanning: an obstacle is on my route -- rerouting around it",
                level="warn")
            self._commit(cand)      # follower stops-and-turns onto the new route
            return
        # Boxed in: no A* route around the obstacle. Report the failure (drives the
        # NavDP fallback) and STOP rather than fly the colliding route.
        self._collision_streak = 0
        self._publish_status(False)
        if self.stop_on_blocked_plan_fail and not self._blocked_hold:
            self._publish_stop()
            self._blocked_hold = True
            self._publish_event(
                "STOP: boxed in - no A* route (NavDP takes over)",
                "No possible route using A* -- I am boxed in. Stopping and "
                "handing over to NavDP",
                level="error")
            rospy.logwarn("astar_planner: no route around the obstacle -- STOP + "
                          "hold (status=False drives the NavDP fallback)")

    def _opportunistic_trigger(self, pose):
        """Return a short reason string when an opportunistic replan should run
        this frame, else None. Checked in order:

          * ROTATION -- the yaw swung >= ~replan_rotation_min_deg since the last
            commit/evaluation (mid-turn or just finished): the camera swept new
            terrain, usually revealed off-corridor to the side the drone faces.
          * DISCOVERY, corridor -- >= ~replan_min_new_cells CHANGED cells (newly
            observed, or flipped free<->occupied) inside the route corridor.
          * DISCOVERY, map-wide -- >= ~replan_min_new_cells_any changed cells
            anywhere (an off-route reveal or new obstacles in mapped free space
            can still change the best route).
          * PERIODIC -- ~replan_period_s since the last A* evaluation: catch-all.
        """
        if self.replan_rotation_min_deg > 0.0:
            turned = abs(math.degrees(normalize_angle(pose.yaw - self._yaw_baseline)))
            if turned >= self.replan_rotation_min_deg:
                return "rotated %.0f deg" % turned
        n_corr = self._changed_in_corridor()
        if n_corr - self._discovery_baseline >= self.replan_min_new_cells:
            return "%d changed corridor cells" % n_corr
        if self.replan_min_new_cells_any > 0:
            n_any = self._changed_total()
            if n_any - self._discovery_baseline_any >= self.replan_min_new_cells_any:
                return "%d changed cells map-wide" % n_any
        if (self.replan_period_s > 0.0
                and (rospy.Time.now() - self._commit_time).to_sec()
                >= self.replan_period_s):
            return "periodic %.1fs check" % self.replan_period_s
        return None

    def _opportunistic_replan(self, remaining, reason):
        """An opportunistic trigger fired (rotation / discovery / periodic): replan,
        and ADOPT the candidate only if it is meaningfully shorter than the
        remaining committed route. Refresh the commit clock either way so
        opportunistic A* runs at most once per commit window (never per frame)."""
        cand = self._plan_candidate()
        # One evaluation regardless of outcome. Do NOT reset _map_at_commit here:
        # a gradual reveal must keep accumulating toward the threshold; only an
        # actual commit resets the map snapshot.
        self._commit_time = rospy.Time.now()
        if cand is None:
            # Opportunistic re-plan failed, but the committed route is still valid
            # (the collision gate passed and already published status=True this
            # frame). Keep flying it; do not emit a spurious failure. Baselines are
            # NOT raised: the trigger stays armed and retries next commit window.
            return
        old_len = polyline_length(remaining)
        new_len = polyline_length(cand.points)
        if new_len <= old_len * (1.0 - self.replan_improve_frac):
            rospy.loginfo("astar_planner: replan adopted (%s): remaining "
                          "%.2fm -> %.2fm", reason, old_len, new_len)
            self._publish_event(
                "REPLAN: shorter route found (%.1fm -> %.1fm)" % (old_len, new_len),
                "The map opened up (%s): taking a shorter route, %.1fm instead "
                "of %.1fm" % (reason, new_len, old_len))
            self._commit(cand)     # resets the snapshot + every trigger baseline
        else:
            # Kept the route: raise every high-water mark so the same reveal/turn
            # does not re-run A* next commit window; a FURTHER one still fires.
            self._raise_trigger_baselines()
            rospy.loginfo_throttle(
                5.0, "astar_planner: %s but candidate not %.0f%% shorter "
                "(%.2fm vs %.2fm) -- keeping committed route",
                reason, 100.0 * self.replan_improve_frac, new_len, old_len)

    def _raise_trigger_baselines(self):
        """High-water marks after a kept (non-adopting) evaluation: the current
        reveals and yaw swing were considered against the committed route -- only
        a FURTHER reveal or turn may trigger again. _map_at_commit is untouched:
        gradual changes keep accumulating; only a real commit resets the snapshot."""
        self._discovery_baseline = self._changed_in_corridor()
        self._discovery_baseline_any = self._changed_total()
        if self.pose is not None:
            self._yaw_baseline = self.pose.yaw

    def _plan_candidate(self):
        """Plan once from the live pose; return the drone-anchored candidate path,
        or None on failure. Status is left to the caller, which knows whether the
        failure is safety-relevant (first plan / boxed in -> status=False) or merely
        an opportunistic re-plan that failed while the committed route is still valid.
        """
        if self.grid is None or self.goal is None or self.pose is None:
            return None
        if not self._warmup_ok():
            return None            # a goal click must NOT bypass the warmup gate
        t0 = rospy.Time.now()
        req = PlanRequest(start=self.pose, goal=self.goal, frame_id=self.frame_id)
        res = self.planner.plan(req, self.grid)
        if not res.ok:
            self.fail_reason = res.message
            rospy.logwarn_throttle(
                5.0, "astar_planner: PLAN FAILED start=(%.2f,%.2f) goal=(%.2f,%.2f) "
                "status=%s reason=%s", self.pose.x, self.pose.y, self.goal.x,
                self.goal.y, res.status.value, res.message)
            return None
        self._last_plan_dt = (rospy.Time.now() - t0).to_sec()
        return self._anchor_to_drone(res.path)

    def _commit(self, cand):
        """Publish ``cand`` as the new committed route and snapshot the map so the
        next discovery diff is measured from here. Resets all per-route state."""
        self.last_points = cand.points
        self._publish(cand.points, getattr(self, "_last_plan_dt", 0.0))
        self._publish_status(True)
        self.has_plan = True
        self._commit_time = rospy.Time.now()
        self._map_at_commit = self.grid.grid.copy()
        self._grid_key = self._grid_key_of(self.grid)
        self._progress_idx = 0
        self._discovery_baseline = 0
        self._discovery_baseline_any = 0
        self._yaw_baseline = self.pose.yaw if self.pose is not None else 0.0
        self._collision_streak = 0
        self._blocked_hold = False
        self.fail_reason = "(success)"
        # A committed route restarts this node's narration too. Without it, the
        # gate suppresses an unchanged line forever: the SECOND "obstacle on the
        # route" reroute of a flight -- and every one after it -- would be
        # silently swallowed as a repeat of the first, which is precisely the
        # event the operator most needs to see.
        self.thinker.forget("plan")

    def _changed_in_corridor(self):
        """Count changed cells (vs the last commit) inside the route corridor.

        Changed = newly observed OR flipped free<->occupied, so new obstacles
        appearing in already-mapped free space count too."""
        radius_cells = int(round(self.replan_corridor_radius_m / self.grid.resolution))
        corridor = corridor_mask(self.last_points, self.grid, radius_cells)
        return count_changed_in_corridor(self._map_at_commit, self.grid, corridor)

    def _changed_total(self):
        """Count changed cells (vs the last commit) anywhere on the grid."""
        return int(changed_mask(self._map_at_commit, self.grid).sum())

    def _reseed_known(self, key):
        """Re-snapshot the map on a new BEV lattice (origin/res/shape change),
        without diffing across the shifted grid. Only the map-diff state resets:
        the commit clock and yaw baseline are lattice-independent, so the rotation
        and periodic triggers keep their cadence even if the lattice keeps moving."""
        self._map_at_commit = self.grid.grid.copy()
        self._grid_key = key
        self._progress_idx = 0
        self._discovery_baseline = 0
        self._discovery_baseline_any = 0
        rospy.logwarn_throttle(
            10.0, "astar_planner: BEV lattice changed %s -- reseeding map snapshot, "
            "skipping discovery this frame", key)

    def _reset_smart_state(self):
        """Clear all smart-replan bookkeeping (used on a fresh goal click)."""
        self._map_at_commit = None
        self._grid_key = None
        self._commit_time = rospy.Time(0)
        self._progress_idx = 0
        self._discovery_baseline = 0
        self._discovery_baseline_any = 0
        self._yaw_baseline = self.pose.yaw if self.pose is not None else 0.0
        self._blocked_hold = False
        self._collision_streak = 0

    @staticmethod
    def _grid_key_of(grid):
        """Lattice identity guarding the cross-frame index diff (world-fixed grid)."""
        return (grid.height, grid.width, round(grid.origin_x, 6),
                round(grid.origin_y, 6), round(grid.resolution, 6))

    def _publish_status(self, ok):
        """Publish one A* plan-attempt outcome (True=route, False=no route)."""
        self.pub_status.publish(Bool(data=bool(ok)))

    def _publish_event(self, text, thought, level="info"):
        """Announce a discrete A* planning event (String) for the viewer HUD.

        The single funnel for every planning event, so it is also the single
        place that narrates one. ``text`` keeps the terse wording the HUD parses;
        ``thought`` is the first-person line the operator reads instead.

        Args:
            text: HUD event string, published verbatim on ~event_topic.
            thought: First-person narration of the same event.
            level: Severity of the narration; "warn" for a reroute the operator
                should notice, "error" for the boxed-in STOP.
        """
        self.pub_event.publish(String(data=text))
        rospy.loginfo("astar_planner event: %s", text)
        self.thinker.say(thought, category="plan", level=level)

    def _publish_stop(self):
        """Command the drone to STOP: a 2-point hold at the current pose collapses
        (via the follower's reanchor) to a single waypoint it is already on, so it
        brakes to DONE. Status is left to the caller (already False)."""
        if self.pose is None:
            return
        hold = (self.pose, self.pose)
        self.last_points = hold
        self._publish(hold, 0.0)

    def _plan_tick(self, _evt):
        """Strict A* cadence: every plan_period_s recompute A* + APF once and
        publish, then freeze until the next tick. Fires regardless of has_plan, so
        each cycle yields a fresh path from the latest map and current pose -- the
        one periodic source of a new trajectory once the first plan exists.

        allow_suppress=True: when the periodic replan reproduces the SAME path
        (typical while the drone is stopped to yaw, so the start anchor has not
        moved) the re-publish is skipped -- see _try_plan."""
        self._try_plan(allow_suppress=True)

    @staticmethod
    def _paths_equal(a, b, eps=1e-3):
        """True if two waypoint tuples are the same length and match within eps (m).

        Used to decide whether a periodic replan actually changed the route. The
        start waypoint is anchored to the live pose, so while the drone moves this
        is False every tick (the route genuinely shifts) and the replan publishes;
        while it is stopped (yaw settle, map wait) the anchor is fixed and an
        identical A* result compares equal, so the redundant re-publish is skipped.
        """
        if not a or len(a) != len(b):
            return False
        return all(abs(p.x - q.x) < eps and abs(p.y - q.y) < eps
                   for p, q in zip(a, b))

    def _try_plan(self, allow_suppress=False):
        """Plan once and (re)publish the raw A* path.

        allow_suppress (set by the periodic / replan_on_bev ticks): if the freshly
        planned path is identical to the one already published, skip the re-publish
        so the downstream corrector and follower are not reset by an unchanged
        route -- re-emitting it resets the follower's waypoint index and re-enters
        its yaw state machine, stuttering the path it is flying. The first plan,
        goal clicks and collision replans leave this False, so an intentional
        new/changed path always reaches the follower.
        """
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
            self.pub_status.publish(Bool(data=False))   # no route this attempt
            rospy.logwarn_throttle(
                5.0, "astar_planner: PLAN FAILED start=(%.2f,%.2f) goal=(%.2f,%.2f) "
                "status=%s reason=%s", self.pose.x, self.pose.y, self.goal.x,
                self.goal.y, res.status.value, res.message)
            return False

        anchored = self._anchor_to_drone(res.path)   # path begins at the drone pose
        # Freeze the trajectory when a periodic replan reproduces the CURRENT path:
        # re-publishing it would only reset the follower (waypoint index + yaw
        # state), never change where the drone goes. has_plan guards the first plan.
        if (allow_suppress and self.has_plan
                and self._paths_equal(anchored.points, self.last_points)):
            self.fail_reason = "(unchanged -- not re-published)"
            self.pub_status.publish(Bool(data=True))    # still a valid route
            rospy.loginfo_throttle(
                10.0, "astar_planner: periodic replan reproduced the current path "
                "-- not re-publishing (trajectory frozen)")
            return True

        # Publish the raw A* path; the path_corrector recentres it downstream.
        # Collision re-checks here run against THIS published path.
        self.last_points = anchored.points
        self._publish(anchored.points, (rospy.Time.now() - t0).to_sec())
        self.pub_status.publish(Bool(data=True))    # a fresh route was found
        self.has_plan = True
        self._collision_streak = 0    # every fresh path starts its debounce clean
        self._pred_collision_streak = 0
        self.fail_reason = "(success)"
        return True

    # ─── Path anchoring ──────────────────────────────────────────
    def _anchor_to_drone(self, path):
        """Make the path originate exactly at the drone's current pose.

        The A* planner trims waypoints within ~start_skip_m of the start and never
        re-inserts the start pose, so the published path begins a step ahead of the
        drone (it appears to "start at the second waypoint"). Prepend the live pose
        as waypoint 0 -- or replace a near-coincident first point with it -- so the
        published path starts exactly at the drone. The downstream corrector pins
        waypoint 0, so the origin stays fixed through correction.
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
        """Publish the raw A* path on ~path_topic and stamp it."""
        m = self._path_msg(points)
        self._last_plan_stamp = m.header.stamp   # freshness ref for predicted check
        self.pub_path.publish(m)
        length = sum(a.distance_to(b) for a, b in zip(points[:-1], points[1:]))
        rospy.loginfo("astar_planner: PATH PUBLISHED %d wp %.2fm plan=%.0fms "
                      "first=(%.2f,%.2f) last=(%.2f,%.2f)", len(points), length,
                      1000.0 * plan_dt_s, points[0].x, points[0].y,
                      points[-1].x, points[-1].y)

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
        L("  path out = %s   (raw A* -> path_corrector)", self.path_topic)
        L("  status out = %s   (Bool: True=route found, False=no route)", self.status_topic)
        if self.smart_replan:
            L("  replan = SMART (event-driven: collision + rotation + discovery "
              "[corridor/map-wide] + periodic)")
            L("    corridor=%.2fm min_new_cells=%d (any=%d) rotation>=%.0fdeg "
              "period=%.1fs", self.replan_corridor_radius_m,
              self.replan_min_new_cells, self.replan_min_new_cells_any,
              self.replan_rotation_min_deg, self.replan_period_s)
            L("    commit_min=%.1fs improve>=%.0f%% collision_confirm=%d "
              "stop_on_fail=%s", self.replan_commit_min_s,
              100.0 * self.replan_improve_frac,
              self.replan_collision_confirm, self.stop_on_blocked_plan_fail)
        else:
            L("  replan = LEGACY periodic %.1fs  (path frozen between ticks)",
              self.plan_period_s)
            L("    mid-cycle: bev=%s collision=%s predicted=%s",
              self.replan_on_bev, self.replan_on_collision,
              self.replan_on_predicted_collision)
        L("  goal init= %s", "(%.2f,%.2f)" % (self.goal.x, self.goal.y) if self.goal else "none")
        L("  conn=%d inflate=%.2fm unknown=%s max_seg=%.2fm los=%s turn_pen=%.2f",
          p.connectivity, p.inflate_radius_m,
          "blocked" if p.unknown_blocked else "free x%.1f" % p.unknown_cost,
          p.waypoint_spacing_m, p.los_smoothing, p.turn_penalty)
        L("  search_margin=%.1fm start_skip=%.2fm snap=%.1fm collision_replan=%s",
          p.search_margin_m, p.start_skip_m, p.goal_snap_radius_m, self.replan_on_collision)
        L("  collision replan confirm=%d frame(s)", self.replan_collision_confirm)
        L("  obstacle confidence gate: conf_in=%s thresh=%.2f ceiling=%s",
          self.conf_topic, self.replan_confirm_conf,
          ("%d frames" % self.replan_collision_confirm_max
           if self.replan_collision_confirm_max else "off")
          if self.replan_confirm_conf > 0.0 else "DISABLED")
        L("  corner_round=%s merge=%.0fdeg max_turn=%.0fdeg chamfer<=%.0fdeg "
          "dist=%.2fm runup=%.2fm", p.corner_round, math.degrees(p.corner_merge_rad),
          math.degrees(p.corner_max_turn_rad), math.degrees(p.corner_chamfer_max_rad),
          p.corner_chamfer_dist_m, p.corner_min_runup_m)
        L("  predicted-collision replan=%s cap=%d topic=%s",
          self.replan_on_predicted_collision, self.max_predicted_replans,
          self.predicted_path_topic)
        L("  path correction runs in path_corrector_node (subscribes %s)",
          self.path_topic)
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
#       ~goal_topic (/waypoint_nav/goal)
#       ~path_topic (/path/waypoints_astar; raw A* -> path_corrector_node, which
#         recentres it and republishes the flown path on /path/waypoints)
#       ~status_topic (/path/astar_status; std_msgs/Bool published once per real
#         planning attempt -- True=route found, False=no route (res.ok False). Not
#         emitted for the not-ready gates (no BEV/goal/pose, warmup). The fallback
#         arbiter counts consecutive False/True to switch to/from NavDP.)
#       ~event_topic (/path/astar_event; std_msgs/String, latched) -- human-readable
#         replan events for the BEV viewer HUD: first route, obstacle reroute, boxed
#         STOP, discovery reroute. Distinct from the per-attempt Bool status.
#       ~frame_id (world) ~goal_x ~goal_y (initial goal; unset = wait for a click)
#       thinking (inherited from thinking.Thinker): ~thinking (true; false silences
#         this node's narration) ~thinking_topic (/nav/thinking; the shared thought
#         log the BEV viewer renders) ~thinking_echo (true; also mirror to rosout).
#         Each ~event_topic event is narrated there in the first person.
#   planner: ~connectivity (8) ~inflate_radius_m (0.4) ~unknown_blocked (false)
#       ~unknown_cost (1.0) ~search_margin_m (3.0) ~turn_penalty (0.3)
#       ~los_smoothing (true) ~waypoint_spacing_m (3.0) ~goal_snap_radius_m (2.0)
#       ~start_skip_m (0.4) ~max_expansions (200000)
#       (When the drone reads INSIDE a wall -- hugging it / a noisy frame paints its
#        cell occupied -- the planner clears the drone's own footprint + inflation
#        skirt so A* plans out; a genuinely walled-in start returns NO_PATH and the
#        node STOPs + hands off to NavDP rather than route through the wall.)
#   corner rounding (gentler turns): ~corner_round (true) ~corner_merge_deg (8)
#       ~corner_max_turn_deg (14) ~corner_chamfer_max_deg (28)
#       ~corner_chamfer_dist_m (0.5) ~corner_min_runup_m (0.6)
#   SMART replanning (~smart_replan, DEFAULT true) -- event-driven off the BEV,
#     built for the slow stop-and-turn platform. The committed route is FROZEN
#     and re-evaluated only on:
#       * COLLISION -- the remaining committed route crosses a confirmed obstacle
#         (~replan_collision_confirm consecutive frames; the drone's own inflated
#         cell is exempted). Safety: bypasses the commit window. If A* finds no way
#         around, the drone STOPS (~stop_on_blocked_plan_fail) and status=False
#         hands off to the NavDP fallback.
#       * DISCOVERY -- a large map CHANGE (newly-observed cells OR free<->occupied
#         flips in already-mapped space, e.g. new obstacles), either ROUTE-RELEVANT
#         (>= ~replan_min_new_cells changed cells inside the route corridor of
#         half-width ~replan_corridor_radius_m) or MAP-WIDE
#         (>= ~replan_min_new_cells_any changed cells anywhere -- a change to the
#         side can still change the best route; higher floor, 0 disables).
#       * ROTATION -- the drone swung its yaw >= ~replan_rotation_min_deg since
#         the last commit/evaluation (mid-turn or just finished): the camera swept
#         new terrain. 0 disables.
#       * PERIODIC -- ~replan_period_s since the last A* evaluation: catch-all
#         re-evaluation even with no other trigger. 0 disables. (NOT the legacy
#         ~plan_period_s strict re-publish timer -- adoption still needs the
#         hysteresis below, so a static map re-evaluates but never re-publishes.)
#     All but COLLISION are opportunistic: they respect the commit window
#     (~replan_commit_min_s), and the new route is ADOPTED only if it is
#     >= ~replan_improve_frac shorter than the remaining committed route
#     (length hysteresis -- what stops the L/R flip-flop and the periodic
#     re-publish oscillation against a moving anchor).
#     Evaluation runs on every BEV frame AND on a 1 Hz quiet-map tick: the BEV
#     publisher skips unchanged maps and mapping freezes during turns, so without
#     the tick a silent BEV would starve the rotation and periodic triggers.
#     Params (defaults): ~smart_replan (true) ~replan_corridor_radius_m (1.0)
#       ~replan_min_new_cells (60) ~replan_min_new_cells_any (240)
#       ~replan_rotation_min_deg (30) ~replan_period_s (7.0)
#       ~replan_commit_min_s (4.0) ~replan_improve_frac (0.15)
#       ~replan_collision_confirm (2)
#       ~stop_on_blocked_plan_fail (true). NavDP is the fallback if A* cannot plan.
#   CONFIDENCE-AWARE obstacle confirmation (a SECOND gate after the frame count;
#     only ever DELAYS a reroute, never bypasses the frame debounce):
#       ~conf_topic (/falcon/bev_2d_conf) -- companion OccupancyGrid from the BEV
#         publisher (int8 0..100 = temporal evidence / t_max).
#       ~replan_confirm_conf (0.75) -- reroute only once the on-route obstacle's
#         confidence reaches this fraction of full evidence; a marginal cell is
#         kept on-route so the drone keeps observing (and can clean) it. Must be
#         above the BEV's latch ratio t_on/t_max (else every latched cell already
#         qualifies and the gate is a no-op). 0 disables the gate (frame count
#         only). Without a co-registered ~conf_topic grid the node also falls back
#         to the frame count.
#       ~replan_collision_confirm_max (30) -- safety ceiling: reroute anyway once a
#         collision has persisted this many frames even if the confidence gate is
#         unmet (bounds flying toward a persistently-flagged, never-confirmed cell).
#         0 = no ceiling (pure confidence gate).
#   LEGACY cadence (~smart_replan:=false): ~plan_period_s (3.0) strict timer +
#     the OPTIONAL mid-cycle hooks below (ALL OFF by default):
#       ~replan_on_collision (false) ~replan_on_bev (false)
#       ~replan_collision_confirm (2; consecutive colliding BEV frames required
#         before a collision replan -- debounces single-frame depth noise)
#       ~replan_on_predicted_collision (false; replan when the follower's predicted
#         stop-and-turn trajectory on ~predicted_path_topic (/path/predicted)
#         collides even if the geometric path does not -- dynamics-aware; legacy)
#       ~max_predicted_replans (3; consecutive cap, reset once prediction is clear)
#   warmup gate: ~min_free_cells_to_plan (80; 0 disables)
#
# PATH CORRECTION moved out of this node. The repulsive-field recentring (and its
# F_rep force-arrow viz) now lives in path_corrector_node.py, which subscribes to
# ~path_topic (/path/waypoints_astar), reshapes the path against the same BEV and
# republishes the flown path on /path/waypoints. Its ~apf_* / ~inflate_radius_m /
# ~publish_forces rosparams are documented in that node.
# ============================================================================
