#!/usr/bin/env python3
"""
astar_planner_node.py -- ROS1 adapter: 2D BEV OccupancyGrid -> smoothed waypoints.

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
  - publishing nav_msgs/Path and logging.

Drop-in replacement for the legacy falcon_adapter ``astar_planner.py``:
identical topics and message types.

  in   ~bev_topic  (OccupancyGrid)  /falcon/bev_2d
  in   ~drone_ns + /gt_pose (Pose)
  in   ~goal_topic (Point)          /waypoint_nav/goal
  out  ~path_topic (Path, latched)  /path/waypoints

See the file footer for the full rosparam list.
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path

from sparx_agency.core.common.types import Pose2D, PlanStatus
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar import (
    WeightedAStarPlanner2D, WeightedAStarParams)

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
            turn_penalty=float(G("~turn_penalty", 0.0)),
            los_smoothing=bool(G("~los_smoothing", True)),
            waypoint_spacing_m=float(G("~waypoint_spacing_m", 3.0)),
            goal_snap_radius_m=float(G("~goal_snap_radius_m", 2.0)),
            start_skip_m=float(G("~start_skip_m", 0.4)),
            max_expansions=int(G("~max_expansions", 200000)),
        )
        self.planner = WeightedAStarPlanner2D(self.params)

        self.replan_on_collision = bool(G("~replan_on_collision", True))
        self.replan_on_bev = bool(G("~replan_on_bev", False))
        self.replan_period_s = float(G("~replan_period_s", 0.0))

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

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.drone_ns + "/gt_pose", Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.goal_topic, Point, self._goal_cb, queue_size=1)

        if self.replan_period_s > 0:
            rospy.Timer(rospy.Duration(self.replan_period_s), lambda _e: self._try_plan())
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
        self.planner.invalidate_cache()
        if not self._try_plan():
            rospy.logwarn("astar_planner: click goal (%.2f, %.2f) accepted but no "
                          "path yet -- reason=%s (will retry on next BEV)",
                          new.x, new.y, self.fail_reason)

    def _bev_cb(self, msg):
        first = self.grid is None
        self.grid = self._decode(msg)
        if first:
            i = msg.info
            rospy.loginfo("astar_planner: first BEV W=%d H=%d res=%.2f origin=(%.1f,%.1f)",
                          i.width, i.height, i.resolution,
                          i.origin.position.x, i.origin.position.y)
        if not self.has_plan:
            self._try_plan()
        elif self.replan_on_bev:
            self._try_plan()
        elif self.replan_on_collision and self.planner.path_collides(self.grid, self.last_points):
            rospy.logwarn("astar_planner: published path now crosses an occupied "
                          "cell -- replanning")
            self.has_plan = False
            self._try_plan()

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

        self.last_points = res.path.points
        self._publish(res.path.points, (rospy.Time.now() - t0).to_sec())
        self.has_plan = True
        self.fail_reason = "(success)"
        return True

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
    def _publish(self, points, plan_dt_s):
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
        L("  path out = %s", self.path_topic)
        L("  goal init= %s", "(%.2f,%.2f)" % (self.goal.x, self.goal.y) if self.goal else "none")
        L("  conn=%d inflate=%.2fm unknown=%s max_seg=%.2fm los=%s turn_pen=%.2f",
          p.connectivity, p.inflate_radius_m,
          "blocked" if p.unknown_blocked else "free x%.1f" % p.unknown_cost,
          p.waypoint_spacing_m, p.los_smoothing, p.turn_penalty)
        L("  search_margin=%.1fm start_skip=%.2fm snap=%.1fm collision_replan=%s",
          p.search_margin_m, p.start_skip_m, p.goal_snap_radius_m, self.replan_on_collision)
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
#       ~goal_topic (/waypoint_nav/goal) ~path_topic (/path/waypoints)
#       ~frame_id (world) ~goal_x ~goal_y (initial goal; unset = wait for a click)
#   planner: ~connectivity (8) ~inflate_radius_m (0.4) ~unknown_blocked (false)
#       ~unknown_cost (1.0) ~search_margin_m (3.0) ~turn_penalty (0.0)
#       ~los_smoothing (true) ~waypoint_spacing_m (3.0) ~goal_snap_radius_m (2.0)
#       ~start_skip_m (0.4) ~max_expansions (200000)
#   replanning: ~replan_on_collision (true) ~replan_on_bev (false) ~replan_period_s (0.0)
#   warmup gate: ~min_free_cells_to_plan (80; 0 disables)
# ============================================================================
