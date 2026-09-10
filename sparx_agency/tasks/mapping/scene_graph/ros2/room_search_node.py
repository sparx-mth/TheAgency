"""room_search_node — close the loop: the room ranking chooses where to fly.

The half of the original sjtu_project stack that was never ported. Everything
else in this task *observes*: the mapper segments rooms, the classifier names
them, the oracle ranks them by how likely the target is inside, and the viz
draws all of it. Nothing acted on any of it, so the ranking was a picture. This
node makes it the goal.

One tick, four steps:

1. join the oracle's ``/llm_oracle/probabilities`` ranking to the
   ``/scene_graph`` centroids (same pids, two topics);
2. hand them to the ROS-free
   :class:`~sparx_agency.core.planning.exploration.room_search_policy.RoomSearchPolicy`,
   which draws one room, holds it until arrival or a deadline, then dwells in
   it so something else can search it;
3. when a room is drawn, plan to its centroid with the shared weighted A* over
   a FRESH ``OccupancyGrid2D`` built from the live BEV;
4. publish the goal, the route and an operator payload.

FLIGHT IS OPT-IN AND OFF BY DEFAULT (``fly``, default False). Unarmed, this
node is still the whole feature: RViz and the dashboard draw the chosen room
and the route to it while FALCON keeps flying its own exploration, which is
exactly the picture the user remembers wanting. Armed, it publishes the SAME
route a second time on ``follow_path_topic``.

**It does not fly the aircraft itself, on purpose.** A second SJTU follower
would be this repo's most-broken rule made flesh -- there is already a proven,
policy-agnostic one at
``tasks/planning/sjtu_internvla_n1/ros2/trajectory_follower_node.py`` whose own
docstring says it "would fly a NavDP path, an A* path or a hand-drawn one
identically", and which carries the altitude capture, the odom timeout, the
capsize guard and the airframe clamp that a fresh 80-line follower would not.
So flight is delegated: point that node's ``topics.trajectory`` at
``follow_path_topic`` and its ``topics.cmd_vel`` at ``/simple_drone/cmd_vel_raw``
(``config/room_search_follower.yaml`` beside this task does both) and it flies
this route.

THE ARBITRATION IS NOT OPTIONAL AND IS NOT DONE HERE. FALCON's
``bspline_follower`` publishes a Twist EVERY tick at 50 Hz in every state, and
so does that follower at 20 Hz -- two continuous writers on one ``cmd_vel`` is
last-writer-wins, not a handover. What this node contributes to the handover is
the *fact*: a latched Bool on ``active_topic`` that is true only while a
planned route is being pursued with flight armed, and its inverse on
``falcon_active_topic``. Two of FALCON's own ``cmd_vel_gate_node`` instances
driven by that pair are a real arbiter, because a closed gate publishes
nothing at all. The README and this task's report say what to wire.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.room_search_node \\
        --ros-args -p use_sim_time:=true -p fly:=false
"""
from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.exploration.room_search_policy import (
    PURSUING, PublishGoal, ReSample, RoomSearchParams, RoomSearchPolicy)
from sparx_agency.core.planning.interfaces import PlanRequest
from sparx_agency.core.planning.planners.astar import (WeightedAStarParams,
                                                       WeightedAStarPlanner2D)
from sparx_agency.robots.SJTU.adapters import topics
from sparx_agency.tasks.mapping.scene_graph.ros2.room_search_payloads import (
    centroids_from_scene_graph, grid_from_bev, room_options, route_points,
    search_info_payload)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import (latched_qos,
                                                            sensor_qos)


class RoomSearchNode(Node):
    """Draw a room from the ranking, plan to it, and say so on the wire."""

    def __init__(self):
        super().__init__("room_search")

        p = self.declare_parameter
        p("probabilities_topic", "/llm_oracle/probabilities")
        p("scene_graph_topic", "/scene_graph")
        p("bev_topic", "/falcon/bev_2d")
        p("odom_topic", topics.ODOM)
        p("target_seen_topic", "/target_seen")
        p("path_topic", "/scene_graph/planned_path")
        # Armed only. The follower reads THIS one, so an unarmed node starves
        # it by construction even if it was launched by mistake.
        p("follow_path_topic", "/scene_graph/follow_path")
        p("goal_topic", "/scene_graph/goal")
        p("info_topic", "/room_search/info")
        p("active_topic", "/room_search/active")
        p("falcon_active_topic", "/room_search/falcon_active")
        p("fly", False)
        p("frame_id", "")           # empty = adopt the BEV's own frame
        p("tick_hz", 1.0)
        p("replan_period_s", 3.0)
        # The policy's knobs, flown values (see RoomSearchParams).
        p("min_prob", 0.01)
        p("arrival_tol_m", 0.6)
        p("plan_grace_s", 5.0)
        p("max_pursue_s", 60.0)
        p("dwell_after_arrival_s", 15.0)
        p("seed", -1)
        p("visit_cooldown", True)
        p("visit_cooldown_s", 120.0)
        # A*'s. The standoff is the airframe's 0.63 m width plus margin; the
        # BEV is unknown almost everywhere early on, so unknown must stay
        # traversable or nothing is ever reachable.
        p("inflate_radius_m", 0.4)
        p("unknown_blocked", False)
        p("waypoint_spacing_m", 1.0)
        p("goal_snap_radius_m", 2.0)

        g = lambda name: self.get_parameter(name).value
        self._fly = bool(g("fly"))
        self._frame_param = str(g("frame_id"))
        self._replan_period_s = float(g("replan_period_s"))
        tick_hz = max(0.1, float(g("tick_hz")))

        self._policy = RoomSearchPolicy(RoomSearchParams(
            min_prob=float(g("min_prob")),
            arrival_tol_m=float(g("arrival_tol_m")),
            plan_grace_s=float(g("plan_grace_s")),
            max_pursue_s=float(g("max_pursue_s")),
            dwell_after_arrival_s=float(g("dwell_after_arrival_s")),
            tick_hz=tick_hz, seed=int(g("seed")),
            visit_cooldown=bool(g("visit_cooldown")),
            visit_cooldown_s=float(g("visit_cooldown_s"))))
        self._planner = WeightedAStarPlanner2D(WeightedAStarParams(
            inflate_radius_m=float(g("inflate_radius_m")),
            unknown_blocked=bool(g("unknown_blocked")),
            waypoint_spacing_m=float(g("waypoint_spacing_m")),
            goal_snap_radius_m=float(g("goal_snap_radius_m"))))

        self._ranked = []          # the oracle's rooms list, verbatim
        self._target = "?"
        self._centroids = {}       # pid -> (x, y), from /scene_graph
        self._bev = None           # decode arguments of the latest BEV
        self._bev_geometry = None  # the shape/origin those arguments describe
        self._pose = None          # (x, y, z, yaw)
        self._target_seen = False
        self._last_plan_s = None   # the policy's plan-grace fact
        self._last_replan_s = -1e9
        self._route = []           # the route currently published
        self._active = None        # what was last published on active_topic
        self._room_id = None       # the room in force, for the heartbeat

        latched = latched_qos()
        sensor = sensor_qos()

        self.create_subscription(String, str(g("probabilities_topic")),
                                 self._probabilities_cb, latched)
        self.create_subscription(String, str(g("scene_graph_topic")),
                                 self._scene_graph_cb, latched)
        self.create_subscription(OccupancyGrid, str(g("bev_topic")),
                                 self._bev_cb, latched)
        self.create_subscription(Odometry, str(g("odom_topic")),
                                 self._odom_cb, sensor)
        self.create_subscription(Bool, str(g("target_seen_topic")),
                                 self._target_seen_cb, latched)

        self._pub_path = self.create_publisher(Path, str(g("path_topic")), latched)
        self._pub_follow = self.create_publisher(
            Path, str(g("follow_path_topic")), latched)
        self._pub_goal = self.create_publisher(
            PoseStamped, str(g("goal_topic")), latched)
        self._pub_info = self.create_publisher(String, str(g("info_topic")), latched)
        self._pub_active = self.create_publisher(Bool, str(g("active_topic")), latched)
        self._pub_falcon = self.create_publisher(
            Bool, str(g("falcon_active_topic")), latched)
        self._publish_handover(False)

        self.create_timer(1.0 / tick_hz, self._tick)
        self.create_timer(10.0, self._heartbeat)
        self.get_logger().info(
            "room_search up: fly=%s  ranking=%s  route=%s%s  tick=%.1f Hz | "
            "min_prob=%.2f arrival=%.2fm grace=%.0fs pursue=%.0fs dwell=%.0fs "
            "cooldown=%s | A* inflate=%.2fm unknown_blocked=%s"
            % (self._fly, g("probabilities_topic"), g("path_topic"),
               (" + " + str(g("follow_path_topic"))) if self._fly else "",
               tick_hz, float(g("min_prob")), float(g("arrival_tol_m")),
               float(g("plan_grace_s")), float(g("max_pursue_s")),
               float(g("dwell_after_arrival_s")),
               ("%.0fs" % float(g("visit_cooldown_s")))
               if bool(g("visit_cooldown")) else "off",
               float(g("inflate_radius_m")), bool(g("unknown_blocked"))))

    # ── input callbacks ──────────────────────────────────────────────
    def _probabilities_cb(self, msg: String):
        payload = self._decode(msg, "probabilities")
        if payload is None:
            return
        self._ranked = payload.get("rooms") or []
        self._target = str(payload.get("target", "?"))

    def _scene_graph_cb(self, msg: String):
        payload = self._decode(msg, "scene_graph")
        if payload is not None:
            # Never cached across a tick: room pids restart whenever the BEV
            # geometry changes, so yesterday's centroid map is not stale, it is
            # about a different set of rooms.
            self._centroids = centroids_from_scene_graph(payload)

    def _bev_cb(self, msg: OccupancyGrid):
        geometry = (int(msg.info.height), int(msg.info.width),
                    round(float(msg.info.resolution), 6),
                    round(float(msg.info.origin.position.x), 6),
                    round(float(msg.info.origin.position.y), 6))
        if self._bev_geometry is not None and geometry != self._bev_geometry:
            self._forget_rooms()
        self._bev_geometry = geometry
        self._bev = (msg.data, int(msg.info.height), int(msg.info.width),
                     float(msg.info.resolution),
                     float(msg.info.origin.position.x),
                     float(msg.info.origin.position.y),
                     self._frame_param or msg.header.frame_id or "world")

    def _forget_rooms(self):
        """Drop everything keyed by a room pid, because the pids just restarted.

        The mapper resets its registry, its dwell times and its door discovery
        whenever the BEV geometry changes, so room 3 after the reshape is a
        different room from room 3 before it. Two things here are keyed by that
        id and would otherwise survive it: the ranking, which is LATCHED and so
        would be joined to the new run's centroids the moment they arrive, and
        the policy's visit cooldowns, which would skip whichever rooms happened
        to inherit a cooling id rather than the rooms that were searched. Both
        come back on their own -- the scene graph within a tick, the ranking on
        the oracle's next publish.
        """
        self._ranked = []
        self._centroids = {}
        self._policy.forget_visits()
        self.get_logger().warn(
            "BEV geometry changed — room ids have restarted; dropping the "
            "ranking and the visit cooldowns until the oracle republishes")

    def _odom_cb(self, msg: Odometry):
        position, orientation = msg.pose.pose.position, msg.pose.pose.orientation
        self._pose = (float(position.x), float(position.y), float(position.z),
                      yaw_from_quaternion((orientation.x, orientation.y,
                                           orientation.z, orientation.w)))

    def _target_seen_cb(self, msg: Bool):
        if bool(msg.data) and not self._target_seen:
            self.get_logger().info("target seen — room search standing down")
        self._target_seen = bool(msg.data)

    def _decode(self, msg: String, what: str):
        """Parse one JSON payload, or warn and return None."""
        try:
            return json.loads(msg.data)
        except (ValueError, TypeError) as exc:
            self.get_logger().warn("undecodable %s payload: %s" % (what, exc),
                                   throttle_duration_sec=5.0)
            return None

    # ── the tick ─────────────────────────────────────────────────────
    def _now(self):
        """Seconds on the NODE's clock, i.e. sim time under use_sim_time.

        Not a wall clock. Every timer in this node already fires on this
        clock, the mapper's dwell accounting is measured in it, and the
        dashboard prints it -- and this box runs the Gazebo world well below
        real time, so a wall-clock ``max_pursue_s`` would be a small and
        varying fraction of that many seconds of actual flying. Mixing the two
        gives up on a room after a distance nobody chose.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self._now()
        if self._target_seen:
            self._clear_route()
            self._publish_handover(False)
            return
        xy = None if self._pose is None else (self._pose[0], self._pose[1])
        state = self._policy.update(
            room_options(self._ranked, self._centroids), xy, now,
            self._last_plan_s)
        action = state.action

        if isinstance(action, PublishGoal):
            self.get_logger().info(action.note)
            self._publish_goal(action.xy)
            self._plan(action.xy, now)
        elif isinstance(action, ReSample):
            self.get_logger().warn(action.note)
        elif state.state == PURSUING and (now - self._last_replan_s) >= self._replan_period_s:
            # The BEV grows under the aircraft: a route planned through what
            # was unknown space is worth re-deriving as the walls appear. Only
            # a SUCCESSFUL replan republishes -- a failed one leaves the
            # committed route in force, because re-deciding every tick is the
            # failure mode that grounded the NavDP flights.
            self._plan(state.goal_xy, now)

        if state.state != PURSUING:
            # Dwelling and idling do not command the aircraft. Withdraw the
            # route as well as the handover flag: the flag closes the arbiter,
            # and an empty Path makes a follower reading it directly hold
            # station rather than fly on to a goal already reached.
            self._clear_route()

        self._room_id = state.room_id
        self._publish_handover(
            self._fly and state.state == PURSUING and bool(self._route))
        self._publish_info(state, now)

    # ── planning ─────────────────────────────────────────────────────
    def _plan(self, goal_xy, now):
        """Plan to ``goal_xy`` over a fresh grid and publish the route."""
        self._last_replan_s = now
        if self._bev is None or self._pose is None:
            self.get_logger().warn(
                "cannot plan: %s" % ("no /falcon/bev_2d yet" if self._bev is None
                                     else "no odometry yet"),
                throttle_duration_sec=5.0)
            return
        try:
            # A NEW grid object per plan: the planner caches its cost field on
            # grid IDENTITY, so a reused object silently flies a stale map.
            world = grid_from_bev(*self._bev)
        except ValueError as exc:
            self.get_logger().error("malformed BEV grid: %s" % (exc,))
            return
        request = PlanRequest(
            start=Pose2D(self._pose[0], self._pose[1], self._pose[3]),
            goal=Pose2D(float(goal_xy[0]), float(goal_xy[1]), 0.0),
            frame_id=world.params.frame_id)
        result = self._planner.plan(request, world=world)
        if not result.ok:
            self.get_logger().warn(
                "no route to (%.2f, %.2f): %s%s — plan_grace_s will re-draw"
                % (goal_xy[0], goal_xy[1], result.status.name,
                   (" (%s)" % result.message) if result.message else ""),
                throttle_duration_sec=5.0)
            return
        points = route_points((self._pose[0], self._pose[1]),
                              [(pose.x, pose.y) for pose in result.path.points],
                              self._pose[2])
        if len(points) < 2:
            self.get_logger().warn(
                "route to (%.2f, %.2f) collapsed to one point — not flyable"
                % (goal_xy[0], goal_xy[1]), throttle_duration_sec=5.0)
            return
        self._route = points
        self._last_plan_s = now
        self._publish_path(points)

    # ── output ───────────────────────────────────────────────────────
    def _header(self, message):
        message.header.frame_id = self._bev[6] if self._bev else (
            self._frame_param or "world")
        message.header.stamp = self.get_clock().now().to_msg()
        return message

    def _publish_path(self, points):
        path = self._header(Path())
        for x, y, z in points:
            pose = self._header(PoseStamped())
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._pub_path.publish(path)
        if self._fly:
            self._pub_follow.publish(path)

    def _clear_route(self):
        """Stop commanding: an empty Path is the follower's hold, not a stop."""
        if not self._route:
            return
        self._route = []
        empty = self._header(Path())
        self._pub_path.publish(empty)
        if self._fly:
            self._pub_follow.publish(empty)

    def _publish_goal(self, xy):
        goal = self._header(PoseStamped())
        goal.pose.position.x, goal.pose.position.y = float(xy[0]), float(xy[1])
        goal.pose.position.z = self._pose[2] if self._pose else 0.0
        goal.pose.orientation.w = 1.0
        self._pub_goal.publish(goal)

    def _publish_handover(self, active):
        """Latch who owns the aircraft. Republished only when it changes."""
        if active == self._active:
            return
        self._active = bool(active)
        self._pub_active.publish(Bool(data=self._active))
        self._pub_falcon.publish(Bool(data=not self._active))
        self.get_logger().info(
            "handover: %s has the aircraft"
            % ("room_search" if self._active else "FALCON"))

    def _publish_info(self, state, now):
        payload = search_info_payload(
            stamp=now, state=state, target=self._target, fly=self._fly,
            planned=bool(self._route), route_length=len(self._route),
            note=getattr(state.action, "note", ""), stats=self._policy.stats)
        self._pub_info.publish(String(data=json.dumps(payload)))

    def _heartbeat(self):
        stats = self._policy.stats
        self.get_logger().info(
            "hb state=%s room=%s ranking=%d centroids=%d bev=%s odom=%s "
            "route=%d fly=%s | samples=%d arrivals=%d plan_fails=%d "
            "timeouts=%d dwells=%d"
            % (self._policy.state, self._room_id, len(self._ranked),
               len(self._centroids), self._bev is not None,
               self._pose is not None, len(self._route), self._fly,
               stats["samples"], stats["arrivals"], stats["plan_fails"],
               stats["timeouts"], stats["dwell_completes"]))


def main(args=None):
    rclpy.init(args=args)
    node = RoomSearchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
