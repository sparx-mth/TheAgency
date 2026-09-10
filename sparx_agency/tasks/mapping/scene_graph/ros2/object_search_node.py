"""object_search_node — find one named object: choose a room, fly, map, repeat.

The loop the method describes, closed end to end:

1. **arc weights.** Every tick (throttled by ``cost_period_s``) the live BEV,
   the scene graph's room centres and the oracle's ranking are assembled into
   an :class:`HppPtInstance` -- a complete, symmetric, metric cost matrix
   between room centres plus a probability per room -- and published on
   ``/object_search/costs`` so the graph a visit order was computed from is
   visible in a recording.
2. **the order.** That instance and the eligible rooms go to a solver. Until
   RPT* exists the built-in stub commits to ONE room, drawn weighted by
   probability; the seam is a plain callable, so RPT* drops in without this
   node changing.
3. **transit.** We fly there ourselves, with the shared weighted A* over a
   fresh grid from the live BEV and the existing trajectory follower. FALCON
   is muted for the whole leg by ``cmd_vel_arbiter_node``, which this node
   drives with one latched Bool.
4. **the room.** On arrival the room is mapped under a budget: the sweep aims
   at frontier clusters INSIDE the room's own mask, which is what makes "does
   not leave the room" structural rather than a hope. It ends when the room's
   frontier count clears, when it stops falling, or when the budget expires.
5. **detection preempts everything.** ``/target_seen`` is latched; on it this
   node stands down permanently and hands the aircraft to the approach.

**FALCON is not asked to map the room, and that is a deliberate deviation
from the method as written.** FALCON's exploration bounds are read once at
construction, its FSM's FINISH state is terminal, and its coverage tour is
built over unknown centres a keep-out box does not touch -- so bounding it to
a room needs a coordinated C++ patch set against a container image that is a
``docker commit`` lineage rather than a reproducible build. Holding the mute
for the whole mission instead costs nothing, keeps FALCON mapping in the
background (so the BEV and the whole scene graph stay alive), and as a side
effect means FALCON never runs out of frontiers and so never reaches FINISH.
``search_backend:=falcon`` releases the mute during SEARCH and lets FALCON
explore for the budget instead -- honest about what it is, which is
unbounded: it may leave the room.

FLIGHT IS OPT-IN AND OFF BY DEFAULT (``fly``). Unarmed, everything still runs
and publishes -- the order, the costs, the route, the sweep goals -- so the
whole pipeline can be watched in RViz while FALCON keeps flying its own
exploration. Armed, the same route also goes to ``follow_path_topic`` and the
arbiter is told to take the aircraft.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.object_search_node \\
        --ros-args -p use_sim_time:=true -p fly:=false
"""
from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Int8, String

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    FOUND, SEARCH, SELECT, TRANSIT, FlyTo, ObjectSearchParams,
    ObjectSearchSupervisor, Release, SearchRoom, StandDown)
from sparx_agency.core.planning.interfaces import PlanRequest
from sparx_agency.core.planning.planners.astar import (WeightedAStarParams,
                                                       WeightedAStarPlanner2D)
from sparx_agency.robots.SJTU.adapters import topics
from sparx_agency.tasks.mapping.scene_graph.ros2.confine_payloads import (
    confine_payload, release_payload)
from sparx_agency.tasks.mapping.scene_graph.ros2.object_search_payloads import (
    centroids_from_scene_graph, costs_payload, facts_from_scene_graph,
    grid_from_bev, in_room_frontier_goals, instance_from_wire,
    room_mask_from_labels, room_options, route_points, search_info_payload)
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import (latched_qos,
                                                             sensor_qos)

HOST_SWEEP = "host_sweep"
"""Map the room ourselves, aiming only at frontier inside its mask.

Confined by construction, and worse at coverage than FALCON by some margin
nobody has measured. Kept as the fallback for a stack without the confinement
patch, and as the A/B arm the campaign compares against.
"""
FALCON_BACKEND = "falcon"
"""Hand the room to FALCON, fenced into it by a leased keep-in box.

The default. FALCON's coverage is the reason this whole stack composes with it,
and a room mapped by our own frontier sweep is a room mapped by code written in
an afternoon. Needs falcon-ros-custom:v2-confine (see
tasks/planning/falcon_sjtu/patches/falcon_room_confine.patch) and
``room_confine:=true`` on the FALCON launch; without both, the fence is simply
never applied and FALCON explores the whole building during the room budget.
"""

FLYING = 1
"""``/simple_drone/state`` value that means airborne."""


class ObjectSearchNode(Node):
    """Runs the select / transit / map loop and owns the route to the room."""

    def __init__(self):
        super().__init__("object_search")

        p = self.declare_parameter
        p("probabilities_topic", "/llm_oracle/probabilities")
        p("scene_graph_topic", "/scene_graph")
        p("room_grid_topic", "/scene_graph/room_labels_grid")
        p("bev_topic", "/falcon/bev_2d")
        p("odom_topic", topics.ODOM)
        p("state_topic", topics.STATE)
        p("target_seen_topic", "/target_seen")
        p("blocked_topic", "/room_search/follower/blocked")
        p("path_topic", "/object_search/planned_path")
        # The follower reads THIS one, so an unarmed node starves it by
        # construction even if it was launched by mistake.
        p("follow_path_topic", "/scene_graph/follow_path")
        p("goal_topic", "/object_search/goal")
        p("costs_topic", "/object_search/costs")
        p("info_topic", "/object_search/info")
        p("active_topic", "/object_search/active")
        p("fly", False)
        p("frame_id", "")
        p("tick_hz", 1.0)
        p("replan_period_s", 3.0)
        p("cost_period_s", 5.0)
        # The instance.
        p("snap_radius_m", 2.0)
        p("cruise_speed_mps", 0.30)
        p("frontier_weight", 0.0)
        # Fold the per-room mapping budget into every arc ENTERING a room.
        # OFF by default, and the default is the important half: HPP-PT as
        # published has NO per-vertex service time and assumes a SYMMETRIC
        # cost matrix, so an unfolded C is exactly the contract a solver
        # written from the paper expects. Folding is still the more faithful
        # model of THIS mission -- the budget spent in a hospital room
        # dominates the flight between rooms, so optimising pure transit
        # optimises the wrong quantity -- and it provably keeps the triangle
        # inequality (c(u,w)+T_w <= c(u,v)+T_v+c(v,w)+T_w for T_v >= 0). What
        # it costs is symmetry, and only on the depot's column: you pay a
        # room's budget on arrival but never pay one to arrive back at the
        # aircraft's own start, which it never returns to. Turn it on once
        # the solver is known to tolerate that.
        p("fold_search_budget", False)
        # The loop.
        p("search_backend", FALCON_BACKEND)
        p("confine_topic", "/scene_graph/confine")
        # Short, and refreshed every tick while the room phase holds. This is
        # the safety property: every failure on the path -- this node dying,
        # the bridge dropping, the shim killed -- must end in NO confinement.
        p("confine_lease_s", 6.0)
        p("confine_margin_m", 0.5)
        p("confine_door_half_m", 0.9)
        p("confine_drone_halo_m", 1.5)
        p("min_prob", 0.01)
        p("arrival_tol_m", 0.6)
        p("plan_grace_s", 5.0)
        p("max_transit_s", 120.0)
        p("blocked_abandon_s", 6.0)
        p("search_grace_s", 8.0)
        p("search_timeout_s", 90.0)
        p("min_frontier_clusters", 0)
        p("frontier_clear_ticks", 3)
        p("frontier_stall_s", 30.0)
        p("seed", -1)
        p("visit_cooldown", True)
        p("visit_cooldown_s", 120.0)
        p("max_attempts", 3)
        p("defer_s", 180.0)
        # The in-room sweep.
        p("sweep_period_s", 6.0)
        p("sweep_min_cluster_cells", 4)
        p("sweep_arrival_tol_m", 0.8)
        # A*'s. Unknown must stay traversable or nothing is ever reachable
        # early, when the BEV is unknown almost everywhere.
        p("inflate_radius_m", 0.4)
        p("unknown_blocked", False)
        p("waypoint_spacing_m", 1.0)
        p("goal_snap_radius_m", 2.0)

        g = lambda name: self.get_parameter(name).value
        self._fly = bool(g("fly"))
        self._frame_param = str(g("frame_id"))
        self._replan_period_s = float(g("replan_period_s"))
        self._cost_period_s = float(g("cost_period_s"))
        self._sweep_period_s = float(g("sweep_period_s"))
        self._sweep_tol_m = float(g("sweep_arrival_tol_m"))
        self._sweep_min_cells = int(g("sweep_min_cluster_cells"))
        self._confine_lease_s = float(g("confine_lease_s"))
        self._confine_margin_m = float(g("confine_margin_m"))
        self._confine_door_half_m = float(g("confine_door_half_m"))
        self._confine_drone_halo_m = float(g("confine_drone_halo_m"))
        self._snap_radius_m = float(g("snap_radius_m"))
        self._cruise_speed = float(g("cruise_speed_mps"))
        self._frontier_weight = float(g("frontier_weight"))
        self._search_timeout_s = float(g("search_timeout_s"))
        self._fold_budget = bool(g("fold_search_budget"))
        tick_hz = max(0.1, float(g("tick_hz")))

        backend = str(g("search_backend"))
        if backend not in (HOST_SWEEP, FALCON_BACKEND):
            raise ValueError(
                "search_backend must be %r or %r, got %r"
                % (HOST_SWEEP, FALCON_BACKEND, backend))
        self._backend = backend

        self._supervisor = ObjectSearchSupervisor(ObjectSearchParams(
            min_prob=float(g("min_prob")),
            seed=int(g("seed")),
            visit_cooldown=bool(g("visit_cooldown")),
            visit_cooldown_s=float(g("visit_cooldown_s")),
            max_attempts=int(g("max_attempts")),
            defer_s=float(g("defer_s")),
            arrival_tol_m=float(g("arrival_tol_m")),
            plan_grace_s=float(g("plan_grace_s")),
            max_transit_s=float(g("max_transit_s")),
            blocked_abandon_s=float(g("blocked_abandon_s")),
            search_grace_s=float(g("search_grace_s")),
            search_timeout_s=self._search_timeout_s,
            min_frontier_clusters=int(g("min_frontier_clusters")),
            frontier_clear_ticks=int(g("frontier_clear_ticks")),
            frontier_stall_s=float(g("frontier_stall_s")),
            tick_hz=tick_hz))
        self._planner = WeightedAStarPlanner2D(WeightedAStarParams(
            inflate_radius_m=float(g("inflate_radius_m")),
            unknown_blocked=bool(g("unknown_blocked")),
            waypoint_spacing_m=float(g("waypoint_spacing_m")),
            goal_snap_radius_m=float(g("goal_snap_radius_m"))))

        self._ranked = []
        self._prob_source = "unknown"
        self._target = "?"
        self._scene_graph = {}
        self._sg_stamp = 0.0
        self._facts = {}
        self._room_grid = None      # (data, height, width) of the label grid
        self._bev = None
        self._bev_geometry = None
        self._pose = None
        self._airborne = False
        self._target_seen = False
        self._blocked_since = None
        self._last_plan_s = None
        self._last_replan_s = -1e9
        self._last_cost_s = -1e9
        self._last_sweep_s = -1e9
        self._instance = None
        self._dropped = []
        self._route = []
        self._sweep_goal = None
        self._active = None
        self._confined_room = None

        latched = latched_qos()
        sensor = sensor_qos()

        self.create_subscription(String, str(g("probabilities_topic")),
                                 self._probabilities_cb, latched)
        self.create_subscription(String, str(g("scene_graph_topic")),
                                 self._scene_graph_cb, latched)
        self.create_subscription(OccupancyGrid, str(g("room_grid_topic")),
                                 self._room_grid_cb, latched)
        self.create_subscription(OccupancyGrid, str(g("bev_topic")),
                                 self._bev_cb, latched)
        self.create_subscription(Odometry, str(g("odom_topic")),
                                 self._odom_cb, sensor)
        self.create_subscription(Int8, str(g("state_topic")),
                                 self._state_cb, 1)
        self.create_subscription(Bool, str(g("target_seen_topic")),
                                 self._target_seen_cb, latched)
        self.create_subscription(Bool, str(g("blocked_topic")),
                                 self._blocked_cb, latched)

        self._pub_path = self.create_publisher(Path, str(g("path_topic")), latched)
        self._pub_follow = self.create_publisher(
            Path, str(g("follow_path_topic")), latched)
        self._pub_goal = self.create_publisher(
            PoseStamped, str(g("goal_topic")), latched)
        self._pub_costs = self.create_publisher(String, str(g("costs_topic")),
                                                latched)
        self._pub_info = self.create_publisher(String, str(g("info_topic")),
                                               latched)
        self._pub_active = self.create_publisher(Bool, str(g("active_topic")),
                                                 latched)
        self._pub_confine = self.create_publisher(
            String, str(g("confine_topic")), latched)
        self._publish_active(False)

        self.create_timer(1.0 / tick_hz, self._tick)
        self.create_timer(10.0, self._heartbeat)
        self.get_logger().info(
            "object_search up: fly=%s backend=%s tick=%.1fHz | budget=%.0fs "
            "grace=%.0fs clear=%dx stall=%.0fs | arrival=%.2fm transit_max=%.0fs "
            "| costs every %.0fs, cruise=%.2fm/s | A* inflate=%.2fm unknown_blocked=%s"
            % (self._fly, self._backend, tick_hz, self._search_timeout_s,
               float(g("search_grace_s")), int(g("frontier_clear_ticks")),
               float(g("frontier_stall_s")), float(g("arrival_tol_m")),
               float(g("max_transit_s")), self._cost_period_s,
               self._cruise_speed, float(g("inflate_radius_m")),
               bool(g("unknown_blocked"))))
        self.get_logger().info(
            "arc weights: %s -- C is %s"
            % ("travel time + %.0fs room budget on entering arcs"
               % self._search_timeout_s if self._fold_budget
               else "pure travel time",
               "asymmetric on the depot column (HPP-PT assumes symmetric)"
               if self._fold_budget else "symmetric and metric"))
        if self._backend == FALCON_BACKEND:
            self.get_logger().warn(
                "search_backend=falcon: FALCON explores UNBOUNDED during the "
                "room budget and may leave the room -- see the module docstring")

    # -- input callbacks --------------------------------------------------
    def _probabilities_cb(self, msg: String) -> None:
        payload = self._decode(msg, "probabilities")
        if payload is None:
            return
        self._ranked = payload.get("rooms") or []
        self._target = str(payload.get("target", "?"))
        self._prob_source = str(payload.get("source", "unknown"))

    def _scene_graph_cb(self, msg: String) -> None:
        payload = self._decode(msg, "scene_graph")
        if payload is None:
            return
        # Never cached across a tick: room pids restart whenever the BEV
        # geometry changes, so a kept copy is not stale, it is about a
        # different set of rooms.
        self._scene_graph = payload
        self._sg_stamp = float(payload.get("stamp", 0.0))
        self._facts = facts_from_scene_graph(payload)

    def _room_grid_cb(self, msg: OccupancyGrid) -> None:
        self._room_grid = (msg.data, int(msg.info.height), int(msg.info.width))

    def _bev_cb(self, msg: OccupancyGrid) -> None:
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

    def _forget_rooms(self) -> None:
        """Drop everything keyed by a room pid, because the pids just restarted.

        The mapper resets its registry, its dwell times and its door discovery
        whenever the BEV geometry changes, so room 3 after the reshape is a
        different room from room 3 before it. Three things here are keyed by
        that id and would otherwise survive it: the LATCHED ranking, which
        would be joined to the new run's centroids the moment they arrive; the
        supervisor's cooldowns and deferrals, which would skip whichever rooms
        happened to inherit an id rather than the rooms that were searched;
        and the instance, whose whole index space is pids. All of them come
        back on their own within a tick or two.
        """
        self._ranked = []
        self._scene_graph = {}
        self._facts = {}
        self._instance = None
        self._supervisor.forget_rooms()
        self.get_logger().warn(
            "BEV geometry changed -- room ids have restarted; dropping the "
            "ranking, the arc weights and the visit memory until they return")

    def _odom_cb(self, msg: Odometry) -> None:
        position, orientation = msg.pose.pose.position, msg.pose.pose.orientation
        self._pose = (float(position.x), float(position.y), float(position.z),
                      yaw_from_quaternion((orientation.x, orientation.y,
                                           orientation.z, orientation.w)))

    def _state_cb(self, msg: Int8) -> None:
        airborne = int(msg.data) == FLYING
        if airborne and not self._airborne:
            self.get_logger().info("aircraft is airborne -- the search may start")
        self._airborne = airborne

    def _target_seen_cb(self, msg: Bool) -> None:
        if bool(msg.data) and not self._target_seen:
            self.get_logger().info("target seen -- object search standing down")
        self._target_seen = bool(msg.data) or self._target_seen

    def _blocked_cb(self, msg: Bool) -> None:
        if bool(msg.data):
            if self._blocked_since is None:
                self._blocked_since = self._now()
        else:
            self._blocked_since = None

    def _decode(self, msg: String, what: str):
        try:
            return json.loads(msg.data)
        except (ValueError, TypeError) as exc:
            self.get_logger().warn("undecodable %s payload: %s" % (what, exc),
                                   throttle_duration_sec=5.0)
            return None

    # -- the tick ---------------------------------------------------------
    def _now(self) -> float:
        """Seconds on the NODE's clock, i.e. sim time under use_sim_time.

        Not a wall clock. This box runs the Gazebo world well below real time,
        so a wall-clock room budget would be a small and varying fraction of
        that many seconds of actual mapping.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self) -> None:
        now = self._now()
        if self._target_seen:
            self._clear_route()
            self._publish_active(False)
            # The approach has to fly to wherever the target is, which is very
            # often not the room we were searching.
            self._release_confine()
            return

        self._refresh_instance(now)
        xy = None if self._pose is None else (self._pose[0], self._pose[1])
        state = self._supervisor.update(
            rooms=room_options(self._ranked, self._centroids()),
            facts=self._facts, xy=xy, now=now, last_plan_s=self._last_plan_s,
            target_seen=self._target_seen, instance=self._instance,
            airborne=self._airborne or not self._fly,
            blocked_since=self._blocked_since)
        action = state.action

        if isinstance(action, FlyTo):
            self.get_logger().info(action.note)
            self._publish_goal(action.xy)
            self._sweep_goal = None
            self._plan(action.xy, now)
        elif isinstance(action, SearchRoom):
            self.get_logger().info(action.note)
            if self._backend == FALCON_BACKEND:
                # Fence FIRST, hand the aircraft over SECOND. The reverse
                # order gives FALCON an unfenced instant in which its coverage
                # tour can commit to a frontier in the next building over, and
                # a committed FALCON trajectory is not cheap to withdraw.
                self._confine(state.room_id)
            else:
                self._last_sweep_s = -1e9
                self._sweep(state.room_id, now, force=True)
        elif isinstance(action, Release):
            self.get_logger().warn(action.note)
            self._sweep_goal = None
            # Lift the fence at once rather than letting it lapse: the
            # aircraft has to fly OUT through a door this fence is sealing,
            # and waiting a whole lease to do it is wasted mission time.
            self._release_confine()
        elif isinstance(action, StandDown):
            self.get_logger().info(action.note)
        elif state.state == TRANSIT and (now - self._last_replan_s) >= self._replan_period_s:
            # The BEV grows under the aircraft: a route planned through what
            # was unknown space is worth re-deriving as the walls appear. Only
            # a SUCCESSFUL replan republishes -- re-deciding every tick is the
            # failure mode that grounded the earlier flights.
            self._plan(state.goal_xy, now)
        elif state.state == SEARCH and self._backend == HOST_SWEEP:
            self._sweep(state.room_id, now)

        # The fence is a LEASE and must be refreshed for as long as the room
        # phase holds -- including on the arrival tick, where the SearchRoom
        # branch above has already claimed it once.
        if state.state == SEARCH and self._backend == FALCON_BACKEND:
            self._confine(state.room_id)
        elif self._confined_room is not None:
            self._release_confine()

        if state.state in (SELECT, FOUND):
            self._clear_route()
        if state.state == SEARCH and self._backend == FALCON_BACKEND:
            # Hand the aircraft back for the budget. The arbiter releases the
            # mute on this flag, and FALCON resumes its own exploration.
            self._clear_route()

        self._publish_active(self._we_are_flying(state))
        self._publish_info(state, now)

    def _we_are_flying(self, state) -> bool:
        """Whether the arbiter should mute FALCON and open our gate."""
        if not self._fly or self._target_seen:
            return False
        if state.state == TRANSIT:
            return bool(self._route)
        if state.state == SEARCH:
            # Under the FALCON backend the aircraft is FALCON's for the whole
            # room phase: we hold the fence, it does the flying.
            return self._backend == HOST_SWEEP and bool(self._route)
        return False

    def _centroids(self):
        """``{pid: (x, y)}``, preferring the instance's SNAPPED room centres.

        A raw centroid is the mean of a room's mask cells and can sit inside
        the room's own wall or inside the planner's inflation skirt; the
        instance already moved every one of those onto a cell the planner
        accepts. Flying to the raw value instead is how a reachable room reads
        as unreachable and is silently never visited.
        """
        centroids = centroids_from_scene_graph(self._scene_graph)
        if self._instance is not None:
            for node in self._instance.nodes:
                if node.pid >= 0:
                    centroids[node.pid] = node.xy
        return centroids

    # -- the arc weights --------------------------------------------------
    def _refresh_instance(self, now: float) -> None:
        """Rebuild the HPP-PT instance, throttled, and publish it."""
        if (now - self._last_cost_s) < self._cost_period_s:
            return
        if self._bev is None or not self._scene_graph:
            return
        self._last_cost_s = now
        try:
            world = grid_from_bev(*self._bev)
        except ValueError as exc:
            self.get_logger().error("malformed BEV grid: %s" % (exc,))
            return
        cost, _lethal, _clear = self._planner.cost_for(world)
        depot = None if self._pose is None else (self._pose[0], self._pose[1])
        try:
            instance, dropped = instance_from_wire(
                world, cost, self._scene_graph, self._ranked, depot_xy=depot,
                snap_radius_m=self._snap_radius_m,
                cruise_speed_mps=self._cruise_speed,
                search_time_s=(self._search_timeout_s if self._fold_budget
                               else 0.0),
                frontier_weight=self._frontier_weight)
        except ValueError as exc:
            self.get_logger().warn("no arc weights this tick: %s" % (exc,),
                                   throttle_duration_sec=10.0)
            return
        if instance is None:
            return
        self._instance = instance
        self._dropped = dropped
        if dropped:
            self.get_logger().warn(
                "withheld %d room(s) from the solver -- unreachable on the "
                "current map: %s" % (len(dropped), dropped),
                throttle_duration_sec=20.0)
        self._pub_costs.publish(String(data=json.dumps(costs_payload(
            instance, self._sg_stamp, now, dropped, self._prob_source))))

    # -- the fence --------------------------------------------------------
    def _confine(self, room_id) -> None:
        """Claim or refresh FALCON's keep-in lease on one room.

        Silently does nothing when the room's mask cannot be resolved -- a pid
        that has just been renumbered. That is the right failure: no request
        means no fence, and FALCON explores unconfined for a few seconds until
        the room set settles, which costs mission time and nothing else. A
        fence built from a stale mask would fence the wrong room.
        """
        if room_id is None or self._room_grid is None or self._bev is None:
            return
        data, height, width = self._room_grid
        if (height, width) != (self._bev[1], self._bev[2]):
            self.get_logger().warn(
                "room grid is %dx%d but the BEV is %dx%d -- different ticks; "
                "not fencing this tick" % (height, width, self._bev[1],
                                           self._bev[2]),
                throttle_duration_sec=10.0)
            return
        try:
            mask = room_mask_from_labels(
                data, height, width,
                self._scene_graph.get("grid_pid_map") or {}, int(room_id))
        except ValueError as exc:
            self.get_logger().error("malformed room label grid: %s" % (exc,))
            return
        if mask is None or not mask.any():
            self.get_logger().warn(
                "R%s has no mask on the current grid -- not fencing"
                % (room_id,), throttle_duration_sec=10.0)
            return
        payload = confine_payload(
            int(room_id), mask,
            resolution=float(self._bev[3]),
            origin_xy=(float(self._bev[4]), float(self._bev[5])),
            doors=self._scene_graph.get("doors") or [],
            drone_xy=None if self._pose is None else (self._pose[0],
                                                      self._pose[1]),
            lease_s=self._confine_lease_s,
            margin_m=self._confine_margin_m,
            door_half_m=self._confine_door_half_m,
            drone_halo_m=self._confine_drone_halo_m)
        if payload is None:
            return
        self._pub_confine.publish(String(data=json.dumps(payload)))
        if self._confined_room != int(room_id):
            self.get_logger().info(
                "fencing FALCON into R%d: %d keep-in box(es), %d door seal(s), "
                "lease %.1fs" % (int(room_id), len(payload["keep_in"]),
                                 len(payload["keep_out"]),
                                 self._confine_lease_s))
        self._confined_room = int(room_id)

    def _release_confine(self) -> None:
        """Lift the fence. Idempotent, and safe to call when none is held."""
        if self._confined_room is None:
            return
        room_id = self._confined_room
        self._confined_room = None
        self._pub_confine.publish(String(data=json.dumps(
            release_payload(room_id))))
        self.get_logger().info("lifted the fence on R%s" % (room_id,))

    # -- the in-room sweep ------------------------------------------------
    def _sweep(self, room_id, now: float, force: bool = False) -> None:
        """Aim at the largest unscanned region INSIDE this room, and re-aim.

        The room mask is the whole safety argument: every goal comes from
        inside it, so the sweep cannot walk out through a door. A room whose
        mask cannot be resolved -- a pid that has just been renumbered -- gets
        no goal at all rather than a goal somewhere else.
        """
        if room_id is None or self._bev is None or self._pose is None:
            return
        arrived = (self._sweep_goal is not None
                   and math.hypot(self._sweep_goal[0] - self._pose[0],
                                  self._sweep_goal[1] - self._pose[1])
                   < self._sweep_tol_m)
        if not force and not arrived and (now - self._last_sweep_s) < self._sweep_period_s:
            return
        self._last_sweep_s = now
        if self._room_grid is None:
            self.get_logger().warn(
                "no /scene_graph/room_labels_grid yet -- cannot bound the "
                "sweep to R%s, holding" % (room_id,),
                throttle_duration_sec=10.0)
            return
        try:
            world = grid_from_bev(*self._bev)
        except ValueError as exc:
            self.get_logger().error("malformed BEV grid: %s" % (exc,))
            return
        data, height, width = self._room_grid
        if (height, width) != (world.height, world.width):
            self.get_logger().warn(
                "room grid is %dx%d but the BEV is %dx%d -- they are from "
                "different ticks; skipping this sweep"
                % (height, width, world.height, world.width),
                throttle_duration_sec=10.0)
            return
        try:
            mask = room_mask_from_labels(
                data, height, width,
                self._scene_graph.get("grid_pid_map") or {}, int(room_id))
        except ValueError as exc:
            self.get_logger().error("malformed room label grid: %s" % (exc,))
            return
        if mask is None or not mask.any():
            self.get_logger().warn(
                "R%s has no mask on the current grid -- it may have been "
                "renumbered; holding station" % (room_id,),
                throttle_duration_sec=10.0)
            return
        cost, _lethal, _clear = self._planner.cost_for(world)
        goals = in_room_frontier_goals(world, cost, mask,
                                       min_cluster_cells=self._sweep_min_cells)
        if not goals:
            self.get_logger().info(
                "R%s has no unscanned region left to aim at" % (room_id,),
                throttle_duration_sec=10.0)
            self._clear_route()
            return
        goal = goals[0]
        if (self._sweep_goal is not None and not arrived
                and math.hypot(goal[0] - self._sweep_goal[0],
                               goal[1] - self._sweep_goal[1]) < 0.5):
            return          # same target as last time; keep the committed route
        self._sweep_goal = goal
        self.get_logger().info(
            "sweeping R%s: %d region(s) left, aiming at (%.2f, %.2f)"
            % (room_id, len(goals), goal[0], goal[1]))
        self._publish_goal(goal)
        self._plan(goal, now)

    # -- planning ---------------------------------------------------------
    def _plan(self, goal_xy, now: float) -> None:
        """Plan to ``goal_xy`` over a fresh grid and publish the route."""
        if goal_xy is None:
            return
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
                "no route to (%.2f, %.2f): %s%s -- plan_grace_s will re-choose"
                % (goal_xy[0], goal_xy[1], result.status.name,
                   (" (%s)" % result.message) if result.message else ""),
                throttle_duration_sec=5.0)
            return
        points = route_points((self._pose[0], self._pose[1]),
                              [(pose.x, pose.y) for pose in result.path.points],
                              self._pose[2])
        if len(points) < 2:
            self.get_logger().warn(
                "route to (%.2f, %.2f) collapsed to one point -- not flyable"
                % (goal_xy[0], goal_xy[1]), throttle_duration_sec=5.0)
            return
        self._route = points
        self._last_plan_s = now
        self._publish_path(points)

    # -- output -----------------------------------------------------------
    def _header(self, message):
        message.header.frame_id = self._bev[6] if self._bev else (
            self._frame_param or "world")
        message.header.stamp = self.get_clock().now().to_msg()
        return message

    def _publish_path(self, points) -> None:
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

    def _clear_route(self) -> None:
        """Stop commanding: an empty Path is the follower's hold, not a stop."""
        if not self._route:
            return
        self._route = []
        empty = self._header(Path())
        self._pub_path.publish(empty)
        if self._fly:
            self._pub_follow.publish(empty)

    def _publish_goal(self, xy) -> None:
        goal = self._header(PoseStamped())
        goal.pose.position.x, goal.pose.position.y = float(xy[0]), float(xy[1])
        goal.pose.position.z = self._pose[2] if self._pose else 0.0
        goal.pose.orientation.w = 1.0
        self._pub_goal.publish(goal)

    def _publish_active(self, active: bool) -> None:
        """Latch who owns the aircraft. Republished only when it changes."""
        if bool(active) == self._active:
            return
        self._active = bool(active)
        self._pub_active.publish(Bool(data=self._active))
        self.get_logger().info(
            "handover: %s has the aircraft"
            % ("object_search" if self._active else "FALCON"))

    def _publish_info(self, state, now: float) -> None:
        payload = search_info_payload(
            stamp=now, state=state, target=self._target, fly=self._fly,
            planned=bool(self._route), route_length=len(self._route),
            note=getattr(state.action, "note", ""),
            stats=self._supervisor.stats,
            room_facts=(None if state.room_id is None
                        else self._facts.get(int(state.room_id))),
            backend=self._backend)
        self._pub_info.publish(String(data=json.dumps(payload)))

    def _heartbeat(self) -> None:
        stats = self._supervisor.stats
        inst = self._instance
        self.get_logger().info(
            "hb state=%s room=%s ranking=%d rooms=%d bev=%s odom=%s air=%s "
            "route=%d fly=%s | C=%s build=%s dropped=%d | sel=%d arr=%d "
            "mapped=%d spent=%d stall=%d planfail=%d ttimeout=%d blocked=%d "
            "done=%d"
            % (self._supervisor.state, self._supervisor.room_id,
               len(self._ranked), len(self._facts), self._bev is not None,
               self._pose is not None, self._airborne, len(self._route),
               self._fly,
               "none" if inst is None else "%dx%d" % (inst.n, inst.n),
               "-" if inst is None else "%.0fms" % inst.build_ms,
               len(self._dropped),
               stats["selections"], stats["arrivals"], stats["mapped"],
               stats["budget_spent"], stats["stalls"], stats["plan_fails"],
               stats["transit_timeouts"], stats["blocked"],
               sum(1 for _, v, _ in self._supervisor.history
                   if v in ("mapped", "budget_spent", "stalled"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectSearchNode()
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
