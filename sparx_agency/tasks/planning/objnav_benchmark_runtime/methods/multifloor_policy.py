"""Building-level scheduling around the unchanged floor-local ZSON search.

Portals are depth proposals until traversed. RPT* orders reachable proposals,
using the existing observed travel-cost builder. Actual traversal is closed-loop
support-surface navigation; neither floor destinations nor GT goals are inputs.
"""
from __future__ import annotations

import math

from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.room_search_policy import RoomCandidate
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_terrain import StairTerrain


class MultiFloorSearch:
    def __init__(self, policy):
        self.policy, self.params = policy, policy.settings.multifloor
        s = policy.settings
        self.terrain = StairTerrain(self.params, s.map_resolution_m, s.body_radius_m, s.body_height_m, s.depth_stride)
        policy.mapping.atlas.plateau_observer = self.terrain.plateau_area
        self.portals = []
        self.active = None
        self.entered_step = 0
        self.floor_id = 0
        self.events = []
        self.path = []
        self.phase = "floor_search"
        self.scans = self.failures = 0
        self._traversing = False
        self._trace = []
        self._last_progress_step = 0
        self._last_pose = None
        self._looked = set()

    @property
    def traversing(self):
        return self._traversing or self.policy.mapping.atlas.in_transition

    def observe(self, obs):
        atlas = self.policy.mapping.atlas
        if atlas.active_id != self.floor_id:
            old = self.floor_id
            self.floor_id, self.entered_step = atlas.active_id, obs.step
            self.events.append({"action": obs.step, "event": "floor_arrival", "source": old, "destination": self.floor_id})
            if self.active is not None:
                self.active["destination"] = self.floor_id
                self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions
            self.active = None
            self._traversing = False
            self.path = []
            self.phase = "floor_search"
            self.failures = self.scans = 0
        floor_world = self.policy.mapping.worlds.get(self.floor_id) if not atlas.in_transition else None
        self.terrain.update(obs, floor_world, atlas.elevation_m if atlas.floors else None)
        if not self.traversing:
            if not atlas.floors or abs(obs.pose.z - atlas.elevation_m) <= self.params.stable_height_m:
                self._discover(obs)
        else:
            xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
            if not self._trace or math.dist(xyz, self._trace[-1]) > 0.05:
                self._trace.append(xyz)
            if self._last_pose is None or math.dist(xyz, self._last_pose) > 0.06:
                self._last_progress_step = obs.step
                self._last_pose = xyz
        if atlas.in_transition and self.active is None:
            direction = 1 if obs.pose.z > atlas.elevation_m else -1
            self.active = self._portal(obs, direction, [(obs.pose.x, obs.pose.y, atlas.elevation_m)])
            self._start(obs)

    def _portal(self, obs, direction, path):
        portal = {"id": len(self.portals), "floor_id": self.floor_id,
                  "entry": list(path[0]), "direction": direction, "path": list(path),
                  "destination": None, "cooldown_until": 0, "observations": 1,
                  "last_step": obs.step}
        self.portals.append(portal)
        return portal

    def _discover(self, obs):
        for candidate in self.terrain.candidates:
            path = candidate["path"]
            # Last near-level tread is the observed portal approach, not an
            # inaccessible point halfway up a stair projected into a room.
            near = [i for i, xyz in enumerate(path) if abs(xyz[2] - obs.pose.z) < 0.20]
            entry_index = max(0, (near[-1] if near else 0) - 2)
            path = path[entry_index:]
            match = next((p for p in self.portals if p["floor_id"] == self.floor_id
                          and p["direction"] == candidate["direction"]
                          and math.dist(p["entry"][:2], path[0][:2]) < 1.25), None)
            if match is None:
                self._portal(obs, candidate["direction"], path)
            elif match["last_step"] != obs.step:
                match["observations"] += 1
                match["last_step"] = obs.step
                if self.active is not match:
                    match["path"] = path
        # Completed edges supply reversible routes, never an inferred teleport.
        for edge in self.policy.mapping.atlas.connections:
            if self.floor_id not in (edge.source, edge.destination):
                continue
            destination, path = edge.oriented(self.floor_id)
            if not any(p.get("edge_id") == edge.id and p["floor_id"] == self.floor_id for p in self.portals):
                portal = self._portal(obs, 1 if path[-1][2] > path[0][2] else -1, path)
                portal.update(destination=destination, edge_id=edge.id, observations=2,
                              cooldown_until=obs.step + self.params.floor_search_actions)

    def plan(self, obs, world, exhausted=False):
        if self.traversing:
            return self._traverse(obs)
        if self.active is None and (exhausted or obs.step - self.entered_step >= self.params.floor_search_actions):
            self._select(obs, world)
        if self.active is not None:
            entry = tuple(self.active["entry"][:2])
            if math.dist((obs.pose.x, obs.pose.y), entry) <= 0.65:
                self._start(obs)
                return self._traverse(obs)
            self.phase = "floor_transit"
            command = self.policy._navigate(obs, world, entry, "portal/%d" % self.active["id"])
            if command is not None:
                return command
            self._abandon(obs, "portal approach unavailable")
        # A single downward inspection at each exhausted location reveals
        # descending treads below the horizon, only when LOOK is legal.
        key = (self.floor_id, round(obs.pose.x), round(obs.pose.y), round(obs.pose.yaw, 1), round(obs.pose.camera_pitch, 1))
        if exhausted and obs.camera and self.policy.episode.action_spec.allows(DiscreteAction.LOOK_DOWN) and key not in self._looked:
            self._looked.add(key)
            return NavigationCommand.hold(camera_pitch=math.radians(60), info={"kind": "stair_inspection"})
        self.phase = "floor_search"
        return None

    def _select(self, obs, world):
        p = self.policy
        eligible = [q for q in self.portals if q["floor_id"] == self.floor_id
                    and q["observations"] >= 2 and obs.step >= q["cooldown_until"]]
        if not eligible:
            return
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        goals = {q["id"]: tuple(q["entry"][:2]) for q in eligible}
        states = p.floors.save()
        probabilities = {}
        for q in eligible:
            state = states.get(q["destination"])
            # Unknown-floor mass is kept explicit; local room probabilities
            # are never renormalized to assert the building was fully observed.
            probabilities[q["id"]] = 0.30 if state is None else max(0.02, min(0.30, sum(state["graph"].probs.values()) / (1 + state["_floor_time"] / 60)))
        instance, _ = build_instance(world, cost, goals, probabilities, depot_xy=(obs.pose.x, obs.pose.y),
                                     cruise_speed_mps=p.episode.action_spec.forward_step_m / p.settings.action_time_s)
        if instance is None:
            return
        candidates = [RoomCandidate(room_id=q["id"], label="observed stair portal", prob=probabilities[q["id"]],
                                    prob_renorm=probabilities[q["id"]], xy=goals[q["id"]]) for q in eligible]
        # Unreachable portals must not enter the room solver's fallback draw.
        allowed = set(instance.index_to_pid)
        order = p.solver([q for q in candidates if q.room_id in allowed], instance)
        if order:
            self.active = next(q for q in eligible if q["id"] == order[0])
            p.route_memory.clear("RPT selected floor portal")
            self.events.append({"action": obs.step, "event": "portal_selected", "portal_id": self.active["id"],
                                "solver_source": p.solver.last.source, "order": list(order)})

    def _start(self, obs):
        self._traversing = True
        self.phase = "stairs"
        self.started_step = obs.step
        self._last_progress_step = obs.step
        self._last_pose = (obs.pose.x, obs.pose.y, obs.pose.z)
        self.source_height = self.policy.mapping.atlas.elevation_m
        self.path = []
        self._trace = [(obs.pose.x, obs.pose.y, obs.pose.z)]
        self._retreat_path = None
        self._surface_goal = None
        self.failures = self.scans = 0
        self.policy.route_memory.clear("stair traversal")
        h = self.policy.hierarchy
        if h is not None:
            if h.regions.mask is not None:
                h.routes.suspend(h.regions.mask, h.current, self.policy, h.machine.actions)
            h.machine.end("floor_departure")
            h.machine.transition("stairs", "observed stair traversal; global allowance only")

    def _traverse(self, obs):
        if self.active is None:
            return NavigationCommand.hold(info={"reason": "unsettled floor; inspect support"})
        p = self.policy
        timed_out = obs.step - self.started_step >= self.params.transition_actions
        stalled = obs.step - self._last_progress_step >= 18
        if self._retreat_path is None and (self.failures >= self.params.max_failures or timed_out or stalled):
            self._retreat_path = list(reversed(self._trace))
        if self._retreat_path is not None:
            if not p.mapping.atlas.in_transition:
                self._abandon(obs, "bounded stair attempt ended")
                return NavigationCommand.hold(camera_pitch=self._pitch(0), info={"reason": "stair proposal rejected"})
            self.phase = "stair_retreat"
            xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
            index = min(range(len(self._retreat_path)), key=lambda i: math.dist(self._retreat_path[i], xyz))
            path = self._retreat_path[index:]
        else:
            self.phase = "stairs"
            path = self._surface_route(obs)
        path = self.terrain.steering_path(obs.pose, path, p.episode.action_spec)
        if path:
            # A short horizon follows bends and treads without committing to a
            # projected end point directly above/below the current position.
            length, end = 0.0, len(path) - 1
            for i, (a, b) in enumerate(zip(path, path[1:]), 1):
                length += math.dist(a[:2], b[:2])
                if length >= 1.25:
                    end = i
                    break
            self.path = path[:end + 1]
            p._route, p._goal = self.path, tuple(self.path[-1][:2])
            p.last_world = self.terrain.world
            self.scans = 0
            pitch = 30 if self.active["direction"] > 0 else 60
            return NavigationCommand.follow(self.path, camera_pitch=self._pitch(pitch), info={"kind": self.phase, "portal": self.active["id"]})
        self.scans += 1
        if self.scans >= 12:
            self.failures += 1
            self.scans = 0
        pitch = 30 if self.active["direction"] > 0 and self.scans < 6 else 60
        return NavigationCommand.hold(camera_pitch=self._pitch(pitch), final_yaw=obs.pose.yaw + p.episode.action_spec.turn_angle_rad,
                                      info={"kind": self.phase, "reason": "acquire connected tread support"})

    def _pitch(self, degrees):
        return math.radians(degrees) if self.policy.episode.action_spec.allows(DiscreteAction.LOOK_DOWN) else None

    def _surface_route(self, obs):
        goal = self._surface_goal
        terrain = self.terrain
        if goal is not None:
            arrived = math.dist((obs.pose.x, obs.pose.y), goal[:2]) <= 0.35 and abs(obs.pose.z - goal[2]) <= 0.45
            x, y = terrain.world.world_to_grid(goal[0], goal[1])
            if not arrived and terrain.world.in_bounds(x, y) and abs(terrain.heights[y, x] - goal[2]) <= self.params.max_step_m:
                path = terrain.path((x, y))
                if path:
                    return path
        path = terrain.continuation(obs.pose, self.active["direction"], self.source_height)
        self._surface_goal = path[-1] if path else None
        return path

    def _abandon(self, obs, reason):
        self.events.append({"action": obs.step, "event": "portal_deferred", "reason": reason,
                            "portal_id": self.active["id"] if self.active else None})
        if self.active:
            self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions
        self.active = None
        self._traversing = False
        self.path = []
        self.phase = "floor_search"
        self.policy.route_memory.clear("portal deferred")
        if self.policy.hierarchy is not None:
            self.policy.hierarchy.machine.transition("room_reasoning", "stair proposal deferred")

    def blocked(self, obs):
        self.failures += 1
        self.path = []
        self.terrain.blocked(obs.pose, self.policy.episode.action_spec.forward_step_m)

    def filter_action(self, obs, action):
        if not self.traversing:
            h = self.policy.hierarchy
            if h is not None:
                h.motion.update(obs, self.policy.mapping.worlds[self.floor_id])
                if not h.motion.permits(obs, action):
                    return DiscreteAction.TURN_LEFT
            return action
        if action == DiscreteAction.MOVE_FORWARD and not self.terrain.permits(obs.pose, self.policy.episode.action_spec.forward_step_m):
            return DiscreteAction.TURN_LEFT
        return action

    def diagnostics(self):
        return {"phase": self.phase, "atlas": self.policy.mapping.atlas.diagnostics(),
                "floor_contexts": self.policy.floors.diagnostics(), "events": list(self.events),
                "portals": [{k: v for k, v in q.items() if k != "path"} for q in self.portals],
                "active_portal": self.active["id"] if self.active else None}


