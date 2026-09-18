"""Committed ground-robot connector lifecycle driven only by measured RGB-D/pose."""
from __future__ import annotations

from collections import deque
import math

from sparx_agency.core.planning.objnav.types.command import NavigationCommand


class StairTraversal:
    """Keep connector/direction/trace until destination exit or a verified retreat.

    A height threshold is neither an arrival nor a retreat-completion test.
    Small landings and stationary turns cannot settle the transaction. Exhausted
    recovery explicitly halts; it never relabels a robot on stairs as searching.
    """

    def __init__(self, coordinator, obs):
        self.coordinator = coordinator
        self.policy, self.terrain, self.params = coordinator.policy, coordinator.terrain, coordinator.params
        self.portal = coordinator.active
        self.direction = self.portal["direction"]
        self.source_floor = coordinator.floor_id
        self.source_height = self.policy.mapping.atlas.elevation_m
        current = (obs.pose.x, obs.pose.y, obs.pose.z)
        approach = []
        for point in reversed(coordinator.source_trail):
            approach.append(point)
            if math.dist(point[:2], current[:2]) >= self.params.exit_distance_m + 0.5:
                break
        self.trace = list(reversed(approach))
        if not self.trace or self.trace[-1] != current:
            self.trace.append(current)
        self.source_xyz = self.trace[0]
        self.started_step = self.last_progress_step = obs.step
        self.phase = "TRAVERSE"
        self.samples = deque(maxlen=self.params.stable_samples)
        self.last_height_change = current
        self.previous = current
        self.best_rise_m = 0.0
        self.retreat_path = None
        self.retreat_step = None
        self.retreat_index = 0
        self.surface_goal = None
        self.path = []
        self.scans = self.failures = 0
        self.close_support = False
        self.arrival_allowed = False
        self.completion_reason = None
        self.destination_height_m = None
        self.guide_index = 0
        self.policy.mapping.atlas.begin_transition(obs.pose)

    def observe(self, obs):
        xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        moved = math.dist(xyz[:2], self.previous[:2]) > 0.04
        if moved:
            self.trace.append(xyz)
            self.samples.append(xyz)
            self.last_progress_step = obs.step
        if abs(xyz[2] - self.previous[2]) > 0.04:
            self.last_height_change = xyz
        self.previous = xyz
        self.best_rise_m = max(self.best_rise_m, self.direction * (xyz[2] - self.source_height))
        flat = (len(self.samples) == self.params.stable_samples
                and max(p[2] for p in self.samples) - min(p[2] for p in self.samples) <= self.params.stable_height_m
                and math.dist(self.samples[0][:2], self.samples[-1][:2]) >= self.params.stable_distance_m)
        area = self.terrain.plateau_area(obs.pose)
        supported = flat and area >= self.params.min_floor_area_m2
        exited = (math.dist(xyz[:2], self.last_height_change[:2]) >= self.params.exit_distance_m
                  and self.terrain.stair_distance(obs.pose) >= self.params.exit_distance_m / 2)
        self.arrival_allowed = False
        if self.phase == "SAFE_HALT":
            return
        if self.retreat_path is not None:
            source = abs(xyz[2] - self.source_height) <= self.params.stable_height_m
            near_entry = math.dist(xyz[:2], self.source_xyz[:2]) <= 0.45
            # Returning merely inside departure_m (or floor_match_m) is unsafe.
            self.arrival_allowed = supported and source and near_entry and exited
            if self.arrival_allowed:
                self.completion_reason = "source_platform_and_entry_verified"
            return
        expected = self.portal.get("destination")
        atlas = self.policy.mapping.atlas
        if expected is not None:
            destination = abs(xyz[2] - atlas.floors[expected].elevation_m) <= self.params.floor_match_m
        else:
            destination = self.direction * (xyz[2] - self.source_height) >= self.params.min_floor_separation_m
        if supported and destination:
            self.phase = "CONFIRM_DESTINATION"
            self.destination_height_m = xyz[2]
            self.arrival_allowed = exited
            if exited:
                self.completion_reason = "stable_platform_and_stair_exit"
        else:
            self.phase = "TRAVERSE"
            self.destination_height_m = None

    def plan(self, obs):
        elapsed = obs.step - self.started_step
        stalled = obs.step - self.last_progress_step >= 18
        if self.phase == "SAFE_HALT":
            return NavigationCommand.stop_here(info={"kind": self.phase, "reason": self.completion_reason,
                                                     "target_confirmed": False})
        if self.retreat_path is None and (self.failures >= self.params.max_failures
                                          or elapsed >= self.params.transition_actions or stalled):
            self.retreat_path = list(reversed(self.trace))
            self.retreat_step = obs.step
            self.surface_goal = None
            self.phase = "RETREAT"
            self.coordinator.events.append({"action": obs.step, "event": "retreat_started", "portal_id": self.portal["id"],
                                             "reason": "timeout" if elapsed >= self.params.transition_actions else "blocked_or_stalled"})
        if self.retreat_path is not None:
            if obs.step - self.retreat_step >= self.params.retreat_actions:
                self.phase = "SAFE_HALT"
                self.completion_reason = "retreat_budget_exhausted_on_unconfirmed_support"
                return self.plan(obs)
            path = self._retreat_route(obs)
        else:
            path = self._surface_route(obs)
        path = self.terrain.steering_path(obs.pose, path, self.policy.episode.action_spec)
        if path:
            self.path = path
            self.policy._route, self.policy._goal = path, tuple(path[-1][:2])
            self.policy.last_world = self.terrain.world
            self.scans = 0
            return NavigationCommand.follow(path, info={"kind": self.phase, "portal": self.portal["id"]})
        self.scans += 1
        if self.scans >= 6:
            # Latch the deeper inspection; never alternate 30/60 on scan count.
            self.close_support = True
        if self.scans >= 12:
            self.failures += 1
            self.scans = 0
        return NavigationCommand.hold(final_yaw=obs.pose.yaw + self.policy.episode.action_spec.turn_angle_rad,
                                      info={"kind": self.phase, "portal": self.portal["id"],
                                            "reason": "acquire connected tread support"})

    def _surface_route(self, obs):
        goal = self.surface_goal
        if goal is not None:
            arrived = math.dist((obs.pose.x, obs.pose.y), goal[:2]) <= 0.35 and abs(obs.pose.z - goal[2]) <= 0.35
            x, y = self.terrain.world.world_to_grid(*goal[:2])
            if not arrived and self.terrain.world.in_bounds(x, y) and abs(self.terrain.heights[y, x] - goal[2]) <= self.params.max_step_m:
                path = self.terrain.path((x, y))
                if path:
                    return path
        guide = self.portal["path"]
        xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        if guide:
            nearest = min(range(self.guide_index, len(guide)), key=lambda i: math.dist(xyz, guide[i]))
            if math.dist(xyz, guide[nearest]) < 0.5:
                self.guide_index = nearest
            # The safe route to the flight contains the lateral alignment that
            # a nearest-waypoint chord loses at a narrow tread entrance.
            for point in reversed(guide[self.guide_index + 1:]):
                if math.dist(xyz, point) < 0.5:
                    continue
                cell = self.terrain.world.world_to_grid(*point[:2])
                if not self.terrain.world.in_bounds(*cell):
                    continue
                x, y = cell
                if abs(self.terrain.heights[y, x] - point[2]) <= self.params.max_step_m:
                    path = self.terrain.path(cell)
                    if path:
                        self.surface_goal = path[-1]
                        return path
        path = self.terrain.continuation(obs.pose, self.direction, self.source_height,
                                         guide=guide[-1] if guide else None)
        self.surface_goal = path[-1] if path else None
        return path

    def _retreat_route(self, obs):
        xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        # A failed attempt can revisit the same coordinate many times. Pick
        # the furthest height-matching revisit in the retreat direction, and
        # never jump back to the beginning of that loop on the next action.
        close = [i for i in range(self.retreat_index, len(self.retreat_path))
                 if math.dist(self.retreat_path[i], xyz) <= 0.4]
        if close:
            self.retreat_index = max(close)
        index = self.retreat_index
        # Route through current connected depth support, not a chord across a bend.
        for point in self.retreat_path[index + 1:]:
            if math.dist(xyz, point) < 0.4:
                continue
            cell = self.terrain.world.world_to_grid(*point[:2])
            if self.terrain.world.in_bounds(*cell):
                path = self.terrain.path(cell)
                if path:
                    return path
            break
        return []

    def diagnostics(self):
        return {"phase": self.phase, "portal_id": self.portal["id"], "source_floor": self.source_floor,
                "direction": self.direction, "source_height_m": self.source_height,
                "source_platform_xyz": self.source_xyz, "retreat_index": self.retreat_index,
                "destination": self.portal.get("destination"), "destination_height_m": self.destination_height_m,
                "best_rise_m": self.best_rise_m, "started_step": self.started_step,
                "arrival_allowed": self.arrival_allowed, "completion_reason": self.completion_reason,
                "retreat_step": self.retreat_step, "failures": self.failures}
