"""The kinematic follower: every committed route becomes the exact actions that walk it, re-planned when the agent leaves them.

The 2D planner says where the route goes; the discrete-action lattice
(``core/planning/planners/lattice``) says which MOVE_FORWARDs and TURNs walk
it, collision-checked on the benchmark's own step and turn; the converter
executes them verbatim while the agent stands where the plan expects it
(``NavigationCommand.actions``). This module owns the plan between the two:
the cursor into it (every executed action is reported back through
``notify_action``), the re-plan when the route changes, the agent is not
where the plan says, a fresh frame puts an obstacle on it, or the simulator
refused a step -- and the refused step itself, remembered as a pose the
lattice may not step forward from again, instead of a cell painted
occupied that would seal a doorway.
"""
from __future__ import annotations

from dataclasses import replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid, build_collision_mask
from sparx_agency.core.planning.planners.lattice import DiscreteActionLatticePlanner, LatticePlannerParams

#: How many of the latest plan events are kept for the record.
EVENT_MEMORY = 60


class KinematicFollower:
    """Attach the lattice's action plan to the policy's follow commands and keep it in step with the agent.

    Args:
        policy: The search policy (its ``episode``, ``converter_params`` and ``route_memory`` are read).
        settings: :class:`~sparx_agency.core.planning.object_search.navigation.settings.KinematicSettings`.
    """

    def __init__(self, policy, settings):
        self.policy, self.settings = policy, settings
        converter = policy.converter_params
        self.lattice = DiscreteActionLatticePlanner(policy.episode.action_spec, LatticePlannerParams(
            turn_action_cost=settings.turn_action_cost, corridor_m=settings.corridor_m,
            route_offset_cost=settings.route_offset_cost,
            heuristic_weight=settings.heuristic_weight, max_expansions=settings.max_expansions,
            refused_move_actions=settings.refused_move_actions, refused_move_radius_m=settings.refused_move_radius_m,
            goal_tolerance_m=converter.goal_tolerance_m))
        self.position_tolerance_m = converter.plan_position_tolerance_m
        self.heading_tolerance_rad = math.radians(converter.plan_heading_tolerance_deg)
        self._plan = None
        self._failed = None                     # (route key, step) of the last failed plan from this route
        self._failure = None                    # why, for the command's record
        self._emitted_step = -1
        self._mask = None                       # (world id, radius) -> collision mask of the plan's radius
        self.events = []
        self.stats = {"commands": 0, "planned": 0, "continued": 0, "resynced": 0, "completed": 0, "unplanned": 0,
                      "executed": 0, "replans": 0, "refused": 0, "squeezed": 0}

    # ------------------------------------------------------------------
    # The command
    # ------------------------------------------------------------------
    def command(self, observation, world, planner, command):
        """``command`` with the plan's remaining actions attached, or as it came when no plan exists.

        Args:
            observation: The current observation.
            world: The map the route was planned on.
            planner: The 2D planner that planned it (its fields price the lattice).
            command: The follow command the policy is about to return.
        """
        if not self.settings.enabled or not command.waypoints:
            return command
        self.stats["commands"] += 1
        step = int(observation.step)
        pose = observation.pose
        route_key = tuple(command.waypoints)
        plan, reason, resynced = self._plan, None, False
        if plan is None:
            reason = "no_plan"
        elif plan["route"] != route_key:
            reason = "route_changed"
        elif not self._at_expected(pose, plan):
            # The full state against the plan, every step: a heading the plan did not predict is a re-plan;
            # a displacement of centimetres at the planned heading (the simulator's navmesh snap) keeps
            # the plan, re-anchored on the true pose -- the same actions, the remaining poses moved with
            # the agent -- when the moved trajectory still clears the map.
            resynced = self._resync(pose, plan, world, planner)
            if not resynced:
                reason = "off_plan"
        if reason is None and not self._still_feasible(plan, world, planner):
            reason = "plan_obstructed"
        if reason is not None:
            plan = self._replan(observation, world, planner, command, reason)
        self._emitted_step = step
        if plan is None:
            self.stats["unplanned"] += 1
            return replace(command, info=dict(command.info, kinematic=dict(self._failure or {}, status="unplanned")))
        remaining = plan["actions"][plan["index"]:]
        if not remaining:
            self.stats["completed"] += 1
            return replace(command, info=dict(command.info, kinematic={"status": "complete", "radius_m": plan["radius"]}))
        if reason is None:
            self.stats["resynced" if resynced else "continued"] += 1
        expected = plan["poses"][plan["index"]]
        turns = sum(1 for a in remaining if a != DiscreteAction.MOVE_FORWARD)
        info = {"status": "planned" if reason is not None else ("resynced" if resynced else "continuing"),
                "actions_left": len(remaining), "turns_left": turns, "radius_m": plan["radius"],
                "corridor_m": self.settings.corridor_m, "planned_at": plan["step"]}
        if reason is not None:
            info["reason"] = reason
        return replace(command, actions=remaining, plan_pose=expected, info=dict(command.info, kinematic=info))

    def _at_expected(self, pose, plan):
        ex, ey, eyaw = plan["poses"][plan["index"]]
        return (math.hypot(pose.x - ex, pose.y - ey) <= self.position_tolerance_m
                and abs(normalize_angle(pose.yaw - eyaw)) <= self.heading_tolerance_rad)

    def _resync(self, pose, plan, world, planner):
        """Re-anchor the plan on a pose displaced by at most ``resync_m`` at the planned heading; False to re-plan."""
        ex, ey, eyaw = plan["poses"][plan["index"]]
        if abs(normalize_angle(pose.yaw - eyaw)) > self.heading_tolerance_rad:
            return False
        dx, dy = float(pose.x) - ex, float(pose.y) - ey
        if math.hypot(dx, dy) > self.settings.resync_m:
            return False
        moved = tuple(plan["poses"][:plan["index"]]) + tuple(
            (x + dx, y + dy, yaw) for x, y, yaw in plan["poses"][plan["index"]:])
        trial = dict(plan, poses=moved)
        if not self._still_feasible(trial, world, planner):
            return False
        plan["poses"] = moved
        plan["resyncs"] = plan.get("resyncs", 0) + 1
        self._event(self._emitted_step, "resynced", "%.3f m" % math.hypot(dx, dy), "index %d" % plan["index"])
        return True

    def _collision_mask(self, world, planner, radius):
        key = (id(world), radius)
        if self._mask is None or self._mask[0] != key:
            self._mask = (key, build_collision_mask(world, planner.params, radius))
        return self._mask[1]

    def _still_feasible(self, plan, world, planner):
        """Whether every remaining forward step of the plan still clears the map in force."""
        mask = self._collision_mask(world, planner, plan["radius"])
        h, w = mask.shape
        origin_x, origin_y, res = world.params.origin_x, world.params.origin_y, world.resolution
        samples = self.lattice.primitives.samples
        step = self.lattice.primitives.step_m
        actions, poses = plan["actions"], plan["poses"]
        for i in range(plan["index"], len(actions)):
            if actions[i] != DiscreteAction.MOVE_FORWARD:
                continue
            x, y, yaw = poses[i]
            c, s = math.cos(yaw), math.sin(yaw)
            for f in samples:
                gx = int(math.floor((x + f * step * c - origin_x) / res))
                gy = int(math.floor((y + f * step * s - origin_y) / res))
                if not (0 <= gx < w and 0 <= gy < h) or mask[gy, gx]:
                    return False
        return True

    def _replan(self, observation, world, planner, command, reason):
        """A fresh plan from the agent's pose along ``command``'s route, or None (recorded in ``_failure``)."""
        step = int(observation.step)
        route_key = tuple(command.waypoints)
        if (self._failed is not None and self._failed[0] == route_key
                and step - self._failed[1] < self.settings.replan_backoff_actions):
            return None
        self._plan = None
        self.stats["replans"] += 1
        pose = observation.pose
        fields = planner.fields_for(world)
        path = getattr(self.policy.route_memory, "path", None)
        metadata = getattr(path, "metadata", None) or {}
        radius = float(metadata.get("inflate_used_m", planner.last_inflate_m))
        floor = float(planner.params.inflate_floor_m)
        # The standoff ladder: the route's own, the planner's floor, then the lattice floor -- the 2D route joins
        # cell centres and passes an obstacle corner diagonally 0.21 m off it, which the sampled step refuses at
        # 0.25 m; a plan at 0.20 m is the same clearance the route already has, and still an exact plan.
        rungs = sorted({radius, floor, float(self.settings.lattice_floor_m)}, reverse=True)
        goal_yaw = command.final_yaw if self.settings.plan_final_yaw else None
        start = (float(pose.x), float(pose.y), float(pose.yaw))
        result = None
        for attempt in rungs:
            cost, _lethal = assemble_cost_grid(fields, planner.params, attempt)
            result = self.lattice.plan(start, route_key, world.params.origin_x, world.params.origin_y, world.resolution,
                                       cost, fields.clearance, attempt, goal_yaw=goal_yaw, step=step,
                                       start_free_radius_m=attempt)
            if result.ok:
                if attempt < floor:
                    self.stats["squeezed"] += 1
                radius = attempt
                break
        if result is None or not result.ok:
            self._failed = (route_key, step)
            self._failure = {"reason": reason, "lattice": result.status if result is not None else "no_attempt",
                             "message": result.message if result is not None else ""}
            self._event(step, "failed", reason, result.status if result is not None else "no_attempt")
            return None
        self._failed = self._failure = None
        self._plan = {"route": route_key, "actions": tuple(result.actions), "poses": tuple(result.poses), "index": 0,
                      "radius": radius, "step": step}
        self.stats["planned"] += 1
        self._event(step, "planned", reason, "%d actions, %d turns, %.2f m" % (len(result.actions), result.turns, radius))
        return self._plan

    # ------------------------------------------------------------------
    # What the agent did
    # ------------------------------------------------------------------
    def note_action(self, observation, action):
        """The action executed on this observation: advance the cursor, or drop a plan the agent left."""
        plan = self._plan
        if plan is None or int(observation.step) != self._emitted_step:
            return
        if plan["index"] < len(plan["actions"]) and action == plan["actions"][plan["index"]]:
            plan["index"] += 1
            self.stats["executed"] += 1
        elif action in (DiscreteAction.LOOK_UP, DiscreteAction.LOOK_DOWN, DiscreteAction.STOP):
            return                                  # the camera's or the policy's action: the plan waits
        else:
            if plan["index"] < len(plan["actions"]) and plan["actions"][plan["index"]] == DiscreteAction.MOVE_FORWARD:
                # The safety filter turned the planned step into a turn: that step is refused from this pose as
                # surely as if the simulator had refused it, or the next plan asks for it again from one turn over.
                pose = observation.pose
                self.lattice.note_refused(float(pose.x), float(pose.y), float(pose.yaw), int(observation.step))
                self.stats["refused"] += 1
            self.invalidate("vetoed_" + action.name)

    def note_refused(self, observation):
        """The simulator refused the MOVE_FORWARD from this pose: no plan steps forward from here again."""
        pose = observation.pose
        self.lattice.note_refused(float(pose.x), float(pose.y), float(pose.yaw), int(observation.step))
        self.stats["refused"] += 1
        self.invalidate("refused_step")

    def invalidate(self, reason):
        """Drop the plan in force; the next command plans afresh."""
        if self._plan is not None:
            self._event(self._emitted_step, "dropped", reason, "")
        self._plan = None
        self._failed = None
        self.stats["dropped:" + reason] = self.stats.get("dropped:" + reason, 0) + 1

    def _event(self, step, kind, reason, detail):
        self.events.append({"step": int(step), "event": kind, "reason": reason, "detail": detail})
        del self.events[:-EVENT_MEMORY]

    @property
    def active(self):
        """Whether a plan with actions left is in force."""
        plan = self._plan
        return plan is not None and plan["index"] < len(plan["actions"])

    def diagnostics(self):
        return {"enabled": self.settings.enabled, "stats": dict(self.stats), "lattice": self.lattice.diagnostics(),
                "events": list(self.events)}
