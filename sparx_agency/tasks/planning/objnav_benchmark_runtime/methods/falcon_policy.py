"""Observed-only FALCON → room LLM → RPT* → A*/WA* ObjectNav orchestration."""
from __future__ import annotations

from dataclasses import asdict
import math
import time
from types import SimpleNamespace
import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D, normalize_angle
from sparx_agency.core.planning.exploration.falcon.bursts import BurstMachine, DONE, EXPLORE, REASON, RECOVER, SELECT, TRANSIT, VERIFY
from sparx_agency.core.planning.exploration.falcon.ordering import PlanningDeadline
from sparx_agency.core.planning.exploration.falcon.planner import FalconPlanner
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.room_search_policy import RoomCandidate
from sparx_agency.core.planning.objnav.action_converter.ladder import reach_floor
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_regions import ObservedRegions
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_routes import LocalRouteBank
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_motion import GroundMotion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute


class FalconObjectNav:
    def __init__(self, policy):
        self.policy, self.params = policy, policy.settings.falcon
        self.machine = BurstMachine(self.params, policy.episode.max_steps)
        self.entry_margin = max(policy.converter_params.goal_tolerance_m, reach_floor(policy.episode.action_spec)) + 1.5 * policy.settings.map_resolution_m
        self.regions = ObservedRegions(self.params, self.entry_margin)
        self.explorer = FalconPlanner(self.params)
        self.routes, self.motion = LocalRouteBank(), GroundMotion(policy)
        self.archived_plans, self.archived_regions = [], []
        self.safety_vetoes = 0
        self.current = self.saved_route = self.selected = None
        self.failures = self.stagnant = 0
        self.gain = 0.0
        self.errors = []
        self._burst_recorded = -1
        self._floor = policy.mapping.floor_revision
        self._transit_start = 0
        self.recovery_phase = EXPLORE
        self.nonlocal_failures = {TRANSIT: 0, VERIFY: 0}

    def plan(self, observation, world, confirmed):
        p, m = self.policy, self.machine
        m.check_limits()
        if self._floor != p.mapping.floor_revision:
            m.end("floor_change")
            m.transition(REASON, "observed floor changed; geometry invalidated, allowance retained")
            self._floor = p.mapping.floor_revision
            self.archived_plans.extend(self.explorer.records)
            self.archived_regions.extend(self.regions.diagnostics())
            self.regions = ObservedRegions(self.params, self.entry_margin)
            self.explorer = FalconPlanner(self.params)
            self.routes = LocalRouteBank()
            self.current = self.saved_route = self.selected = None
        self.explorer.observe(world)
        self.cost = self.motion.update(observation, world)
        if self.regions.anchor is None and m.burst is None:
            self._begin(observation, world)
        if confirmed and p._target_xy is not None and math.dist((observation.pose.x, observation.pose.y), p._target_xy) <= p.settings.stop_distance_m:
            m.end("verified_target")
            m.transition(DONE, "fresh multi-view-confirmed target")
            return NavigationCommand.stop_here(info={"reason": "fresh multi-view-confirmed target"})
        interrupted = self._target(observation, world)
        if interrupted is not None:
            return interrupted
        # All zero-motion transitions happen in one bounded call, not idle turns.
        for _ in range(5):
            if m.phase == DONE:
                return NavigationCommand.stop_here(info={"reason": "bounded search terminated; not target-free proof"})
            if m.phase == REASON:
                self._reason(observation, world)
            elif m.phase == SELECT:
                selection_started = time.monotonic()
                self._select(observation, world)
                p.telemetry.latencies["room_selection"].append((time.monotonic() - selection_started) * 1000)
            elif m.phase == TRANSIT:
                command = self._transit(observation, world)
                if command is not None:
                    return command
            elif m.phase == VERIFY:
                command = self._target(observation, world)
                if command is not None:
                    return command
            elif m.phase == RECOVER:
                if m.phase_actions >= self.params.recovery_actions:
                    m.transition(self.recovery_phase, "bounded recovery completed")
                else:
                    return NavigationCommand.hold(final_yaw=observation.pose.yaw + p.episode.action_spec.turn_angle_rad,
                                                  info={"reason": "bounded local recovery"})
            else:
                command = self._explore(observation, world)
                if command is not None:
                    return command
        m.transition(DONE, "bounded transition guard")
        return NavigationCommand.stop_here(info={"reason": "no reachable useful observed region"})

    def _begin(self, obs, world, room_id=None):
        self.regions.start(world, obs.pose, self.policy.graph.registry.rooms, room_id, self.machine.actions)
        self.machine.begin("floor%d/region%d" % (self._floor, len(self.regions.visits)))
        self.current = None
        self.failures = self.stagnant = 0
        self.gain = 0.0
        self.policy.route_memory.clear("exploration_burst_started")
        self.current = self.routes.restore(self.regions.mask, self.policy, self.machine.actions)

    def _target(self, obs, world):
        p, m = self.policy, self.machine
        if m.phase == RECOVER and self.recovery_phase == VERIFY and self.saved_route is not None:
            if p._target_xy is not None and m.actions - self._target_pause < self.params.target_actions:
                return None
            m.transition(VERIFY, "target recovery allowance ended")
        if self.saved_route is not None and m.phase not in (VERIFY, RECOVER):
            m.transition(VERIFY, "finish existing target interrupt without refilling it")
        if p._target_xy is not None and m.phase != VERIFY:
            self.saved_route = (p.route_memory, p._route, p._goal)
            self._target_pause = m.actions
            self.nonlocal_failures[VERIFY] = 0
            p.route_memory = CommittedRoute(p.episode.action_spec, p.converter_params, p.route_settings)
            m.interrupt()
        if m.phase != VERIFY:
            return None
        if p._target_xy is not None and m.actions - self._target_pause < self.params.target_actions:
            command = p._approach(obs, world)
            if command is not None:
                return command
        if p._target_id is not None:
            p.target_evidence.reject(p._target_id, obs.step)
        p._target_xy = p._target_id = None
        if self.saved_route is not None:
            p.route_memory, p._route, p._goal = self.saved_route
            p.route_memory.resume_after_pause(m.actions - self._target_pause)
            self.saved_route = None
        m.resume()
        return None

    def _reason(self, obs, world):
        p, m = self.policy, self.machine
        if m.burst is not None and m.burst.token != self._burst_recorded:
            self.regions.finish(m.burst.reason, m.actions, self.gain)
            if self.regions.mask is not None and m.burst.reason == "action_budget":
                self.routes.suspend(self.regions.mask, self.current, p, m.actions)
            self._burst_recorded = m.burst.token
        started = time.monotonic()
        try:
            p.graph.update(world, p.landmarks.confirmed(), p.target, doors=p.doors.confirmed(), step=obs.step, reason=False)
            p.graph.reason(world, p.target, obs.step)
        except Exception as exc:
            # Existing bounded HTTP retries/schema repair remain authoritative.
            # Failure is visible and terminal, never a uniform-or-frontier run.
            self.errors.append({"phase": REASON, "error": str(exc), "action": m.actions})
            m.transition(DONE, "LLM unavailable after bounded retries")
            return
        finally:
            p.telemetry.latencies["room_reasoning"].append((time.monotonic() - started) * 1000)
        m.transition(SELECT, "accumulated room evidence refreshed")

    def _select(self, obs, world):
        p, m = self.policy, self.machine
        try:
            goals, histories = self.regions.room_goals(world, self.cost, p.graph.registry.rooms, obs.pose, m.actions)
            if not goals:
                m.transition(DONE, "no useful reachable region eligible")
                return
            instance, _ = build_instance(world, self.cost, goals, p.graph.probs,
                                         depot_xy=(obs.pose.x, obs.pose.y), snap_radius_m=0,
                                         cruise_speed_mps=p.episode.action_spec.forward_step_m / p.settings.action_time_s)
            total = sum(p.graph.probs.get(pid, 0.0) for pid in goals)
            candidates = [RoomCandidate(room_id=pid, label="observed", prob=p.graph.probs.get(pid, 0.0),
                                        prob_renorm=p.graph.probs.get(pid, 0.0) / total, xy=xy)
                          for pid, xy in sorted(goals.items())] if total > 0 else []
            order = p.solver(candidates, instance)
            if p.solver.last is not None:
                data = asdict(p.solver.last)
                p._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in data.items()})
            if not order:
                m.transition(DONE, "RPT* returned no eligible room")
                return
            pid = int(order[0])
            self.selected = (pid, tuple(goals[pid]), p.graph.registry.rooms[pid].mask.copy())
            self._transit_start = m.actions
            self.nonlocal_failures[TRANSIT] = 0
            m._entry_streak = 0
            p.route_memory.clear("RPT room selected")
            m.transition(TRANSIT, "RPT* selected room %s" % pid)
        except (ValueError, PlanningDeadline) as exc:
            self.errors.append({"phase": SELECT, "error": str(exc), "action": m.actions})
            m.transition(DONE, "bounded room planning failed")

    def _transit(self, obs, world):
        p, m = self.policy, self.machine
        pid, goal, mask = self.selected
        x, y = world.world_to_grid(obs.pose.x, obs.pose.y)
        inside = world.in_bounds(x, y) and mask[y, x]
        if m.entered(inside, obs.step) and (p.route_memory.arrived(obs) or math.dist((obs.pose.x, obs.pose.y), goal) < self.entry_margin):
            self._begin(obs, world, pid if pid in p.graph.registry.rooms else None)
            return None
        if m.actions - self._transit_start >= self.params.transit_actions:
            m.transition(DONE, "committed transit action limit")
            return None
        command = p._navigate(obs, world, goal, "transit/%s" % pid)
        if command is None:
            m.transition(DONE, "selected room route unavailable")
        return command

    def _explore(self, obs, world):
        p, m = self.policy, self.machine
        gain = p.telemetry.curve[-1]["new_m2"] if p.telemetry.curve else 0.0
        self.gain += gain
        self.stagnant = 0 if gain >= self.params.low_gain_m2 else self.stagnant + 1
        if self.stagnant >= self.params.low_gain_actions:
            m.end("negligible_gain")
            return None
        scope = self.regions.scope(world, p.graph.registry.rooms)
        self.motion.scope = scope
        if self.current is not None:
            x, y = self.current.cell
            arrived = p.route_memory.arrived(obs)
            facing = abs(normalize_angle(self.current.yaw - obs.pose.yaw)) <= p.episode.action_spec.turn_angle_rad / 2 + 1e-6
            if arrived and facing:
                self.explorer.retire(self.current)
                self.current = None
                p.route_memory.clear("falcon_viewpoint_observed")
            elif not scope[y, x] or not np.isfinite(self.cost[y, x]) or not p.route_memory.reusable(obs, world, self.motion, world.grid_to_world(x, y), "falcon"):
                self.blocked("viewpoint_or_route_invalid")
                return None
            else:
                return NavigationCommand.follow(p.route_memory.path, final_yaw=self.current.yaw, info={"kind": "falcon", "route": "committed"})
        # Pass the public action spec explicitly; observations carry no private
        # simulator handle or evaluator telemetry.
        request = SimpleNamespace(pose=obs.pose, camera=obs.camera, step=obs.step,
                                  action_spec=p.episode.action_spec, topology_cost=self.motion.topology_cost,
                                  arrival_m=max(p.converter_params.goal_tolerance_m, reach_floor(p.episode.action_spec)))
        result = self.explorer.plan(world, self.cost, scope, request, p.settings.body_height_m)
        p.telemetry.latencies["falcon_planning"].append(result.diagnostics["planning_ms"])
        if result.status != "planned":
            bootstrap = result.status == "no_useful_reachable_frontiers" and m.burst.actions < self.params.recovery_actions
            if (result.status == "start_not_safe" or bootstrap) and self.failures < self.params.max_failed_goals:
                self.blocked(result.status)
                return None
            m.end(result.status)
            return None
        self.current = result.viewpoints[0]
        points = [Pose2D(obs.pose.x, obs.pose.y, obs.pose.yaw)]
        points.extend(Pose2D(*world.grid_to_world(*cell)) for cell in result.routes[0][1:])
        if len(points) == 1:
            points.append(Pose2D(*world.grid_to_world(*self.current.cell)))
        path = Path2D(tuple(points), "world", {"planner": "falcon-planar-free-route", "inflate_used_m": p.settings.body_radius_m})
        goal = world.grid_to_world(*self.current.cell)
        p.route_memory.adopt(path, goal, "falcon", obs)
        p._route, p._goal = path, goal
        return NavigationCommand.follow(path, final_yaw=self.current.yaw, info={"kind": "falcon", "frontier": self.current.frontier})

    def filter_action(self, obs, action):
        """Veto the actual converter output, including after cursor restoration."""
        if self.motion.permits(obs, action, local=self.machine.phase == EXPLORE):
            return action
        self.safety_vetoes += 1
        self.blocked("discrete_swept_motion_unsafe")
        return DiscreteAction.TURN_LEFT

    def blocked(self, reason="forward_blocked"):
        self.failures += 1
        if self.machine.phase == EXPLORE:
            self.explorer.retire(self.current, rejected=True)
            self.current = None
            self.policy.route_memory.clear(reason)
            if self.failures >= self.params.max_failed_goals:
                self.machine.end("unrecoverable_blockage")
            else:
                self.recovery_phase = EXPLORE
                self.machine.transition(RECOVER, reason)
        elif self.machine.phase in (TRANSIT, VERIFY):
            phase = self.machine.phase
            self.nonlocal_failures[phase] += 1
            self.policy.route_memory.clear("route_obstructed")
            if self.nonlocal_failures[phase] >= self.params.max_failed_goals:
                if phase == TRANSIT:
                    self.machine.transition(DONE, "unrecoverable discrete transit blockage")
                else:
                    self.policy.target_evidence.reject(self.policy._target_id, self.machine.actions)
                    self.policy._target_xy = self.policy._target_id = None
            else:
                self.recovery_phase = phase
                self.machine.transition(RECOVER, reason)

    def diagnostics(self):
        self.machine.check_limits()
        return {**self.machine.snapshot(), "regions": self.archived_regions + self.regions.diagnostics(),
                "falcon_plans": self.archived_plans + list(self.explorer.records), "frontier_updates": self.explorer.frontiers.updates,
                "rejected_viewpoints": len(self.explorer.rejected), "visited_viewpoints": len(self.explorer.visited),
                "route_bank": self.routes.diagnostics(), "safety_vetoes": self.safety_vetoes,
                "errors": list(self.errors), "fallback": None}






