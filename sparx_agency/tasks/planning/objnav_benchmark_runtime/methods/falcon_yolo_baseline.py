"""FALCON + YOLO + closing: the goal-free exploration baseline of the ZSON method.

The ablation of the room-search method (``rpt_policy``) down to three parts:

1. **Exploration** -- planar FALCON (``core/planning/exploration/falcon``)
   run goal-free over the whole storey, as Daphna's ``StandaloneFalcon``
   ran it on HM3D (``feat/multi_resolution_falcon_daphna``): one bounded
   FALCON transaction in a disc of ``falcon.scope_radius_m`` about the
   agent, the walk to its first viewpoint, the viewpoint retired, the next
   transaction. When the disc holds no useful frontier the agent relocates
   to the nearest reachable frontier of the storey (A* on the observed map)
   and plans again from there; when the storey has none left it looks all
   round once more and STOPs. No room LLM, no RPT*, no scene graph, no room
   partition, no doors, openings, peeks, glances or sweeps, no probabilities.
2. **Perception** -- the same YOLO-World frame per action and the same fusion
   as the full method (``PerceptionCycle``: landmark map, floor support,
   border and refusal gates, target evidence).
3. **The approach** -- the full method's ``TargetClosing`` takeover, verbatim,
   with its legacy pursuit (``_approach``) behind it: the moment a confident
   box of the target's class has a coherent 3-D projection the takeover
   owns the episode -- verification, standoff, pitch servo, inspection and
   STOP -- exactly as in ``rpt_policy``.

Spawn-floor confinement is the full method's own, unchanged: the building
coordinator (``MultiFloorSearch``) sees ground-truth navmesh stairs through
its perfect detector and places them as portals, and the ``SpawnFloorGuard``
writes every seen flight and every observed drop below the storey plane
occupied on the world the policy plans on, and refuses any goal off the spawn
plane (``allow_stair_traversal`` False, the benchmark default). FALCON's
frontiers, routes and swept-motion vetoes all read that confined world.

The policy hosts the shared components under the same attribute names the
full method gives them (``mapping``, ``planner``, ``route_memory``,
``perception``, ``closing``, ``building``, ``floor_guard``, ``landmarks``,
``target_evidence``, ``_target_xy`` ...), so the components run the code they
run there; what is missing (``graph``, ``loop``, ``solver``, ``floors``) is
what the recorder's panels show as absent.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from sparx_agency.core.common.types import Path2D, Pose2D, normalize_angle
from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.exploration.falcon.params import SOURCE as FALCON_SOURCE
from sparx_agency.core.planning.exploration.falcon.planner import FalconPlanner
from sparx_agency.core.planning.exploration.view_gain import camera_rotation_turns
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.objnav.action_converter.ladder import reach_floor
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.camera_control import CameraControlSettings, CameraController
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DoorSettings, ObservedDoors
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import ExplorationFallback, FallbackSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_metrics import ExplorationMetrics
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_motion import GroundMotion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import TargetEvidence, TargetEvidenceSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception_cycle import PerceptionCycle
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute, RouteSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.sightlines import SightLedger, SightSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.spawn_floor_guard import SpawnFloorGuard
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosing

#: Relocation events kept for the recording; the counters are complete.
MAX_EVENTS = 60


@dataclass(frozen=True)
class BaselineSettings:
    """The knobs of the goal-free exploration loop; everything else is ``RPTSettings``.

    Attributes:
        bootstrap_turns: In-place turns before the first FALCON plan (Daphna's
            ``--bootstrap-turns``). 0: the first frame's cone is frontier
            enough, and FALCON's own viewpoints look where the gain is.
        relocation_candidates: How many frontier clusters of the storey, nearest
            first, a relocation tries to route to on one action before giving up.
        relocation_retire_m: A relocation goal given up (no route, no progress, a
            blocked step at it) refuses every goal within this of it ...
        relocation_retire_actions: ... for this many actions.
        relocation_done_m: A relocation is over early when no frontier cell is
            left within this of its goal: the walk there has shown what it was
            walking to see.
        exhausted_turns: With no FALCON plan and no reachable frontier anywhere,
            the agent turns in place this many actions -- a full look-around at
            the protocol's 30 degrees -- and STOPs if nothing has opened. A new
            frame can turn an unknown cell into a frontier; a boxed-in agent
            cannot be helped by turning, and the recording says which it was.
        stop_when_exhausted: False keeps turning instead of STOPping (the
            episode then runs to its action limit; the score is the same).
    """

    bootstrap_turns: int = 0
    relocation_candidates: int = 8
    relocation_retire_m: float = 1.0
    relocation_retire_actions: int = 60
    relocation_done_m: float = 0.75
    exhausted_turns: int = 12
    stop_when_exhausted: bool = True

    def __post_init__(self):
        for key in ("bootstrap_turns", "relocation_candidates", "relocation_retire_actions", "exhausted_turns"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError("baseline.%s must be a non-negative integer" % key)
        if self.relocation_candidates < 1 or self.exhausted_turns < 1:
            raise ValueError("baseline.relocation_candidates and baseline.exhausted_turns must be positive")
        for key in ("relocation_retire_m", "relocation_done_m"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("baseline.%s must be positive and finite" % key)
        if type(self.stop_when_exhausted) is not bool:
            raise ValueError("baseline.stop_when_exhausted must be boolean")


class _NoPeek:
    """The doorway-peek interface the closing and the building call into; the baseline has no peeks."""

    def cancel(self, obs, reason, restore=True):
        return None

    def floor_ready(self):
        return True

    def pending_rooms(self):
        return []

    def diagnostics(self):
        return {"enabled": False, "coverage": None}


class GoalFreeFalcon:
    """Planar FALCON run goal-free: bounded transactions about the agent, relocation between them.

    Owns the FALCON planner (its frontier memory, visited and rejected
    viewpoints), the ground-motion safety boundary and the relocation
    bookkeeping. Every route it commits goes through the policy's route
    memory, like the full method's FALCON hybrid; every goal passes the
    spawn-floor guard.
    """

    def __init__(self, policy):
        self.policy = policy
        self.params = policy.settings.falcon
        self.settings = policy.baseline
        self.explorer = FalconPlanner(self.params)
        self.motion = GroundMotion(policy)
        self.current = None                # the FALCON viewpoint being walked to
        self.scope = None                  # the disc the transaction in force was planned in
        self.relocation = None             # {"goal", "step", "cells", "distance_m"} of the relocation in force
        self.retired = []                  # [(xy, action)] relocation goals given up
        self.bootstrap_left = self.settings.bootstrap_turns
        self.exhausted = 0                 # consecutive decisions with neither a FALCON plan nor a relocation
        self.cost = self.topology_cost = None
        self.stats = Counter()
        self.events = []
        self._floor_id = None
        self._step = None

    # -- per action ---------------------------------------------------------------------
    def prepare(self, obs, world):
        """Feed the frame's world to FALCON and the safety boundary; once per action, before any decision.

        Called for every action the closing does not own, so the swept-motion
        veto in :meth:`filter_action` reads this action's confined map even
        when the legacy pursuit, not FALCON, chose the command.
        """
        if self._step == obs.step:
            return self.cost
        self._step = obs.step
        self._sync_floor(obs)
        self.explorer.observe(world)
        self.cost = self.motion.update(obs, world)
        self.topology_cost = self.motion.topology_cost
        return self.cost

    def plan(self, obs, world):
        p = self.policy
        cost = self.prepare(obs, world)
        if self.bootstrap_left > 0:
            self.bootstrap_left -= 1
            self.stats["bootstrap_turns"] += 1
            return NavigationCommand.hold(
                final_yaw=normalize_angle(obs.pose.yaw + p.episode.action_spec.turn_angle_rad),
                info={"kind": "bootstrap_scan", "remaining": self.bootstrap_left})
        if self.current is not None:
            command = self._continue_viewpoint(obs, world, cost)
            if command is not None:
                return command
        if self.relocation is not None:
            command = self._continue_relocation(obs, world)
            if command is not None:
                return command
        command = self._falcon(obs, world, cost)
        if command is None:
            command = self._relocate(obs, world)
        if command is not None:
            self.exhausted = 0
            return command
        self.exhausted += 1
        self.stats["exhausted_decisions"] += 1
        turn = p.episode.action_spec.turn_angle_rad
        if self.settings.stop_when_exhausted and self.exhausted >= self.settings.exhausted_turns:
            self.stats["exhausted_stops"] += 1
            self._record({"action": obs.step, "event": "exploration_exhausted_stop",
                          "last_falcon_status": self._last_falcon_status()})
            return NavigationCommand.stop_here(info={
                "kind": "exploration_exhausted",
                "reason": "no FALCON plan in the disc and no reachable frontier on the storey for %d actions"
                          % self.exhausted})
        return NavigationCommand.hold(
            final_yaw=normalize_angle(obs.pose.yaw + turn),
            info={"kind": "exploration_exhausted_turn", "falcon_status": self._last_falcon_status(),
                  "turns": self.exhausted, "turns_before_stop": self.settings.exhausted_turns})

    def filter_action(self, obs, action):
        """Veto a forward step off FALCON's own route that its safety map says is unsafe, held to the disc.

        FALCON's routes are planned on the swept-motion map this check reads,
        so a veto there is the converter's 30-degree quantisation clipping a
        corner, and the viewpoint is rejected. A relocation or the legacy
        pursuit walks an A* route the planner qualified at the body radius --
        through the squeeze of a doorway the stricter swept check refuses --
        and is not re-judged here, as the full method's frontier explorer
        does not re-judge its A* routes: the simulator's blocked step is the
        check (Wiconisco/000000, 2026-10-18: 179 vetoes in the doorway of the
        first room, every relocation retired as ``no_progress``).
        """
        if self.current is None or self.policy.route_memory.kind != "falcon" or self.cost is None:
            return action
        if self.motion.permits(obs, action, local=True):
            return action
        self.stats["safety_vetoes"] += 1
        self._retire_current("discrete_swept_motion_unsafe", rejected=True)
        return DiscreteAction.TURN_LEFT

    def suspend(self, reason):
        """Another owner (the legacy pursuit) takes the action: the viewpoint in force is dropped, not rejected.

        Its frontier is left for the next transaction to sample again; the
        relocation in force, if any, keeps its goal and replans when it resumes.
        """
        if self.current is not None:
            self.current = None
            self.stats["viewpoints_suspended"] += 1
            self.stats["suspended:" + str(reason)] += 1

    def blocked(self, obs):
        """A forward step the simulator refused: the FALCON viewpoint is rejected; a relocation replans."""
        self.stats["blocked"] += 1
        if self.current is not None:
            self._retire_current("forward_blocked", rejected=True)

    # -- FALCON ------------------------------------------------------------------------------
    def _continue_viewpoint(self, obs, world, cost):
        p = self.policy
        spec = p.episode.action_spec
        x, y = self.current.cell
        arrived = p.route_memory.arrived(obs)
        facing = abs(normalize_angle(self.current.yaw - obs.pose.yaw)) <= spec.turn_angle_rad / 2 + 1e-6
        if arrived and facing:
            self.explorer.retire(self.current)
            self.current = None
            self.stats["viewpoints_observed"] += 1
            p.route_memory.clear("falcon_viewpoint_observed")
            p._route = p._goal = None
            return None
        goal = world.grid_to_world(x, y)
        if not np.isfinite(cost[y, x]) or not p.route_memory.reusable(obs, world, self.motion, goal, "falcon"):
            self._retire_current(p.route_memory.cleared or "viewpoint_or_route_invalid", rejected=True)
            return None
        return NavigationCommand.follow(p.route_memory.path, final_yaw=self.current.yaw,
                                        info={"kind": "falcon", "route": "committed", "frontier": self.current.frontier})

    def _falcon(self, obs, world, cost):
        """One bounded FALCON transaction in the disc about the agent; its first viewpoint, or None."""
        p = self.policy
        spec = p.episode.action_spec
        self.scope = self._disc(world, obs.pose)
        self.motion.scope = self.scope
        request = SimpleNamespace(pose=obs.pose, camera=obs.camera, step=obs.step, action_spec=spec,
                                  topology_cost=self.topology_cost,
                                  arrival_m=max(p.converter_params.goal_tolerance_m, reach_floor(spec)))
        result = self.explorer.plan(world, cost, self.scope, request, p.settings.body_height_m)
        p.telemetry.latencies["falcon_planning"].append(float(result.diagnostics.get("planning_ms", 0.0)))
        self.stats["falcon_plans"] += 1
        self.stats["falcon:" + result.status] += 1
        if result.status != "planned":
            return None
        view = result.viewpoints[0]
        goal = world.grid_to_world(*view.cell)
        if not p.floor_guard.goal_allowed(obs, world, goal, "falcon"):
            self.explorer.retire(view, rejected=True)
            self.stats["viewpoints_refused_by_guard"] += 1
            return None
        points = [Pose2D(obs.pose.x, obs.pose.y, obs.pose.yaw)]
        points.extend(Pose2D(*world.grid_to_world(*cell)) for cell in result.routes[0][1:])
        if len(points) == 1:
            points.append(Pose2D(*goal))
        path = Path2D(tuple(points), "world", {"planner": "falcon-planar-free-route",
                                               "inflate_used_m": p.settings.body_radius_m})
        p.route_memory.adopt(path, goal, "falcon", obs)
        p._route, p._goal = path, goal
        self.current = view
        return NavigationCommand.follow(path, final_yaw=view.yaw,
                                        info={"kind": "falcon", "route": "planned", "frontier": view.frontier,
                                              "gain_m2": round(view.gain * world.resolution ** 2, 3)})

    def _retire_current(self, reason, rejected):
        p = self.policy
        if self.current is not None:
            self.explorer.retire(self.current, rejected=rejected)
            self.stats["viewpoints_rejected" if rejected else "viewpoints_observed"] += 1
        self.current = None
        p.route_memory.clear(reason)
        p._route = p._goal = None

    def _disc(self, world, pose):
        x, y = world.world_to_grid(pose.x, pose.y)
        rows, columns = np.ogrid[:world.grid.shape[0], :world.grid.shape[1]]
        distance2 = ((columns - x) ** 2 + (rows - y) ** 2) * world.resolution ** 2
        return distance2 <= self.params.scope_radius_m ** 2

    def _last_falcon_status(self):
        records = self.explorer.records
        return records[-1]["status"] if records else "not_planned"

    # -- relocation ---------------------------------------------------------------------------
    def _relocate(self, obs, world):
        """Route to the nearest reachable frontier of the storey outside the exhausted disc; None when there is none."""
        p = self.policy
        boundary = self.explorer.frontiers.boundary
        if boundary is None or self.topology_cost is None:
            return None
        candidates = boundary & np.isfinite(self.topology_cost)
        if self.scope is not None:
            candidates &= ~self.scope
        labels, count = ndimage.label(candidates, structure=np.ones((3, 3), dtype=bool))
        if count == 0:
            return None
        hx, hy = world.world_to_grid(obs.pose.x, obs.pose.y)
        ranked = []
        for index in range(1, count + 1):
            ys, xs = np.nonzero(labels == index)
            if len(xs) < self.params.min_cluster_cells:
                continue
            distance2 = (xs - hx) ** 2 + (ys - hy) ** 2
            nearest = int(np.argmin(distance2))
            ranked.append((int(distance2[nearest]), (int(xs[nearest]), int(ys[nearest])), int(len(xs))))
        ranked.sort()
        self.motion.scope = None
        tried = 0
        for distance2, cell, size in ranked:
            goal = world.grid_to_world(*cell)
            if self._retired_near(goal, obs.step):
                continue
            tried += 1
            if tried > self.settings.relocation_candidates:
                break
            command = p._navigate(obs, world, goal, "relocate")
            if command is None:
                self._retire_goal(goal, obs.step, "pathless")
                continue
            self.relocation = {"goal": [round(float(goal[0]), 3), round(float(goal[1]), 3)], "step": int(obs.step),
                               "cells": size, "distance_m": round(math.sqrt(distance2) * world.resolution, 2)}
            self.stats["relocations"] += 1
            self._record(dict(self.relocation, action=obs.step, event="relocation_started",
                              falcon_status=self._last_falcon_status()))
            return replace(command, info=dict(command.info, kind="relocate", frontier_cells=size))
        return None

    def _continue_relocation(self, obs, world):
        p = self.policy
        goal = tuple(self.relocation["goal"])
        boundary = self.explorer.frontiers.boundary
        if boundary is not None and not self._frontier_near(world, boundary, goal):
            self._end_relocation(obs, "frontier_gone")
            return None
        if p.route_memory.kind == "relocate" and p.route_memory.arrived(obs):
            self._end_relocation(obs, "arrived")
            return None
        command = p._navigate(obs, world, goal, "relocate")
        if command is None:
            self._retire_goal(goal, obs.step, p.route_memory.reason)
            self._end_relocation(obs, "no route: %s" % p.route_memory.reason)
            return None
        return replace(command, info=dict(command.info, kind="relocate", frontier_cells=self.relocation["cells"]))

    def _end_relocation(self, obs, why):
        p = self.policy
        self.stats["relocations_" + ("arrived" if why == "arrived" else "ended")] += 1
        self._record({"action": obs.step, "event": "relocation_ended", "why": why, "goal": self.relocation["goal"],
                      "actions": int(obs.step) - self.relocation["step"]})
        self.relocation = None
        p.route_memory.clear("relocation_" + ("arrived" if why == "arrived" else "ended"))
        p._route = p._goal = None

    def _frontier_near(self, world, boundary, goal):
        gx, gy = world.world_to_grid(*goal)
        n = int(math.ceil(self.settings.relocation_done_m / world.resolution))
        patch = boundary[max(0, gy - n):gy + n + 1, max(0, gx - n):gx + n + 1]
        return bool(patch.any())

    def _retire_goal(self, goal, step, why):
        self.retired.append(((float(goal[0]), float(goal[1])), int(step)))
        self.stats["relocations_retired"] += 1
        self._record({"action": int(step), "event": "relocation_goal_retired", "why": str(why),
                      "goal": [round(float(goal[0]), 2), round(float(goal[1]), 2)]})

    def _retired_near(self, goal, step):
        s = self.settings
        self.retired = [(xy, at) for xy, at in self.retired if step - at < s.relocation_retire_actions]
        return any(math.dist(xy, goal) <= s.relocation_retire_m for xy, _ in self.retired)

    # -- housekeeping -------------------------------------------------------------------------
    def _sync_floor(self, obs):
        """A storey never planned on (an unplanned height departure): fresh FALCON memories, nothing in force."""
        floor_id = self.policy.mapping.floor_id
        if floor_id == self._floor_id:
            return
        if self._floor_id is not None:
            self.explorer = FalconPlanner(self.params)
            self.current = self.relocation = None
            self.retired = []
            self.stats["floor_changes"] += 1
            self._record({"action": obs.step, "event": "floor_changed", "from": self._floor_id, "to": floor_id})
        self._floor_id = floor_id

    def _record(self, event):
        if len(self.events) < MAX_EVENTS:
            self.events.append(event)
        elif len(self.events) == MAX_EVENTS:
            self.events.append({"event": "events_truncated", "limit": MAX_EVENTS})

    def snapshot(self):
        """What the explorer is doing this action, JSON-ready, for the recording's per-step row and the HUD."""
        return {"mode": ("bootstrap" if self.bootstrap_left else "relocate" if self.relocation is not None
                         else "falcon" if self.current is not None else "idle"),
                "viewpoint": None if self.current is None else {"cell": list(self.current.cell),
                                                                 "yaw_rad": round(float(self.current.yaw), 3),
                                                                 "frontier": int(self.current.frontier)},
                "relocation": self.relocation, "last_falcon_status": self._last_falcon_status(),
                "exhausted_decisions": self.exhausted}

    def diagnostics(self):
        return {"explorer": "falcon-planar-goal-free", "falcon_source": dict(FALCON_SOURCE),
                "scope_radius_m": self.params.scope_radius_m, "settings": asdict(self.settings),
                "stats": dict(self.stats), "frontier_updates": self.explorer.frontiers.updates,
                "visited_viewpoints": len(self.explorer.visited), "rejected_viewpoints": len(self.explorer.rejected),
                "retired_relocations": len(self.retired), "relocation": self.relocation,
                "events": list(self.events), "records": list(self.explorer.records)}


class FalconYoloBaseline:
    """Goal-free planar FALCON until YOLO sees the target, then the full method's closing."""

    name = "sparx-falcon-yolo-baseline"

    def __init__(self, detector, settings=None, segmenter=None, baseline=None):
        self.detector, self.segmenter = detector, segmenter
        self.llm_client = None
        settings = settings or RPTSettings()
        if settings.local_exploration != "falcon":
            settings = replace(settings, local_exploration="falcon")
        self.settings = settings
        self.baseline = baseline or BaselineSettings()
        s = self.settings
        if s.segmentation and segmenter is None:
            raise ValueError("RPTSettings.segmentation is on but no segmenter was given; no silent box fallback")
        # The planner, the converter and the route memory are the full method's, parameter for parameter.
        self.converter_params = ActionConverterParams(heading_hysteresis_deg=1.0)
        self.planner_params = WeightedAStarParams(
            inflate_radius_m=s.preferred_clearance_m, inflate_floor_m=s.body_radius_m,
            detour_factor=s.detour_factor,
            unknown_blocked=True, goal_snap_radius_m=0.3, waypoint_spacing_m=0.25,
            start_skip_m=0.0, search_margin_m=s.map_size_m)
        self.route_settings, self.target_settings = RouteSettings(), TargetEvidenceSettings()
        self.fallback_settings = FallbackSettings()
        self.door_settings = DoorSettings()
        self.sight_settings = SightSettings(enabled=False)    # the walked trail only: no pockets, no settled unknown
        self.target_projector = None       # optional simulator-local geometry, never goal annotations
        self.hierarchy = None              # the building coordinator asks; the baseline has no FALCON/RPT hybrid

    def configuration(self):
        s = self.settings
        return {"method": self.name, "adaptation": asdict(s), "baseline": asdict(self.baseline),
                "planner": asdict(self.planner_params), "converter": asdict(self.converter_params),
                "route_commitment": asdict(self.route_settings), "target_evidence": asdict(self.target_settings),
                "target_closing": asdict(s.target_closing),
                "target_navmesh_projection": self.target_projector is not None,
                "exploration": ("planar FALCON run goal-free: one bounded transaction in a %.1f m disc about the agent, "
                                "its first viewpoint walked and retired, the next transaction; with no useful frontier in "
                                "the disc, A* to the nearest reachable frontier of the storey and FALCON again from there; "
                                "with none left on the storey, %d turns in place and %s"
                                % (s.falcon.scope_radius_m, self.baseline.exhausted_turns,
                                   "STOP" if self.baseline.stop_when_exhausted else "turns until the action limit")),
                "bootstrap_turns": self.baseline.bootstrap_turns,
                "local_exploration": "falcon", "falcon_running": True, "falcon_source": dict(FALCON_SOURCE),
                "llm": None, "room_llm": "none", "node_oracle": "none", "rpt_star": "none", "scene_graph": "none",
                "room_partition": "none", "openings": "none", "doorway_peek": "none", "glances": "none",
                "frontier_sweep": "none", "exploration_fallback": "none (service back-off ledger only)",
                "escape_probe": "none",
                "target_approach": "the full method's TargetClosing takeover and legacy pursuit, unchanged",
                "camera_control": asdict(CameraControlSettings()),
                "perception_fusion": "coherent-depth/floor-qualified-v1" + ("/target-masked" if s.segmentation else ""),
                "segmentation": ({"enabled": True, "scope": "target boxes at or above detection_confidence"}
                                 if s.segmentation else {"enabled": False, "depth": "robust_bbox_depth over the shrunken box"}),
                "ground_truth_semantics": False,
                "ground_truth_stairs": s.multifloor.enabled and s.multifloor.stair_source == "ground_truth",
                "stair_source": s.multifloor.stair_source, "training_free": True,
                "allow_stair_traversal": s.allow_stair_traversal,
                "floor_plane_bound_m": s.floor_plane_bound_m, "step_down_m": s.step_down_m,
                "spawn_floor_confinement": ("the full method's: seen ground-truth stairs and observed drops below the "
                                            "storey plane written occupied on the planning world; goals off the spawn "
                                            "plane refused (methods/spawn_floor_guard.py)"),
                "non_metric": False, "publishable": True}

    # -- the episode ----------------------------------------------------------------------------
    def reset(self, episode, target):
        self.episode, self.target = episode, target
        s = self.settings
        self.mapping = ObservedMap(s.map_size_m, s.map_resolution_m, s.depth_stride, s.body_height_m, s.body_radius_m, s.multifloor)
        self.planner = WeightedAStarPlanner2D(self.planner_params)
        # The LOCKED target's last resort before a boxed-in release (``target_closing.through_unknown``).
        self.optimistic_planner = WeightedAStarPlanner2D(replace(
            self.planner_params, unknown_blocked=False, unknown_cost=s.target_closing.through_unknown_cost))
        self.route_memory = CommittedRoute(episode.action_spec, self.converter_params, self.route_settings)
        # The service back-off ledger and the failure record the perception cycle writes to; its rungs are never run.
        self.fallback = ExplorationFallback(self, self.fallback_settings)
        # Where the agent has stood, per storey: the stair mask never covers it (``peek_stairs.walked_exemption``).
        self.sight = SightLedger(self, self.sight_settings)
        self._reset_floor_context()
        self._route = self._goal = None
        self._blocked = self._plan_calls = self._duplicates_removed = 0
        self._last_pose = None
        self.last_world = None
        self.telemetry = ExplorationMetrics()
        self.camera_control = CameraController(episode.action_spec)
        self.perception = PerceptionCycle(self)
        self.building = None
        if s.multifloor.enabled:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.multifloor_policy import MultiFloorSearch
            self.building = MultiFloorSearch(self)
        self.floor_guard = SpawnFloorGuard(self)
        self.peek = _NoPeek()
        self.closing = TargetClosing(self)
        self.explorer = GoalFreeFalcon(self)
        self._action_owner = "search"
        self._notified_step = self._decision_step = -1
        self._decision_command = None

    def _reset_floor_context(self):
        """The floor-local evidence the perception cycle and the closing read, as ``FloorContextBank.new`` sets it."""
        s = self.settings
        self.doors = ObservedDoors(self.door_settings)
        self.landmarks = ObjectLandmarkMap(dedupe_radius_m=s.landmark_dedupe_radius_m, nearest_match=True,
                                           class_votes=False, footprint_iou=s.landmark_footprint_iou)
        self.target_evidence = TargetEvidence(self.target_settings)
        self._target_xy = self._target_id = self._target_floor_id = None
        self._object_geometry = {}
        self._landmark_best_conf = {}
        self._target_step = -s.target_memory_steps - 1
        self._floor_time = 0.0

    def rotation_turns(self):
        """In-place turns one full look-around takes (``RPTSearchPolicy.rotation_turns``)."""
        k = self.episode.camera.intrinsics
        return camera_rotation_turns(k.width, k.fx, self.episode.action_spec.turn_angle_rad)

    # -- the agent's hooks ---------------------------------------------------------------------
    def notify_blocked(self, observation):
        if self.closing.active:
            self.closing.note_blocked()
            self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
            self._blocked += 1
            self.route_memory.clear("target_forward_blocked")
            self._route = self._goal = None
            return
        if self.building and self.building.traversing:
            self._blocked += 1
            self.building.blocked(observation)
            return
        self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
        self._blocked += 1
        self.route_memory.clear("forward_blocked")
        self._route, self._goal = None, None
        self.explorer.blocked(observation)

    def plan(self, observation):
        if observation.step == self._decision_step:
            return self._decision_command
        self._action_owner = "search"
        started = time.monotonic()
        try:
            command = self._plan(observation)
            phase = str(command.info.get("kind", "falcon"))
            if self.closing.active:
                phase = "target_" + self.closing.phase.lower()
            elif self.building and self.building.phase != "SEARCH":
                phase = self.building.phase
            transition = self.building.transition if self.building else None
            command = self.camera_control.apply(
                observation, command, "TARGET_CLOSING" if self.closing.active else self.building.phase if self.building else "SEARCH",
                transition.direction if transition else 0, transition.close_support if transition else False)
            self.telemetry.phase = phase
            self._decision_step = observation.step
            self._decision_command = replace(command, info=dict(command.info, explorer="falcon", phase=phase,
                                                                floor_id=self.mapping.floor_id))
            return self._decision_command
        finally:
            self.telemetry.latencies["policy_decision"].append((time.monotonic() - started) * 1000)

    def filter_action(self, observation, action):
        if self.closing.active:
            return action  # closing A* and native collision checking, never an exploration mask
        if self.building and self.building.committed:
            return self.building.filter_action(observation, action)
        return self.explorer.filter_action(observation, action)

    def notify_action(self, observation, action):
        if observation.step == self._notified_step:
            return
        if observation.step < self._notified_step:
            raise ValueError("Action ledger regressed")
        self._notified_step = observation.step
        if self.closing.active:
            self.closing.note_action(action)
        self._floor_time += self.settings.action_time_s
        self.telemetry.emitted(observation, action, self.telemetry.phase)

    # -- the decision ---------------------------------------------------------------------------
    def _plan(self, observation):
        self.perception.observe(observation)
        self.closing.observe(observation)
        if self.closing.active:
            self._action_owner = "target_closing"
            # No exploration try block may catch closing failures; the map stays fresh under the takeover.
            try:
                world = self.mapping.update(observation, arrival_allowed=False)
                world = self._confine(observation, world)
                self.last_world = world
                self.sight.observe(observation, world)
                self._perceive(observation)
                return self.closing.plan(observation, world)
            except Exception as exc:
                self.closing.phase = "FAILED"
                if self.closing.failure is None:
                    self.closing.failure = "%s: %s" % (type(exc).__name__, exc)
                raise
        if self.building:
            self.building.prepare_observation(observation)
        started = time.monotonic()
        floor_revision = self.mapping.floor_revision
        transition = self.building.transition if self.building else None
        try:
            world = self.mapping.update(observation, integrate=not (self.building and self.building.traversing),
                                        arrival_allowed=transition.arrival_allowed if transition else True)
        except ValueError as exc:
            # The agent walked within the camera's range of the map's edge: the frame cannot be
            # integrated, but the episode goes on with the map as it stands.
            if self.last_world is None:
                raise
            self.fallback.record_failure(observation, "mapping", exc)
            world = self.last_world
        world = self._confine(observation, world)
        self.telemetry.latencies["mapping"].append((time.monotonic() - started) * 1000)
        if self.mapping.floor_revision != floor_revision:
            # A storey change the guard did not prevent (a step down the atlas read as a floor): the
            # floor-local evidence starts afresh, as the full method's floor contexts do.
            self._reset_floor_context()
            self.route_memory.clear("floor_context_changed")
            self._route = self._goal = None
        self.last_world = world
        if self.building:
            self.building.observe(observation)
        if not self.building or not self.building.traversing:
            self.telemetry.observe(observation, world, self.mapping.floor_id)
            self.sight.observe(observation, world)
        confirmed = self._perceive(observation)
        try:
            return self._decide(observation, world, confirmed)
        except Exception as exc:  # FALCON, A* or a bug in the decision: keep moving, loudly
            self.fallback.record_failure(observation, "decision", exc)
            self.explorer.stats["decision_failures"] += 1
            return NavigationCommand.hold(
                final_yaw=normalize_angle(observation.pose.yaw + self.episode.action_spec.turn_angle_rad),
                info={"kind": "decision_failure_turn", "reason": "decision failure: %s" % type(exc).__name__})

    def _decide(self, observation, world, confirmed):
        if self.building and self.building.committed:
            command = self.building.plan(observation, world)
            if command is not None:
                return command
        inspection = self.camera_control.inspection_command(observation)
        if inspection is not None:
            return inspection
        return self._search(observation, world, confirmed)

    def _search(self, observation, world, confirmed):
        s, pose = self.settings, observation.pose
        self._last_pose = (pose.x, pose.y)
        self.explorer.prepare(observation, world)
        # The legacy pursuit, as the full method runs it under its frontier explorer: a confirmed target
        # landmark the takeover did not start on is walked to and verified in place.
        if confirmed and self._target_xy is not None and math.dist((pose.x, pose.y), self._target_xy) <= s.stop_distance_m:
            return NavigationCommand.stop_here(info={"reason": "fresh multi-view-confirmed target", "target_confirmed": True})
        if self._target_xy is not None and observation.step - self._target_step <= self.target_settings.max_unseen_steps:
            approach = self._approach(observation, world)
            if approach is not None:
                self.explorer.suspend("target_pursuit")
                return approach
        return self.explorer.plan(observation, world)

    def _confine(self, observation, world):
        """The world every decision plans on: seen stairs and observed drops impassable while traversal is forbidden."""
        confined = self.floor_guard.observe(observation, world)
        if confined is not world:
            self.mapping.worlds[self.mapping.floor_id] = confined
        return confined

    def _navigate(self, observation, world, goal, kind, final_yaw=None, planner=None):
        planner = self.planner if planner is None else planner
        if not self.floor_guard.goal_allowed(observation, world, goal, kind):
            self.route_memory.clear("goal off the spawn floor")
            self._route = self._goal = None
            return None
        if not self.route_memory.reusable(observation, world, planner, goal, kind):
            if self.route_memory.reason == "no_progress":
                self._route = self._goal = None
                return None
            path = (self._plan_to(observation, world, goal) if planner is self.planner
                    else self._plan_to(observation, world, goal, planner=planner))
            if path is None:
                self._route = self._goal = None
                return None
            self.route_memory.adopt(path, goal, kind, observation)
        self._route, self._goal = self.route_memory.path, self.route_memory.goal
        info = {"route": self.route_memory.reason, "kind": kind}
        if planner is not self.planner:
            info["through_unknown"] = True
        if self.route_memory.reason == "new_goal_or_invalid_route":
            info["route_replaced"] = self.route_memory.replaced
        return NavigationCommand.follow(self._route, final_yaw=final_yaw, info=info)

    def _perceive(self, observation):
        return self.perception.fuse(observation)

    def _approach(self, observation, world):
        p = observation.pose
        dx, dy = self._target_xy[0] - p.x, self._target_xy[1] - p.y
        distance = math.hypot(dx, dy)
        at_path_end = self.route_memory.kind == "target" and self.route_memory.arrived(observation)
        if distance <= self.settings.stop_distance_m or at_path_end or self.target_evidence.verification_started(self._target_id):
            self.route_memory.clear("target_route_arrived")
            self._route = self._goal = None
            yaw = self.target_evidence.verification_yaw(self._target_id, p.yaw, self.episode.action_spec.turn_angle_rad)
            if yaw is not None:
                return NavigationCommand.hold(final_yaw=yaw, info={"reason": "bounded target verification"})
            self.target_evidence.reject(self._target_id, observation.step)
            self._target_xy = self._target_id = None
            return None
        margin = self.converter_params.goal_tolerance_m + world.resolution + self.planner_params.goal_snap_radius_m
        standoff = max(self.settings.body_radius_m, self.settings.stop_distance_m - margin)
        scale = (distance - standoff) / distance
        command = self._navigate(observation, world, (p.x + dx * scale, p.y + dy * scale), "target", final_yaw=math.atan2(dy, dx))
        if command is None and self.route_memory.reason == "no_progress":
            self.target_evidence.reject(self._target_id, observation.step)
            self._target_xy = self._target_id = None
        return command

    def _plan_to(self, observation, world, goal, planner=None):
        p = observation.pose
        planner = self.planner if planner is None else planner
        self._plan_calls += 1
        started = time.monotonic()
        result = planner.plan(PlanRequest(Pose2D(p.x, p.y, p.yaw), Pose2D(*goal), frame_id="world"), world)
        self.telemetry.latencies["astar"].append((time.monotonic() - started) * 1000)
        if not result.ok:
            return None
        return result.path

    # -- the record ------------------------------------------------------------------------------
    def _building_diagnostics(self):
        """The coordinator's record without its room-coverage and floor-context readings (the baseline has neither)."""
        b = self.building
        if b is None:
            return None
        return {"phase": b.phase, "stair_source": b.params.stair_source, "allow_stair_traversal": b.allowed,
                "ground_truth": b.ground_truth.diagnostics() if b.ground_truth is not None else None,
                "stairs_seen": b.sightings.diagnostics(), "atlas": self.mapping.atlas.diagnostics(),
                "transition": b.transition.diagnostics() if b.transition else None,
                "completed_transitions": list(b.completed), "safety_vetoes": b.safety_vetoes,
                "events": list(b.events),
                "portals": [dict({k: v for k, v in q.items() if k not in ("path", "footprint")},
                                 footprint_points=len(q.get("footprint", ()))) for q in b.portals],
                "active_portal": b.active["id"] if b.active else None,
                "decides_floor_changes": "never: stair traversal forbidden; stairs are seen, placed and masked"}

    def episode_info(self):
        return {"method": self.name, "landmarks": len(self.landmarks),
                "blocked": self._blocked, "plan_calls": self._plan_calls, "duplicates_removed": self._duplicates_removed,
                "route_commitment": dict(self.route_memory.stats),
                "target_evidence": self.target_evidence.diagnostics(), "floor_revisions": self.mapping.floor_revision,
                "target_closing": self.closing.diagnostics(),
                "perception": self.perception.diagnostics(), "camera": dict(self.camera_control.last),
                "floor_maps": self.mapping.integrity(),
                "building": self._building_diagnostics(),
                "spawn_floor_guard": self.floor_guard.diagnostics(),
                "doors": self.doors.diagnostics(),
                "exploration": self.explorer.diagnostics(),
                "exploration_fallback": self.fallback.diagnostics(),
                "exploration_metrics": self.telemetry.report(), "hierarchy": None}
