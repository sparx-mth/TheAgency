"""The greedy look planner: turn here or walk to where the camera learns more, by expected useful area per action.

One machine for two uses. The **vantage warm-up**
(:mod:`~sparx_agency.core.planning.object_search.exploration.vantage_warmup`)
runs it unscoped at the episode's start within an action budget; the
**viewpoint explore visit** (``LoopSettings.explore_goals == "viewpoint"``,
:mod:`~sparx_agency.core.planning.object_search.rooms.room_search_loop`) runs
it scoped to the room in force, within the visit's bound. Both are MapEx's
viewpoint selection (raycast from a candidate, count the unknown it would
see, divide by the cost to get there; Kim, Ho et al. 2024, arXiv 2409.15590,
whose observed-map ablation needs no learned map) with this camera's sensor
model (:mod:`~sparx_agency.core.planning.exploration.viewpoint_gain`) and the
cost in the benchmark's own actions
(:mod:`~sparx_agency.core.planning.exploration.viewpoint_candidates`).

Every action, standing at a *spot*:

1. **The next sweep here.** The frame at the current heading has already gone
   into the map, and every frame taken from this spot since the agent arrived
   is remembered as an arc of bearings. A turn option is a sweep of ``k``
   turns one way, valued at its best ``k`` -- the expected area in the part
   of the swept cones no frame from this spot has covered, over ``k`` actions.
   The next sliver alone may show a wall while the two behind it show the
   room; a greedy that read one sliver at a time ended there.
2. **A walk to a candidate.** Reachable cells on a lattice within
   ``max_travel_m`` along the floor (and within what is left of the budget),
   at least ``min_move_m`` from every spot looked around from, inside the
   caller's scope when it gives one, plus any extra points the caller names
   (a room's frontier goals); each is valued by the better of two looks from
   there -- one frame at its best heading, or the whole circle -- over the
   actions it costs: the forward steps along the geodesic, the turn that
   faces the way, the route's own bends and the turns the look takes. At
   most ``max_points`` such walks when the caller bounds them.
3. **The choice.** The option of greatest rate, if it clears
   ``min_rate_m2_per_action``; otherwise the planner ends (``ended``) and
   the caller spends its actions elsewhere. A chosen walk is committed until
   arrival (the route planned and walked by the policy's own ``_navigate``
   and lattice follower, on the caller's planning map), a refused or
   stalled route drops the candidate, and arrival opens a new spot.

A spot is *complete* when the frames covered its circle to within half a
turn, or when the uncovered arc promises no more than ``scan_skip_gain_m2``;
the caller is told when a spot is closed (``on_spot_closed``) and decides
what that means -- the warm-up records a completed scan, the explore visit
nothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Optional

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.viewpoint_candidates import (
    ActionPricing, Viewpoint, geodesic_field, sample_viewpoints)
from sparx_agency.core.planning.exploration.viewpoint_gain import (
    ExpectedView, arc_uncovered, near_floor_mask, sensor_from_camera)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand

TURN_LEFT, TURN_RIGHT, MOVE = "turn_left", "turn_right", "move"


@dataclass(frozen=True)
class LookSettings:
    """What a look is worth and what it costs. Distances in metres, counts in actions.

    Attributes:
        detection_range_m: How far an object can be detected by the
            perception in force -- the range the gain is counted to, not
            the depth sensor's clip (``SensorModel.detection_range_m``).
            3.5 m by default, the method's own far limit for a clear view of
            a landmark (``RPTSettings.landmark_miss_max_range_m``) and the
            object-inclusion range 3D-Mem ablated to (ApexNav reads its
            frontiers as seen within 4.0 m of a 4.99 m depth clip); the
            ground-truth perception detects to the depth range, 5 m, and
            the studies' arms set that.
        unknown_free_run_m: The survival prior on unknown space
            (``SensorModel.unknown_free_run_m``): a ray that has crossed
            ``d`` metres of unknown reaches the next cell with probability
            ``exp(-d / this)``; None is MapEx's transparent unknown.
        near_horizon_m: Floor within this of a pose the agent stood at,
            inside the frame's cone, is as good as seen: a level camera
            0.88 m up sees the floor from about 1.4 m out, so the unknown
            ring under its feet is floor nothing stands on (anything standing
            there was mapped), not area a walk would reveal.
        candidate_spacing_m: The lattice pitch the candidates are sampled on.
        min_move_m: No candidate within this of a spot already looked around
            from (a shorter move shows the same room).
        max_travel_m: How far along the floor a candidate may lie.
        min_rate_m2_per_action: The least expected useful area per action an
            option must promise to be taken; below it the planner ends. 0
            takes the best option whatever it promises.
        scan_skip_gain_m2: A spot whose uncovered arc promises no more than
            this is a completed look-around even though the body did not
            turn through it. 1 m2: half of what one turn into open unknown
            promises under the default prior (1.8 m2), so an arc left
            unswept because every bearing in it meets a wall within a metre
            or two completes the spot, and one with an unseen room behind it
            does not.
        plan_attempts: Candidates the planner is asked about per action
            before the turn option stands in.
        max_points: How many walks the planner may take; None is unbounded
            (the explore visit, bounded by the visit's own actions).
    """

    detection_range_m: float = 3.5
    unknown_free_run_m: Optional[float] = 2.0
    near_horizon_m: float = 1.4
    candidate_spacing_m: float = 0.5
    min_move_m: float = 0.75
    max_travel_m: float = 4.0
    min_rate_m2_per_action: float = 0.25
    scan_skip_gain_m2: float = 1.0
    plan_attempts: int = 3
    max_points: Optional[int] = None

    def __post_init__(self):
        for name in ("max_travel_m", "candidate_spacing_m", "min_move_m", "detection_range_m", "near_horizon_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("look.%s must be positive and finite" % name)
        for name in ("min_rate_m2_per_action", "scan_skip_gain_m2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("look.%s must be finite and non-negative" % name)
        if type(self.plan_attempts) is not int or self.plan_attempts < 1:
            raise ValueError("look.plan_attempts must be a positive integer")
        if self.max_points is not None and (type(self.max_points) is not int or self.max_points < 0):
            raise ValueError("look.max_points must be None or a non-negative integer")
        if self.unknown_free_run_m is not None and (
                isinstance(self.unknown_free_run_m, bool) or not math.isfinite(self.unknown_free_run_m)
                or self.unknown_free_run_m <= 0):
            raise ValueError("look.unknown_free_run_m must be None or positive and finite")


class GreedyLookPlanner:
    """The greedy look of the module docstring, as a component of the policy.

    Args:
        policy: The composition root (``episode``, ``sight``, ``graph``,
            ``route_memory``, ``_navigate``).
        settings: The look's knobs.
        kind: The route kind the walks are navigated under (``"warmup"``,
            ``"viewpoint"``): the policy's goal-retirement and bump rules
            read it, and the recording shows it.
        on_spot_closed: Optional callback ``(obs, world, spot)`` when a
            spot is left or the planner ends; ``spot["complete"]`` says
            whether its look-around is finished.
    """

    def __init__(self, policy, settings, kind, on_spot_closed=None):
        self.policy = policy
        self.settings = settings
        self.kind = kind
        self.on_spot_closed = on_spot_closed
        self.reset()

    def reset(self):
        self.spots = []           # dicts: xy, step, arcs [(start, end)], complete, closed
        self.goal = None          # the committed candidate's xy while a walk is in force
        self.goal_step = None
        self.moves = 0
        self.failed = []          # candidate xy the planner refused or the walk stalled on
        self.events = []
        self.decisions = []       # one record per action: the options weighed and the choice
        self.ended = None         # why the planner ended, once it has
        self.stats = {"turns": 0, "moves": 0, "move_actions": 0, "arrivals": 0, "walk_failures": 0,
                      "early_end": 0, "candidates": 0}
        self._view_cache = None

    # -- the decision --------------------------------------------------------------
    def plan(self, obs, world, cost, remaining, within=None, planning_world=None, extra=()):
        """One action, or None when nothing is worth one (``ended`` says why) or the budget is spent.

        Args:
            obs: The frame; its map has already been integrated.
            world: The observed map the gain is read on.
            cost: The passable cost for the geodesic distances (None: the
                scene graph's frontier inventory alone).
            remaining: Actions the caller still allows.
            within: Optional ``(H, W)`` bool scope for the candidates.
            planning_world: The map routes are planned on (``world`` when
                None): the explore visit's room-confined copy.
            extra: Extra candidate points ``(x, y)`` the caller names.
        """
        p, s = self.policy, self.settings
        self.ended = None
        here = (float(obs.pose.x), float(obs.pose.y))
        yaw = float(obs.pose.yaw)
        turn = float(p.episode.action_spec.turn_angle_rad)
        half = float(self.sensor().half_fov_rad)
        remaining = int(remaining)
        if remaining <= 0:
            self.ended = "budget"
            return None
        if self.goal is not None:
            command = self._walk(obs, world, planning_world, remaining)
            if command is not None:
                return command
        spot = self._spot(obs, world, here)
        spot["arcs"].append((yaw - half, yaw + half))
        view = self.view(obs, world)
        profile = view.profile(here)
        covered = self.covered_rays(view, spot["arcs"])
        options = []
        circle_turns = self.sensor().rotation_turns(turn)
        for name, signed in ((TURN_LEFT, 1.0), (TURN_RIGHT, -1.0)):
            swept = np.zeros(view.rays, dtype=bool)
            best = dict(kind=name, gain=0.0, actions=1, rate=0.0, turns=1)
            for k in range(1, min(circle_turns, remaining) + 1):
                swept |= view.sector_mask(yaw + signed * k * turn, half)
                gain = float(profile[swept & ~covered].sum())
                if gain / k > best["rate"]:
                    best = dict(kind=name, gain=gain, actions=k, rate=gain / k, turns=k)
            options.append(best)
        candidates = []
        if (s.max_points is None or self.moves < s.max_points) and remaining >= 2:
            candidates = self._candidates(obs, world, cost, view, remaining, within, extra)
            options.extend(candidates)
        options.sort(key=lambda o: -o["rate"])
        best = options[0] if options else None
        record = dict(step=int(obs.step), spot=len(self.spots), remaining=remaining,
                      turn_left=round(rate_of(options, TURN_LEFT), 3), turn_right=round(rate_of(options, TURN_RIGHT), 3),
                      candidates=len(candidates),
                      best_candidate=None if not candidates else {
                          k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in max(candidates, key=lambda o: o["rate"]).items()})
        if best is None or best["rate"] < s.min_rate_m2_per_action:
            return self._end(obs, world, record, "nothing worth an action", best)
        if best["kind"] == MOVE:
            for option in [o for o in options if o["kind"] == MOVE][:s.plan_attempts]:
                if option["rate"] < s.min_rate_m2_per_action:
                    break
                command = self._begin_walk(obs, world, planning_world, option, remaining)
                if command is not None:
                    record["choice"] = dict(kind=MOVE, xy=[round(v, 2) for v in option["xy"]], gain=round(option["gain"], 3),
                                            actions=round(option["actions"], 1), rate=round(option["rate"], 3), look=option["look"])
                    self.decisions.append(record)
                    return command
            best = max((o for o in options if o["kind"] != MOVE), key=lambda o: o["rate"], default=None)
            if best is None or best["rate"] < s.min_rate_m2_per_action:
                return self._end(obs, world, record, "walks refused, no turn worth an action", best)
        record["choice"] = dict(kind=best["kind"], gain=round(best["gain"], 3), rate=round(best["rate"], 3),
                                turns=best["turns"])
        self.decisions.append(record)
        self.stats["turns"] += 1
        signed = 1.0 if best["kind"] == TURN_LEFT else -1.0
        return NavigationCommand.hold(final_yaw=normalize_angle(yaw + signed * turn),
                                      info={"kind": self.kind, "choice": best["kind"], "sweep_turns": best["turns"],
                                            "sweep_gain_m2": round(best["gain"], 3),
                                            "rate_m2_per_action": round(best["rate"], 3), "spot": len(self.spots)})

    def _end(self, obs, world, record, reason, best):
        record["choice"] = "end"
        self.decisions.append(record)
        self.stats["early_end"] += 1
        self.ended = reason
        self._log(obs, "look_end", reason=reason, best_rate=None if best is None else round(best["rate"], 3))
        self.close(obs, world)
        return None

    # -- the walk ------------------------------------------------------------------------
    def _begin_walk(self, obs, world, planning_world, option, remaining):
        command = self._route(obs, world, planning_world, tuple(option["xy"]))
        if command is None:
            self.failed.append(tuple(option["xy"]))
            self.stats["walk_failures"] += 1
            self._log(obs, "look_walk_refused", xy=[round(v, 2) for v in option["xy"]])
            return None
        self.goal, self.goal_step = tuple(option["xy"]), int(obs.step)
        self.moves += 1
        self.stats["moves"] += 1
        self._log(obs, "look_walk", xy=[round(v, 2) for v in option["xy"]], gain=round(option["gain"], 3),
                  actions=round(option["actions"], 1), rate=round(option["rate"], 3), look=option["look"],
                  geodesic_m=round(option["geodesic_m"], 2))
        return replace(command, info=dict(command.info, kind=self.kind, choice=MOVE))

    def _walk(self, obs, world, planning_world, remaining):
        """The committed walk's next action; None once arrived, refused, stalled or out of budget (the goal is dropped)."""
        p = self.policy
        here = (float(obs.pose.x), float(obs.pose.y))
        arrived = (p.route_memory.kind == self.kind and p.route_memory.arrived(obs)) or math.dist(here, self.goal) <= 0.35
        if arrived:
            self.stats["arrivals"] += 1
            self._log(obs, "look_arrived", xy=[round(v, 2) for v in self.goal], actions=int(obs.step) - int(self.goal_step))
            self._drop_goal("%s_arrived" % self.kind)
            return None
        if math.dist(here, self.goal) / float(p.episode.action_spec.forward_step_m) + 1 > remaining:
            # A takeover walked the agent away mid-walk, or the route grew: the budget left cannot reach the
            # point and look once. The walk is dropped, not finished for its own sake.
            self.failed.append(tuple(self.goal))
            self.stats["walk_failures"] += 1
            self._log(obs, "look_walk_dropped", xy=[round(v, 2) for v in self.goal], reason="out of budget")
            self._drop_goal("%s_out_of_budget" % self.kind)
            return None
        command = self._route(obs, world, planning_world, self.goal)
        if command is None:
            self.failed.append(tuple(self.goal))
            self.stats["walk_failures"] += 1
            self._log(obs, "look_walk_dropped", xy=[round(v, 2) for v in self.goal], reason=p.route_memory.reason)
            self.goal = self.goal_step = None
            return None
        self.stats["move_actions"] += 1
        return replace(command, info=dict(command.info, kind=self.kind, choice=MOVE))

    def _route(self, obs, world, planning_world, goal):
        """The policy's route to ``goal`` on the planning map, else on the observed map when the confinement walled it off."""
        p = self.policy
        command = p._navigate(obs, planning_world if planning_world is not None else world, goal, self.kind)
        if command is None and planning_world is not None and planning_world is not world:
            command = p._navigate(obs, world, goal, self.kind)
        return command

    def _drop_goal(self, reason):
        p = self.policy
        self.goal = self.goal_step = None
        p.route_memory.clear(reason)
        p._route = p._goal = None

    # -- candidates ---------------------------------------------------------------------------
    def _candidates(self, obs, world, cost, view, remaining, within, extra):
        """Every reachable lattice point (and extra point) within reach of the budget, valued by its best look per action."""
        p, s = self.policy, self.settings
        here = (float(obs.pose.x), float(obs.pose.y))
        pricing = ActionPricing(float(p.episode.action_spec.forward_step_m), float(p.episode.action_spec.turn_angle_rad))
        inventory = getattr(p.graph, "frontier_inventory", None)
        distance = getattr(inventory, "distance_m", None)
        if distance is None or distance.shape != world.grid.shape:
            if cost is None:
                return []
            distance = geodesic_field(world, cost, here, s.max_travel_m)
        reach = min(s.max_travel_m, max(0.0, (remaining - 1)) * pricing.forward_step_m)
        avoid = [tuple(spot["xy"]) for spot in self.spots] + list(self.failed)
        points = sample_viewpoints(world, distance, here, float(obs.pose.yaw), s.candidate_spacing_m, reach,
                                   min_distance_m=s.min_move_m, avoid=avoid, avoid_radius_m=s.min_move_m, within=within)
        points += self._extra_points(world, distance, here, float(obs.pose.yaw), reach, avoid, extra)
        self.stats["candidates"] += len(points)
        turn = pricing.turn_rad
        circle_turns = self.sensor().rotation_turns(turn)
        headings = [-math.pi + k * turn for k in range(int(round(2.0 * math.pi / turn)))]
        options = []
        for point in points:
            travel = pricing.travel_actions(point.geodesic_m, point.heading_error_rad)
            if travel + 1 > remaining:
                continue
            profile = view.profile(point.xy)
            circle = float(profile.sum())
            frames = view.frame_gains(profile, headings)
            arrival = math.atan2(point.xy[1] - here[1], point.xy[0] - here[0])
            best_frame, best_rate_frame, best_heading = 0.0, 0.0, None
            for heading, gain in zip(headings, frames):
                actions = travel + pricing.turns(normalize_angle(heading - arrival))
                if actions > remaining or gain <= 0.0:
                    continue
                rate = gain / max(actions, 1.0)
                if rate > best_rate_frame:
                    best_frame, best_rate_frame, best_heading = float(gain), rate, heading
            option = dict(kind=MOVE, xy=point.xy, geodesic_m=point.geodesic_m, look="frame", gain=best_frame,
                          actions=travel + (pricing.turns(normalize_angle(best_heading - arrival)) if best_heading is not None else 0),
                          rate=best_rate_frame, heading=best_heading)
            circle_actions = travel + circle_turns
            if circle_actions <= remaining and circle > 0.0 and circle / circle_actions > option["rate"]:
                option = dict(kind=MOVE, xy=point.xy, geodesic_m=point.geodesic_m, look="circle", gain=circle,
                              actions=circle_actions, rate=circle / circle_actions, heading=None)
            if option["rate"] > 0.0:
                options.append(option)
        return options

    def _extra_points(self, world, distance, here, yaw, reach, avoid, extra):
        """The caller's own points as viewpoints, when reachable within reach and clear of the spots and failures."""
        out = []
        s = self.settings
        for xy in extra:
            x, y = float(xy[0]), float(xy[1])
            gx, gy = world.world_to_grid(x, y)
            if not world.in_bounds(gx, gy):
                continue
            geodesic = float(distance[gy, gx])
            if not math.isfinite(geodesic) or geodesic > reach or geodesic < s.min_move_m:
                continue
            if any(math.dist((x, y), a) <= s.min_move_m for a in avoid):
                continue
            error = normalize_angle(math.atan2(y - here[1], x - here[0]) - yaw)
            out.append(Viewpoint(xy=(x, y), cell=(int(gx), int(gy)), geodesic_m=geodesic, heading_error_rad=float(error)))
        return out

    # -- spots ---------------------------------------------------------------------------------------------
    def _spot(self, obs, world, here):
        """The spot the agent stands at: the last one while within half a metre of it, else a new one."""
        if self.spots and math.dist(here, self.spots[-1]["xy"]) <= 0.5:
            return self.spots[-1]
        if self.spots:
            self._close_spot(obs, world, self.spots[-1])
        spot = dict(xy=here, step=int(obs.step), arcs=[], complete=False, closed=False)
        self.spots.append(spot)
        self._log(obs, "look_spot", index=len(self.spots) - 1, xy=[round(v, 2) for v in here])
        return spot

    def _close_spot(self, obs, world, spot):
        """Leaving ``spot`` (or ending): judge its look-around complete or not and tell the caller."""
        if spot.get("closed") or not spot["arcs"]:
            return
        spot["closed"] = True
        p = self.policy
        world = world if world is not None else p.last_world
        uncovered = arc_uncovered(spot["arcs"], 0.0, math.pi)
        turn = float(p.episode.action_spec.turn_angle_rad)
        complete = uncovered <= 0.5 * turn + 1e-9
        remaining_gain = None
        if not complete and world is not None:
            view = self.view(obs, world)
            covered = self.covered_rays(view, spot["arcs"])
            remaining_gain = float(view.profile(tuple(spot["xy"]))[~covered].sum())
            complete = remaining_gain <= self.settings.scan_skip_gain_m2
        spot["uncovered_deg"] = round(math.degrees(uncovered), 1)
        spot["remaining_gain_m2"] = None if remaining_gain is None else round(remaining_gain, 3)
        spot["complete"] = bool(complete)
        if self.on_spot_closed is not None:
            self.on_spot_closed(obs, world, spot)

    def close(self, obs, world):
        """The caller is done with the planner for now: the current frame joins the spot in force, and it is closed."""
        here = (float(obs.pose.x), float(obs.pose.y))
        if self.spots and not self.spots[-1].get("closed") and math.dist(here, self.spots[-1]["xy"]) <= 0.5:
            half = float(self.sensor().half_fov_rad)
            self.spots[-1]["arcs"].append((float(obs.pose.yaw) - half, float(obs.pose.yaw) + half))
        if self.spots:
            self._close_spot(obs, world, self.spots[-1])
        if self.goal is not None:
            self._drop_goal("%s_over" % self.kind)

    # -- the sensor and the rays --------------------------------------------------------------------------
    def sensor(self):
        p, s = self.policy, self.settings
        k = p.episode.camera.intrinsics
        return sensor_from_camera(k.width, k.fx, s.detection_range_m, float(p.episode.camera.min_depth_m), s.unknown_free_run_m)

    def view(self, obs, world):
        """The ray caster over this action's map, built once per action."""
        cached = self._view_cache
        if cached is not None and cached[0] == int(obs.step) and cached[1] is world:
            return cached[2]
        p, s = self.policy, self.settings
        sight = getattr(p, "sight", None)
        resolved = sight.resolved(world) if sight is not None else None
        sensor = self.sensor()
        poses = [(x, y, yaw) for _, x, y, yaw, _ in sight.poses()] if sight is not None else []
        poses.append((float(obs.pose.x), float(obs.pose.y), float(obs.pose.yaw)))
        ignore = near_floor_mask(world, poses, s.near_horizon_m, sensor.half_fov_rad)
        view = ExpectedView(world, sensor, resolved=resolved, ignore=ignore)
        self._view_cache = (int(obs.step), world, view)
        return view

    @staticmethod
    def covered_rays(view, arcs):
        """``(rays,)`` bool over ``view.bearings``: the bearings some frame from the spot has covered."""
        covered = np.zeros(view.rays, dtype=bool)
        two_pi = 2.0 * math.pi
        for start, end in arcs:
            covered |= ((view.bearings - float(start)) % two_pi) <= (float(end) - float(start)) + 1e-9
        return covered

    def _log(self, obs, event, **fields):
        self.events.append(dict(step=int(obs.step), event=event, **fields))

    def diagnostics(self):
        return {"settings": asdict(self.settings), "kind": self.kind, "stats": dict(self.stats), "moves": self.moves,
                "ended": self.ended,
                "spots": [dict({k: (list(v) if isinstance(v, tuple) else v) for k, v in spot.items() if k != "arcs"},
                               frames=len(spot["arcs"])) for spot in self.spots],
                "goal": None if self.goal is None else [round(v, 2) for v in self.goal],
                "events": list(self.events), "decisions": list(self.decisions)}


def rate_of(options, kind):
    """The rate of the option of ``kind`` in ``options`` (0.0 when absent)."""
    for option in options:
        if option["kind"] == kind:
            return float(option["rate"])
    return 0.0
