"""Informative glances along a route: where to stop and look, and the look itself.

The map is observed free space, and the camera's cone is narrow: an agent
walking a corridor sees the corridor. Doors on either side pass through
the edge of the frame or not at all, and the recordings of 2026-10-04 show
it -- at Hanson's action 35 an open door stood a step to the left of a
committed route and was never looked into. Turning in place costs no path
length (nothing under SPL), but under the 500-action budget a full circle
every few steps is unaffordable, so the question is *where* along the route
a look pays: a step too early and the door frame hides the room behind it,
a step too late and the frame hides it again; a left look is six actions, a
look to both sides twelve -- the price of a full circle, which sees more.

This module schedules at most one such **glance** on the route in force:

1. **Candidates** are the route's points ahead of the agent, one every
   ``stride_m`` up to ``horizon_m``, each with the route's heading there.
2. **Gain** per candidate is the unknown floor a glance from there would
   sweep -- left, right or the full circle
   (:func:`~sparx_agency.core.planning.exploration.view_gain.glance_gains`)
   -- minus what the walk reveals anyway: the forward cones along the whole
   route ahead. A doorway's room counts once, at the point it shows best.
3. **Choice**: the candidate and kind of greatest gain per action, if the
   gain clears ``min_gain_m2`` and the rate ``min_gain_per_action_m2``; a
   side glance costs the turns out and back, a circle the full turn count.
4. **The look** begins when the agent reaches the chosen point: one in-place
   turn per action, toward the side (and back, which the route follower
   does by itself when the glance ends) or all the way round, read off the
   measured yaw as the room scan's rotation is. It is a *suspended phase*
   (``discovery.SUSPENDED_PHASES``): the room-search loop is not ticked and
   not charged, the supervisor's clocks pause, and the committed route's
   watchdogs are told how long the pause was.

A target sighting or a committed stair traversal ends a glance at once;
the warm-up, a peek's look, a scan's rotation and the target takeover never
start one. The candidates are re-scored every ``revalue_actions`` and
whenever the route changes, and the map's growth as the agent walks moves
the best point with it -- the glance is taken where it is worth most *now*,
not where it was worth most three actions ago.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.view_gain import UnknownView, cone_from_camera, glance_gains
from sparx_agency.core.planning.objnav.types.command import NavigationCommand

LEFT, RIGHT, FULL = "left", "right", "full"


@dataclass(frozen=True)
class GlanceSettings:
    """What a glance is worth and when one is taken. Distances in metres, counts in actions.

    Attributes:
        enabled: Schedule glances at all (the ablation is ``False``).
        horizon_m: How far ahead along the route candidates are scored.
        stride_m: Spacing of candidate points along the route.
        arrival_m: Within this of the chosen point the glance begins.
        min_gain_m2: Least unknown floor a glance must promise beyond what
            the walk reveals.
        min_gain_per_action_m2: ... and per action it costs: a six-action
            side glance needs 3 m2 at the default, a twelve-action circle
            6 m2. The bar is set where walking stops being the better buy:
            six forward steps into unknown space sweep the cone over
            1.5 m of new floor, around 10 m2 in a room; a doorway beside
            the route shows nothing to the walk and 5-15 m2 to a glance
            (the Hanson routes of 2026-10-04 replayed on their final maps:
            14.8 m2 at action 33, 7.6 m2 at 164).
        side_rad: How far a side glance turns (a right angle).
        cooldown_actions: After a glance, no new one for this many actions.
        revalue_actions: Candidates are re-scored at most this often while
            the route stands; a new route is scored at once.
        max_range_m: The farthest a ray is followed; the camera's depth
            range when None.
        blind_m: The floor under the camera that no glance reveals.
        scanned_m: No glance within this of a completed full rotation (the
            warm-up, a room scan -- the scan ledger): whatever is still
            unknown from there is beyond range or behind furniture, and a
            second look from the same spot reveals none of it.
    """

    enabled: bool = True
    horizon_m: float = 6.0
    stride_m: float = 0.5
    arrival_m: float = 0.35
    min_gain_m2: float = 3.0
    min_gain_per_action_m2: float = 0.5
    side_rad: float = math.pi / 2
    cooldown_actions: int = 8
    revalue_actions: int = 3
    max_range_m: Optional[float] = 5.0
    blind_m: float = 0.6
    scanned_m: float = 1.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("glances.enabled must be a bool")
        for name in ("horizon_m", "stride_m", "arrival_m", "min_gain_m2", "min_gain_per_action_m2", "side_rad", "blind_m",
                     "scanned_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("glances.%s must be positive and finite" % name)
        for name in ("cooldown_actions", "revalue_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("glances.%s must be a positive integer" % name)
        if self.max_range_m is not None and (not math.isfinite(self.max_range_m) or self.max_range_m <= self.blind_m):
            raise ValueError("glances.max_range_m must exceed blind_m")
        if self.side_rad > math.pi:
            raise ValueError("glances.side_rad must be at most pi")


@dataclass(frozen=True)
class GlanceCandidate:
    """One scored point of the route.

    Attributes:
        index: The candidate's position in the resampled route (0 = here).
        xy: Where the glance would stand.
        heading: The route's heading there.
        along_m: Route distance from the agent.
        left_m2 / right_m2 / full_m2: The gains.
    """

    index: int
    xy: Tuple[float, float]
    heading: float
    along_m: float
    left_m2: float
    right_m2: float
    full_m2: float


@dataclass(frozen=True)
class GlancePlan:
    """The glance chosen on the route in force."""

    candidate: GlanceCandidate
    kind: str
    gain_m2: float
    actions: int

    @property
    def rate(self) -> float:
        return self.gain_m2 / max(1, self.actions)


class GlanceScheduler:
    """Scores the route in force for a glance, and performs the one it chose.

    Attributes:
        plan_in_force: The :class:`GlancePlan` waiting for the agent to reach
            its point, or None.
        active: The glance being performed, or None.
        events: One entry per scheduled, started, completed or aborted glance.
        stats: Counters for the record.
    """

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or GlanceSettings()
        self.plan_in_force: Optional[GlancePlan] = None
        self.active: Optional[Dict] = None
        self.candidates: Tuple[GlanceCandidate, ...] = ()
        self.events: List[Dict] = []
        self.stats = {"evaluations": 0, "scheduled": 0, "started": 0, "completed": 0, "aborted": 0,
                      "turns": 0, "gain_m2": 0.0}
        self._last_eval_step = -10 ** 9
        self._last_glance_step = -10 ** 9
        self._route_key = None

    # -- the wrapper ----------------------------------------------------------
    def apply(self, obs, world, command):
        """The command for this action: a glance's turn, or ``command`` as the search produced it."""
        if not self.settings.enabled or command is None or command.stop:
            return command
        if self.active is not None:
            return self.continue_look(obs) or command
        if not command.waypoints:
            return command
        if not self._may_glance(obs):
            return command
        route = self._route_points()
        if route is None:
            return command
        key = (tuple(round(v, 2) for v in route[-1]), len(route))
        if key != self._route_key or obs.step - self._last_eval_step >= self.settings.revalue_actions:
            self._route_key = key
            self._evaluate(obs, world, route)
        plan = self.plan_in_force
        if plan is None:
            return command
        here = (float(obs.pose.x), float(obs.pose.y))
        if math.dist(here, plan.candidate.xy) <= self.settings.arrival_m or self._passed(route, here, plan):
            return self._start(obs, plan)
        return command

    def continue_look(self, obs):
        """The next turn of the glance in force, or None the action it is complete (the search resumes at once)."""
        if self.active is None:
            return None
        return self._continue(obs)

    def abort(self, obs, reason):
        """End the glance in force, if any, without finishing it (a target sighting, a stair traversal)."""
        if self.active is None:
            return
        self.stats["aborted"] += 1
        self._log(obs, "glance_aborted", kind=self.active["kind"], reason=reason, turns=self.active["turns"])
        self._finish_pause(self.active["turns"])
        self.active = None
        self.plan_in_force = None
        self._last_glance_step = int(obs.step)

    # -- gating ---------------------------------------------------------------
    def _may_glance(self, obs):
        p = self.policy
        if getattr(p, "_action_owner", "search") != "search":
            return False                                           # warm-up, a peek, a return leg
        if getattr(p, "warmup_actions", 0) < getattr(p.settings, "warmup_steps", 0):
            return False
        closing = getattr(p, "closing", None)
        if closing is not None and getattr(closing, "active", False):
            return False
        if getattr(p, "_target_xy", None) is not None:
            return False
        building = getattr(p, "building", None)
        if building is not None and getattr(building, "traversing", False):
            return False
        if obs.step - self._last_glance_step < self.settings.cooldown_actions:
            return False
        return True

    def _route_points(self):
        path = getattr(self.policy.route_memory, "path", None)
        if path is None:
            return None
        points = [(float(q.x), float(q.y)) if hasattr(q, "x") else (float(q[0]), float(q[1]))
                  for q in getattr(path, "points", path)]
        return points if len(points) >= 2 else None

    # -- scoring --------------------------------------------------------------
    def _view(self, obs, world):
        camera = obs.camera
        k = camera.intrinsics
        max_range = self.settings.max_range_m
        if max_range is None or not math.isfinite(max_range):
            max_range = float(camera.max_depth_m)
        max_range = min(max_range, float(camera.max_depth_m)) if math.isfinite(camera.max_depth_m) else max_range
        return UnknownView(world, cone_from_camera(k.width, k.fx, max_range, self.settings.blind_m))

    def _resample(self, obs, route):
        """Candidate points: the agent's position, then one every ``stride_m`` along the route ahead."""
        here = (float(obs.pose.x), float(obs.pose.y))
        nearest = min(range(len(route)), key=lambda i: math.dist(route[i], here))
        ahead = [here] + list(route[nearest + 1:])
        out = []
        along, since = 0.0, self.settings.stride_m
        for i in range(len(ahead)):
            if i > 0:
                seg = math.dist(ahead[i - 1], ahead[i])
                along += seg
                since += seg
            if along > self.settings.horizon_m:
                break
            if i == 0 or since >= self.settings.stride_m - 1e-9:
                nxt = ahead[i + 1] if i + 1 < len(ahead) else None
                prv = ahead[i - 1] if i > 0 else None
                if nxt is not None and math.dist(nxt, ahead[i]) > 1e-6:
                    heading = math.atan2(nxt[1] - ahead[i][1], nxt[0] - ahead[i][0])
                elif prv is not None and math.dist(prv, ahead[i]) > 1e-6:
                    heading = math.atan2(ahead[i][1] - prv[1], ahead[i][0] - prv[0])
                else:
                    heading = float(obs.pose.yaw)
                out.append((len(out), ahead[i], heading, along))
                since = 0.0
        return out

    def _evaluate(self, obs, world, route):
        """Score every candidate and keep the best glance, or none."""
        s = self.settings
        self.stats["evaluations"] += 1
        self._last_eval_step = int(obs.step)
        view = self._view(obs, world)
        points = self._resample(obs, route)
        walk = np.zeros(world.grid.shape, dtype=bool)
        for _, xy, heading, _ in points:
            walk |= view.unknown_ahead(xy, heading)
        scanned = self._scanned_points()
        turn = float(self.policy.episode.action_spec.turn_angle_rad)
        side_actions = 2 * int(math.ceil(s.side_rad / turn - 1e-9))
        full_actions = int(math.ceil(2 * math.pi / turn - 1e-9))
        candidates, best = [], None
        for index, xy, heading, along in points:
            if any(math.dist(xy, point) <= s.scanned_m for point in scanned):
                continue                                           # a full rotation stood here: nothing more to see
            gain = glance_gains(view, xy, heading, s.side_rad, already=walk)
            candidate = GlanceCandidate(index, (float(xy[0]), float(xy[1])), float(heading), float(along),
                                        gain.left_m2, gain.right_m2, gain.full_m2)
            candidates.append(candidate)
            for kind, value, actions in ((LEFT, gain.left_m2, side_actions), (RIGHT, gain.right_m2, side_actions),
                                         (FULL, gain.full_m2, full_actions)):
                if value < s.min_gain_m2 or value / actions < s.min_gain_per_action_m2:
                    continue
                plan = GlancePlan(candidate, kind, float(value), int(actions))
                if best is None or plan.rate > best.rate + 1e-9 or (abs(plan.rate - best.rate) <= 1e-9
                                                                   and plan.candidate.along_m < best.candidate.along_m):
                    best = plan
        self.candidates = tuple(candidates)
        previous = self.plan_in_force
        self.plan_in_force = best
        if best is not None and (previous is None or previous.kind != best.kind
                                 or math.dist(previous.candidate.xy, best.candidate.xy) > s.stride_m / 2):
            self.stats["scheduled"] += 1
            self._log(obs, "glance_scheduled", kind=best.kind, gain_m2=round(best.gain_m2, 2), actions=best.actions,
                      along_m=round(best.candidate.along_m, 2), xy=[round(v, 2) for v in best.candidate.xy],
                      candidates=len(candidates))

    def _scanned_points(self):
        """Where completed rotations stood on this storey (the scan ledger), as ``(x, y)`` pairs."""
        ledger = getattr(self.policy, "scans", None)
        if ledger is None or not hasattr(ledger, "on_floor"):
            return []
        return [tuple(record["xy"]) for record in ledger.on_floor()]

    @staticmethod
    def _passed(route, here, plan):
        """Whether the agent has walked past the plan's point without coming within ``arrival_m`` of it."""
        nearest = min(range(len(route)), key=lambda i: math.dist(route[i], here))
        target = min(range(len(route)), key=lambda i: math.dist(route[i], plan.candidate.xy))
        return nearest > target

    # -- the look -------------------------------------------------------------
    def _start(self, obs, plan):
        p = self.policy
        yaw = float(obs.pose.yaw)
        heading = plan.candidate.heading
        if plan.kind == FULL:
            direction = 1.0 if plan.candidate.left_m2 >= plan.candidate.right_m2 else -1.0
            target = None
        else:
            direction = 1.0 if plan.kind == LEFT else -1.0
            target = float(normalize_angle(heading + direction * self.settings.side_rad))
        self.active = dict(kind=plan.kind, direction=direction, target=target, started=int(obs.step),
                           turns=0, swept=0.0, last_yaw=yaw, gain_m2=plan.gain_m2,
                           xy=[round(v, 2) for v in plan.candidate.xy])
        self.stats["started"] += 1
        self.plan_in_force = None
        self._log(obs, "glance_started", kind=plan.kind, gain_m2=round(plan.gain_m2, 2), actions=plan.actions,
                  xy=self.active["xy"], heading_deg=round(math.degrees(heading), 1))
        return self._turn(obs)

    def _continue(self, obs):
        active = self.active
        turn = float(self.policy.episode.action_spec.turn_angle_rad)
        yaw = float(obs.pose.yaw)
        active["swept"] += abs(normalize_angle(yaw - active["last_yaw"]))
        active["last_yaw"] = yaw
        done = False
        if active["kind"] == FULL:
            full_circle = int(math.ceil(2 * math.pi / turn - 1e-9))
            done = active["swept"] >= 2 * math.pi - 0.5 * turn or active["turns"] >= full_circle + 2
        else:
            side_turns = int(math.ceil(self.settings.side_rad / turn - 1e-9))
            done = (abs(normalize_angle(active["target"] - yaw)) <= 0.5 * turn + 1e-6
                    or active["turns"] >= side_turns + 2)
        if not done:
            return self._turn(obs)
        self.stats["completed"] += 1
        self.stats["gain_m2"] += float(active["gain_m2"])
        self._log(obs, "glance_complete", kind=active["kind"], turns=active["turns"],
                  swept_deg=round(math.degrees(active["swept"]), 1), actions=int(obs.step) - active["started"])
        self._finish_pause(active["turns"])
        self.active = None
        self._last_glance_step = int(obs.step)
        return None

    def _turn(self, obs):
        active = self.active
        p = self.policy
        turn = float(p.episode.action_spec.turn_angle_rad)
        yaw = float(obs.pose.yaw)
        if active["kind"] == FULL:
            step = active["direction"] * turn
        else:
            error = normalize_angle(active["target"] - yaw)
            step = max(-turn, min(turn, error))
        active["turns"] += 1
        self.stats["turns"] += 1
        p._action_owner = "glance"
        return NavigationCommand.hold(
            final_yaw=float(normalize_angle(yaw + step)),
            info={"kind": "glance", "glance": active["kind"], "turn": active["turns"],
                  "gain_m2": round(float(active["gain_m2"]), 2), "xy": list(active["xy"])})

    def _finish_pause(self, turns):
        memory = getattr(self.policy, "route_memory", None)
        if memory is not None and turns > 0 and hasattr(memory, "resume_after_pause"):
            memory.resume_after_pause(int(turns))

    # -- the record -----------------------------------------------------------
    def _log(self, obs, event, **fields):
        self.events.append(dict(step=int(obs.step), floor_id=getattr(getattr(self.policy, "mapping", None), "floor_id", None),
                                event=event, **fields))

    def state(self):
        """The glance in force (or the one planned) for the HUD and the per-step record."""
        plan = self.plan_in_force
        return {"active": None if self.active is None else {k: v for k, v in self.active.items() if k != "last_yaw"},
                "planned": None if plan is None else {"kind": plan.kind, "gain_m2": round(plan.gain_m2, 2),
                                                      "actions": plan.actions,
                                                      "xy": [round(v, 2) for v in plan.candidate.xy],
                                                      "along_m": round(plan.candidate.along_m, 2)}}

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats), "events": list(self.events),
                "state": self.state()}
