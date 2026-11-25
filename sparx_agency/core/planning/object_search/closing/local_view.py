"""One short, bounded change of viewpoint on the agent's observed standing floor."""
from __future__ import annotations

from dataclasses import replace
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.object_search.closing.map_sight import MEMORY_SIGHT_MARGIN_M, map_sight_clear
from sparx_agency.core.planning.objnav.action_converter.action_choice import turn_action
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


class LocalView:
    """A single endpoint, rechecked each decision; no pose outside RGB-D mapping is consulted."""

    def __init__(self, obs, floor_id, kind, max_walk, max_actions):
        self.start = int(obs.step)
        self.origin = self.previous = (obs.pose.x, obs.pose.y)
        self.floor_id, self.kind = floor_id, kind
        self.max_walk, self.max_actions = max_walk, max_actions
        self.walked = 0.0
        self.goal = None
        self.finished = None

    def choose(self, closing, obs, world, points, acceptable, focus=None):
        p = closing.policy
        if (self.floor_id != p.mapping.floor_id or obs.step - self.start >= self.max_actions
                or obs.step >= p.episode.max_steps - 1):
            return False
        floor = closing.standing_floor(obs, world)
        options = []
        for point in points:
            xy = floor.nearest(point, world.resolution)
            if xy is None or not acceptable(xy) or not p.floor_guard.goal_allowed(obs, world, xy, self.kind):
                continue
            if focus is not None and not map_sight_clear(world, xy, focus, skip_to_m=MEMORY_SIGHT_MARGIN_M):
                continue
            route = p._plan_to(obs, world, xy, exact_goal=True)
            if route is not None and route.length() <= self.max_walk:
                options.append((route.length(), xy))
        if options:
            self.goal = min(options)[1]
        return self.goal is not None

    def end(self, p, why):
        self.finished = why
        if p.route_memory.kind == self.kind:
            p.route_memory.clear(self.kind + "_ended")
            p._route = p._goal = None

    def advance(self, closing, obs, world, yaw, acceptable):
        p = closing.policy
        here = (obs.pose.x, obs.pose.y)
        self.walked += math.dist(here, self.previous)
        self.previous = here
        if (p.mapping.floor_id != self.floor_id or self.goal is None or
                obs.step - self.start >= self.max_actions or obs.step >= p.episode.max_steps - 1):
            self.end(p, "expired or unavailable")
            return None
        if math.dist(here, self.goal) <= p.converter_params.goal_tolerance_m and acceptable(here):
            if turn_action(normalize_angle(yaw - obs.pose.yaw), p.episode.action_spec.turn_angle_rad) is None:
                self.end(p, "observed alternative view")
                return None
            return NavigationCommand.hold(final_yaw=yaw, info={"kind": self.kind, "stage": "face"})
        if p.route_memory.kind == self.kind and p.route_memory.arrived(obs) and not acceptable(here):
            self.end(p, "arrival did not change viewing angle")
            return None
        if (self.walked + p.episode.action_spec.forward_step_m > self.max_walk + 1e-6
                or not closing.standing_floor(obs, world).standable_at(self.goal)):
            self.end(p, "walk limit or endpoint invalid")
            return None
        command = p._navigate(obs, world, self.goal, self.kind, exact_goal=True)
        if command is None:
            self.end(p, "route unavailable")
            return None
        # Re-observe between actions; do not queue a lattice plan past the retry budget.
        return replace(command, actions=(), plan_pose=None)


def ring_views(here, focus, radius, angles_deg):
    """Alternative viewing positions relative to the observing agent, not a scene axis."""
    direction = math.atan2(here[1] - focus[1], here[0] - focus[0])
    return [(focus[0] + radius * math.cos(direction + math.radians(a)),
             focus[1] + radius * math.sin(direction + math.radians(a))) for a in angles_deg]


def angle_change(focus, witness, xy):
    first = math.atan2(witness[1] - focus[1], witness[0] - focus[0])
    second = math.atan2(xy[1] - focus[1], xy[0] - focus[0])
    return abs(normalize_angle(second - first))
