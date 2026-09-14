"""Committed-route lifetime using the existing monotone progress and safety APIs."""
from __future__ import annotations

from dataclasses import dataclass
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.objnav.action_converter.ladder import choose_action, parallel_offset
from sparx_agency.core.planning.objnav.action_converter.path_progress import PathProgress
from sparx_agency.core.planning.objnav.action_converter.types import STATUS_ARRIVED
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


@dataclass(frozen=True)
class RouteSettings:
    goal_shift_m: float = 0.5
    cross_track_m: float = 0.5
    progress_timeout_steps: int = 30
    progress_epsilon_m: float = 0.10


class CommittedRoute:
    """Keep a safe route through turns and map updates; do not restart on a timer.

    Safety and blocked-motion invalidation override commitment immediately.
    Progress is monotone on the same leg, reusing the converter's geometry.
    One planner owns this route, so its last_inflate_m is the actual accepted
    clearance, not a radius from a discarded alternative candidate.
    """

    def __init__(self, spec, converter, settings=None):
        self.spec, self.converter = spec, converter
        self.settings = settings or RouteSettings()
        self.path = self.goal = self.kind = None
        self.progress = PathProgress()
        self.last_progress_step = 0
        self.best_remaining = math.inf
        self._motion_xy = None
        self._motion_step = 0
        self.stats = {"adoptions": 0, "kept": 0, "invalidations": 0}
        self.reason = "unplanned"

    def clear(self, reason):
        if self.path is not None:
            self.stats["invalidations"] += 1
        self.path = self.goal = self.kind = None
        self.progress = PathProgress()
        # A replacement path is not evidence of movement. Keep the positional
        # watchdog across safety replans, including a failed forward action.
        if reason not in ("route_obstructed", "off_route", "forward_blocked"):
            self._motion_xy = None
        self.reason = reason

    def adopt(self, path, goal, kind, observation):
        self.path, self.goal, self.kind = path, tuple(goal), kind
        raw = getattr(path, "points", path)
        self._points = tuple(Pose2D(float(p.x), float(p.y)) if hasattr(p, "x")
                             else Pose2D(float(p[0]), float(p[1])) for p in raw)
        self.progress = PathProgress()
        self.progress.adopt(tuple((p.x, p.y) for p in self._points))
        self.last_progress_step = observation.step
        self.best_remaining = math.inf
        self.stats["adoptions"] += 1
        self.reason = "new_goal_or_invalid_route"

    def _sample(self, observation):
        return self.progress.advance(
            observation.pose, self.converter.lookahead_m, self.spec.forward_step_m / 2,
            parallel_offset(self.spec, self.converter.lookahead_m))

    def arrived(self, observation):
        """Use the executor's arrival rule on the actual (possibly snapped) path.

        Checking distance to the requested goal both misses snapped arrivals
        and prematurely finishes return legs passing near their endpoint.
        Final facing is deliberately excluded: this asks if translation is done.
        """
        if self.path is None:
            return False
        result = choose_action(
            observation.pose, NavigationCommand.follow(self.path), self._sample(observation),
            False, self.spec, self.converter)
        return result.status == STATUS_ARRIVED

    def reusable(self, observation, world, planner, goal, kind):
        if self.path is not None and (
                kind != self.kind or math.dist(goal, self.goal) > self.settings.goal_shift_m):
            self.clear("goal_changed")
        xy = (observation.pose.x, observation.pose.y)
        if self._motion_xy is None or math.dist(xy, self._motion_xy) >= self.settings.progress_epsilon_m:
            self._motion_xy, self._motion_step = xy, observation.step
        elif observation.step - self._motion_step > self.settings.progress_timeout_steps:
            self.clear("no_progress")
            return False
        if self.path is None:
            self.reason = "unplanned"
            return False
        sample = self._sample(observation)
        if sample.cross_track_m > self.settings.cross_track_m:
            self.clear("off_route")
            return False
        points = list(self._points)
        remaining = [Pose2D(observation.pose.x, observation.pose.y, observation.pose.yaw)] + points[sample.segment_index + 1:]
        if planner.path_collides(world, remaining, passable_start=remaining[0]):
            self.clear("route_obstructed")
            return False
        if sample.remaining_path_m < self.best_remaining - self.settings.progress_epsilon_m:
            self.best_remaining = sample.remaining_path_m
            self.last_progress_step = observation.step
        elif observation.step - self.last_progress_step > self.settings.progress_timeout_steps:
            self.clear("no_progress")
            return False
        self.stats["kept"] += 1
        self.reason = "committed_safe_route"
        return True



