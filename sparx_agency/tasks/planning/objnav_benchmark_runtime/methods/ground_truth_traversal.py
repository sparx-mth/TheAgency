"""Committed floor transition along a ground-truth stair connector.

The connector's polyline is the walkable path the navmesh itself reports, so
the traversal is a route to follow, not a surface to discover: no depth
support search, no in-place scans to "acquire tread support" -- the
behaviour that spun for 120 actions at the top of a real staircase and
again on a raised bathroom floor. Every action returns a FOLLOW command:
along the polyline while climbing, past its far anchor to step clear of the
stairs once the destination height is reached, or back down it in retreat.

Arrival is still the atlas's to confirm (a settled, translated plateau at a
storey height); this class only says WHEN the atlas may confirm it --
``arrival_allowed`` -- which is at the far anchor at the destination height,
or back at the source anchor after a retreat.

Retreat is bounded and never halts the episode: when its budget is spent the
transition stops claiming to know where the agent is and lets the atlas
settle wherever it stands, so the search resumes rather than stopping.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sparx_agency.core.planning.objnav.types.command import NavigationCommand

Xyz = Tuple[float, float, float]

#: Actions without XY progress before the traversal turns back.
STALL_ACTIONS = 18
#: How far past the far anchor the agent walks to clear the stairs.
EXIT_M = 0.75
#: A polyline vertex counts as reached within this XY radius.
REACHED_M = 0.35


class GroundTruthTraversal:
    """Follow the committed connector; report when the atlas may confirm arrival."""

    def __init__(self, coordinator, obs):
        self.coordinator = coordinator
        self.policy, self.params = coordinator.policy, coordinator.params
        self.portal = coordinator.active
        self.direction = int(self.portal["direction"])
        self.source_floor = coordinator.floor_id
        self.source_height = float(self.policy.mapping.atlas.elevation_m)
        self.polyline: List[Xyz] = [tuple(float(v) for v in p[:3]) for p in self.portal["path"]]
        if len(self.polyline) < 2:
            raise ValueError("A ground-truth connector needs at least two polyline points")
        self.destination_height = float(self.portal.get("destination_z", self.polyline[-1][2]))
        self.route: List[Xyz] = list(self.polyline)
        self.cursor = 0
        self.started_step = self.last_progress_step = obs.step
        self.phase = "TRAVERSE"
        self.failures = 0
        self.exit_attempts = 0
        self.retreat_path: Optional[List[Xyz]] = None
        self.retreat_step: Optional[int] = None
        self.arrival_allowed = False
        self.completion_reason: Optional[str] = None
        self.close_support = self.direction < 0          # descending: look down the flight
        self.surface_goal = None
        self.path: List[Xyz] = []
        self.best_rise_m = 0.0
        self.previous: Xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        self.trace: List[Xyz] = [self.previous]
        self.policy.mapping.atlas.begin_transition(obs.pose)

    # -- observation ---------------------------------------------------------
    def observe(self, obs):
        xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        if math.dist(xyz[:2], self.previous[:2]) > 0.04:
            self.trace.append(xyz)
            self.last_progress_step = obs.step
        self.previous = xyz
        self.best_rise_m = max(self.best_rise_m, self.direction * (xyz[2] - self.source_height))
        self._advance(xyz)
        if self.phase == "RETREAT":
            at_source = (abs(xyz[2] - self.source_height) <= self.params.floor_match_m
                         and math.dist(xyz[:2], self.polyline[0][:2]) <= self.params.exit_distance_m + 0.5)
            self.arrival_allowed = at_source or self.completion_reason == "ground_truth_retreat_budget_exhausted"
            if at_source:
                self.completion_reason = "ground_truth_source_returned"
            return
        at_height = abs(xyz[2] - self.destination_height) <= self.params.floor_match_m
        near_end = (self.cursor >= len(self.route) - 1
                    or math.dist(xyz[:2], self.polyline[-1][:2]) <= self.params.exit_distance_m)
        if at_height and near_end:
            self.phase = "CONFIRM_DESTINATION"
            self.arrival_allowed = True
            self.completion_reason = "ground_truth_connector_end_reached"
        else:
            self.phase = "TRAVERSE"
            self.arrival_allowed = False

    def _advance(self, xyz):
        """Move the cursor to the furthest route vertex reached; never backwards."""
        for index in range(self.cursor, len(self.route)):
            if math.dist(xyz[:2], self.route[index][:2]) <= REACHED_M:
                self.cursor = max(self.cursor, index)

    # -- command ---------------------------------------------------------------
    def plan(self, obs):
        elapsed = obs.step - self.started_step
        stalled = obs.step - self.last_progress_step >= STALL_ACTIONS
        if self.retreat_path is None and self.phase != "CONFIRM_DESTINATION" and (
                self.failures >= self.params.max_failures or elapsed >= self.params.transition_actions or stalled):
            self._begin_retreat(obs, "timeout" if elapsed >= self.params.transition_actions else
                                "blocked" if self.failures >= self.params.max_failures else "stalled")
        if self.retreat_path is not None and obs.step - self.retreat_step >= self.params.retreat_actions:
            # Out of retreat budget: stop claiming to know where the agent is and let the
            # atlas settle wherever it stands. The search continues; nothing halts.
            self.completion_reason = "ground_truth_retreat_budget_exhausted"
            self.arrival_allowed = True
        points = self._remaining(obs)
        self.path = points
        self.policy._route, self.policy._goal = points, tuple(points[-1][:2])
        return NavigationCommand.follow(points, info={"kind": self.phase, "portal": self.portal["id"],
                                                      "stair_source": "ground_truth",
                                                      "direction": "up" if self.direction > 0 else "down"})

    def _begin_retreat(self, obs, reason):
        self.retreat_path = list(reversed(self.route[:self.cursor + 1]))
        if len(self.retreat_path) < 2:
            self.retreat_path = [self.previous, self.polyline[0]]
        self.route, self.cursor = list(self.retreat_path), 0
        self.retreat_step = obs.step
        self.phase = "RETREAT"
        self.arrival_allowed = False
        self.coordinator.events.append({"action": obs.step, "event": "retreat_started",
                                        "portal_id": self.portal["id"], "reason": reason})

    def _remaining(self, obs) -> List[Xyz]:
        """The route ahead of the agent, or the exit stub once the route is walked."""
        pose = (obs.pose.x, obs.pose.y, obs.pose.z)
        start = self.cursor if math.dist(pose[:2], self.route[self.cursor][:2]) > REACHED_M else self.cursor + 1
        ahead = self.route[start:]
        if ahead:
            return ahead
        return [self._exit_target()]

    def _exit_target(self) -> Xyz:
        """A point ``EXIT_M`` past the route's end, rotated a quarter turn per blocked attempt."""
        (ax, ay, _), (bx, by, bz) = self.route[-2], self.route[-1]
        heading = math.atan2(by - ay, bx - ax) + (math.pi / 2) * (self.exit_attempts % 4)
        return (bx + EXIT_M * math.cos(heading), by + EXIT_M * math.sin(heading), bz)

    def blocked(self, obs):
        """A forward step did not move: skip the corner, or try another exit heading."""
        self.failures += 1
        if self.cursor + 1 < len(self.route):
            self.cursor += 1
        else:
            self.exit_attempts += 1

    def diagnostics(self) -> Dict:
        return {"phase": self.phase, "portal_id": self.portal["id"], "source_floor": self.source_floor,
                "direction": self.direction, "stair_source": "ground_truth",
                "source_height_m": self.source_height, "destination_height_m": self.destination_height,
                "destination": self.portal.get("destination"), "cursor": self.cursor, "route_points": len(self.route),
                "best_rise_m": self.best_rise_m, "started_step": self.started_step,
                "arrival_allowed": self.arrival_allowed, "completion_reason": self.completion_reason,
                "retreat_step": self.retreat_step, "failures": self.failures, "exit_attempts": self.exit_attempts}

