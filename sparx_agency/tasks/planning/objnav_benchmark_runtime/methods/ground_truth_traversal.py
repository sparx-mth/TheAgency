"""Committed floor transition along a ground-truth stair connector.

The connector's polyline is the walkable path the navmesh itself reports, so
the traversal is a route to follow, not a surface to discover: no depth
support search, no in-place scans to "acquire tread support" -- the
behaviour that spun for 120 actions at the top of a real staircase and
again on a raised bathroom floor. Every action returns a FOLLOW command:
along the polyline while climbing, past its far anchor to step clear of the
stairs once the destination height is reached, or back down it in retreat.

**Once begun, a transition is finished.** The rules that make that so:

* **The route handed to the converter starts where the agent stands.** The
  approach ends up to ``approach_snap_m`` short of the entry anchor (the
  anchor itself is rarely a passable cell of the observed map), and a
  staircase folds back on itself: on a U-shaped flight the bottom anchor
  passes within a metre of the top one in XY, one storey down. Handed the
  bare polyline, the converter's nearest-leg projection landed on the far
  leg and aimed at the bottom anchor through the stair-well wall -- 463
  actions grinding on the spot in the first recorded campaign. So every
  command is a LEAD-IN leg from the point where the agent stood when its
  next vertex was decided, then the route from that vertex on; the
  converter's progress runs along the lead-in first and can only ever reach
  the far leg by walking there.
* **A vertex is reached in three dimensions.** Switchback flights lie 0.3-0.6
  m apart in XY and a storey apart in height; an XY test alone marked the
  flight above as walked from the flight below (Pomaria: the cursor jumped
  a storey at the first landing and the agent paced 460 actions against
  the landing wall).
* A blocked forward step TIGHTENS the following before anything else: the
  next command aims straight at the next polyline vertex, with no lookahead
  to cut the corner into the banister the shortest path grazes.
* **No vertex is ever skipped.** On a staircase the polyline is the only
  way; a route that gave a vertex up aimed at the one after it THROUGH the
  landing wall (Ranchester: three skips, then a retreat, twice). When the
  tight aim is blocked too the agent goes BACK to the last vertex it did
  reach -- a point of the centreline, from which the next leg is known to
  be walkable -- and comes at the vertex again along the route.
* Turning back is the last resort, not the first: it takes
  :attr:`MultiFloorParams.commit_failures` blocked steps or
  :attr:`MultiFloorParams.commit_stall_actions` actions in which the
  distance still to walk along the route never shrank. Displacement alone
  is not progress: an agent skidding back and forth along a wall moves
  every action and gets nowhere, and a stall clock reset by any movement
  never fired on 300 such actions. The transition budget running out no
  longer turns the agent round -- it keeps following and lets the atlas
  settle wherever the agent actually stands.
* At the far anchor at the destination height the atlas is asked to
  confirm; if its translated-plateau test has not fired within
  :attr:`MultiFloorParams.confirm_actions` the storey is confirmed
  outright -- the navmesh said this height is a storey, and the agent is
  standing on it.

Arrival is still the atlas's to confirm (a settled, translated plateau at a
storey height); this class only says WHEN the atlas may confirm it --
``arrival_allowed`` -- which is at the far anchor at the destination height,
or back at the retreat's destination after a retreat: the source anchor when
the flight was entered, the point the traversal began from when it never was.

Retreat is bounded and never halts the episode: when its budget is spent the
transition stops claiming to know where the agent is and lets the atlas
settle wherever it stands, so the search resumes rather than stopping.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sparx_agency.core.planning.objnav.types.command import NavigationCommand

Xyz = Tuple[float, float, float]

#: Actions without XY progress before the traversal turns back (observed-mode heritage; the
#: committed ground-truth rule reads ``MultiFloorParams.commit_stall_actions`` instead).
STALL_ACTIONS = 18
#: How far past the far anchor the agent walks to clear the stairs when the connector has no verified exit.
EXIT_M = 0.75
#: Swings of the exit stub either side of the last leg per blocked attempt, radians; never behind the agent.
EXIT_SWINGS_RAD = (0.0, math.radians(45), math.radians(-45), math.radians(90), math.radians(-90))
#: A polyline vertex counts as reached within this radius in plan view. It must exceed the converter's
#: arrival tolerance (``goal_tolerance_m``, 0.25 m): a route handed to the converter ends at a corner
#: vertex, and the traversal must move on to the next leg before the converter declares the path
#: walked and idles -- but no wider than that, since the agent then leaves the vertex from that far off.
REACHED_XY_M = 0.26
#: ... and within this much of its height: the flight above a switchback is a storey away.
REACHED_Z_M = 0.45
#: An agent farther than this from the vertex it just reached is led back THROUGH it before the next leg.
VIA_M = 0.10
#: Blocked steps aimed straight at one vertex before the agent falls back to the last vertex it reached.
TIGHT_ATTEMPTS = 2
#: The least the distance still to walk must shrink by for an action to count as progress.
PROGRESS_M = 0.04
#: A route leg turning more than this from the leg the agent is on is a corner: the route handed to the
#: converter ends at that corner's vertex, so the corner is walked to, not cut (one turn action's worth).
BEND_RAD = math.radians(35.0)


def near(a: Xyz, b: Xyz) -> bool:
    """Whether ``a`` stands on vertex ``b``: within :data:`REACHED_XY_M` in plan view and :data:`REACHED_Z_M` in height."""
    return math.dist(a[:2], b[:2]) <= REACHED_XY_M and abs(a[2] - b[2]) <= REACHED_Z_M


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
        #: Whether the cursor's vertex has been reached (the cursor starts at 0 short of the foot).
        self._reached = False
        #: Whether any vertex of the flight was ever reached: the retreat's way back depends on it.
        self._entered = False
        self.started_step = self.last_progress_step = obs.step
        self.phase = "TRAVERSE"
        self.failures = 0
        self.exit_attempts = 0
        self.tight = False
        #: Times the agent went back to the last vertex it reached because the tight aim was blocked too.
        self.recoveries = 0
        #: The vertex the agent is walking BACK to, while it is.
        self._recovery: Optional[Xyz] = None
        self._blocked_at: Dict[int, int] = {}
        self.confirm_step: Optional[int] = None
        self.forced = False
        self.retreat_path: Optional[List[Xyz]] = None
        self.retreat_step: Optional[int] = None
        self.arrival_allowed = False
        self.completion_reason: Optional[str] = None
        self.close_support = self.direction < 0          # descending: look down the flight
        self.surface_goal = None
        self.path: List[Xyz] = []
        self.best_rise_m = 0.0
        self.previous: Xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        #: Where the traversal began: the retreat's destination when the flight was never entered.
        self.origin: Xyz = self.previous
        self.trace: List[Xyz] = [self.previous]
        #: The lead-in leg's start and the route index it leads to; renewed whenever that index changes.
        self._lead_in: Xyz = self.previous
        self._lead_start: Optional[int] = None
        #: The vertex the lead-in passes back through when the agent reached it from off to one side.
        self._via: Optional[Xyz] = None
        #: The least distance still to walk seen so far; progress is shrinking it.
        self._best_remaining_m = math.inf
        self.policy.mapping.atlas.begin_transition(obs.pose)

    # -- observation ---------------------------------------------------------
    def observe(self, obs):
        xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
        if math.dist(xyz[:2], self.previous[:2]) > 0.04:
            self.trace.append(xyz)
        self.previous = xyz
        self.best_rise_m = max(self.best_rise_m, self.direction * (xyz[2] - self.source_height))
        if self._recovery is not None and near(xyz, self._recovery):
            # Back on a vertex of the centreline: the next leg is walkable from here.
            self._recovery, self.tight, self._lead_start, self._via = None, False, None, None
        if self._advance(xyz):
            self.last_progress_step = obs.step             # a vertex reached is progress too
            self._best_remaining_m = math.inf              # a new leg: measure it afresh
        remaining = self._remaining_m(xyz)
        if remaining < self._best_remaining_m - PROGRESS_M:
            # The route got shorter by a real amount since the last time it did: progress.
            # Not the running minimum -- a skid back and forth along a wall changes the
            # distance every action and shortens it never.
            self._best_remaining_m = remaining
            self.last_progress_step = obs.step
        if self.phase == "RETREAT":
            destination = self.retreat_path[-1] if self.retreat_path else self.polyline[0]
            at_source = (abs(xyz[2] - self.source_height) <= self.params.floor_match_m
                         and math.dist(xyz[:2], destination[:2]) <= self.params.exit_distance_m + 0.5)
            exhausted = self.completion_reason == "ground_truth_retreat_budget_exhausted"
            self.arrival_allowed = at_source or exhausted
            if at_source:
                self.completion_reason = "ground_truth_source_returned"
            if at_source or (exhausted and self._on_known_floor(xyz[2])):
                # Back on a storey the atlas has measured: confirm the return now rather than
                # wait for a plateau test an agent turning on the spot never passes.
                self.policy.mapping.atlas.settle(obs.pose, xyz[2])
            return
        at_height = abs(xyz[2] - self.destination_height) <= self.params.floor_match_m
        near_end = (self.cursor >= len(self.route) - 1
                    or math.dist(xyz[:2], self.polyline[-1][:2]) <= self.params.exit_distance_m)
        if at_height and near_end:
            if self.phase != "CONFIRM_DESTINATION":
                self.confirm_step = obs.step
            self.phase = "CONFIRM_DESTINATION"
            self.arrival_allowed = True
            if not self.forced:
                self.completion_reason = "ground_truth_connector_end_reached"
        else:
            self.phase = "TRAVERSE"
            self.confirm_step = None
            # The transition budget is spent: stop claiming to know where the agent
            # is and let the atlas settle if it stands on a storey -- but keep
            # following; a committed climb is not turned round by a clock.
            self.arrival_allowed = obs.step - self.started_step >= self.params.transition_actions

    def _advance(self, xyz):
        """Move the cursor to the furthest route vertex reached; never backwards. True when it moved.

        Reached means :func:`near` -- within ``REACHED_XY_M`` in plan view
        and ``REACHED_Z_M`` in height: the flight above a switchback passes
        within a hand's breadth in XY of the flight below, a storey up.
        Reaching the cursor's own vertex for the first time counts too -- the
        cursor starts at 0 before the foot of the flight is reached.
        """
        before = (self.cursor, self._reached)
        for index in range(self.cursor, len(self.route)):
            if near(xyz, self.route[index]):
                self.cursor, self._reached, self._entered = max(self.cursor, index), True, True
        if (self.cursor, self._reached) != before:
            self.tight = False                              # the corner is behind us
        return (self.cursor, self._reached) != before

    def _next_index(self) -> int:
        """The route index the agent walks toward now.

        The cursor's own vertex until it has been reached, then the one after
        it -- and it stays the one after it however far the agent walks on:
        a distance test here sent the route back to the vertex just left.
        """
        return self.cursor + 1 if self._reached else self.cursor

    def _on_known_floor(self, height) -> bool:
        """Whether ``height`` is within the floor-match tolerance of a storey the atlas has measured."""
        return any(abs(floor.elevation_m - height) <= self.params.floor_match_m
                   for floor in self.policy.mapping.atlas.floors.values())

    def _remaining_m(self, xyz) -> float:
        """The distance still to walk: to the next vertex and along the route past it, or to the exit stub."""
        start = self._next_index()
        if start >= len(self.route):
            return math.dist(xyz[:2], self._exit_target()[:2])
        ahead = self.route[start:]
        return math.dist(xyz[:2], ahead[0][:2]) + sum(math.dist(a[:2], b[:2]) for a, b in zip(ahead, ahead[1:]))

    # -- command ---------------------------------------------------------------
    def plan(self, obs):
        stalled = obs.step - self.last_progress_step >= self.params.commit_stall_actions
        if self.retreat_path is None and self.phase != "CONFIRM_DESTINATION" and (
                self.failures >= self.params.commit_failures or stalled):
            self._begin_retreat(obs, "blocked" if self.failures >= self.params.commit_failures else "stalled")
        if self.retreat_path is not None and obs.step - self.retreat_step >= self.params.retreat_actions:
            # Out of retreat budget: stop claiming to know where the agent is and let the
            # atlas settle wherever it stands. The search continues; nothing halts.
            self.completion_reason = "ground_truth_retreat_budget_exhausted"
            self.arrival_allowed = True
        if (self.phase == "CONFIRM_DESTINATION" and not self.forced and self.confirm_step is not None
                and obs.step - self.confirm_step >= self.params.confirm_actions):
            self._force_confirmation(obs)
        points = self._remaining(obs)
        self.path = points
        self.policy._route, self.policy._goal = points, tuple(points[-1][:2])
        return NavigationCommand.follow(points, info={"kind": self.phase, "portal": self.portal["id"],
                                                      "stair_source": "ground_truth",
                                                      "direction": "up" if self.direction > 0 else "down",
                                                      "tight": self.tight, "recovering": self._recovery is not None})

    def _force_confirmation(self, obs):
        """The agent has stood at the destination height by the far anchor long enough: it has arrived."""
        self.forced = True
        if self.policy.mapping.atlas.settle(obs.pose, self.destination_height):
            self.completion_reason = "ground_truth_destination_forced"
            self.coordinator.events.append({"action": obs.step, "event": "destination_forced",
                                            "portal_id": self.portal["id"],
                                            "waited_actions": obs.step - self.confirm_step,
                                            "height_m": round(obs.pose.z, 3)})
        else:
            self.coordinator.events.append({"action": obs.step, "event": "destination_refused",
                                            "portal_id": self.portal["id"],
                                            "reason": "destination height is neither a known floor nor a new storey"})

    def _begin_retreat(self, obs, reason):
        entered = self._entered
        if entered:
            # Back down the vertices walked, then off the flight to where the approach ended.
            self.retreat_path = list(reversed(self.route[:self.cursor + 1])) + [self.origin]
        else:
            # The entry was never reached: the way back is to where the traversal began, not
            # to the anchor the agent could not get to.
            self.retreat_path = [self.previous, self.origin]
        if math.dist(self.retreat_path[-1][:2], self.retreat_path[-2][:2]) <= REACHED_XY_M and len(self.retreat_path) > 2:
            self.retreat_path.pop()                         # the approach ended on the anchor itself
        self.route, self.cursor = list(self.retreat_path), 0
        self._reached = near(self.previous, self.route[0])
        self.retreat_step = obs.step
        self.phase = "RETREAT"
        self.tight = False
        self._recovery = None
        self.arrival_allowed = False
        self._lead_start = None
        self._via = None
        self._best_remaining_m = math.inf
        self.coordinator.events.append({"action": obs.step, "event": "retreat_started",
                                        "portal_id": self.portal["id"], "reason": reason,
                                        "failures": self.failures, "recoveries": self.recoveries,
                                        "entered_flight": entered})

    def _remaining(self, obs) -> List[Xyz]:
        """The route ahead of the agent, led into from where the agent stands, or the exit stub.

        The first point is the LEAD-IN: where the agent stood when the vertex
        it walks toward was decided (the traversal's start, or the action a
        vertex was reached). The converter projects the agent onto that leg
        first and walks the route from there, so a far leg of a folded flight
        passing nearer in XY is never mistaken for progress. The lead-in is
        held fixed until the next vertex changes, so the converter sees the
        same path from one action to the next and keeps its forward-only
        progress along it.

        After a blocked step the command is a straight line from the agent to
        the next vertex and nothing more: the converter's lookahead can then
        only aim along that line, never across the corner it just hit. While
        the agent is going back to the last vertex it reached, the command
        is the straight line to that vertex.
        """
        pose = (obs.pose.x, obs.pose.y, obs.pose.z)
        if self._recovery is not None:
            return [pose, self._recovery]
        start = self._next_index()
        ahead = self.route[start:]
        if not ahead:
            # The route is walked: off the flight to the exit, so the atlas sees a translated
            # plateau -- by way of the far anchor when the agent has drifted back off the storey.
            exit_target = self._exit_target()
            off_storey = abs(pose[2] - self.destination_height) > self.params.floor_match_m
            if off_storey and self.phase != "RETREAT" and math.dist(pose[:2], self.route[-1][:2]) > REACHED_XY_M:
                return [pose, self.route[-1], exit_target]
            return [pose, exit_target]
        if self.tight:
            return [pose, ahead[0]]
        if start != self._lead_start:
            self._lead_start, self._lead_in = start, pose
            # Reached the last vertex from off to one side: lead back THROUGH it, so the converter's
            # aim lies on the centreline past the vertex, not on a chord from beside the route to the
            # vertex after -- a chord that ran through the wall of a narrow stair head (Hanson, Coffeen).
            via = self.route[start - 1] if self._reached and start >= 1 else None
            self._via = via if via is not None and math.dist(pose[:2], via[:2]) > VIA_M else None
        origin = self._via if self._via is not None else self._lead_in
        lead = [self._lead_in] + ([self._via] if self._via is not None else [])
        return lead + self._to_next_bend(origin, ahead)

    @staticmethod
    def _to_next_bend(lead_in: Xyz, ahead: List[Xyz]) -> List[Xyz]:
        """The route ahead up to and including the first vertex where it bends: the corner is walked TO, never cut.

        The converter aims a lookahead along the path past the agent's
        projection, so a path that continues round a corner is aimed across
        the corner -- into the inner wall of a landing, on every landing of
        every stairwell. A path that ENDS at the corner vertex clamps the aim
        to that vertex: the agent walks to the middle of the landing, turns
        there, and only then is handed the next leg. On a straight flight the
        whole flight is handed and the lookahead smooths the following as
        before.
        """
        if len(ahead) <= 1:
            return list(ahead)
        first = (ahead[0][0] - lead_in[0], ahead[0][1] - lead_in[1])
        if math.hypot(*first) < 0.05:
            first = (ahead[1][0] - ahead[0][0], ahead[1][1] - ahead[0][1])
        heading = math.atan2(first[1], first[0])
        for k in range(len(ahead) - 1):
            leg = (ahead[k + 1][0] - ahead[k][0], ahead[k + 1][1] - ahead[k][1])
            if math.hypot(*leg) < 0.05:
                continue
            turn = math.atan2(leg[1], leg[0]) - heading
            turn = math.atan2(math.sin(turn), math.cos(turn))
            if abs(turn) > BEND_RAD:
                return list(ahead[:k + 1])
        return list(ahead)

    def _exit_target(self) -> Xyz:
        """Where to stand for the atlas to confirm the storey: the connector's verified exit, else a stub.

        The extraction hands each connector an exit point on the flat floor
        off the flight, walkable from the anchor; it is used first. Without
        one, or once it has been blocked, a stub ``EXIT_M`` past the route's
        end is aimed along the last leg and swung a little to either side per
        blocked attempt -- never behind: a stub turned a quarter turn per
        block pointed back up the flight on the second block and walked the
        agent out of the destination height (Coffeen). When every swing is
        spent the target is the route's end itself, and the traversal waits
        there for the atlas or the forced confirmation.
        """
        (ax, ay, _), (bx, by, bz) = self.route[-2], self.route[-1]
        exit_xyz = self.portal.get("exit_xyz") if self.phase != "RETREAT" else None
        attempts = self.exit_attempts
        if exit_xyz is not None:
            if attempts == 0:
                return tuple(float(v) for v in exit_xyz[:3])
            attempts -= 1                                   # the verified exit was the first attempt
        if attempts >= len(EXIT_SWINGS_RAD):
            return (bx, by, bz)
        heading = math.atan2(by - ay, bx - ax) + EXIT_SWINGS_RAD[attempts]
        return (bx + EXIT_M * math.cos(heading), by + EXIT_M * math.sin(heading), bz)

    def blocked(self, obs):
        """A forward step did not advance: aim straight at the next vertex; fall back to the last one when that fails too.

        Never skip: on a staircase the polyline is the only way, and a route
        that gave a vertex up aimed at the one after it through the landing
        wall. The straight line from where the agent stands to the next
        vertex being obstructed means the agent has drifted off the leg;
        the last vertex it reached is a point of the centreline from which
        that leg is known to be walkable, so the agent goes back there and
        comes at the vertex again. Every block counts toward
        ``commit_failures``; the stall clock keeps running while the agent
        walks back, so a corner that never yields ends in a retreat.
        """
        self.failures += 1
        target = self._next_index()
        if target >= len(self.route):
            self.exit_attempts += 1                          # the exit stub itself is blocked: turn it
            return
        if self._recovery is not None:
            return                                           # blocked on the way back: keep going back
        hits = self._blocked_at.get(target, 0) + 1
        self._blocked_at[target] = hits
        if not self.tight or hits <= TIGHT_ATTEMPTS:
            self.tight = True
            return
        here = (obs.pose.x, obs.pose.y, obs.pose.z)
        fallback = self._fallback_vertex(here)
        self._blocked_at[target] = 0
        if fallback is None:
            return                                           # nowhere behind to go back to: keep aiming
        self._recovery = fallback
        self.recoveries += 1
        self.tight = False
        self.coordinator.events.append({"action": obs.step, "event": "traversal_recovery",
                                        "portal_id": self.portal["id"], "vertex": target,
                                        "back_to": [round(v, 3) for v in fallback], "failures": self.failures})

    def _fallback_vertex(self, here: Xyz) -> Optional[Xyz]:
        """The nearest vertex behind the agent that it does not already stand on; the traversal's origin at last."""
        walked = self.route[:self.cursor + (1 if self._reached else 0)]
        for vertex in list(reversed(walked)) + [self.origin]:
            if math.dist(here[:2], vertex[:2]) > REACHED_XY_M:
                return tuple(vertex)
        return None

    def diagnostics(self) -> Dict:
        return {"phase": self.phase, "portal_id": self.portal["id"], "source_floor": self.source_floor,
                "direction": self.direction, "stair_source": "ground_truth",
                "source_height_m": self.source_height, "destination_height_m": self.destination_height,
                "destination": self.portal.get("destination"), "cursor": self.cursor, "route_points": len(self.route),
                "best_rise_m": self.best_rise_m, "started_step": self.started_step,
                "last_progress_step": self.last_progress_step,
                "arrival_allowed": self.arrival_allowed, "completion_reason": self.completion_reason,
                "retreat_step": self.retreat_step, "failures": self.failures, "exit_attempts": self.exit_attempts,
                "tight": self.tight, "recoveries": self.recoveries, "recovering": self._recovery is not None,
                "confirm_step": self.confirm_step, "forced": self.forced}

