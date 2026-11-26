"""Building scheduling around floor-local LLM/RPT* search: which stairs, when, up or down.

**The stairs are the ones the robot has seen** (``MultiFloorParams.stair_source``,
``observed`` -- the default since 2026-10-21 and the only source since
2026-11-15): the RGB-D support-surface discovery (:mod:`stair_terrain`). A
reachable run of measured treads rising ``stair_min_rise_m`` or more within
5 m is a portal of the storey, placed from the measurement, and its path is
what the planning grid masks (:mod:`world_model.peek_stairs`, the spawn-floor
guard). No simulator geometry is read -- the footing of the published
baselines. With stair traversal forbidden (the benchmark default) that mask,
the observed drops and the observed map's own obstacle band are the whole of
the stair handling; with it allowed, a floor change is decided by the
``floor_search_actions`` allowance and look-down inspections
(:meth:`MultiFloorSearch._select`, :meth:`MultiFloorSearch._inspect`) and
walked on measured tread support (:mod:`stair_traversal`) -- which once
started a "traversal" on a 13 cm raised bathroom floor.

Until 2026-11-15 there was a second source, ``ground_truth``: stair
connectors and storey levels the evaluator read from the simulator's navmesh
and attached to the episode metadata, sighted by a "perfect detector" and
climbed along the connector's centreline. That was privileged scene geometry
-- no published zero-shot baseline is told where the stairs are or how many
storeys the building has -- and it is removed with everything that read it;
asking for it raises (``floor_atlas.REMOVED_STAIR_SOURCES``). It was also
the only source that told a portal which storey its flight leads to, which
the room-search loop needed to offer a staircase to the oracle and RPT* as a
node beside the rooms; that side -- ``rooms/stair_nodes.py``,
``stair_approach.py``, the commitment API here and the two parameters that
priced it -- lay dormant from then and was deleted on 2026-11-16, until the
multi-storey work brings it back. No staircase is a node of the search, and
nothing but the allowance decides a floor change.

The floor atlas, the per-floor contexts and the portal bookkeeping are as
they were.

**A spawn on the stairs** (since 2026-10-26). The atlas makes a storey of
the first pose, and the spawn-floor guard confines the search to it; when
that pose is a tread, the "storey" is the middle of a flight -- nothing is
level with it, every flight off it is masked, and the agent stands boxed in
by its own rules (00873/000002, ``02cc26973cab``: three treads down, the
trail retreat back, then 447 actions of probing the stairwell's walls). The
spawn storey is therefore **provisional** until the agent has walked
``stable_distance_m`` on it at its height (:attr:`MultiFloorSearch.storey_confirmed`,
the atlas's own plateau criterion); a height departure before that is the
agent leaving the staircase it spawned on, and with traversal forbidden it
is walked as an observed traversal toward the nearer storey the terrain can
see -- not retreated from -- and the storey it settles on becomes the spawn
storey (``SpawnFloorGuard.rebase``). A departure from a confirmed storey is
the accidental flight as before.

Three rules since 2026-11-02, from the 174 HM3D v1 val episodes that start on
stairs (``runs/stair_spawn``; 86 of them with the target 0.45-2 m up or down
succeeded 9 % of the time, and most failures never left the start height):

* **A storey is confirmed on support of floor size** (``confirm_area_m2``):
  walking ``stable_distance_m`` across a landing and onto the first tread had
  confirmed the landing, and the flight became a departure to retreat from.
* **The flight is walked on purpose** (:meth:`_spawn_departure_due`): no
  frontier route leads off a landing, so a spawn storey still provisional at
  ``spawn_leave_after_actions``, with a flight seen from it and less than
  ``min_floor_area_m2`` of level support under the agent, walks that flight to
  the nearer storey.
* **One switch to the flight's other end** (room policy only, ``spawn_switch.py``): once the
  nearer storey is searched out, the agent walks back along the flight to the
  storey at its far end -- once, and that storey is final. Two storeys at
  most, never back: the agent cannot go up and down between floors.
"""
from __future__ import annotations

import math
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.room_search_policy import RoomCandidate
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.object_search.multifloor.floor_departure import FloorDepartureGuard
from sparx_agency.core.planning.object_search.world_model.stair_terrain import StairTerrain
from sparx_agency.core.planning.object_search.multifloor.stair_traversal import StairTraversal
from sparx_agency.core.planning.object_search.multifloor.spawn_switch import SpawnStairsSwitch

#: A flight's far end must lie this much (half a riser) farther from the storey plane than the mask geometry's before
#: the mask takes the newer path.
MASK_PATH_GROWTH_M = 0.08


def _flight_extent(path, anchor_z) -> float:
    """How far from the storey plane ``anchor_z`` the flight path's far end lies, metres; 0 for an empty path.

    The far end is the top (or bottom) of the flight the terrain has measured
    so far, which grows as more treads are seen; the near end is trimmed at
    the agent's own height and says nothing about the flight.
    """
    if not len(path):
        return 0.0
    base = float(path[0][2]) if anchor_z is None else float(anchor_z)
    return abs(float(path[-1][2]) - base)


class MultiFloorSearch:
    def __init__(self, policy, *, confinement_only=False):
        self.policy, self.params = policy, policy.settings.multifloor
        self.confinement_only = confinement_only
        s = policy.settings
        #: ``RPTSettings.allow_stair_traversal``: False confines the search to the spawn
        #: storey -- stairs are still SEEN and placed (the guard masks them), never taken.
        self.allowed = s.allow_stair_traversal
        self.terrain = StairTerrain(self.params, s.map_resolution_m, s.body_radius_m, s.body_height_m, s.depth_stride)
        # A storey is confirmed by depth-measured flat support of floor size: nobody tells the atlas the storey heights.
        policy.mapping.atlas.plateau_observer = self.terrain.plateau_area
        self.portals = []
        self.active = None
        self.entered_step = 0
        self.floor_id = 0
        self.events = []
        self.transition = None
        self.completed = []
        self._observed_step = None
        self.semantic_stair_hint = False
        #: A stair detection seen while a route was committed is kept until the
        #: next action without one, rather than interrupting the route.
        self.stair_hint_pending = False
        self.safety_vetoes = 0
        self.source_trail = []
        #: Whether the spawn storey is a storey at all: True once the agent has walked ``stable_distance_m`` on it
        #: at its height, or arrived on a storey. False while a tread may be all the atlas's floor 0 ever was.
        self.storey_confirmed = False
        self._small_plateau_seen = False   # one event per episode for a level walk on a landing or a tread
        self.flights_seen_from_spawn = False   # the terrain saw a flight while the spawn storey was provisional
        self._spawn_flights = []               # the flight candidates last seen from it
        self._departures = 0                   # deliberate departures from a stair spawn started (bounded)
        self._departure_ended = None           # the action the last one ended back at the spawn storey's height
        self._last_decision = None
        #: ``stairs_id`` of the stairs the agent last came up or down (None when its portal carries none -- every
        #: portal of the observed discovery), and the action it arrived on. For the record.
        self.arrived_by = None
        self.arrived_step = None
        self._departure_ignored = False  # one event per off-level excursion, not one per action
        #: The forbidden-mode retreat in force (:meth:`_retreat_to_storey`), or None.
        self._retreat = None
        self.departure = None if confinement_only else FloorDepartureGuard(self)
        #: The one switch to the other storey of the flight a stair spawn settled off (``spawn_switch.py``).
        self.switch = None if confinement_only else SpawnStairsSwitch(self)

    @property
    def phase(self):
        if self.transition:
            return self.transition.phase
        if self.retreating:
            return "RETREAT"
        return "APPROACH_STAIRS" if self.active is not None else "SEARCH"

    @property
    def retreating(self):
        """Whether the forbidden-mode trail retreat is in force: started and not given up (a given-up one is the floor's search)."""
        return self._retreat is not None and not self._retreat["given_up"]

    @property
    def path(self):
        return self.transition.path if self.transition else []

    @property
    def traversing(self):
        return self.transition is not None or self.policy.mapping.atlas.in_transition

    @property
    def committed(self):
        return self.active is not None or self.traversing

    def can_leave_floor(self, obs=None):
        """Pending unknown rooms block departure; classified or attempted rooms do not."""
        return self.departure is not None and self.departure.ready(obs)

    def prepare_observation(self, obs):
        """Current depth before atlas settlement; never previous-view support."""
        if obs.step == self._observed_step:
            return
        self._observed_step = obs.step
        atlas = self.policy.mapping.atlas
        height = atlas.elevation_m if atlas.floors else obs.pose.z
        floor_world = self.policy.mapping.worlds.get(self.floor_id)
        self.terrain.traversing = self.transition is not None
        self.terrain.update(obs, floor_world, height)
        if not self.storey_confirmed and self.terrain.candidates:
            self.flights_seen_from_spawn = True       # a flight in view of the spawn storey: a stair spawn's flight
            self._spawn_flights = [dict(c) for c in self.terrain.candidates]
        if self.transition is None and abs(obs.pose.z - height) <= self.params.stable_height_m:
            self._departure_ignored = False
            xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
            if not self.source_trail or math.dist(xyz[:2], self.source_trail[-1][:2]) > 0.04:
                self.source_trail.append(xyz)
                self.source_trail = self.source_trail[-24:]
            if not self.storey_confirmed and math.dist(xyz[:2], self.source_trail[0][:2]) >= self.params.stable_distance_m:
                self._confirm_storey(obs)
        if self.transition is None and atlas.floors and abs(obs.pose.z - height) > self.params.stable_height_m:
            if not self.allowed:
                if self.storey_confirmed:
                    self._forbid_departure(obs, height)
                # A provisional spawn storey: the atlas's own departure test decides, in ``observe``.
            else:
                direction = 1 if obs.pose.z > height else -1
                if self.active is None:
                    self.active = self._portal(obs, direction, [(obs.pose.x, obs.pose.y, height)])
                self._start(obs)
        if self.transition is not None:
            self.transition.observe(obs)

    def _spawn_departure_due(self, obs):
        """Stair traversal forbidden: the agent still stands on the provisional spawn storey, a landing or a tread.

        Waiting for the search to walk it onto the flight by accident left it boxed in: from a landing no frontier
        route leads up or down (00800/000028: the flight up seen, 300 tread cells walkable, 500 actions at the
        spawn height). After ``spawn_leave_after_actions``, with a flight seen from it and the level support under
        the agent smaller than ``min_floor_area_m2``, the flight is walked to the nearer storey on purpose. On the
        probes of 2026-11-02 that fired for 6 of 7 stuck stair starts and none of 47 floor starts. One bumped back
        to the spawn storey's height is tried again ``spawn_leave_after_actions`` later, from wherever the agent
        then stands, up to ``spawn_departure_attempts`` in all.
        """
        if self.allowed or self.storey_confirmed:
            return False
        if self._departures >= self.params.spawn_departure_attempts or not self.flights_seen_from_spawn:
            return False
        if self._departures and self._departure_ended is None:
            return False                                # one under way
        since = obs.step if self._departure_ended is None else obs.step - self._departure_ended
        if since < self.params.spawn_leave_after_actions:
            return False
        atlas = self.policy.mapping.atlas
        if not atlas.floors or atlas.in_transition or abs(obs.pose.z - atlas.elevation_m) > self.params.stable_height_m:
            return False
        return float(self.terrain.plateau_area(obs.pose)) < self.params.min_floor_area_m2

    def visual_spawn_pending(self, obs):
        """A measured flight on a small provisional platform, or a departure before the first floor walk."""
        if not self.confinement_only or self.storey_confirmed:
            return False
        if self.transition is not None:
            return True
        if self.retreating:
            return False
        atlas = self.policy.mapping.atlas
        departed = (self._departures < self.params.spawn_departure_attempts and atlas.floors
                    and abs(obs.pose.z - atlas.elevation_m) > self.params.departure_m)
        return bool(departed or (self.flights_seen_from_spawn
                                 and self.terrain.plateau_area(obs.pose) < self.params.min_floor_area_m2))

    def visual_spawn_command(self, obs):
        """Own motion until the first supported floor; never select a portal from a confirmed floor."""
        if self.transition is not None:
            return self.transition.plan(obs)
        if not self.visual_spawn_pending(obs):
            # The guard or atlas settled during this observation. The next frame uses the destination grid.
            return NavigationCommand.hold(info={"kind": "spawn_handoff"})
        if self._departures >= self.params.spawn_departure_attempts:
            return NavigationCommand.stop_here(info={"kind": "SAFE_HALT", "target_confirmed": False,
                                                      "reason": "stair spawn: no supported floor after bounded attempts"})
        delta = obs.pose.z - self.policy.mapping.atlas.elevation_m
        since = obs.step if self._departure_ended is None else obs.step - self._departure_ended
        if abs(delta) <= self.params.departure_m and since < self.params.spawn_leave_after_actions:
            self.policy.camera_control.begin_inspection(obs, self.floor_id)
            return (self.policy.camera_control.inspection_command(obs)
                    or NavigationCommand.hold(final_yaw=obs.pose.yaw + self.policy.episode.action_spec.turn_angle_rad,
                                              info={"kind": "spawn_survey"}))
        return self._leave_spawn(obs, departed=1 if delta >= 0 else -1)

    def _leave_spawn(self, obs, departed=1):
        """Walk the flight seen from a stair spawn toward the nearer storey; at most ``spawn_departure_attempts`` times."""
        self._departures += 1
        self._departure_ended = None
        flights = sorted(self._spawn_flights, key=lambda c: float(c["rise_m"]))
        direction = int(flights[0]["direction"]) if flights else departed
        height = self.policy.mapping.atlas.elevation_m
        # The flight seen is the traversal's guide: from where the agent stands the terrain may no longer
        # see it (00802/000008 turned in place for 18 actions and retreated without a tread climbed).
        guide = [tuple(float(v) for v in point) for point in (flights[0].get("path") or [])] if flights else []
        self.active = self._portal(obs, direction, [(obs.pose.x, obs.pose.y, height)] + guide)
        self.active["selected_by"] = "spawn_departure"
        self.events.append({"action": obs.step, "event": "spawn_stairs_departure", "attempt": self._departures,
                            "direction": "up" if direction > 0 else "down",
                            "plateau_m2": round(float(self.terrain.plateau_area(obs.pose)), 2),
                            "flights_seen": [{"direction": "up" if c["direction"] > 0 else "down",
                                              "rise_m": round(float(c["rise_m"]), 2)} for c in flights],
                            "reason": "a stair spawn on a landing or a tread: the search alone never walks it onto "
                                      "the flight; walking it to the nearer storey, which becomes the spawn storey"})
        return self.transition.plan(obs) if self._start(obs, force=True) else None

    def _confirm_storey(self, obs):
        """Walked ``stable_distance_m`` level: a storey if the level support under the agent is of floor size.

        A landing or a tread is walked level too -- across it, onto the next
        tread within ``stable_height_m`` -- and confirming it made the flight
        a departure to retreat from (00800/000040, boxed in for 460 actions);
        it stays provisional, so the flight is walked to the storey it joins.
        """
        area = float(self.terrain.plateau_area(obs.pose))
        if area >= self.params.confirm_area_m2:
            self.storey_confirmed = True          # walked level this far on support of floor size: a storey
            self.events.append({"action": obs.step, "event": "spawn_storey_confirmed", "plateau_m2": round(area, 2)})
        elif not self._small_plateau_seen:
            self._small_plateau_seen = True
            self.events.append({"action": obs.step, "event": "spawn_storey_kept_provisional", "plateau_m2": round(area, 2),
                                "reason": "walked level on support smaller than confirm_area_m2: a landing or a tread"})

    def _forbid_departure(self, obs, height):
        """Stair traversal is off: a height departure is recorded once per excursion and starts nothing.

        The atlas alone decides what the excursion was; the guard refuses
        every goal while the storey in force is off the spawn plane.
        """
        if self._departure_ignored:
            return
        self._departure_ignored = True
        self.events.append({"action": obs.step, "event": "height_departure_ignored",
                            "height_m": round(obs.pose.z - height, 3),
                            "reason": "stair traversal disabled (allow_stair_traversal is false)"})

    def _retreat_to_storey(self, obs):
        """Traversal forbidden, the atlas in transition, no traversal of ours: walk the storey trail back.

        The agent is on a flight it was never meant to take -- a committed
        route whose end lay past the top tread walked it down three risers
        before the atlas called the height a departure (00814/000009,
        00873/000049, 2026-10-25; 00873/000002 turned out to have SPAWNED on
        the flight, which since 2026-10-26 is :meth:`_leave_spawn_stairs`,
        not a retreat). Nothing is climbed here and
        no map is read: the planning world is blank while the atlas is in
        transition. The retreat follows the storey poses the coordinator
        kept (``source_trail``: the last poses within ``stable_height_m`` of
        the plane) from the one nearest the agent back to the earliest -- far
        enough onto the plane for the atlas's plateau test to record a
        source return, after which the search resumes with the flight's foot
        among the observed drops. A retreat that has regained nothing within
        ``retreat_actions``, or whose trail is spent, is given up once: the
        floor's search and the spawn-floor guard's half-level adoption then
        take over, as they would have without the crash.
        """
        atlas = self.policy.mapping.atlas
        here = (float(obs.pose.x), float(obs.pose.y))
        if self._retreat is None:
            trail = [tuple(float(v) for v in point) for point in self.source_trail]
            if not trail:
                return None
            self._retreat = {"started": int(obs.step), "trail": trail, "index": len(trail) - 1, "given_up": False,
                             "blocked": 0}
            self.events.append({"action": obs.step, "event": "retreat_started", "portal_id": None,
                                "reason": "height departure with stair traversal disabled",
                                "height_m": round(float(obs.pose.z) - float(atlas.elevation_m), 3),
                                "trail_points": len(trail)})
            self.policy.route_memory.clear("retreat to the storey plane")
        state = self._retreat
        if abs(float(obs.pose.z) - float(atlas.elevation_m)) <= self.params.stable_height_m and obs.step > state["started"]:
            # Back at the plane: confirm the return now, given up or not. The atlas's own plateau test wants
            # three translated samples at the height, which a retreat ending at the flight's head may never
            # supply (00873/000002, 2026-10-25: back on the landing at action 23, "in transition" for 477 more).
            if atlas.settle(obs.pose, float(atlas.elevation_m)):
                self._end_retreat(obs, atlas.completion_reason or "storey height regained")
                return None
        if state["given_up"]:
            return None
        if obs.step - state["started"] >= self.params.retreat_actions:
            state["given_up"] = True
            self.events.append({"action": obs.step, "event": "retreat_given_up",
                                "reason": "retreat_actions exhausted off the storey plane"})
            return None
        if state["blocked"] >= self.params.retreat_blocked:
            # The trail is not walkable back (a ledge dropped off, a step the body cannot climb): pushing at
            # it for the rest of the allowance is what 00802/000015 did for 351 actions. The floor's search and
            # the spawn-floor guard's half-level adoption take over now (since 2026-10-28).
            state["given_up"] = True
            self.events.append({"action": obs.step, "event": "retreat_given_up",
                                "reason": "%d refused steps on the storey trail" % state["blocked"]})
            return None
        trail = state["trail"]
        reached = self.policy.converter_params.goal_tolerance_m + self.policy.settings.map_resolution_m
        while state["index"] >= 0 and math.dist(trail[state["index"]][:2], here) <= reached:
            state["index"] -= 1
        if state["index"] < 0 and not state.get("extended"):
            # The trail is walked and the plane not regained: the storey lies past its last point (a trail of
            # one pose, the spawn, when the first step was already a tread). One point two steps beyond it,
            # along the way back, so the follower has a goal past the edge; then the retreat is spent.
            last = trail[0][:2]
            dx, dy = last[0] - here[0], last[1] - here[1]
            span = math.hypot(dx, dy)
            if span < 0.05:
                dx, dy, span = math.cos(obs.pose.yaw), math.sin(obs.pose.yaw), 1.0
            reach = 2.0 * self.policy.episode.action_spec.forward_step_m
            state["trail"] = trail = [(last[0] + reach * dx / span, last[1] + reach * dy / span)]
            state["index"], state["extended"] = 0, True
        if state["index"] < 0:
            state["given_up"] = True
            self.events.append({"action": obs.step, "event": "retreat_given_up", "reason": "storey trail spent"})
            return None
        points = [here] + [trail[i][:2] for i in range(state["index"], -1, -1)]
        return NavigationCommand.follow(points, info={"kind": "RETREAT", "portal": None,
                                                      "reason": "off the storey with stair traversal disabled; "
                                                                "walking the storey trail back",
                                                      "trail_points_left": state["index"] + 1})

    def _end_retreat(self, obs, reason):
        if self._retreat is None:
            return
        self.events.append({"action": obs.step, "event": "retreat_ended", "reason": reason,
                            "actions": int(obs.step) - self._retreat["started"]})
        self._retreat = None
        self.policy.route_memory.clear("retreat ended")

    def observe(self, obs):
        atlas = self.policy.mapping.atlas
        if self.switch is not None:
            self.switch.accept_arrival(obs)
        # A floor one riser from the start matches the atlas's original id.
        # Finishing the verified exit there is an arrival, not a failed return to the tread.
        settled_spawn = False
        t = self.transition
        if (self.confinement_only and t is not None and t.retreat_path is None and t.arrival_allowed
                and not atlas.in_transition and atlas.active_id == self.floor_id
                and atlas.completion_reason == "source_platform_returned"):
            settled_spawn = atlas.remeasure(t.destination_height_m)
            if settled_spawn:
                self.policy.floor_guard.rebase(obs, self.floor_id, atlas.elevation_m, "verified floor beside the spawn tread")
        if atlas.active_id != self.floor_id or settled_spawn:
            old = self.floor_id
            self.floor_id, self.entered_step = atlas.active_id, obs.step
            was_provisional = not self.storey_confirmed
            self.storey_confirmed = True              # the atlas settled on it: a storey, whatever the spawn was
            old_height = atlas.floors[old].elevation_m if old in atlas.floors else obs.pose.z
            new_height = atlas.floors[self.floor_id].elevation_m
            if self.switch is not None:
                self.switch.arrived(obs, self.floor_id, new_height)
            if was_provisional and not self.allowed and not self.confinement_only:
                self.switch.settled(obs, self.floor_id, old_height, new_height, "arrival")
            reason = self.transition.completion_reason if self.transition else atlas.completion_reason
            self.events.append({"action": obs.step, "event": "floor_arrival", "source": old,
                                "destination": self.floor_id, "reason": reason})
            if self.active is not None:
                self.active["destination"] = self.floor_id
                self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions   # a rest for a portal just walked
                self.active["edge_id"] = atlas.last_connection
                self.active.pop("selected_by", None)
                self.arrived_by, self.arrived_step = self.active.get("stairs_id"), obs.step
            if self.transition:
                self.completed.append(dict(self.transition.diagnostics(), destination=self.floor_id, action=obs.step))
            self.active = None
            self.transition = None
            self.source_trail = []
            self._last_decision = None
            self.policy.route_memory.clear("completed floor transition")
        elif self.transition and self.transition.retreat_path is not None and not atlas.in_transition:
            self._abandon(obs, self.transition.completion_reason)
        elif (self.transition and not atlas.in_transition and atlas.completion_reason == "source_platform_returned"
              and abs(obs.pose.z - atlas.elevation_m) <= self.params.stable_height_m):
            # The atlas settled back on the floor the traversal left (the guard's known-floor rule): no tread was
            # taken, and the traversal is over (2026-11-02).
            self._abandon(obs, "settled back on the source floor without a tread taken")
        if self._retreat is not None and not atlas.in_transition:
            self._end_retreat(obs, atlas.completion_reason or "atlas settled")
        if not self.traversing:
            if not atlas.floors or abs(obs.pose.z - atlas.elevation_m) <= self.params.stable_height_m:
                self._discover(obs)
        if atlas.in_transition and self.transition is None:
            if self.visual_spawn_pending(obs):
                return  # visual_spawn_command owns this departure, without a room-coverage gate
            if not self.allowed and (self.storey_confirmed or self.confinement_only):
                self._forbid_departure(obs, atlas.elevation_m)
            else:
                direction = 1 if obs.pose.z > atlas.elevation_m else -1
                if not self.allowed:
                    direction = self._leave_spawn_stairs(obs, atlas.elevation_m, direction)
                if self.active is None:
                    self.active = self._portal(obs, direction, [(obs.pose.x, obs.pose.y, atlas.elevation_m)])
                self._start(obs)

    def _leave_spawn_stairs(self, obs, height, departed):
        """Traversal forbidden, the spawn storey never walked: the agent spawned on a flight and is leaving it.

        Nothing confines a search to the middle of a staircase, so the
        departure is walked as a traversal rather than retreated from, and
        the storey it settles on becomes the spawn storey (the guard's
        ``rebase``). The direction is the nearer storey the terrain can see
        -- the flight candidate with the smaller rise from here, up or down
        -- and the way already taken when it sees none: the observed map
        walks the agent DOWN a flight by itself (the treads below the plane
        are cleared by the rays over them, the ones above are obstacles), so
        the accidental direction alone would never choose the storey three
        treads up (00873/000002, the bed upstairs).
        """
        candidates = sorted(self.terrain.candidates, key=lambda c: float(c["rise_m"]))
        direction = int(candidates[0]["direction"]) if candidates else int(departed)
        self.events.append({"action": obs.step, "event": "spawn_storey_provisional",
                            "height_m": round(float(obs.pose.z) - float(height), 3),
                            "direction": "up" if direction > 0 else "down",
                            "flights_seen": [{"direction": "up" if c["direction"] > 0 else "down",
                                              "rise_m": round(float(c["rise_m"]), 2)} for c in candidates],
                            "reason": "height departure before the spawn storey was ever walked level: the agent "
                                      "spawned on a staircase; walking it to the nearer storey, which becomes the "
                                      "spawn storey"})
        return direction

    def _portal(self, obs, direction, path):
        atlas = self.policy.mapping.atlas
        portal = {"id": len(self.portals), "floor_id": self.floor_id,
                  "entry": list(path[0]), "direction": direction, "path": list(path),
                  # The storey plane the flight was placed from: the capsule is drawn about it, not about the plane
                  # in force later (a re-measure elsewhere on the storey must not move a staircase, peek_stairs.py).
                  "anchor_z": float(atlas.elevation_m) if atlas.floors else float(obs.pose.z),
                  # The flight geometry the MASK is drawn from (``stair_peek_mask``): what was measured of the flight,
                  # grown only as more of it is measured (``_extend_mask_path``), never the route from where the
                  # agent stands now -- ``path`` is that route, kept fresh for an approach.
                  "mask_path": list(path),
                  "destination": None, "cooldown_until": 0, "observations": 1, "last_step": obs.step}
        self.portals.append(portal)
        return portal

    @staticmethod
    def _extend_mask_path(portal, path):
        """Replace the portal's mask geometry only when more of the flight is on record than before.

        A terrain candidate's path is the route from the agent's own cell to
        the top of the flight, trimmed at the agent's height, so it changes
        with every action the agent takes -- and until 2026-11-03 the mask
        followed it: 00878-XB4GS9ShBRE/000000 redrew its capsules on 47 of
        500 actions, 627 cells out of the camera's view, as the agent's own
        level alternated between 0.0 and 0.2 m. What grows when more treads
        are measured is how far the flight's far end lies from the storey
        plane (:func:`_flight_extent`); the mask takes a path whose extent
        exceeds the one on record by more than half a riser, and nothing else.
        """
        anchor = portal.get("anchor_z")
        current = portal.get("mask_path") or []
        if _flight_extent(path, anchor) > _flight_extent(current, anchor) + MASK_PATH_GROWTH_M:
            portal["mask_path"] = list(path)

    def _discover(self, obs):
        for candidate in self.terrain.candidates:
            path = candidate["path"]
            near = [i for i, xyz in enumerate(path) if abs(xyz[2] - obs.pose.z) < 0.20]
            entry_index = max(0, (near[-1] if near else 0) - 2)
            path = path[entry_index:]
            match = next((p for p in self.portals if p["floor_id"] == self.floor_id
                          and p["direction"] == candidate["direction"]
                          and math.dist(p["entry"][:2], path[0][:2]) < 1.25), None)
            if match is None:
                self._portal(obs, candidate["direction"], path)
            elif match["last_step"] != obs.step:
                match["observations"] += 1
                match["last_step"] = obs.step
                # Preserve commitment, not stale pre-approach geometry. An
                # executed edge is different: its measured trail is immutable.
                if "edge_id" not in match:
                    match["path"] = path
                    self._extend_mask_path(match, path)
        for edge in self.policy.mapping.atlas.connections:
            if self.floor_id not in (edge.source, edge.destination):
                continue
            destination, path = edge.oriented(self.floor_id)
            if not any(p.get("edge_id") == edge.id and p["floor_id"] == self.floor_id for p in self.portals):
                portal = self._portal(obs, 1 if path[-1][2] > path[0][2] else -1, path)
                portal.update(destination=destination, edge_id=edge.id, observations=2,
                              cooldown_until=obs.step + self.params.floor_search_actions)

    def plan(self, obs, world, exhausted=False):
        """The building's command for this action, or None when the floor's search owns it.

        A portal is selected once the floor's ``floor_search_actions``
        allowance is spent, or at once when ``exhausted`` (the exploration
        fallback saying the floor has nothing left), by RPT* over the seen
        portals (:meth:`_select`); an active portal is approached and its
        flight walked; otherwise a look-down inspection may be due.

        With stair traversal forbidden nothing is selected, approached or
        climbed here, and a portal somehow left active is deferred.
        """
        if self.confinement_only:
            if self.visual_spawn_pending(obs):
                return self.visual_spawn_command(obs)
            if self.traversing:
                return self._retreat_to_storey(obs)
            if self.semantic_stair_hint or self.stair_hint_pending:
                return self._inspect(obs, exhausted=False)
            return None
        if not self.allowed and not self.traversing:
            # The one exception: the switch to the other storey of the flight a stair spawn came off (spawn_switch.py).
            if self.active is not None and not self.switch.owns(self.active):
                self._abandon(obs, "stair traversal disabled (allow_stair_traversal is false)")
                return None
            if self.active is None and self._spawn_departure_due(obs):
                return self._leave_spawn(obs)
            if self.active is None:
                if not self.switch.due(obs, exhausted):
                    return None
                self.switch.begin(obs)
            return self.switch.plan(obs, world)
        if self.traversing:
            if self.transition is None:
                # Traversal forbidden: the atlas alone says the agent is off its
                # storey (``_forbid_departure`` started nothing). Until 2026-10-25 this line
                # dereferenced the missing traversal on every action for the rest of the
                # episode (00814, 00873 x2: three treads down, then 400+ recorded crashes).
                return self._retreat_to_storey(obs)
            return self.transition.plan(obs)
        if self.active is not None and not self.can_leave_floor(obs):
            # A room may be discovered while approaching a previously allowed
            # staircase. Recheck before taking even the first tread.
            self._abandon(obs, "new room needs a peek before floor departure")
            return None
        if self.active is None and (exhausted or obs.step - self.entered_step >= self.params.floor_search_actions):
            self._select(obs, world)
        if self.active is not None:
            entry = tuple(self.active["entry"][:2])
            here = (obs.pose.x, obs.pose.y)
            kind = "portal/%d" % self.active["id"]
            arrived = self.policy.route_memory.kind == kind and self.policy.route_memory.arrived(obs)
            if math.dist(here, entry) <= 0.65 or arrived:
                return self.transition.plan(obs) if self._start(obs) else None
            command = self.policy._navigate(obs, world, entry, kind)
            if command is not None:
                return command
            self._abandon(obs, "portal approach unavailable")
        return self._inspect(obs, exhausted)

    def _inspect(self, obs, exhausted):
        """Start a look-down sweep only at a natural pause, and say what asked for it.

        The three triggers, in precedence: the floor has no frontier left
        (``exhausted`` -- the robot has nothing else to do); a stair detection
        is pending; or the floor allowance is spent and the long periodic
        cadence has elapsed. The last two wait until no route is committed,
        because a sweep mid-route costs its own actions plus the turns to
        recover the heading it leaves behind. The former per-cooldown timer
        interrupted routes ten times in one 472-action recording.
        """
        camera = self.policy.camera_control
        if self.semantic_stair_hint:
            self.stair_hint_pending = True
        idle = self.policy.route_memory.path is None
        allowance_spent = obs.step - self.entered_step >= self.params.floor_search_actions
        if exhausted:
            reason = "floor_exhausted"
        elif idle and self.stair_hint_pending:
            reason = "stair_detection"
        elif idle and allowance_spent and camera.periodic_due(obs.step):
            reason = "periodic"
        else:
            reason = None
        if reason is not None and camera.begin_inspection(obs, self.floor_id):
            self.stair_hint_pending = False
            self.events.append({"action": obs.step, "event": "inspection_started", "reason": reason,
                                "floor_id": self.floor_id})
        return camera.inspection_command(obs)

    def _select(self, obs, world):
        if not self.can_leave_floor(obs):
            return
        p = self.policy
        eligible = [q for q in self.portals if q["floor_id"] == self.floor_id
                    and q["observations"] >= 2 and obs.step >= q["cooldown_until"]]
        if not eligible:
            return
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.planner_params.inflate_floor_m)[0]
        goals = {q["id"]: tuple(q["entry"][:2]) for q in eligible}
        states = p.floors.save()
        probabilities = {}
        for q in eligible:
            state = states.get(q["destination"])
            probabilities[q["id"]] = 0.30 if state is None else max(0.02, min(0.30, sum(state["graph"].probs.values()) / (1 + state["_floor_time"] / 60)))
        instance, _ = build_instance(world, cost, goals, probabilities, depot_xy=(obs.pose.x, obs.pose.y),
                                     cruise_speed_mps=p.episode.action_spec.forward_step_m / p.settings.action_time_s)
        if instance is None:
            return
        candidates = [RoomCandidate(room_id=q["id"], label="observed stair portal", prob=probabilities[q["id"]],
                                    prob_renorm=probabilities[q["id"]], xy=goals[q["id"]]) for q in eligible]
        allowed = set(instance.index_to_pid)
        order = p.solver([q for q in candidates if q.room_id in allowed], instance)
        if order:
            self.active = next(q for q in eligible if q["id"] == order[0])
            p.route_memory.clear("RPT selected floor portal")
            self.events.append({"action": obs.step, "event": "portal_selected", "portal_id": self.active["id"],
                                "solver_source": p.solver.last.source, "order": list(order)})

    def _start(self, obs, force=False):
        """Begin the active portal's traversal; ``force`` skips the room-coverage gate (a stair spawn's landing)."""
        if self.transition is not None:
            return True
        allowed = self.can_leave_floor(obs)
        height = self.policy.mapping.atlas.elevation_m
        if not allowed and not force and abs(obs.pose.z - height) <= self.params.stable_height_m:
            self._abandon(obs, "rooms still need interior peeks")
            return False
        self.transition = StairTraversal(self, obs, first_floor=self.confinement_only)
        if not allowed and self.departure is not None:
            self.departure.recover_unplanned(obs)
        self.policy.route_memory.clear("stair traversal")
        self.events.append({"action": obs.step, "event": "traversal_started", "portal_id": self.active["id"],
                            "direction": self.active["direction"],
                            "stair_source": self.params.stair_source,
                            "selected_by": self.active.get("selected_by", "unplanned"),
                            "room_coverage_complete": allowed})
        h = self.policy.hierarchy
        if h is not None:
            if h.regions.mask is not None:
                h.routes.suspend(h.regions.mask, h.current, self.policy, h.machine.actions)
            h.machine.end("floor_departure")
            h.machine.transition("TRAVERSE", "observed stair traversal; global allowance only")
        return True

    def _abandon(self, obs, reason):
        # The atlas settled back on a storey is a verified return too: the traversal's own test needs level
        # support of floor size at the source, which a stair spawn's landing never has, and refusing here ended
        # 00800/000040 and 00802/000008 (2026-11-02) as agent errors back on the landing they had left.
        if self.transition and not self.transition.arrival_allowed and self.policy.mapping.atlas.in_transition:
            raise RuntimeError("Cannot abandon a committed transition before verified source return")
        self.events.append({"action": obs.step, "event": "portal_deferred", "reason": reason,
                            "portal_id": self.active["id"] if self.active else None})
        if self.switch is not None and self.switch.owns(self.active):
            self.switch.abandoned(obs, reason)
        if self.active and self.active.get("selected_by") == "spawn_departure":
            self._departure_ended = obs.step
        if self.active:
            self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions
            self.active.pop("selected_by", None)
        if self.transition:
            self.completed.append(dict(self.transition.diagnostics(), action=obs.step))
        self.active = None
        self.transition = None
        self.policy.route_memory.clear("portal deferred")
        if self.policy.hierarchy is not None:
            self.policy.hierarchy.machine.transition("room_reasoning", "stair proposal deferred")

    def blocked(self, obs):
        if self.transition:
            self.transition.failures += 1
            self.transition.path = []
            self.transition.surface_goal = None
        elif self.retreating:
            self._retreat["index"] -= 1      # this heading is refused: aim at the next trail point back
            self._retreat["blocked"] += 1
        self.terrain.blocked(obs.pose, self.policy.episode.action_spec.forward_step_m)

    def filter_action(self, obs, action):
        guarded = self.departure.filter_action(obs, action) if self.departure else action
        if guarded != action:
            return guarded
        if not self.traversing:
            h = self.policy.hierarchy
            if h is not None:
                h.motion.update(obs, self.policy.mapping.worlds[self.floor_id])
                if not h.motion.permits(obs, action):
                    return DiscreteAction.TURN_LEFT
            return action
        if self.transition is None:
            # The trail retreat walks ground the body stood on this episode: depth cannot veto it.
            return action
        if action == DiscreteAction.MOVE_FORWARD and not self.terrain.permits(obs.pose, self.policy.episode.action_spec.forward_step_m):
            self.safety_vetoes += 1
            return DiscreteAction.TURN_LEFT
        return action

    def diagnostics(self):
        return {"phase": self.phase, "stair_source": self.params.stair_source,
                "allow_stair_traversal": self.allowed, "storey_confirmed": self.storey_confirmed,
                "spawn_switch": self.switch.diagnostics() if self.switch else None,
                "atlas": self.policy.mapping.atlas.diagnostics(),
                "transition": self.transition.diagnostics() if self.transition else None,
                "trail_retreat": ({k: self._retreat[k] for k in ("started", "index", "given_up", "blocked")}
                                  if self._retreat is not None else None),
                "completed_transitions": list(self.completed), "safety_vetoes": self.safety_vetoes,
                "room_coverage": None if self.confinement_only else self.policy.peek.diagnostics()["coverage"],
                "coverage_vetoes": self.departure.vetoes if self.departure else 0,
                "floor_contexts": self.policy.floors.diagnostics() if not self.confinement_only else [],
                "events": list(self.events),
                "portals": [{k: v for k, v in q.items() if k != "path"} for q in self.portals],
                "active_portal": self.active["id"] if self.active else None,
                "selected_by": self.active.get("selected_by") if self.active else None,
                "entered_step": self.entered_step, "arrived_by": self.arrived_by, "arrived_step": self.arrived_step,
                "decides_floor_changes": "provisional stair spawn only; confined after settlement" if self.confinement_only else
                                      "allowance (%d actions) + RPT* over portals" % self.params.floor_search_actions}
