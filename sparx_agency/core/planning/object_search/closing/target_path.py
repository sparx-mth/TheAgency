"""Stable target standoff selection; visibility is not a route-validity test."""
from __future__ import annotations

import math

from sparx_agency.core.planning.object_search.closing.map_sight import MEMORY_SIGHT_MARGIN_M, map_sight_clear


class TargetApproachPath:
    """Own the standoff -- a standing point on the agent's own map -- while the existing route memory owns A*.

    A refinement of the 3-D memory beyond ``refine_distance_m`` used to drop
    the standoff goal and re-select one from a bearing-relative ring about
    the new position -- and the ring is a different set of points, each
    needing floor to stand on and a path, so a goal that still had a
    path could be traded for none (Collierville/000194, 2026-10-11: the
    along-ray association refined the television's memory by 0.11 m at
    2.5 m, the re-selected ring found no path from a spot the old goal was
    reachable from, and the lock was released as "no safe path from here";
    the un-refined baseline walked in and STOPped). Since then a goal that
    still lies within ``terminal_distance_m`` of the refined memory is KEPT
    (``kept``): arriving at it puts the agent inside the terminal radius,
    which is all a standoff is for, and the inspection reads the fresh
    range from there. The reference is refreshed on every refinement, so a
    re-selection that finds nothing is not counted again every action; and
    a kept goal that a slow drift carries past the terminal radius is
    re-selected the action it falls out, reference or no reference.

    **An arrived standoff is terminal** (since 2026-10-11, the same
    recordings). The executor reports ``arrived`` when the route to the goal
    is walked, and the headless agent spends an idle result on a TURN: at a
    kept standoff 1.15 m from the refined memory -- inside the evaluator's
    success region -- the agent turned twenty times until the stall clock
    released the lock as "went nowhere" (Collierville/000194 and /000043).
    ``arrived`` is read with the route memory's own rule the action the
    route completes, so the closing inspects from the standoff instead; a
    fresh range that still says "too far" drops the goal (:meth:`drop`) and
    the next selection is a closer ring about the memory.

    **A standoff is floor the agent has seen, with room for its body, that
    it can walk to on its own map** (since 2026-11-15,
    ``world_model/standing_floor.py``). Each ring point is moved to the
    nearest such cell within ``TargetClosingSettings.standing_snap_m`` and
    dropped when there is none: a cell observed free, every cell within the
    body radius of it observed free too, and joined to the agent through
    cells like it. The route is then planned to that cell exactly (the
    planner's goal snap is withheld): by the default planner, of the cells
    that keep its standoff (0.25 m) from every observed obstacle, and --
    when it has no route -- by the retry planner, of the cells that keep
    the body radius (0.18 m; ``TargetClosing.standing_planners``,
    ``TargetClosingSettings.standing_retry_clearance_m``: one floor, two
    planners -- the first paired runs asked the default planner only of
    cells with its whole 0.25 m disc observed free, which a map of a few
    frames rarely has, and the second planner, which prefers a tight short
    way, took a standoff in 32 of 85 episodes). Until then
    the point was projected onto the simulator's navigation mesh, which says
    where floor is whether or not the agent ever looked there, and when the
    known floor had no route the ring was planned THROUGH unknown cells to
    it (2026-10-14 to 2026-11-15, ``through_unknown``). Both are gone: no
    simulator geometry is read, and unknown cells are never a standoff,
    never part of its clearance and never on the way to it. A standoff kept
    from an earlier action is given up the action the map stops showing it
    as standing floor (``revoked``). A point is never moved across an
    obstacle the map shows (``standing_floor.nearest_cell_in_sight``): the
    floor nearest a ring point beside the wall a toilet stands against is
    not the next room's.

    **Floor not yet seen is looked at, not walked through.** A level camera
    leaves the floor under its view unknown for 1.4 m ahead, so the floor
    between a corridor and a toilet seen through its doorway may have no
    observed route (Collierville/000023, action 375). The closing's own
    rungs answer that without assuming anything about the unknown: the walk
    to the observed floor nearest the target, then the footing sweep
    (``TargetClosing._no_path``), with the ring tried again on every action
    as the floor between fills in.

    **The ring widens before the spot is given up, and a stale goal is kept
    over no goal** (since 2026-10-15, Collierville/000009). The standoff
    ring stands ``goal_tolerance_m`` inside the terminal radius so that
    arriving within the tolerance lands inside it -- 0.70 m about the 3-D
    memory. The memory is a point on the object's near surface, and a big
    object has no floor there: a 1.5 m television on its cabinet put every
    0.70 m point inside the cabinet, against its front or behind it, in both
    planners, from 3 m away, three times in one episode; each lock was
    released as "no safe path from here" with the television in view. When
    neither planner reaches the standoff ring, wider rings are tried in the
    front arc out to the terminal radius itself (``widened``): a goal there
    is reached within the tolerance up to one forward step outside the
    terminal range, which the inspection's step rule closes. And a goal a
    refinement carries just past the terminal radius -- 1.01 m against
    1.00, with a seventeen-waypoint route to it -- is dropped only for a
    ring point that has a route; when none has, the stale goal is kept
    while it lies within the terminal radius plus the tolerance
    (``stale_kept``).

    **The ring is drawn about the agent's own bearing, for every target**
    (again since 2026-11-16). From 2026-11-10 the ring of a raised planar
    target -- a picture, a wall cabinet's front -- was tried about its
    facing normal first, because the evaluator's view points of a thin
    object lie only in front of it. That is a property of how view points
    are generated, measured on validation goals, and nothing the agent
    observes asks for it: the side the target was seen from is a side it is
    seen from. The rule is removed (``TargetClosingSettings``).

    **Within a ring, the front arc with every planner before any point
    beside or behind, and a detour-long route last** (since 2026-11-05,
    ``BACK_ARC_FROM_DEG``, ``TargetClosingSettings.standoff_detour_factor``).
    A toilet, a bed, a television stands against a wall: the ring point
    behind it is in the wall or in the next room, and a known-floor route to
    it goes round the wall -- 00891/000006 and /000027 took the point behind
    their toilet on a 7.5 m route round the outside of the bathroom,
    inspected a wall from 0.9 m and STOPped 6 m of geodesic from the goal.
    So each ring is tried arc by arc -- the offsets under
    ``BACK_ARC_FROM_DEG`` with every planner, then the rest with any
    (``behind``, counted as ``back_arc``) -- and a point whose walk over
    standing floor is longer than the factor times the range plus the slack
    is not taken by :meth:`command` at all: it is kept as ``deferred``, and
    :meth:`take_deferred` adopts it only when the closing has nothing left
    to try from where it stands -- no walk toward the target on observed
    floor, no footing sweep still owed (``TargetClosing._no_path``;
    ``detours`` counts the ones taken).

    **While floor in front of the target is unobserved, its far side
    waits** (since 2026-11-15, :meth:`_front_unseen`). Until then the front
    arc was planned through the unknown before any point beside or behind
    was asked. Without those routes the front of a target seen through a
    doorway is often simply not mapped yet, and the floor that IS mapped
    near it is the next room's: so while a front-arc point of the agent's
    ring lies on floor nobody has observed, a standoff beside or behind the
    target, or one the map puts an obstacle between and the target
    (``closing/map_sight.py``), is set aside with the detours -- behind any
    detour that does face the target -- until the front has been looked at.
    Walking round the wall first would be the 00891 failure again. With the
    front observed nothing waits but the detours, as before.

    The order in full: ring (the standoff ring, then the widened ones), then
    arc, then planner, then the offsets of the arc.
    """

    #: The bearings tried about the memory, from the agent's side round to behind it.
    OFFSETS_DEG = (0, 30, -30, 60, -60, 90, -90, 180)
    #: The front arc alone for the widened rings: beside and behind a big object is where its floor is not.
    FRONT_OFFSETS_DEG = (0, 30, -30, 60, -60)
    #: Offsets of this magnitude and beyond are beside or behind the target: tried, with every planner, only after
    #: the front arc has been tried with every planner (``_arc``).
    BACK_ARC_FROM_DEG = 90

    def __init__(self, policy):
        self.policy = policy
        self.goal_xyz = self.reference_xyz = None
        self.refinements = 0
        self.kept = 0                      # refinements through which a still-valid standoff was kept
        self.arrived = False               # the route to the current standoff is walked (the executor's rule)
        self.arrivals = 0                  # standoffs reached over this lock
        self.dropped = 0                   # arrived standoffs given up for a closer one
        self.goal_planner = None           # the planner the current goal's route was found with (None: the default)
        self.retried = 0                   # standoffs the default planner had no route to (the retry clearance's)
        self.revoked = 0                   # standoffs given up because the map stopped showing them as standing floor
        self.deferred = None               # the last selection's shortest detour-long standoff, for ``take_deferred``
        self.widened = False               # the current goal lies beyond the standoff ring: no closer point had a route
        self.widenings = 0                 # standoffs found on a widened ring over this lock
        self.stale_kept = 0                # stale goals kept because no ring point had a route
        self.behind = False                # the current goal lies beside or behind the target (the back arc)
        self.back_arc = 0                  # standoffs taken beside or behind the target over this lock
        self.detours = 0                   # standoffs taken on a detour-long route, nothing else left to try, over this lock
        self.far_side = 0                  # standoffs taken on the target's far side with its front still unobserved

    def drop(self):
        """Give up the current standoff so the next :meth:`command` selects one anew, closer to the memory."""
        if self.goal_xyz is not None:
            self.dropped += 1
        self.goal_xyz = self.reference_xyz = self.deferred = None
        self.arrived = self.widened = self.behind = False
        self.policy.route_memory.clear("standoff_dropped")
        self.policy._route = self.policy._goal = None

    def command(self, target, obs, world):
        p = self.policy
        s = target.settings
        self.deferred = None
        stale = self.goal_xyz is not None and math.dist(self.goal_xyz[:2], target.xyz[:2]) > s.terminal_distance_m
        moved = self.reference_xyz is not None and math.dist(self.reference_xyz[:2], target.xyz[:2]) > s.refine_distance_m
        previous = None
        if moved or stale:
            self.refinements += 1
            self.reference_xyz = tuple(target.xyz)
            if self.goal_xyz is not None and not stale:
                self.kept += 1                  # the standoff still ends inside the terminal radius: no re-selection
            else:
                previous = self.goal_xyz        # a goal carried out of the terminal radius: the ring's fallback
                self.goal_xyz = None
                self.arrived = self.widened = self.behind = False
                p.route_memory.clear("target_refined")
                p._route = p._goal = None
        if self.goal_xyz is not None and not target.standing_floor(obs, world).standable_at(self.goal_xyz[:2]):
            # The map moved under the standoff: its cell, or one the body would cover there, no longer reads
            # FREE (an obstacle seen since, a cell the evidence has turned ambiguous). It is not a place to stand
            # any more, whatever route still leads to it. Only the CELL is asked about: whether the agent is
            # joined to it is the committed route's business, re-checked below like any other.
            self.revoked += 1
            target.standoffs_revoked += 1
            self.goal_xyz = None
            self.arrived = self.widened = self.behind = False
            p.route_memory.clear("standoff_not_standing")
            p._route = p._goal = None
        if self.goal_xyz is not None:
            # Do not generate a different bearing-relative endpoint every frame.
            # _navigate still checks collisions, blocked motion and path progress.
            command = p._navigate(obs, world, self.goal_xyz[:2], "target_closing", planner=self.goal_planner,
                                  exact_goal=True)
            if command is not None:
                reached = p.route_memory.arrived(obs)
                if reached and not self.arrived:
                    self.arrivals += 1
                self.arrived = reached
                return command
            self.goal_xyz = None
            self.arrived = self.widened = self.behind = False
        tolerance = p.converter_params.goal_tolerance_m
        limit = s.terminal_distance_m - tolerance
        radius = max(p.settings.body_radius_m, limit - world.resolution)
        # The ring is drawn about the agent's own bearing from the memory: the side the target was seen from.
        bearing = math.atan2(obs.pose.y - target.xyz[1], obs.pose.x - target.xyz[0])
        # The planners a standoff's route is asked of, each with the standing cells it can end on: the default
        # planner and the cells that keep its floor standoff from every observed obstacle, then -- since
        # 2026-10-25: the floor beside a sofa corner or a cabinet is often drawn narrower than
        # ``RPTSettings.min_clearance_m``, and 00890/000074 released its locked television, 3.5 m away in plain
        # view, as "no safe path from here" -- the retry planner and the cells that keep its clearance (0.18 m
        # since 2026-11-16; the body radius until then).
        planners = target.standing_planners()
        retry = getattr(p, "standing_retry_planner", None)
        # The standoff ring first, with every bearing and both planners; then, when no point of it has a
        # route, wider rings in the front arc out to the terminal radius (since 2026-10-15): a goal there is
        # reached within the tolerance up to a forward step outside the terminal range, which the inspection
        # closes with a step (``TargetClosing._inspect``) and STOPs on when it cannot (``closest_approach``).
        # The rings in the order tried, each with the offsets it is tried at: the widened ones in the front arc
        # alone.
        rings = [(radius, limit, self.OFFSETS_DEG)]
        widest = s.terminal_distance_m - world.resolution
        if widest > radius + world.resolution:
            reach = s.terminal_distance_m + world.resolution
            rings.append((0.5 * (radius + widest), reach, self.FRONT_OFFSETS_DEG))
            rings.append((widest, reach, self.FRONT_OFFSETS_DEG))
        # A walk much longer than the straight line to the target goes round something -- the wall the target
        # stands against, the point on its far side (since 2026-11-05, ``standoff_detour_factor``; see the setting):
        # such a point is not taken here. The shortest is left in ``deferred`` for :meth:`take_deferred`.
        range_m = math.dist((obs.pose.x, obs.pose.y), target.xyz[:2])
        allowance = self.detour_allowance(s, range_m)
        detour = None
        front_unseen = self._front_unseen(target, obs, world, radius)
        tried = set()                      # (planner, cell): two ring points often share their nearest standing cell
        for ring_radius, reach, offsets in rings:
            # The front arc -- the side the target was seen from -- with every planner before any point beside
            # or behind it with any (since 2026-11-05). A toilet, a bed, a television stands against a wall: the
            # ring point behind it is in the wall or in the next room, and a known-floor route to it goes round
            # the wall. 00891/000006 and /000027 saw their toilet through the bathroom doorway from 3 m, took the
            # point behind the toilet on a 7.5 m known-floor route round the outside of the bathroom, inspected a
            # wall from 0.9 m, and STOPped on the memory 6 m of geodesic from the goal.
            for arc in ("front", "back"):
                for planner, standoff_m in planners:
                    floor = target.standing_floor(obs, world, standoff_m)
                    for offset in self._arc(offsets, arc):
                        angle = bearing + math.radians(offset)
                        # The ring point is geometry about the target; where the agent can STAND near it is
                        # the map's to say: the nearest cell observed free with the body's clearance all
                        # observed free, joined to the agent through such cells. No such cell, no standoff.
                        point = (target.xyz[0] + ring_radius * math.cos(angle),
                                 target.xyz[1] + ring_radius * math.sin(angle))
                        spot = floor.nearest(point, s.standing_snap_m)
                        if spot is None or math.dist(spot, target.xyz[:2]) > reach:
                            continue
                        key = (id(planner), spot)
                        if key in tried:
                            continue
                        tried.add(key)
                        goal = (spot[0], spot[1], float(obs.pose.z))
                        behind = abs(offset) >= self.BACK_ARC_FROM_DEG
                        # Two kinds of standoff are set aside rather than taken (``deferred``). One whose walk
                        # over standing floor -- read off the floor itself, no route planned for it -- is a
                        # detour long. And, while floor in FRONT of the target is still unobserved, one on the
                        # target's far side: beside or behind it, or with something the map has seen standing
                        # between it and the target. The front was where the routes through the unknown used
                        # to go first; it is still to be looked at before the far side is walked to.
                        far_side = front_unseen and (arc == "back" or not map_sight_clear(
                            world, spot, target.xyz[:2], skip_to_m=MEMORY_SIGHT_MARGIN_M))
                        if far_side or math.isfinite(allowance):
                            gx, gy = world.world_to_grid(*spot)
                            length = float(floor.geodesic_m()[gy, gx])
                            if far_side or length > allowance:
                                rank = (far_side, length)        # a detour that faces the target before the far side
                                if detour is None or rank < detour[0]:
                                    detour = (rank, ring_radius > radius, planner, goal, behind, length > allowance)
                                continue
                        command = p._navigate(obs, world, spot, "target_closing", planner=planner, exact_goal=True)
                        if command is not None and math.dist(command.waypoints[-1], target.xyz[:2]) <= reach:
                            self._adopt(target, obs, goal, planner, widened=ring_radius > radius,
                                        retried=planner is not None and planner is retry, behind=behind)
                            return command
        self.deferred = detour
        if previous is not None and math.dist(previous[:2], target.xyz[:2]) <= s.terminal_distance_m + tolerance:
            # No ring point has a route, and the goal the refinement made stale -- by centimetres -- had one:
            # arriving at it puts the agent within a step of the terminal range, where the inspection reads
            # the fresh range. Kept over no goal at all (since 2026-10-15; Collierville/000009, action 15) --
            # while the map still shows it as floor the agent can stand on and walk to.
            for planner, standoff_m in planners:
                if not target.standing_floor(obs, world, standoff_m).contains(previous[:2]):
                    continue
                command = p._navigate(obs, world, previous[:2], "target_closing", planner=planner, exact_goal=True)
                if command is not None:
                    self.stale_kept += 1
                    target.stale_standoffs_kept += 1
                    self._adopt(target, obs, previous, planner, widened=True, retried=False)
                    return command
        return None

    def take_deferred(self, target, obs, world):
        """Adopt the standoff the last :meth:`command` set aside, or None when there is none to take.

        The shortest walk among the detour-long standoffs and the ones
        beside or behind a target whose front is still unobserved. For the
        caller that has nothing else left from where the agent stands
        (``TargetClosing._no_path``): it is a real standoff on observed
        floor, only on the far side or a long way round, and it is better
        than giving the target up.
        """
        if self.deferred is None:
            return None
        (far_side, _length), widened, planner, goal, behind, detour = self.deferred
        self.deferred = None
        command = self.policy._navigate(obs, world, goal[:2], "target_closing", planner=planner, exact_goal=True)
        if command is None:
            return None
        if detour:
            self.detours += 1
            target.standoff_detours += 1
        if far_side:
            self.far_side += 1
            target.far_side_standoffs += 1
        self._adopt(target, obs, goal, planner, widened=widened,
                    retried=planner is not None and planner is getattr(self.policy, "standing_retry_planner", None),
                    behind=behind)
        return command

    @staticmethod
    def detour_allowance(settings, gain_m):
        """The longest walk that is not a detour for the straight-line distance it covers; ``inf`` with the rule off.

        ``standoff_detour_factor`` times ``gain_m`` plus ``standoff_detour_slack_m``. For a standoff ``gain_m`` is
        the range to the target (the standoff is at the target, as near as makes no difference); for a walk to
        floor that is merely nearer the target, how much nearer (``TargetClosing._nearest_reachable``). Scalars or
        arrays alike.
        """
        if settings.standoff_detour_factor <= 0:
            return math.inf
        return settings.standoff_detour_factor * gain_m + settings.standoff_detour_slack_m

    def _front_unseen(self, target, obs, world, radius):
        """Whether a front-arc point of the agent's standoff ring lies on floor nobody has observed.

        The side the target was seen from, at the standoff radius: the
        bearings under ``BACK_ARC_FROM_DEG`` of the agent's own. An UNKNOWN
        cell there is floor (or not) that a look will settle; an occupied or
        a free one is settled already, whatever it says.
        """
        bearing = math.atan2(obs.pose.y - target.xyz[1], obs.pose.x - target.xyz[0])
        for offset in self._arc(self.OFFSETS_DEG, "front"):
            angle = bearing + math.radians(offset)
            if world.is_unknown(*world.world_to_grid(target.xyz[0] + radius * math.cos(angle),
                                                     target.xyz[1] + radius * math.sin(angle))):
                return True
        return False

    def _arc(self, offsets, which):
        """The offsets of one arc: ``"front"`` under ``BACK_ARC_FROM_DEG``, ``"back"`` the rest (beside or behind the target)."""
        if which == "front":
            return tuple(o for o in offsets if abs(o) < self.BACK_ARC_FROM_DEG)
        return tuple(o for o in offsets if abs(o) >= self.BACK_ARC_FROM_DEG)

    def _adopt(self, target, obs, goal, planner, widened, retried, behind=False):
        """Take ``goal`` as the standoff and record how it was found."""
        p = self.policy
        self.goal_xyz, self.reference_xyz = tuple(goal), tuple(target.xyz)
        self.goal_planner = planner
        self.widened = bool(widened)
        self.behind = bool(behind)
        if behind:
            self.back_arc += 1
            target.back_arc_standoffs += 1
        self.arrived = p.route_memory.arrived(obs)      # a ring point the agent already stands on
        if self.arrived:
            self.arrivals += 1
        if widened:
            self.widenings += 1
            target.widened_standoffs += 1
        if retried:
            self.retried += 1
            target.retry_standoffs += 1
