"""Stable target standoff selection; visibility is not a route-validity test."""
from __future__ import annotations

import math


class TargetApproachPath:
    """Own the projected footpoint while the existing route memory owns A*.

    A refinement of the 3-D memory beyond ``refine_distance_m`` used to drop
    the standoff goal and re-select one from a bearing-relative ring about
    the new position -- and the ring is a different set of points, each
    needing a navmesh projection and a path, so a goal that still had a
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

    **The ring is planned through the unknown before the spot is given up**
    (since 2026-10-14, ``TargetClosingSettings.through_unknown``). The
    planner blocks unknown cells and a level camera leaves the floor under
    its view unknown for 1.4 m ahead, so the floor between a corridor and a
    toilet seen through its doorway routinely has no known route; the lock
    was then released as boxed in with the target in plain view
    (Collierville/000023, action 375). When no ring point has a route on the
    known floor the same ring is tried with the policy's
    ``optimistic_planner`` -- the strict planner with unknown passable at
    ``through_unknown_cost`` times open floor -- and the committed route is
    re-checked against the map as it fills in like any other
    (``through_unknown`` counts them).

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
    """

    #: The bearings tried about the memory, from the agent's side round to behind it.
    OFFSETS_DEG = (0, 30, -30, 60, -60, 90, -90, 180)
    #: The front arc alone for the widened rings: beside and behind a big object is where its floor is not.
    FRONT_OFFSETS_DEG = (0, 30, -30, 60, -60)

    def __init__(self, policy):
        self.policy = policy
        self.goal_xyz = self.reference_xyz = None
        self.refinements = 0
        self.kept = 0                      # refinements through which a still-valid standoff was kept
        self.arrived = False               # the route to the current standoff is walked (the executor's rule)
        self.arrivals = 0                  # standoffs reached over this lock
        self.dropped = 0                   # arrived standoffs given up for a closer one
        self.through_unknown = 0           # standoffs planned through unknown cells (no route on the known floor)
        self.widened = False               # the current goal lies beyond the standoff ring: no closer point had a route
        self.widenings = 0                 # standoffs found on a widened ring over this lock
        self.stale_kept = 0                # stale goals kept because no ring point had a route

    def drop(self):
        """Give up the current standoff so the next :meth:`command` selects one anew, closer to the memory."""
        if self.goal_xyz is not None:
            self.dropped += 1
        self.goal_xyz = self.reference_xyz = None
        self.arrived = self.widened = False
        self.policy.route_memory.clear("standoff_dropped")
        self.policy._route = self.policy._goal = None

    def command(self, target, obs, world):
        p = self.policy
        s = target.settings
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
                self.arrived = self.widened = False
                p.route_memory.clear("target_refined")
                p._route = p._goal = None
        if self.goal_xyz is not None:
            # Do not generate a different bearing-relative endpoint every frame.
            # _navigate still checks collisions, blocked motion and path progress.
            command = p._navigate(obs, world, self.goal_xyz[:2], "target_closing")
            if command is not None:
                reached = p.route_memory.arrived(obs)
                if reached and not self.arrived:
                    self.arrivals += 1
                self.arrived = reached
                return command
            self.goal_xyz = None
            self.arrived = self.widened = False
        tolerance = p.converter_params.goal_tolerance_m
        limit = s.terminal_distance_m - tolerance
        radius = max(p.settings.body_radius_m, limit - world.resolution)
        bearing = math.atan2(obs.pose.y - target.xyz[1], obs.pose.x - target.xyz[0])
        planners = [None]
        squeeze = getattr(p, "squeeze_planner", None)
        if squeeze is not None and squeeze.params.inflate_floor_m < p.planner_params.inflate_floor_m:
            # The known floor at the body radius before the unknown at any price (since 2026-10-25): the
            # routes above relax no further than ``RPTSettings.min_clearance_m``, and the floor beside a sofa
            # corner or a cabinet is often drawn narrower than that -- 00890/000074 released its locked
            # television, 3.5 m away in plain view, as "no safe path from here".
            planners.append(squeeze)
        if s.through_unknown and getattr(p, "optimistic_planner", None) is not None:
            # No route on the known floor: the ring again with unknown cells passable at a price.
            # The target was seen along this line of sight, and a level camera leaves the floor
            # under its view unknown for 1.4 m ahead; the committed route is re-checked against
            # the map as it fills in like any other (since 2026-10-14).
            planners.append(p.optimistic_planner)
        # The standoff ring first, with every bearing and both planners; then, when no point of it has a
        # route, wider rings in the front arc out to the terminal radius (since 2026-10-15): a goal there is
        # reached within the tolerance up to a forward step outside the terminal range, which the inspection
        # closes with a step (``TargetClosing._inspect``) and STOPs on when it cannot (``closest_approach``).
        rings = [(radius, limit, self.OFFSETS_DEG)]
        widest = s.terminal_distance_m - world.resolution
        if widest > radius + world.resolution:
            reach = s.terminal_distance_m + world.resolution
            rings.append((0.5 * (radius + widest), reach, self.FRONT_OFFSETS_DEG))
            rings.append((widest, reach, self.FRONT_OFFSETS_DEG))
        for ring_radius, reach, offsets in rings:
            for planner in planners:
                for offset in offsets:
                    angle = bearing + math.radians(offset)
                    goal = (target.xyz[0] + ring_radius * math.cos(angle), target.xyz[1] + ring_radius * math.sin(angle), obs.pose.z)
                    if p.target_projector is not None:
                        goal = p.target_projector(goal)
                        if goal is None:
                            continue
                    if abs(goal[2] - obs.pose.z) > 0.2 or math.dist(goal[:2], target.xyz[:2]) > reach:
                        continue
                    command = p._navigate(obs, world, goal[:2], "target_closing", planner=planner)
                    if command is not None and math.dist(command.waypoints[-1], target.xyz[:2]) <= reach:
                        self._adopt(target, obs, goal, widened=ring_radius > radius,
                                    through_unknown=planner is not None and planner is not squeeze)
                        return command
        if previous is not None and math.dist(previous[:2], target.xyz[:2]) <= s.terminal_distance_m + tolerance:
            # No ring point has a route, and the goal the refinement made stale -- by centimetres -- had one:
            # arriving at it puts the agent within a step of the terminal range, where the inspection reads
            # the fresh range. Kept over no goal at all (since 2026-10-15; Collierville/000009, action 15).
            command = p._navigate(obs, world, previous[:2], "target_closing")
            if command is not None:
                self.stale_kept += 1
                target.stale_standoffs_kept += 1
                self._adopt(target, obs, previous, widened=True, through_unknown=False)
                return command
        return None

    def _adopt(self, target, obs, goal, widened, through_unknown):
        """Take ``goal`` as the standoff and record how it was found."""
        p = self.policy
        self.goal_xyz, self.reference_xyz = tuple(goal), tuple(target.xyz)
        self.widened = bool(widened)
        self.arrived = p.route_memory.arrived(obs)      # a ring point the agent already stands on
        if self.arrived:
            self.arrivals += 1
        if widened:
            self.widenings += 1
            target.widened_standoffs += 1
        if through_unknown:
            self.through_unknown += 1
            target.through_unknown_routes += 1
