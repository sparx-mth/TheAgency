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
    """

    def __init__(self, policy):
        self.policy = policy
        self.goal_xyz = self.reference_xyz = None
        self.refinements = 0
        self.kept = 0                      # refinements through which a still-valid standoff was kept
        self.arrived = False               # the route to the current standoff is walked (the executor's rule)
        self.arrivals = 0                  # standoffs reached over this lock
        self.dropped = 0                   # arrived standoffs given up for a closer one

    def drop(self):
        """Give up the current standoff so the next :meth:`command` selects one anew, closer to the memory."""
        if self.goal_xyz is not None:
            self.dropped += 1
        self.goal_xyz = self.reference_xyz = None
        self.arrived = False
        self.policy.route_memory.clear("standoff_dropped")
        self.policy._route = self.policy._goal = None

    def command(self, target, obs, world):
        p = self.policy
        s = target.settings
        stale = self.goal_xyz is not None and math.dist(self.goal_xyz[:2], target.xyz[:2]) > s.terminal_distance_m
        moved = self.reference_xyz is not None and math.dist(self.reference_xyz[:2], target.xyz[:2]) > s.refine_distance_m
        if moved or stale:
            self.refinements += 1
            self.reference_xyz = tuple(target.xyz)
            if self.goal_xyz is not None and not stale:
                self.kept += 1                  # the standoff still ends inside the terminal radius: no re-selection
            else:
                self.goal_xyz = None
                self.arrived = False
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
            self.arrived = False
        limit = s.terminal_distance_m - p.converter_params.goal_tolerance_m
        radius = max(p.settings.body_radius_m, limit - world.resolution)
        bearing = math.atan2(obs.pose.y - target.xyz[1], obs.pose.x - target.xyz[0])
        for offset in (0, 30, -30, 60, -60, 90, -90, 180):
            angle = bearing + math.radians(offset)
            goal = (target.xyz[0] + radius * math.cos(angle), target.xyz[1] + radius * math.sin(angle), obs.pose.z)
            if p.target_projector is not None:
                goal = p.target_projector(goal)
                if goal is None:
                    continue
            if abs(goal[2] - obs.pose.z) > 0.2 or math.dist(goal[:2], target.xyz[:2]) > limit:
                continue
            command = p._navigate(obs, world, goal[:2], "target_closing")
            if command is not None and math.dist(command.waypoints[-1], target.xyz[:2]) <= limit:
                self.goal_xyz, self.reference_xyz = tuple(goal), tuple(target.xyz)
                self.arrived = p.route_memory.arrived(obs)      # a ring point the agent already stands on
                if self.arrived:
                    self.arrivals += 1
                return command
        return None
