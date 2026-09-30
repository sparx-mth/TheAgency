"""One room-coverage gate for planned, fallback and accidental stair departures."""
from __future__ import annotations

import math

from sparx_agency.core.planning.objnav.types.actions import DiscreteAction


class FloorDepartureGuard:
    """Read the peek ledger, never substitute a label, timeout or failed attempt.

    Only observed rooms and seen stair portals are used. A floor with no room
    geometry yet is not complete. Unreachable rooms remain pending, visibly;
    this guard cannot certify rooms that the sensors have not discovered.
    """

    def __init__(self, building):
        self.building = building
        self._last_pending = None
        self.vetoes = 0

    def ready(self, obs=None):
        p = self.building.policy
        if p.peek.floor_ready():
            self._last_pending = None
            return True
        pending = p.peek.pending_rooms()
        signature = (p.mapping.floor_id, tuple(pending))
        if obs is not None and signature != self._last_pending:
            self._last_pending = signature
            self.building.events.append(dict(
                action=obs.step, event="floor_change_held", floor_id=p.mapping.floor_id,
                pending_rooms=pending, known_rooms=len(p.graph.registry.rooms),
                reason="every observed room needs an interior peek and a measured 180-degree scan"))
        return False

    def filter_action(self, obs, action):
        """Do not let a floor-wide frontier route bypass the stair-choice gate."""
        b, p = self.building, self.building.policy
        if action != DiscreteAction.MOVE_FORWARD or b.traversing or self.ready():
            return action
        step = p.episode.action_spec.forward_step_m
        here = (obs.pose.x, obs.pose.y)
        ahead = (here[0] + step * math.cos(obs.pose.yaw), here[1] + step * math.sin(obs.pose.yaw))
        height = p.mapping.atlas.elevation_m
        for portal in b.portals:
            if portal["floor_id"] != b.floor_id:
                continue
            path = portal.get("path", ())
            for a, z in zip(path, path[1:]):
                # Only the departing flight at this height, never an overhead
                # switchback or a flat approach through the same room.
                levels = (abs(a[2] - height), abs(z[2] - height))
                if max(levels) <= b.params.stable_height_m or min(levels) > b.params.departure_m:
                    continue
                dx, dy = z[0] - a[0], z[1] - a[1]
                length2 = dx * dx + dy * dy
                if length2 < 1e-9:
                    continue
                before = ((here[0] - a[0]) * dx + (here[1] - a[1]) * dy) / length2
                after = ((ahead[0] - a[0]) * dx + (ahead[1] - a[1]) * dy) / length2
                t = max(0.0, min(1.0, after))
                separation = math.dist(ahead, (a[0] + t * dx, a[1] + t * dy))
                if after > max(0.0, before) and before < 1.0 and separation <= p.settings.body_radius_m + 0.1:
                    self.ready(obs)
                    self.vetoes += 1
                    if p._goal is not None and not p._action_owner.startswith("doorway_peek"):
                        p._visited_frontiers.append(tuple(p._goal))
                    p.peek.cancel(obs, "floor_coverage_guard")
                    p.route_memory.clear("floor_coverage_guard")
                    p._route = p._goal = None
                    return DiscreteAction.TURN_LEFT
        return action

    def recover_unplanned(self, obs):
        """If already on a tread before coverage, return safely instead of climbing.

        Reuse the traversal's existing retreat machinery; never drop its
        safety ownership and resume planar navigation while on the flight.
        """
        b, p = self.building, self.building.policy
        if self.ready(obs):
            return
        t = b.transition
        if b.ground_truth is not None:
            flat = [point for point in reversed(b.source_trail)
                    if abs(point[2] - t.source_height) <= b.params.stable_height_m]
            source = next((point for point in flat if math.dist(point[:2], t.polyline[0][:2])
                           > p.converter_params.goal_tolerance_m + p.settings.map_resolution_m),
                          flat[0] if flat else t.polyline[0])
            t.origin = tuple(source)
            height = t.direction * (obs.pose.z - t.source_height)
            walked = [i for i, point in enumerate(t.polyline)
                      if t.direction * (point[2] - t.source_height) <= height + 0.05]
            t.cursor = max(walked, default=0)
            t._entered = True
            t._begin_retreat(obs, "rooms_not_peeked")
        else:
            t.retreat_path = list(reversed(t.trace))
            t.retreat_step, t.phase = obs.step, "RETREAT"
            b.events.append(dict(action=obs.step, event="retreat_started", portal_id=t.portal["id"],
                                 reason="rooms_not_peeked"))
        p.peek.cancel(obs, "stairs_recovery", restore=False)
