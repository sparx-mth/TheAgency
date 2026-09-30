"""Observed, reachable room entrances for bounded opportunistic inspection."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from sparx_agency.core.planning.objnav.action_converter.ladder import reach_floor


@dataclass(frozen=True)
class PeekSettings:
    """A peek is local, finite and optional; distances are metres, budgets actions."""

    enabled: bool = True
    trigger_distance_m: float = 3.0
    door_radius_m: float = 1.25
    approach_actions: int = 24
    return_actions: int = 60
    scan_actions: int = 24
    retry_actions: int = 50
    max_attempts: int = 2

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("doorway_peek.enabled must be boolean")
        for name in ("trigger_distance_m", "door_radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        for name in ("approach_actions", "return_actions", "scan_actions", "retry_actions", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive action count" % name)


def same_region(left, right):
    """A substantial two-way overlap, not merely touching or one child of a split."""
    if left.shape != right.shape:
        return False
    overlap = int(np.count_nonzero(left & right))
    return overlap >= 0.6 * max(1, int(left.sum()), int(right.sum()))


def threshold_cells(policy, world, room):
    """An arrival-tolerance inset keeps the actual stopping pose inside the room."""
    margin = max(policy.converter_params.goal_tolerance_m, reach_floor(policy.episode.action_spec)) + world.resolution
    return distance_transform_edt(room.mask) * world.resolution > margin


def doorway_candidates(policy, observation, world, settings):
    """Nearest reachable threshold in each nearby room, never a centroid detour.

    Confirmed doors constrain the entry to their neighbourhood. When a new
    geometric room has no confirmed door yet, its nearest reachable boundary
    is an observed opening, not an invented doorway behind an occupied wall.
    The normal A* must still accept the route before a peek can move.
    """
    graph = policy.graph
    inventory = graph.frontier_inventory
    if inventory is None:
        return []
    here = graph.room_at(world, (observation.pose.x, observation.pose.y))
    candidates = []
    for pid, room in graph.registry.rooms.items():
        if pid == here:
            continue
        ys, xs = np.nonzero(threshold_cells(policy, world, room) & np.isfinite(inventory.distance_m))
        if not len(xs):
            continue
        distances = inventory.distance_m[ys, xs]
        door_points = [door["xy"] for door in graph.doors if pid in door.get("rooms", ())]
        allowed = distances <= settings.trigger_distance_m
        if door_points:
            door_distance = np.full(len(xs), np.inf)
            for point in door_points:
                gx, gy = world.world_to_grid(*point)
                door_distance = np.minimum(door_distance, np.hypot(xs - gx, ys - gy) * world.resolution)
            allowed &= door_distance <= settings.door_radius_m
        if not allowed.any():
            continue
        index = int(np.argmin(np.where(allowed, distances, np.inf)))
        xy = tuple(float(v) for v in world.grid_to_world(int(xs[index]), int(ys[index])))
        local = np.hypot(xs - xs[index], ys - ys[index]) * world.resolution <= settings.door_radius_m
        inward = (float(xs[local].mean() - xs[index]), float(ys[local].mean() - ys[index]))
        if math.hypot(*inward) < 1e-6:
            inward = (xy[0] - observation.pose.x, xy[1] - observation.pose.y)
        heading = math.atan2(inward[1], inward[0])
        candidates.append((float(distances[index]), pid, xy, heading, room.mask))
    return sorted(candidates, key=lambda row: (row[0], row[1]))
