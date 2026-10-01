"""Observed, reachable room entrances for bounded opportunistic inspection."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from sparx_agency.core.planning.objnav.action_converter.ladder import reach_floor
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import stair_peek_mask


@dataclass(frozen=True)
class PeekSettings:
    """A peek is local, finite and optional; distances are metres, budgets actions."""

    enabled: bool = True
    trigger_distance_m: float = 3.0
    door_radius_m: float = 1.25
    inset_m: float = 1.0
    approach_actions: int = 36
    return_actions: int = 60
    scan_actions: int = 24
    retry_actions: int = 50
    max_attempts: int = 1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("doorway_peek.enabled must be boolean")
        for name in ("trigger_distance_m", "door_radius_m", "inset_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        for name in ("approach_actions", "return_actions", "scan_actions", "retry_actions", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive action count" % name)
        if self.max_attempts != 1:
            raise ValueError("Each room may be peeked at most once per episode")


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


def doorway_candidates(policy, observation, world, settings, *, nearby=True, room_ids=None):
    """Reachable interior viewpoints, including the room the robot starts in.

    Door proximity and the local trigger constrain the THRESHOLD, not the
    viewpoint. Move a further ``inset_m`` inward, allowing for early stopping
    by the converter. Small rooms use the deepest safe point actually seen;
    neither unknown nor unreachable cells are invented to meet the inset.
    ``nearby=False`` lets the exhausted-floor fallback finish outstanding peeks.
    """
    graph = policy.graph
    inventory = graph.frontier_inventory
    if inventory is None:
        return []
    here = graph.room_at(world, (observation.pose.x, observation.pose.y))
    candidates = []
    excluded = stair_peek_mask(policy, world)
    for pid, room in graph.registry.rooms.items():
        if room_ids is not None and pid not in room_ids:
            continue
        ys, xs = np.nonzero(threshold_cells(policy, world, room) & ~excluded & np.isfinite(inventory.distance_m))
        if not len(xs):
            continue
        distances = inventory.distance_m[ys, xs]
        door_points = [door["xy"] for door in graph.doors if pid in door.get("rooms", ())]
        allowed = distances <= settings.trigger_distance_m if nearby and pid != here else np.ones(len(xs), bool)
        if door_points and pid != here:
            door_distance = np.full(len(xs), np.inf)
            for point in door_points:
                gx, gy = world.world_to_grid(*point)
                door_distance = np.minimum(door_distance, np.hypot(xs - gx, ys - gy) * world.resolution)
            allowed &= door_distance <= settings.door_radius_m
        if not allowed.any():
            continue
        index = int(np.argmin(np.where(allowed, distances, np.inf)))
        entry = tuple(float(v) for v in world.grid_to_world(int(xs[index]), int(ys[index])))
        local = np.hypot(xs - xs[index], ys - ys[index]) * world.resolution <= settings.door_radius_m
        inward = (float(xs[local].mean() - xs[index]), float(ys[local].mean() - ys[index]))
        if math.hypot(*inward) < 1e-6:
            inward = (room.centroid[0] - entry[0], room.centroid[1] - entry[1])
        heading = math.atan2(inward[1], inward[0])
        margin = max(policy.converter_params.goal_tolerance_m, reach_floor(policy.episode.action_spec)) + world.resolution
        if pid == here:
            # Already inside is not already scanned. Avoid a needless trip to a
            # door: stand well inside the observed region and scan from there.
            depth = distance_transform_edt(room.mask)[ys, xs] * world.resolution
            enough = depth >= min(settings.inset_m + margin, float(depth.max())) - 1e-6
            index = int(np.argmin(np.where(enough, distances, np.inf)))
        else:
            depth = ((xs - xs[index]) * math.cos(heading) + (ys - ys[index]) * math.sin(heading)) * world.resolution
            # Require a straight, room-contained connection to the threshold:
            # a distant lobe across a wall is not a deeper doorway viewpoint.
            # Try useful points first and stop at the first valid ray. Do not
            # ray-cast every room cell, or allocate a whole grid per ray.
            shortfall = np.maximum(0.0, settings.inset_m + margin - depth)
            ranked = np.lexsort((distances, shortfall))
            blocked = ~room.mask | excluded
            index = next((int(i) for i in ranked if line_of_sight_clear(
                blocked, int(xs[index]), int(ys[index]), int(xs[i]), int(ys[i]))), None)
            if index is None:
                continue
        xy = tuple(float(v) for v in world.grid_to_world(int(xs[index]), int(ys[index])))
        candidates.append((float(distances[index]), pid, xy, heading, room.mask))
    return sorted(candidates, key=lambda row: (row[0], row[1]))
