"""Observed room scopes and persistent coverage/visit bookkeeping; no GT inputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from sparx_agency.core.planning.exploration.falcon.ordering import Deadline
from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes


class ObservedRegions:
    """An anchored local envelope survives segmentation renumbering and door jitter.

    A split inherits overlapping visit records; a merge inherits every overlap.
    Geometry is never transferred across floors. Budget tokens are owned by the
    state machine, not this registry, and cannot be refilled by matching a mask.
    """

    def __init__(self, params, entry_margin_m=0.35):
        self.params = params
        self.entry_margin_m = entry_margin_m
        self.anchor = None
        self.mask = None
        self.envelope = None
        self.room_id = None
        self.visits = []

    def start(self, world, pose, rooms, room_id=None, action=0):
        self.anchor = (pose.x, pose.y)
        x, y = world.world_to_grid(*self.anchor)
        if room_id is None:
            room_id = next((pid for pid, room in rooms.items() if room.mask[y, x]), None)
        self.room_id = room_id
        yy, xx = np.ogrid[:world.grid.shape[0], :world.grid.shape[1]]
        self.envelope = ((xx - x) ** 2 + (yy - y) ** 2) * world.resolution ** 2 <= self.params.scope_radius_m ** 2
        self.mask = self.envelope.copy() if room_id not in rooms else rooms[room_id].mask & self.envelope
        self.visits.append({"anchor": self.anchor, "action": action, "end_action": action,
                            "mask": self.mask.copy(), "status": "partial", "gain_m2": 0.0})

    def scope(self, world, rooms):
        # Match the anchored geometry, not the robot's current side of a doorway.
        ax, ay = world.world_to_grid(*self.anchor)
        overlaps = [(int(np.count_nonzero(room.mask & self.mask)), pid) for pid, room in rooms.items()]
        overlap, pid = max(overlaps, default=(0, None))
        # A larger newly exposed neighbour must not steal a still-live anchor
        # room. This also selects the correct child after an observed split.
        anchored = [key for key, room in sorted(rooms.items()) if room.mask[ay, ax]]
        if anchored:
            pid, overlap = anchored[0], 1
        if overlap:
            self.room_id = pid
            self.mask = rooms[pid].mask & self.envelope
            self.visits[-1]["mask"] |= self.mask
            other = np.zeros(world.grid.shape, bool)
            for key, room in rooms.items():
                if key != pid:
                    other |= room.mask
            # Unknown space is guidance only; free destinations must belong to
            # the matched room. A one-sensor-range unknown apron bootstraps it.
            apron = binary_dilation(self.mask, iterations=max(1, int(self.params.cell_size_m / world.resolution)))
            return self.envelope & ~other & apron
        return self.envelope.copy()  # provisional observed local region

    def finish(self, reason, action, gain):
        if self.visits:
            self.visits[-1].update(status=reason, end_action=action, gain_m2=gain)

    def history_for(self, mask):
        """Overlap threshold is relative to the smaller region, retaining splits."""
        result = []
        for visit in self.visits:
            overlap = np.count_nonzero(mask & visit["mask"])
            denominator = min(np.count_nonzero(mask), np.count_nonzero(visit["mask"]))
            if denominator and overlap / denominator >= 0.25:
                result.append(visit)
        return result

    def room_goals(self, world, cost, rooms, pose, action):
        """Reachable representative *inside* each room, not an obstructed centroid."""
        routes = GridRoutes(cost, world.resolution, self.params.max_grid_nodes)
        start = world.world_to_grid(pose.x, pose.y)
        data = routes.distances(start, Deadline(self.params.planning_deadline_s))
        if data is None:
            return {}, {}
        distance = np.full(cost.shape, np.inf)
        distance[routes.safe] = data[0]
        goals, histories, partial = {}, {}, {}
        for pid, room in sorted(rooms.items()):
            history = self.history_for(room.mask)
            histories[pid] = history
            if len(history) >= self.params.max_region_bursts:
                continue
            cooling = history and action - max(v["end_action"] for v in history) < self.params.revisit_cooldown_actions
            eligible = room.mask & np.isfinite(distance)
            # The discrete converter stops within an arrival band. A goal on
            # the first room cell can therefore leave the body outside forever.
            clearance = distance_transform_edt(np.pad(room.mask, 1))[1:-1, 1:-1] * world.resolution
            interior = eligible & (clearance >= self.entry_margin_m)
            if interior.any():
                eligible = interior
            elif eligible.any():
                eligible &= clearance >= float(clearance[eligible].max()) - 1e-9
            ys, xs = np.nonzero(eligible)
            if not len(xs):
                continue
            cx, cy = world.world_to_grid(*room.centroid)
            # Prefer a short reachable entry while avoiding a boundary-only goal.
            ranks = distance[ys, xs] + 0.2 * np.hypot(xs - cx, ys - cy)
            k = int(np.argmin(ranks))
            goal = world.grid_to_world(int(xs[k]), int(ys[k]))
            if not cooling:
                goals[pid] = goal
            elif history[-1]["status"] == "action_budget" and history[-1]["gain_m2"] >= self.params.low_gain_m2:
                partial[pid] = goal
        # A completed budget with useful gain is not a target-free verdict. Only
        # RPT* can authorize these capped revisits after refreshed reasoning.
        return goals or partial, histories

    def diagnostics(self):
        return [{key: (list(value) if key == "anchor" else value) for key, value in v.items() if key != "mask"}
                for v in self.visits]




