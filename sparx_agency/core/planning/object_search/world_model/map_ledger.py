"""Per-action ledger of what changed on the planning grid, where, by which layer, and whether the camera could see it.

The planning world of an action is the observed map's export (log-odds
thresholded, body footprint, bump bars) with the spawn-floor guard's layers
on top (drops and stair capsules written occupied, walkable steps written
free). A cell may change for three kinds of reason:

* **observation** -- the observed map's export changed there: a depth frame
  was integrated, the body footprint moved, a bump bar was written or
  expired. Legitimate inside the camera's footprint and the body's reach;
* **guard** -- the export did not change but the guard's verdict did: a
  cell masked (``mask_on``) or released (``mask_off``), a step freed
  (``free_on``) or no longer freed (``free_off``);
* **both** -- the export and the guard changed the same cell this action.

Each change is also classed by whether the cell lay inside the camera's
field of view this action (:func:`field_of_view`: the horizontal frustum to
the camera's range, plus the body's own radius). A planning grid whose
cells the camera cannot see keep changing is a planning grid rewritten
from something other than observation -- 00878-XB4GS9ShBRE/000000
(2026-11-03): 369 cells at action 84, 733 at 150, 612 at 159, none of them
in view, every one on an action the storey plane was re-measured, because
the guard re-thresholded every measured cell of the grid against the new
plane. The ledger is what makes that visible in the record: the per-action
summary rides in the policy's diagnostics (``method.map_changes`` in the
recording's ``steps.jsonl``), and actions with more than ``jump_cells``
changes out of view are kept as events with the plane before and after.

Pure numpy over the two int8 grids; a few boolean operations per action.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

#: Changes out of view on one action from this count up are an event (and, in the tests, a failure).
JUMP_CELLS = 50
#: Events kept; the per-action counters are complete.
MAX_EVENTS = 60
#: Half-angle slack around the horizontal field of view, degrees: a cell at the frame's edge is in view.
FOV_MARGIN_DEG = 6.0


def field_of_view(world, pose, camera, body_radius_m=0.25) -> np.ndarray:
    """``(H, W)`` bool: the cells the camera could have measured this action, and the body's own cells.

    The horizontal frustum -- heading ``pose.yaw``, half-angle from the
    intrinsics (``atan(cx / fx)``) plus :data:`FOV_MARGIN_DEG`, out to the
    camera's maximum depth and a cell -- regardless of pitch: a pitched
    camera sees less of the floor, never more of the grid outside this
    sector. The disc of ``body_radius_m`` about the agent is included for the
    footprint the observed map clears under it.
    """
    k = camera.intrinsics
    half = math.atan2(float(k.cx), float(k.fx)) + math.radians(FOV_MARGIN_DEG)
    reach = float(camera.max_depth_m) + float(world.resolution)
    ys, xs = np.mgrid[0:world.height, 0:world.width]
    dx = world.origin_x + (xs + 0.5) * world.resolution - float(pose.x)
    dy = world.origin_y + (ys + 0.5) * world.resolution - float(pose.y)
    distance = np.hypot(dx, dy)
    bearing = np.abs((np.arctan2(dy, dx) - float(pose.yaw) + math.pi) % (2 * math.pi) - math.pi)
    return ((distance <= reach) & (bearing <= half)) | (distance <= float(body_radius_m))


class MapChangeLedger:
    """Diff the planning world of each action against the floor's previous one and attribute the changes."""

    def __init__(self, jump_cells: int = JUMP_CELLS):
        self.jump_cells = int(jump_cells)
        self._observed: Dict[int, np.ndarray] = {}     # floor id -> last observed export
        self._planning: Dict[int, np.ndarray] = {}     # floor id -> last planning world
        self._masked: Dict[int, np.ndarray] = {}       # floor id -> last guard mask
        self._freed: Dict[int, np.ndarray] = {}        # floor id -> last guard step overlay
        self.last: Optional[dict] = None
        self.events: List[dict] = []
        self.stats = {"actions": 0, "changed": 0, "outside_fov": 0, "guard_outside_fov": 0, "jumps": 0}
        self.max_outside_fov = 0

    def record(self, obs, floor_id, observed, planning, mask=None, freed=None, plane_m=None, plane_before_m=None):
        """Attribute this action's changes; returns the summary also kept as :attr:`last`.

        Args:
            obs: The observation (pose and camera for the field of view, ``step``).
            floor_id: The floor the worlds belong to.
            observed: The observed map's export this action (``OccupancyGrid2D``).
            planning: The world the policy plans on this action (the guard's output).
            mask: The guard's occupied mask this action (bool grid or None).
            freed: The guard's step overlay written free this action (bool grid or None).
            plane_m: The storey plane in force after this action's guard pass.
            plane_before_m: The plane in force before it (a re-measure this action shows as a difference).
        """
        floor_id = int(floor_id)
        shape = planning.grid.shape
        observed_grid = np.asarray(observed.grid, dtype=np.int8)
        planning_grid = np.asarray(planning.grid, dtype=np.int8)
        mask = np.zeros(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        freed = np.zeros(shape, dtype=bool) if freed is None else np.asarray(freed, dtype=bool)
        previous = self._planning.get(floor_id)
        summary = {"action": int(obs.step), "floor_id": floor_id, "changed": 0, "in_fov": 0, "outside_fov": 0,
                   "observation": 0, "guard": 0, "both": 0,
                   "guard_detail": {"mask_on": 0, "mask_off": 0, "free_on": 0, "free_off": 0},
                   "outside_fov_detail": {"observation": 0, "guard": 0, "both": 0},
                   "outside_fov_bbox_m": None, "plane_m": None if plane_m is None else round(float(plane_m), 3),
                   "plane_moved": bool(plane_before_m is not None and plane_m is not None
                                       and abs(float(plane_m) - float(plane_before_m)) > 1e-6)}
        if previous is not None and previous.shape == shape:
            changed = planning_grid != previous
            if changed.any():
                observed_changed = observed_grid != self._observed[floor_id]
                guard_changed = ((mask != self._masked[floor_id]) | (freed != self._freed[floor_id])) & changed
                fov = field_of_view(planning, obs.pose, obs.camera)
                outside = changed & ~fov
                kinds = {"observation": observed_changed & ~guard_changed, "guard": guard_changed & ~observed_changed,
                         "both": observed_changed & guard_changed}
                summary["changed"] = int(changed.sum())
                summary["in_fov"] = int((changed & fov).sum())
                summary["outside_fov"] = int(outside.sum())
                for name, cells in kinds.items():
                    summary[name] = int((cells & changed).sum())
                    summary["outside_fov_detail"][name] = int((cells & outside).sum())
                # Cells the guard changed without the export changing are ones the observation did not touch.
                unexplained = changed & ~observed_changed & ~guard_changed
                summary["unattributed"] = int(unexplained.sum())
                detail = summary["guard_detail"]
                was_masked, was_freed = self._masked[floor_id], self._freed[floor_id]
                detail["mask_on"] = int((mask & ~was_masked & changed).sum())
                detail["mask_off"] = int((~mask & was_masked & changed).sum())
                detail["free_on"] = int((freed & ~was_freed & changed).sum())
                detail["free_off"] = int((~freed & was_freed & changed).sum())
                if outside.any():
                    ys, xs = np.nonzero(outside)
                    summary["outside_fov_bbox_m"] = [round(planning.origin_x + float(xs.min()) * planning.resolution, 2),
                                                     round(planning.origin_y + float(ys.min()) * planning.resolution, 2),
                                                     round(planning.origin_x + float(xs.max() + 1) * planning.resolution, 2),
                                                     round(planning.origin_y + float(ys.max() + 1) * planning.resolution, 2)]
                self.stats["changed"] += summary["changed"]
                self.stats["outside_fov"] += summary["outside_fov"]
                self.stats["guard_outside_fov"] += summary["outside_fov_detail"]["guard"] + summary["outside_fov_detail"]["both"]
                self.max_outside_fov = max(self.max_outside_fov, summary["outside_fov"])
                if summary["outside_fov"] >= self.jump_cells:
                    self.stats["jumps"] += 1
                    if len(self.events) < MAX_EVENTS:
                        self.events.append(dict(summary, event="map_changed_out_of_view",
                                                plane_before_m=None if plane_before_m is None else round(float(plane_before_m), 3)))
                    elif len(self.events) == MAX_EVENTS:
                        self.events.append({"event": "events_truncated", "limit": MAX_EVENTS})
        self.stats["actions"] += 1
        self._observed[floor_id] = observed_grid.copy()
        self._planning[floor_id] = planning_grid.copy()
        self._masked[floor_id] = mask.copy()
        self._freed[floor_id] = freed.copy()
        self.last = summary
        return summary

    def diagnostics(self) -> dict:
        return {"jump_cells": self.jump_cells, "stats": dict(self.stats), "max_outside_fov": int(self.max_outside_fov),
                "events": list(self.events)}
