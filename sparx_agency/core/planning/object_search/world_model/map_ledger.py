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

The guard's verdict is itself the sum of layers, and the ledger names the
layer that moved under every changed cell (``layers`` in the summary: the
observed drops, the ray crossings, the stair capsules, the walkable steps,
the trail exemption, the bump bars -- whatever the caller hands in), so a
cell rewritten out of view says WHICH code path rewrote it, not only that
one did.

Each change is also classed by whether the cell lay inside the camera's
field of view this action (:func:`field_of_view`: the footprint of the
camera frustum to its range -- pitch included -- plus the body's own
radius). A planning grid whose cells the camera cannot see keep changing is
a planning grid rewritten from something other than observation --
00878-XB4GS9ShBRE/000000 (2026-11-03): 369 cells at action 84, 735 at 150,
641 at 159, none of them in view, every one on an action the storey plane
was re-measured, because the guard re-thresholded every measured cell of
the grid against the new plane. The ledger is what makes that visible in
the record: the per-action summary rides in the policy's diagnostics
(``method.map_changes`` in the recording's ``steps.jsonl``), and actions
with more than ``jump_cells`` changes out of view are kept as events with
the plane before and after.

Pure numpy over the int8 grids and the bool layers; a few operations per action.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical

#: Changes out of view on one action from this count up are an event (and, in the tests, a failure).
JUMP_CELLS = 50
#: Events kept; the per-action counters are complete.
MAX_EVENTS = 60
#: Slack around the frustum's footprint, cells: a reading at the frame's edge lands in the cell beside it.
FOV_MARGIN_CELLS = 1


def field_of_view(world, pose, camera, body_radius_m=0.25) -> np.ndarray:
    """``(H, W)`` bool: the cells a depth reading of this frame could have landed in, and the body's own cells.

    The camera frustum is the pyramid from the optical centre through the
    four corners of the image, cut at ``camera.max_depth_m`` along the
    optical axis (a reading at the bound is dropped by ``backproject_depth``,
    so nothing lands beyond it). Its footprint on the map is the convex hull
    of the optical centre and the four far corners projected to the ground
    -- a convex body's shadow is the hull of its vertices' shadows -- with
    the pose's pitch in the transform: a camera pitched 30 degrees down
    reads the floor at its feet across some 55 degrees of azimuth at the
    frame's corners, not the 39.5 degrees of a level frame, so a horizontal
    sector at the nominal half-angle calls those readings "out of view". The
    hull is widened by :data:`FOV_MARGIN_CELLS`, and the disc of
    ``body_radius_m`` about the agent is included for the footprint the
    observed map clears under it.
    """
    k = camera.intrinsics
    depth = float(camera.max_depth_m)
    corners_px = np.array([[-0.5, -0.5], [k.width - 0.5, -0.5], [k.width - 0.5, k.height - 0.5], [-0.5, k.height - 0.5]])
    optical = np.column_stack([(corners_px[:, 0] - k.cx) / k.fx * depth, (corners_px[:, 1] - k.cy) / k.fy * depth,
                               np.full(4, depth), np.ones(4)])
    transform = world_T_camera_optical(pose, camera)
    far = (transform @ optical.T).T[:, :2]
    points = np.vstack([[pose.x, pose.y], far])
    cells = np.column_stack([(points[:, 0] - world.origin_x) / world.resolution,
                             (points[:, 1] - world.origin_y) / world.resolution])
    hull = cv2.convexHull(np.round(cells).astype(np.int32).reshape(-1, 1, 2))
    canvas = np.zeros((world.height, world.width), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, hull, 1)
    if FOV_MARGIN_CELLS > 0:
        size = 2 * FOV_MARGIN_CELLS + 1
        canvas = cv2.dilate(canvas, np.ones((size, size), np.uint8))
    fov = canvas.astype(bool)
    ys, xs = np.mgrid[0:world.height, 0:world.width]
    dx = world.origin_x + (xs + 0.5) * world.resolution - float(pose.x)
    dy = world.origin_y + (ys + 0.5) * world.resolution - float(pose.y)
    return fov | (np.hypot(dx, dy) <= float(body_radius_m))


class MapChangeLedger:
    """Diff the planning world of each action against the floor's previous one and attribute the changes."""

    def __init__(self, jump_cells: int = JUMP_CELLS):
        self.jump_cells = int(jump_cells)
        self._observed: Dict[int, np.ndarray] = {}     # floor id -> last observed export
        self._planning: Dict[int, np.ndarray] = {}     # floor id -> last planning world
        self._masked: Dict[int, np.ndarray] = {}       # floor id -> last guard mask
        self._freed: Dict[int, np.ndarray] = {}        # floor id -> last guard step overlay
        self._layers: Dict[int, Dict[str, np.ndarray]] = {}   # floor id -> layer name -> last layer
        self.last: Optional[dict] = None
        self.events: List[dict] = []
        self.stats = {"actions": 0, "changed": 0, "outside_fov": 0, "guard_outside_fov": 0, "jumps": 0}
        self.outside_fov_by_layer: Dict[str, int] = {}
        self.max_outside_fov = 0

    def record(self, obs, floor_id, observed, planning, mask=None, freed=None, layers=None, plane_m=None,
               plane_before_m=None):
        """Attribute this action's changes; returns the summary also kept as :attr:`last`.

        Args:
            obs: The observation (pose and camera for the field of view, ``step``).
            floor_id: The floor the worlds belong to.
            observed: The observed map's export this action (``OccupancyGrid2D``).
            planning: The world the policy plans on this action (the guard's output).
            mask: The guard's occupied mask this action (bool grid or None).
            freed: The guard's step overlay written free this action (bool grid or None).
            layers: ``{name: bool grid or None}`` -- the layers the mask and the overlay are made of
                (``SpawnFloorGuard.layers``) and anything else the caller wants a changed cell attributed
                to; a changed cell is credited to every layer that changed under it.
            plane_m: The storey plane in force after this action's guard pass.
            plane_before_m: The plane in force before it (a re-measure this action shows as a difference).
        """
        floor_id = int(floor_id)
        shape = planning.grid.shape
        observed_grid = np.asarray(observed.grid, dtype=np.int8)
        planning_grid = np.asarray(planning.grid, dtype=np.int8)
        mask = np.zeros(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        freed = np.zeros(shape, dtype=bool) if freed is None else np.asarray(freed, dtype=bool)
        layers = {name: (np.zeros(shape, dtype=bool) if layer is None else np.asarray(layer, dtype=bool))
                  for name, layer in (layers or {}).items()}
        previous = self._planning.get(floor_id)
        summary = {"action": int(obs.step), "floor_id": floor_id, "changed": 0, "in_fov": 0, "outside_fov": 0,
                   "observation": 0, "guard": 0, "both": 0, "unattributed": 0,
                   "guard_detail": {"mask_on": 0, "mask_off": 0, "free_on": 0, "free_off": 0},
                   "outside_fov_detail": {"observation": 0, "guard": 0, "both": 0},
                   "layers": {}, "outside_fov_bbox_m": None,
                   "plane_m": None if plane_m is None else round(float(plane_m), 3),
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
                before_layers = self._layers.get(floor_id, {})
                for name, layer in layers.items():
                    was = before_layers.get(name)
                    moved = (layer != was) & changed if was is not None and was.shape == shape else changed & layer
                    if moved.any():
                        summary["layers"][name] = {"changed": int(moved.sum()), "outside_fov": int((moved & outside).sum()),
                                                   "on": int((moved & layer).sum()), "off": int((moved & ~layer).sum())}
                        self.outside_fov_by_layer[name] = (self.outside_fov_by_layer.get(name, 0)
                                                           + summary["layers"][name]["outside_fov"])
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
        self._layers[floor_id] = {name: layer.copy() for name, layer in layers.items()}
        self.last = summary
        return summary

    def diagnostics(self) -> dict:
        return {"jump_cells": self.jump_cells, "stats": dict(self.stats), "max_outside_fov": int(self.max_outside_fov),
                "outside_fov_by_layer": dict(self.outside_fov_by_layer), "events": list(self.events)}
