"""Display-only floor slots. Configured count must never enter navigation state."""
from __future__ import annotations

import hashlib
import math
import cv2
import numpy as np

UNKNOWN_GRAY = 128


class FloorPanels:
    """One map per storey, not K layers per floor; bindings use discovery IDs.

    The evaluator supplies only the number of reserved display slots, never
    heights or ordering. Full grids are retained; a fixed 32 m observed-origin
    viewport keeps the same scale across visits and all panels. Overflow slots
    are explicit if observations discover more floors than configured.
    """

    def __init__(self, count, settings, pose):
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError("Display floor count must be an integer in [1, 16]")
        self.resolution = settings.map_resolution_m
        n = int(round(settings.map_size_m / self.resolution))
        self.shape = (n, n)
        self.origin = (pose.x - settings.map_size_m / 2, pose.y - settings.map_size_m / 2)
        self.span_m = min(32.0, settings.map_size_m)
        self.center = (pose.x, pose.y)
        self.slots = [self._unknown() for _ in range(count)]
        self.bindings = {}

    def _unknown(self):
        return {"floor_id": None, "grid": np.full(self.shape, -1, np.int8), "objects": [], "trail": [],
                "rooms": 0, "search_time_s": 0.0, "pose": None}

    def capture(self, policy, observation):
        mapping = getattr(policy, "mapping", None)
        if mapping is None:
            return
        for floor_id, world in sorted(mapping.worlds.items()):
            if floor_id not in self.bindings:
                free = next((i for i, s in enumerate(self.slots) if s["floor_id"] is None), None)
                if free is None:
                    free = len(self.slots)
                    self.slots.append(self._unknown())
                self.bindings[floor_id] = free
                self.slots[free]["floor_id"] = floor_id
            slot = self.slots[self.bindings[floor_id]]
            if world.grid.shape != self.shape or not np.allclose((world.origin_x, world.origin_y), self.origin):
                raise ValueError("Floor map identity/scale changed inside an episode")
            np.copyto(slot["grid"], world.grid)
            context = policy.floors.contexts.get(floor_id)
            if floor_id == policy.floors.active:
                context = {key: getattr(policy, key) for key in ("landmarks", "graph", "_floor_time")}
            if context is not None:
                slot["objects"] = [{"id": lm.id, "class": lm.class_name, "xy": list(lm.xy), "count": lm.count}
                                   for lm in context["landmarks"].all_landmarks()]
                slot["rooms"] = len(context["graph"].registry.rooms)
                slot["search_time_s"] = context["_floor_time"]
        if mapping.floor_id in self.bindings and not (getattr(policy, "building", None) and policy.building.traversing):
            slot = self.slots[self.bindings[mapping.floor_id]]
            pose = observation.pose
            point = (pose.x, pose.y)
            if not slot["trail"] or slot["trail"][-1] != point:
                slot["trail"].append(point)
            slot["pose"] = (pose.x, pose.y, pose.yaw)

    def metadata(self):
        return [{"slot": i, "floor_id": s["floor_id"], "unknown": s["floor_id"] is None,
                 "shape": list(s["grid"].shape), "origin": list(self.origin), "resolution_m": self.resolution,
                 "known_cells": int(np.count_nonzero(s["grid"] >= 0)), "rooms": s["rooms"],
                 "objects": list(s["objects"]), "search_time_s": s["search_time_s"],
                 "occupancy_sha256": hashlib.sha256(s["grid"].tobytes()).hexdigest(),
                 "viewport_span_m": self.span_m}
                for i, s in enumerate(self.slots)]

    def save(self, directory):
        # Numeric arrays only; no pickled state. These are display artifacts,
        # never restored into policy knowledge on another episode.
        np.savez_compressed(directory / "floor_maps.npz", **{"slot_%d" % i: s["grid"] for i, s in enumerate(self.slots)})

    def render(self, active_id, planned_path=(), size=(960, 720)):
        w, h = size
        cols = int(math.ceil(math.sqrt(len(self.slots))))
        rows = int(math.ceil(len(self.slots) / cols))
        panel_w, panel_h = w // cols, h // rows
        image = np.full((h, w, 3), 20, np.uint8)
        side = int(round(self.span_m / self.resolution))
        y0, x0 = (self.shape[0] - side) // 2, (self.shape[1] - side) // 2
        for index, slot in enumerate(self.slots):
            left, top = index % cols * panel_w, index // cols * panel_h
            width, height = panel_w - 8, panel_h - 68
            edge = min(width, height)
            x = left + (panel_w - edge) // 2
            y = top + 34
            grid = slot["grid"][y0:y0+side, x0:x0+side]
            colors = np.full((*grid.shape, 3), UNKNOWN_GRAY, np.uint8)
            colors[grid == 0] = 235
            colors[grid == 100] = 35
            canvas = cv2.resize(np.flipud(colors), (edge, edge), interpolation=cv2.INTER_NEAREST)
            def pixel(xy):
                return (int((xy[0] - self.center[0] + self.span_m / 2) / self.span_m * edge),
                        int((self.center[1] + self.span_m / 2 - xy[1]) / self.span_m * edge))
            for trail, color in ((slot["trail"], (0, 180, 0)),
                                 (planned_path if slot["floor_id"] == active_id else (), (255, 180, 0))):
                if len(trail) > 1:
                    cv2.polylines(canvas, [np.array([pixel(p) for p in trail], np.int32)], False, color, 1)
            for obj in slot["objects"]:
                cv2.circle(canvas, pixel(obj["xy"]), 3, (200, 30, 180), -1)
            if slot["pose"] is not None:
                px, py, yaw = slot["pose"]
                a = pixel((px, py))
                b = pixel((px + math.cos(yaw), py + math.sin(yaw)))
                cv2.arrowedLine(canvas, a, b, (0, 60, 255), 2)
            image[y:y+edge, x:x+edge] = canvas
            active = slot["floor_id"] is not None and slot["floor_id"] == active_id
            title = "Slot %d | UNKNOWN" % index if slot["floor_id"] is None else "F%d%s | rooms %d objects %d" % (
                slot["floor_id"], " ACTIVE" if active else "", slot["rooms"], len(slot["objects"]))
            cv2.putText(image, title, (left + 8, top + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (0, 220, 255) if active else (230, 230, 230), 1, cv2.LINE_AA)
            caption = "UNKNOWN" if slot["floor_id"] is None else "observed %d cells | search %.0fs" % (
                np.count_nonzero(slot["grid"] >= 0), slot["search_time_s"])
            cv2.putText(image, caption, (left + 8, top + panel_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)
        return image
