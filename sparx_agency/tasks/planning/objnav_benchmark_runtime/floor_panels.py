"""Minimal observed floor plans; display geometry never enters navigation.

Persistent room outlines, short room IDs, seen stair entries, robot, trail
and committed route only. Objects, probabilities and visit order belong in
the unchanged search column. Every storey shares an observed-footprint
viewport and metric scale, fitted to the available panel without clipping
large houses or shrinking small ones into a fixed 32-metre square.
"""
from __future__ import annotations

import hashlib
import math
import cv2
import numpy as np

UNKNOWN_GRAY = 128
LABEL_SCALE = 0.38
#: Colour of a seen staircase's entry marker (BGR).
STAIRS_COLOR = (60, 140, 255)


class FloorPanels:
    """One map per storey, not K layers per floor; bindings use discovery IDs.

    The evaluator supplies only the number of reserved display slots, never
    heights or ordering. Full grids and room partitions are retained. The
    shared viewport grows with observed geometry, never surveyed house bounds.
    Overflow slots are explicit if more floors are discovered than configured.
    """

    def __init__(self, count, settings, pose):
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError("Display floor count must be an integer in [1, 16]")
        self.resolution = settings.map_resolution_m
        n = int(round(settings.map_size_m / self.resolution))
        self.shape = (n, n)
        self.origin = (pose.x - settings.map_size_m / 2, pose.y - settings.map_size_m / 2)
        self.span_m = min(4.0, settings.map_size_m)
        self.center = (pose.x, pose.y)
        self.slots = [self._unknown() for _ in range(count)]
        self.bindings = {}
        self._bounds = None

    def _unknown(self):
        return {"floor_id": None, "grid": np.full(self.shape, -1, np.int8), "objects": [], "trail": [],
                "room_labels": np.zeros(self.shape, np.int32),
                "stairs": [], "rooms": 0, "search_time_s": 0.0, "pose": None}

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
                labels = context["graph"].labels
                slot["room_labels"].fill(0)
                if labels is not None and labels.shape == self.shape:
                    np.copyto(slot["room_labels"], labels)
            slot["stairs"] = self._stairs_on(policy, floor_id)
        if mapping.floor_id in self.bindings and not (getattr(policy, "building", None) and policy.building.traversing):
            slot = self.slots[self.bindings[mapping.floor_id]]
            pose = observation.pose
            point = (pose.x, pose.y)
            if not slot["trail"] or slot["trail"][-1] != point:
                slot["trail"].append(point)
            slot["pose"] = (pose.x, pose.y, pose.yaw)
        self._fit_view()

    def _fit_view(self):
        """Monotonic, metre-quantised bounds of observations and measured poses."""
        known = np.logical_or.reduce([slot["grid"] >= 0 for slot in self.slots])
        for slot in self.slots:
            for xy in slot["trail"]:
                gx, gy = (int(math.floor((xy[i] - self.origin[i]) / self.resolution)) for i in (0, 1))
                if 0 <= gx < self.shape[1] and 0 <= gy < self.shape[0]:
                    known[gy, gx] = True
        ys, xs = np.nonzero(known)
        quantum = max(1, int(round(1.0 / self.resolution)))
        if len(xs):
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        else:
            x0, y0 = self.shape[1] // 2 - quantum, self.shape[0] // 2 - quantum
            x1, y1 = self.shape[1] // 2 + quantum, self.shape[0] // 2 + quantum
        bounds = (max(0, (x0 // quantum - 1) * quantum), max(0, (y0 // quantum - 1) * quantum),
                  min(self.shape[1], (math.ceil(x1 / quantum) + 1) * quantum),
                  min(self.shape[0], (math.ceil(y1 / quantum) + 1) * quantum))
        if self._bounds is not None:
            bounds = (min(bounds[0], self._bounds[0]), min(bounds[1], self._bounds[1]),
                      max(bounds[2], self._bounds[2]), max(bounds[3], self._bounds[3]))
        self._bounds = bounds
        x0, y0, x1, y1 = bounds
        self.center = (self.origin[0] + (x0 + x1) * self.resolution / 2,
                       self.origin[1] + (y0 + y1) * self.resolution / 2)
        self.span_m = max(x1 - x0, y1 - y0) * self.resolution

    @staticmethod
    def _stairs_on(policy, floor_id):
        """The staircases the building knows on ``floor_id`` -- seen ones, placed where they were seen."""
        building = getattr(policy, "building", None)
        if building is None:
            return []
        out = []
        for portal in getattr(building, "portals", ()):
            if portal.get("floor_id") != floor_id:
                continue
            out.append({"portal_id": portal["id"], "connector_id": portal.get("connector_id"),
                        "direction": int(portal.get("direction", 0)), "entry": list(portal["entry"][:2]),
                        "footprint": [list(p[:2]) for p in portal.get("footprint", ())],
                        "seen_step": portal.get("seen_step"), "cooling_until": portal.get("cooldown_until", 0)})
        return out

    def metadata(self):
        return [{"slot": i, "floor_id": s["floor_id"], "unknown": s["floor_id"] is None,
                 "shape": list(s["grid"].shape), "origin": list(self.origin), "resolution_m": self.resolution,
                 "known_cells": int(np.count_nonzero(s["grid"] >= 0)), "rooms": s["rooms"],
                 "objects": list(s["objects"]), "search_time_s": s["search_time_s"],
                 "stairs": [dict({k: v for k, v in st.items() if k != "footprint"}, footprint_points=len(st["footprint"]))
                            for st in s["stairs"]],
                 "occupancy_sha256": hashlib.sha256(s["grid"].tobytes()).hexdigest(),
                  "room_labels_sha256": hashlib.sha256(s["room_labels"].tobytes()).hexdigest(),
                  "map_style": "minimal-room-outlines", "viewport_cells": self._bounds,
                  "viewport_span_m": self.span_m}
                for i, s in enumerate(self.slots)]

    def save(self, directory):
        # Numeric arrays only; no pickled state. These are display artifacts,
        # never restored into policy knowledge on another episode.
        np.savez_compressed(directory / "floor_maps.npz", **{"slot_%d" % i: s["grid"] for i, s in enumerate(self.slots)})
        np.savez_compressed(directory / "room_partitions.npz",
                            **{"slot_%d" % i: s["room_labels"] for i, s in enumerate(self.slots)})

    def render(self, active_id, planned_path=(), size=(960, 720), search=None):
        """Fit persistent floor plans into ``size=(w, h)`` at one shared scale."""
        w, h = size
        self._fit_view()
        x0, y0, x1, y1 = self._bounds
        gw, gh = x1 - x0, y1 - y0
        # Stacked maps suit wide houses; side-by-side maps suit tall ones.
        cols = max(range(1, len(self.slots) + 1), key=lambda c: min(
            (w // c - 16) / gw, (h // math.ceil(len(self.slots) / c) - 64) / gh))
        rows = int(math.ceil(len(self.slots) / cols))
        panel_w, panel_h = w // cols, h // rows
        image = np.full((h, w, 3), 20, np.uint8)
        scale = min((panel_w - 16) / gw, (panel_h - 64) / gh)
        width, height = max(1, int(round(gw * scale))), max(1, int(round(gh * scale)))
        for index, slot in enumerate(self.slots):
            left, top = index % cols * panel_w, index // cols * panel_h
            x, y = left + (panel_w - width) // 2, top + 32 + (panel_h - 64 - height) // 2
            grid = slot["grid"][y0:y1, x0:x1]
            colors = np.full((*grid.shape, 3), UNKNOWN_GRAY, np.uint8)
            colors[grid == 0] = 242
            colors[grid == 100] = 35
            canvas = cv2.resize(np.flipud(colors), (width, height), interpolation=cv2.INTER_NEAREST)
            def pixel(xy):
                return (int((xy[0] - self.origin[0] - x0 * self.resolution) / (gw * self.resolution) * width),
                        int((self.origin[1] + y1 * self.resolution - xy[1]) / (gh * self.resolution) * height))
            active = slot["floor_id"] is not None and slot["floor_id"] == active_id
            labels = np.where(grid == 0, slot["room_labels"][y0:y1, x0:x1], 0)
            labels = cv2.resize(np.flipud(labels), (width, height), interpolation=cv2.INTER_NEAREST)
            self._draw_rooms(canvas, labels)
            for trail, color in ((slot["trail"], (185, 145, 90)),
                                 (planned_path if active else (), (40, 170, 235))):
                if len(trail) > 1:
                    cv2.polylines(canvas, [np.array([pixel(p) for p in trail], np.int32)], False, color, 1, cv2.LINE_AA)
            self._draw_stairs(canvas, pixel, slot["stairs"])
            if active and search:
                self._draw_search(canvas, pixel, search)
            if slot["pose"] is not None:
                px, py, yaw = slot["pose"]
                a = pixel((px, py))
                b = (int(a[0] + 12 * math.cos(yaw)), int(a[1] - 12 * math.sin(yaw)))
                cv2.arrowedLine(canvas, a, b, (0, 60, 255), 2)
            image[y:y+height, x:x+width] = canvas
            title = "Floor slot %d | UNKNOWN" % index if slot["floor_id"] is None else "Floor %d%s" % (
                slot["floor_id"], " | ACTIVE" if active else "")
            cv2.putText(image, title, (left + 8, top + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (0, 220, 255) if active else (230, 230, 230), 1, cv2.LINE_AA)
            caption = "Unobserved" if slot["floor_id"] is None else "%d rooms | %d stairs" % (slot["rooms"], len(slot["stairs"]))
            cv2.putText(image, caption, (left + 8, top + panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)
            metres = 1.0 if self.span_m <= 12 else 2.0 if self.span_m <= 25 else 5.0
            bar = int(metres * scale / self.resolution)
            bx, by = left + panel_w - bar - 12, top + panel_h - 15
            cv2.line(image, (bx, by), (bx + bar, by), (215, 215, 215), 2)
            cv2.putText(image, "%g m" % metres, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (215, 215, 215), 1)
        return image

    @staticmethod
    def _label(canvas, text, anchor, color, placed, scale=LABEL_SCALE):
        """Write ``text`` beside ``anchor`` on the first side where it overlaps no label placed before; False when none is free.

        Every label is drawn twice -- a thick dark stroke, then the colour --
        so it reads on white floor, grey unknown and the trail alike.
        """
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ax, ay = anchor
        candidates = ((ax + 8, ay - 5), (ax - w - 8, ay - 5), (ax + 8, ay + h + 7), (ax - w - 8, ay + h + 7),
                      (ax - w // 2, ay - 10), (ax - w // 2, ay + h + 12))
        height, width = canvas.shape[:2]
        for x0, y0 in candidates:
            box = (x0 - 2, y0 - h - 2, x0 + w + 2, y0 + 3)
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                continue
            if any(not (box[2] < other[0] or box[0] > other[2] or box[3] < other[1] or box[1] > other[3]) for other in placed):
                continue
            cv2.putText(canvas, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, (242, 242, 242), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            placed.append(box)
            return True
        return False

    @classmethod
    def _draw_stairs(cls, canvas, pixel, stairs):
        """One compact entry symbol per seen staircase, not a cloud of tread dots."""
        for stair in stairs:
            entry = pixel(stair["entry"])
            half = 4
            cv2.rectangle(canvas, (entry[0] - half, entry[1] - half), (entry[0] + half, entry[1] + half), STAIRS_COLOR, -1)
            cv2.rectangle(canvas, (entry[0] - half, entry[1] - half), (entry[0] + half, entry[1] + half), (15, 15, 15), 1)

    @classmethod
    def _draw_rooms(cls, canvas, labels):
        """Thin simplified logical boundaries from the live observed partition."""
        placed = []
        for value in np.unique(labels):
            if value <= 0:
                continue
            mask = (labels == value).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            outlines = [cv2.approxPolyDP(c, 1.5, True) for c in contours if cv2.contourArea(c) >= 12]
            cv2.polylines(canvas, outlines, True, (150, 155, 160), 1, cv2.LINE_AA)
            if int(mask.sum()) >= 180:
                depth = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                row, col = np.unravel_index(int(depth.argmax()), depth.shape)
                cls._label(canvas, "R%d" % (int(value) - 1), (int(col), int(row)), (65, 70, 75), placed)

    @staticmethod
    def _draw_search(canvas, pixel, search):
        """Only current/next-room emphasis; verbose search facts stay off the map."""
        nodes = {n["id"]: n for n in list(search.get("rooms", ())) + list(search.get("stairs", ())) if n.get("centroid")}
        for key, color in (("room_in_force", (40, 170, 235)), ("next_room", (75, 160, 75))):
            node = nodes.get(search.get(key))
            if node is not None:
                cv2.circle(canvas, pixel(node["centroid"]), 8, color, 2, cv2.LINE_AA)

