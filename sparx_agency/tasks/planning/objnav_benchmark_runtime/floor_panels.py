"""Display-only floor slots. Configured count must never enter navigation state.

What a floor map shows, and how much: the observed occupancy; the trail; the
committed route; confirmed objects as small dots; every staircase the agent
has SEEN, drawn where its treads were seen (the sightings' footprint) with
its entry marked; and the search's nodes -- but named and valued only for
the nodes in the RPT* order, the node in force and the next one. A floor
with twelve rooms once carried twelve overlapping "3:R7 living_ro 0.04"
labels; a node the order does not visit is a dot now, and a label that
would overlap one already placed moves to the next free side or is dropped.
Text is drawn with a dark outline so it reads on white floor and grey unknown
alike.
"""
from __future__ import annotations

import hashlib
import math
import cv2
import numpy as np

UNKNOWN_GRAY = 128
#: Labels are written for at most this many nodes of the visit order (plus the node in force and the next).
LABELLED_ORDER = 6
#: Font scale of the node labels.
LABEL_SCALE = 0.34
#: Colour of a seen staircase: its footprint dots, its entry square and its label (BGR).
STAIRS_COLOR = (60, 140, 255)


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
            slot["stairs"] = self._stairs_on(policy, floor_id)
        if mapping.floor_id in self.bindings and not (getattr(policy, "building", None) and policy.building.traversing):
            slot = self.slots[self.bindings[mapping.floor_id]]
            pose = observation.pose
            point = (pose.x, pose.y)
            if not slot["trail"] or slot["trail"][-1] != point:
                slot["trail"].append(point)
            slot["pose"] = (pose.x, pose.y, pose.yaw)

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
                 "viewport_span_m": self.span_m}
                for i, s in enumerate(self.slots)]

    def save(self, directory):
        # Numeric arrays only; no pickled state. These are display artifacts,
        # never restored into policy knowledge on another episode.
        np.savez_compressed(directory / "floor_maps.npz", **{"slot_%d" % i: s["grid"] for i, s in enumerate(self.slots)})

    def render(self, active_id, planned_path=(), size=(960, 720), search=None):
        """Every slot as a map; the active one also carries the search's rooms and visit order.

        Args:
            active_id: The floor in force.
            planned_path: The committed route, drawn on the active floor.
            size: ``(w, h)`` of the whole image.
            search: The ``search`` block of a method snapshot -- rooms with
                centroids, labels and search values, the RPT* order, the
                next room and the room in force. Drawn on the active floor
                only; None draws maps alone.
        """
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
            active = slot["floor_id"] is not None and slot["floor_id"] == active_id
            for trail, color in ((slot["trail"], (0, 180, 0)),
                                 (planned_path if active else (), (255, 180, 0))):
                if len(trail) > 1:
                    cv2.polylines(canvas, [np.array([pixel(p) for p in trail], np.int32)], False, color, 1)
            for obj in slot["objects"]:
                cv2.circle(canvas, pixel(obj["xy"]), 2, (200, 30, 180), -1)
            self._draw_stairs(canvas, pixel, slot["stairs"])
            if active and search:
                self._draw_search(canvas, pixel, search)
            if slot["pose"] is not None:
                px, py, yaw = slot["pose"]
                a = pixel((px, py))
                b = pixel((px + math.cos(yaw), py + math.sin(yaw)))
                cv2.arrowedLine(canvas, a, b, (0, 60, 255), 2)
            image[y:y+edge, x:x+edge] = canvas
            title = "Slot %d | UNKNOWN" % index if slot["floor_id"] is None else "F%d%s | rooms %d objects %d stairs %d" % (
                slot["floor_id"], " ACTIVE" if active else "", slot["rooms"], len(slot["objects"]), len(slot["stairs"]))
            cv2.putText(image, title, (left + 8, top + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (0, 220, 255) if active else (230, 230, 230), 1, cv2.LINE_AA)
            caption = "UNKNOWN" if slot["floor_id"] is None else "observed %d cells | search %.0fs" % (
                np.count_nonzero(slot["grid"] >= 0), slot["search_time_s"])
            if active and search:
                caption += " | accessible frontiers %d" % len(search.get("accessible_frontiers", ()))
            cv2.putText(image, caption, (left + 8, top + panel_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1)
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
            cv2.putText(canvas, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            placed.append(box)
            return True
        return False

    @classmethod
    def _draw_stairs(cls, canvas, pixel, stairs):
        """Every seen staircase: its footprint where the treads were seen, its entry as a square, a short name."""
        placed = []
        for stair in stairs:
            for point in stair.get("footprint", ()):
                cv2.circle(canvas, pixel(point), 1, STAIRS_COLOR, -1)
            entry = pixel(stair["entry"])
            half = 5
            cv2.rectangle(canvas, (entry[0] - half, entry[1] - half), (entry[0] + half, entry[1] + half), STAIRS_COLOR, -1)
            cv2.rectangle(canvas, (entry[0] - half, entry[1] - half), (entry[0] + half, entry[1] + half), (15, 15, 15), 1)
            name = "S%d %s" % (stair["portal_id"], "up" if stair.get("direction", 0) > 0 else "down")
            cls._label(canvas, name, entry, STAIRS_COLOR, placed)

    @classmethod
    def _draw_search(cls, canvas, pixel, search):
        """The RPT* order as a chain through its nodes; the visited nodes named and valued; every other node a dot.

        Only the nodes of the order (the first :data:`LABELLED_ORDER`), the
        node in force and the next one carry a label; a room the order does
        not visit is a small dot, and a label that finds no free side is
        dropped rather than written over another.
        """
        for goal in search.get("accessible_frontiers", ()):
            cv2.drawMarker(canvas, pixel(goal), (255, 120, 0), cv2.MARKER_DIAMOND, 7, 1)
        nodes = {node["id"]: node for node in list(search.get("rooms", ())) + list(search.get("stairs", ()))
                 if node.get("centroid")}
        order = [nid for nid in search.get("order", ()) if nid in nodes]
        in_force, next_room = search.get("room_in_force"), search.get("next_room")
        if len(order) > 1:
            chain = np.array([pixel(nodes[nid]["centroid"]) for nid in order], np.int32)
            cv2.polylines(canvas, [chain], False, (0, 200, 255), 1, cv2.LINE_AA)
        labelled = set(order[:LABELLED_ORDER]) | {nid for nid in (in_force, next_room) if nid in nodes}
        placed = []
        # The nodes that matter most are labelled first, so they get the free sides.
        ranking = sorted(nodes, key=lambda nid: (nid not in (in_force, next_room), order.index(nid) if nid in order else 1 << 30))
        for nid in ranking:
            node = nodes[nid]
            centre = pixel(node["centroid"])
            stairs = node.get("kind") == "stairs"
            if nid == in_force:
                color = (0, 200, 255)
            elif nid == next_room:
                color = (80, 220, 80)
            elif stairs:
                color = STAIRS_COLOR
            elif (node.get("prob") or 0.0) < 0.01:
                color = (90, 90, 235)
            elif node.get("cooling"):
                color = (120, 120, 120)
            else:
                color = (40, 40, 40)
            if stairs:
                half = 6
                cv2.rectangle(canvas, (centre[0] - half, centre[1] - half), (centre[0] + half, centre[1] + half), color, -1)
            else:
                cv2.circle(canvas, centre, 5 if nid in labelled else 3, color, -1)
            if nid in (in_force, next_room):
                cv2.circle(canvas, centre, 11, color, 2, cv2.LINE_AA)
            if nid not in labelled:
                continue
            prob = node.get("prob")
            name = ("S%d %s" % (node.get("portal_id", nid), "up" if node.get("direction", 0) > 0 else "down")
                    if stairs else "R%d %s" % (nid, (node.get("label") or "?")[:8]))
            text = "%s%s" % (name, "" if prob is None else " %.2f" % prob)
            if nid in order:
                text = "%d:%s" % (order.index(nid) + 1, text)
            cls._label(canvas, text, centre, color, placed)

