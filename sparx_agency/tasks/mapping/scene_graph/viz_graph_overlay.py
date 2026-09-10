"""The topological room graph, drawn onto the map panel.

This is the half of the old RViz ``/scene_graph/markers`` view that the OpenCV
dashboard was missing: rooms as **nodes**, doors as the **state of the
frontier** (seen or still pending), the gold **room -> door -> room edges**
that make the tinted blobs read as a graph, and the green **object -> room
links** that place every detected object in the room that owns it. Nothing here
computes topology — ``/scene_graph`` already carries it, ``doors[].room_pairs``
*is* the adjacency edge (already vetted against which rooms genuinely touch,
so a pair here never crosses a wall) and ``rooms[].objects`` *is* the
containment edge — these functions only paint what the mapper published.

Geometry and colors match the flown markers: the edge is the same gold
``(1.0, 0.85, 0.0)`` dog-leg through the door rather than a straight
centroid-to-centroid line (a straight line cuts through the wall the door is
in), the object link is the same semi-transparent green
``(0.2, 0.7, 0.2, 0.55)`` straight run from the object to its room centroid,
the room chip is the old ``R{pid} t={tau}s F={F}`` text plus the LLM label and
the oracle probability, and door labels are ``d{index}``.

Draw order matters and is owned by ``viz_render.render_map_panel``: object
links above the room tint, gold edges above the links, doors and object dots
above the edges, room nodes and their text chips above everything but the
drone.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from sparx_agency.tasks.mapping.scene_graph.viz_canvas import (
    CHIP_ALPHA, CHIP_BG, DIM_TEXT_BGR, DOOR_BGR, DOOR_LABEL_BGR,
    DOOR_UNDISCOVERED_BGR, DRONE_BGR, EDGE_BGR, EDGE_OUTLINE_BGR, FONT,
    LEGEND_ALPHA, LEGEND_BG, LEGEND_BORDER_BGR, NODE_OUTLINE_BGR,
    NODE_RING_BGR, OBJ_EDGE_ALPHA, OBJ_EDGE_BGR, TEXT_BGR, TRAIL_BGR,
    blend_box, blend_color, class_bgr, pixels_per_metre, room_bgr, world_to_px)

DOOR_RADIUS_M = 0.25
"""Door glyph half-width in metres (the old RViz ``viz_door_r``)."""

MAX_OBJECT_LINKS_PER_ROOM = 40
"""Object -> room links drawn per room before the rest are dropped.

A ward the drone has stared at can hold hundreds of confirmed landmarks, and
every one of them wants a line to the same centroid: past a few dozen the star
stops being a graph and becomes a solid green fan that buries the gold door
topology underneath it. The links are drawn in the order ``/scene_graph``
lists them (landmark id order, so it is stable frame to frame rather than
flickering between which links survive), and :func:`draw_legend` says on the
picture itself how many were hidden.
"""

OBJ_EDGE_PX = 1
"""Object-link line width, in pixels — thin, and deliberately not antialiased.

An antialiased line would blend a *ramp* of greens into the panel; a hairline
blended once through a binary mask lands on exactly one predictable color, so
"is this pixel an object link?" stays a question the tests can answer.
"""

DOOR_LABEL_MIN_PPM = 12.0
"""Below this zoom a ``d{index}`` label is smaller than the door — drop it."""

DOOR_HALF_PX = (3, 10)
"""Clamp on the door glyph half-size, so it stays visible and never a blob."""

LEGEND_MIN_SIZE = (480, 360)
"""A legend below this panel size is unreadable clutter, so it is not drawn."""

_LEGEND_ROWS = (
    ("rooms", "room (one hue each)"),
    ("room_outline", "room outline = its extent"),
    ("edge", "gold link = shared door"),
    ("door", "door reached"),
    ("door_pending", "door not reached yet"),
    ("object", "object (hue = class)"),
    ("object_link", "green link = object in room"),
    ("drone", "drone"),
    ("trail", "flown trail"),
)


def _edge_pids(door, known):
    # type: (dict, dict) -> list
    """The room pid pairs a door carries, restricted to known rooms.

    Read off ``room_pairs`` rather than paired up from ``rooms``: the rooms
    around one door are not all mutually adjacent, so enumerating pairs of the
    room list draws edges through walls. A pair naming an unknown pid (a stale
    payload after a BEV reshape) or naming one room twice is dropped.
    """
    out = []
    for pair in door.get("room_pairs") or []:
        if len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if a == b or a not in known or b not in known:
            continue
        if (a, b) not in out:
            out.append((a, b))
    return out


def draw_room_edges(panel, graph, extent):
    """Gold room -> door -> room links, one per discovered door pair.

    An undiscovered door is not an edge: the drone has not been through it, so
    the mapper has not proved the two rooms connect. A dark halo goes down
    first so the gold reads on both the pale floor plan and a dark room tint.
    """
    if not graph:
        return
    size = (panel.shape[1], panel.shape[0])
    centroids = {int(r["id"]): r["centroid"] for r in graph.get("rooms") or []}
    for door in graph.get("doors") or []:
        if not bool(door.get("discovered")):
            continue
        pairs = _edge_pids(door, centroids)
        if not pairs:
            continue
        via = world_to_px(extent, size, float(door["xy"][0]),
                          float(door["xy"][1]))
        for pid_a, pid_b in pairs:
            a = world_to_px(extent, size, *centroids[pid_a])
            b = world_to_px(extent, size, *centroids[pid_b])
            leg = np.array([a, via, b], dtype=np.int32)
            cv2.polylines(panel, [leg], False, EDGE_OUTLINE_BGR, 5,
                          cv2.LINE_AA)
            cv2.polylines(panel, [leg], False, EDGE_BGR, 2, cv2.LINE_AA)


def draw_room_object_edges(panel, graph, extent):
    # type: (np.ndarray, Optional[dict], tuple) -> int
    """Green object -> room links: the containment half of the scene graph.

    ``/scene_graph`` already answers "which room owns this object?" — every
    room carries its own ``objects`` list — so this needs no matching, only a
    line per object from where it was mapped to the node of the room that
    claims it. That line is what turns a field of loose dots into a graph.

    Every link goes into one binary mask and is blended in a single pass, so
    crossing links never blend twice into a darker green than one link alone,
    and the color a link paints is the same everywhere it crosses the same
    background.

    Args:
        panel: The map panel, modified in place.
        graph: The parsed ``/scene_graph`` payload, or None.
        extent: World rectangle the panel shows, ``(min_x, min_y, max_x,
            max_y)``.

    Returns:
        How many links were **not** drawn because their room was over
        :data:`MAX_OBJECT_LINKS_PER_ROOM`; 0 when the whole graph fits.
        :func:`draw_legend` prints this so a thinned picture never passes for
        a complete one.
    """
    if not graph:
        return 0
    size = (panel.shape[1], panel.shape[0])
    mask = np.zeros((panel.shape[0], panel.shape[1]), dtype=np.uint8)
    dropped = 0
    for room in graph.get("rooms") or []:
        centroid = room.get("centroid")
        objects = room.get("objects") or []
        if centroid is None:
            continue
        if len(objects) > MAX_OBJECT_LINKS_PER_ROOM:
            dropped += len(objects) - MAX_OBJECT_LINKS_PER_ROOM
            objects = objects[:MAX_OBJECT_LINKS_PER_ROOM]
        node = world_to_px(extent, size, float(centroid[0]),
                           float(centroid[1]))
        for obj in objects:
            xy = obj.get("xy")
            if xy is None:
                continue
            here = world_to_px(extent, size, float(xy[0]), float(xy[1]))
            cv2.line(mask, here, node, 255, OBJ_EDGE_PX)
    blend_color(panel, mask != 0, OBJ_EDGE_BGR, OBJ_EDGE_ALPHA)
    return dropped


def draw_doors(panel, graph, extent):
    """Doors: filled once reached, a hollow ghost while still pending.

    The glyph is sized from the live zoom (``DOOR_RADIUS_M`` on the ground)
    and the ``d{index}`` label is dropped entirely when the panel shows too few
    pixels per metre to read it — a hospital-wide view would otherwise be a
    field of overlapping text.
    """
    if not graph:
        return
    size = (panel.shape[1], panel.shape[0])
    ppm = pixels_per_metre(extent, size)
    half = int(min(max(round(DOOR_RADIUS_M * ppm), DOOR_HALF_PX[0]),
                   DOOR_HALF_PX[1]))
    for door in graph.get("doors") or []:
        px, py = world_to_px(extent, size, float(door["xy"][0]),
                             float(door["xy"][1]))
        discovered = bool(door.get("discovered"))
        color = DOOR_BGR if discovered else DOOR_UNDISCOVERED_BGR
        cv2.rectangle(panel, (px - half, py - half), (px + half, py + half),
                      color, -1 if discovered else 1)
        if ppm < DOOR_LABEL_MIN_PPM or "index" not in door:
            continue
        label_bgr = DOOR_LABEL_BGR if discovered else DOOR_UNDISCOVERED_BGR
        cv2.putText(panel, "d%d" % int(door["index"]),
                    (px + half + 2, py - half - 2), FONT, 0.32, label_bgr, 1,
                    cv2.LINE_AA)


def draw_room_nodes(panel, graph, labels, probs, extent):
    # type: (np.ndarray, Optional[dict], Optional[dict], Optional[dict], tuple) -> None
    """A graph node per room: colored disc, white ring, and its text chip."""
    if not graph:
        return
    size = (panel.shape[1], panel.shape[0])
    for room in graph.get("rooms") or []:
        pid = int(room["id"])
        color = room_bgr(pid)
        px, py = world_to_px(extent, size, float(room["centroid"][0]),
                             float(room["centroid"][1]))
        cv2.circle(panel, (px, py), 8, NODE_OUTLINE_BGR, -1, cv2.LINE_AA)
        cv2.circle(panel, (px, py), 7, NODE_RING_BGR, -1, cv2.LINE_AA)
        cv2.circle(panel, (px, py), 5, color, -1, cv2.LINE_AA)
        _draw_room_chip(panel, _room_chip_text(room, pid, labels, probs),
                        (px, py), color)


def _room_chip_text(room, pid, labels, probs):
    # type: (dict, int, Optional[dict], Optional[dict]) -> str
    """``R3 ward  t=12s F=2  p=0.90`` — the old marker text, plus what we know.

    Dwell time and frontier count are the two numbers the search loop reasons
    about (a room with time on it and no frontiers left is done), which is why
    the old RViz label carried them; the LLM label and oracle probability are
    appended only where they exist, so an unlabeled room still shows its stats.
    """
    text = "R%d" % pid
    label = ((labels or {}).get(str(pid)) or {}).get("label", "")
    if label:
        text += " %s" % label
    text += "  t=%.0fs F=%d" % (float(room.get("time_in_room_s", 0.0)),
                                int(room.get("frontier_clusters", 0)))
    prob = (probs or {}).get(pid)
    if prob is not None:
        text += "  p=%.2f" % float(prob)
    return text


def _draw_room_chip(panel, text, anchor, color):
    """A label plate floating above a node, like the old RViz text marker.

    Above and centred rather than hung off the side: a chip beside the node
    lies straight across the gold edges leaving it, and those edges are the
    whole point of this view. The plate is blended, not opaque, so a wall or an
    edge running under a long chip still reads through it.
    """
    px, py = anchor
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    x0 = min(max(px - tw // 2, 4), max(4, panel.shape[1] - tw - 4))
    y0 = min(max(py - 14, th + 6), panel.shape[0] - 4)
    box = (x0 - 3, y0 - th - 3, x0 + tw + 3, y0 + 3)
    blend_box(panel, box[0], box[1], box[2], box[3], CHIP_BG, CHIP_ALPHA)
    cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), color, 1)
    cv2.putText(panel, text, (x0, y0), FONT, 0.42, TEXT_BGR, 1, cv2.LINE_AA)


def _legend_glyph(panel, kind, x, cy, width):
    """One legend swatch, drawn in the span ``[x, x + width]`` about ``cy``."""
    if kind == "rooms":
        for i, pid in enumerate((0, 1)):
            left = x + i * 12
            cv2.rectangle(panel, (left, cy - 5), (left + 9, cy + 4),
                          room_bgr(pid), -1)
    elif kind == "room_outline":
        cv2.rectangle(panel, (x + 4, cy - 5), (x + 17, cy + 4), room_bgr(0), 1)
    elif kind == "object_link":
        # Opaque here, unlike the blended link on the map: a legend swatch has
        # to be readable against the legend plate, not sample the floor plan.
        cv2.circle(panel, (x + 3, cy + 3), 2, class_bgr("object"), -1,
                   cv2.LINE_AA)
        cv2.line(panel, (x + 3, cy + 3), (x + width - 2, cy - 4), OBJ_EDGE_BGR,
                 2, cv2.LINE_AA)
    elif kind == "edge":
        leg = np.array([(x, cy + 4), (x + width // 2, cy - 4),
                        (x + width, cy + 4)], dtype=np.int32)
        cv2.polylines(panel, [leg], False, EDGE_OUTLINE_BGR, 5, cv2.LINE_AA)
        cv2.polylines(panel, [leg], False, EDGE_BGR, 2, cv2.LINE_AA)
    elif kind in ("door", "door_pending"):
        seen = kind == "door"
        color = DOOR_BGR if seen else DOOR_UNDISCOVERED_BGR
        cv2.rectangle(panel, (x + 4, cy - 5), (x + 13, cy + 4), color,
                      -1 if seen else 1)
    elif kind == "object":
        cv2.circle(panel, (x + 9, cy), 4, class_bgr("object"), -1, cv2.LINE_AA)
    elif kind == "drone":
        nose = np.array([(x + 15, cy), (x + 4, cy - 6), (x + 4, cy + 6)],
                        dtype=np.int32)
        cv2.fillConvexPoly(panel, nose, DRONE_BGR, cv2.LINE_AA)
    elif kind == "trail":
        cv2.line(panel, (x + 2, cy), (x + width - 2, cy), TRAIL_BGR, 2,
                 cv2.LINE_AA)


def draw_legend(panel, dropped_object_links=0):
    # type: (np.ndarray, int) -> None
    """The vocabulary of the map panel, bottom-left, on a dimmed plate.

    Sized from the measured text so no caption is ever clipped, and skipped
    entirely on a panel too small to spare the room (``LEGEND_MIN_SIZE``).

    Args:
        panel: The map panel, modified in place.
        dropped_object_links: The count :func:`draw_room_object_edges`
            returned. Anything above zero adds a footnote naming the cap, so
            an operator counting green links off a busy frame is told the
            picture is thinned rather than left to miscount.
    """
    h, w = panel.shape[0], panel.shape[1]
    if w < LEGEND_MIN_SIZE[0] or h < LEGEND_MIN_SIZE[1]:
        return
    pad, glyph_w, row_h = 8, 26, 16
    note = ""
    if int(dropped_object_links) > 0:
        note = "+%d links hidden (cap %d/room)" % (int(dropped_object_links),
                                                   MAX_OBJECT_LINKS_PER_ROOM)
    content_w = max([glyph_w + 6 + cv2.getTextSize(t, FONT, 0.36, 1)[0][0]
                     for _, t in _LEGEND_ROWS]
                    + [cv2.getTextSize("SCENE GRAPH", FONT, 0.4, 1)[0][0]]
                    + ([cv2.getTextSize(note, FONT, 0.34, 1)[0][0]]
                       if note else []))
    x0 = 10
    x1 = x0 + 2 * pad + content_w
    y1 = h - 10
    y0 = y1 - (2 * pad + 18 + row_h * (len(_LEGEND_ROWS) + bool(note)))
    blend_box(panel, x0, y0, x1, y1, LEGEND_BG, LEGEND_ALPHA)
    cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), LEGEND_BORDER_BGR, 1)
    cv2.putText(panel, "SCENE GRAPH", (x0 + pad, y0 + pad + 11), FONT, 0.4,
                TEXT_BGR, 1, cv2.LINE_AA)
    y = y0 + pad + 18
    for kind, caption in _LEGEND_ROWS:
        cy = y + row_h // 2
        _legend_glyph(panel, kind, x0 + pad, cy, glyph_w)
        cv2.putText(panel, caption, (x0 + pad + glyph_w + 6, cy + 4), FONT,
                    0.36, TEXT_BGR, 1, cv2.LINE_AA)
        y += row_h
    if note:
        cv2.putText(panel, note, (x0 + pad, y + row_h // 2 + 4), FONT, 0.34,
                    DIM_TEXT_BGR, 1, cv2.LINE_AA)
