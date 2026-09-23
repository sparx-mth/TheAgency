"""Pure rendering for the live scene-graph visualization.

Every pixel of the mission dashboard is drawn here, from plain dicts (parsed
topic JSON) and numpy arrays, so the whole picture is unit-testable without
ROS. The node in ``ros2/scene_graph_viz_node.py`` only collects state and
calls :func:`render_scene`.

Canvas layout (default 1600x900):

* **left map panel** -- the hospital floor plan backdrop (dimmed), the live BEV
  over it (occupied dark, free washed light), per-room tints and crisp per-room
  outlines from the latched room-label grid, the **room graph** (gold room ->
  door -> room edges, green object -> room links, room nodes, door state) from
  :mod:`sparx_agency.tasks.mapping.scene_graph.viz_graph_overlay`, object dots,
  the drone triangle + trail, a legend, and a target-seen banner.
* **right panel** -- mission title (target / sim time / oracle source), one row
  per room sorted by search probability, and a heartbeat footer.

World is ENU metres (the ``/simple_drone/odom`` frame); grids follow the
``nav_msgs/OccupancyGrid`` layout, row 0 at minimum y. Panel images are
ordinary pictures, row 0 at maximum y — :func:`viz_canvas.world_to_px` owns the
flip, and the palette and grid resampling live there too.

The Voronoi skeleton the old RViz view drew in "open space" is deliberately
absent: it is not on any topic, and this module renders only what the mapper
actually published.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from sparx_agency.tasks.mapping.scene_graph.viz_canvas import (
    BACKDROP_DIM, BAR_BG_BGR, BEV_FREE_ALPHA, BEV_FREE_BGR, BEV_OCC_BGR,
    DIM_TEXT_BGR, DRONE_BGR, FONT, PANEL_BG, ROOM_TINT_ALPHA, SIDE_BG,
    TARGET_BGR, TEXT_BGR, TRAIL_BGR, blend_color, class_bgr, compute_extent,
    room_bgr, warp_grid, world_to_px)
from sparx_agency.tasks.mapping.scene_graph.viz_graph_overlay import (
    draw_doors, draw_legend, draw_room_edges, draw_room_nodes,
    draw_room_object_edges)


def _draw_bev(panel, bev, extent):
    """The live BEV: free space washed light, occupied bright, unknown absent."""
    if bev is None:
        return
    # int8 {-1,0,100} -> uint8 {0,1,101}; warpAffine cannot take int8.
    shifted = (bev["grid"].astype(np.int16) + 1).astype(np.uint8)
    warped = warp_grid(shifted, bev["resolution"], bev["origin"], extent,
                       (panel.shape[1], panel.shape[0]))
    blend_color(panel, warped == 1, BEV_FREE_BGR, BEV_FREE_ALPHA)
    panel[warped == 101] = BEV_OCC_BGR


def _warp_room_values(room_grid, extent, size):
    """The latched room-value grid resampled into panel pixels, or None.

    Both room layers — the tint and the outline — read the same warped image,
    so the resample happens once per frame rather than once per layer.
    """
    if room_grid is None:
        return None
    return warp_grid(np.clip(room_grid["grid"], 0, 127).astype(np.uint8),
                     room_grid["resolution"], room_grid["origin"], extent,
                     size)


def _room_color_lut(warped, graph):
    """``grid value -> BGR`` lookup for the room values present on the panel.

    The grid carries a small per-room *value*, not the pid (see
    ``ros2/payloads.py``), and ``/scene_graph`` carries the ``grid_pid_map``
    that resolves one to the other; the documented fallback when no map has
    arrived yet is to treat the value as the pid. Resolving every value once
    into a 256-entry table lets both room layers color a whole panel with one
    indexing pass, instead of one full-image comparison per room.
    """
    pid_map = (graph or {}).get("grid_pid_map") or {}
    lut = np.zeros((256, 3), dtype=np.uint8)
    for value in np.unique(warped):
        value = int(value)
        if value == 0:
            continue
        lut[value] = room_bgr(int(pid_map.get(str(value), value)))
    return lut


def _tint_rooms(panel, warped, lut):
    """Tint each room's area with its stable color, via the latched pid grid."""
    mask = warped != 0
    if not np.any(mask):
        return
    tint = lut[warped[mask]].astype(np.float32)
    panel[mask] = (panel[mask].astype(np.float32) * (1.0 - ROOM_TINT_ALPHA)
                   + tint * ROOM_TINT_ALPHA).astype(np.uint8)


def _outline_rooms(panel, warped, lut):
    """A crisp full-opacity boundary around every room, in the room's color.

    The old RViz view drew this as a ``room_polygons`` LINE_STRIP per contour
    of the room mask; here the mask is already rasterized into panel pixels, so
    the boundary is the set of room cells whose 4-neighbour holds a *different*
    value. That is four whole-image comparisons regardless of how many rooms
    are on the map — a ``findContours`` pass per room would grow with the
    hospital, and this runs every render tick.

    Each room paints only its own side of a shared boundary, so two rooms that
    touch get one pixel of each color rather than one arbitrary winner, and a
    room running off the panel is left open there rather than being closed off
    by a border that is not a wall.
    """
    edge = np.zeros(warped.shape, dtype=bool)
    vertical = warped[:-1, :] != warped[1:, :]
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    horizontal = warped[:, :-1] != warped[:, 1:]
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge &= warped != 0
    if np.any(edge):
        panel[edge] = lut[warped[edge]]


def _draw_objects(panel, objects, extent):
    if not objects:
        return
    size = (panel.shape[1], panel.shape[0])
    for obj in objects.get("objects", []):
        color = class_bgr(str(obj["class"]))
        px, py = world_to_px(extent, size, obj["xy"][0], obj["xy"][1])
        cv2.circle(panel, (px, py), 4, color, -1, cv2.LINE_AA)
        text = str(obj["class"])
        if int(obj.get("count", 1)) > 1:
            text += " x%d" % int(obj["count"])
        cv2.putText(panel, text, (px + 6, py + 3), FONT, 0.32, color, 1, cv2.LINE_AA)


def _draw_trail(panel, trail, extent):
    if not trail or len(trail) < 2:
        return
    size = (panel.shape[1], panel.shape[0])
    pts = np.array([world_to_px(extent, size, p[0], p[1]) for p in trail],
                   dtype=np.int32)
    cv2.polylines(panel, [pts], False, TRAIL_BGR, 2, cv2.LINE_AA)


def _draw_drone(panel, pose, extent):
    """A yaw-oriented triangle: long nose, base corners swept back +/-140 deg.

    Pixel rows grow downward while world y grows upward, hence the minus on
    every sin term — without it the triangle points the mirror heading.
    """
    if pose is None:
        return
    yaw = float(pose[2])
    px, py = world_to_px(extent, (panel.shape[1], panel.shape[0]),
                         float(pose[0]), float(pose[1]))
    corners = np.array([
        (px + 12 * np.cos(yaw), py - 12 * np.sin(yaw)),
        (px + 7 * np.cos(yaw + 2.45), py - 7 * np.sin(yaw + 2.45)),
        (px + 7 * np.cos(yaw - 2.45), py - 7 * np.sin(yaw - 2.45)),
    ], dtype=np.int32)
    cv2.fillConvexPoly(panel, corners, DRONE_BGR, cv2.LINE_AA)


def _draw_target(panel, state, extent):
    """Bold banner + a ring at the matched object once the target is seen."""
    if not state.get("target_seen"):
        return
    info = state.get("target_info") or {}
    size = (panel.shape[1], panel.shape[0])
    xy = info.get("xy")
    if xy is not None:
        px, py = world_to_px(extent, size, float(xy[0]), float(xy[1]))
        cv2.circle(panel, (px, py), 14, TARGET_BGR, 3, cv2.LINE_AA)
    text = "TARGET SEEN: %s" % (info.get("target") or "?")
    matched = info.get("matched_class")
    if matched and matched != info.get("target"):
        text += " (%s)" % matched
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 120), -1)
    cv2.putText(panel, text, (10, 24), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def _oracle_probs(state):
    # type: (dict) -> Dict[int, float]
    """``pid -> search probability`` from the latched oracle, for the chips."""
    return {int(entry["id"]): float(entry.get("prob", 0.0))
            for entry in (state.get("oracle") or {}).get("rooms", [])}


def _has_map_content(state):
    # type: (dict) -> bool
    """True once any live layer has arrived — the legend explains nothing
    before that, and a startup frame should stay clean."""
    if state.get("bev") is not None or state.get("room_grid") is not None:
        return True
    if state.get("pose") is not None or state.get("trail"):
        return True
    graph = state.get("scene_graph") or {}
    if graph.get("rooms") or graph.get("doors"):
        return True
    return bool((state.get("objects") or {}).get("objects"))


def render_map_panel(state, backdrop, size):
    # type: (dict, object, Tuple[int, int]) -> np.ndarray
    """The left panel: floor plan + live layers. See the module docstring.

    The order below is the picture's legibility contract: room tint and the
    crisp room outline over it, then the green object -> room links, then the
    gold graph edges over those, then doors and object dots over the edges,
    then the room nodes and their chips, then the aircraft, and the legend and
    banner last so nothing ever paints over them. The two link layers are
    stacked that way on purpose — the gold door topology is the mission's
    skeleton and has to stay dominant, while the green containment links read
    as a softer underlay saying which room owns what.
    """
    w, h = int(size[0]), int(size[1])
    extent = compute_extent(state, backdrop, (w, h))
    if backdrop is not None:
        base = backdrop.whole((w, h)).image
        panel = (base.astype(np.float32) * BACKDROP_DIM).astype(np.uint8)
    else:
        panel = np.full((h, w, 3), PANEL_BG, dtype=np.uint8)
    graph = state.get("scene_graph")
    _draw_bev(panel, state.get("bev"), extent)
    warped_rooms = _warp_room_values(state.get("room_grid"), extent, (w, h))
    if warped_rooms is not None:
        lut = _room_color_lut(warped_rooms, graph)
        _tint_rooms(panel, warped_rooms, lut)
        _outline_rooms(panel, warped_rooms, lut)
    dropped_links = draw_room_object_edges(panel, graph, extent)
    draw_room_edges(panel, graph, extent)
    draw_doors(panel, graph, extent)
    _draw_objects(panel, state.get("objects"), extent)
    draw_room_nodes(panel, graph, state.get("room_labels"),
                    _oracle_probs(state), extent)
    _draw_trail(panel, state.get("trail"), extent)
    _draw_drone(panel, state.get("pose"), extent)
    if _has_map_content(state):
        draw_legend(panel, dropped_links)
    _draw_target(panel, state, extent)
    return panel


def _room_rows(state):
    # type: (dict) -> List[dict]
    """Merge oracle probabilities, scene-graph stats and LLM labels into rows.

    The oracle is the sort order; rooms the oracle has not scored yet still get
    a row (prob None) so the panel never hides a discovered room.
    """
    graph = state.get("scene_graph") or {}
    labels = state.get("room_labels") or {}
    oracle = state.get("oracle") or {}
    rows = {}  # type: Dict[int, dict]
    for room in graph.get("rooms", []):
        pid = int(room["id"])
        rows[pid] = {"id": pid, "prob": None,
                     "label": (labels.get(str(pid)) or {}).get("label", ""),
                     "time_s": float(room.get("time_in_room_s", 0.0)),
                     "frontiers": int(room.get("frontier_clusters", 0)),
                     "objects": len(room.get("objects", []))}
    for entry in oracle.get("rooms", []):
        pid = int(entry["id"])
        row = rows.setdefault(pid, {"id": pid, "label": "", "objects": 0,
                                    "time_s": 0.0, "frontiers": 0})
        row["prob"] = float(entry.get("prob", 0.0))
        row["label"] = row["label"] or entry.get("label", "")
        row["time_s"] = row.get("time_s") or float(entry.get("time_in_room_s", 0.0))
        row["frontiers"] = row.get("frontiers") or int(entry.get("frontier_clusters", 0))
    return sorted(rows.values(),
                  key=lambda r: (-(r["prob"] if r["prob"] is not None else -1.0),
                                 r["id"]))


def render_side_panel(state, size):
    # type: (dict, Tuple[int, int]) -> np.ndarray
    """The right panel: mission title, per-room probability rows, footer."""
    w, h = int(size[0]), int(size[1])
    panel = np.full((h, w, 3), SIDE_BG, dtype=np.uint8)
    oracle = state.get("oracle") or {}
    target = oracle.get("target") or (state.get("target_info") or {}).get("target")
    cv2.putText(panel, "TARGET: %s" % (target or "(none yet)"), (16, 34),
                FONT, 0.8, TEXT_BGR, 2, cv2.LINE_AA)
    sub = "sim t=%.1fs   oracle: %s" % (
        float(state.get("sim_time") or 0.0), oracle.get("source") or "-")
    if oracle.get("model"):
        sub += "   model: %s" % oracle["model"]
    cv2.putText(panel, sub, (16, 62), FONT, 0.48, DIM_TEXT_BGR, 1, cv2.LINE_AA)
    cv2.line(panel, (0, 76), (w, 76), (70, 70, 76), 1)

    y, row_h = 96, 58
    rows = _room_rows(state)
    if not rows:
        cv2.putText(panel, "waiting for rooms...", (16, y + 12), FONT, 0.55,
                    DIM_TEXT_BGR, 1, cv2.LINE_AA)
    for i, row in enumerate(rows):
        if y + row_h > h - 60:
            cv2.putText(panel, "(+%d more rooms)" % (len(rows) - i),
                        (16, y + 12), FONT, 0.45, DIM_TEXT_BGR, 1, cv2.LINE_AA)
            break
        _draw_room_row(panel, row, y, w)
        y += row_h
    for i, line in enumerate(reversed(state.get("footer") or [])):
        cv2.putText(panel, str(line), (16, h - 14 - 20 * i), FONT, 0.42,
                    DIM_TEXT_BGR, 1, cv2.LINE_AA)
    return panel


def _draw_room_row(panel, row, y, w):
    """One room: swatch, id+label, probability bar, stats line."""
    color = room_bgr(row["id"])
    cv2.rectangle(panel, (16, y), (34, y + 18), color, -1)
    title = "R%d  %s" % (row["id"], row["label"] or "unlabeled")
    cv2.putText(panel, title, (44, y + 15), FONT, 0.55, TEXT_BGR, 1, cv2.LINE_AA)
    bar_x0, bar_x1 = 44, w - 90
    prob = row.get("prob")
    cv2.rectangle(panel, (bar_x0, y + 24), (bar_x1, y + 36), BAR_BG_BGR, -1)
    if prob is not None:
        fill = int(round((bar_x1 - bar_x0) * max(0.0, min(1.0, prob))))
        cv2.rectangle(panel, (bar_x0, y + 24), (bar_x0 + fill, y + 36), color, -1)
        cv2.putText(panel, "%.2f" % prob, (bar_x1 + 8, y + 35), FONT, 0.5,
                    TEXT_BGR, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "--", (bar_x1 + 8, y + 35), FONT, 0.5,
                    DIM_TEXT_BGR, 1, cv2.LINE_AA)
    stats = "searched %.0fs   frontiers %d   objects %d" % (
        row.get("time_s", 0.0), row.get("frontiers", 0), row.get("objects", 0))
    cv2.putText(panel, stats, (44, y + 52), FONT, 0.42, DIM_TEXT_BGR, 1, cv2.LINE_AA)


def render_scene(state, backdrop=None, size=(1600, 900), map_panel_w=760):
    # type: (dict, object, Tuple[int, int], int) -> np.ndarray
    """The whole dashboard frame.

    Args:
        state: Plain-data snapshot; every key optional. Keys: ``bev`` and
            ``room_grid`` (dicts of ``grid`` int8 (H, W) row 0 = min y,
            ``resolution``, ``origin`` (x, y)), ``scene_graph`` /
            ``room_labels`` / ``oracle`` / ``objects`` / ``target_info``
            (parsed topic JSON per the scene-graph contract), ``target_seen``
            (bool), ``pose`` ((x, y, yaw)), ``trail`` (list of (x, y)),
            ``sim_time`` (float sec), ``footer`` (list of status strings).
        backdrop: A ``map_backdrop.OccupancyMapImage`` or None.
        size: ``(width, height)`` of the canvas.
        map_panel_w: Width of the left map panel; the rest is the side panel.

    Returns:
        ``(height, width, 3)`` uint8 BGR canvas.
    """
    w, h = int(size[0]), int(size[1])
    map_w = max(120, min(int(map_panel_w), w - 120))
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:, :map_w] = render_map_panel(state, backdrop, (map_w, h))
    canvas[:, map_w:] = render_side_panel(state, (w - map_w, h))
    cv2.line(canvas, (map_w, 0), (map_w, h), (90, 90, 95), 1)
    return canvas
