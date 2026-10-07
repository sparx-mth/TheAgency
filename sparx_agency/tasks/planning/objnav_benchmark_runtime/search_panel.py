"""The search panel of the ObjectNav dashboard: what the loop believes, and what it will do next.

One column beside the floor maps that reads, top to bottom, like the loop's
own reasoning: independent node success estimates and their joint-failure
approximation; the RPT* visit order with the node it is
heading to; every room with its type, the oracle's probability and its
reason, frontier left, time spent and the objects seen in it; every
staircase offered with its probability, direction, the storey beyond and the
climb's cost; every opening offered with its probability, the room it opens
from and what was glimpsed through it; the objects on the floor; and the
loop's last few events.
Everything drawn here comes from :func:`~sparx_agency.tasks.planning
.objnav_benchmark_runtime.visualization.method_snapshot`, so the video and
``steps.jsonl`` agree by construction.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_PX = 17
WHITE, GREY, DIM = (230, 230, 230), (170, 170, 170), (110, 110, 110)
GREEN, AMBER, RED, CYAN, BLUE = (80, 220, 80), (0, 200, 255), (90, 90, 235), (255, 200, 0), (255, 140, 60)
VIOLET = (220, 120, 220)
#: Node id bands: stairs from 100 000, openings from 200 000 (``stair_nodes`` / ``opening_nodes``); within the
#: opening band, target landmarks from index 50 000.
STAIR_BASE, OPENING_BASE, LANDMARK_INDEX_BASE = 100_000, 200_000, 50_000


class _Column:
    """A cursor that writes one line at a time and stops at the bottom."""

    def __init__(self, image, x=8, y=22, scale=0.42):
        self.image, self.x, self.y, self.scale = image, x, y, scale
        self.bottom = image.shape[0] - 6

    def line(self, text, color=WHITE, indent=0, bold=False):
        if self.y > self.bottom:
            return False
        cv2.putText(self.image, str(text)[:64], (self.x + indent, self.y), FONT, self.scale, color,
                    2 if bold else 1, cv2.LINE_AA)
        self.y += LINE_PX
        return True

    def gap(self, px=6):
        self.y += px


def _fmt(value, digits=2, width=5):
    if value is None:
        return "-".rjust(width)
    return ("%%.%df" % digits % float(value)).rjust(width)


def _node_color(node, search):
    if node["id"] == search.get("room_in_force"):
        return AMBER
    if node["id"] == search.get("next_room"):
        return GREEN
    if node.get("kind") == "stairs":
        return BLUE
    if node.get("kind") == "landmark":
        return CYAN
    if node.get("kind") == "opening":
        return VIOLET
    if (node.get("prob") or 0.0) < 0.01:
        return RED
    if node.get("cooling"):
        return DIM
    return WHITE


def _name(node):
    kind, nid = node.get("kind"), int(node["id"])
    if kind == "landmark" and node.get("landmark_id") is not None:
        return "T%d" % int(node["landmark_id"])
    if kind in ("opening", "landmark") or nid >= OPENING_BASE:
        index = nid - OPENING_BASE if nid >= OPENING_BASE else nid
        return "T%d" % (index - LANDMARK_INDEX_BASE) if index >= LANDMARK_INDEX_BASE else "O%d" % index
    if kind == "stairs" or nid >= STAIR_BASE:
        return "S%d" % (nid - STAIR_BASE if nid >= STAIR_BASE else nid)
    return "R%d" % nid


def _header(col, search):
    col.line("TARGET: %s" % search.get("target", "?"), CYAN, bold=True)
    col.line("P(any success) %s | all fail %s | F%s | %s" % (
        _fmt(search.get("p_present"), 2, 4).strip(), _fmt(search.get("elsewhere"), 2, 4).strip(),
        search.get("floor_id", "?"), search.get("supervisor_state", "?")), GREY)
    warmup = search.get("warmup") or {}
    peek = (search.get("doorway_peek") or {}).get("active")
    if warmup and warmup.get("actions", 0) < warmup.get("budget", 0):
        col.line("WARM-UP %d/%d actions" % (warmup["actions"], warmup["budget"]), AMBER)
    if peek:
        col.line("PEEK R%d: %s" % (peek["room"], peek["phase"]), AMBER)
    glance = search.get("glance") or {}
    if glance.get("active"):
        active = glance["active"]
        col.line("GLANCE %s: turn %d, %.1f m2 of unknown in view" % (active.get("kind", "?"), active.get("turns", 0),
                                                                     active.get("gain_m2", 0.0)), AMBER)
    elif glance.get("planned"):
        planned = glance["planned"]
        col.line("glance %s planned %.1f m ahead: %.1f m2 for %d actions" % (
            planned.get("kind", "?"), planned.get("along_m", 0.0), planned.get("gain_m2", 0.0), planned.get("actions", 0)), GREY)
    col.line("Accessible frontiers: %d (blue diamonds)" % len(search.get("accessible_frontiers", ())), GREY)
    floor = search.get("floor") or {}
    if floor:
        col.line("building: %s | arrived by stairs %s | active portal %s (%s)" % (
            floor.get("phase", "SEARCH"), floor.get("arrived_by", "-"),
            floor.get("active_portal", "-"), floor.get("selected_by") or "-"), GREY)
    reading = search.get("reading") or {}
    if reading.get("home"):
        col.line("LLM: home = %s" % reading["home"], GREY)
    if reading.get("house"):
        col.line("LLM: house = %s" % reading["house"], GREY)
    if reading.get("stage"):
        col.line("LLM: stage = %s" % reading["stage"], GREY)
    if reading.get("pass"):
        col.line("LLM: pass = %s%s" % (reading["pass"].upper(),
                                        " -- finished rooms are nodes again" if reading["pass"] == "second" else ""), GREY)
    if reading.get("storey"):
        col.line("LLM: storey = %s" % reading["storey"], GREY)
    if reading.get("home_here"):
        col.line("LLM: home type %s on this storey%s" % (
            {"found": "FOUND", "missing": "MISSING", "elsewhere": "ELSEWHERE"}.get(reading["home_here"], reading["home_here"]),
            " -- unexplored places read at the elsewhere value" if reading["home_here"] == "elsewhere" else ""), GREY)
    col.gap()


def _order(col, search):
    order, head = list(search.get("order") or ()), search.get("order_index", 0)
    nodes = {n["id"]: n for n in list(search.get("rooms", ())) + list(search.get("stairs", ()))
             + list(search.get("openings", ()))}
    col.line("RPT* VISIT ORDER (rooms, stairs, openings; costs charged)", CYAN, bold=True)
    if not order:
        col.line("(no order: nothing in force)", DIM, indent=8)
    else:
        names = [_name(nodes.get(nid, {"id": nid})) for nid in order]
        col.line(" -> ".join(("[%s]" if i == head else "%s") % name for i, name in enumerate(names)), WHITE, indent=8)
    room_id = search.get("room_in_force")
    if room_id is not None:
        node = nodes.get(room_id, {"id": room_id})
        peek = search.get("peek") or {}
        scan = search.get("scan") or {}
        phase = ""
        if node.get("kind") == "opening" or (peek and peek.get("node") == room_id):
            phase = "  peek: look %d/%d (%d turns)" % (peek.get("look", 0), peek.get("of", 0), peek.get("turns", 0))
        elif scan:
            phase = "  %s%s" % (scan.get("phase", ""), " (%d turns, %d deg)" % (
                scan.get("turns", 0), int(round(math.degrees(float(scan.get("swept", 0.0))))))
                if scan.get("phase") == "rotate" else " (%d actions)" % scan.get("approach_actions", 0))
        col.line("IN FORCE: %s %s  visit step %d/%d%s" % (_name(node), node.get("label", "?"),
                                                            search.get("local_steps", 0), search.get("local_budget", 0),
                                                            phase), AMBER, indent=8)
    elif search.get("next_room") is not None:
        node = nodes.get(search["next_room"], {"id": search["next_room"]})
        peek = search.get("peek") or {}
        how = ("approaching the stairs" if node.get("kind") == "stairs" else
               "walking to the threshold (%d/%d actions)" % (peek.get("approach_actions", 0), peek.get("approach_bound", 0))
               if node.get("kind") == "opening" else "in transit")
        col.line("NEXT: %s %s (%s)" % (_name(node), node.get("label", "?"), how), GREEN, indent=8)
    else:
        col.line("NEXT: %s" % search.get("fallback", "floor-wide frontier"), GREY, indent=8)
    excluded = search.get("excluded") or {}
    if excluded:
        col.line("NOT NODES: " + ", ".join("R%s %s" % (pid, why) for pid, why in list(excluded.items())[:6])
                 + (" ..." if len(excluded) > 6 else ""), DIM, indent=8)
    col.gap()


def _rooms(col, search):
    excluded = search.get("excluded") or {}
    col.line("ROOMS  id type            p   F(access) s ago", CYAN, bold=True)
    for room in search.get("rooms", ())[:10]:
        label = (room.get("label") or "?")[:14]
        if room.get("strength") == "weak":
            label = label[:13] + "?"
        ago = room.get("last_inside_step")
        col.line("R%-3d %-15s%s %2s %3s %4s%s" % (
            room["id"], label, _fmt(room.get("prob")),
            "-" if room.get("frontier_clusters") is None else room["frontier_clusters"],
            "-" if room.get("searched_s") is None else "%d" % room["searched_s"],
            "-" if ago is None else "%d" % ago,
            "  [%s]" % excluded[str(room["id"])].split(":")[0] if str(room["id"]) in excluded else ""),
            DIM if str(room["id"]) in excluded else _node_color(room, search))
        detail = []
        if room.get("why"):
            detail.append(room["why"])
        if room.get("objects"):
            detail.append("seen: " + ", ".join(sorted(set(room["objects"]))[:5]))
        if detail:
            col.line(" | ".join(detail), DIM, indent=24)
    col.gap()


def _stairs(col, search):
    stairs = search.get("stairs", ())
    col.line("STAIRS (nodes)  p     climb m  storey beyond", CYAN, bold=True)
    if not stairs:
        col.line("(none reachable on the observed map)", DIM, indent=8)
    for node in stairs:
        col.line("%-5s %-5s %s  %6s  %s%s" % (
            _name(node), "up" if node.get("direction", 0) > 0 else "down", _fmt(node.get("prob")),
            _fmt(node.get("leaf_m"), 1, 6).strip(),
            "visited" if node.get("destination_visited") else "unvisited",
            " (arrived by)" if node.get("arrived_by") else ""), _node_color(node, search))
        if node.get("why"):
            col.line(node["why"], DIM, indent=24)
    col.gap()


def _openings(col, search):
    openings = search.get("openings", ())
    if not openings:
        return
    col.line("OPENINGS (nodes)  p     off room  glimpsed / target landmark", CYAN, bold=True)
    for node in openings[:6]:
        via = node.get("via") or "?"
        col.line("%-5s %-9s %s  %-9s %s" % (
            _name(node), (node.get("label") or "gap")[:9], _fmt(node.get("prob")),
            via.split(" (")[0].replace("room ", "R") if via != "?" else "-",
            ("look at the %s" % (node.get("glimpsed") or ["target"])[0]) if node.get("kind") == "landmark"
            else ", ".join(sorted(set(node.get("glimpsed") or ()))[:4]) or "nothing yet"), _node_color(node, search))
        if node.get("why"):
            col.line(node["why"], DIM, indent=24)
    if len(openings) > 6:
        col.line("... %d more" % (len(openings) - 6), DIM, indent=8)
    col.gap()


def _objects(col, search):
    objects = search.get("objects", ())
    col.line("OBJECTS (%d confirmed)" % len(objects), CYAN, bold=True)
    for obj in objects[:6]:
        col.line("%s x%d  @ %s" % (obj["class"], obj.get("count", 1),
                                    "R%d" % obj["room"] if obj.get("room") is not None else "no room"), WHITE, indent=8)
    if len(objects) > 6:
        col.line("... %d more" % (len(objects) - 6), DIM, indent=8)
    col.gap()


def _events(col, search):
    col.line("LOOP EVENTS", CYAN, bold=True)
    for event in search.get("events", ())[-4:]:
        text = " ".join("%s=%s" % (k, v) for k, v in event.items()
                        if k not in ("floor_id", "goal", "order", "objects", "stairs"))
        col.line(text, GREY, indent=8)


def render_search_panel(search, size=(320, 720)):
    """Draw the search block of a snapshot as one text column of ``size`` (w, h)."""
    image = np.full((size[1], size[0], 3), 24, np.uint8)
    col = _Column(image)
    for section in (_header, _order, _rooms, _stairs, _openings, _objects, _events):
        section(col, search or {})
        if col.y > col.bottom:
            break
    return image

