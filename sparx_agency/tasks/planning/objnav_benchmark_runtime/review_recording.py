"""Read a recording the way a reviewer should: by step, from the log, never from a frame.

A recording directory (``runs/<campaign>/<scene>/recordings/<key>/``) holds one
``steps.jsonl`` row per decision -- pose, action, the method's snapshot (the
search column: rooms, order, openings, the frontier inventory, the loop's
events), the command -- plus ``metrics.json``, ``trajectory.csv`` and, since
2026-10-22, ``map_journal.npz``: the cells of the occupancy grid, the settled
unknown and the room partition that changed on every action. Together they
answer, in text, every question a frame of the video answers in pixels, and
several it does not: *what did the agent see to its left at action 261, how
much of it was unknown, how much of that had the sight ledger settled, and
which frontier goals and openings lay in that sector.*

Subcommands (``python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording <dir> <command>``):

* ``summary`` -- the episode's outcome, where the wall time went (the policy's
  latency ledger), how the selections ended (release verdicts), LLM calls.
* ``timeline [--start A --end B]`` -- one line per action: state, phase,
  room in force / next, order, control-counter deltas (A* / classifier /
  room LLM / RPT*), the route's reason, the pose, the events NEW on that
  action.
* ``step N [--radius R]`` -- everything the log holds about action N, and
  the map as it stood then summarised by sector round the agent (ahead,
  left, behind, right; within R metres): free / occupied / unknown /
  settled cells, raw frontier cells, the frontier goals and openings that
  fall in the sector, the rooms the sector touches.
* ``map N [--radius R --cell C]`` -- the map at action N round the agent as
  text: ``#`` occupied, ``.`` free, blank unknown, ``:`` settled unknown,
  ``F`` a frontier goal, ``O`` an opening, ``*`` the route, ``@`` the agent
  (``^ > v <`` beside it for the heading).
* ``events [--name NAME]`` -- every loop event with its action, or one kind.

Without ``map_journal.npz`` (recordings before 2026-10-22) the map commands
fall back to the FINAL map and say so. numpy and the standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

SECTORS = (("ahead", -45.0, 45.0), ("left", 45.0, 135.0), ("behind", 135.0, 225.0), ("right", 225.0, 315.0))
COUNTER_NAMES = (("astar", "A*"), ("room_classifier", "classifier"), ("room_llm", "room LLM"), ("rpt", "RPT*"))


# -- loading ---------------------------------------------------------------------------------------------------
def load_steps(directory):
    """Every ``steps.jsonl`` row, in order."""
    with (Path(directory) / "steps.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_json(directory, name):
    path = Path(directory) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class MapAtStep:
    """The three map layers of a floor slot as they stood after a given action, from the journal or the final maps."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.journal = None
        path = self.directory / "map_journal.npz"
        if path.is_file():
            self.journal = dict(np.load(path, allow_pickle=False))
            self.shape = tuple(int(v) for v in self.journal["shape"])
            self.origin = tuple(float(v) for v in self.journal["origin"])
            self.resolution = float(self.journal["resolution_m"])
            self.floor_ids = [int(v) for v in self.journal["floor_ids"]]
        else:
            maps = np.load(self.directory / "floor_maps.npz", allow_pickle=False)
            self.final_grids = {int(k.split("_")[1]): np.asarray(maps[k]) for k in maps.files}
            parts = np.load(self.directory / "room_partitions.npz", allow_pickle=False)
            self.final_labels = {int(k.split("_")[1]): np.asarray(parts[k]) for k in parts.files}
            self.shape = next(iter(self.final_grids.values())).shape
            panels = load_json(self.directory, "floor_panels.json") or []
            panel = panels[0] if panels else {}
            self.origin = tuple(float(v) for v in panel.get("origin", (0.0, 0.0)))
            self.resolution = float(panel.get("resolution_m", 0.1))
            self.floor_ids = [p.get("floor_id") if p.get("floor_id") is not None else -1 for p in panels] or [0]

    @property
    def exact(self):
        """Whether the map at a step is the step's own (the journal) rather than the final map."""
        return self.journal is not None

    def slot_of(self, floor_id):
        return self.floor_ids.index(int(floor_id)) if int(floor_id) in self.floor_ids else 0

    def at(self, step, floor_id=0):
        """``(grid int8, settled bool, room_labels int32)`` of ``floor_id`` after action ``step``."""
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import replay_journal
        slot = self.slot_of(floor_id)
        if self.journal is None:
            return (self.final_grids.get(slot, np.full(self.shape, -1, np.int8)), np.zeros(self.shape, bool),
                    self.final_labels.get(slot, np.zeros(self.shape, np.int32)))
        grid = replay_journal(self.journal, slot, "grid", step, self.shape, -1)
        settled = replay_journal(self.journal, slot, "settled", step, self.shape, 0).astype(bool)
        labels = replay_journal(self.journal, slot, "room_labels", step, self.shape, 0)
        return grid, settled, labels

    def world_to_grid(self, x, y):
        return (int(math.floor((x - self.origin[0]) / self.resolution)),
                int(math.floor((y - self.origin[1]) / self.resolution)))


# -- the rows' vocabulary -----------------------------------------------------------------------------------------
def counters(row):
    cc = (row.get("method") or {}).get("control_counters") or {}
    return tuple(int(cc.get(name, 0)) for name, _ in COUNTER_NAMES)


def counter_delta(previous, current):
    parts = ["%s+%d" % (label, current[i] - previous[i]) for i, (_, label) in enumerate(COUNTER_NAMES)
             if current[i] - previous[i]]
    return " ".join(parts)


def new_events(rows):
    """``{step: [event, ...]}`` -- every loop event the moment it first appears (the rows carry a sliding window)."""
    seen, out = set(), {}
    for row in rows:
        search = (row.get("method") or {}).get("search") or {}
        for event in search.get("events", ()):
            key = json.dumps(event, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.setdefault(int(event.get("step", row["step"])), []).append(event)
    return out


def event_text(event):
    fields = {k: v for k, v in event.items() if k not in ("event", "floor_id", "step")}
    if not fields:
        return event["event"]
    return "%s(%s)" % (event["event"], ", ".join("%s=%s" % (k, _short(v)) for k, v in fields.items()))


def _short(value):
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= 60 else text[:57] + "..."


def bearing_sector(dx, dy, yaw):
    """Which of :data:`SECTORS` a point at ``(dx, dy)`` from the agent falls in, given the agent's heading."""
    relative = (math.degrees(math.atan2(dy, dx) - yaw) + 45.0) % 360.0 - 45.0
    for name, low, high in SECTORS:
        if low <= relative < high:
            return name
    return "ahead"


# -- commands ------------------------------------------------------------------------------------------------------
def summary(directory, rows, out):
    metrics = load_json(directory, "metrics.json") or {}
    episode = load_json(directory, "episode.json") or {}
    out("episode %s  target %s  steps %s  success %s  SPL %s  DTG %s m  termination %s  wall %s s" % (
        episode.get("episode_id"), episode.get("target"), metrics.get("steps"), metrics.get("success"),
        _fmt(metrics.get("spl")), _fmt(metrics.get("distance_to_goal_m")), metrics.get("termination"),
        _fmt(metrics.get("wall_s"))))
    policy = ((metrics.get("agent_info") or {}).get("policy") or {})
    latency = ((policy.get("exploration_metrics") or {}).get("latency") or {})
    if latency:
        out("where the wall time went (the policy's latency ledger):")
        for name, item in sorted(latency.items(), key=lambda kv: -float(kv[1].get("total_ms", 0.0))):
            out("  %-18s n=%-4d total %7.1f s  median %7.0f ms  p95 %7.0f ms  max %7.0f ms" % (
                name, int(item.get("count", 0)), float(item.get("total_ms", 0.0)) / 1000.0,
                float(item.get("median_ms", 0.0)), float(item.get("p95_ms", 0.0)), float(item.get("max_ms", 0.0))))
    out("LLM: %s room-reasoning calls, %s classifier calls; A* plans %s; route commitment %s" % (
        policy.get("llm_queries"), policy.get("room_label_queries"), policy.get("plan_calls"),
        json.dumps(policy.get("route_commitment"))))
    verdicts, selections = {}, 0
    for events in new_events(rows).values():
        for event in events:
            if event["event"] == "release":
                verdicts[event.get("verdict")] = verdicts.get(event.get("verdict"), 0) + 1
            elif event["event"] == "transit":
                selections += 1
    out("selections %d; releases by verdict: %s" % (selections, json.dumps(verdicts, sort_keys=True)))
    loop = policy.get("room_search_loop") or {}
    stats = loop.get("stats") or {}
    keys = ("rooms_rebound", "room_missing_held", "relabels", "relabels_deferred", "entry_failures",
            "excluded_unreachable", "finished_in_transit", "explore_visits", "llm_reasonings")
    out("loop: " + ", ".join("%s=%s" % (k, stats.get(k)) for k in keys if k in stats))
    if loop.get("entry_failures"):
        out("places failed: %s" % json.dumps(loop["entry_failures"]))
    actions = (policy.get("exploration_metrics") or {}).get("actions_by_phase")
    if actions:
        out("actions by phase: %s" % json.dumps(actions, sort_keys=True))


def timeline(rows, out, start=None, end=None):
    events = new_events(rows)
    previous = None
    for row in rows:
        step = int(row["step"])
        current = counters(row)
        delta = "" if previous is None else counter_delta(previous, current)
        previous = current
        if (start is not None and step < start) or (end is not None and step > end):
            continue
        method = row.get("method") or {}
        search = method.get("search") or {}
        info = (row.get("command") or {}).get("info") or {}
        pose = row.get("pose") or {}
        transit = search.get("transit") or {}
        out("%4d %-10s %-16s %-10s room=%-4s next=%-4s order=%-18s rev=%-4s %-26s %-28s (%6.2f,%6.2f,%4.0f°) %s" % (
            step, str(method.get("state"))[:10], str(info.get("kind"))[:16], str(row.get("decision", {}).get("action"))[:10],
            method.get("room_id"), search.get("next_room"), str(search.get("order"))[:18],
            (method.get("reasoning") or {}).get("partition_revision"), delta,
            str(info.get("route"))[:28], float(pose.get("x", 0.0)), float(pose.get("y", 0.0)),
            math.degrees(float(pose.get("yaw", 0.0))),
            " | ".join(event_text(e) for e in events.get(step, ()))
            + (" | rebound->%s" % transit["room"] if transit.get("committed_room") not in (None, transit.get("room")) else "")))


def step_report(directory, rows, step, out, radius_m=6.0):
    row = next((r for r in rows if int(r["step"]) == int(step)), None)
    if row is None:
        raise SystemExit("no step %s in the recording (%d rows)" % (step, len(rows)))
    method = row.get("method") or {}
    search = method.get("search") or {}
    decision = row.get("decision") or {}
    command = row.get("command") or {}
    info = command.get("info") or {}
    pose = row.get("pose") or {}
    yaw = float(pose.get("yaw", 0.0))
    out("== step %d ==" % int(step))
    out("pose (%.2f, %.2f) yaw %.0f°  action %s  phase %s  state %s  owner %s" % (
        pose.get("x", 0.0), pose.get("y", 0.0), math.degrees(yaw), decision.get("action"), info.get("phase"),
        method.get("state"), (info.get("camera") or {}).get("owner")))
    dinfo = decision.get("info") or {}
    out("follower: status %s  heading error %.0f°  cross-track %.2f m  remaining path %.2f m  blocked %s" % (
        dinfo.get("status"), math.degrees(float(dinfo.get("heading_error_rad", 0.0) or 0.0)),
        float(dinfo.get("cross_track_m", 0.0) or 0.0), float(dinfo.get("remaining_path_m", 0.0) or 0.0),
        dinfo.get("forward_blocked")))
    out("route: %s  kind %s  waypoints %d%s" % (info.get("route"), info.get("kind"), len(command.get("waypoints") or ()),
                                                 "  replaced after %s" % info["route_replaced"] if info.get("route_replaced") else ""))
    out("room in force %s  next %s  order %s  visit %s  local steps %s/%s  supervisor %s  p_present %s" % (
        search.get("room_in_force"), search.get("next_room"), search.get("order"), search.get("visit"),
        search.get("local_steps"), search.get("local_budget"), search.get("supervisor_state"), _fmt(search.get("p_present"))))
    transit = search.get("transit")
    if transit:
        out("transit: room %s (committed as %s)  entry %s  from_frontier %s  room missing for %s action(s)" % (
            transit.get("room"), transit.get("committed_room"), transit.get("entry"), transit.get("from_frontier"),
            transit.get("room_missing_actions")))
    explore = search.get("explore")
    if explore:
        out("explore visit: room %s started %s goals walked %s frontier left %s" % (
            explore.get("room"), explore.get("started"), explore.get("goals"), explore.get("frontier_left")))
        for goal in explore.get("ranked", ())[:5]:
            out("   ranked goal %s utility %s size %s cells walk %s m" % (goal["xy"], goal["utility"], goal["size_cells"], goal["geodesic_m"]))
    events = new_events(rows).get(int(step), ())
    out("events this action: %s" % ("; ".join(event_text(e) for e in events) if events else "none"))
    excluded = search.get("excluded") or {}
    if excluded:
        out("excluded rooms: %s" % json.dumps(excluded, sort_keys=True))
    out("rooms:")
    for room in search.get("rooms", ()):
        out("   R%-3s %-14s p=%-5s frontier=%-3s searched=%-6s entry=%-10s dist=%-6s objects=%s%s" % (
            room.get("id"), str(room.get("label"))[:14], _fmt(room.get("prob")), room.get("frontier_clusters"),
            _fmt(room.get("searched_s")), room.get("entry"), _fmt(room.get("distance_m")),
            ",".join(sorted(set(room.get("objects") or ())))[:60], "  COOLING" if room.get("cooling") else ""))
    reading = search.get("reading") or {}
    if reading:
        out("LLM reading: home=%s | house=%s | stage=%s" % (reading.get("home"), reading.get("house"), reading.get("stage")))
    closing = method.get("target_closing") or {}
    if closing.get("active") or closing.get("phase") not in (None, "SEARCH"):
        out("target closing: phase %s label %s confidence %s xyz %s" % (closing.get("phase"), closing.get("label"),
                                                                         _fmt(closing.get("confidence")), closing.get("xyz")))
    detections = sorted(method.get("detections") or (), key=lambda d: -float(d.get("confidence", 0.0)))[:6]
    if detections:
        out("detections: " + ", ".join("%s %.2f" % (d["label"], d["confidence"]) for d in detections))
    glance = search.get("glance") or {}
    if glance.get("here"):
        out("glance gains here: %s  planned: %s  active: %s" % (json.dumps(glance.get("here")), _short(glance.get("planned")),
                                                               _short(glance.get("active"))))
    sector_report(directory, row, out, radius_m)


def _goal_rows(search):
    goals = search.get("frontier_goals")
    if goals is None:
        goals = [{"xy": xy} for xy in search.get("accessible_frontiers", ())]
    return goals


def sector_report(directory, row, out, radius_m):
    """The map round the agent at this action, by sector: what lay where, and what the inventory made of it."""
    maps = MapAtStep(directory)
    pose = row["pose"]
    x, y, yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
    floor_id = ((row.get("method") or {}).get("floor_id")) or 0
    grid, settled, labels = maps.at(int(row["step"]), floor_id)
    if not maps.exact:
        out("(no map_journal.npz: the sectors below read the FINAL map, not the map at this action)")
    gx, gy = maps.world_to_grid(x, y)
    r = int(math.ceil(radius_m / maps.resolution))
    h, w = grid.shape
    y0, y1, x0, x1 = max(0, gy - r), min(h, gy + r + 1), max(0, gx - r), min(w, gx + r + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = (xs - gx) * maps.resolution
    dy = (ys - gy) * maps.resolution
    within = dx * dx + dy * dy <= radius_m * radius_m
    relative = (np.degrees(np.arctan2(dy, dx) - yaw) + 45.0) % 360.0 - 45.0
    window, settled_w, labels_w = grid[y0:y1, x0:x1], settled[y0:y1, x0:x1], labels[y0:y1, x0:x1]
    unknown = window == -1
    free = window == 0
    live_unknown = unknown & ~settled_w
    adjacent = np.zeros_like(free)
    adjacent[:-1, :] |= live_unknown[1:, :]
    adjacent[1:, :] |= live_unknown[:-1, :]
    adjacent[:, :-1] |= live_unknown[:, 1:]
    adjacent[:, 1:] |= live_unknown[:, :-1]
    raw_frontier = free & adjacent
    search = (row.get("method") or {}).get("search") or {}
    goals = _goal_rows(search)
    openings = [o for o in search.get("openings", ()) if o.get("centroid")]
    area = maps.resolution ** 2
    out("map by sector within %.0f m (heading %.0f°):" % (radius_m, math.degrees(yaw)))
    for name, low, high in SECTORS:
        sector = within & (relative >= low) & (relative < high)
        rooms = sorted({int(v) - 1 for v in np.unique(labels_w[sector & free]) if v > 0})
        in_sector = lambda xy: bearing_sector(float(xy[0]) - x, float(xy[1]) - y, yaw) == name and math.dist(xy, (x, y)) <= radius_m
        sector_goals = [g for g in goals if in_sector(g["xy"])]
        sector_openings = [o for o in openings if in_sector(o["centroid"])]
        out("  %-6s free %6.1f m2  occupied %5.1f m2  unknown %6.1f m2 (settled %5.1f)  raw frontier cells %4d  "
            "frontier goals %d  openings %d  rooms %s" % (
                name, (sector & free).sum() * area, (sector & (window > 0)).sum() * area, (sector & unknown).sum() * area,
                (sector & unknown & settled_w).sum() * area, int((sector & raw_frontier).sum()), len(sector_goals),
                len(sector_openings), rooms))
        for goal in sector_goals:
            out("      goal %s %s" % (goal["xy"], ", ".join("%s=%s" % (k, v) for k, v in goal.items() if k not in ("xy", "cell"))))
        for opening in sector_openings:
            out("      opening O%d room %s at %s heading %s° gain %s m2 folded %s" % (
                int(opening["id"]) - 200000, opening.get("room_pid"), [round(float(v), 2) for v in opening["centroid"]],
                opening.get("heading_deg"), opening.get("gain_m2"), opening.get("folded")))


def ascii_map(directory, rows, step, out, radius_m=6.0, cell_m=0.2):
    row = next((r for r in rows if int(r["step"]) == int(step)), None)
    if row is None:
        raise SystemExit("no step %s in the recording" % step)
    maps = MapAtStep(directory)
    pose = row["pose"]
    x, y, yaw = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
    floor_id = ((row.get("method") or {}).get("floor_id")) or 0
    grid, settled, _ = maps.at(int(step), floor_id)
    if not maps.exact:
        out("(no map_journal.npz: this is the FINAL map, not the map at this action)")
    search = (row.get("method") or {}).get("search") or {}
    marks = {}
    for point in (row.get("command") or {}).get("waypoints") or ():
        marks[_bin(point, x, y, cell_m)] = "*"
    for goal in _goal_rows(search):
        marks[_bin(goal["xy"], x, y, cell_m)] = "F"
    for opening in search.get("openings", ()):
        if opening.get("centroid"):
            marks[_bin(opening["centroid"], x, y, cell_m)] = "O"
    n = int(math.ceil(radius_m / cell_m))
    per = max(1, int(round(cell_m / maps.resolution)))
    gx, gy = maps.world_to_grid(x, y)
    lines = []
    for j in range(n, -n - 1, -1):
        chars = []
        for i in range(-n, n + 1):
            cy0, cx0 = gy + j * per, gx + i * per
            patch = grid[max(0, cy0):max(0, cy0 + per), max(0, cx0):max(0, cx0 + per)]
            settled_patch = settled[max(0, cy0):max(0, cy0 + per), max(0, cx0):max(0, cx0 + per)]
            if (i, j) == (0, 0):
                char = "@"
            elif (i, j) in marks:
                char = marks[(i, j)]
            elif patch.size == 0:
                char = " "
            elif (patch > 0).any():
                char = "#"
            elif (patch == 0).any():
                char = "."
            elif settled_patch.any():
                char = ":"
            else:
                char = " "
            chars.append(char)
        lines.append("".join(chars))
    arrow = "^" if 45 <= math.degrees(yaw) % 360 < 135 else "<" if 135 <= math.degrees(yaw) % 360 < 225 else \
        "v" if 225 <= math.degrees(yaw) % 360 < 315 else ">"
    out("map at step %d round (%.2f, %.2f), %.0f m radius, %.1f m per character, heading %s (%.0f°); north up" % (
        int(step), x, y, radius_m, cell_m, arrow, math.degrees(yaw)))
    out("legend: # occupied  . free  (blank) unknown  : settled unknown  F frontier goal  O opening  * route  @ agent")
    for line in lines:
        out(line)


def _bin(point, x, y, cell_m):
    return (int(math.floor((float(point[0]) - x) / cell_m + 0.5)), int(math.floor((float(point[1]) - y) / cell_m + 0.5)))


def events_command(rows, out, name=None):
    for step, items in sorted(new_events(rows).items()):
        for event in items:
            if name is None or event["event"] == name:
                out("%4d %s" % (step, event_text(event)))


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "%.2f" % value
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", help="a recording directory holding steps.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary")
    p = sub.add_parser("timeline")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p = sub.add_parser("step")
    p.add_argument("step", type=int)
    p.add_argument("--radius", type=float, default=6.0)
    p = sub.add_parser("map")
    p.add_argument("step", type=int)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--cell", type=float, default=0.2)
    p = sub.add_parser("events")
    p.add_argument("--name")
    args = parser.parse_args(argv)
    rows = load_steps(args.directory)
    out = lambda line: print(line)
    if args.command == "summary":
        summary(args.directory, rows, out)
    elif args.command == "timeline":
        timeline(rows, out, args.start, args.end)
    elif args.command == "step":
        step_report(args.directory, rows, args.step, out, args.radius)
    elif args.command == "map":
        ascii_map(args.directory, rows, args.step, out, args.radius, args.cell)
    elif args.command == "events":
        events_command(rows, out, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
