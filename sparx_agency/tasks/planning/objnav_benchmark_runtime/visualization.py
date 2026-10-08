"""Observed-only ObjectNav dashboard, with recorder-owned persistent floor panels."""
from __future__ import annotations

import json
import math
import textwrap
import cv2
import numpy as np

from sparx_agency.tasks.mapping.scene_graph.viz_canvas import compute_extent, world_to_px
from sparx_agency.tasks.mapping.scene_graph.viz_render import render_scene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.search_panel import render_search_panel
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.discovery import SUSPENDED_PHASES

FRAME_SIZE = (1600, 900)
#: The right-hand side of the frame: floor maps, then the search column.
MAP_PANEL = (640, 720)
SEARCH_PANEL = (320, 720)


def search_snapshot(policy):
    """What the room-search loop believes and intends, JSON-ready, for the panel and the trace.

    Every node the loop can go to -- the rooms of the floor with the oracle's
    probability, type, frontier, time searched and objects, and the
    staircases it offered with their probability, direction and the storey
    beyond -- plus the RPT* order, the node in force or in transit, the
    objects by room, the oracle's ``elsewhere`` and the loop's last events.
    Built from policy state only, so a frame and its ``steps.jsonl`` row
    cannot disagree.
    """
    loop, graph, supervisor = (getattr(policy, name, None) for name in ("loop", "graph", "supervisor"))
    if loop is None or graph is None:
        return {}
    world = getattr(policy, "last_world", None)
    now = float(getattr(policy, "_floor_time", 0.0))
    cooling = {pid for pid in graph.registry.rooms if supervisor is not None and supervisor.is_cooling(pid, now)}
    reasons = graph.last_reasoning.get("oracle", {}).get("reasons", {})
    rooms = []
    for pid, room in graph.registry.rooms.items():
        estimate = dict(loop.estimates.get(pid, {}))
        info = graph.label_info(pid) or {}
        fact = graph.facts.get(pid)
        rooms.append({"id": pid, "kind": "room", "label": estimate.get("label") or info.get("label", "unknown"),
                      "strength": estimate.get("strength", info.get("strength")),
                      "prob": estimate.get("prob", graph.probs.get(pid)),
                      "why": estimate.get("why") or reasons.get(pid, reasons.get(str(pid), "")),
                      "frontier_clusters": None if fact is None else fact.frontier_clusters,
                      "searched_s": None if fact is None else fact.time_in_room_s,
                      "last_inside_step": graph.last_inside.get(pid),
                      "objects": graph.objects_in(pid), "distance_m": estimate.get("distance_m"),
                      "entry": estimate.get("entry"), "cooling": pid in cooling,
                      "centroid": [float(v) for v in room.centroid]})
    rooms.sort(key=lambda r: -(r["prob"] or 0.0))
    stairs = []
    for nid, option in sorted(loop._stairs.items()):
        estimate = dict(loop.estimates.get(nid, {}))
        stairs.append({"id": nid, "kind": "stairs", "label": option.node.label, "direction": option.node.direction,
                       "prob": estimate.get("prob", graph.stair_probs.get(nid)),
                       "why": estimate.get("why") or reasons.get(nid, reasons.get(str(nid), "")),
                       "destination_visited": option.node.destination_visited, "destination": option.node.destination,
                       "arrived_by": option.node.arrived_by, "leaf_m": round(option.leaf_m, 2),
                       "approach_m": round(option.approach.distance_m, 2), "distance_m": estimate.get("distance_m"),
                       "portal_id": option.portal["id"], "centroid": [float(v) for v in option.approach.xy]})
    openings = []
    for nid, option in sorted(getattr(loop, "_openings", {}).items()):
        estimate = dict(loop.estimates.get(nid, {}))
        o = option.opening
        landmark = o.kind == "landmark"
        openings.append({"id": nid, "kind": "landmark" if landmark else "opening", "label": option.node.label,
                         "via": option.node.via, "room_pid": o.room_pid, "door_id": o.door_id, "glimpsed": list(o.glimpsed),
                         "landmark_id": o.landmark_id,
                         "landmark_xy": None if o.landmark_xy is None else [float(v) for v in o.landmark_xy],
                         "size_cells": int(o.size_cells), "heading_deg": round(math.degrees(o.heading), 1),
                         "prob": estimate.get("prob", loop.settings.openings.landmark_prob if landmark
                                              else graph.stair_probs.get(nid)),
                         "why": estimate.get("why") or reasons.get(nid, reasons.get(str(nid), "")),
                         "distance_m": estimate.get("distance_m"), "centroid": [float(v) for v in o.xy]})
    objects = [{"id": lm.id, "class": lm.class_name, "count": lm.count, "xy": list(lm.xy),
                "room": graph.object_room(world, lm) if world is not None else None}
               for lm in policy.landmarks.confirmed()] if hasattr(policy, "landmarks") else []
    building = getattr(policy, "building", None)
    floor = {}
    if building is not None and building.ground_truth is not None:
        floor = {"phase": building.phase, "spent": int(round(now / max(1e-6, policy.settings.action_time_s))),
                 "arrived_by": building.arrived_by, "active_portal": building.active["id"] if building.active else None,
                 "selected_by": building.active.get("selected_by") if building.active else None}
    return {"target": policy.target.query, "p_present": graph.p_present, "elsewhere": max(0.0, 1.0 - graph.p_present),
            "probability_model": "independent_search_success",
            "accessible_frontiers": [list(goal.xy) for goal in graph.frontier_inventory.goals]
            if graph.frontier_inventory is not None else [],
            "warmup": {"actions": getattr(policy, "warmup_actions", 0), "budget": policy.settings.warmup_steps},
            "doorway_peek": policy.peek.diagnostics() if hasattr(policy, "peek") else {},
            "reading": dict(graph.last_reasoning.get("oracle", {}).get("reading", {})),
            "floor_id": policy.mapping.floor_id,
            "supervisor_state": supervisor.state if supervisor is not None else "?",
            "room_in_force": loop.room_id, "local_steps": loop.local_steps, "local_budget": loop.visit_budget(),
            "visit": loop.settings.visit, "scan": None if loop._scan is None else dict(loop._scan),
            "peek": loop.peek_state() if hasattr(loop, "peek_state") else None,
            "glance": policy.glances.state() if hasattr(policy, "glances") else None,
            "excluded": {str(pid): why for pid, why in sorted(loop._excluded.items())},
            "order": list(loop.order), "order_index": loop.order_index, "next_room": loop.next_room,
            "rooms": rooms, "stairs": stairs, "openings": openings, "objects": objects,
            "floor": floor, "events": list(loop.events[-6:])}


def visual_state(policy, observation, trail):
    """Adapt method-owned rooms/maps into the existing scene-graph drawing schema."""
    pose = observation.pose
    graph = getattr(policy, "graph", None)
    landmarks = getattr(policy, "landmarks", None)
    world = getattr(policy, "last_world", None)
    mapping = getattr(policy, "mapping", None)
    if mapping is not None:
        world = getattr(mapping, "worlds", {}).get(getattr(mapping, "floor_id", 0), world)
    state = {"pose": (pose.x, pose.y, pose.yaw), "trail": list(trail), "sim_time": float(observation.step),
             "oracle": {"target": observation.target_category, "source": "waiting", "rooms": []}}
    if graph is None or world is None:
        return state
    objects = [{"id": lm.id, "class": lm.class_name, "xy": lm.xy, "count": lm.count} for lm in landmarks.all_landmarks()]
    rooms, grid_values = [], np.zeros(world.grid.shape, np.uint8)
    pid_map = {}
    for value, (pid, room) in enumerate(graph.registry.rooms.items(), 1):
        if value > 127:
            break
        grid_values[room.mask] = value
        pid_map[str(value)] = pid
        members = []
        for obj in objects:
            gx, gy = world.world_to_grid(*obj["xy"])
            if world.in_bounds(gx, gy) and room.mask[gy, gx]:
                members.append(obj)
        fact = graph.facts.get(pid)
        rooms.append({"id": pid, "centroid": room.centroid, "objects": members,
                      "time_in_room_s": fact.time_in_room_s if fact else 0,
                      "frontier_clusters": fact.frontier_clusters if fact else 0})
    known_y, known_x = np.where(world.grid >= 0)
    gx, gy = world.world_to_grid(pose.x, pose.y)
    x0, x1 = max(0, min(int(known_x.min()) if known_x.size else gx, gx) - 5), min(world.width, max(int(known_x.max()) if known_x.size else gx, gx) + 6)
    y0, y1 = max(0, min(int(known_y.min()) if known_y.size else gy, gy) - 5), min(world.height, max(int(known_y.max()) if known_y.size else gy, gy) + 6)
    origin = (world.origin_x + x0 * world.resolution, world.origin_y + y0 * world.resolution)
    state.update(
        bev={"grid": world.grid[y0:y1, x0:x1], "resolution": world.resolution, "origin": origin},
        room_grid={"grid": grid_values[y0:y1, x0:x1], "resolution": world.resolution, "origin": origin},
        scene_graph={"rooms": rooms, "doors": graph.doors, "grid_pid_map": pid_map},
        room_labels={pid: dict(item, label=item["label"] + (" (provisional)" if item.get("provisional") else ""))
                     for pid, item in graph.last_reasoning.get("labels", {}).items()}, objects={"objects": objects},
        oracle={"target": observation.target_category,
                "source": graph.last_reasoning.get("oracle", {}).get("source", "waiting"),
                "rooms": [{"id": r.room_id, "label": r.label, "prob": r.prob} for r in graph.options]},
        target_seen=getattr(policy, "_target_xy", None) is not None,
        target_info={"target": observation.target_category, "xy": getattr(policy, "_target_xy", None)})
    return state


def method_snapshot(policy):
    """JSON-ready policy observations, excluding arrays and private goal data."""
    graph = getattr(policy, "graph", None)
    solver = getattr(policy, "solver", None)
    route = getattr(policy, "_route", None)
    points = [] if route is None else getattr(route, "points", route)
    detector = getattr(policy, "detector", None)
    perception = getattr(policy, "perception", None)
    raw = perception.raw if perception is not None else getattr(detector, "last_detections", ())
    boxes = [{"label": d.cls, "confidence": float(d.conf), "xyxy": list(d.xyxy)} for d in raw]
    hierarchy = getattr(policy, "hierarchy", None)
    supervisor = getattr(policy, "supervisor", None)
    burst = hierarchy.machine.burst if hierarchy is not None else None
    building = getattr(policy, "building", None)
    # The perfect stair detector's boxes ride beside the object detector's: a staircase is
    # "detected" only in the frames it is actually in, and the overlay shows exactly those.
    stair_boxes = list(getattr(building, "last_sightings", ()) or ()) if building is not None else []
    atlas = building.policy.mapping.atlas.diagnostics() if building else {}
    state = hierarchy.machine.phase if hierarchy is not None else getattr(supervisor, "state", "starting")
    if getattr(policy, "_action_owner", None) in SUSPENDED_PHASES:
        state = policy._action_owner
    if building and building.phase != "SEARCH":
        state = building.phase
    closing = getattr(policy, "closing", None)
    if closing is not None and closing.active:
        state = "TARGET_" + closing.phase
    return {"state": state, "floor_id": getattr(getattr(policy, "mapping", None), "floor_id", 0),
            "target_closing": closing.diagnostics() if closing is not None else {},
            "control_counters": {"room_llm": getattr(graph, "queries", 0),
                                 "room_classifier": getattr(getattr(graph, "label_tracker", None), "queries", 0),
                                 "rpt": getattr(solver, "calls", 0),
                                 "astar": getattr(policy, "_plan_calls", 0)},
            "floor_atlas": atlas, "transition": building.transition.diagnostics() if building and building.transition else None,
            "completion_reason": atlas.get("completion_reason"),
            "camera": dict(policy.camera_control.last) if hasattr(policy, "camera_control") else {},
            "perception": perception.diagnostics() if perception is not None else {},
            "room_id": hierarchy.regions.room_id if hierarchy is not None else getattr(supervisor, "room_id", None),
            "explorer": getattr(getattr(policy, "settings", None), "local_exploration", "unknown"),
            "burst_actions_left": max(0, hierarchy.params.burst_actions - burst.actions) if burst is not None else None,
            "planned_path": [[float(p.x), float(p.y)] if hasattr(p, "x") else list(p[:2]) for p in points],
            "detections": boxes + stair_boxes, "detector_ms": getattr(detector, "last_inference_ms", None),
            "stairs_seen": building.sightings.diagnostics().get("seen", []) if building is not None and hasattr(building, "sightings") else [],
            "objects": [{"id": lm.id, "class": lm.class_name, "xy": list(lm.xy), "count": lm.count}
                        for lm in policy.landmarks.all_landmarks()] if hasattr(policy, "landmarks") else [],
            "reasoning": graph.last_reasoning if graph is not None else {},
            "doors": policy.doors.diagnostics() if hasattr(policy, "doors") else {},
            "rooms": len(graph.registry.rooms) if graph is not None else 0,
            "solver": str(solver.last.reason) if solver is not None else "not called",
            "fallback": policy.fallback.snapshot() if hasattr(policy, "fallback") and hasattr(policy.fallback, "snapshot") else {},
            "frontier_gaps": ({"step": policy.frontier_gaps.step, "gaps": list(policy.frontier_gaps.last),
                               "stats": dict(policy.frontier_gaps.stats)}
                              if hasattr(policy, "frontier_gaps") else {}),
            "search": search_snapshot(policy)}


def _text(image, value, origin, width=115, lines=4, color=(230, 230, 230)):
    x, y = origin
    for index, line in enumerate(textwrap.wrap(str(value), width=width)[:lines]):
        cv2.putText(image, line, (x, y + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)


def render_dashboard(policy, observation, trail, decision, episode_id, snapshot, final=False, floor_panels=None):
    """RGB predictions, depth, persistent observed floors, the search column and actual decision reasons.

    Layout, left to right: the camera and depth; the floor maps (rooms and
    stair nodes named and valued, the RPT* order drawn through them, the
    next node ringed); the search column -- target, the oracle's split
    between the nodes and elsewhere, the visit order, every room's
    probability and reason, the staircases with the climb's cost, the
    objects, the loop's last events.
    """
    frame = np.full((900, 1600, 3), 20, np.uint8)
    action = "TERMINAL" if final else str(decision.get("action", "waiting"))
    search = snapshot.get("search") or {}
    _text(frame, "%s | %s | target: %s | step %d | %s | floor %s | z %.2fm | levels %d | links %d" %
          (episode_id, snapshot.get("explorer", "unknown"), observation.target_category,
           observation.step, action, snapshot.get("floor_id", 0), observation.pose.z,
           len(snapshot.get("floor_atlas", {}).get("floors", [0])),
           len(snapshot.get("floor_atlas", {}).get("connections", []))), (12, 26), width=190, lines=1)
    rgb = np.ascontiguousarray(observation.rgb[..., ::-1])
    for detection in snapshot.get("detections", ()) if not final else ():
        x1, y1, x2, y2 = (int(v) for v in detection["xyxy"])
        ground_truth = detection.get("source") == "ground_truth"
        color = (60, 140, 255) if ground_truth else (0, 220, 255)        # orange for the perfect stair detector
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        text = ("%s GT %.1fm" % (detection["label"], detection.get("distance_m", 0.0)) if ground_truth
                else "%s %.2f" % (detection["label"], detection["confidence"]))
        cv2.putText(rgb, text, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    frame[40:520, :640] = cv2.resize(rgb, (640, 480))
    depth = observation.depth_m
    finite = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros(depth.shape, np.uint8)
    scaled[finite] = np.clip(depth[finite] / observation.camera.max_depth_m * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~finite] = 0
    frame[545:785, :320] = cv2.resize(colored, (320, 240))
    _text(frame, "Metric depth; black = invalid/clipped", (8, 539), width=76, lines=1)
    _text(frame, "STATE: %s ROOM: %s RAW BOXES: %d OBJECTS: %d ROOMS: %d DOORS: %d" %
          (snapshot["state"], snapshot["room_id"], len(snapshot["detections"]), len(snapshot["objects"]),
           snapshot.get("rooms", 0), snapshot.get("doors", {}).get("confirmed", 0)), (332, 563), width=37, lines=6)
    if floor_panels is not None:
        panel = floor_panels.render(snapshot.get("floor_id", 0), snapshot.get("planned_path", ()), size=MAP_PANEL,
                                    search=search)
    else:
        state = visual_state(policy, observation, trail)
        panel = render_scene(state, size=MAP_PANEL, map_panel_w=MAP_PANEL[0])
        extent = compute_extent(state, None, MAP_PANEL)
        for goal in search.get("accessible_frontiers", ()):
            cv2.drawMarker(panel, world_to_px(extent, MAP_PANEL, *goal), (255, 120, 0), cv2.MARKER_DIAMOND, 7, 1)
        points = [world_to_px(extent, MAP_PANEL, *xy) for xy in snapshot.get("planned_path", ())]
        if len(points) >= 2:
            cv2.polylines(panel, [np.asarray(points, np.int32)], False, (255, 180, 0), 2, cv2.LINE_AA)
    frame[40:40 + MAP_PANEL[1], 640:640 + MAP_PANEL[0]] = panel
    frame[40:40 + SEARCH_PANEL[1], 640 + MAP_PANEL[0]:] = render_search_panel(search, SEARCH_PANEL)
    transition = snapshot.get("transition") or {}
    camera = snapshot.get("camera", {})
    _text(frame, "pitch %.0f deg down | %s | connector %s | %s" % (
        math.degrees(observation.pose.camera_pitch), camera.get("owner", "unknown"), transition.get("portal_id", "-"),
        snapshot.get("completion_reason") or "transition not completed"), (654, 783), width=120, lines=1)
    reason = decision.get("info", {}).get("policy", {})
    _text(frame, "DECISION: " + json.dumps(reason, ensure_ascii=True), (12, 814), width=175, lines=2)
    oracle = snapshot.get("reasoning", {}).get("oracle", {})
    _text(frame, "LLM ROOM REASONS: " + json.dumps(oracle.get("reasons", {}), ensure_ascii=True),
          (12, 858), width=175, lines=2)
    return frame
