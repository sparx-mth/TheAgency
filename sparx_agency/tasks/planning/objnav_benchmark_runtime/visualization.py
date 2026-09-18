"""Observed-only ObjectNav dashboard, with recorder-owned persistent floor panels."""
from __future__ import annotations

import json
import math
import textwrap
import cv2
import numpy as np

from sparx_agency.tasks.mapping.scene_graph.viz_canvas import compute_extent, world_to_px
from sparx_agency.tasks.mapping.scene_graph.viz_render import render_scene

FRAME_SIZE = (1600, 900)


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
    atlas = building.policy.mapping.atlas.diagnostics() if building else {}
    state = hierarchy.machine.phase if hierarchy is not None else getattr(supervisor, "state", "starting")
    if building and building.phase != "SEARCH":
        state = building.phase
    return {"state": state, "floor_id": getattr(getattr(policy, "mapping", None), "floor_id", 0),
            "floor_atlas": atlas, "transition": building.transition.diagnostics() if building and building.transition else None,
            "completion_reason": atlas.get("completion_reason"),
            "camera": dict(policy.camera_control.last) if hasattr(policy, "camera_control") else {},
            "perception": perception.diagnostics() if perception is not None else {},
            "room_id": hierarchy.regions.room_id if hierarchy is not None else getattr(supervisor, "room_id", None),
            "explorer": getattr(getattr(policy, "settings", None), "local_exploration", "unknown"),
            "burst_actions_left": max(0, hierarchy.params.burst_actions - burst.actions) if burst is not None else None,
            "planned_path": [[float(p.x), float(p.y)] if hasattr(p, "x") else list(p[:2]) for p in points],
            "detections": boxes, "detector_ms": getattr(detector, "last_inference_ms", None),
            "objects": [{"id": lm.id, "class": lm.class_name, "xy": list(lm.xy), "count": lm.count}
                        for lm in policy.landmarks.all_landmarks()] if hasattr(policy, "landmarks") else [],
            "reasoning": graph.last_reasoning if graph is not None else {},
            "doors": policy.doors.diagnostics() if hasattr(policy, "doors") else {},
            "rooms": len(graph.registry.rooms) if graph is not None else 0,
            "solver": str(solver.last.reason) if solver is not None else "not called"}


def _text(image, value, origin, width=115, lines=4, color=(230, 230, 230)):
    x, y = origin
    for index, line in enumerate(textwrap.wrap(str(value), width=width)[:lines]):
        cv2.putText(image, line, (x, y + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)


def render_dashboard(policy, observation, trail, decision, episode_id, snapshot, final=False, floor_panels=None):
    """RGB predictions, depth, persistent observed floors and actual decision reasons."""
    frame = np.full((900, 1600, 3), 20, np.uint8)
    action = "TERMINAL" if final else str(decision.get("action", "waiting"))
    _text(frame, "%s | %s | target: %s | step %d | %s | floor %s | z %.2fm | levels %d | links %d" %
          (episode_id, snapshot.get("explorer", "unknown"), observation.target_category,
           observation.step, action, snapshot.get("floor_id", 0), observation.pose.z,
           len(snapshot.get("floor_atlas", {}).get("floors", [0])),
           len(snapshot.get("floor_atlas", {}).get("connections", []))), (12, 26), width=190, lines=1)
    rgb = np.ascontiguousarray(observation.rgb[..., ::-1])
    for detection in snapshot.get("detections", ()) if not final else ():
        x1, y1, x2, y2 = (int(v) for v in detection["xyxy"])
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(rgb, "%s %.2f" % (detection["label"], detection["confidence"]),
                    (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
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
        panel = floor_panels.render(snapshot.get("floor_id", 0), snapshot.get("planned_path", ()))
    else:
        state = visual_state(policy, observation, trail)
        panel = render_scene(state, size=(960, 720), map_panel_w=640)
        extent = compute_extent(state, None, (640, 720))
        points = [world_to_px(extent, (640, 720), *xy) for xy in snapshot.get("planned_path", ())]
        if len(points) >= 2:
            cv2.polylines(panel, [np.asarray(points, np.int32)], False, (255, 180, 0), 2, cv2.LINE_AA)
    frame[40:760, 640:] = panel
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
