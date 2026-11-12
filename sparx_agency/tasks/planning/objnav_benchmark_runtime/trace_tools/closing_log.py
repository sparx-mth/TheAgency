"""One row per step of what the target closing sees and decides, for the verification diagnosis. Read-only."""
import math

COUNTERS = ("border_rejections", "context_rejections", "glimpse_steps", "verify_pauses", "weak_resightings", "releases",
            "memory_sight_refusals", "inspection_steps", "inspection_releases", "support_holds", "history_holds")


CANDIDATE_ROOM_REACH_M = 0.6


def room_of(policy, xy):
    """The room the policy's own context check would read at ``xy``: its id, label and strength, or None."""
    graph, world = getattr(policy, "graph", None), getattr(policy, "last_world", None)
    if xy is None or graph is None or world is None or getattr(graph, "labels", None) is None:
        return None
    pid = graph.room_near(world, (float(xy[0]), float(xy[1])), CANDIDATE_ROOM_REACH_M)
    info = None if pid is None else graph.label_info(pid)
    return None if not info else {"room": int(pid), "label": info.get("label"), "strength": info.get("strength")}


def target_boxes(policy, observation):
    closing, intrinsics = policy.closing, observation.camera.intrinsics
    return [{"cls": d.cls, "conf": round(float(d.conf), 3), "clipped": bool(closing._clipped(d.xyxy, intrinsics))}
            for d in policy.perception.detections
            if policy.target.accepts(d.cls) and d.conf >= closing.settings.confidence]


def closing_row(policy, observation):
    closing = policy.closing
    here = (observation.pose.x, observation.pose.y)
    distance = None if closing.xyz is None else round(math.dist(here, closing.xyz[:2]), 2)
    candidate_xy = None if closing.xyz is None else [round(float(closing.xyz[0]), 2), round(float(closing.xyz[1]), 2)]
    return {"step": int(observation.step), "phase": closing.phase, "active": closing.active, "locked": closing.locked,
            "count": closing.count, "suspect": closing.suspect, "best_conf": round(float(closing.best_conf), 3),
            "candidate_distance_m": distance, "candidate_xy": candidate_xy, "candidate_room": room_of(policy, closing.xyz),
            "agent_room": room_of(policy, here), "last_release": closing.last_release,
            "counters": {name: getattr(closing, name) for name in COUNTERS}, "target_boxes": target_boxes(policy, observation)}


def worth_logging(row):
    return bool(row["target_boxes"]) or row["phase"] != "SEARCH"
