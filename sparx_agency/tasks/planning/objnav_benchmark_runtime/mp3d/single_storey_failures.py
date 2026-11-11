"""What each failed episode of a single-storey campaign did, read from its recording, and a first mechanism tag.

For every row of a campaign root (``<root>/<scene>/episodes.jsonl``) that failed, the script reads the episode's
recording (``steps.jsonl``, ``trajectory.csv``) and the published episode (``<episodes-dir>/content/<scene>.json.gz``)
and reports:

* the outcome: STOP or step limit, the geodesic DTG the scorer measured, the STOP's reason;
* the final pose against the goals: the planar distance to the nearest published view point and to the nearest
  goal object, and the height of the closing's memory over the agent's floor;
* every lock the closing took: the action it started on, the first box (width x height in pixels), the memory's
  range and height band, how it ended (STOP / release reason) and after how many actions;
* a first mechanism tag (``MECHANISMS``), from those facts alone -- a reading to be checked against the recording,
  not a verdict; the ground-truth sensor's phantom ids (``GT_ARTEFACTS``) are counted apart from the method's;
* for a near-miss STOP, the standing spot against the goal's ``.house`` box (``--scene-dir``): the generator's own
  distance test, how far beside the object's width the agent stands and how far off its normal.

Also tallies the room labeller over EVERY episode of the root (successes too): the labels adopted, the object
sets behind each ``dining_room`` verdict, and the classifier calls made on rooms with no objects.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_failures runs/ss_baseline \\
        [--episodes-dir ~/datasets/objectnav/mp3d/v1/val] [--scene-dir ~/datasets/scene_datasets/mp3d] \\
        [--json out.json] [--locks locks.json] [--all]

Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_EPISODES_DIR = Path.home() / "datasets" / "objectnav" / "mp3d" / "v1" / "val"
DEFAULT_SCENE_DIR = Path.home() / "datasets" / "scene_datasets" / "mp3d"

#: The first-pass tags, in the order they are tried; the first that fits is the episode's.
#: ``gt_phantom_*``: the STOP's lock had its 3-D memory more than PHANTOM_M from every published goal object of
#: the class -- pixels the semantic sensor labelled with the class where no such object stands (a sliver through a
#: gap, an annotation leaking through a wall); ``_sliver`` when the first sighting's visible surface was under
#: SLIVER_HEIGHT_M tall. ``stop_other_level``: the memory stands more than OTHER_LEVEL_M below or above the agent's
#: base (seen through a stairwell or over a railing). ``stop_near_miss``: a STOP within a metre of a view point.
MECHANISMS = ("gt_phantom_sliver", "gt_phantom_id", "stop_near_miss", "stop_other_level",
              "stop_seen_through_barrier", "stop_other",
              "limit_never_seen", "limit_wedged", "limit_locks_released", "limit_other")
#: The tags that are the ground-truth sensor's doing, not the method's (the user's decision of 2026-11-10: GT
#: artefacts are outside the failure analysis; the on-object filter removes them from later runs).
GT_ARTEFACTS = ("gt_phantom_sliver", "gt_phantom_id")
PHANTOM_M = 1.5
SLIVER_HEIGHT_M = 0.10
#: An object whose top is this far below the agent's base, or whose lowest point is BELOW_M above it, is on another level.
BELOW_M = 0.8
ABOVE_M = 1.9
#: The MP3D camera's focal length in pixels (640 px wide at HFOV 79 degrees), for a box's width in metres.
FX_PX = 320.0 / math.tan(math.radians(79.0) / 2.0)
#: A near-miss STOP's standing spot against the goal's box (``standing_spot``): ``beside`` when the agent stands
#: more than BESIDE_M beyond the object's width along its mounting and BESIDE_DEG or more off its normal -- where a
#: thin wall-mounted object has no view point (its view points reach 0.26 m beyond its width at the median; see
#: ``perception/face_normal.py``); ``far`` when the agent stands farther from the box than the goal's farthest view
#: point; ``in_band`` otherwise -- within the band and in front, yet off every view point (furniture under the
#: object, an occluded strip, a geodesic detour).
BESIDE_M = 0.2
BESIDE_DEG = 45.0


def enu(position) -> tuple:
    """Habitat's Y-up ``[x, y, z]`` as the recording's ENU ``(x, y, z)``: ``(-z, -x, y)``."""
    return (-float(position[2]), -float(position[0]), float(position[1]))


def load_goals(episodes_dir: Path, scene: str) -> dict:
    with gzip.open(episodes_dir / "content" / ("%s.json.gz" % scene)) as fh:
        return json.load(fh)


def published_episode(data: dict, index: int) -> dict:
    episode = data["episodes"][index]
    key = "%s_%s" % (Path(episode["scene_id"]).name, episode["object_category"])
    goals = data["goals_by_category"][key]
    return {"category": episode["object_category"],
            "start": enu(episode["start_position"]),
            "goals": [enu(g["position"]) for g in goals],
            "goal_ids": [g["object_id"] for g in goals],
            "goal_view_points": [[enu(v["agent_state"]["position"]) for v in g["view_points"]] for g in goals],
            "view_points": [enu(v["agent_state"]["position"]) for g in goals for v in g["view_points"]]}


def house_boxes(scene_dir: Path, scene: str) -> Dict[int, tuple]:
    """Every object's oriented box from the scene's ``.house`` file, in the recording's ENU frame.

    An ``O`` line holds the object index, its centre, two axes and three radii in the house frame (z up); habitat
    reads that frame as ``(x, z, -y)`` and the recording's ENU is ``(-hab_z, -hab_x, hab_y)``, so a house vector
    ``(x, y, z)`` is the ENU ``(y, -x, z)``. Returns ``{object_index: (centre, axes, radii)}`` with ``axes`` the
    three unit rows (the third the cross product of the two given) -- the generator's own geometry, which keeps a
    candidate view point only within 1 m of this box at the object's height.
    """
    path = scene_dir / scene / ("%s.house" % scene)
    out: Dict[int, tuple] = {}
    if not path.is_file():
        return out
    to_enu = lambda v: (float(v[1]), -float(v[0]), float(v[2]))
    for line in open(path):
        if not line.startswith("O "):
            continue
        f = line.split()
        centre = to_enu(list(map(float, f[4:7])))
        a0, a1 = to_enu(list(map(float, f[7:10]))), to_enu(list(map(float, f[10:13])))
        a2 = (a0[1] * a1[2] - a0[2] * a1[1], a0[2] * a1[0] - a0[0] * a1[2], a0[0] * a1[1] - a0[1] * a1[0])
        out[int(f[1])] = (centre, (a0, a1, a2), tuple(map(float, f[13:16])))
    return out


def box_distance(point, box) -> float:
    """The distance from ``point`` to the oriented ``box`` (0 inside), as the generator's ``OBB.distance``."""
    centre, axes, radii = box
    excess = 0.0
    for axis, radius in zip(axes, radii):
        local = sum(a * (p - c) for a, p, c in zip(axis, point, centre))
        excess += max(abs(local) - radius, 0.0) ** 2
    return math.sqrt(excess)


def standing_spot(pose, published: dict, boxes: Dict[int, tuple]) -> Optional[dict]:
    """The final pose against the box of the goal whose view point is nearest: the generator's distance test, and
    where the agent stands relative to the object's mounting (beside it, in front of it, too far)."""
    best = None
    for goal_id, points in zip(published["goal_ids"], published["goal_view_points"]):
        for v in points:
            d = math.dist(pose[:3], v)                   # in 3-D: the same object's view points a storey up are not it
            if best is None or d < best[0]:
                best = (d, goal_id, points)
    if best is None or best[1] not in boxes:
        return None
    centre, axes, radii = boxes[best[1]]
    at_height = (pose[0], pose[1], centre[2])
    box_m = box_distance(at_height, boxes[best[1]])
    farthest_vp = max(box_distance((v[0], v[1], centre[2]), boxes[best[1]]) for v in best[2])
    horizontal = sorted(range(3), key=lambda i: abs(axes[i][2]))[:2]      # the two axes nearest the floor plane
    thin, wide = sorted(horizontal, key=lambda i: radii[i])
    along = sum(a * (p - c) for a, p, c in zip(axes[wide], pose, centre))
    out = sum(a * (p - c) for a, p, c in zip(axes[thin], pose, centre))
    beside = max(0.0, abs(along) - radii[wide])
    graze = math.degrees(math.atan2(abs(along), abs(out))) if (along or out) else 0.0
    vertical = max(range(3), key=lambda i: abs(axes[i][2]))
    bottom_above_base = (centre[2] - radii[vertical]) - pose[2]
    kind = ("beside" if beside > BESIDE_M and graze >= BESIDE_DEG else
            "far" if box_m > farthest_vp + 0.05 else "in_band")
    return {"goal_id": best[1], "box_m": round(box_m, 2), "farthest_view_point_m": round(farthest_vp, 2),
            "beside_m": round(beside, 2), "graze_deg": round(graze), "box_bottom_above_base_m": round(bottom_above_base, 2),
            "thin_half_m": round(radii[thin], 2), "wide_half_m": round(radii[wide], 2), "kind": kind}


def recordings_by_episode(scene_dir: Path) -> Dict[str, Path]:
    """``{episode_id: recording directory}`` of a scene: its own ``recordings/`` and, for a scene merged from
    shards (``campaign.py --shard-episodes``), the recordings its ``shard_<k>/`` directories keep."""
    out = {}
    patterns = [str(scene_dir / "recordings" / "*" / "episode.json"),
                str(scene_dir / "shard_*" / "recordings" / "*" / "episode.json")]
    for meta in sorted(path for pattern in patterns for path in glob.glob(pattern)):
        try:
            out.setdefault(json.load(open(meta))["episode_id"], Path(meta).parent)
        except (ValueError, KeyError):
            continue
    return out


def final_pose(recording: Path) -> Optional[tuple]:
    rows = list(csv.DictReader(open(recording / "trajectory.csv")))
    if not rows:
        return None
    last = rows[-1]
    return (float(last["x"]), float(last["y"]), float(last["z"]), float(last["yaw"]))


def planar(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def locks(recording: Path) -> List[dict]:
    """Every closing lock in the step log: start, first box, memory, end."""
    out: List[dict] = []
    current = None
    for line in open(recording / "steps.jsonl"):
        step = json.loads(line)
        closing = step["method"].get("target_closing") or {}
        policy = (step["decision"].get("info") or {}).get("policy") or {}
        started = closing.get("started_step")
        if closing.get("active") and started is not None and started >= 0:
            if current is None or current["start"] != started:
                box = closing.get("bbox")
                size = None if not box else (round(box[2] - box[0]), round(box[3] - box[1]))
                heights = (closing.get("lowest_m"), closing.get("top_m"))
                current = {"start": int(started), "label": closing.get("label"), "first_box_wh": size,
                           "first_range_m": None, "xyz": closing.get("xyz"), "lowest_m": closing.get("lowest_m"),
                           "top_m": closing.get("top_m"), "end": None, "end_step": None, "phases": Counter(),
                           "first_height_extent_m": None if None in heights else round(float(heights[1]) - float(heights[0]), 3),
                           "first_xyz": closing.get("xyz"), "agent_z": float(step["pose"].get("z", 0.0))}
                out.append(current)
            current["phases"][str(closing.get("phase"))] += 1
            if closing.get("xyz") is not None:
                current["xyz"] = closing["xyz"]
                current["lowest_m"], current["top_m"] = closing.get("lowest_m"), closing.get("top_m")
            approach = closing.get("approach") or {}
            if current["first_range_m"] is None and approach.get("start_range_m") is not None:
                current["first_range_m"] = round(float(approach["start_range_m"]), 2)
            if step["decision"].get("action") == "STOP":
                current["end"] = "STOP: %s" % policy.get("reason")
                current["end_step"] = int(step["step"])
        if policy.get("kind") == "target_released" and current is not None and current["end"] is None:
            current["end"] = "released: %s" % policy.get("reason")
            current["end_step"] = int(step["step"])
            current = None
    for lock in out:
        lock["phases"] = dict(lock["phases"])
    return out


def mechanism(row: dict, facts: dict) -> str:
    stop = bool(row.get("stop_called"))
    dtg = float(row["distance_to_goal_m"])
    last = facts["locks"][-1] if facts["locks"] else None
    if stop:
        if facts.get("memory_to_goal_m") is not None and facts["memory_to_goal_m"] > PHANTOM_M:
            extent = None if last is None else last.get("first_height_extent_m")
            return "gt_phantom_sliver" if extent is not None and extent < SLIVER_HEIGHT_M else "gt_phantom_id"
        if dtg <= 1.0:
            return "stop_near_miss"
        top, lowest = facts.get("memory_top_above_base_m"), facts.get("memory_lowest_above_base_m")
        if (top is not None and top < -BELOW_M) or (lowest is not None and lowest > ABOVE_M):
            return "stop_other_level"
        if facts.get("agent_to_view_point_m") is not None and facts["agent_to_view_point_m"] < 0.5 * dtg:
            return "stop_seen_through_barrier"
        return "stop_other"
    closing = ((row.get("agent_info") or {}).get("policy") or {}).get("target_closing") or {}
    if int(closing.get("last_seen_step", -1)) < 0 and not facts["locks"]:
        return "limit_never_seen"
    blocked = int((row.get("agent_info") or {}).get("blocked_forward", 0) or 0)
    if int(closing.get("stall_releases", 0)) >= 1 or blocked >= 40:
        return "limit_wedged"
    if facts["locks"]:
        return "limit_locks_released"
    return "limit_other"


def analyse_episode(row: dict, recording: Optional[Path], published: dict, boxes: Optional[Dict[int, tuple]] = None) -> dict:
    facts = {"episode_id": row["episode_id"], "category": row["target_category"], "success": bool(row["success"]),
             "steps": int(row["steps"]), "stop": bool(row.get("stop_called")), "dtg_m": round(float(row["distance_to_goal_m"]), 2),
             "spl": round(float(row["spl"]), 3), "locks": [], "recording": None if recording is None else str(recording)}
    policy = (row.get("agent_info") or {}).get("policy") or {}
    closing = policy.get("target_closing") or {}
    facts["closing_counters"] = {k: v for k, v in closing.items()
                                 if k in ("releases", "stall_releases", "border_rejections", "same_spot_rejections",
                                          "inspection_releases", "blind_inspection_releases", "map_releases",
                                          "boxed_releases", "bearing_starts", "bearing_abandoned", "history_stops",
                                          "support_stops", "closest_approach_stops", "blind_cone_stops",
                                          "last_seen_step", "occluded_path_steps")}
    facts["blocked_forward"] = (row.get("agent_info") or {}).get("blocked_forward")
    facts["llm"] = {"oracle": policy.get("llm_queries"), "labels": policy.get("room_label_queries"),
                    "answers": policy.get("llm_answers")}
    if recording is not None and (recording / "steps.jsonl").is_file():
        facts["locks"] = locks(recording)
        pose = final_pose(recording)
        if pose is not None:
            facts["final_pose"] = [round(v, 2) for v in pose]
            if published["view_points"]:
                facts["agent_to_view_point_m"] = round(min(planar(pose, v) for v in published["view_points"]), 2)
            if published["goals"]:
                facts["agent_to_goal_m"] = round(min(planar(pose, g) for g in published["goals"]), 2)
            if boxes and facts["stop"]:
                facts["standing_spot"] = standing_spot(pose, published, boxes)
            last = facts["locks"][-1] if facts["locks"] else None
            if last and last.get("xyz"):
                xyz = last["xyz"]
                facts["memory_to_goal_m"] = round(min(planar(xyz, g) for g in published["goals"]), 2) if published["goals"] else None
                facts["memory_height_m"] = None if last.get("lowest_m") is None else round(float(last["lowest_m"]), 2)
                facts["memory_elevation_m"] = round(float(xyz[2]) - pose[2], 2)
                facts["memory_lowest_above_base_m"] = None if last.get("lowest_m") is None else round(float(last["lowest_m"]) - pose[2], 2)
                facts["memory_top_above_base_m"] = None if last.get("top_m") is None else round(float(last["top_m"]) - pose[2], 2)
                facts["memory_range_m"] = round(planar(pose, xyz), 2)
        facts["release_reasons"] = dict(Counter(
            (lock["end"] or "unfinished").split(":", 1)[-1].strip().split(";")[0][:60] for lock in facts["locks"]))
    facts["mechanism"] = mechanism(row, facts)
    return facts


def lock_table(rows: List[dict], recordings: Dict[str, Path], episodes_dir: Path) -> List[dict]:
    """Every closing lock of every episode: its first box, its memory against the goals, how it ended.

    The data behind a minimum-size rule for the ground-truth detector: a lock whose memory lies within a metre
    of a published goal object was a lock on the target; one far from every goal was a lock on pixels the
    semantic sensor labelled with the class where no instance stands (a sliver through a gap, a misaligned
    annotation).
    """
    goals_cache: Dict[str, dict] = {}
    out = []
    for row in rows:
        recording = recordings.get(row["episode_id"])
        if recording is None or not (recording / "steps.jsonl").is_file():
            continue
        scene, index = row["episode_id"].split("/")
        if scene not in goals_cache:
            goals_cache[scene] = load_goals(episodes_dir, scene)
        published = published_episode(goals_cache[scene], int(index))
        for lock in locks(recording):
            xyz = lock.get("xyz")
            to_goal = None if not xyz or not published["goals"] else round(min(planar(xyz, g) for g in published["goals"]), 2)
            box = lock["first_box_wh"]
            out.append({"episode_id": row["episode_id"], "category": row["target_category"], "success": bool(row["success"]),
                        "start": lock["start"], "box_w": None if not box else box[0], "box_h": None if not box else box[1],
                        "area_px": None if not box else box[0] * box[1], "first_range_m": lock.get("first_range_m"),
                        "memory_to_goal_m": to_goal, "on_target": None if to_goal is None else to_goal <= 1.0,
                        "end": lock["end"], "lowest_m": lock.get("lowest_m"), "top_m": lock.get("top_m"),
                        "first_height_extent_m": lock.get("first_height_extent_m"),
                        "first_width_m": None if not box or not lock.get("first_range_m") else
                        round(box[0] * float(lock["first_range_m"]) / FX_PX, 3),
                        "first_elevation_m": None if lock.get("lowest_m") is None else round(float(lock["lowest_m"]) - lock["agent_z"], 2)})
    return out


def labeller_tally(rows: List[dict], recordings: Dict[str, Path]) -> dict:
    """The room labeller over every episode: adopted labels, the evidence behind dining_room, zero-object calls."""
    adopted: Counter = Counter()
    dining_evidence: Counter = Counter()
    dining_rooms_per_episode: List[int] = []
    shape_only_calls = 0
    shape_only_labels: Counter = Counter()
    calls = 0
    for row in rows:
        policy = (row.get("agent_info") or {}).get("policy") or {}
        calls += int(policy.get("room_label_queries") or 0)
        recording = recordings.get(row["episode_id"])
        if recording is None or not (recording / "steps.jsonl").is_file():
            continue
        history_seen = set()
        last_labels: Dict[str, dict] = {}
        for line in open(recording / "steps.jsonl"):
            step = json.loads(line)
            reasoning = step["method"].get("reasoning") or {}
            for entry in reasoning.get("label_history") or ():
                key = (entry.get("step"), entry.get("room_id"), entry.get("label"), entry.get("trigger"))
                if key in history_seen or entry.get("held"):
                    continue
                history_seen.add(key)
                adopted[entry.get("label")] += 1
                if entry.get("label") == "dining_room":
                    dining_evidence[tuple(sorted((entry.get("objects") or {}).keys()))] += 1
                if entry.get("trigger") == "shape":
                    shape_only_calls += 1
                    shape_only_labels[entry.get("label")] += 1
            for pid, meta in (reasoning.get("labels") or {}).items():
                last_labels[pid] = meta
        dining_rooms_per_episode.append(sum(1 for m in last_labels.values() if m.get("label") == "dining_room"))
    return {"classifier_calls": calls, "labels_adopted": dict(adopted.most_common()),
            "dining_room_evidence": {" + ".join(k) or "(none)": v for k, v in dining_evidence.most_common(15)},
            "episodes_ending_with_2_or_more_dining_rooms": sum(1 for n in dining_rooms_per_episode if n >= 2),
            "shape_only_adoptions": shape_only_calls, "shape_only_labels": dict(shape_only_labels.most_common())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR,
                        help="the MP3D scenes (<scene>/<scene>.house), for the standing spot against the goal's box")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--all", action="store_true", help="analyse the successes too")
    parser.add_argument("--locks", type=Path, help="write every lock of every episode (the sliver data) as JSON here")
    args = parser.parse_args(argv)
    rows: List[dict] = []
    recordings: Dict[str, Path] = {}
    for path in sorted(glob.glob(str(args.root / "*" / "episodes.jsonl"))):
        scene_dir = Path(path).parent
        recordings.update(recordings_by_episode(scene_dir))
        for line in open(path):
            if line.strip():
                rows.append(json.loads(line))
    goals_cache: Dict[str, dict] = {}
    boxes_cache: Dict[str, Dict[int, tuple]] = {}
    results = []
    for row in rows:
        if row["success"] and not args.all:
            continue
        scene, index = row["episode_id"].split("/")
        if scene not in goals_cache:
            goals_cache[scene] = load_goals(args.episodes_dir, scene)
            boxes_cache[scene] = house_boxes(args.scene_dir, scene)
        published = published_episode(goals_cache[scene], int(index))
        results.append(analyse_episode(row, recordings.get(row["episode_id"]), published, boxes_cache[scene]))
    tally = Counter(r["mechanism"] for r in results if not r["success"])
    spots = Counter((r.get("standing_spot") or {}).get("kind") for r in results
                    if not r["success"] and r["mechanism"] == "stop_near_miss" and r.get("standing_spot"))
    failures = sum(1 for r in rows if not r["success"])
    artefacts = sum(tally.get(name, 0) for name in GT_ARTEFACTS)
    print("%d episodes, %d failures: %d the method's, %d the ground-truth sensor's (phantom ids, outside the analysis)"
          % (len(rows), failures, failures - artefacts, artefacts))
    for name in MECHANISMS:
        if tally.get(name):
            print("  %-28s %d%s" % (name, tally[name], "   (GT artefact)" if name in GT_ARTEFACTS else ""))
            if name == "stop_near_miss" and spots:
                print("  %-28s %s" % ("", ", ".join("%s %d" % kv for kv in spots.most_common())))
    print()
    for r in sorted(results, key=lambda r: (r["mechanism"] in GT_ARTEFACTS, r["episode_id"])):
        if r["success"]:
            continue
        last = r["locks"][-1] if r["locks"] else None
        print("%-20s %-16s %-26s %s DTG %5.2f  steps %3d  locks %d  agent->vp %s  mem->goal %s  mem dz %s  first box %s H %s%s"
              % (r["episode_id"], r["category"], r["mechanism"], "STOP " if r["stop"] else "LIMIT", r["dtg_m"], r["steps"],
                 len(r["locks"]), r.get("agent_to_view_point_m"), r.get("memory_to_goal_m"), r.get("memory_elevation_m"),
                 None if last is None else last["first_box_wh"], None if last is None else last.get("first_height_extent_m"),
                 "" if last is None else "\n%s      last lock @%d: %s" % (" " * 20, last["start"], last["end"])))
        spot = r.get("standing_spot")
        if spot and r["mechanism"] == "stop_near_miss":
            print("%s      standing spot: %s -- %.2f m from the box (its view points reach %.2f), %.2f m beside its width, "
                  "%d deg off its normal, box bottom %.2f m above the base"
                  % (" " * 20, spot["kind"], spot["box_m"], spot["farthest_view_point_m"], spot["beside_m"], spot["graze_deg"],
                     spot["box_bottom_above_base_m"]))
    print()
    print("labeller:", json.dumps(labeller_tally(rows, recordings), indent=1))
    if args.locks:
        args.locks.write_text(json.dumps(lock_table(rows, recordings, args.episodes_dir), indent=1))
    if args.json:
        args.json.write_text(json.dumps({"episodes": results, "mechanisms": dict(tally),
                                         "labeller": labeller_tally(rows, recordings)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
