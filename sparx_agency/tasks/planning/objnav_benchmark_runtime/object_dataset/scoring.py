"""Score a detector's boxes on the object views against the exact boxes: hits on the target, wrong boxes, votes over views."""
import collections
import math

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.evaluate import iou, merged_by_kind
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND

KIND_OF_CATEGORY = {"chair": "chair", "bed": "bed", "sofa": "sofa", "tv_monitor": "tv", "toilet": "toilet", "plant": "plant"}
MATCH_IOU = 0.3
HEADINGS_PER_STEP = 3


def classify_box(row, detection, category):
    """What one box of the goal kind covers: ``target``, ``other_goal``, ``non_goal`` (a kind object outside the goal set) or ``nothing``."""
    kind = KIND_OF_CATEGORY[category]
    best = max(((iou(detection["xyxy"], b["xyxy"]), b) for b in row["boxes"] if b["kind"] == kind), key=lambda pair: pair[0], default=(0.0, None))
    if best[0] < MATCH_IOU:
        return "nothing"
    if best[1]["instance"] == row.get("target_instance"):
        return "target"
    return "other_goal" if best[1]["is_goal"] else "non_goal"


def score_view(row, detections, threshold, category=None):
    """What the boxes of the goal kind at ``threshold`` did in one view: on the target, on another goal object, or wrong."""
    kind = KIND_OF_CATEGORY[category or row["category"]]
    score = collections.Counter(detections=0, on_target=0, on_other_goal=0, on_non_goal=0, on_nothing=0)
    target = next((b for b in row["boxes"] if b["instance"] == row.get("target_instance")), None)
    for d in merged_by_kind(detections):
        if DETECTOR_KIND[d["cls"]] != kind or d["conf"] < threshold:
            continue
        score["detections"] += 1
        score["on_" + classify_box(row, d, category or row["category"])] += 1
    score["wrong"] = score["on_non_goal"] + score["on_nothing"]
    score["hit_goal"] = bool(score["on_target"] or score["on_other_goal"])
    score["hit_target"] = bool(score.pop("on_target"))
    score["target_pixels"] = target["pixels"] if target else 0
    score["target_truncated"] = bool(target and target["truncated"])
    return dict(score)


def accepted_by_vote(flags, need):
    return sum(bool(f) for f in flags) >= need


def summarise(scored_rows, key):
    """Per ``key``: views, views with the target visible, recall over those, wrong boxes per view."""
    groups = collections.defaultdict(list)
    for r in scored_rows:
        groups[r[key]].append(r)
    table = {}
    for name, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        visible = [r for r in rows if r["target_visible"]]
        table[name] = {"views": len(rows), "visible": len(visible),
                       "recall": sum(r["hit_target"] for r in visible) / len(visible) if visible else None,
                       "recall_goal": sum(r["hit_goal"] for r in visible) / len(visible) if visible else None,
                       "wrong_per_view": sum(r["wrong"] for r in rows) / len(rows)}
    return table


def vote_summary(scored_rows, need):
    """Over the full groups of the headings of one approach step: groups voting a hit, groups voting a wrong box, and the single-view rates."""
    groups = collections.defaultdict(list)
    for r in scored_rows:
        groups[(r["scene"], r["target_instance"], r["sequence"], r["step"])].append(r)
    full = [g for g in groups.values() if len(g) == HEADINGS_PER_STEP]
    views = [r for g in full for r in g]
    return {"groups": len(full),
            "hit": sum(accepted_by_vote([r["hit_target"] for r in g], need) for g in full),
            "wrong": sum(accepted_by_vote([r["wrong"] > 0 for r in g], need) for g in full),
            "views": len(views), "single_hit": sum(bool(r["hit_target"]) for r in views),
            "single_wrong": sum(r["wrong"] > 0 for r in views)}


def vlm_summary(answers):
    """Per category and overall: right boxes (on a goal object) the vision model kept, wrong boxes it rejected, and the unsure ones."""
    cells = collections.defaultdict(collections.Counter)
    for a in answers:
        right = a["label"] in ("target", "other_goal")
        for name in (a["category"], "all"):
            cell = cells[name]
            cell["right" if right else "wrong"] += 1
            cell["kept" if right and a["answer"] != "no" else "rejected" if not right and a["answer"] == "no" else "other"] += 1
            if a["answer"] == "unsure":
                cell["unsure_right" if right else "unsure_wrong"] += 1
    keys = ("right", "kept", "wrong", "rejected", "unsure_wrong", "unsure_right")
    return {name: {k: cell[k] for k in keys} for name, cell in sorted(cells.items())}


def sequence_vote_summary(scored_rows, need, max_distance_m):
    """Per approach sequence: a STOP once ``need`` consecutive steps within ``max_distance_m`` each have a box, right if the last step's box is on a goal object."""
    sequences = collections.defaultdict(dict)
    for r in scored_rows:
        if r["bucket"] <= max_distance_m:
            sequences[(r["scene"], r["target_instance"], r["sequence"])].setdefault(r["step"], r)
    result = collections.Counter(sequences=len(sequences))
    for steps in sequences.values():
        streak = 0
        for _, r in sorted(steps.items()):
            streak = streak + 1 if r["detections"] else 0
            if streak >= need:
                result["stopped_right" if r["hit_goal"] else "stopped_wrong"] += 1
                break
        else:
            result["never"] += 1
    return {k: result[k] for k in ("sequences", "stopped_right", "stopped_wrong", "never")}


def answers_within(answers, min_conf, max_distance_m):
    return [a for a in answers if a["conf"] >= min_conf and a["distance_m"] <= max_distance_m]


def consistent_stop_summary(sequences, need, radius_m):
    """Per approach sequence of boxes (``step``, floor ``point``, ``right``): a STOP at the first box that ``need - 1`` earlier steps also put within ``radius_m``."""
    result = collections.Counter(sequences=len(sequences))
    for boxes in sequences:
        for index, current in enumerate(boxes):
            supporting = {b["step"] for b in boxes[:index] if b["step"] < current["step"] and math.dist(b["point"], current["point"]) <= radius_m}
            if len(supporting) >= need - 1:
                result["stopped_right" if current["right"] else "stopped_wrong"] += 1
                break
        else:
            result["never"] += 1
    return {k: result[k] for k in ("sequences", "stopped_right", "stopped_wrong", "never")}
