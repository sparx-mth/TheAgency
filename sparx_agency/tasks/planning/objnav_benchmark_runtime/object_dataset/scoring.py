"""Score a detector's boxes on the object views against the exact boxes: hits on the target, wrong boxes, votes over views."""
import collections

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.evaluate import iou, merged_by_kind
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND

KIND_OF_CATEGORY = {"chair": "chair", "bed": "bed", "sofa": "sofa", "tv_monitor": "tv", "toilet": "toilet", "plant": "plant"}
MATCH_IOU = 0.3
HEADINGS_PER_STEP = 3


def score_view(row, detections, threshold, category=None):
    """What the boxes of the goal kind at ``threshold`` did in one view: on the target, on another goal object, or wrong."""
    kind = KIND_OF_CATEGORY[category or row["category"]]
    ground_truth = [b for b in row["boxes"] if b["kind"] == kind]
    score = collections.Counter(detections=0, hit_target=0, on_other_goal=0, on_non_goal=0, on_nothing=0)
    target = next((b for b in row["boxes"] if b["instance"] == row.get("target_instance")), None)
    for d in merged_by_kind(detections):
        if DETECTOR_KIND[d["cls"]] != kind or d["conf"] < threshold:
            continue
        score["detections"] += 1
        best = max(((iou(d["xyxy"], b["xyxy"]), b) for b in ground_truth), key=lambda pair: pair[0], default=(0.0, None))
        if best[0] < MATCH_IOU:
            score["on_nothing"] += 1
        elif best[1]["instance"] == row.get("target_instance"):
            score["hit_target"] += 1
        elif best[1]["is_goal"]:
            score["on_other_goal"] += 1
        else:
            score["on_non_goal"] += 1
    score["wrong"] = score["on_non_goal"] + score["on_nothing"]
    score["hit_goal"] = bool(score["hit_target"] or score["on_other_goal"])
    score["hit_target"] = bool(score["hit_target"])
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
