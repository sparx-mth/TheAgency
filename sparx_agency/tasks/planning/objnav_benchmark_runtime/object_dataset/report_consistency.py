"""STOP rules that need the same place confirmed from several approach steps (boxes projected with the views' depth), scored against the exact boxes."""
import argparse
import json
from pathlib import Path

import cv2

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.evaluate import merged_by_kind
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.projection import box_floor_point
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import (
    KINDS_OF_CATEGORY, classify_box, confirmed_by, consistent_stop_summary)

THRESHOLDS = (0.3, 0.5, 0.7)
STOP_RANGES_M = (2.5, 4.5)
RADII_M = (0.3, 0.5, 1.0)
NEEDED_STEPS = (1, 2, 3)
MAX_MEAN_RGB_DIFF = 6.0
CONFIRM_IOU = 0.5
WIDTH, HEIGHT, HFOV_DEG = 640, 480, 79.0


def camera(row):
    """The stored pose is public ENU (x = -habitat z, y = -habitat x); the box maths uses habitat x, z."""
    return (-row["y"], -row["x"], row["yaw"])


def sequences_of_boxes(folder, name, threshold, max_distance_m, confirmer=None):
    """One list per approach sequence: its goal-kind boxes in step order with their floor point and whether they lie on a goal object.

    With ``confirmer=(detector name, min conf)`` only boxes that detector also finds (IoU 0.5) are kept.
    """
    found = json.loads((folder / ("det_%s.json" % name)).read_text())
    others = json.loads((folder / ("det_%s.json" % confirmer[0])).read_text()) if confirmer else {}
    differences = json.loads((folder / "depth_check.json").read_text())
    grouped, skipped = {}, 0
    for line in (folder / "views.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["role"] != "object" or row["distance_m"] > max_distance_m:
            continue
        key = (row["scene"], row["target_instance"], row["sequence"])
        grouped.setdefault(key, [])
        if differences[row["frame"]] > MAX_MEAN_RGB_DIFF:
            skipped += 1
            continue
        depth = cv2.imread(str(folder / "depth" / row["frame"].replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED) / 1000.0
        for d in merged_by_kind(found[row["frame"]]):
            if DETECTOR_KIND[d["cls"]] not in KINDS_OF_CATEGORY[row["category"]] or d["conf"] < threshold:
                continue
            if confirmer and not confirmed_by(d, others[row["frame"]], confirmer[1], CONFIRM_IOU):
                continue
            point = box_floor_point(d["xyxy"], depth, camera(row), WIDTH, HEIGHT, HFOV_DEG)
            if point is not None:
                label = classify_box(row, d, row["category"])
                grouped[key].append({"step": row["step"], "point": point, "right": label in ("target", "other_goal")})
    return [sorted(boxes, key=lambda b: b["step"]) for boxes in grouped.values()], skipped


def report(views, name, held_out_fraction, include_held_out):
    folders = [d for d in sorted(views.iterdir()) if (d / ("det_%s.json" % name)).exists() and (d / "depth_check.json").exists()
               and (include_held_out or not is_held_out(d.name, held_out_fraction))]
    out = ["# Same place from several approach steps: `%s` on %d scenes\n" % (name, len(folders)),
           "A box counts when its depth puts it on the floor plan; the STOP is the first box that the needed number of earlier steps also put within the radius, "
           "judged right when the box lies on a goal object. Views whose re-rendered frame differs from the stored one are left out.\n"]
    for max_distance in STOP_RANGES_M:
        for threshold in THRESHOLDS:
            out += ["## Steps within %.1f m, confidence >= %.1f\n" % (max_distance, threshold),
                    "| steps that agree | radius m | sequences | stop on a goal object | stop on a wrong box | never stops |", "|---|---|---|---|---|---|"]
            per_folder = [sequences_of_boxes(f, name, threshold, max_distance) for f in folders]
            sequences = [s for found, _ in per_folder for s in found]
            for need in NEEDED_STEPS:
                for radius in ((RADII_M[0],) if need == 1 else RADII_M):
                    v = consistent_stop_summary(sequences, need, radius)
                    out.append("| %d | %s | %d | %d | %d | %d |" % (need, "-" if need == 1 else radius, v["sequences"], v["stopped_right"], v["stopped_wrong"], v["never"]))
            out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--include-held-out", action="store_true")
    args = p.parse_args(argv)
    text = report(args.views, args.name, args.held_out_fraction, args.include_held_out)
    (args.views / ("report_consistency_%s.md" % args.name)).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
