"""Paired HTTP detector evaluation on an immutable annotated development capture.

Reports point precision/recall at IoU >= .5, NOT COCO/LVIS AP. Calibrate only on
the declared calibration scenes, then measure on other training scenes. Raw
predictions and unchanged shared alias-suppressed predictions are both reported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from sparx_agency.core.common.math.bbox import iou
from sparx_agency.tasks.common.model_registry.download.verify import sha256_of
from sparx_agency.tasks.mapping.scene_graph.serve.contract import detections_from_json, detections_to_json
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import deduplicate_detections
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector

# Accepted RoboTHOR names, NOT fuzzy substring matches (e.g. cup != mug).
TARGET_ALIASES = {"tv": "television", "trash can": "garbage can", "potted plant": "house plant",
                  "clock": "alarm clock", "sports ball": "basketball"}


def assess(frames, predictions, targets, threshold, *, deduplicate=False):
    """Greedy score-ordered, one-to-one target matching with visible-mask GT."""
    if len(frames) != len(predictions):
        raise ValueError("Every annotated frame needs a prediction record, including negatives")
    counts, confusion = Counter(), Counter()
    for frame, items in zip(frames, predictions):
        gt = [a for a in frame["annotations"] if not a.get("ignore") and a["label"] in targets]
        ignored = [a for a in frame["annotations"] if a.get("ignore")]
        matched = set()
        dets = detections_from_json(items)
        if deduplicate:
            dets = deduplicate_detections(dets)
        for det in sorted(dets, key=lambda d: -d.conf):
            label = TARGET_ALIASES.get(det.cls, det.cls)
            if det.conf < threshold or label not in targets:
                continue
            overlaps = [(iou(det.xyxy, a["xyxy"]), i) for i, a in enumerate(gt) if a["label"] == label]
            candidates = [(v, i) for v, i in overlaps if i not in matched]
            best, index = max(candidates, default=(0, -1))
            if best >= .5:
                matched.add(index)
                counts["tp"] += 1
            elif any(a["label"] == label and iou(det.xyxy, a["xyxy"]) >= .5 for a in ignored):
                counts["ignored_predictions"] += 1
            else:
                counts["fp"] += 1
                counts["duplicates"] += int(any(v >= .5 for v, _ in overlaps))
                other = [(iou(det.xyxy, a["xyxy"]), a["label"]) for a in gt if a["label"] != label]
                overlap, actual = max(other, default=(0, ""))
                if overlap >= .5:
                    confusion[label + " -> " + actual] += 1
        counts["gt"] += len(gt)
        counts["fn"] += len(gt) - len(matched)
        for i, annotation in enumerate(gt):
            x1, y1, x2, y2 = annotation["xyxy"]
            if (x2-x1)*(y2-y1) < 32*32:
                counts["small_gt"] += 1
                counts["small_tp"] += int(i in matched)
    for key in ("tp", "fp", "fn", "gt", "duplicates", "small_gt", "small_tp"):
        counts.setdefault(key, 0)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    recall = counts["tp"] / counts["gt"] if counts["gt"] else None
    return dict(counts, precision=precision, recall=recall,
                small_recall=counts["small_tp"] / counts["small_gt"] if counts["small_gt"] else None,
                confusion=dict(confusion))


def choose_threshold(frames, predictions, targets):
    """Predeclared precision-weighted F0.5 grid, independently for each model."""
    grid = []
    for threshold in (round(float(v), 2) for v in np.arange(.1, .81, .05)):
        result = assess(frames, predictions, targets, threshold, deduplicate=True)
        p, r = result["precision"] or 0, result["recall"] or 0
        f = 1.25*p*r / (.25*p+r) if .25*p+r else 0
        grid.append(dict(threshold=round(float(threshold), 2), f05=f, **result))
    if not any(row["tp"] for row in grid):
        raise ValueError("No calibration true positives; cannot choose a meaningful operating point")
    best = max(grid, key=lambda row: (row["f05"], row["precision"] or 0, row["threshold"]))
    return best["threshold"], grid


def collect(manifest, root, url):
    """Use the actual ObjectNav HTTP client, including JPEG and identity checks."""
    client = HttpDetector(url, manifest["vocabulary"], timeout_s=180)
    records = []
    try:
        identity = client.health()
        for frame in manifest["frames"]:
            path = root / frame["file"]
            if sha256_of(path) != frame["sha256"]:
                raise ValueError("Capture changed: " + frame["file"])
            with np.load(path, allow_pickle=False) as data:
                rgb = data["rgb"]
            start = time.perf_counter()
            detections = client.detect(rgb)
            records.append({"file": frame["file"], "detections": detections_to_json(detections),
                            "client_ms": 1000*(time.perf_counter()-start),
                            "server_ms": client.last_inference_ms,
                            "peak_rss_mib": client.last_peak_rss_mib})
            print(len(records), frame["file"], round(records[-1]["client_ms"], 1), flush=True)
    finally:
        client.session.close()
    return {"identity": identity, "frames": records}


def summarize(manifest, run):
    frames = manifest["frames"]
    predictions = [r["detections"] for r in run["frames"]]
    if [r["file"] for r in run["frames"]] != [r["file"] for r in frames]:
        raise ValueError("Predictions do not match capture order")
    indices = lambda part: [i for i, f in enumerate(frames) if f["partition"] == part]
    cal, check = indices("calibration"), indices("measurement")
    cal_scenes, check_scenes = ({frames[i]["scene"] for i in group} for group in (cal, check))
    if not cal or not check or cal_scenes & check_scenes:
        raise ValueError("Calibration and measurement must have disjoint scenes")
    targets = manifest["targets"]
    threshold, grid = choose_threshold([frames[i] for i in cal], [predictions[i] for i in cal], targets)
    result = {"threshold": threshold, "calibration_grid": grid,
              "operating_point_objective": "max calibration F0.5 after unchanged shared deduplication",
              "accepted_target_aliases": TARGET_ALIASES}
    result["measurement_per_target"] = {
        target: assess([frames[i] for i in check], [predictions[i] for i in check],
                       [target], threshold, deduplicate=True) for target in targets}
    for point in sorted({.35, threshold}):
        for mode in ("raw", "shared_deduplicated"):
            result["measurement_%.2f_%s" % (point, mode)] = assess(
                [frames[i] for i in check], [predictions[i] for i in check], targets, point,
                deduplicate=mode != "raw")
    for key in ("client_ms", "server_ms"):
        values = [r[key] for r in run["frames"]]
        result[key] = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
    result["peak_rss_mib"] = max(r["peak_rss_mib"] or 0 for r in run["frames"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", help="Dedicated HTTP detector, already configured")
    parser.add_argument("--predictions", type=Path, help="Rescore saved predictions without inference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.url) == bool(args.predictions):
        parser.error("Select exactly one of --url or --predictions")
    if args.output.exists():
        parser.error("Use a fresh output path")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") not in ("train", "development", "synthetic"):
        parser.error("Refusing to calibrate on a validation/test split")
    run = (json.loads(args.predictions.read_text()) if args.predictions else
           collect(manifest, args.manifest.parent, args.url))
    digest = sha256_of(args.manifest)
    if args.predictions and run.get("manifest_sha256") != digest:
        raise ValueError("Cached predictions belong to a different capture manifest")
    if tuple(run["identity"]["classes"]) != tuple(manifest["vocabulary"]):
        raise ValueError("Prediction vocabulary does not match the evaluation vocabulary")
    run["manifest_sha256"] = digest
    # Save expensive inference even if calibration is impossible (e.g. no TP).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run, indent=2) + "\n")
    run["summary"] = summarize(manifest, run)
    args.output.write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run["summary"], indent=2))


if __name__ == "__main__":
    main()
