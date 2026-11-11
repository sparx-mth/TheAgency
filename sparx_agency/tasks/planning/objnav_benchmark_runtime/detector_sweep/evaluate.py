"""Precision and recall of each detector, per kind of object, by confidence threshold, against the sweep's exact boxes."""
import argparse
import collections
import json
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND

THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MATCH_IOU = 0.3
SAME_KIND_IOU = 0.6
MIN_GT_PIXELS = 1000


def iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merged_by_kind(detections):
    """The detections of the kinds, one per object: a box of the same kind inside a better one (IoU over SAME_KIND_IOU) is dropped."""
    kept = []
    for d in sorted((d for d in detections if d["cls"] in DETECTOR_KIND), key=lambda d: -d["conf"]):
        kind = DETECTOR_KIND[d["cls"]]
        if not any(DETECTOR_KIND[k["cls"]] == kind and iou(d["xyxy"], k["xyxy"]) >= SAME_KIND_IOU for k in kept):
            kept.append(d)
    return kept


def match_frame(detections, boxes, threshold):
    """``{kind: [true positives, false positives, matched objects]}`` for one frame at one threshold."""
    result = collections.defaultdict(lambda: [0, 0, 0])
    taken = set()
    for d in (d for d in merged_by_kind(detections) if d["conf"] >= threshold):
        kind = DETECTOR_KIND[d["cls"]]
        best = max(((iou(d["xyxy"], b["xyxy"]), i) for i, b in enumerate(boxes) if b["kind"] == kind and i not in taken and b["pixels"] >= MIN_GT_PIXELS),
                   default=(0.0, None))
        if best[0] >= MATCH_IOU:
            taken.add(best[1])
            result[kind][0] += 1
            result[kind][2] += 1
        else:
            result[kind][1] += 1
    return result


def objects_per_kind(sweep_rows):
    count = collections.Counter()
    for row in sweep_rows:
        count.update(b["kind"] for b in row["boxes"] if b["pixels"] >= MIN_GT_PIXELS)
    return count


def curves(sweep, name):
    """``{kind: {threshold: (precision, recall, tp, fp)}}`` and ``{kind: objects}`` over every scene of the sweep."""
    totals = {t: collections.defaultdict(lambda: [0, 0, 0]) for t in THRESHOLDS}
    objects = collections.Counter()
    for folder in sorted(d for d in sweep.iterdir() if (d / ("det_%s.json" % name)).exists()):
        rows, found = json.loads((folder / "gt.json").read_text()), json.loads((folder / ("det_%s.json" % name)).read_text())
        objects.update(objects_per_kind(rows))
        for row in rows:
            for t in THRESHOLDS:
                for kind, (tp, fp, hit) in match_frame(found[row["frame"]], row["boxes"], t).items():
                    cell = totals[t][kind]
                    cell[0] += tp
                    cell[1] += fp
                    cell[2] += hit
    table = collections.defaultdict(dict)
    for t, by_kind in totals.items():
        for kind in objects:
            tp, fp, hit = by_kind.get(kind, [0, 0, 0])
            table[kind][t] = (tp / (tp + fp) if tp + fp else None, hit / objects[kind], tp, fp)
    return table, objects


def suggested_threshold(row, target_precision):
    """The lowest threshold whose precision reaches the target (it keeps the most recall), or None."""
    for t in THRESHOLDS:
        precision = row[t][0]
        if precision is not None and precision >= target_precision:
            return t
    return None


def report(sweep, names, target_precision):
    lines = []
    for name in names:
        table, objects = curves(sweep, name)
        lines.append("\n== %s   (objects in view, >= %d px: %s)" % (name, MIN_GT_PIXELS, dict(objects.most_common())))
        for kind, _ in objects.most_common():
            cells = " | ".join("%.2f: P %s R %2d%%" % (t, "%3d%%" % (100 * p) if p is not None else "  - ", 100 * r)
                               for t, (p, r, _, _) in table[kind].items() if t in (0.05, 0.3, 0.5, 0.7, 0.9))
            lines.append("  %-12s n=%4d  %s  -> threshold for precision >= %.0f%%: %s" % (
                kind, objects[kind], cells, 100 * target_precision, suggested_threshold(table[kind], target_precision)))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True, help="the --name of each run_detectors run")
    p.add_argument("--target-precision", type=float, default=0.8)
    args = p.parse_args(argv)
    print(report(args.sweep, args.names, args.target_precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
