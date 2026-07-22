"""Run the open-set YOLO-World TRT split over a folder of RGB frames.

Given the two engines (backbone DLA + head GPU), a ``.pt`` for the CLIP text
branch, and a comma-separated prompt list, this loads :class:`YoloTRTDetector`,
sets the prompts once, and detects them in every image in ``--images``. For each
frame it writes an annotated ``.jpg`` and appends the detections to a CSV and a
JSONL; a ``summary.json`` records the run (prompts, thresholds, per-class counts).

The confidence threshold (``--conf``) is a **runtime** NMS parameter applied in
:func:`postprocess.decode` -- it is independent of the TensorRT engines, so raising
it to reduce false positives needs no rebuild. Prompts are open-vocabulary and set
at run time; nothing here is baked into an engine.

Run (target Orin, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.detect_folder \\
        --backbone .../orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \\
        --head     .../orin_sm87/yolo_world_s.head.fp16.gpu.engine \\
        --text-weights /path/to/yolov8s-worldv2.pt \\
        --images /path/to/rgb --out /path/to/identifications \\
        --labels "chair, bottle, table" --conf 0.4
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector

_IMG_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
# A small fixed palette (BGR) so each class draws in a stable, distinct colour.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
]


def parse_labels(value: str) -> List[str]:
    """Split a comma-separated prompt string into cleaned, ordered class names.

    Multi-word prompts (e.g. ``"computer screen"``) keep their spaces; only commas
    separate classes. The order is preserved -- it maps to the head's class rows.
    """
    labels = [p.strip() for p in value.split(",")]
    labels = [p for p in labels if p]
    if not labels:
        raise argparse.ArgumentTypeError("--labels needs at least one class name")
    return labels


def find_images(images_dir: str) -> List[str]:
    """Sorted list of image files in ``images_dir`` (non-recursive)."""
    files = sorted(p for p in glob.glob(os.path.join(images_dir, "*"))
                   if p.rsplit(".", 1)[-1].lower() in _IMG_EXTS)
    if not files:
        raise RuntimeError("No images (%s) found in: %s"
                           % ("/".join(_IMG_EXTS), images_dir))
    return files


def draw(bgr: np.ndarray, dets, labels: List[str]) -> np.ndarray:
    """Return a copy of ``bgr`` with each detection's box + ``label score`` drawn."""
    out = bgr.copy()
    index = {name: i for i, name in enumerate(labels)}
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        color = _PALETTE[index.get(d.label, 0) % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = "%s %.2f" % (d.label, d.score)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, tag, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def run(args) -> None:
    """Detect ``args.labels`` in every frame of ``args.images`` and persist results."""
    labels = args.labels
    files = find_images(args.images)
    out_dir = Path(args.out)
    vis_dir = out_dir / "annotated"
    vis_dir.mkdir(parents=True, exist_ok=True)

    det = YoloTRTDetector(args.backbone, args.head, text_weights=args.text_weights,
                          text_device=args.text_device, conf_thresh=args.conf,
                          iou_thresh=args.iou, max_det=args.max_det)
    det.set_prompts(labels)
    print("[detect] %d frames | %d prompts=%s | conf=%.2f iou=%.2f max_det=%d"
          % (len(files), len(labels), labels, det.conf_thresh, det.iou_thresh,
             det.max_det))

    per_class = {name: 0 for name in labels}
    n_dets = 0
    csv_path = out_dir / "detections.csv"
    jsonl_path = out_dir / "detections.jsonl"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf, \
            open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.writer(cf)
        writer.writerow(["image", "label", "score", "x1", "y1", "x2", "y2"])
        for path in files:
            name = os.path.basename(path)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                print("  skipped unreadable image:", name)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = det.detect(rgb)
            n_dets += len(dets)
            records = []
            for d in dets:
                x1, y1, x2, y2 = d.bbox_xyxy
                writer.writerow([name, d.label, "%.4f" % d.score, x1, y1, x2, y2])
                records.append({"label": d.label, "score": round(float(d.score), 4),
                                "bbox_xyxy": [x1, y1, x2, y2]})
                if d.label in per_class:
                    per_class[d.label] += 1
            jf.write(json.dumps({"image": name, "detections": records}) + "\n")
            if not args.no_vis:
                cv2.imwrite(str(vis_dir / (Path(name).stem + ".jpg")),
                            draw(bgr, dets, labels))

    summary = {
        "images": len(files), "prompts": labels, "total_detections": n_dets,
        "conf_thresh": det.conf_thresh, "iou_thresh": det.iou_thresh,
        "max_det": det.max_det, "per_class_counts": per_class,
        "backbone": Path(args.backbone).name, "head": Path(args.head).name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] %d detections over %d frames -> %s" % (n_dets, len(files), out_dir))
    print("       per-class:", per_class)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, help="backbone .engine path")
    ap.add_argument("--head", required=True, help="head .engine path")
    ap.add_argument("--text-weights", required=True,
                    help=".pt YOLO-World checkpoint driving the CLIP text branch")
    ap.add_argument("--images", required=True, help="folder of RGB frames")
    ap.add_argument("--out", required=True, help="output folder for identifications")
    ap.add_argument("--labels", required=True, type=parse_labels,
                    help="comma-separated open-vocab class prompts")
    ap.add_argument("--conf", type=float, default=0.40,
                    help="min class confidence (runtime NMS; higher = fewer false "
                         "positives). Default 0.40 (build default is 0.25).")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU (default: manifest)")
    ap.add_argument("--max-det", type=int, default=None,
                    help="max detections per frame (default: manifest)")
    ap.add_argument("--text-device", default="cpu", help="torch device for text encode")
    ap.add_argument("--no-vis", action="store_true", help="skip annotated images")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
