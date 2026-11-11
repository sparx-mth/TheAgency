"""Run a COCO-trained ultralytics YOLO (e.g. yolov8x.pt) in this process over the frames of a sweep (``det_<name>.json`` per scene).

Standalone on purpose: it needs only ultralytics and OpenCV, so it runs in the detector environment. COCO names that are
benchmark words (chair, couch, bed, toilet, tv, sink, oven, refrigerator, potted plant, dining table) are kept, the rest dropped.
"""
import argparse
import json
from pathlib import Path

import cv2

KEPT = {"chair", "couch", "bed", "toilet", "tv", "sink", "oven", "refrigerator", "potted plant", "dining table"}


def detect_scene(model, folder, name, conf):
    rows = json.loads((folder / "gt.json").read_text())
    found = {}
    for row in rows:
        result = model.predict(cv2.imread(str(folder / "frames" / row["frame"])), conf=conf, imgsz=640, device="cpu", verbose=False)[0]
        found[row["frame"]] = [{"cls": model.names[int(c)], "conf": round(float(s), 4), "xyxy": [round(float(v), 1) for v in box]}
                               for box, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())
                               if model.names[int(c)] in KEPT]
    (folder / ("det_%s.json" % name)).write_text(json.dumps(found))
    return len(rows)


def main(argv=None):
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--name", default="yolov8x_coco")
    p.add_argument("--conf", type=float, default=0.05)
    args = p.parse_args(argv)
    model = YOLO(str(args.weights))
    for folder in sorted(d for d in args.sweep.iterdir() if (d / "gt.json").exists()):
        if (folder / ("det_%s.json" % args.name)).exists():
            continue
        print("%s: %d frames" % (folder.name, detect_scene(model, folder, args.name, args.conf)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
