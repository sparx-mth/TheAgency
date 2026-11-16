"""Run a detector server over the frames of the object views and store every detection (``det_<name>.json`` per scene)."""
import argparse
import json
from pathlib import Path

import cv2

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.run_detectors import DEFAULT_VOCABULARY
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector


def detect_scene(detector, folder, name):
    rows = [json.loads(line) for line in (folder / "views.jsonl").read_text().splitlines()]
    found = {}
    for row in rows:
        rgb = cv2.cvtColor(cv2.imread(str(folder / "frames" / row["frame"])), cv2.COLOR_BGR2RGB)
        found[row["frame"]] = [{"cls": d.cls, "conf": round(float(d.conf), 4), "xyxy": [round(float(v), 1) for v in d.xyxy]}
                               for d in detector.detect(rgb)]
    (folder / ("det_%s.json" % name)).write_text(json.dumps(found))
    return len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--detector-url", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--include-held-out", action="store_true", help="only for the final numbers")
    args = p.parse_args(argv)
    vocabulary = [w for w in DEFAULT_VOCABULARY.read_text().strip().split(",") if w]
    detector = HttpDetector(args.detector_url, vocabulary, timeout_s=120.0, expected_backend=args.backend)
    for folder in sorted(d for d in args.views.iterdir() if (d / "views.jsonl").exists()):
        if is_held_out(folder.name, args.held_out_fraction) and not args.include_held_out:
            continue
        if not (folder / ("det_%s.json" % args.name)).exists():
            print("%s: %d frames" % (folder.name, detect_scene(detector, folder, args.name)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
