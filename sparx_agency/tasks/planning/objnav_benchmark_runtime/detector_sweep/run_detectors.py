"""Run one detector server over the frames of a sweep and store every detection (``det_<name>.json`` per scene)."""
import argparse
import json
from pathlib import Path

import cv2

from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector

DEFAULT_VOCABULARY = Path(__file__).resolve().parents[1] / "trace_tools" / "vocabulary.txt"


def detect_scene(detector, folder, name):
    rows = json.loads((folder / "gt.json").read_text())
    found = {}
    for row in rows:
        rgb = cv2.cvtColor(cv2.imread(str(folder / "frames" / row["frame"])), cv2.COLOR_BGR2RGB)
        found[row["frame"]] = [{"cls": d.cls, "conf": round(float(d.conf), 4), "xyxy": [round(float(v), 1) for v in d.xyxy]}
                               for d in detector.detect(rgb)]
    (folder / ("det_%s.json" % name)).write_text(json.dumps(found))
    return len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, required=True, help="the --out of sample_frames")
    p.add_argument("--detector-url", required=True)
    p.add_argument("--backend", required=True, help="the backend the server reports (yolo_world, closed_open, ...)")
    p.add_argument("--name", required=True, help="label for the output files")
    p.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY, help="comma-separated words, in the order the server was started with")
    args = p.parse_args(argv)
    vocabulary = [w for w in args.vocabulary.read_text().strip().split(",") if w]
    detector = HttpDetector(args.detector_url, vocabulary, timeout_s=120.0, expected_backend=args.backend)
    for folder in sorted(d for d in args.sweep.iterdir() if (d / "gt.json").exists()):
        if (folder / ("det_%s.json" % args.name)).exists():
            continue
        print("%s: %d frames" % (folder.name, detect_scene(detector, folder, args.name)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
