"""Objects a detector finds in the frames of each Matterport3D region: with depth, each box is placed on the floor plan and kept as inside the region or not."""
import argparse
import json
from pathlib import Path

import cv2

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.run_detectors import DEFAULT_VOCABULARY
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.projection import box_floor_point
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_eval import unique_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector

MIN_CONF = 0.3
SAME_OBJECT_M = 0.6
BOX_MARGIN_M = 0.3
HALF = 0.5


def inside(point, box_xz):
    x0, z0, x1, z1 = box_xz
    return x0 - BOX_MARGIN_M <= point[0] <= x1 + BOX_MARGIN_M and z0 - BOX_MARGIN_M <= point[1] <= z1 + BOX_MARGIN_M


def room_objects(detector, folder, room, width=640, height=480, hfov_deg=79.0):
    placed = []
    for f in room["frames"]:
        rgb = cv2.cvtColor(cv2.imread(str(folder / "frames" / (f["frame"] + ".jpg"))), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(folder / "depth" / (f["frame"] + ".png")), cv2.IMREAD_UNCHANGED) / 1000.0
        for d in detector.detect(rgb):
            if d.conf < MIN_CONF:
                continue
            point = box_floor_point([v * HALF for v in d.xyxy], depth, (f["x"], f["z"], f["yaw"]), width * HALF, height * HALF, hfov_deg)
            if point is not None:
                placed.append({"cls": d.cls, "point": point})
    return {"all": unique_objects(placed, SAME_OBJECT_M),
            "inside": unique_objects([p for p in placed if inside(p["point"], room["box_xz"])], SAME_OBJECT_M)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--detector-url", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY, help="comma-separated words, in the order the server was started with")
    args = p.parse_args(argv)
    vocabulary = [w for w in args.vocabulary.read_text().strip().split(",") if w]
    detector = HttpDetector(args.detector_url, vocabulary, timeout_s=120.0, expected_backend=args.backend)
    found = {}
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "regions.json").exists()):
        for room in json.loads((folder / "regions.json").read_text())["rooms"]:
            found[room["room"]] = room_objects(detector, folder, room)
        print(folder.name, flush=True)
    (args.rooms / ("objects_%s.json" % args.name)).write_text(json.dumps(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
