"""Ask a vision model about every detector box of the goal kind on the object views, with the in-loop prompt, and store its answers."""
import argparse
import collections
import json
import time
from pathlib import Path

import cv2

from sparx_agency.tasks.planning.objnav_benchmark_runtime.box_check_client import OllamaBoxCheck
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.evaluate import merged_by_kind
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.scoring import KINDS_OF_CATEGORY, classify_box

RIGHT = ("target", "other_goal")


def boxes_to_check(views, name, threshold, held_out_fraction):
    """``(scene, frame, xyxy, conf, category, label, truncated)`` for every box of the view's goal kind, label = what it covers."""
    for folder in sorted(d for d in views.iterdir() if (d / ("det_%s.json" % name)).exists()):
        if is_held_out(folder.name, held_out_fraction):
            continue
        found = json.loads((folder / ("det_%s.json" % name)).read_text())
        for line in (folder / "views.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["role"] != "object":
                continue
            kinds = KINDS_OF_CATEGORY[row["category"]]
            for d in merged_by_kind(found[row["frame"]]):
                if DETECTOR_KIND[d["cls"]] in kinds and d["conf"] >= threshold:
                    yield folder.name, row["frame"], d["xyxy"], d["conf"], row["category"], classify_box(row, d, row["category"]), row["distance_m"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--per-cell", type=int, default=0, help="at most this many boxes per category and right/wrong; 0 = all")
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    args = p.parse_args(argv)
    out = args.views / ("vlm_%s_%s.jsonl" % (args.name, args.model.replace(":", "_")))
    done = {tuple(json.loads(l)["id"]) for l in out.read_text().splitlines()} if out.exists() else set()
    check = OllamaBoxCheck(args.url, args.model, timeout_s=120.0)
    taken = collections.Counter()
    with open(out, "a") as handle:
        for scene, frame, xyxy, conf, category, label, distance in boxes_to_check(args.views, args.name, args.threshold, args.held_out_fraction):
            cell = (category, label in RIGHT)
            if args.per_cell and taken[cell] >= args.per_cell:
                continue
            taken[cell] += 1
            key = (scene, frame, *[round(v) for v in xyxy])
            if key in done:
                continue
            rgb = cv2.cvtColor(cv2.imread(str(args.views / scene / "frames" / frame)), cv2.COLOR_BGR2RGB)
            started = time.monotonic()
            answer = check.verdict(rgb, xyxy, category)
            handle.write(json.dumps({"id": list(key), "category": category, "label": label, "conf": conf, "distance_m": distance,
                                     "answer": answer, "seconds": round(time.monotonic() - started, 2)}) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
