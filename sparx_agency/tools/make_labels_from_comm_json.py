#!/usr/bin/env python3
"""
Build a labels.json for run_room_mapper.py from Comm Manager per-frame JSONs.

Usage:
    python3 sparx_agency/tools/make_labels_from_comm_json.py \
        --session-dir /home/user/jetson-containers/data/R1/R2/20260623_134637 \
        --comm-json-dir /home/user/jetson-containers/data/R1/2026_06_23___16_29_29 \
        --output /tmp/labels.json
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir",   required=True,
                   help="Original capture session dir (JPGs + NPY files)")
    p.add_argument("--comm-json-dir", required=True,
                   help="Comm Manager output dir containing updated JSONs with nanoowl key")
    p.add_argument("--output",        default="labels.json")
    p.add_argument("--depth-subdir",  default=".",
                   help="Where NPY files live relative to session-dir (default '.')")
    p.add_argument("--min-score",     type=float, default=0.25,
                   help="Min NanoOWL detection score to include (default 0.25)")
    args = p.parse_args()

    session_dir = Path(args.session_dir)
    comm_dir    = Path(args.comm_json_dir)
    depth_dir   = (session_dir / args.depth_subdir).resolve()

    depth_stems = {p.stem for p in depth_dir.glob("*.npy")}
    jpgs = sorted(
        [p for p in session_dir.glob("*.jpg") if p.stem in depth_stems],
        key=lambda p: p.stem,
    )

    labels = []
    missing = 0
    for i, jpg in enumerate(jpgs):
        json_path = comm_dir / jpg.with_suffix(".json").name
        if not json_path.exists():
            missing += 1
            continue

        data = json.loads(json_path.read_text())
        nanoowl  = data.get("nanoowl", {})
        result   = nanoowl.get("result", {})
        dets     = result.get("detections", [])
        img_info = result.get("image", {})
        src_w    = img_info.get("width",  720)
        src_h    = img_info.get("height", 420)

        for det in dets:
            if det.get("score", 0) < args.min_score:
                continue
            x1, y1, x2, y2 = det["bbox"]
            clean_label = det["label"].removeprefix("a ").removeprefix("an ")
            labels.append({
                "frame_idx":   i,
                "label":       clean_label,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "source_size": [src_w, src_h],
                "score":       round(det["score"], 4),
            })

    out = Path(args.output)
    out.write_text(json.dumps(labels, indent=2))
    print(f"[labels] {len(labels)} detections from {len(jpgs) - missing}/{len(jpgs)} frames → {out}")
    if missing:
        print(f"[labels] {missing} frames had no matching Comm Manager JSON")


if __name__ == "__main__":
    main()
