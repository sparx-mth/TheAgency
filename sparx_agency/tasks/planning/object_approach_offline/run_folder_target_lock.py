#!/usr/bin/env python3
"""run_folder_target_lock.py -- offline replay: find a named object in a folder of
RGB frames, visually lock onto it, and show exactly what command would be sent to
the drone. No ROS, no Falcon, no live drone connection required.

Drives the same mission stack ``object_approach_node.py`` wires to ROS
(:class:`TargetConfirmationGate` -> :class:`TargetTracker` ->
:class:`VisualServoController` -> :class:`VisualApproachStateMachine`, with
:class:`ReSearchPolicy` on a lost track) purely in Python, frame by frame, over a
folder such as a ``compare_folder``/``detect_folder`` capture. For every frame it
writes an annotated ``.jpg`` (raw detections + the tracked box on the image, the
mission state/offsets/range and ROLL/PITCH/YAW command gauges in a side panel)
plus a JSONL log, so you can watch the lock happen and check the commanded
velocity before wiring up the real drone.

For a folder that is still being written to live (e.g. the XTEND frame publisher),
use :mod:`.run_live_target_lock` instead -- this tool is for a finished capture.

Run (target Orin, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.object_approach_offline.run_folder_target_lock \\
        --backbone .../orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \\
        --head     .../orin_sm87/yolo_world_s.head.fp16.gpu.engine \\
        --text-weights /path/to/yolov8s-worldv2.pt \\
        --images /path/to/rgb --out /path/to/target_lock \\
        --target bottle --distractors "chair, table"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from sparx_agency.tasks.planning.object_approach_offline import overlay
from sparx_agency.tasks.planning.object_approach_offline.cli_common import (
    add_target_lock_args,
    build_pipeline,
    depth_path_for,
    load_intrinsics,
)
from sparx_agency.tasks.mapping.yolo_world_trt.detect_folder import find_images
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector


def _first_readable(files):
    """First ``(path, image)`` in ``files`` that actually decodes, skipping the rest.

    One corrupt or still-being-written file (e.g. a live frame publisher racing
    this script) must not abort the whole run just because it happens to sort first.
    """
    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return path, img
    return None, None


def _snapshot_frames(files: List[str], snapshot_dir: Path) -> List[str]:
    """Copy ``files`` into ``snapshot_dir`` right away and return the copies' paths.

    ``--images`` may be a live rolling buffer (e.g. the XTEND frame publisher,
    which deletes its oldest frame on every new write) rather than a finished
    capture. Loading the TensorRT engines and the CLIP text branch takes long
    enough that such a buffer can rotate out every frame this script originally
    listed before it gets around to reading them. Snapshotting immediately after
    listing -- before that slow setup -- shrinks the race window from "engine load
    time" to "a few file copies". A file that vanishes even during the copy (the
    buffer is that fast, or genuinely corrupt) is skipped, not fatal.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    skipped = 0
    for path in files:
        dst = snapshot_dir / os.path.basename(path)
        try:
            shutil.copy2(path, dst)
        except OSError:
            skipped += 1
            continue
        copied.append(str(dst))
    if skipped:
        print("[warn] %d/%d frame(s) vanished before they could be snapshotted -- "
              "the source folder is being actively written/rotated (e.g. a live "
              "frame publisher); consider stopping it first for a reproducible run, "
              "or use run_live_target_lock instead" % (skipped, len(files)))
    return copied


def run(args: argparse.Namespace) -> None:
    files = find_images(args.images)
    n_listed = len(files)
    out_dir = Path(args.out)
    vis_dir = out_dir / "annotated"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_snapshot:
        files = _snapshot_frames(files, out_dir / "raw")
        if not files:
            raise RuntimeError(
                "All %d frame(s) vanished before they could be snapshotted -- "
                "--images is rotating faster than this process can copy it. Stop "
                "the live frame publisher first, use run_live_target_lock instead, "
                "or point --images at a finished capture." % n_listed)

    prompts = [args.target] + args.distractors
    det = YoloTRTDetector(args.backbone, args.head, text_weights=args.text_weights,
                          text_device=args.text_device, conf_thresh=args.conf,
                          iou_thresh=args.iou, max_det=args.max_det)
    det.set_prompts(prompts)

    first_path, first = _first_readable(files)
    if first is None:
        raise RuntimeError("No readable image in --images (all %d file(s) failed to "
                           "open) -- if this folder is still being written by a live "
                           "frame publisher, stop it first or point --images at a "
                           "finished capture." % len(files))
    if first_path != files[0]:
        print("[warn] %s is unreadable (corrupt or still being written); using %s "
              "to derive intrinsics" % (os.path.basename(files[0]),
                                        os.path.basename(first_path)))
    intr = load_intrinsics(args, first)
    pipeline = build_pipeline(args, intr)

    print("[target_lock] %d frames | target=%r distractors=%s | intrinsics %dx%d"
          % (len(files), args.target, args.distractors, intr.width, intr.height))

    mode_counts = {}
    jsonl_path = out_dir / "target_lock.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, path in enumerate(files):
            name = os.path.basename(path)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                print("  skipped unreadable image:", name)
                continue
            if bgr.shape[:2] != (intr.height, intr.width):
                raise ValueError(
                    "%s is %dx%d, expected %dx%d (fix --img-width/--img-height)"
                    % (name, bgr.shape[1], bgr.shape[0], intr.width, intr.height))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = det.detect(rgb)

            depth = None
            if not args.no_depth:
                dpath = depth_path_for(path, args.depth)
                if dpath is not None:
                    depth = np.load(dpath).astype(np.float32)

            result = pipeline.step(bgr, i / args.fps, dets, depth)
            mode_counts[result.fsm_mode] = mode_counts.get(result.fsm_mode, 0) + 1

            cv2.imwrite(str(vis_dir / (Path(name).stem + ".jpg")), overlay.render(bgr, result))

            jf.write(json.dumps({
                "image": name, "stamp_s": round(result.stamp_s, 3), "state": result.fsm_mode,
                "confirmed": result.confirmed, "streak": result.streak,
                "track_valid": None if result.track is None else result.track.valid,
                "x_offset": result.x_offset, "y_offset": result.y_offset,
                "area_frac": result.area_frac, "range_m": result.range_m,
                "at_target": result.at_target,
                "command": None if result.command is None else {
                    "vx": result.command.x, "vy": result.command.y,
                    "vz": result.command.z, "yaw_rate": result.command.yaw_rate},
                "cmd_source": result.cmd_source,
            }) + "\n")

    summary = {"images": len(files), "target": args.target,
              "distractors": args.distractors, "mode_counts": mode_counts}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] mode counts:", mode_counts)
    print("       wrote annotated frames + log -> %s" % out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of RGB frames (finished capture)")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="read frames from --images directly instead of copying them "
                         "to <out>/raw first; only safe against a finished capture, "
                         "not a live rolling frame buffer")
    ap.add_argument("--depth", default=None,
                    help="optional folder of per-frame depth .npy (HxW meters, same "
                         "basename as the image); omit to use the area-fraction proxy")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="synthetic frame rate for dt/timestamps (matches ctrl_hz)")
    add_target_lock_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
