#!/usr/bin/env python3
"""run_live_target_lock.py -- live target lock over a growing frame folder (e.g.
the XTEND frame publisher's live stream), shown in an on-screen window. No ROS, no
Falcon, no live drone connection required -- this only reads whatever a frame
publisher is already writing to disk and shows what command *would* be sent.

Continuously polls ``--images`` for its newest frame -- never a stale/backlogged
one; if the detector is slower than the incoming frame rate it always jumps to the
latest, the same way the live ROS node always acts on ``self.rgb`` rather than a
queue. Each new frame runs through the same detect -> confirm -> track -> servo ->
FSM stack as :mod:`.run_folder_target_lock` and is shown in a window: the camera
frame (with the tracked box) beside a status panel with the mission state,
offsets/range, and ROLL/PITCH/YAW gauges for the exact body-frame command that
would be published to ``/cmd_vel``.

Needs a display: run on the Jetson's own screen, or ``ssh -X``/VNC into it.
Press 'q' in the window (or Ctrl+C) to stop.

Run (target Orin, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.object_approach_offline.run_live_target_lock \\
        --backbone .../orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \\
        --head     .../orin_sm87/yolo_world_s.head.fp16.gpu.engine \\
        --text-weights /path/to/yolov8s-worldv2.pt \\
        --images /tmp/xtend_frames \\
        --target bottle --distractors "chair, table, shelf"
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Optional

import cv2
import numpy as np

from sparx_agency.tasks.planning.object_approach_offline import overlay
from sparx_agency.tasks.planning.object_approach_offline.cli_common import (
    add_target_lock_args,
    build_pipeline,
    depth_path_for,
    load_intrinsics,
)
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector

_IMG_EXTS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
_WINDOW_NAME = "object_approach -- target lock (live)"


def _newest_image(images_dir: str) -> Optional[str]:
    """The newest image in ``images_dir`` by filename, or None if empty.

    Ranked by filename (zero-padded sequence numbers sort lexicographically =
    chronologically -- the same assumption the frame publisher's own rolling
    cleanup makes), not modification time: a rolling live buffer deletes its
    oldest frames continuously, and stat-ing every listed file to compare mtimes
    (a syscall per file) can race that deletion and raise FileNotFoundError.
    Comparing the already-listed names does no further filesystem access.
    """
    files = [p for p in glob.glob(os.path.join(images_dir, "*"))
             if p.rsplit(".", 1)[-1].lower() in _IMG_EXTS]
    return max(files, key=os.path.basename) if files else None


def run(args: argparse.Namespace) -> None:
    det = YoloTRTDetector(args.backbone, args.head, text_weights=args.text_weights,
                          text_device=args.text_device, conf_thresh=args.conf,
                          iou_thresh=args.iou, max_det=args.max_det)
    det.set_prompts([args.target] + args.distractors)

    pipeline = None
    intr = None
    last_path = None
    start_t = time.monotonic()
    n_frames = 0

    print("[live] watching %s for target=%r ... press 'q' in the window to stop"
          % (args.images, args.target))
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            path = _newest_image(args.images)
            if path is None or path == last_path:
                if (cv2.waitKey(args.poll_ms) & 0xFF) == ord("q"):
                    break
                continue

            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue  # rare: caught mid-rename despite the publisher's atomic write
            last_path = path

            if pipeline is None:
                intr = load_intrinsics(args, bgr)
                pipeline = build_pipeline(args, intr)
            elif bgr.shape[:2] != (intr.height, intr.width):
                print("[warn] %s is %dx%d, expected %dx%d -- skipping"
                      % (os.path.basename(path), bgr.shape[1], bgr.shape[0],
                         intr.width, intr.height))
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = det.detect(rgb)

            depth = None
            if not args.no_depth:
                dpath = depth_path_for(path, args.depth)
                if dpath is not None:
                    depth = np.load(dpath).astype(np.float32)

            result = pipeline.step(bgr, time.monotonic() - start_t, dets, depth)
            n_frames += 1
            cv2.imshow(_WINDOW_NAME, overlay.render(bgr, result))

            if (cv2.waitKey(args.poll_ms) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("[live] stopped after %d frame(s)" % n_frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="folder a live frame publisher is writing into")
    ap.add_argument("--depth", default=None,
                    help="optional folder of matching depth .npy (HxW meters, same "
                         "basename as the image)")
    ap.add_argument("--poll-ms", type=int, default=30,
                    help="how often (ms) to check --images for a newer frame; also "
                         "the GUI event-loop tick")
    add_target_lock_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
