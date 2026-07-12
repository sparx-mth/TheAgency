#!/usr/bin/env python3
"""run_webcam_target_lock.py -- drive the object-approach target-lock stack from a laptop.

Runs the *exact* ``tasks/planning/object_approach_offline`` pipeline (detect ->
confirm -> track -> visual servo -> SEARCH/APPROACH/HOVER_LOCK/RECOVER) and its HUD,
but sourced from a laptop webcam and a laptop detector instead of the drone + the
Jetson TensorRT detector. No drone, no depth, no localization, no TensorRT needed --
so every new mechanism can be exercised at home:

  * detector vs detector+tracker closure (``--lock-mode``),
  * the robust Median-Flow tracker (hold an object, then cover it -- the box does
    not run off onto the background),
  * the HUD colours: green (detector sees it) / orange (tracking only) / red
    whole-frame border (RECOVER) / grey border (SEARCH),
  * the RECOVER manoeuvres: move the object off to one side (directional chase) vs
    hide it behind something in the centre (occluder peek).

Two frame sources:
  * ``--images /tmp/xtend_frames`` (default) -- read the rolling folder the mock
    ``webcam_frame_publisher`` writes, i.e. the exact two-process drone setup;
  * ``--camera 0`` -- open the webcam directly in this one process (no publisher).

Two detectors (``--detector``):
  * ``yoloworld`` (default; ``yolo`` is an alias) -- the project's real
    **open-vocabulary YOLO-World** (`core/mapping/detection/YoloWorldDetector`), the
    torch analog of the drone's TensorRT YOLO-World; ``--target`` can be any prompt.
    Needs ``torch``/``ultralytics``; GPU auto-selected. NOT plain COCO.
  * ``color`` -- zero-dependency colour-blob mock; hold a coloured object.

Run (two-process, faithful to the drone):
    # terminal 1 -- the drone RGB mock:
    python -m sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher
    # terminal 2 -- the mission + HUD (YOLO-World by default):
    python -m sparx_agency.tasks.planning.object_approach_webcam.run_webcam_target_lock \
        --target person

Or one-process with the colour mock (no model):
    python -m sparx_agency.tasks.planning.object_approach_webcam.run_webcam_target_lock \
        --camera 0 --target cup --detector color --color red --lock-mode detector

Press 'q' in the window (or Ctrl+C) to stop.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from sparx_agency.core.common.types import Intrinsics, KinematicLimits
from sparx_agency.core.mapping.tracking import DetectionOnlyConfig, TargetTrackerConfig
from sparx_agency.core.planning.visual_servo import (
    ApproachFSMConfig,
    ConfirmationGateConfig,
    ReSearchConfig,
    VisualServoParams,
)
from sparx_agency.tasks.planning.object_approach_offline import overlay
from sparx_agency.tasks.planning.object_approach_offline.pipeline import TargetLockPipeline
from sparx_agency.tasks.planning.object_approach_webcam.detector_factory import (
    COLOR,
    YOLOWORLD,
    make_webcam_detector,
)
from sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher import (
    center_crop_resize,
    open_camera,
)

_IMG_EXTS = ("jpg", "jpeg", "png", "bmp")
_WINDOW = "object_approach -- webcam target lock"


def _center_intrinsics(w: int, h: int) -> Intrinsics:
    """Image-centred intrinsics. Without depth the servo uses only width/height and
    the frame centre, so fx/fy/cx/cy need only be self-consistent, not calibrated."""
    return Intrinsics(width=int(w), height=int(h), fx=float(w), fy=float(w),
                      cx=w / 2.0, cy=h / 2.0)


def _build_pipeline(args: argparse.Namespace, w: int, h: int) -> TargetLockPipeline:
    intr = _center_intrinsics(w, h)
    limits = KinematicLimits(max_speed_xy=0.4, max_speed_z=0.3, max_yaw_rate=0.6)
    servo = VisualServoParams(
        mode="holonomic", use_lateral=True, use_depth=False,   # no depth at home
        vx_max=args.vx_max, center_tol=args.center_tol,
        target_area_frac=args.target_area_frac)
    return TargetLockPipeline(
        target=args.target, intrinsics=intr, limits=limits, servo_params=servo,
        gate_config=ConfirmationGateConfig(n_confirm=args.n_confirm,
                                           min_score=args.min_score),
        lock_mode=args.lock_mode,
        tracker_config=TargetTrackerConfig(max_predict_s=args.max_predict_s,
                                           max_unconfirmed_s=args.max_unconfirmed_s),
        detection_config=DetectionOnlyConfig(max_det_age_s=args.max_det_age_s),
        fsm_config=ApproachFSMConfig(recover_timeout_s=args.recover_timeout_s),
        recovery_config=ReSearchConfig(max_search_s=args.recover_timeout_s),
        confirm_iou=args.confirm_iou,
        soft_confirm_min_score=args.soft_confirm_min_score)


class _FolderSource:
    """Yield the newest frame from a rolling folder (the webcam publisher's output)."""

    def __init__(self, images_dir: str) -> None:
        self._dir = images_dir
        self._last = None

    def read(self) -> Optional[np.ndarray]:
        # Pick the most-recently-MODIFIED frame, not the highest-named one: a
        # leftover high-numbered file from a previous run would otherwise fool a
        # name-ordered pick forever. mtime is stat'd race-safe (a rolling buffer may
        # delete a file between listing and stat -> skip it).
        newest, newest_mtime = None, -1.0
        for p in glob.glob(os.path.join(self._dir, "*")):
            if p.rsplit(".", 1)[-1].lower() not in _IMG_EXTS:
                continue
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue                       # vanished mid-scan (rolled away)
            if m > newest_mtime:
                newest, newest_mtime = p, m
        if newest is None or newest == self._last:
            return None                        # empty, or no new frame yet
        img = cv2.imread(newest, cv2.IMREAD_COLOR)
        if img is None:
            return None                        # caught mid-write; try again next tick
        self._last = newest
        return img

    def release(self) -> None:
        pass


class _CameraSource:
    """Open the webcam directly and centre-crop/resize each frame to (w, h)."""

    def __init__(self, index: int, w: int, h: int) -> None:
        self._cap = open_camera(index)         # V4L2-preferring (continuous streaming)
        if not self._cap.isOpened():
            raise RuntimeError("could not open camera index %d" % index)
        self._w, self._h = w, h

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return center_crop_resize(frame, self._w, self._h)

    def release(self) -> None:
        self._cap.release()


def run(args: argparse.Namespace) -> None:
    det = make_webcam_detector(args.detector, target=args.target,
                               distractors=args.distractors, color=args.color,
                               weights=args.weights, conf=args.conf, device=args.device)
    source = (_CameraSource(args.camera, args.img_width, args.img_height)
              if args.camera is not None
              else _FolderSource(args.images))
    pipeline: Optional[TargetLockPipeline] = None
    start = time.monotonic()
    n = 0
    last_state = None
    wait_hb = time.monotonic()
    src_desc = ("camera %d" % args.camera) if args.camera is not None else args.images

    det_desc = ("colour mock (%s blob)" % args.color if args.detector == COLOR
                else "YOLO-World open-vocab (%s)" % args.weights)
    print("[webcam-lock] target=%r detector=%s source=%s lock_mode=%s -- press 'q' to stop"
          % (args.target, det_desc,
             ("camera %d" % args.camera) if args.camera is not None else args.images,
             args.lock_mode))
    if not args.no_window:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            bgr = source.read()
            if bgr is None:
                now = time.monotonic()
                if now - wait_hb >= 3.0:       # no new frame for a while -> say so
                    print("[webcam-lock] waiting for new frames from %s ... "
                          "(is the publisher writing? check `ls %s`)"
                          % (src_desc, args.images if args.camera is None else "the camera"))
                    wait_hb = now
                if not args.no_window and (cv2.waitKey(args.poll_ms) & 0xFF) == ord("q"):
                    break
                if args.no_window:
                    time.sleep(args.poll_ms / 1000.0)
                continue
            wait_hb = time.monotonic()
            if pipeline is None:
                h, w = bgr.shape[:2]
                pipeline = _build_pipeline(args, w, h)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = det.detect(rgb)
            result = pipeline.step(bgr, time.monotonic() - start, dets)  # no depth
            n += 1
            if result.fsm_mode != last_state:
                print("  [state] %-11s (detector_hit=%s track_valid=%s)"
                      % (result.fsm_mode, result.target_detection is not None,
                         result.track is not None and result.track.valid))
                last_state = result.fsm_mode

            if args.no_window:
                if n >= args.max_frames > 0:
                    break
                continue
            cv2.imshow(_WINDOW, overlay.render(bgr, result))
            if (cv2.waitKey(args.poll_ms) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        print("[webcam-lock] stopped after %d frame(s)" % n)


def _distractors(value: str):
    return [s.strip().lower() for s in value.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--images", default="/tmp/xtend_frames",
                     help="rolling folder the webcam publisher writes (default)")
    src.add_argument("--camera", type=int, default=None,
                     help="open this webcam index directly (one-process mode)")

    ap.add_argument("--target", required=True, type=lambda s: s.strip().lower(),
                    help="object to lock onto (any open-vocab prompt for YOLO-World; "
                         "a colour label for --detector color)")
    ap.add_argument("--distractors", default="", type=_distractors,
                    help="comma-separated extra prompts to also score (context)")
    ap.add_argument("--detector", choices=(YOLOWORLD, "yolo", COLOR), default=YOLOWORLD,
                    help="yoloworld (default; 'yolo' is an alias) = open-vocab "
                         "YOLO-World; color = zero-dependency colour-blob mock")
    ap.add_argument("--color", default="red",
                    help="blob colour for --detector color "
                         "(red/orange/yellow/green/blue/purple)")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt",
                    help="YOLO-World checkpoint (yolov8{n,s,m,l,x}-worldv2.pt); must "
                         "be a YOLO-World model, not a plain COCO one")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="YOLO-World min confidence FLOOR (kept low so weak boxes are "
                         "emitted for the tracking soft-confirm; the gate's --min-score "
                         "is the higher bar that actually acquires/greens a target)")
    ap.add_argument("--device", default="",
                    help="torch device for YOLO-World (blank = auto: your RTX GPU if "
                         "available; or force 'cpu'/'cuda:0')")

    ap.add_argument("--lock-mode", choices=("detector_tracker", "detector"),
                    default="detector_tracker")
    ap.add_argument("--n-confirm", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.15,
                    help="hard-confirmation floor: score to acquire / show green "
                         "(keep it above --conf so a soft band exists)")
    ap.add_argument("--max-predict-s", type=float, default=0.4)
    ap.add_argument("--max-det-age-s", type=float, default=0.5)
    ap.add_argument("--recover-timeout-s", type=float, default=6.0)

    # Tracking-only guard: drop the lock if the detector hasn't re-confirmed the
    # target for this long (stops tracking the background); a weak detection ON the
    # tracked box (>= --soft-confirm-min-score, IoU >= --confirm-iou) counts as a
    # re-confirmation and keeps it alive.
    ap.add_argument("--max-unconfirmed-s", type=float, default=2.0)
    ap.add_argument("--confirm-iou", type=float, default=0.4)
    ap.add_argument("--soft-confirm-min-score", type=float, default=0.05)

    ap.add_argument("--vx-max", type=float, default=0.35)
    ap.add_argument("--center-tol", type=float, default=0.15)
    ap.add_argument("--target-area-frac", type=float, default=0.12,
                    help="box area fraction that counts as 'close enough' (no depth)")

    ap.add_argument("--img-width", type=int, default=504, help="camera-mode resize width")
    ap.add_argument("--img-height", type=int, default=294, help="camera-mode resize height")
    ap.add_argument("--poll-ms", type=int, default=30, help="frame poll / GUI tick (ms)")
    ap.add_argument("--no-window", action="store_true",
                    help="headless: log state transitions instead of showing a window")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = run forever); for headless testing")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
