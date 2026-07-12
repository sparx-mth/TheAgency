"""Shared CLI plumbing for the batch (:mod:`.run_folder_target_lock`) and live
(:mod:`.run_live_target_lock`) target-lock tools: the flags that configure
:class:`~.pipeline.TargetLockPipeline` are identical either way -- only how frames
are sourced (a finished folder vs. a live-growing one) differs between the two.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from sparx_agency.core.common.types import Intrinsics, KinematicLimits
from sparx_agency.core.planning.visual_servo import (
    ApproachFSMConfig,
    ConfirmationGateConfig,
    ReSearchConfig,
    VisualServoParams,
)
from sparx_agency.core.mapping.tracking import (
    DetectionOnlyConfig,
    DETECTOR,
    DETECTOR_TRACKER,
    TargetTrackerConfig,
)
from sparx_agency.tasks.mapping.yolo_world_trt.detect_folder import parse_labels
from sparx_agency.tasks.planning.object_approach_offline.pipeline import TargetLockPipeline


def labels_or_empty(value: str) -> List[str]:
    """argparse ``type`` for ``--distractors``: ``""`` -> ``[]``, else :func:`parse_labels`."""
    return parse_labels(value) if value.strip() else []


def depth_path_for(image_path: str, depth_dir: Optional[str]) -> Optional[str]:
    """Matching ``<depth_dir>/<image_stem>.npy``, or None if absent/not requested."""
    if not depth_dir:
        return None
    p = Path(depth_dir) / (Path(image_path).stem + ".npy")
    return str(p) if p.exists() else None


def add_target_lock_args(ap: argparse.ArgumentParser) -> None:
    """Register every flag that configures the detector + :class:`TargetLockPipeline`.

    Callers still add their own frame-source flags (``--images``/``--out`` for a
    batch folder, ``--images``/``--poll-ms`` for a live one, etc.).
    """
    ap.add_argument("--backbone", required=True, help="backbone .engine path")
    ap.add_argument("--head", required=True, help="head .engine path")
    ap.add_argument("--text-weights", required=True,
                    help=".pt YOLO-World checkpoint driving the CLIP text branch")
    ap.add_argument("--target", required=True, type=lambda s: s.strip().lower(),
                    help="the object to lock onto, e.g. 'bottle'")
    ap.add_argument("--distractors", default="", type=labels_or_empty,
                    help="extra comma-separated prompts the detector also scores "
                         "(context only; the gate/tracker only lock onto --target)")
    ap.add_argument("--conf", type=float, default=0.40, help="min class confidence")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU (default: manifest)")
    ap.add_argument("--max-det", type=int, default=None, help="max detections/frame")
    ap.add_argument("--text-device", default="cpu", help="torch device for text encode")

    # Camera model -- must match the frames for offsets/range to be meaningful.
    ap.add_argument("--img-width", type=int, default=None, help="default: first frame's width")
    ap.add_argument("--img-height", type=int, default=None, help="default: first frame's height")
    ap.add_argument("--fx", type=float, default=322.6351083474948)
    ap.add_argument("--fy", type=float, default=323.3893307141174)
    ap.add_argument("--cx", type=float, default=242.06479658679714)
    ap.add_argument("--cy", type=float, default=90.03019076680604)

    # Acquisition (TargetConfirmationGate)
    ap.add_argument("--n-confirm", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.30)

    # Closure strategy: what we use to keep the box on the target.
    ap.add_argument("--lock-mode", choices=(DETECTOR_TRACKER, DETECTOR),
                    default=DETECTOR_TRACKER,
                    help="detector_tracker (default): detector seeds an optical-flow "
                         "tracker propagated every frame; detector: the detector's "
                         "box alone, no tracking (use when the detector keeps up "
                         "with the RGB stream)")
    ap.add_argument("--max-det-age-s", type=float, default=0.5,
                    help="(detector mode) hold the last detection as a valid track "
                         "for this long before declaring loss")

    # Tracking (TargetTracker)
    ap.add_argument("--max-predict-s", type=float, default=0.4)
    ap.add_argument("--no-reseed", action="store_true",
                    help="seed the tracker once on acquisition only "
                         "(default: re-seed every matching detection)")

    # Servo (VisualServoController)
    ap.add_argument("--servo-mode", choices=("holonomic", "yaw_forward_xor"),
                    default="holonomic")
    ap.add_argument("--kp-yaw", type=float, default=1.2)
    ap.add_argument("--vx-max", type=float, default=0.35)
    ap.add_argument("--no-lateral", action="store_true")
    ap.add_argument("--use-vertical", action="store_true")
    ap.add_argument("--no-depth", action="store_true",
                    help="ignore --depth even if given; use the area-fraction proxy")
    ap.add_argument("--target-range-m", type=float, default=0.8)
    ap.add_argument("--slowdown-range-m", type=float, default=2.0)
    ap.add_argument("--target-area-frac", type=float, default=0.12)
    ap.add_argument("--center-tol", type=float, default=0.15)

    # Kinematic limits
    ap.add_argument("--max-speed-xy", type=float, default=0.4)
    ap.add_argument("--max-speed-z", type=float, default=0.3)
    ap.add_argument("--max-yaw-rate", type=float, default=0.6)

    # Recovery / FSM
    ap.add_argument("--search-yaw-rate", type=float, default=0.5)
    ap.add_argument("--recover-timeout-s", type=float, default=6.0)


def build_pipeline(args: argparse.Namespace, intr: Intrinsics) -> TargetLockPipeline:
    """Construct a :class:`TargetLockPipeline` from the flags :func:`add_target_lock_args` adds."""
    limits = KinematicLimits(max_speed_xy=args.max_speed_xy, max_speed_z=args.max_speed_z,
                             max_yaw_rate=args.max_yaw_rate)
    servo_params = VisualServoParams(
        mode=args.servo_mode, kp_yaw=args.kp_yaw, max_yaw_rate=args.max_yaw_rate,
        use_lateral=not args.no_lateral, use_vertical=args.use_vertical,
        vx_max=args.vx_max, use_depth=not args.no_depth,
        target_range_m=args.target_range_m, slowdown_range_m=args.slowdown_range_m,
        target_area_frac=args.target_area_frac, center_tol=args.center_tol)
    return TargetLockPipeline(
        target=args.target, intrinsics=intr, limits=limits,
        reseed_on_detection=not args.no_reseed, servo_params=servo_params,
        gate_config=ConfirmationGateConfig(n_confirm=args.n_confirm,
                                           min_score=args.min_score),
        lock_mode=args.lock_mode,
        tracker_config=TargetTrackerConfig(max_predict_s=args.max_predict_s),
        detection_config=DetectionOnlyConfig(max_det_age_s=args.max_det_age_s),
        fsm_config=ApproachFSMConfig(recover_timeout_s=args.recover_timeout_s),
        recovery_config=ReSearchConfig(search_yaw_rate=args.search_yaw_rate,
                                       max_search_s=args.recover_timeout_s))


def load_intrinsics(args: argparse.Namespace, first_frame: np.ndarray) -> Intrinsics:
    """Camera model for the run: ``--img-width/height`` or the first frame's size."""
    h, w = first_frame.shape[:2]
    intr = Intrinsics(width=args.img_width or w, height=args.img_height or h,
                      fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    if (intr.width, intr.height) != (w, h):
        print("[warn] intrinsics %dx%d != first frame %dx%d -- offsets/range will be "
              "wrong unless --img-width/--img-height match the real capture"
              % (intr.width, intr.height, w, h))
    return intr
