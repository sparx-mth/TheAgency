"""Pinhole camera intrinsics calibration from chessboard views.

Robot-agnostic: takes already-captured BGR frames and pattern geometry, runs
OpenCV's calibrateCamera, and writes the result in the same
camera_matrix/distortion_coefficients YAML schema already used by
tasks/localization/config/front_camera_calib.yaml and the robots/XTEND/config
camera_*.yaml files, so it's a drop-in replacement for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# OpenCV pattern size is INNER corners, not squares: an NxM squares board has
# (N-1)x(M-1) inner corners. See robots/common/calibrate_small_depth_from_chessboard.py.
PatternSize = Tuple[int, int]


@dataclass
class CalibrationResult:
    """Result of a single calibrate_camera() run."""

    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (5,) for the default plumb_bob model
    rms_reprojection_error: float
    per_view_errors: List[float]


def make_object_points(pattern_size: PatternSize, square_size_m: float) -> np.ndarray:
    """Builds the flat (N, 3) grid of 3D corner positions for one chessboard view.

    The grid lies in the Z=0 plane in the board's own frame; units are meters.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * square_size_m
    return objp


def find_chessboard_corners(
    image_bgr: np.ndarray, pattern_size: PatternSize
) -> Optional[np.ndarray]:
    """Finds sub-pixel inner-corner positions, or None if the board isn't visible.

    Prefers cv2.findChessboardCornersSB (more robust, self-refining); falls
    back to the classic detector + cornerSubPix.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if found:
            return corners.reshape(-1, 1, 2).astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def calibrate_camera(
    object_points_list: Sequence[np.ndarray],
    image_points_list: Sequence[np.ndarray],
    image_size: Tuple[int, int],
) -> CalibrationResult:
    """Runs cv2.calibrateCamera and computes a per-view reprojection error.

    Args:
        object_points_list: one (N, 3) array per captured view (see make_object_points).
        image_points_list: one (N, 1, 2) array per captured view (see find_chessboard_corners).
        image_size: (width, height) in pixels.

    Raises:
        ValueError: if fewer than 3 views are given (cv2.calibrateCamera's own minimum).
    """
    if len(object_points_list) < 3:
        raise ValueError(
            f"Need at least 3 chessboard views to calibrate, got {len(object_points_list)}."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None
    )

    per_view_errors = []
    for objp, imgp, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.reshape(-1),
        rms_reprojection_error=float(rms),
        per_view_errors=per_view_errors,
    )


def save_calibration_yaml(
    path: Path,
    image_width: int,
    image_height: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    distortion_model: str = "plumb_bob",
) -> None:
    """Writes calibration results in the schema read by
    tasks/localization/common/apriltag_cv_common.py,
    tasks/localization/publish_camera_info_from_yaml.py, and
    robots/common/helpers.py:load_camera_info_from_yaml (used by
    depth_processor_node.py — this one requires rectification_matrix and
    projection_matrix to be present, not just optional).

    rectification_matrix is identity and projection_matrix is [K|0] (no
    stereo baseline, Tx=0) — the same no-rectification convention already
    used in robots/ROBOTICAN/adapters/rooster_video_adapter.py's
    VideoStreamManager.make_camera_info().
    """
    k = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    fx, cx, fy, cy = k[0, 0], k[0, 2], k[1, 1], k[1, 2]
    projection = [
        fx, 0.0, cx, 0.0,
        0.0, fy, cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    rectification = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    data = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_matrix": {"data": [float(v) for v in k.reshape(-1)]},
        "distortion_model": distortion_model,
        "distortion_coefficients": {
            "data": [float(v) for v in np.asarray(dist_coeffs).reshape(-1)]
        },
        "rectification_matrix": {"data": [float(v) for v in rectification]},
        "projection_matrix": {"data": [float(v) for v in projection]},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
