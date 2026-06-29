#!/usr/bin/env python3
"""
Estimate camera-relative position from DA3 depth using simple plane fitting.

Coordinate convention used here:
  x = camera right
  y = camera down
  z = camera forward

Outputs per frame:
  - distance to left wall
  - distance to right wall
  - lateral offset from corridor/room center, if both walls are visible
  - distance to front plane / door area
  - approximate camera height from floor, if floor is visible

This is not full SLAM. It estimates position relative to currently visible planes.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml


PlaneFit = Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]


from sparx_agency.robots.common.helpers import load_intrinsics_from_yaml, valid_depth_mask


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_intrinsics_for_depth(camera_yaml: Path, depth_w: int, depth_h: int) -> tuple[float, float, float, float]:
    """Load rectified pinhole intrinsics and scale them to the depth image size."""
    return load_intrinsics_from_yaml(camera_yaml, depth_w=depth_w, depth_h=depth_h)


def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Return HxWx3 XYZ points in camera coordinates."""
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.dstack((x, y, z)).astype(np.float32)


def roi_points(points: np.ndarray, roi: tuple[float, float, float, float], depth: np.ndarray,
               min_depth: float, max_depth: float, stride: int) -> np.ndarray:
    """Extract valid points from normalized ROI: x0,x1,y0,y1 in 0..1."""
    h, w = depth.shape[:2]
    x0 = int(np.clip(roi[0], 0.0, 1.0) * w)
    x1 = int(np.clip(roi[1], 0.0, 1.0) * w)
    y0 = int(np.clip(roi[2], 0.0, 1.0) * h)
    y1 = int(np.clip(roi[3], 0.0, 1.0) * h)

    patch_points = points[y0:y1:stride, x0:x1:stride].reshape(-1, 3)
    patch_depth = depth[y0:y1:stride, x0:x1:stride].reshape(-1)

    valid = valid_depth_mask(patch_depth, min_depth=min_depth, max_depth=max_depth)
    valid &= np.all(np.isfinite(patch_points), axis=1)

    return patch_points[valid]


def fit_plane_ransac(points: np.ndarray, axis: np.ndarray, min_axis_dot: float,
                     iterations: int, threshold_m: float, min_inliers: int) -> PlaneFit:
    """
    Fit plane n.x + d = 0 using simple RANSAC.
    axis is the expected normal direction, e.g. x-axis for side walls, z-axis for front wall.
    """
    if points.shape[0] < max(min_inliers, 30):
        return None, None, None, None

    rng = np.random.default_rng(3)
    axis = axis.astype(np.float64)
    axis = axis / np.linalg.norm(axis)

    best_n = None
    best_d = None
    best_inliers = None
    best_count = 0

    n_points = points.shape[0]
    pts = points.astype(np.float64)

    for _ in range(iterations):
        idx = rng.choice(n_points, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm

        if abs(float(np.dot(n, axis))) < min_axis_dot:
            continue

        d = -float(np.dot(n, p0))
        dist = np.abs(pts @ n + d)
        inliers = dist < threshold_m
        count = int(np.count_nonzero(inliers))

        if count > best_count:
            best_count = count
            best_n = n
            best_d = d
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return None, None, None, None

    inlier_pts = pts[best_inliers]

    # Refine plane with PCA/SVD on inliers.
    centroid = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n_refined = vh[-1, :]
    n_refined = n_refined / np.linalg.norm(n_refined)

    if abs(float(np.dot(n_refined, axis))) < min_axis_dot:
        # Keep the RANSAC plane if SVD normal flips to a less expected orientation.
        n_refined = best_n
        centroid = np.mean(inlier_pts, axis=0)

    d_refined = -float(np.dot(n_refined, centroid))

    return n_refined, d_refined, best_inliers, inlier_pts.astype(np.float32)


def safe_median(values: np.ndarray) -> float:
    if values is None or values.size == 0:
        return float("nan")
    return float(np.median(values))


def plane_distance_to_origin(n: Optional[np.ndarray], d: Optional[float]) -> float:
    if n is None or d is None:
        return float("nan")
    return float(abs(d) / max(np.linalg.norm(n), 1e-12))


def process_frame(depth_path: Path, camera_yaml: Path, args) -> dict:
    depth = np.load(depth_path).astype(np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]

    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0

    h, w = depth.shape[:2]
    fx, fy, cx, cy = load_intrinsics_for_depth(camera_yaml, w, h)
    points = backproject_depth(depth, fx, fy, cx, cy)

    left_pts = roi_points(points, args.left_roi, depth, args.min_depth, args.max_depth, args.stride)
    right_pts = roi_points(points, args.right_roi, depth, args.min_depth, args.max_depth, args.stride)
    front_pts = roi_points(points, args.front_roi, depth, args.min_depth, args.max_depth, args.stride)
    floor_pts = roi_points(points, args.floor_roi, depth, args.min_depth, args.max_depth, args.stride)

    side_axis = np.array([1.0, 0.0, 0.0])
    front_axis = np.array([0.0, 0.0, 1.0])
    floor_axis = np.array([0.0, 1.0, 0.0])

    _, _, _, left_inliers = fit_plane_ransac(
        left_pts, side_axis, args.min_axis_dot, args.ransac_iters, args.plane_threshold_m, args.min_inliers
    )
    _, _, _, right_inliers = fit_plane_ransac(
        right_pts, side_axis, args.min_axis_dot, args.ransac_iters, args.plane_threshold_m, args.min_inliers
    )
    n_front, d_front, _, front_inliers = fit_plane_ransac(
        front_pts, front_axis, args.min_axis_dot, args.ransac_iters, args.plane_threshold_m, args.min_inliers
    )
    n_floor, d_floor, _, floor_inliers = fit_plane_ransac(
        floor_pts, floor_axis, args.min_axis_dot, args.ransac_iters, args.plane_threshold_m, args.min_inliers
    )

    # For side walls, median X is more interpretable than perpendicular distance.
    # left wall should have negative x, right wall positive x.
    left_x = safe_median(left_inliers[:, 0]) if left_inliers is not None else float("nan")
    right_x = safe_median(right_inliers[:, 0]) if right_inliers is not None else float("nan")

    d_left = abs(left_x) if np.isfinite(left_x) and left_x < 0 else float("nan")
    d_right = abs(right_x) if np.isfinite(right_x) and right_x > 0 else float("nan")

    if np.isfinite(d_left) and np.isfinite(d_right):
        corridor_width = d_left + d_right
        x_from_center = (d_left - d_right) / 2.0
    else:
        corridor_width = float("nan")
        x_from_center = float("nan")

    z_front = safe_median(front_inliers[:, 2]) if front_inliers is not None else float("nan")
    floor_distance = plane_distance_to_origin(n_floor, d_floor)
    y_floor_median = safe_median(floor_inliers[:, 1]) if floor_inliers is not None else float("nan")

    return {
        "depth_width": w,
        "depth_height": h,
        "fx_depth": fx,
        "fy_depth": fy,
        "cx_depth": cx,
        "cy_depth": cy,
        "left_wall_x_m": left_x,
        "right_wall_x_m": right_x,
        "distance_left_wall_m": d_left,
        "distance_right_wall_m": d_right,
        "estimated_room_width_m": corridor_width,
        "x_from_center_m": x_from_center,
        "z_to_front_plane_m": z_front,
        "floor_distance_plane_m": floor_distance,
        "floor_y_median_m": y_floor_median,
        "left_inliers": 0 if left_inliers is None else int(left_inliers.shape[0]),
        "right_inliers": 0 if right_inliers is None else int(right_inliers.shape[0]),
        "front_inliers": 0 if front_inliers is None else int(front_inliers.shape[0]),
        "floor_inliers": 0 if floor_inliers is None else int(floor_inliers.shape[0]),
    }


def parse_roi(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must contain 4 values: x0 x1 y0 y1")
    return tuple(float(v) for v in values)


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate camera position from side walls, front door/wall, and floor using depth planes.")
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--camera-yaml", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--min-depth", type=float, default=0.3)
    p.add_argument("--max-depth", type=float, default=8.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--ransac-iters", type=int, default=250)
    p.add_argument("--plane-threshold-m", type=float, default=0.04)
    p.add_argument("--min-axis-dot", type=float, default=0.55)
    p.add_argument("--min-inliers", type=int, default=150)
    p.add_argument("--left-roi", nargs=4, type=float, default=[0.00, 0.32, 0.18, 0.88])
    p.add_argument("--right-roi", nargs=4, type=float, default=[0.68, 1.00, 0.18, 0.88])
    p.add_argument("--front-roi", nargs=4, type=float, default=[0.35, 0.65, 0.18, 0.82])
    p.add_argument("--floor-roi", nargs=4, type=float, default=[0.18, 0.82, 0.62, 1.00])
    args = p.parse_args()

    metadata_csv = Path(args.metadata_csv).expanduser()
    camera_yaml = Path(args.camera_yaml).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    if "depth_npy_path" not in df.columns:
        raise RuntimeError("metadata CSV must contain a depth_npy_path column")

    end_idx = len(df) if args.end_idx < 0 else min(args.end_idx, len(df))
    rows = df.iloc[args.start_idx:end_idx:args.step]

    out_rows = []
    for _, row in rows.iterrows():
        frame_idx = int(row["frame_idx"])
        depth_path = Path(str(row["depth_npy_path"]))
        if not depth_path.exists():
            print(f"[warn] missing depth file for frame {frame_idx}: {depth_path}")
            continue

        try:
            result = process_frame(depth_path, camera_yaml, args)
        except Exception as exc:
            print(f"[warn] failed frame {frame_idx}: {exc}")
            continue

        out = row.to_dict()
        out.update(result)
        out_rows.append(out)

        print(
            f"frame={frame_idx:06d} "
            f"left={result['distance_left_wall_m']:.3f}m "
            f"right={result['distance_right_wall_m']:.3f}m "
            f"x_center={result['x_from_center_m']:.3f}m "
            f"front={result['z_to_front_plane_m']:.3f}m "
            f"floor={result['floor_distance_plane_m']:.3f}m"
        )

    if not out_rows:
        raise RuntimeError("No frames processed successfully")

    pd.DataFrame(out_rows).to_csv(output_csv, index=False)
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
