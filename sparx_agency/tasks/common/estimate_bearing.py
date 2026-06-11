#!/usr/bin/env python3
"""
estimate_bearing.py
-------------------
Generate a placeholder bearing (compass heading, degrees) for every frame in a
drone capture folder, when the real bearing telemetry was not recorded.

The output is a CSV keyed by filename. When real bearing data becomes available
later, only the `bearing_deg` column needs to be replaced; the rest of your
pipeline does not have to change.

Three modes are supported:
  - estimate (default): rough yaw estimation using ORB feature matching between
                        consecutive frames. Median horizontal pixel shift of
                        inlier matches is converted to a yaw delta via the
                        camera's horizontal FOV, then accumulated.
  - constant:           every frame gets the same bearing (`--initial-bearing`).
  - linear:             bearing increases linearly from initial to `--linear-end`.

Usage examples:
  python estimate_bearing.py /path/to/rgb_rectified
  python estimate_bearing.py /path/to/rgb_rectified -o bearings_placeholder.csv --hfov 78
  python estimate_bearing.py /path/to/rgb_rectified --mode constant --initial-bearing 90
  python estimate_bearing.py /path/to/rgb_rectified --mode linear --initial-bearing 0 --linear-end 360

Requirements:
  pip install opencv-python numpy
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def list_frames(folder: Path):
    """Return frame paths in sorted (lexicographic) order."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    frames = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return frames


def estimate_yaw_delta(prev_gray, curr_gray, hfov_deg, orb, matcher):
    """
    Estimate yaw delta (degrees) between two grayscale frames using ORB matches.

    Convention: positive yaw delta = clockwise rotation = bearing increases
    (e.g. North -> East). When the camera rotates clockwise, the world shifts
    leftward in the image (negative dx), so we negate the median dx.

    Returns (yaw_delta_deg, confidence_0_to_1).
    """
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0.0

    matches = matcher.match(des1, des2)
    if len(matches) < 10:
        return 0.0, 0.0

    matches = sorted(matches, key=lambda m: m.distance)[:300]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # RANSAC homography to discard outlier matches (moving objects, mismatches).
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if mask is None:
        return 0.0, 0.0
    inliers = mask.ravel().astype(bool)
    n_in = int(inliers.sum())
    if n_in < 8:
        return 0.0, 0.0

    dx = pts2[inliers, 0] - pts1[inliers, 0]
    median_dx = float(np.median(dx))

    width = curr_gray.shape[1]
    deg_per_pixel = hfov_deg / width
    yaw_delta = -median_dx * deg_per_pixel  # see convention note above

    confidence = n_in / len(matches)
    return float(yaw_delta), float(confidence)


def build_rows_constant(frames, initial_bearing):
    rows = []
    for i, f in enumerate(frames):
        rows.append({
            "frame_index": i,
            "filename": f.name,
            "bearing_deg": initial_bearing % 360.0,
            "delta_deg": 0.0,
            "confidence": 1.0,
            "source": "placeholder_constant",
        })
    return rows


def build_rows_linear(frames, initial_bearing, end_bearing):
    bearings = np.linspace(initial_bearing, end_bearing, len(frames))
    rows = []
    for i, f in enumerate(frames):
        delta = float(bearings[i] - bearings[i - 1]) if i > 0 else 0.0
        rows.append({
            "frame_index": i,
            "filename": f.name,
            "bearing_deg": float(bearings[i]) % 360.0,
            "delta_deg": delta,
            "confidence": 1.0,
            "source": "placeholder_linear",
        })
    return rows


def build_rows_estimate(frames, initial_bearing, hfov, downscale):
    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    bearing = float(initial_bearing)
    prev_gray = None
    rows = []

    for i, f in enumerate(frames):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  warn: could not read {f.name}", file=sys.stderr)
            rows.append({
                "frame_index": i, "filename": f.name,
                "bearing_deg": bearing % 360.0, "delta_deg": 0.0,
                "confidence": 0.0, "source": "read_error",
            })
            continue

        if downscale != 1.0:
            img = cv2.resize(img, None, fx=1.0 / downscale, fy=1.0 / downscale,
                             interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            rows.append({
                "frame_index": i, "filename": f.name,
                "bearing_deg": bearing % 360.0, "delta_deg": 0.0,
                "confidence": 1.0, "source": "estimated_initial",
            })
        else:
            delta, conf = estimate_yaw_delta(prev_gray, gray, hfov, orb, matcher)
            bearing += delta
            rows.append({
                "frame_index": i, "filename": f.name,
                "bearing_deg": bearing % 360.0, "delta_deg": delta,
                "confidence": conf, "source": "estimated",
            })

        prev_gray = gray
        if (i + 1) % 50 == 0 or (i + 1) == len(frames):
            print(f"  processed {i + 1}/{len(frames)} frames")

    return rows


def bearing_to_yaw_deg(bearing_deg: float, frame: str = "NED") -> float:
    """
    Convert a compass bearing in [0, 360) to a signed yaw in (-180, 180].

    NED (drone / aviation default): yaw axis matches bearing axis exactly,
        so this is just a range remap from [0, 360) to (-180, 180].
    ENU (ROS / math convention):    yaw is counter-clockwise from East, so
        yaw_enu = 90 - bearing  (then wrapped to the signed range).
    """
    if frame.upper() == "ENU":
        deg = 90.0 - bearing_deg
    else:  # NED
        deg = bearing_deg
    # Wrap to (-180, 180]
    deg = ((deg + 180.0) % 360.0) - 180.0
    if deg == -180.0:
        deg = 180.0
    return deg


def write_csv(rows, out_path: Path, yaw_frame: str = "NED"):
    # Add the yaw columns derived from bearing. Done here so all three modes
    # (estimate / constant / linear) share one source of truth for the formula.
    for row in rows:
        yaw_deg = bearing_to_yaw_deg(row["bearing_deg"], frame=yaw_frame)
        row["yaw_deg"] = yaw_deg
        row["yaw_rad"] = math.radians(yaw_deg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_index", "filename", "bearing_deg", "yaw_deg", "yaw_rad",
                  "delta_deg", "confidence", "source"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate placeholder bearing for drone frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("frames_dir", help="Folder containing the drone frames")
    parser.add_argument("-o", "--output", default="bearings_placeholder.csv",
                        help="Output CSV path. Default name flags that this is "
                             "not real telemetry — rename or replace once real "
                             "bearing data is available.")
    parser.add_argument("--mode", choices=["estimate", "constant", "linear"],
                        default="estimate",
                        help="estimate=feature-based; constant=fixed; "
                             "linear=evenly interpolated")
    parser.add_argument("--initial-bearing", type=float, default=0.0,
                        help="Starting bearing in degrees (0=N, 90=E, ...)")
    parser.add_argument("--linear-end", type=float, default=None,
                        help="End bearing for linear mode (default: initial+360)")
    parser.add_argument("--hfov", type=float, default=78.0,
                        help="Camera horizontal field-of-view in degrees")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Downscale factor for speed (e.g. 2.0 = half size)")
    parser.add_argument("--yaw-frame", choices=["NED", "ENU"], default="NED",
                        help="Convention for the yaw_deg column. NED (drone "
                             "default) treats bearing and yaw as the same axis; "
                             "ENU (ROS/math) flips and offsets by 90 deg.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir).expanduser()
    frames = list_frames(frames_dir)
    if not frames:
        print(f"No image frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(frames)} frames in {frames_dir}")
    print(f"Mode: {args.mode}")

    if args.mode == "constant":
        rows = build_rows_constant(frames, args.initial_bearing)
    elif args.mode == "linear":
        end = args.linear_end if args.linear_end is not None else args.initial_bearing + 360.0
        rows = build_rows_linear(frames, args.initial_bearing, end)
    else:
        rows = build_rows_estimate(frames, args.initial_bearing,
                                   args.hfov, args.downscale)

    out_path = Path(args.output).expanduser()
    write_csv(rows, out_path, yaw_frame=args.yaw_frame)
    print(f"Wrote bearings for {len(rows)} frames to {out_path}")


if __name__ == "__main__":
    main()