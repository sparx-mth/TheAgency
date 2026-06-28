#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import yaml

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel


Roi = Tuple[float, float, float, float]  # x, y, w, h


from sparx_agency.robots.common.helpers import load_intrinsics_from_yaml as _load_intrinsics

def load_intrinsics_from_yaml(path: str, prefer_projection: bool = True):
    fx, fy, cx, cy = _load_intrinsics(path, prefer_projection=prefer_projection)
    return fx, fy, cx, cy, 0.5 * (fx + fy)


def list_images(image_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])

    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")

    return paths


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()

    raise RuntimeError(
        "No OpenCV tracker found. Install opencv-contrib-python or use --fixed-roi."
    )


def pad_720_to_728(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    if (h, w) != (420, 720):
        raise ValueError(f"Expected image shape (420, 720), got {(h, w)}")

    return cv2.copyMakeBorder(
        image_bgr,
        0,
        0,
        4,
        4,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def crop_728_to_720(depth_728: np.ndarray) -> np.ndarray:
    if depth_728.shape[:2] != (420, 728):
        raise ValueError(f"Expected depth shape (420, 728), got {depth_728.shape[:2]}")

    return depth_728[:, 4:724].astype(np.float32)


def resize_depth_to_rgb(depth: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    rgb_h, rgb_w = image_bgr.shape[:2]

    if depth.shape[:2] != (rgb_h, rgb_w):
        depth = cv2.resize(
            depth,
            (rgb_w, rgb_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return depth.astype(np.float32)


def robust_depth_stats(
    depth: np.ndarray,
    roi: Roi,
    min_value: float,
    max_value: float,
) -> Optional[Dict[str, Any]]:
    x, y, w, h = roi

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(depth.shape[1], int(round(x + w)))
    y2 = min(depth.shape[0], int(round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth[y1:y2, x1:x2]

    valid = (
        np.isfinite(patch)
        & (patch > min_value)
        & (patch < max_value)
    )

    vals = patch[valid]

    if vals.size < 20:
        return None

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "std": float(np.std(vals)),
        "n": int(vals.size),
    }


def depth_to_vis(depth: np.ndarray, roi: Optional[Roi] = None, label: str = ""):
    valid = np.isfinite(depth)
    vis_gray = np.zeros(depth.shape, dtype=np.uint8)

    if np.any(valid):
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        vis_gray = (norm * 255).astype(np.uint8)

    vis = cv2.applyColorMap(vis_gray, cv2.COLORMAP_TURBO)

    if roi is not None:
        x, y, w, h = [int(round(v)) for v in roi]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if label:
            cv2.putText(
                vis,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return vis


def draw_rgb_roi(image_bgr: np.ndarray, roi: Roi, label: str):
    out = image_bgr.copy()

    x, y, w, h = [int(round(v)) for v in roi]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        out,
        label,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def infer_large_meters(
    da3_large: DA3TensorRTModel,
    image_bgr: np.ndarray,
    focal_px: float,
    large_uses_728_width: bool,
    large_raw_needs_focal_scale: bool,
):
    if large_uses_728_width:
        image_728 = pad_720_to_728(image_bgr)
        large_raw_728, _ = da3_large.infer_all(image_728)
        large_raw = crop_728_to_720(large_raw_728)
    else:
        large_raw, _ = da3_large.infer_all(image_bgr)
        large_raw = resize_depth_to_rgb(large_raw, image_bgr)

    if large_raw_needs_focal_scale:
        large_m = focal_px * large_raw / 300.0
    else:
        large_m = large_raw

    return large_m.astype(np.float32)


def infer_small_raw(
    da3_small: DA3TensorRTModel,
    image_bgr: np.ndarray,
):
    small_raw, _ = da3_small.infer_all(image_bgr)
    small_raw = resize_depth_to_rgb(small_raw, image_bgr)
    return small_raw.astype(np.float32)


def parse_fixed_roi(value: str) -> Roi:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--fixed-roi must be x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--camera-yaml", required=True)
    parser.add_argument("--large-engine", required=True)
    parser.add_argument("--small-engine", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--fixed-roi", default=None, help="Optional ROI as x,y,w,h")
    parser.add_argument("--prefer-projection", action="store_true")

    parser.add_argument(
        "--large-uses-728-width",
        action="store_true",
        help="Use this for DA3METRIC-LARGE 420x728 engine with 720x420 RGB frames.",
    )

    parser.add_argument(
        "--large-raw-needs-focal-scale",
        action="store_true",
        help="Use this if large infer_all() returns raw DA3 output, not meters.",
    )

    parser.add_argument("--min-large-m", type=float, default=0.2)
    parser.add_argument("--max-large-m", type=float, default=30.0)
    parser.add_argument("--min-small-raw", type=float, default=0.0)
    parser.add_argument("--max-small-raw", type=float, default=100.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    debug_dir = out_dir / "debug"
    npy_dir = out_dir / "npy"

    debug_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.image_dir)

    fx, fy, cx, cy, focal = load_intrinsics_from_yaml(
        args.camera_yaml,
        prefer_projection=args.prefer_projection,
    )

    print(f"Loaded {len(image_paths)} images")
    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, focal={focal:.3f}")

    print("Loading DA3METRIC-LARGE...")
    da3_large = DA3TensorRTModel(
        engine_path=args.large_engine,
        yaml_path=args.camera_yaml,
    )

    print("Loading DA3-SMALL...")
    da3_small = DA3TensorRTModel(
        engine_path=args.small_engine,
        yaml_path=args.camera_yaml,
    )

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    if first.shape[:2] != (420, 720):
        print(f"[WARN] Expected RGB shape (420, 720), got {first.shape[:2]}")

    if args.fixed_roi is not None:
        roi = parse_fixed_roi(args.fixed_roi)
        tracker = None
        print(f"Using fixed ROI: {roi}")
    else:
        print("Select the door ROI, then press ENTER or SPACE.")
        roi = cv2.selectROI(
            "select_door_roi",
            first,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow("select_door_roi")

        if roi[2] <= 0 or roi[3] <= 0:
            raise RuntimeError("Invalid ROI selected")

        tracker = create_tracker()
        tracker.init(first, tuple(roi))

    csv_path = out_dir / "door_depth_compare.csv"
    rows = []

    for idx, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Skipping unreadable image: {img_path}")
            continue

        if tracker is None:
            ok = True
            tracked_roi = roi
        elif idx == 0:
            ok = True
            tracked_roi = roi
        else:
            ok, tracked_roi = tracker.update(image_bgr)

        if not ok:
            print(f"[WARN] Tracker failed at frame {idx}: {img_path.name}")
            continue

        tracked_roi = tuple(float(v) for v in tracked_roi)

        large_m = infer_large_meters(
            da3_large=da3_large,
            image_bgr=image_bgr,
            focal_px=focal,
            large_uses_728_width=args.large_uses_728_width,
            large_raw_needs_focal_scale=args.large_raw_needs_focal_scale,
        )

        small_raw = infer_small_raw(
            da3_small=da3_small,
            image_bgr=image_bgr,
        )

        large_stats = robust_depth_stats(
            large_m,
            tracked_roi,
            min_value=args.min_large_m,
            max_value=args.max_large_m,
        )

        small_stats = robust_depth_stats(
            small_raw,
            tracked_roi,
            min_value=args.min_small_raw,
            max_value=args.max_small_raw,
        )

        if large_stats is None or small_stats is None:
            print(f"[WARN] Invalid depth stats at frame {idx}: {img_path.name}")
            continue

        scale_mean = large_stats["mean"] / max(small_stats["mean"], 1e-6)
        scale_median = large_stats["median"] / max(small_stats["median"], 1e-6)

        x, y, w, h = tracked_roi

        row = {
            "frame_idx": idx,
            "image_name": img_path.name,

            "roi_x": x,
            "roi_y": y,
            "roi_w": w,
            "roi_h": h,

            "large_m_mean": large_stats["mean"],
            "large_m_median": large_stats["median"],
            "large_m_p10": large_stats["p10"],
            "large_m_p90": large_stats["p90"],
            "large_m_std": large_stats["std"],
            "large_valid_n": large_stats["n"],

            "small_raw_mean": small_stats["mean"],
            "small_raw_median": small_stats["median"],
            "small_raw_p10": small_stats["p10"],
            "small_raw_p90": small_stats["p90"],
            "small_raw_std": small_stats["std"],
            "small_valid_n": small_stats["n"],

            "small_to_large_scale_mean": float(scale_mean),
            "small_to_large_scale_median": float(scale_median),
        }

        rows.append(row)

        if idx % args.save_every == 0:
            stem = f"{idx:06d}_{img_path.stem}"

            rgb_vis = draw_rgb_roi(
                image_bgr,
                tracked_roi,
                (
                    f"L={large_stats['median']:.2f}m "
                    f"S={small_stats['median']:.3f} "
                    f"scale={scale_median:.3f}"
                ),
            )

            large_vis = depth_to_vis(
                large_m,
                tracked_roi,
                f"large={large_stats['median']:.2f}m",
            )

            small_vis = depth_to_vis(
                small_raw,
                tracked_roi,
                f"small={small_stats['median']:.3f}",
            )

            cv2.imwrite(str(debug_dir / f"{stem}_rgb_roi.jpg"), rgb_vis)
            cv2.imwrite(str(debug_dir / f"{stem}_large_depth_roi.jpg"), large_vis)
            cv2.imwrite(str(debug_dir / f"{stem}_small_depth_roi.jpg"), small_vis)

            np.save(str(npy_dir / f"{stem}_large_m.npy"), large_m)
            np.save(str(npy_dir / f"{stem}_small_raw.npy"), small_raw)

        print(
            f"[{idx:04d}] {img_path.name} "
            f"L={large_stats['median']:.3f}m "
            f"S={small_stats['median']:.4f} "
            f"scale={scale_median:.4f}"
        )

    if not rows:
        raise RuntimeError("No valid rows were collected")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    scales = np.array(
        [r["small_to_large_scale_median"] for r in rows],
        dtype=np.float32,
    )

    large_vals = np.array(
        [r["large_m_median"] for r in rows],
        dtype=np.float32,
    )

    small_vals = np.array(
        [r["small_raw_median"] for r in rows],
        dtype=np.float32,
    )

    global_scale = float(np.median(scales))
    small_scaled = small_vals * global_scale
    residual = small_scaled - large_vals

    summary_path = out_dir / "summary.txt"

    with open(summary_path, "w") as f:
        f.write(f"frames_used: {len(rows)}\n")
        f.write(f"global_scale_median: {global_scale:.8f}\n")
        f.write(f"scale_mean: {float(np.mean(scales)):.8f}\n")
        f.write(f"scale_std: {float(np.std(scales)):.8f}\n")
        f.write(
            f"scale_cv_percent: "
            f"{float(100.0 * np.std(scales) / max(np.mean(scales), 1e-6)):.3f}\n"
        )
        f.write(
            f"scaled_small_vs_large_mae_m: "
            f"{float(np.mean(np.abs(residual))):.8f}\n"
        )
        f.write(
            f"scaled_small_vs_large_rmse_m: "
            f"{float(np.sqrt(np.mean(residual ** 2))):.8f}\n"
        )

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Debug images: {debug_dir}")
    print(f"NPY files: {npy_dir}")
    print(f"Summary: {summary_path}")
    print(f"Global small scale median: {global_scale:.6f}")
    print(f"Scale std: {float(np.std(scales)):.6f}")
    print(
        f"Scale CV %: "
        f"{float(100.0 * np.std(scales) / max(np.mean(scales), 1e-6)):.2f}"
    )


if __name__ == "__main__":
    main()