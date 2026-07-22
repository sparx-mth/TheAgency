#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, Sequence

import cv2
import numpy as np


Roi = Tuple[int, int, int, int]  # x, y, w, h


def parse_distance_from_dir_name(name: str) -> float:
    """
    Supports:
      4_0          -> 4.0
      4_0_seg00    -> 4.0
      2_5_seg08    -> 2.5
      0_5_seg12    -> 0.5
    """
    m = re.match(r"^(\d+)_(\d+)", name)

    if m is None:
        raise ValueError(f"Could not parse distance from folder name: {name}")

    return float(f"{m.group(1)}.{m.group(2)}")


def list_distance_dirs(root_dir: Path):
    dirs = []

    for p in root_dir.iterdir():
        if not p.is_dir():
            continue

        if re.fullmatch(r"\d+_\d+(?:_seg\d+)?", p.name):
            dirs.append(p)

    dirs.sort(key=lambda d: parse_distance_from_dir_name(d.name), reverse=True)
    return dirs


def find_matching_depth_npy(image_path: Path, rgb_root: Path, depth_root: Path) -> Optional[Path]:
    """
    For:
      rgb_root/3_5/frame_001.jpg

    Find:
      depth_root/3_5/frame_001.npy

    Also tries common suffix variants.
    """
    rel = image_path.relative_to(rgb_root)
    depth_folder = depth_root / rel.parent
    stem = image_path.stem

    candidates = [
        depth_folder / f"{stem}.npy",
        depth_folder / f"{stem}_small.npy",
        depth_folder / f"{stem}_depth.npy",
        depth_folder / f"{stem}_depth_small.npy",
        depth_folder / f"{stem}_small_raw.npy",
    ]

    for c in candidates:
        if c.exists():
            return c

    # Fallback: if there is exactly one npy with same numeric index-ish stem
    npys = sorted(depth_folder.glob("*.npy"))
    if len(npys) == 1:
        return npys[0]

    return None


def list_rgb_images(folder: Path):
    """
    Lists RGB images inside one distance folder.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
    )

def detect_chessboard_roi(
    image_bgr: np.ndarray,
    pattern_sizes: Sequence[Tuple[int, int]] = ((8, 5), (5, 8)),
    expand_ratio: float = 0.35,
    min_roi_area: int = 100,
    use_sb_detector: bool = True,
) -> Optional[Roi]:
    """
    Detect chessboard and return expanded ROI.

    OpenCV pattern size is INNER corners, not squares.
    A 9x6 squares board has 8x5 inner corners.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    best_corners = None

    for pattern_size in pattern_sizes:
        found = False
        corners = None

        if use_sb_detector and hasattr(cv2, "findChessboardCornersSB"):
            flags = (
                cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
            )
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=flags,
            )

        if not found:
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=flags,
            )

            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    winSize=(5, 5),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )

        if found and corners is not None:
            best_corners = corners.reshape(-1, 2)
            break

    if best_corners is None:
        return None

    x_min = float(np.min(best_corners[:, 0]))
    y_min = float(np.min(best_corners[:, 1]))
    x_max = float(np.max(best_corners[:, 0]))
    y_max = float(np.max(best_corners[:, 1]))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w * box_h < min_roi_area:
        return None

    pad_x = box_w * expand_ratio
    pad_y = box_h * expand_ratio

    x1 = int(round(max(0, x_min - pad_x)))
    y1 = int(round(max(0, y_min - pad_y)))
    x2 = int(round(min(img_w - 1, x_max + pad_x)))
    y2 = int(round(min(img_h - 1, y_max + pad_y)))

    roi_w = x2 - x1 + 1
    roi_h = y2 - y1 + 1

    if roi_w <= 0 or roi_h <= 0:
        return None

    return x1, y1, roi_w, roi_h


def resize_depth_to_rgb(depth: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    rgb_h, rgb_w = image_bgr.shape[:2]

    if depth.shape[:2] != (rgb_h, rgb_w):
        depth = cv2.resize(
            depth.astype(np.float32),
            (rgb_w, rgb_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return depth.astype(np.float32)


def depth_stats_in_roi(
    depth: np.ndarray,
    roi: Roi,
    min_depth: float,
    max_depth: float,
) -> Optional[dict]:
    x, y, w, h = roi

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(depth.shape[1], int(x + w))
    y2 = min(depth.shape[0], int(y + h))

    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth[y1:y2, x1:x2]

    valid = (
        np.isfinite(patch)
        & (patch > min_depth)
        & (patch < max_depth)
    )

    vals = patch[valid]

    if vals.size < 10:
        return None

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "std": float(np.std(vals)),
        "n": int(vals.size),
    }


def draw_roi(image_bgr: np.ndarray, roi: Roi, text: str) -> np.ndarray:
    out = image_bgr.copy()
    x, y, w, h = roi

    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        out,
        text,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return out

def select_manual_roi_for_folder(folder_name: str, image_bgr: np.ndarray) -> Optional[Roi]:
    """
    Opens an OpenCV window and lets the user manually drag an ROI.

    Controls:
      - Drag rectangle with mouse
      - Press ENTER or SPACE to accept
      - Press ESC or C to cancel
    """
    win_name = f"Select ROI for {folder_name}"

    display = image_bgr.copy()
    cv2.putText(
        display,
        f"Folder {folder_name}: drag ROI, ENTER/SPACE=accept, ESC/C=cancel",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    roi = cv2.selectROI(
        win_name,
        display,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow(win_name)

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def depth_to_vis(depth: np.ndarray, roi: Optional[Roi] = None, text: str = "") -> np.ndarray:
    valid = np.isfinite(depth)
    vis_gray = np.zeros(depth.shape[:2], dtype=np.uint8)

    if np.any(valid):
        vals = depth[valid]
        lo, hi = np.percentile(vals, [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        vis_gray = (norm * 255).astype(np.uint8)

    vis = cv2.applyColorMap(vis_gray, cv2.COLORMAP_TURBO)

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if text:
            cv2.putText(
                vis,
                text,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return vis


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--out-dir", required=True)


    parser.add_argument("--save-debug-every", type=int, default=10)
    parser.add_argument("--expand-ratio", type=float, default=0.35)

    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=100.0)

    parser.add_argument(
        "--manual-roi-file",
        default="",
        help="Optional CSV file with manual ROIs: folder,x,y,w,h",
    )

    parser.add_argument(
        "--reuse-folder-roi",
        action="store_true",
        help="Detect chessboard once per distance folder and reuse ROI for all frames in that folder.",
    )

    parser.add_argument(
        "--manual-roi-on-fail",
        action="store_true",
        help="If chessboard detection fails, open a window and let user select ROI once per folder.",
    )

    parser.add_argument(
        "--force-manual-roi",
        action="store_true",
        help="Always open a window and select ROI manually once per folder/segment.",
    )

    args = parser.parse_args()

    rgb_root = Path(args.rgb_root).expanduser()
    depth_root = Path(args.depth_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    debug_dir = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    distance_dirs = list_distance_dirs(rgb_root)

    if not distance_dirs:
        raise RuntimeError(f"No distance folders found in {rgb_root}")

    rows = []
    failures = []

    for rgb_dist_dir in distance_dirs:
        dist_name = rgb_dist_dir.name
        gt_m = parse_distance_from_dir_name(dist_name)

        depth_dist_dir = depth_root / dist_name
        if not depth_dist_dir.exists():
            print(f"[WARN] Missing depth folder: {depth_dist_dir}")
            continue

        image_paths = list_rgb_images(rgb_dist_dir)

        if not image_paths:
            print(f"[WARN] No RGB images in {rgb_dist_dir}")
            continue

        folder_roi = None

        print(f"\n=== {dist_name} | gt={gt_m:.2f}m | images={len(image_paths)} ===")

        for i, img_path in enumerate(image_paths):
            image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            if image_bgr is None:
                failures.append((dist_name, img_path.name, "failed_read_rgb"))
                continue

            if args.reuse_folder_roi and folder_roi is not None:
                roi = folder_roi

            elif args.force_manual_roi:
                print(f"[{dist_name}] Opening manual ROI window on {img_path.name}...")
                roi = select_manual_roi_for_folder(dist_name, image_bgr)

                if roi is None:
                    failures.append((dist_name, img_path.name, "manual_roi_cancelled"))
                    print(f"[{dist_name}] Manual ROI cancelled. Skipping this folder.")
                    break

                print(f"[{dist_name}] Manual ROI selected: {roi}")

                if args.reuse_folder_roi:
                    folder_roi = roi

            else:
                roi = detect_chessboard_roi(
                    image_bgr,
                    pattern_sizes=((8, 5), (5, 8)),
                    expand_ratio=args.expand_ratio,
                )

                if roi is None:
                    if args.manual_roi_on_fail:
                        print(f"[{dist_name}] Chessboard detection failed on {img_path.name}")
                        print(f"[{dist_name}] Opening manual ROI window...")

                        roi = select_manual_roi_for_folder(dist_name, image_bgr)

                        if roi is None:
                            failures.append((dist_name, img_path.name, "manual_roi_cancelled"))
                            print(f"[{dist_name}] Manual ROI cancelled. Skipping this folder.")
                            break

                        print(f"[{dist_name}] Manual ROI selected: {roi}")

                        if args.reuse_folder_roi:
                            folder_roi = roi
                    else:
                        failures.append((dist_name, img_path.name, "failed_detect_chessboard"))
                        continue
                else:
                    if args.reuse_folder_roi:
                        folder_roi = roi

            depth_path = find_matching_depth_npy(
                image_path=img_path,
                rgb_root=rgb_root,
                depth_root=depth_root,
            )

            if depth_path is None:
                failures.append((dist_name, img_path.name, "missing_depth_npy"))
                continue

            depth = np.load(str(depth_path))
            depth = resize_depth_to_rgb(depth, image_bgr)

            stats = depth_stats_in_roi(
                depth,
                roi,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )

            if stats is None:
                failures.append((dist_name, img_path.name, "invalid_depth_roi"))
                continue

            x, y, w, h = roi

            row = {
                "distance_folder": dist_name,
                "gt_m": gt_m,
                "image_name": img_path.name,
                "depth_name": depth_path.name,

                "rgb_width": image_bgr.shape[1],
                "rgb_height": image_bgr.shape[0],
                "depth_width": depth.shape[1],
                "depth_height": depth.shape[0],

                "roi_x": x,
                "roi_y": y,
                "roi_w": w,
                "roi_h": h,

                "small_raw_mean": stats["mean"],
                "small_raw_median": stats["median"],
                "small_raw_p10": stats["p10"],
                "small_raw_p90": stats["p90"],
                "small_raw_std": stats["std"],
                "small_raw_n": stats["n"],

                "small_to_gt_scale_mean": gt_m / max(stats["mean"], 1e-6),
                "small_to_gt_scale_median": gt_m / max(stats["median"], 1e-6),
            }

            rows.append(row)

            if args.save_debug_every > 0 and i % args.save_debug_every == 0:
                stem = f"{dist_name}_{img_path.stem}"

                rgb_debug = draw_roi(
                    image_bgr,
                    roi,
                    f"gt={gt_m:.1f}m raw={stats['median']:.3f}",
                )

                depth_debug = depth_to_vis(
                    depth,
                    roi,
                    f"raw={stats['median']:.3f}",
                )

                cv2.imwrite(str(debug_dir / f"{stem}_rgb_roi.jpg"), rgb_debug)
                cv2.imwrite(str(debug_dir / f"{stem}_depth_roi.jpg"), depth_debug)

            print(
                f"[{dist_name}] {img_path.name} "
                f"raw_med={stats['median']:.4f} "
                f"scale={row['small_to_gt_scale_median']:.4f} "
                f"roi={roi}"
            )

    if not rows:
        raise RuntimeError("No valid samples collected")

    csv_path = out_dir / "small_depth_chessboard_calibration.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fail_path = out_dir / "failures.csv"
    with open(fail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance_folder", "image_name", "reason"])
        writer.writerows(failures)

    gt = np.array([r["gt_m"] for r in rows], dtype=np.float64)
    small = np.array([r["small_raw_median"] for r in rows], dtype=np.float64)

    # Constant scale through origin: gt ~= k * small
    k = float(np.sum(small * gt) / max(np.sum(small * small), 1e-12))
    pred_const = k * small

    # Linear: gt ~= a * small + b
    a, b = np.polyfit(small, gt, deg=1)
    pred_linear = a * small + b

    # Quadratic: gt ~= qa * small^2 + qb * small + qc
    qa, qb, qc = np.polyfit(small, gt, deg=2)
    pred_quad = qa * small * small + qb * small + qc

    def err_stats(pred):
        residual = pred - gt
        return {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "max_abs": float(np.max(np.abs(residual))),
        }

    const_err = err_stats(pred_const)
    linear_err = err_stats(pred_linear)
    quad_err = err_stats(pred_quad)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"samples_used: {len(rows)}\n")
        f.write(f"failures: {len(failures)}\n\n")

        f.write("constant_scale_model:\n")
        f.write(f"  gt_m = {k:.8f} * small_raw\n")
        f.write(f"  mae_m = {const_err['mae']:.8f}\n")
        f.write(f"  rmse_m = {const_err['rmse']:.8f}\n")
        f.write(f"  max_abs_m = {const_err['max_abs']:.8f}\n\n")

        f.write("linear_model:\n")
        f.write(f"  gt_m = {a:.8f} * small_raw + {b:.8f}\n")
        f.write(f"  mae_m = {linear_err['mae']:.8f}\n")
        f.write(f"  rmse_m = {linear_err['rmse']:.8f}\n")
        f.write(f"  max_abs_m = {linear_err['max_abs']:.8f}\n\n")

        f.write("quadratic_model:\n")
        f.write(f"  gt_m = {qa:.8f} * small_raw^2 + {qb:.8f} * small_raw + {qc:.8f}\n")
        f.write(f"  mae_m = {quad_err['mae']:.8f}\n")
        f.write(f"  rmse_m = {quad_err['rmse']:.8f}\n")
        f.write(f"  max_abs_m = {quad_err['max_abs']:.8f}\n")

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"Failures: {fail_path}")
    print(f"Debug: {debug_dir}")
    print(f"Summary: {summary_path}")
    print(f"Constant scale: gt_m = {k:.6f} * small_raw")
    print(f"Linear: gt_m = {a:.6f} * small_raw + {b:.6f}")
    print(f"Quadratic: gt_m = {qa:.6f} * small_raw^2 + {qb:.6f} * small_raw + {qc:.6f}")


if __name__ == "__main__":
    main()