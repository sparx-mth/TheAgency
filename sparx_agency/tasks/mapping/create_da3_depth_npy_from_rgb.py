#!/usr/bin/env python3
"""
Create DA3 depth .npy files from an existing RGB image directory.

This is the offline equivalent of take_xtend_da3_frames.py:
  - no RTSP
  - no XTEND websocket telemetry
  - reads existing RGB images
  - runs DA3 on every image
  - saves depth_npy/<same_stem>.npy as float32 meters/raw model output, depending on model wrapper
  - optionally saves depth_vis/<same_stem>.png

Example:
python3 create_da3_depth_npy_from_rgb.py \
  --rgb-dir /home/user/jetson-containers/data/R1/2026_05_05___18_42_33 \
  --output-dir /home/user/Desktop/xtend_rectified_depth_take_20260503 \
  --engine-path /home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine \
  --config-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth_clean = np.nan_to_num(depth_m.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    depth_clipped = np.clip(depth_clean, 0.0, max_depth_m)
    depth_norm = (depth_clipped / max(max_depth_m, 1e-6) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)


def list_images(rgb_dir: Path, recursive: bool, image_glob: str) -> list[Path]:
    if recursive:
        candidates = sorted(rgb_dir.rglob(image_glob))
    else:
        candidates = sorted(rgb_dir.glob(image_glob))

    images = [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return images


def finite_stats(depth: np.ndarray) -> tuple[float, float, float]:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite)), float(np.mean(finite))


def safe_relative_stem(path: Path, root: Path) -> Path:
    """Return relative path without suffix, preserving subfolders in recursive mode."""
    rel = path.relative_to(root)
    return rel.with_suffix("")


def main() -> None:
    args = parse_args()

    rgb_dir = Path(args.rgb_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    depth_npy_dir = output_dir / args.depth_npy_subdir
    depth_vis_dir = output_dir / args.depth_vis_subdir
    copied_rgb_dir = output_dir / args.rgb_subdir

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir.mkdir(parents=True, exist_ok=True)

    if args.save_depth_vis:
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_rgb:
        copied_rgb_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(rgb_dir=rgb_dir, recursive=args.recursive, image_glob=args.image_glob)
    if not images:
        raise RuntimeError(f"No RGB images found in {rgb_dir} with glob {args.image_glob}")

    print(f"[offline-da3] RGB dir       : {rgb_dir}")
    print(f"[offline-da3] Output dir    : {output_dir}")
    print(f"[offline-da3] Depth NPY dir : {depth_npy_dir}")
    print(f"[offline-da3] Num images    : {len(images)}")
    print(f"[offline-da3] Engine        : {args.engine_path}")
    print(f"[offline-da3] Camera YAML   : {args.config_yaml}")

    depth_model = DA3TensorRTModel(
        engine_path=args.engine_path,
        yaml_path=args.config_yaml,
    )

    metadata_csv_path = output_dir / args.metadata_csv
    metadata_jsonl_path = output_dir / args.metadata_jsonl

    with open(metadata_csv_path, "w", newline="") as csv_fp, open(metadata_jsonl_path, "w") as jsonl_fp:
        writer = csv.writer(csv_fp)
        writer.writerow([
            "frame_idx",
            "rgb_path",
            "depth_npy_path",
            "depth_vis_path",
            "rgb_height",
            "rgb_width",
            "depth_height",
            "depth_width",
            "depth_min",
            "depth_max",
            "depth_mean",
            "elapsed_sec",
        ])

        for frame_idx, image_path in enumerate(images):
            rel_stem = safe_relative_stem(image_path, rgb_dir)
            depth_npy_path = depth_npy_dir / f"{rel_stem}.npy"
            depth_vis_path = depth_vis_dir / f"{rel_stem}.png"
            copied_rgb_path = copied_rgb_dir / image_path.relative_to(rgb_dir) if args.copy_rgb else None

            depth_npy_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_depth_vis:
                depth_vis_path.parent.mkdir(parents=True, exist_ok=True)
            if copied_rgb_path is not None:
                copied_rgb_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and depth_npy_path.exists():
                print(f"[offline-da3] skip existing {depth_npy_path}")
                continue

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[offline-da3][WARN] failed to read image: {image_path}")
                continue

            t0 = time.time()
            depth = depth_model.infer_depth(bgr).astype(np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            if args.clip_max_depth_m > args.clip_min_depth_m:
                depth = np.clip(depth, args.clip_min_depth_m, args.clip_max_depth_m).astype(np.float32)

            np.save(str(depth_npy_path), depth)

            depth_vis_str = ""
            if args.save_depth_vis:
                depth_vis = colorize_depth(depth, max_depth_m=args.max_depth_vis_m)
                cv2.imwrite(str(depth_vis_path), depth_vis)
                depth_vis_str = str(depth_vis_path)

            if copied_rgb_path is not None:
                shutil.copy2(str(image_path), str(copied_rgb_path))

            elapsed = time.time() - t0
            depth_min, depth_max, depth_mean = finite_stats(depth)

            row = [
                frame_idx,
                str(image_path),
                str(depth_npy_path),
                depth_vis_str,
                int(bgr.shape[0]),
                int(bgr.shape[1]),
                int(depth.shape[0]),
                int(depth.shape[1]),
                depth_min,
                depth_max,
                depth_mean,
                elapsed,
            ]
            writer.writerow(row)
            csv_fp.flush()

            jsonl_fp.write(json.dumps({
                "frame_idx": frame_idx,
                "image": image_path.name,
                "rgb_path": str(image_path),
                "depth_npy_path": str(depth_npy_path),
                "depth_vis_path": depth_vis_str,
                "rgb_shape_hw": [int(bgr.shape[0]), int(bgr.shape[1])],
                "depth_shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
                "depth_min": depth_min,
                "depth_max": depth_max,
                "depth_mean": depth_mean,
                "elapsed_sec": elapsed,
            }) + "\n")
            jsonl_fp.flush()

            print(
                f"[offline-da3] {frame_idx + 1:06d}/{len(images):06d} "
                f"{image_path.name} -> {depth_npy_path.name} "
                f"depth={depth.shape[1]}x{depth.shape[0]} "
                f"mean={depth_mean:.3f} elapsed={elapsed:.3f}s"
            )

    print(f"[offline-da3] Done.")
    print(f"[offline-da3] Metadata CSV  : {metadata_csv_path}")
    print(f"[offline-da3] Metadata JSONL: {metadata_jsonl_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run DA3 on an existing RGB image directory and save depth .npy files with matching names."
    )

    p.add_argument(
        "--rgb-dir",
        required=True,
        help="Directory containing RGB images. Example: /data/R1/2026_05_05___18_42_33",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. For FALCON, use something like xtend_rectified_depth_take_YYYYMMDD.",
    )
    p.add_argument(
        "--engine-path",
        required=True,
        help="Path to DA3 TensorRT .engine file.",
    )
    p.add_argument(
        "--config-yaml",
        required=True,
        help="Path to camera intrinsics YAML used by DA3TensorRTModel.",
    )

    p.add_argument(
        "--image-glob",
        default="*",
        help="Image glob inside rgb-dir. Use '*.jpg' for only JPG files.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search RGB images recursively and preserve subfolder structure under depth_npy.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose target .npy already exists.",
    )

    p.add_argument(
        "--depth-npy-subdir",
        default="depth_npy",
        help="Subdirectory under output-dir for .npy files.",
    )
    p.add_argument(
        "--save-depth-vis",
        action="store_true",
        help="Also save colored debug depth images.",
    )
    p.add_argument(
        "--depth-vis-subdir",
        default="depth_vis",
        help="Subdirectory under output-dir for colored depth debug images.",
    )
    p.add_argument(
        "--max-depth-vis-m",
        type=float,
        default=15.0,
        help="Max depth used only for visualization color scaling.",
    )
    p.add_argument(
        "--copy-rgb",
        action="store_true",
        help="Copy RGB images into output-dir/rgb so the recording folder is self-contained.",
    )
    p.add_argument(
        "--rgb-subdir",
        default="rgb",
        help="Subdirectory under output-dir used when --copy-rgb is enabled.",
    )

    p.add_argument(
        "--clip-min-depth-m",
        type=float,
        default=0.0,
        help="Minimum depth clamp. Set >= max to disable clipping.",
    )
    p.add_argument(
        "--clip-max-depth-m",
        type=float,
        default=20.0,
        help="Maximum depth clamp. Set <= min to disable clipping.",
    )

    p.add_argument(
        "--metadata-csv",
        default="depth_metadata.csv",
        help="CSV metadata filename written under output-dir.",
    )
    p.add_argument(
        "--metadata-jsonl",
        default="depth_metadata.jsonl",
        help="JSONL metadata filename written under output-dir.",
    )

    return p.parse_args()


if __name__ == "__main__":
    main()
