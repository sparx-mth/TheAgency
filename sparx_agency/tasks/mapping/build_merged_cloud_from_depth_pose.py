#!/usr/bin/env python3
"""
Build a merged point cloud from raw RGB images, raw depth .npy files, pose JSON files,
camera calibration YAML, and a LUT .npz converting raw model depth to metric meters.

Main use case:
  - RGB raw image: 720x420
  - center crop: 540x420
  - resize: 504x392
  - depth .npy: 504x392
  - pose: x/y/z/yaw per image

Example:
  python3 build_merged_cloud_from_depth_pose_v2.py \
    --data-root mamad.zip \
    --calib-yaml calib_small_depth.yaml \
    --lut-npz lut_small_depth.npz \
    --out-dir mamad_outputs \
    --crop-width 540 \
    --crop-height 420 \
    --output-width 504 \
    --output-height 392 \
    --pixel-step 4 \
    --voxel-size 0.03

Useful yaw tests:
  --yaw-offset-deg 90
  --yaw-offset-deg -90
  --yaw-offset-deg 180
  --yaw-sign -1

Useful local-window tests:
  --start-frame 10 --end-frame 20
  --start-frame 10 --max-frames 5

Useful translation-noise tests:
  --translation-scale 0.0   # pure rotation around the first pose
  --translation-scale 0.5   # damp reported XY/Z translation
  --translation-scale 1.0   # original translation

Useful coloring tests:
  --color-mode frame         # each frame gets a different color
  --color-mode depth         # pseudo-color by metric depth
  --save-per-frame-clouds    # save one PLY per frame for debugging
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def center_crop_resize(
    frame: np.ndarray,
    crop_width: int,
    crop_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    h, w = frame.shape[:2]

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop_width and crop_height must be positive")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_width and output_height must be positive")
    if crop_width > w or crop_height > h:
        raise ValueError(f"Crop {crop_width}x{crop_height} is larger than frame {w}x{h}")

    x0 = (w - crop_width) // 2
    y0 = (h - crop_height) // 2
    cropped = frame[y0:y0 + crop_height, x0:x0 + crop_width]
    return cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def load_intrinsics(calib_yaml: Path, use_projection_matrix: bool = False) -> tuple[float, float, float, float, int, int]:
    with calib_yaml.open("r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    width = int(calib.get("image_width", calib.get("width")))
    height = int(calib.get("image_height", calib.get("height")))

    if use_projection_matrix and "projection_matrix" in calib:
        p = calib["projection_matrix"]["data"]
        fx = float(p[0])
        fy = float(p[5])
        cx = float(p[2])
        cy = float(p[6])
        source = "projection_matrix"
    elif "camera_matrix" in calib:
        k = calib["camera_matrix"]["data"]
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        source = "camera_matrix"
    else:
        fx = float(calib["fx"])
        fy = float(calib["fy"])
        cx = float(calib["cx"])
        cy = float(calib["cy"])
        source = "flat_fx_fy_cx_cy"

    print(f"[INFO] intrinsics source: {source}")
    return fx, fy, cx, cy, width, height


def load_lut(lut_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    lut = np.load(lut_npz)
    if "raw" not in lut or "meters" not in lut:
        raise KeyError(f"LUT must contain arrays named 'raw' and 'meters'. Found keys: {list(lut.keys())}")

    raw = lut["raw"].astype(np.float32)
    meters = lut["meters"].astype(np.float32)

    order = np.argsort(raw)
    raw_sorted = raw[order]
    meters_sorted = meters[order]

    return raw_sorted, meters_sorted


def raw_depth_to_meters(depth_raw: np.ndarray, raw_lut: np.ndarray, meters_lut: np.ndarray) -> np.ndarray:
    # Keep this simple for now. For production, replace with monotonic fit if the LUT is noisy.
    d = np.asarray(depth_raw, dtype=np.float32)
    d_clip = np.clip(d, raw_lut[0], raw_lut[-1])
    return np.interp(d_clip, raw_lut, meters_lut).astype(np.float32)


def yaw_bearing_to_vectors(yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bearing convention:
      yaw=0 deg   -> forward is +world Y
      yaw=90 deg  -> forward is +world X

    Camera convention:
      x_cam = right
      y_cam = down
      z_cam = forward
    """
    theta = np.deg2rad(yaw_deg)
    forward = np.array([np.sin(theta), np.cos(theta), 0.0], dtype=np.float32)
    right = np.array([np.cos(theta), -np.sin(theta), 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return right, up, forward


def frame_color(index: int) -> np.ndarray:
    """
    Deterministic bright-ish pseudo-color per frame.
    Returned as RGB uint8.
    """
    # Use HSV-like cycling without requiring matplotlib colormaps in the hot path.
    hue = (index * 0.61803398875) % 1.0
    h = hue * 6.0
    c = 220
    x = int(c * (1.0 - abs((h % 2.0) - 1.0)))
    m = 35

    if 0 <= h < 1:
        rgb = (c, x, 0)
    elif 1 <= h < 2:
        rgb = (x, c, 0)
    elif 2 <= h < 3:
        rgb = (0, c, x)
    elif 3 <= h < 4:
        rgb = (0, x, c)
    elif 4 <= h < 5:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)

    return np.array([rgb[0] + m, rgb[1] + m, rgb[2] + m], dtype=np.uint8)


def write_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors size mismatch: {points.shape[0]} vs {colors.shape[0]}")

    with path.open("wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {points.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))

        for p, c in zip(points, colors):
            f.write(
                struct.pack(
                    "<fffBBB",
                    float(p[0]), float(p[1]), float(p[2]),
                    int(c[0]), int(c[1]), int(c[2]),
                )
            )


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        return points.astype(np.float32), colors.astype(np.uint8)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    keys = np.floor(points / voxel_size).astype(np.int32)
    _, inv = np.unique(keys, axis=0, return_inverse=True)

    n = int(inv.max()) + 1
    pts_sum = np.zeros((n, 3), dtype=np.float64)
    col_sum = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    np.add.at(pts_sum, inv, points)
    np.add.at(col_sum, inv, colors)
    np.add.at(counts, inv, 1)

    pts_ds = (pts_sum / counts[:, None]).astype(np.float32)
    col_ds = np.clip(col_sum / counts[:, None], 0, 255).astype(np.uint8)
    return pts_ds, col_ds


def maybe_extract_zip(data_root: Path, out_dir: Path) -> Path:
    if data_root.suffix.lower() != ".zip":
        return data_root

    extract_dir = out_dir / "_unzipped_input"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(data_root, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def read_pose_records(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "pose" in data:
        return [data]
    return []


def collect_matched_frames(data_root: Path) -> pd.DataFrame:
    image_files = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in data_root.rglob(ext):
            image_files[p.stem] = p

    npy_files = {p.stem: p for p in data_root.rglob("*.npy")}

    records = []
    for jp in data_root.rglob("*.json"):
        for rec in read_pose_records(jp):
            image_name = rec.get("image", "")
            stem = Path(image_name).stem if image_name else jp.stem
            pose = rec.get("pose", {})

            if stem not in image_files or stem not in npy_files:
                continue

            required = ("x", "y", "z", "yaw")
            if not all(k in pose for k in required):
                continue

            records.append({
                "stem": stem,
                "image": image_name or (stem + ".jpg"),
                "jpg": image_files[stem],
                "npy": npy_files[stem],
                "x_original": float(pose["x"]),
                "y_original": float(pose["y"]),
                "z_original": float(pose["z"]),
                "yaw_original_deg": float(pose["yaw"]),
            })

    if not records:
        raise RuntimeError(
            "No matched frames found. Need matching stems for image, npy, and pose JSON image field."
        )

    df = pd.DataFrame(records)
    df = df.drop_duplicates(
        subset=["stem", "x_original", "y_original", "z_original", "yaw_original_deg"]
    )
    df = df.sort_values("stem").reset_index(drop=True)
    return df


def remove_consecutive_duplicate_poses(df: pd.DataFrame) -> pd.DataFrame:
    pose_cols = ["x_original", "y_original", "z_original", "yaw_original_deg"]
    is_dup_pose = (df[pose_cols].diff().abs().fillna(1.0).sum(axis=1) == 0.0)
    return df.loc[~is_dup_pose].reset_index(drop=True)


def parse_only_images(only_images: str | None) -> set[str] | None:
    if not only_images:
        return None

    raw_items = []
    for part in only_images.split(","):
        item = part.strip()
        if item:
            raw_items.append(Path(item).stem)

    return set(raw_items)


def apply_frame_selection(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    only = parse_only_images(args.only_images)
    if only is not None:
        df = df[df["stem"].isin(only)].reset_index(drop=True)

    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else len(df)
        df = df.iloc[start:end].reset_index(drop=True)

    if args.max_frames is not None and args.max_frames > 0:
        df = df.iloc[:args.max_frames].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("Frame selection produced 0 frames.")

    return df


def apply_pose_adjustments(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()

    yaw = args.yaw_sign * df["yaw_original_deg"].to_numpy(dtype=np.float32) + args.yaw_offset_deg
    yaw = np.mod(yaw, 360.0)

    origin = df[["x_original", "y_original", "z_original"]].iloc[0].to_numpy(dtype=np.float32)
    xyz = df[["x_original", "y_original", "z_original"]].to_numpy(dtype=np.float32)
    xyz_scaled = origin[None, :] + args.translation_scale * (xyz - origin[None, :])

    # Optional fixed additional z offset for quick visual tests.
    xyz_scaled[:, 2] += args.z_offset_m

    df["x"] = xyz_scaled[:, 0]
    df["y"] = xyz_scaled[:, 1]
    df["z"] = xyz_scaled[:, 2]
    df["yaw_deg"] = yaw

    return df


def save_debug_images(
    debug_dir: Path,
    stem: str,
    rgb_crop: np.ndarray,
    depth_raw: np.ndarray,
    depth_m: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(debug_dir / f"{stem}_rgb_crop.png"), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))

    raw_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    m_norm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(str(debug_dir / f"{stem}_depth_raw_norm.png"), raw_norm)
    cv2.imwrite(str(debug_dir / f"{stem}_depth_metric_norm.png"), m_norm)


def build_cloud(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = maybe_extract_zip(Path(args.data_root), out_dir)

    fx, fy, cx, cy, width, height = load_intrinsics(
        Path(args.calib_yaml),
        use_projection_matrix=args.use_projection_matrix,
    )
    raw_lut, meters_lut = load_lut(Path(args.lut_npz))

    if args.output_width is not None:
        width = int(args.output_width)
    if args.output_height is not None:
        height = int(args.output_height)

    df_all = collect_matched_frames(data_root)
    df_no_dups = remove_consecutive_duplicate_poses(df_all)
    df_selected = apply_frame_selection(df_no_dups, args)
    df = apply_pose_adjustments(df_selected, args)

    print(f"[INFO] matched frames before duplicate pose filter: {len(df_all)}")
    print(f"[INFO] frames after duplicate pose filter: {len(df_no_dups)}")
    print(f"[INFO] frames used after selection: {len(df)}")
    print(f"[INFO] intrinsics: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, size={width}x{height}")
    print(f"[INFO] crop: {args.crop_width}x{args.crop_height} -> {width}x{height}")
    print(f"[INFO] yaw used = {args.yaw_sign} * yaw_original + {args.yaw_offset_deg} deg")
    print(f"[INFO] translation_scale: {args.translation_scale}")
    print(f"[INFO] z_offset_m: {args.z_offset_m}")
    print(f"[INFO] color_mode: {args.color_mode}")
    print(f"[INFO] LUT raw sorted: {raw_lut}")
    print(f"[INFO] LUT meters sorted by raw: {meters_lut}")

    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    all_points = []
    all_colors = []
    per_frame_stats = []

    debug_dir = out_dir / "debug_images"

    for i, row in df.iterrows():
        bgr = cv2.imread(str(row["jpg"]), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] failed to read image: {row['jpg']}")
            continue

        rgb_crop = cv2.cvtColor(
            center_crop_resize(
                bgr,
                args.crop_width,
                args.crop_height,
                width,
                height,
            ),
            cv2.COLOR_BGR2RGB,
        )

        depth_raw = np.load(row["npy"]).astype(np.float32)
        if depth_raw.shape != (height, width):
            depth_raw = cv2.resize(depth_raw, (width, height), interpolation=cv2.INTER_LINEAR)

        depth_m = raw_depth_to_meters(depth_raw, raw_lut, meters_lut)

        if args.save_debug_images and i < args.debug_image_count:
            save_debug_images(debug_dir, row["stem"], rgb_crop, depth_raw, depth_m)

        mask = (
            np.isfinite(depth_m)
            & (depth_m >= args.min_depth_m)
            & (depth_m <= args.max_depth_m)
        )

        if args.pixel_step > 1:
            sparse_mask = np.zeros_like(mask, dtype=bool)
            sparse_mask[::args.pixel_step, ::args.pixel_step] = True
            mask &= sparse_mask

        if not np.any(mask):
            print(f"[WARN] no valid depth points for {row['image']}")
            continue

        z_cam = depth_m[mask]
        x_cam = (uu[mask] - cx) * z_cam / fx
        y_cam = (vv[mask] - cy) * z_cam / fy

        right, up, forward = yaw_bearing_to_vectors(float(row["yaw_deg"]))
        origin = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)

        pts_world = (
            origin[None, :]
            + x_cam[:, None] * right[None, :]
            - y_cam[:, None] * up[None, :]
            + z_cam[:, None] * forward[None, :]
        )

        if args.color_mode == "rgb":
            cols = rgb_crop[mask].astype(np.uint8)
        elif args.color_mode == "frame":
            cols = np.repeat(frame_color(i)[None, :], pts_world.shape[0], axis=0)
        elif args.color_mode == "depth":
            depth_norm = (z_cam - args.min_depth_m) / max(1e-6, (args.max_depth_m - args.min_depth_m))
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            cols = np.stack([
                (255.0 * depth_norm),
                (255.0 * (1.0 - np.abs(depth_norm - 0.5) * 2.0)),
                (255.0 * (1.0 - depth_norm)),
            ], axis=1).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported color_mode: {args.color_mode}")

        pts_world = pts_world.astype(np.float32)
        all_points.append(pts_world)
        all_colors.append(cols)

        if args.save_per_frame_clouds:
            frame_dir = out_dir / "per_frame_clouds"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_points_ds, frame_colors_ds = voxel_downsample(pts_world, cols, args.per_frame_voxel_size)
            write_ply_binary(
                frame_dir / f"{i:04d}_{row['stem']}.ply",
                frame_points_ds,
                frame_colors_ds,
            )

        per_frame_stats.append({
            "frame_index_used": int(i),
            "image": row["image"],
            "stem": row["stem"],
            "n_points": int(pts_world.shape[0]),
            "depth_raw_min": float(np.nanmin(depth_raw)),
            "depth_raw_max": float(np.nanmax(depth_raw)),
            "depth_m_p05": float(np.nanpercentile(depth_m, 5)),
            "depth_m_p50": float(np.nanpercentile(depth_m, 50)),
            "depth_m_p95": float(np.nanpercentile(depth_m, 95)),
            "x_original": float(row["x_original"]),
            "y_original": float(row["y_original"]),
            "z_original": float(row["z_original"]),
            "yaw_original_deg": float(row["yaw_original_deg"]),
            "x_used": float(row["x"]),
            "y_used": float(row["y"]),
            "z_used": float(row["z"]),
            "yaw_used_deg": float(row["yaw_deg"]),
        })

        if (i + 1) % 10 == 0:
            print(f"[INFO] processed {i + 1}/{len(df)} selected frames")

    if not all_points:
        raise RuntimeError("No points generated.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if args.filter_vertical_percentiles:
        z_low = np.percentile(points[:, 2], args.vertical_low_percentile)
        z_high = np.percentile(points[:, 2], args.vertical_high_percentile)
        z_mask = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
        points = points[z_mask]
        colors = colors[z_mask]
        print(f"[INFO] vertical filter z range: {z_low:.3f}..{z_high:.3f}")

    points_ds, colors_ds = voxel_downsample(points, colors, args.voxel_size)

    voxel_cm = int(round(args.voxel_size * 100))
    suffix = (
        f"color_{args.color_mode}_"
        f"yawoff_{args.yaw_offset_deg:g}_"
        f"yawsign_{args.yaw_sign:g}_"
        f"tscale_{args.translation_scale:g}"
    ).replace("-", "m").replace(".", "p")

    raw_ply = out_dir / f"merged_cloud_sparse_{suffix}.ply"
    ds_ply = out_dir / f"merged_cloud_voxel_{voxel_cm}cm_{suffix}.ply"
    stats_csv = out_dir / "pointcloud_frame_stats.csv"
    matched_csv = out_dir / "matched_frames_used.csv"
    topdown_png = out_dir / "merged_cloud_topdown_xy.png"
    traj_png = out_dir / "trajectory_used_for_cloud.png"

    write_ply_binary(raw_ply, points, colors)
    write_ply_binary(ds_ply, points_ds, colors_ds)
    pd.DataFrame(per_frame_stats).to_csv(stats_csv, index=False)
    df.to_csv(matched_csv, index=False)

    # Top-down cloud plot.
    plt.figure(figsize=(8, 8))
    sample_n = min(args.plot_max_points, len(points_ds))
    rng = np.random.default_rng(7)
    idx = rng.choice(len(points_ds), size=sample_n, replace=False) if len(points_ds) > sample_n else np.arange(len(points_ds))
    plot_colors = colors_ds[idx].astype(np.float32) / 255.0
    plt.scatter(points_ds[idx, 0], points_ds[idx, 1], s=0.2, c=plot_colors)
    plt.scatter(df["x"], df["y"], s=20, marker="x", label="reported/adjusted camera poses")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("world x [m]")
    plt.ylabel("world y [m]")
    plt.title("Top-down XY of merged point cloud")
    plt.legend()
    plt.tight_layout()
    plt.savefig(topdown_png, dpi=180)
    plt.close()

    # Trajectory plot.
    plt.figure(figsize=(7, 7))
    plt.plot(df["x"], df["y"], marker="o", linewidth=1)
    theta = np.deg2rad(df["yaw_deg"].to_numpy())
    plt.quiver(
        df["x"],
        df["y"],
        np.sin(theta) * 0.04,
        np.cos(theta) * 0.04,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory poses used for point cloud")
    plt.tight_layout()
    plt.savefig(traj_png, dpi=180)
    plt.close()

    print("[DONE]")
    print(f"  raw PLY:         {raw_ply}")
    print(f"  voxel PLY:       {ds_ply}")
    print(f"  frame stats CSV: {stats_csv}")
    print(f"  matched CSV:     {matched_csv}")
    print(f"  topdown PNG:     {topdown_png}")
    print(f"  trajectory PNG:  {traj_png}")
    if args.save_debug_images:
        print(f"  debug images:    {debug_dir}")
    if args.save_per_frame_clouds:
        print(f"  per-frame PLYs:  {out_dir / 'per_frame_clouds'}")
    print(f"  points raw:      {len(points)}")
    print(f"  points voxel:    {len(points_ds)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True, help="Input folder or zip containing jpg/npy/json files.")
    parser.add_argument("--calib-yaml", required=True, help="Camera calibration YAML.")
    parser.add_argument("--lut-npz", required=True, help="LUT .npz with arrays raw and meters.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")

    parser.add_argument("--crop-width", type=int, default=540)
    parser.add_argument("--crop-height", type=int, default=420)
    parser.add_argument("--output-width", type=int, default=None)
    parser.add_argument("--output-height", type=int, default=None)

    parser.add_argument("--pixel-step", type=int, default=4, help="Use every Nth pixel in each direction.")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="Voxel size in meters. 0 disables downsample.")
    parser.add_argument("--min-depth-m", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=5.0)

    parser.add_argument("--use-projection-matrix", action="store_true", help="Use projection_matrix instead of camera_matrix.")

    # Frame selection.
    parser.add_argument("--start-frame", type=int, default=None, help="Start index after duplicate-pose filtering.")
    parser.add_argument("--end-frame", type=int, default=None, help="End index after duplicate-pose filtering, exclusive.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit selected frames.")
    parser.add_argument(
        "--only-images",
        type=str,
        default=None,
        help="Comma-separated image names or stems to use, e.g. R2_..._1.jpg,R2_..._2.jpg",
    )

    # Pose debugging.
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0, help="Add constant yaw offset after yaw sign.")
    parser.add_argument("--yaw-sign", type=float, default=1.0, choices=[-1.0, 1.0], help="Use +yaw or -yaw.")
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale translation relative to first selected pose. 0=pure rotation around first pose.",
    )
    parser.add_argument("--z-offset-m", type=float, default=0.0, help="Add fixed offset to world z.")

    # Filtering.
    parser.add_argument("--filter-vertical-percentiles", action="store_true", default=True)
    parser.add_argument("--no-filter-vertical-percentiles", dest="filter_vertical_percentiles", action="store_false")
    parser.add_argument("--vertical-low-percentile", type=float, default=1.0)
    parser.add_argument("--vertical-high-percentile", type=float, default=99.0)

    # Debug outputs.
    parser.add_argument("--plot-max-points", type=int, default=120000)
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "frame", "depth"],
        default="rgb",
        help="Point color source: real RGB, one color per frame, or depth pseudo-color.",
    )
    parser.add_argument("in", action="store_true")
    parser.add_argument("--per-frame-voxel-size", type=float, default=0.02)

    parser.add_argument("--save-debug-images", action="store_true")
    parser.add_argument("--debug-image-count", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    build_cloud(parse_args())
