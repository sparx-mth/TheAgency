import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from sparx_agency.core.mapping.pipeline.mapping_pipeline import PinholeCloudGenerator
from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel

# Put this script next to map_from_image_clean.py, or change this import to the correct module path.
from map_from_image_clean import (
from sparx_agency.robots.common.helpers import valid_depth_mask
    Intrinsics,
    MapCfg,
    load_yaml,
    resize_intrinsics,
    fit_floor_plane_ransac,
    refine_plane_svd,
    plane_basis_from_camera,
)


class GlobalLogOddsGrid:
    def __init__(self, size_m: float, resolution_m: float, origin_x: Optional[float] = None, origin_y: Optional[float] = None):
        self.size_m = float(size_m)
        self.resolution_m = float(resolution_m)
        self.width = int(np.ceil(self.size_m / self.resolution_m))
        self.height = int(np.ceil(self.size_m / self.resolution_m))

        self.origin_x = -0.5 * self.size_m if origin_x is None else float(origin_x)
        self.origin_y = -0.5 * self.size_m if origin_y is None else float(origin_y)

        self.logodds = np.zeros((self.height, self.width), dtype=np.float32)
        self.hits = np.zeros((self.height, self.width), dtype=np.uint16)
        # For debugging: stores the latest frame index that wrote each occupied cell.
        self.owner = np.full((self.height, self.width), -1, dtype=np.int32)

    def world_to_cell(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((xy[:, 0] - self.origin_x) / self.resolution_m).astype(np.int32)
        iy = np.floor((xy[:, 1] - self.origin_y) / self.resolution_m).astype(np.int32)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        return ix, iy, valid

    def add_occupied_points(
        self,
        xy: np.ndarray,
        inc: float = 0.65,
        max_logodds: float = 4.0,
        frame_id: Optional[int] = None,
    ) -> int:
        if xy.size == 0:
            return 0

        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if xy.size == 0:
            return 0

        ix, iy, valid = self.world_to_cell(xy)
        ix = ix[valid]
        iy = iy[valid]
        if ix.size == 0:
            return 0

        np.add.at(self.logodds, (iy, ix), float(inc))
        np.add.at(self.hits, (iy, ix), 1)
        if frame_id is not None:
            self.owner[iy, ix] = int(frame_id)
        np.clip(self.logodds, -max_logodds, max_logodds, out=self.logodds)
        return int(ix.size)

    def to_probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def to_int8_grid(self, occupied_thresh: float = 0.65, free_thresh: float = 0.35) -> np.ndarray:
        prob = self.to_probability()
        grid = np.full(prob.shape, -1, dtype=np.int8)
        grid[prob < free_thresh] = 0
        grid[prob > occupied_thresh] = 100
        return grid

    def to_png(self, occupied_thresh: float = 0.65, free_thresh: float = 0.35) -> np.ndarray:
        grid = self.to_int8_grid(occupied_thresh=occupied_thresh, free_thresh=free_thresh)
        img = np.full(grid.shape, 127, dtype=np.uint8)
        img[grid == 0] = 255
        img[grid >= 50] = 0
        return np.flipud(img)

    def to_owner_png(self) -> np.ndarray:
        """
        Debug image: colors each occupied cell by the latest frame that wrote it.
        Gray means no frame wrote that cell.
        """
        owner = self.owner.copy()
        valid = owner >= 0
        out = np.full((self.height, self.width, 3), 127, dtype=np.uint8)
        if np.any(valid):
            max_owner = max(int(owner[valid].max()), 1)
            owner_u8 = np.zeros_like(owner, dtype=np.uint8)
            owner_u8[valid] = np.clip(255.0 * owner[valid] / max_owner, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(owner_u8, cv2.COLORMAP_TURBO)
            out[valid] = colored[valid]
        return np.flipud(out)


def load_pose_json(json_path: Path) -> Tuple[Dict[str, float], str]:
    with open(json_path, "r") as f:
        data = json.load(f)

    pose = data.get("pose", {})
    image_name = data.get("image", "")

    required = ["x", "y", "yaw"]
    missing = [k for k in required if k not in pose]
    if missing:
        raise RuntimeError(f"Missing pose keys {missing} in {json_path}")

    if not image_name:
        image_name = json_path.with_suffix(".jpg").name

    return pose, image_name


def find_pose_files(frames_dir: Path, json_glob: str) -> List[Path]:
    pose_files = sorted(frames_dir.glob(json_glob))
    if not pose_files:
        raise RuntimeError(f"No pose JSON files found in {frames_dir} using glob '{json_glob}'")
    return pose_files


def colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 0.0, max_depth_m)
    u8 = (255.0 * d / max(max_depth_m, 1e-6)).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)


def extract_local_obstacle_xy(pts: np.ndarray, n: np.ndarray, d: float, u: np.ndarray, v: np.ndarray, cfg: MapCfg) -> np.ndarray:
    """
    Returns Nx2 local obstacle points.
      local_xy[:, 0] = forward
      local_xy[:, 1] = lateral/left
    """
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    if n[2] < 0:
        n = -n
        d = -d

    signed = (pts @ n + d).astype(np.float32)
    height = signed.copy()
    height[height < 0.0] = 0.0

    p_proj = pts - signed[:, None] * n[None, :]

    lateral = (p_proj @ u).astype(np.float32)
    forward = (p_proj @ v).astype(np.float32)

    finite = np.isfinite(forward) & np.isfinite(lateral) & np.isfinite(height)
    front = forward > 0.0
    obstacle = height >= float(cfg.occupied_height_m)

    mask = finite & front & obstacle
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float32)

    return np.stack([forward[mask], lateral[mask]], axis=1).astype(np.float32)


def local_to_global_xy(
    local_xy: np.ndarray,
    pose: Dict[str, float],
    yaw_sign: float,
    yaw_offset_deg: float,
    lateral_sign: float,
    pose_scale: float,
) -> np.ndarray:
    if local_xy.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    forward = local_xy[:, 0]
    lateral = float(lateral_sign) * local_xy[:, 1]

    yaw_deg = float(yaw_sign) * float(pose["yaw"]) + float(yaw_offset_deg)
    yaw = np.deg2rad(yaw_deg)
    c = np.cos(yaw)
    s = np.sin(yaw)

    tx = float(pose["x"]) * float(pose_scale)
    ty = float(pose["y"]) * float(pose_scale)

    gx = tx + c * forward - s * lateral
    gy = ty + s * forward + c * lateral

    return np.stack([gx, gy], axis=1).astype(np.float32)


def fit_floor_for_frame(pts: np.ndarray, cfg: MapCfg) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    if pts.shape[0] < 500:
        return None

    z = pts[:, 2].astype(np.float32)
    finite = np.isfinite(z)
    if not np.any(finite):
        return None

    bottom_frac = float(np.clip(cfg.floor_bottom_frac, 0.05, 0.9))
    z_thr = np.quantile(z[finite], bottom_frac)
    floor_candidates = finite & (z <= z_thr)
    pts_floor = pts[floor_candidates]

    if pts_floor.shape[0] < 500:
        return None

    plane = fit_floor_plane_ransac(
        pts_floor,
        iters=cfg.floor_ransac_iters,
        dist_thresh=cfg.floor_dist_thresh,
        min_inliers=cfg.min_plane_inliers,
        normal_gate_abs_z=cfg.normal_gate_abs_z,
    )
    if plane is None:
        return None

    n0, d0, inliers0 = plane
    if n0[2] < 0:
        n0 = -n0
        d0 = -d0

    n, d = refine_plane_svd(pts_floor[inliers0])
    if n[2] < 0:
        n = -n
        d = -d

    cam_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u, v = plane_basis_from_camera(n, cam_forward)

    fwd_p = cam_forward - (cam_forward @ n) * n
    nf = np.linalg.norm(fwd_p)
    if nf > 1e-6:
        fwd_p = fwd_p / nf
        if float(v @ fwd_p) < 0.0:
            v = -v
            u = np.cross(n, v)
            u = u / (np.linalg.norm(u) + 1e-9)

    return n.astype(np.float32), float(d), u.astype(np.float32), v.astype(np.float32)


def process_one_frame(
    image_path: Path,
    depth_model: DA3TensorRTModel,
    intr_full: Intrinsics,
    cfg: MapCfg,
    cloud_generator: PinholeCloudGenerator,
    min_depth_m: float,
    max_depth_m: float,
    ignore_top_frac: float,
    ignore_bottom_frac: float,
    debug_dir: Optional[Path],
    frame_index: int,
    debug_every: int,
    depth_lut: Optional[Tuple[np.ndarray, np.ndarray]],
    depth_scale: float,
    depth_shift: float,
    center_crop_width: int,
    center_crop_height: int,
) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return None

    # Center crop raw RGB to the small model input size, e.g. 504x392.
    if center_crop_width > 0 and center_crop_height > 0:
        h_img, w_img = bgr.shape[:2]

        if center_crop_width > w_img or center_crop_height > h_img:
            raise RuntimeError(
                f"Requested crop {center_crop_width}x{center_crop_height}, "
                f"but image is {w_img}x{h_img}"
            )

        x0 = (w_img - center_crop_width) // 2
        y0 = (h_img - center_crop_height) // 2
        bgr = bgr[y0:y0 + center_crop_height, x0:x0 + center_crop_width].copy()

    depth_raw = depth_model.infer_depth(bgr).astype(np.float32)

    # Small model raw output -> calibrated metric depth using LUT.
    depth_m = apply_depth_lut(depth_raw, depth_lut)

    # Optional final linear correction. Use 1.0 and 0.0 for now.
    depth_m = depth_scale * depth_m + depth_shift

    if debug_every > 0 and frame_index % debug_every == 0:
        finite_raw = np.isfinite(depth_raw)
        finite_m = np.isfinite(depth_m)

        if np.any(finite_raw):
            print(
                f"[DEPTH RAW] frame={frame_index} "
                f"p05/p50/p95="
                f"{[float(np.nanpercentile(depth_raw, p)) for p in [5, 50, 95]]}"
            )

        if np.any(finite_m):
            print(
                f"[DEPTH LUT] frame={frame_index} "
                f"p05/p50/p95="
                f"{[float(np.nanpercentile(depth_m, p)) for p in [5, 50, 95]]}"
            )

    valid = valid_depth_mask(depth_m, min_depth=min_depth_m, max_depth=max_depth_m)
    depth_m = depth_m.copy()
    depth_m[~valid] = np.nan

    h = depth_m.shape[0]
    top = int(np.clip(ignore_top_frac, 0.0, 0.95) * h)
    bottom = int(np.clip(ignore_bottom_frac, 0.0, 0.95) * h)
    if top > 0:
        depth_m[:top, :] = np.nan
    if bottom > 0:
        depth_m[h - bottom :, :] = np.nan

    if debug_dir is not None and debug_every > 0 and frame_index % debug_every == 0:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"depth_{frame_index:05d}.png"), colorize_depth(depth_m, max_depth_m))

    intr = resize_intrinsics(intr_full, cfg.inference_width, cfg.inference_height)
    pts = cloud_generator.depth_to_cloud_to_base_xyz(depth_m, intr)
    if pts.size == 0:
        print(f"[WARN] No point cloud points for {image_path.name}")
        return None

    floor_result = fit_floor_for_frame(pts, cfg)
    if floor_result is None:
        print(f"[WARN] Floor fit failed for {image_path.name}")
        return None

    n, d, u, v = floor_result
    local_xy = extract_local_obstacle_xy(pts, n, d, u, v, cfg)
    return local_xy

def load_depth_lut(path: Optional[str]):
    if path is None:
        return None

    data = np.load(path)
    raw = data["raw"].astype(np.float32)
    meters = data["meters"].astype(np.float32)

    order = np.argsort(raw)
    raw = raw[order]
    meters = meters[order]

    print("[LUT] loaded:", path)
    print("[LUT] raw:", raw)
    print("[LUT] meters:", meters)

    return raw, meters


def apply_depth_lut(depth_raw: np.ndarray, lut):
    if lut is None:
        return depth_raw

    raw, meters = lut

    depth_flat = depth_raw.reshape(-1).astype(np.float32)

    calibrated = np.interp(
        depth_flat,
        raw,
        meters,
        left=meters[0],
        right=meters[-1],
    ).astype(np.float32)

    return calibrated.reshape(depth_raw.shape)

def frame_color_bgr(frame_id: int, max_frame_id: int) -> Tuple[int, int, int]:
    """Return the same BGR color used by global_occ_by_frame.png for a frame id."""
    if frame_id < 0:
        return (127, 127, 127)
    max_frame_id = max(int(max_frame_id), 1)
    value = np.uint8(np.clip(255.0 * int(frame_id) / max_frame_id, 0, 255))
    color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def write_frame_color_legend(frame_records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Save a CSV/JSON legend and a PNG showing frame id -> image/yaw/color."""
    if not frame_records:
        return

    max_frame_id = max(int(r["frame_id"]) for r in frame_records)

    legend_json = []
    csv_lines = ["frame_id,image,yaw_deg,pose_x,pose_y,color_b,color_g,color_r,local_points,written_points"]

    row_h = 32
    w = 980
    h = max(80, row_h * (len(frame_records) + 1) + 16)
    legend_img = np.full((h, w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        legend_img,
        "frame_id  color  yaw_deg   pose_x   pose_y   local_pts  written_pts  image",
        (10, 24),
        font,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    print("Frame color legend for global_occ_by_frame.png:")
    print("frame_id | BGR color       | yaw_deg | pose_x  pose_y  | local_pts written_pts | image")
    print("---------+-----------------+---------+----------------+----------------------+----------------")

    for i, r in enumerate(frame_records):
        frame_id = int(r["frame_id"])
        b, g, rr = frame_color_bgr(frame_id, max_frame_id)
        yaw = float(r["yaw_deg"])
        px = float(r["pose_x"])
        py = float(r["pose_y"])
        local_pts = int(r["local_points"])
        written_pts = int(r["written_points"])
        image = str(r["image"])

        print(
            f"{frame_id:8d} | ({b:3d},{g:3d},{rr:3d}) | {yaw:7.1f} | "
            f"{px:+6.3f} {py:+6.3f} | {local_pts:9d} {written_pts:11d} | {image}"
        )

        legend_json.append(
            {
                "frame_id": frame_id,
                "image": image,
                "yaw_deg": yaw,
                "pose_x": px,
                "pose_y": py,
                "color_bgr": [b, g, rr],
                "color_rgb": [rr, g, b],
                "local_points": local_pts,
                "written_points": written_pts,
            }
        )
        csv_lines.append(
            f"{frame_id},{image},{yaw:.6f},{px:.6f},{py:.6f},{b},{g},{rr},{local_pts},{written_pts}"
        )

        y = 52 + i * row_h
        cv2.rectangle(legend_img, (95, y - 17), (150, y + 5), (b, g, rr), -1)
        cv2.rectangle(legend_img, (95, y - 17), (150, y + 5), (0, 0, 0), 1)
        text = (
            f"{frame_id:3d}           yaw={yaw:6.1f}  "
            f"pose=({px:+.3f},{py:+.3f})  "
            f"pts={local_pts}/{written_pts}  {image}"
        )
        cv2.putText(legend_img, text, (10, y), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    with open(out_dir / "frame_color_legend.json", "w") as f:
        json.dump(legend_json, f, indent=2)

    with open(out_dir / "frame_color_legend.csv", "w") as f:
        f.write("".join(csv_lines) + "")

    cv2.imwrite(str(out_dir / "frame_color_legend.png"), legend_img)


def draw_trajectory_overlay(grid: GlobalLogOddsGrid, poses_xy: np.ndarray, out_path: Path) -> None:
    img = grid.to_png()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if poses_xy.size > 0:
        ix, iy, valid = grid.world_to_cell(poses_xy)
        ix = ix[valid]
        iy = iy[valid]
        # Convert grid row to displayed image row because occupancy PNG is flipped vertically.
        iy_disp = grid.height - 1 - iy

        for i in range(len(ix)):
            cv2.circle(img_bgr, (int(ix[i]), int(iy_disp[i])), 2, (0, 0, 255), -1)
            if i > 0:
                cv2.line(
                    img_bgr,
                    (int(ix[i - 1]), int(iy_disp[i - 1])),
                    (int(ix[i]), int(iy_disp[i])),
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    cv2.imwrite(str(out_path), img_bgr)


def save_metadata(args: argparse.Namespace, grid: GlobalLogOddsGrid, used_frames: int, skipped_frames: int, out_dir: Path) -> None:
    metadata: Dict[str, Any] = {
        "format": "global_logodds_occupancy_from_da3_sequence",
        "grid_shape_hw": [int(grid.height), int(grid.width)],
        "resolution_m_per_cell": float(grid.resolution_m),
        "size_m": float(grid.size_m),
        "origin_xy_m": [float(grid.origin_x), float(grid.origin_y)],
        "used_frames": int(used_frames),
        "skipped_frames": int(skipped_frames),
        "pose_transform": {
            "yaw_sign": float(args.yaw_sign),
            "yaw_offset_deg": float(args.yaw_offset_deg),
            "lateral_sign": float(args.lateral_sign),
            "pose_scale": float(args.pose_scale),
        },
        "values_int8": {
            "-1": "unknown",
            "0": "free",
            "100": "occupied",
        },
        "note": "First version accumulates occupied evidence only. Free-space carving is intentionally disabled for noisy telemetry.",
    }
    with open(out_dir / "global_occ_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="Directory containing image files and matching JSON pose files.")
    ap.add_argument("--json-glob", default="*.json", help="Glob for pose JSON files inside --frames-dir.")
    ap.add_argument("--config", required=True, help="Mapping config YAML used by map_from_image_clean.py.")
    ap.add_argument("--calib-yaml", required=True, help="Camera YAML passed to DA3TensorRTModel.")
    ap.add_argument("--engine-path", required=True, help="TensorRT engine path for DA3.")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--grid-size-m", type=float, default=8.0)
    ap.add_argument("--resolution-m", type=float, default=0.05)
    ap.add_argument("--origin-x", type=float, default=None)
    ap.add_argument("--origin-y", type=float, default=None)

    ap.add_argument("--min-depth-m", type=float, default=0.4)
    ap.add_argument("--max-depth-m", type=float, default=7.0)
    ap.add_argument("--ignore-top-frac", type=float, default=0.25)
    ap.add_argument("--ignore-bottom-frac", type=float, default=0.05)

    ap.add_argument("--depth-lut-npz", default=None)
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--depth-shift", type=float, default=0.0)
    ap.add_argument("--center-crop-width", type=int, default=0)
    ap.add_argument("--center-crop-height", type=int, default=0)

    ap.add_argument("--yaw-sign", type=float, default=1.0)
    ap.add_argument("--yaw-offset-deg", type=float, default=0.0)
    ap.add_argument("--lateral-sign", type=float, default=1.0)
    ap.add_argument("--pose-scale", type=float, default=1.0)

    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--occ-inc", type=float, default=0.65)
    ap.add_argument("--debug-every", type=int, default=20)

    ap.add_argument("--cam-pitch-deg", type=float, default=30.0)
    ap.add_argument("--camera-height-m", type=float, default=0.0, help="Use 0.0 first. Do not use the old fake 10 m height.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"

    intr_full, cfg = load_yaml(args.config)

    # Override resolution from CLI for the global grid only.
    cfg.resolution_m = float(args.resolution_m)

    cloud_generator = PinholeCloudGenerator(
        stride=cfg.stride,
        cam_rpy_deg=(0.0, float(args.cam_pitch_deg), 0.0),
        t_base=(0.0, 0.0, float(args.camera_height_m)),
    )

    print(f"[DA3] Loading engine: {args.engine_path}")
    print(f"[DA3] Loading camera YAML: {args.calib_yaml}")
    depth_model = DA3TensorRTModel(engine_path=args.engine_path, yaml_path=args.calib_yaml)

    pose_files = find_pose_files(frames_dir, args.json_glob)
    if args.step < 1:
        raise RuntimeError("--step must be >= 1")

    pose_files = pose_files[:: args.step]
    if args.max_frames > 0:
        pose_files = pose_files[: args.max_frames]

    grid = GlobalLogOddsGrid(
        size_m=args.grid_size_m,
        resolution_m=args.resolution_m,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
    )



    poses_xy: List[List[float]] = []
    frame_records: List[Dict[str, Any]] = []
    used_frames = 0
    skipped_frames = 0

    for frame_index, json_path in enumerate(pose_files):
        try:
            pose, image_name = load_pose_json(json_path)
            image_path = frames_dir / image_name
            depth_lut = load_depth_lut(args.depth_lut_npz)
            local_xy = process_one_frame(
                image_path=image_path,
                depth_model=depth_model,
                intr_full=intr_full,
                cfg=cfg,
                cloud_generator=cloud_generator,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                ignore_top_frac=args.ignore_top_frac,
                ignore_bottom_frac=args.ignore_bottom_frac,
                debug_dir=debug_dir,
                frame_index=frame_index,
                debug_every=args.debug_every,
                depth_lut=depth_lut,
                depth_scale=args.depth_scale,
                depth_shift=args.depth_shift,
                center_crop_width=args.center_crop_width,
                center_crop_height=args.center_crop_height,
            )

            if local_xy is None or local_xy.shape[0] == 0:
                skipped_frames += 1
                continue

            global_xy = local_to_global_xy(
                local_xy=local_xy,
                pose=pose,
                yaw_sign=args.yaw_sign,
                yaw_offset_deg=args.yaw_offset_deg,
                lateral_sign=args.lateral_sign,
                pose_scale=args.pose_scale,
            )

            written = grid.add_occupied_points(global_xy, inc=args.occ_inc, frame_id=frame_index)
            poses_xy.append([float(pose["x"]) * args.pose_scale, float(pose["y"]) * args.pose_scale])
            frame_records.append(
                {
                    "frame_id": int(frame_index),
                    "image": image_name,
                    "yaw_deg": float(pose["yaw"]),
                    "pose_x": float(pose["x"]) * args.pose_scale,
                    "pose_y": float(pose["y"]) * args.pose_scale,
                    "local_points": int(local_xy.shape[0]),
                    "written_points": int(written),
                }
            )
            used_frames += 1

            print(
                f"[{frame_index + 1:04d}/{len(pose_files):04d}] {image_name}: "
                f"local_pts={local_xy.shape[0]} written={written} "
                f"pose=({float(pose['x']):+.3f},{float(pose['y']):+.3f}) yaw={float(pose['yaw']):.1f}"
            )

        except Exception as e:
            skipped_frames += 1
            print(f"[WARN] Skipping {json_path.name}: {e}")

    prob = grid.to_probability()
    int8_grid = grid.to_int8_grid()
    png = grid.to_png()

    np.save(out_dir / "global_occ_prob.npy", prob.astype(np.float32))
    np.save(out_dir / "global_occ_logodds.npy", grid.logodds.astype(np.float32))
    np.save(out_dir / "global_occ_int8.npy", int8_grid.astype(np.int8))
    np.save(out_dir / "global_occ_hits.npy", grid.hits.astype(np.uint16))
    np.save(out_dir / "global_occ_owner.npy", grid.owner.astype(np.int32))
    cv2.imwrite(str(out_dir / "global_occ.png"), png)
    cv2.imwrite(str(out_dir / "global_occ_by_frame.png"), grid.to_owner_png())

    png_big = cv2.resize(png, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "global_occ_big.png"), png_big)

    poses_arr = np.asarray(poses_xy, dtype=np.float32)
    np.save(out_dir / "trajectory_xy.npy", poses_arr)
    draw_trajectory_overlay(grid, poses_arr, out_dir / "global_occ_with_trajectory.png")
    write_frame_color_legend(frame_records, out_dir)

    save_metadata(args, grid, used_frames, skipped_frames, out_dir)

    print("Done")
    print(f"  used_frames={used_frames}")
    print(f"  skipped_frames={skipped_frames}")
    print(f"  saved: {out_dir}")


if __name__ == "__main__":
    main()
