#!/usr/bin/env python3
"""
Offline room mapper: RGB + DA3 depth frames → 2D occupancy map + object poses.

Usage:
    python run_room_mapper.py \
        --data-dir /home/daphnaa/Documents/xtend_da3_takes/xtend_da3_take_20260616_171539 \
        --tag-map  sparx_agency/tasks/localization/config/new_map.yaml \
        --output-dir /tmp/room_map_out \
        --stride 5 \
        --labels labels.json   # optional
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # project root

from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridCostmap, LogOddsGridConfig
from sparx_agency.core.mapping.costmap.depth_to_grid import update_grid_from_depth, texture_confidence_mask
from sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.frame_reader import iter_frames
from sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.pose_fuser import PoseFuser
from sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.map_visualizer import (
    save_map_png, render_map_rgb,
)
from sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.object_placer import (
    ObjectMarker, load_labels, place_objects, cluster_objects,
    flag_objects_beyond_tags, flag_objects_outside_map,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_DIR = _REPO_ROOT / "sparx_agency" / "robots" / "XTEND" / "config"

RGB_CALIB   = str(_CONFIG_DIR / "camera_xtend_ros_calib_720_420.yaml")
DEPTH_CALIB = str(_CONFIG_DIR / "camera_xtend_ros_calib_504_392_crop_resize.yaml")
DEPTH_H, DEPTH_W = 392, 504

_TRAJ_MIN_MOVE_M = 0.10   # skip trajectory point if drone moved less than this
_TRAJ_SMOOTH_WIN = 7      # rolling-average window for trajectory smoothing


def _load_depth_K(calib_path: str) -> np.ndarray:
    with open(calib_path) as f:
        data = yaml.safe_load(f)
    if "projection_matrix" in data:
        P = np.array(data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        return P[:3, :3]
    return np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline room mapper (RGB + DA3 depth).")
    p.add_argument("--data-dir",         required=True)
    p.add_argument("--tag-map",          default=None,
                   help="AprilTag world map YAML. Omit for odometry-only (unscaled) mode.")
    p.add_argument("--output-dir",       default="./room_map_out")
    p.add_argument("--stride",           type=int,   default=5)
    p.add_argument("--labels",           default=None)
    p.add_argument("--grid-size",        type=float, default=14.0)
    p.add_argument("--resolution",       type=float, default=0.05)
    p.add_argument("--depth-max",        type=float, default=5.0)
    p.add_argument("--z-min",            type=float, default=0.0)
    p.add_argument("--z-max",            type=float, default=3.0)
    p.add_argument("--downsample",       type=int,   default=4)
    p.add_argument("--cluster-radius",   type=float, default=2.0,
                   help="Merge same-label object markers within this radius (m)")
    p.add_argument("--texture-thresh",   type=float, default=0.0,
                   help="Laplacian gradient threshold for depth confidence masking. "
                        "0 = disabled. Try 6-12 to ignore white/featureless walls.")
    p.add_argument("--no-scale-correction", action="store_true",
                   help="Disable per-frame DA3 scale correction from AprilTags.")
    p.add_argument("--labels-only-grid",  action="store_true",
                   help="Only update the occupancy grid for labeled frames. "
                        "Guarantees objects are in front of their walls (same-frame DA3 consistency).")
    p.add_argument("--no-convex-walls",  action="store_true",
                   help="Disable convex-hull post-processing of free space. "
                        "By default the free region is convex-hull-filled to fix "
                        "diagonal DA3 corner smear in rectangular rooms.")
    p.add_argument("--no-raytrace",      action="store_true",
                   help="Disable ray-cast free-space (faster but no wall contrast)")
    p.add_argument("--preview",          action="store_true",
                   help="Show live map preview window during processing")
    p.add_argument("--preview-interval", type=int,   default=10,
                   help="Update preview every N processed frames")
    return p.parse_args()


def _show_preview(
    grid, trajectory, objects, tag_fixes, title, win_name="Room Mapper"
) -> bool:
    """Render current map to cv2 window. Returns False if user pressed 'q'."""
    rgb = render_map_rgb(grid, trajectory, objects, tag_fixes, title)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(win_name, bgr)
    return (cv2.waitKey(1) & 0xFF) != ord('q')


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K_depth = _load_depth_K(DEPTH_CALIB)
    grid = LogOddsGridCostmap(LogOddsGridConfig(
        resolution_m=args.resolution, size_m=args.grid_size,
    ))
    fuser = PoseFuser(
        tag_map_path=args.tag_map,
        rgb_calib_path=RGB_CALIB,
        depth_calib_path=DEPTH_CALIB,
        depth_h=DEPTH_H, depth_w=DEPTH_W,
    )

    trajectory: List[Tuple[float, float]] = []
    tag_fix_positions: List[Tuple[float, float]] = []
    frame_poses: Dict[int, np.ndarray] = {}
    frame_tag_ids: Dict[int, List[int]] = {}
    frame_tag_areas: Dict[int, float] = {}
    depth_cache: Dict[int, np.ndarray] = {}
    _prev_tag_set: frozenset = frozenset()

    labeled_frame_idxs: Optional[set] = None
    # frame_idx → list of (x, y, w, h) bboxes in depth-image coordinates
    frame_bbox_masks: Dict[int, List[Tuple[int, int, int, int]]] = {}
    if args.labels and Path(args.labels).exists():
        _raw_labels = load_labels(args.labels)
        if args.labels_only_grid:
            labeled_frame_idxs = {int(e["frame_idx"]) for e in _raw_labels}
            print(f"[mapper] labels-only-grid: will update grid for "
                  f"{len(labeled_frame_idxs)} labeled frames only")
        # Build per-frame bbox list (scaled from source_size to depth resolution)
        for _e in _raw_labels:
            _fidx = int(_e["frame_idx"])
            _bx, _by, _bw, _bh = _e["bbox"]
            _sw, _sh = _e.get("source_size", [DEPTH_W, DEPTH_H])
            _dx = int(_bx * DEPTH_W / _sw); _dy = int(_by * DEPTH_H / _sh)
            _dw = int(_bw * DEPTH_W / _sw); _dh = int(_bh * DEPTH_H / _sh)
            frame_bbox_masks.setdefault(_fidx, []).append((_dx, _dy, _dw, _dh))
        print(f"[mapper] bbox masking: {len(frame_bbox_masks)} labeled frames will "
              f"have object regions excluded from grid")

    frames = list(iter_frames(Path(args.data_dir), stride=args.stride))
    print(f"[mapper] {len(frames)} frames  stride={args.stride}  "
          f"raytrace={'off' if args.no_raytrace else 'on'}  "
          f"preview={'on' if args.preview else 'off'}")

    if args.preview:
        cv2.namedWindow("Room Mapper", cv2.WINDOW_NORMAL)

    prev_n_fixes = 0
    for i, rec in enumerate(frames):
        bgr     = rec.load_rgb()
        depth_m = rec.load_depth()

        world_T_cam = fuser.update(bgr, depth_m)
        frame_tag_ids[rec.frame_idx] = list(fuser.last_tag_ids)
        frame_tag_areas[rec.frame_idx] = fuser.last_tag_total_area

        if world_T_cam is None:
            continue

        # Keep raw depth for object placement (scale is calibrated at wall/tag distance,
        # not at object depth — applying it to foreground objects would over-push them).
        depth_raw = depth_m

        # Per-frame DA3 scale correction from AprilTag ground truth (walls only)
        if not args.no_scale_correction and fuser.last_depth_scale is not None:
            depth_m = depth_m * fuser.last_depth_scale

        if labeled_frame_idxs is None or rec.frame_idx in labeled_frame_idxs:
            conf_mask = None
            if args.texture_thresh > 0.0:
                conf_mask = texture_confidence_mask(
                    bgr, DEPTH_H, DEPTH_W, thresh=args.texture_thresh
                )

            # Mask out labeled-object bboxes so foreground objects don't create
            # phantom wall cells in the occupancy grid at the object's depth.
            bboxes = frame_bbox_masks.get(rec.frame_idx)
            if bboxes:
                obj_mask = np.ones((DEPTH_H, DEPTH_W), dtype=bool)
                for bx, by, bw, bh in bboxes:
                    x1, y1 = max(0, bx), max(0, by)
                    x2, y2 = min(DEPTH_W, bx + bw), min(DEPTH_H, by + bh)
                    obj_mask[y1:y2, x1:x2] = False
                conf_mask = obj_mask if conf_mask is None else (conf_mask & obj_mask)

            update_grid_from_depth(
                grid, depth_m, K_depth, world_T_cam,
                z_min_world=args.z_min,
                z_max_world=args.z_max,
                depth_max_m=args.depth_max,
                downsample=args.downsample,
                raytrace=not args.no_raytrace,
                confidence_mask=conf_mask,
            )

        wx, wy = float(world_T_cam[0, 3]), float(world_T_cam[1, 3])

        # Trajectory dedup: skip if barely moved
        if not trajectory or (wx - trajectory[-1][0])**2 + (wy - trajectory[-1][1])**2 >= _TRAJ_MIN_MOVE_M**2:
            trajectory.append((wx, wy))

        frame_poses[rec.frame_idx] = world_T_cam.copy()

        # Only mark a star when the visible tag set changes (not every frame)
        current_tag_set = frozenset(fuser.last_tag_ids)
        if current_tag_set and current_tag_set != _prev_tag_set:
            tag_fix_positions.append((wx, wy))
            _prev_tag_set = current_tag_set
        prev_n_fixes = fuser.n_tag_fixes

        if args.labels:
            depth_cache[rec.frame_idx] = depth_raw.copy()

        if (i + 1) % 20 == 0:
            scale_str = f"{fuser.last_depth_scale:.3f}" if fuser.last_depth_scale else "none"
            print(f"[mapper] {i+1}/{len(frames)}  tag_fixes={fuser.n_tag_fixes}  "
                  f"traj_pts={len(trajectory)}  depth_scale={scale_str}")

        if args.preview and (i + 1) % args.preview_interval == 0:
            title = f"Frame {rec.frame_idx}  fixes={fuser.n_tag_fixes}"
            if not _show_preview(grid, trajectory, [], tag_fix_positions, title):
                print("[mapper] Preview quit by user.")
                break

    print(f"[mapper] Done. tag_fixes={fuser.n_tag_fixes}  frames_with_pose={len(frame_poses)}")

    objects: List[ObjectMarker] = []
    if args.labels and Path(args.labels).exists():
        labels = load_labels(args.labels)
        objects = place_objects(labels, frame_poses, lambda fidx: depth_cache[fidx],
                                K_depth, frame_tag_ids, frame_tag_areas)
        print(f"[mapper] Placed {len(objects)} raw object markers")
        objects = cluster_objects(objects, radius_m=args.cluster_radius)
        print(f"[mapper] After clustering ({args.cluster_radius}m): {len(objects)} objects")
        if args.tag_map is not None:
            objects = flag_objects_beyond_tags(
                objects, frame_poses, fuser.tag_world_xyz, frame_tag_ids, tolerance_m=0.4)
        objects = flag_objects_outside_map(objects, grid)
        for obj in objects:
            print(f"         {obj.label:20s}  "
                  f"({obj.world_x:+.2f}, {obj.world_y:+.2f}, {obj.world_z:+.2f})m  "
                  f"tags={obj.tag_ids}")

    np.save(str(out_dir / "trajectory.npy"), np.array(trajectory))

    # Smooth trajectory for visualization (raw positions kept in .npy)
    if len(trajectory) >= _TRAJ_SMOOTH_WIN:
        arr = np.array(trajectory)
        k = np.ones(_TRAJ_SMOOTH_WIN) / _TRAJ_SMOOTH_WIN
        traj_smooth = list(zip(
            np.convolve(arr[:, 0], k, mode="valid").tolist(),
            np.convolve(arr[:, 1], k, mode="valid").tolist(),
        ))
    else:
        traj_smooth = trajectory
    np.save(str(out_dir / "frame_poses.npy"), frame_poses, allow_pickle=True)
    _, grid_arr = grid.get_grid()
    np.save(str(out_dir / "occupancy_2d.npy"), grid_arr)

    map_title = f"Room Map — {Path(args.data_dir).name}"
    save_map_png(
        grid=grid,
        trajectory_world=traj_smooth,
        objects=objects,
        output_path=str(out_dir / "map_with_objects.png"),
        title=map_title,
        tag_fixes=tag_fix_positions if tag_fix_positions else None,
        convex_walls=not args.no_convex_walls,
    )

    if args.preview:
        cv2.destroyAllWindows()

    print(f"[mapper] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()