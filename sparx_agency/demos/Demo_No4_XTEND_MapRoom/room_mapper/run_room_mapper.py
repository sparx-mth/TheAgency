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
import math
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
    ObjectMarker, load_labels, load_labels_from_session, place_objects, cluster_objects,
    flag_objects_beyond_tags, flag_objects_outside_map, snap_objects_to_free_space,
    save_objects_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_DIR = _REPO_ROOT / "sparx_agency" / "robots" / "XTEND" / "config"

RGB_CALIB   = str(_CONFIG_DIR / "camera_xtend_ros_calib_720_420.yaml")
DEPTH_CALIB = str(_CONFIG_DIR / "camera_xtend_ros_calib_504_294_resize.yaml")
DEPTH_H, DEPTH_W = 294, 504

_TRAJ_MIN_MOVE_M = 0.05   # skip trajectory point if drone moved less than this
_TRAJ_SMOOTH_WIN = 3      # rolling-average window for trajectory smoothing


def _load_depth_K(calib_path: str) -> np.ndarray:
    """Load the intrinsics matrix used to backproject depth pixels into world points.

    Prefers camera_matrix (the raw/distorted-image K) over projection_matrix (P):
    depth is inferred directly on the raw distorted frame (DA3 never undistorts
    it), so P — valid only after rectification — is the wrong K here. Using P
    introduced a real ~12% fx/fy anisotropy on this rig's calib, which combined
    with per-frame yaw rotation to shear an actually-rectangular room into a
    parallelogram once frames from different viewing angles were merged.
    """
    with open(calib_path) as f:
        data = yaml.safe_load(f)
    if "camera_matrix" in data:
        return np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    P = np.array(data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
    return P[:3, :3]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline room mapper (RGB + DA3 depth).")
    p.add_argument("--data-dir",         required=True)
    p.add_argument("--rgb-subdir",       default=".",
                   help="RGB image subdirectory relative to data-dir (default: '.' = flat layout)")
    p.add_argument("--depth-subdir",     default=".",
                   help="Depth .npy subdirectory relative to data-dir (default: '.' = flat layout)")
    p.add_argument("--tag-map",          default=None,
                   help="AprilTag world map YAML. Omit for odometry-only (unscaled) mode.")
    p.add_argument("--output-dir",       default="./room_map_out")
    p.add_argument("--stride",           type=int,   default=1)
    p.add_argument("--labels",           default=None,
                   help="Path to labels JSON. If omitted, NanoOWL detections are loaded "
                        "automatically from JSON sidecars in data-dir.")
    p.add_argument("--min-score",        type=float, default=0.25,
                   help="Min NanoOWL detection score when loading from session JSON sidecars.")
    p.add_argument("--no-session-labels", action="store_true",
                   help="Disable automatic label loading from JSON sidecars.")
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
    p.add_argument("--wall-angle-thresh",  type=float, default=None,
                   help="Only update occupancy grid for frames where the camera views "
                        "the wall within this many degrees of perpendicular (e.g. 10 for "
                        "80-100 deg). Skipped frames still contribute to pose/trajectory. "
                        "Requires --tag-map. Default: disabled (use all frames).")
    p.add_argument("--labels-only-grid",  action="store_true",
                   help="Only update the occupancy grid for labeled frames. "
                        "Guarantees objects are in front of their walls (same-frame DA3 consistency).")
    p.add_argument("--no-convex-walls",  action="store_true",
                   help="Disable convex-hull post-processing of free space. "
                        "By default the free region is convex-hull-filled to fix "
                        "diagonal DA3 corner smear in rectangular rooms.")
    p.add_argument("--flip-y",           action="store_true",
                   help="Flip Y axis in the output map image. Does not affect saved coordinates.")
    p.add_argument("--north-up",         action="store_true",
                   help="Rotate display so geographic north is up (display_x=−world_y, "
                        "display_y=world_x). Use when the world frame has +X pointing north "
                        "and CCW flight appears CW in the default view.")
    p.add_argument("--show-traj",         action="store_true",
                   help="Draw the drone trajectory on the map (hidden by default).")
    p.add_argument("--show-tag-fixes",    action="store_true",
                   help="Draw AprilTag fix stars on the map (hidden by default).")
    p.add_argument("--sidecar-pose",     action="store_true",
                   help="Use pose from JSON sidecar files (x,y,z,yaw) instead of "
                        "AprilTag/odometry re-estimation. Use when the capture session "
                        "already has good live-flight poses saved alongside each frame.")
    p.add_argument("--no-raytrace",      action="store_true",
                   help="Disable ray-cast free-space (faster but no wall contrast)")
    p.add_argument("--preview",          action="store_true",
                   help="Show live map preview window during processing")
    p.add_argument("--preview-interval", type=int,   default=10,
                   help="Update preview every N processed frames")
    return p.parse_args()


def _sidecar_pose_to_world_T_cam(pose: dict) -> np.ndarray:
    """
    Build a 4x4 world_T_cam matrix from a sidecar pose dict {x, y, z, yaw}.

    Convention:
      World frame  : ROS (X=east / forward, Y=left, Z=up)
      Camera frame : OpenCV (Z=forward, X=right, Y=down)
      yaw          : CCW rotation from +X, stored in DEGREES in the sidecar JSON
                     (dome_main converts radians→degrees before saving)
    """
    x, y, z = pose["x"], pose["y"], pose["z"]
    yaw = math.radians(pose["yaw"])   # sidecar stores degrees
    c, s = np.cos(yaw), np.sin(yaw)
    # Columns = camera X/Y/Z axes expressed in world coordinates
    R = np.array([
        [ s,  0,  c],   # world-X components of [X_cam, Y_cam, Z_cam]
        [-c,  0,  s],   # world-Y components
        [ 0, -1,  0],   # world-Z components
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


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
    frame_world_anchored: Dict[int, bool] = {}
    frame_depth_scales: Dict[int, float] = {}
    depth_cache: Dict[int, np.ndarray] = {}
    _prev_tag_set: frozenset = frozenset()

    # Load detection labels — from explicit file, or auto from session JSON sidecars.
    _raw_labels: list = []
    if args.labels and Path(args.labels).exists():
        _raw_labels = load_labels(args.labels)
        print(f"[mapper] {len(_raw_labels)} labels from {args.labels}")
    elif not args.no_session_labels:
        _raw_labels = load_labels_from_session(
            Path(args.data_dir), args.min_score, DEPTH_H, DEPTH_W)
        if _raw_labels:
            print(f"[mapper] {len(_raw_labels)} detections auto-loaded from session JSON sidecars "
                  f"(min_score={args.min_score})")
    _has_labels = bool(_raw_labels)

    labeled_frame_idxs: Optional[set] = None
    # frame_idx → list of (x, y, w, h) bboxes in depth-image coordinates
    frame_bbox_masks: Dict[int, List[Tuple[int, int, int, int]]] = {}
    if _has_labels:
        if args.labels_only_grid:
            labeled_frame_idxs = {int(e["frame_idx"]) for e in _raw_labels}
            print(f"[mapper] labels-only-grid: will update grid for "
                  f"{len(labeled_frame_idxs)} labeled frames only")
        for _e in _raw_labels:
            _fidx = int(_e["frame_idx"])
            _bx, _by, _bw, _bh = _e["bbox"]
            _sw, _sh = _e.get("source_size", [DEPTH_W, DEPTH_H])
            _dx = int(_bx * DEPTH_W / _sw); _dy = int(_by * DEPTH_H / _sh)
            _dw = int(_bw * DEPTH_W / _sw); _dh = int(_bh * DEPTH_H / _sh)
            frame_bbox_masks.setdefault(_fidx, []).append((_dx, _dy, _dw, _dh))
        if frame_bbox_masks:
            print(f"[mapper] bbox masking: {len(frame_bbox_masks)} labeled frames")

    frames = list(iter_frames(Path(args.data_dir), stride=args.stride,
                              rgb_subdir=args.rgb_subdir, depth_subdir=args.depth_subdir))
    print(f"[mapper] {len(frames)} frames  stride={args.stride}  "
          f"raytrace={'off' if args.no_raytrace else 'on'}  "
          f"preview={'on' if args.preview else 'off'}")

    if args.preview:
        cv2.namedWindow("Room Mapper", cv2.WINDOW_NORMAL)

    prev_n_fixes = 0
    n_grid_frames = 0
    for i, rec in enumerate(frames):
        bgr     = rec.load_rgb()
        depth_m = rec.load_depth()
        print("[mapper] processing frame {}".format(i))

        if args.sidecar_pose and rec.pose is not None:
            world_T_cam = _sidecar_pose_to_world_T_cam(rec.pose)
            frame_tag_ids[rec.frame_idx] = []
            frame_tag_areas[rec.frame_idx] = 0.0
        else:
            world_T_cam = fuser.update(bgr, depth_m)
            frame_tag_ids[rec.frame_idx] = list(fuser.last_tag_ids)
            frame_tag_areas[rec.frame_idx] = fuser.last_tag_total_area

        if world_T_cam is None:
            continue

        # Per-frame DA3 scale correction from AprilTag ground truth.
        # The same linear scale applies to objects and walls alike — the metric
        # DA3 model has a proportional error that the tag-based scale corrects.
        if not args.no_scale_correction and fuser.last_depth_scale is not None:
            depth_m = depth_m * fuser.last_depth_scale
        frame_depth_scales[rec.frame_idx] = fuser.last_depth_scale or 1.0

        # Check if this frame is near-perpendicular to the wall (angle filtering)
        _angle_ok = True
        if args.wall_angle_thresh is not None:
            angle = fuser.last_wall_view_angle
            _angle_ok = (angle is not None and angle <= args.wall_angle_thresh)

        if _angle_ok and (labeled_frame_idxs is None or rec.frame_idx in labeled_frame_idxs):
            n_grid_frames += 1
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
        frame_world_anchored[rec.frame_idx] = (fuser.n_tag_fixes > 0)

        # Only mark a star when the visible tag set changes (not every frame)
        current_tag_set = frozenset(fuser.last_tag_ids)
        if current_tag_set and current_tag_set != _prev_tag_set:
            tag_fix_positions.append((wx, wy))
            _prev_tag_set = current_tag_set
        prev_n_fixes = fuser.n_tag_fixes

        if _has_labels:
            depth_cache[rec.frame_idx] = depth_m.copy()

        if (i + 1) % 20 == 0:
            scale_str = f"{fuser.last_depth_scale:.3f}" if fuser.last_depth_scale else "none"
            print(f"[mapper] {i+1}/{len(frames)}  tag_fixes={fuser.n_tag_fixes}  "
                  f"traj_pts={len(trajectory)}  depth_scale={scale_str}")

        if args.preview and (i + 1) % args.preview_interval == 0:
            title = f"Frame {rec.frame_idx}  fixes={fuser.n_tag_fixes}"
            if not _show_preview(grid, trajectory, [], tag_fix_positions, title):
                print("[mapper] Preview quit by user.")
                break

    angle_str = f"  wall_angle_thresh={args.wall_angle_thresh}°  grid_frames={n_grid_frames}" \
                if args.wall_angle_thresh is not None else ""
    print(f"[mapper] Done. tag_fixes={fuser.n_tag_fixes}  frames_with_pose={len(frame_poses)}{angle_str}")

    objects: List[ObjectMarker] = []
    if _has_labels:
        # Sidecar poses come from the live AprilTag localization — they are already
        # reliable, so suppress the per-frame tag-visibility check inside place_objects
        # by passing frame_tag_ids=None (None means "skip reliability filter").
        _ftids = None if args.sidecar_pose else frame_tag_ids
        _fanchored = None if args.sidecar_pose else frame_world_anchored
        _fscales = None if args.sidecar_pose else frame_depth_scales
        _tag_xyz = None if args.sidecar_pose else fuser.tag_world_xyz
        objects = place_objects(_raw_labels, frame_poses, lambda fidx: depth_cache[fidx],
                                K_depth, _ftids, frame_tag_areas,
                                frame_depth_scales=_fscales, tag_world_xyz=_tag_xyz,
                                frame_world_anchored=_fanchored)
        print(f"[mapper] Placed {len(objects)} raw object markers")

        # In sidecar-pose mode all frames have confidence=0.1 (no tag fixes), which
        # would expand the cluster radius to 5m and merge objects from opposite walls.
        # Cap at 1x (base radius) so clustering stays local.
        _max_r_scale = 1.0 if args.sidecar_pose else None
        _cluster_kwargs = {"radius_m": args.cluster_radius}
        if _max_r_scale is not None:
            _cluster_kwargs["max_radius_scale"] = _max_r_scale

        # Snap objects that DA3 pushed into wall cells back to free space, then cluster.
        objects = snap_objects_to_free_space(objects, frame_poses, grid)
        objects = cluster_objects(objects, **_cluster_kwargs)
        print(f"[mapper] After clustering ({args.cluster_radius}m): {len(objects)} objects")
        if args.tag_map is not None and not args.sidecar_pose:
            objects = flag_objects_beyond_tags(
                objects, frame_poses, fuser.tag_world_xyz, frame_tag_ids, tolerance_m=1.0)
        objects = flag_objects_outside_map(objects, grid)
        # Remove objects that landed outside the observed room volume.
        n_before = len(objects)
        objects = [o for o in objects if not o.suspicious]
        if len(objects) < n_before:
            print(f"[mapper] Removed {n_before - len(objects)} suspicious object(s) outside observed room")
        for obj in objects:
            print(f"         {obj.label:20s}  "
                  f"({obj.world_x:+.2f}, {obj.world_y:+.2f}, {obj.world_z:+.2f})m  "
                  f"tags={obj.tag_ids}")

        save_objects_json(objects, str(out_dir / "objects.json"))
        print(f"[mapper] Saved {len(objects)} object poses to {out_dir / 'objects.json'}")

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
        trajectory_world=traj_smooth if args.show_traj else [],
        objects=objects,
        output_path=str(out_dir / "map_with_objects.png"),
        title=map_title,
        tag_fixes=tag_fix_positions if (args.show_tag_fixes and tag_fix_positions) else None,
        convex_walls=not args.no_convex_walls,
        flip_y=args.flip_y,
        north_up=args.north_up,
    )

    if args.preview:
        cv2.destroyAllWindows()

    print(f"[mapper] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()