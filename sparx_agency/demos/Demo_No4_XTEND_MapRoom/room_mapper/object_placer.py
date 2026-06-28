"""Place manually-labelled objects onto the 2D map via depth backprojection."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# Total tag pixel area that counts as "full confidence" (both tags ~2000px² each)
_REFERENCE_TAG_AREA_PX2 = 4000.0
# Cluster radius is scaled up to this multiplier for low-confidence fixes
_MAX_RADIUS_SCALE = 2.5


@dataclass
class ObjectMarker:
    label: str
    frame_idx: int
    world_x: float
    world_y: float
    world_z: float
    tag_ids: List[int] = field(default_factory=list)
    tag_confidence: float = 1.0   # 0–1: based on total tag pixel area at label frame
    suspicious: bool = False       # True if far from trajectory — placement may be wrong


def load_labels(labels_path: str) -> list:
    """
    Load labels JSON.

    Expected format:
        [{"frame_idx": 350, "bbox": [x, y, w, h], "source_size": [w, h], "label": "chair"}, ...]
    """
    with open(labels_path) as f:
        return json.load(f)


def load_labels_from_session(
    session_dir,
    min_score: float = 0.25,
    depth_h: int = 294,
    depth_w: int = 504,
) -> list:
    """
    Extract NanoOWL detections directly from JSON sidecar files in a flat session dir.

    Iterates sorted JPGs that have a matching .npy (mirroring iter_frames with stride=1),
    so frame_idx values align with those assigned by run_room_mapper.
    """
    from pathlib import Path as _Path

    session_dir = _Path(session_dir)
    depth_stems = {p.stem for p in session_dir.glob("*.npy")}
    jpgs = sorted(
        [p for p in session_dir.glob("*.jpg") if p.stem in depth_stems],
        key=lambda p: p.stem,
    )

    labels = []
    for frame_idx, jpg in enumerate(jpgs):
        json_path = jpg.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        nanoowl = data.get("nanoowl", {})
        result = nanoowl.get("result", {})
        dets = result.get("detections", [])
        img_info = result.get("image", {})
        src_w = img_info.get("width", depth_w)
        src_h = img_info.get("height", depth_h)
        for det in dets:
            if det.get("score", 0) < min_score:
                continue
            x1, y1, x2, y2 = det["bbox"]
            label = det.get("label", "object").removeprefix("a ").removeprefix("an ")
            labels.append({
                "frame_idx": frame_idx,
                "label": label,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "source_size": [src_w, src_h],
                "score": round(float(det.get("score", 1.0)), 4),
            })
    return labels


def place_objects(
    labels: list,
    frame_poses: Dict[int, np.ndarray],
    depth_loader: Callable[[int], np.ndarray],
    K_depth: np.ndarray,
    frame_tag_ids: Optional[Dict[int, List[int]]] = None,
    frame_tag_areas: Optional[Dict[int, float]] = None,
    frame_depth_scales: Optional[Dict[int, float]] = None,
    tag_world_xyz: Optional[Dict[int, np.ndarray]] = None,
    frame_world_anchored: Optional[Dict[int, bool]] = None,
) -> List[ObjectMarker]:
    """
    Backproject each labelled bbox centre through depth to a world 3D position.

    bbox coords are in source_size space; scaled to depth resolution before sampling.
    tag_confidence is derived from total tag pixel area at the label frame.
    frame_depth_scales + tag_world_xyz: if provided, object raw depth is capped to the
    raw DA3 distance of the nearest aligned visible tag (tags are on walls — no object
    can be further than the wall in the same direction).
    """
    fx, fy, cx, cy = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    markers: List[ObjectMarker] = []

    for entry in labels:
        fidx = int(entry["frame_idx"])
        if fidx not in frame_poses:
            continue

        bx, by, bw, bh = entry["bbox"]
        u_src = bx + bw / 2
        v_src = by + bh / 2

        depth_m = depth_loader(fidx)
        dh, dw = depth_m.shape[:2]

        src_w, src_h = entry.get("source_size", [dw, dh])
        u = int(u_src * dw / src_w)
        v = int(v_src * dh / src_h)
        u = max(0, min(u, dw - 1))
        v = max(0, min(v, dh - 1))

        # Find the first coherent depth cluster in the bbox — the object surface.
        # Sort depths ascending, find the first jump > gap_thresh (background starts there),
        # use the median of everything before the jump.
        half_w = max(1, int(bw * dw / src_w) // 4)
        half_h = max(1, int(bh * dh / src_h) // 4)
        u0 = max(0, u - half_w); u1 = min(dw, u + half_w + 1)
        v0 = max(0, v - half_h); v1 = min(dh, v + half_h + 1)
        patch = depth_m[v0:v1, u0:u1].ravel()
        valid_px = np.sort(patch[np.isfinite(patch) & (patch > 0.1)])
        if valid_px.size == 0:
            continue
        # Object pixels are always closer than background wall pixels in raw DA3.
        # Take the nearest 10% of the inner patch (min 3, max 25) and use their
        # median — reliably lands on the object front surface for both large objects
        # (fridge fills most of the patch) and thin ones (gun occupies ~10% of patch).
        n_front = max(3, min(25, max(1, int(valid_px.size * 0.10))))
        z = float(np.median(valid_px[:n_front]))

        tids = frame_tag_ids.get(fidx, []) if frame_tag_ids else []

        # Cap raw depth so the object can't be placed past the wall.
        # Tags sit on walls — compute the raw DA3 equivalent distance to each visible
        # tag and cap z to the minimum aligned one (minus a 5cm safety margin).
        if frame_depth_scales and tag_world_xyz and tids:
            scale = frame_depth_scales.get(fidx, 1.0)
            if scale > 0:
                world_T_cam = frame_poses[fidx]
                cam_T_world = np.linalg.inv(world_T_cam)
                obj_ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
                obj_ray = obj_ray / np.linalg.norm(obj_ray)
                z_cap = None
                for t in tids:
                    if t not in tag_world_xyz:
                        continue
                    tag_cam = (cam_T_world @ np.append(tag_world_xyz[t], 1.0))[:3]
                    tag_metric = float(tag_cam[2])
                    if tag_metric <= 0:
                        continue
                    tag_raw = tag_metric / scale
                    tag_ray = tag_cam / np.linalg.norm(tag_cam)
                    if float(np.dot(obj_ray, tag_ray)) >= 0.3:  # within ~72° of object direction
                        if z_cap is None or tag_raw < z_cap:
                            z_cap = tag_raw
                if z_cap is not None and z > z_cap - 0.05:
                    print(f"  [cap] '{entry['label']}' frame={fidx} "
                          f"z={z:.2f}→{z_cap - 0.05:.2f} (tag_raw={z_cap:.2f})")
                    z = z_cap - 0.05

        p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
        p_world = frame_poses[fidx] @ p_cam

        area = frame_tag_areas.get(fidx, 0.0) if frame_tag_areas else 0.0
        confidence = float(np.clip(area / _REFERENCE_TAG_AREA_PX2, 0.1, 1.0))

        # Skip frames with no tag visible, unless the pose is world-anchored from a
        # prior tag fix (odom-propagated over a few frames is still reliable).
        if frame_tag_ids is not None and not tids:
            anchored = frame_world_anchored.get(fidx, False) if frame_world_anchored else False
            if not anchored:
                print(f"  [skip] '{entry['label']}' frame={fidx} — no tag ever seen, pose unreliable")
                continue
            print(f"  [odom] '{entry['label']}' frame={fidx} — odom-propagated (no tag this frame)")

        markers.append(ObjectMarker(
            label=entry["label"],
            frame_idx=fidx,
            world_x=float(p_world[0]),
            world_y=float(p_world[1]),
            world_z=float(p_world[2]),
            tag_ids=list(tids),
            tag_confidence=confidence,
        ))

    return markers


def flag_objects_beyond_tags(
    markers: List[ObjectMarker],
    frame_poses: Dict[int, np.ndarray],
    tag_world_xyz: Dict[int, np.ndarray],
    frame_tag_ids: Dict[int, List[int]],
    tolerance_m: float = 0.1,
    min_alignment: float = 0.5,
) -> List[ObjectMarker]:
    """
    Flag objects placed further from the camera than the wall tag used to localize.

    Tags are taped to walls — no real object can be further than a tag that sits
    on the SAME wall (i.e. in the same direction as the object).

    min_alignment: cosine similarity threshold; only tags within ~60° of the
    object direction are used as wall proxies.
    """
    for obj in markers:
        if obj.suspicious:
            continue
        fidx = obj.frame_idx
        if fidx not in frame_poses:
            continue
        cam_xyz = frame_poses[fidx][:3, 3]
        obj_xyz = np.array([obj.world_x, obj.world_y, obj.world_z])
        d_obj = float(np.linalg.norm(obj_xyz - cam_xyz))
        if d_obj < 1e-3:
            continue
        dir_obj = (obj_xyz - cam_xyz) / d_obj

        tids = frame_tag_ids.get(fidx, [])
        d_aligned_wall = None
        for t in tids:
            if t not in tag_world_xyz:
                continue
            tag_xyz = tag_world_xyz[t]
            d_tag = float(np.linalg.norm(tag_xyz - cam_xyz))
            if d_tag < 1e-3:
                continue
            dir_tag = (tag_xyz - cam_xyz) / d_tag
            alignment = float(np.dot(dir_obj[:3], dir_tag[:3]))
            if alignment >= min_alignment:
                if d_aligned_wall is None or d_tag < d_aligned_wall:
                    d_aligned_wall = d_tag

        if d_aligned_wall is None:
            continue

        if d_obj > d_aligned_wall + tolerance_m:
            obj.suspicious = True
            print(f"  [flag] '{obj.label}' beyond wall -- "
                  f"obj={d_obj:.2f}m  aligned_tag_wall={d_aligned_wall:.2f}m  frame={fidx}")
    return markers


def flag_objects_outside_map(
    markers: List[ObjectMarker],
    grid,
) -> List[ObjectMarker]:
    """
    Flag objects whose XY lands in a cell that depth rays never reached.

    A cell with _seen=False means no depth ray ever passed through it —
    the object is completely outside the observed volume of the room.

    Note: objects near walls land in occupied cells (the depth sensor sees their
    surface as a wall), so occupancy value alone cannot distinguish inside vs outside.
    """
    for obj in markers:
        if obj.suspicious:
            continue
        gx, gy = grid._world_to_grid(
            np.array([obj.world_x]), np.array([obj.world_y])
        )
        gxi, gyi = int(gx[0]), int(gy[0])
        outside_bounds = not (0 <= gxi < grid.width and 0 <= gyi < grid.height)
        if outside_bounds or not grid._seen[gyi, gxi]:
            obj.suspicious = True
            print(f"  [flag] '{obj.label}' in unobserved cell "
                  f"({obj.world_x:.1f}, {obj.world_y:.1f})")
    return markers


def snap_objects_to_free_space(
    markers: List[ObjectMarker],
    frame_poses: Dict[int, np.ndarray],
    grid,
    step_m: float = 0.05,
    min_free_run: int = 2,
) -> List[ObjectMarker]:
    """
    Walk each object back along its camera ray until it sits in free space.

    When DA3 depth is slightly overestimated (no tag scale correction), objects
    land in wall cells or beyond the room.  This corrects them to the last free
    cell before the nearest wall along that ray, ensuring every object is inside
    the room and just in front of its wall.

    Objects that are already in free space are not moved.
    Objects whose camera pose is unknown are left unchanged.
    """
    for obj in markers:
        fidx = obj.frame_idx
        if fidx not in frame_poses:
            continue

        cam_x = float(frame_poses[fidx][0, 3])
        cam_y = float(frame_poses[fidx][1, 3])
        obj_x, obj_y = obj.world_x, obj.world_y

        dx = obj_x - cam_x
        dy = obj_y - cam_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            continue
        ux, uy = dx / dist, dy / dist

        # Helper: is a world position inside the grid and in free or unknown space?
        def _is_free(wx: float, wy: float) -> bool:
            gxs, gys = grid._world_to_grid(np.array([wx]), np.array([wy]))
            gxi, gyi = int(gxs[0]), int(gys[0])
            if not (0 <= gxi < grid.width and 0 <= gyi < grid.height):
                return False
            # Free = seen AND negative log-odds; unknown = not seen (also acceptable)
            return (not grid._seen[gyi, gxi]) or (grid._L[gyi, gxi] <= 0.0)

        # If already in free space, nothing to do.
        if _is_free(obj_x, obj_y):
            continue

        # Walk backward from object toward camera until we find min_free_run
        # consecutive free cells, then place the object at the first free one.
        best_x, best_y = None, None
        free_run = 0
        r = dist
        while r > step_m:
            r -= step_m
            tx = cam_x + ux * r
            ty = cam_y + uy * r
            if _is_free(tx, ty):
                if best_x is None:
                    best_x, best_y = tx, ty
                free_run += 1
                if free_run >= min_free_run:
                    break
            else:
                best_x, best_y = None, None
                free_run = 0

        if best_x is not None:
            print(f"  [snap] '{obj.label}' ({obj.world_x:+.2f},{obj.world_y:+.2f})"
                  f" → ({best_x:+.2f},{best_y:+.2f})")
            obj.world_x = best_x
            obj.world_y = best_y

    return markers


def cluster_objects(
    markers: List[ObjectMarker],
    radius_m: float = 1.0,
    max_radius_scale: float = _MAX_RADIUS_SCALE,
) -> List[ObjectMarker]:
    """
    Merge same-label markers within an effective radius that scales with
    tag confidence: low-confidence fixes use a larger merge radius.

    Effective radius = min(base * max_radius_scale, base / anchor_confidence).
    Pass max_radius_scale=1.0 (sidecar-pose mode) to disable confidence scaling
    and keep the base radius fixed regardless of tag confidence.
    """
    result: List[ObjectMarker] = []
    for label in sorted({m.label for m in markers}):
        group = [m for m in markers if m.label == label]
        # Sort highest-confidence first so anchors are the reliable detections
        group.sort(key=lambda m: m.tag_confidence, reverse=True)
        used = [False] * len(group)

        for i, anchor in enumerate(group):
            if used[i]:
                continue
            eff_r = min(radius_m * max_radius_scale,
                        radius_m / max(anchor.tag_confidence, 0.1))
            cluster = [anchor]
            used[i] = True
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                dx = group[j].world_x - anchor.world_x
                dy = group[j].world_y - anchor.world_y
                if dx * dx + dy * dy <= eff_r * eff_r:
                    cluster.append(group[j])
                    used[j] = True

            best = max(cluster, key=lambda m: m.tag_confidence)
            all_tags = sorted({t for m in cluster for t in m.tag_ids})
            avg_conf = float(np.mean([m.tag_confidence for m in cluster]))
            result.append(ObjectMarker(
                label=label,
                frame_idx=best.frame_idx,
                world_x=float(np.median([m.world_x for m in cluster])),
                world_y=float(np.median([m.world_y for m in cluster])),
                world_z=float(np.median([m.world_z for m in cluster])),
                tag_ids=all_tags,
                tag_confidence=avg_conf,
            ))
    return result