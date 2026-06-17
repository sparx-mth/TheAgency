"""Place manually-labelled objects onto the 2D map via depth backprojection."""
from __future__ import annotations

import json
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


def place_objects(
    labels: list,
    frame_poses: Dict[int, np.ndarray],
    depth_loader: Callable[[int], np.ndarray],
    K_depth: np.ndarray,
    frame_tag_ids: Optional[Dict[int, List[int]]] = None,
    frame_tag_areas: Optional[Dict[int, float]] = None,
) -> List[ObjectMarker]:
    """
    Backproject each labelled bbox centre through depth to a world 3D position.

    bbox coords are in source_size space; scaled to depth resolution before sampling.
    tag_confidence is derived from total tag pixel area at the label frame.
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

        z = float(depth_m[v, u])
        if not np.isfinite(z) or z < 0.1:
            continue

        p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
        p_world = frame_poses[fidx] @ p_cam

        tids = frame_tag_ids.get(fidx, []) if frame_tag_ids else []
        area = frame_tag_areas.get(fidx, 0.0) if frame_tag_areas else 0.0
        confidence = float(np.clip(area / _REFERENCE_TAG_AREA_PX2, 0.1, 1.0))

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


def cluster_objects(
    markers: List[ObjectMarker],
    radius_m: float = 1.0,
) -> List[ObjectMarker]:
    """
    Merge same-label markers within an effective radius that scales with
    tag confidence: low-confidence fixes use a larger merge radius.

    Effective radius = min(base * MAX_SCALE, base / anchor_confidence).
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
            eff_r = min(radius_m * _MAX_RADIUS_SCALE,
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