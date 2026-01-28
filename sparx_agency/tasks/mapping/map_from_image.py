import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import yaml

from sparx_agency.core.mapping.costmap.probabilistic_grid_config import bresenham
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class MapCfg:
    resolution_m: float
    grid_width_m: float
    grid_height_m: float
    inference_width: int
    inference_height: int
    stride: int
    floor_ransac_iters: int
    floor_dist_thresh: float
    free_height_m: float
    occupied_height_m: float

    # debug / heuristics
    floor_bottom_frac: float = 0.45       # use bottom % of image for plane candidates
    min_plane_inliers: int = 600          # reject tiny planes
    normal_gate_abs_z: float = 0.0        # 0 disables; else require abs(n[2]) >= this


def load_yaml(path: str) -> Tuple[Intrinsics, MapCfg]:
    d = yaml.safe_load(Path(path).read_text())
    cam = d["camera"]
    mp = d["mapping"]
    intr = Intrinsics(
        width=int(cam["width"]),
        height=int(cam["height"]),
        fx=float(cam["fx"]),
        fy=float(cam["fy"]),
        cx=float(cam["cx"]),
        cy=float(cam["cy"]),
    )
    cfg = MapCfg(
        resolution_m=float(mp["resolution_m"]),
        grid_width_m=float(mp["grid_width_m"]),
        grid_height_m=float(mp["grid_height_m"]),
        inference_width=int(mp["inference_width"]),
        inference_height=int(mp["inference_height"]),
        stride=int(mp["stride"]),
        floor_ransac_iters=int(mp["floor_ransac_iters"]),
        floor_dist_thresh=float(mp["floor_dist_thresh"]),
        free_height_m=float(mp["free_height_m"]),
        occupied_height_m=float(mp["occupied_height_m"]),
        floor_bottom_frac=float(mp.get("floor_bottom_frac", 0.45)),
        min_plane_inliers=int(mp.get("min_plane_inliers", 600)),
        normal_gate_abs_z=float(mp.get("normal_gate_abs_z", 0.0)),
    )
    return intr, cfg


def resize_intrinsics(intr: Intrinsics, new_w: int, new_h: int) -> Intrinsics:
    sx = new_w / intr.width
    sy = new_h / intr.height
    return Intrinsics(
        width=new_w,
        height=new_h,
        fx=intr.fx * sx,
        fy=intr.fy * sy,
        cx=intr.cx * sx,
        cy=intr.cy * sy,
    )


def depth_to_pointcloud_sparse(depth_m: np.ndarray,
                               intr: Intrinsics,
                               stride: int,
                               valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    xv, yv = np.meshgrid(xs, ys)

    z = depth_m[yv, xv].reshape(-1).astype(np.float32)
    x = (xv.reshape(-1).astype(np.float32) - intr.cx) * z / intr.fx
    y = (yv.reshape(-1).astype(np.float32) - intr.cy) * z / intr.fy
    pts = np.stack([x, y, z], axis=1)

    y_pix = yv.reshape(-1).astype(np.int32)
    good = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-6)
    m = good
    if valid_mask is not None:
        m = m & valid_mask[yv, xv].reshape(-1)
    return pts[m], y_pix[m]


def fit_floor_plane_ransac(
    pts: np.ndarray,
    iters: int,
    dist_thresh: float,
    min_inliers: int,
    normal_gate_abs_z: float = 0.0,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Returns (n, d, inlier_mask) for plane n·p + d = 0.
    """
    if pts.shape[0] < 500:
        return None

    rng = np.random.default_rng(seed)
    best_score = -1e18
    best = None

    for _ in range(iters):
        idx = rng.choice(pts.shape[0], size=3, replace=False)
        p1, p2, p3 = pts[idx]
        n = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(float(n[1])) < 0.75:
            continue
        # Optional orientation gate (DISABLED by default)
        if normal_gate_abs_z > 0.0 and abs(float(n[2])) < normal_gate_abs_z:
            continue

        d = -float(np.dot(n, p1))

        dist = np.abs(pts @ n + d)
        inlier_mask = dist < dist_thresh
        inliers = int(inlier_mask.sum())
        if inliers < min_inliers:
            continue

        med = float(np.median(dist[inlier_mask]))
        # score prefers many inliers and small median distance
        score = float(inliers) - 2000.0 * med

        if score > best_score:
            best_score = score
            best = (n, d, inlier_mask)

    return best


def refine_plane_svd(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Best-fit plane (least squares) through points.
    """
    c = np.mean(pts, axis=0)
    A = pts - c
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-9)
    d = -float(np.dot(n, c))
    return n.astype(np.float32), float(d)


def plane_basis(n: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Project cam_x onto plane to get a stable 'u'
    u = cam_x - float(np.dot(cam_x, n)) * n
    u = u / (np.linalg.norm(u) + 1e-9)

    # v completes the basis on the plane
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-9)
    return u, v

def plane_basis_from_camera(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (u,v) axes on the floor plane aligned with the camera:
    - v = camera forward projected onto plane
    - u = camera right projected onto plane
    Assumes camera frame: x=right, y=down, z=forward.
    """
    n = n.astype(np.float32)
    n = n / (np.linalg.norm(n) + 1e-9)

    f = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # camera forward (z)
    r = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # camera right (x)

    # project onto plane
    v = f - (np.dot(f, n)) * n
    u = r - (np.dot(r, n)) * n

    v = v / (np.linalg.norm(v) + 1e-9)
    u = u / (np.linalg.norm(u) + 1e-9)

    # re-orthogonalize u to v
    u = u - (np.dot(u, v)) * v
    u = u / (np.linalg.norm(u) + 1e-9)

    return u, v


def build_occupancy_raycast(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    u: np.ndarray,
    v: np.ndarray,
    cfg: MapCfg,
    debug: bool = True,
) -> np.ndarray:
    """
    Better occupancy-from-single-view:
    1) Project points onto floor plane coords (u,v).
    2) Build per-cell max height grid.
    3) Mark occupied cells by height threshold.
    4) Raycast ONLY to occupied cells to carve free space.
    Map size is computed dynamically from data + sensor.
    """

    # Signed distance to plane: positive = above floor
    signed = (pts @ n + d).astype(np.float32)
    height = signed.copy()
    height[height < 0.0] = 0.0

    # Project points onto plane coords
    p_proj = pts - signed[:, None] * n[None, :]
    xp = (p_proj @ u).astype(np.float32)
    yp = (p_proj @ v).astype(np.float32)

    front = (yp > 0.0)
    xp = xp[front]
    yp = yp[front]
    height = height[front]

    # Sensor projection on plane: closest point to camera origin is p0 = -d*n
    p0 = (-d) * n
    sx = float(p0 @ u)
    sy = float(p0 @ v)

    # ---- Dynamic map bounds (include BOTH endpoints and sensor) ----
    margin = 2.0  # meters
    min_x = float(np.percentile(xp, 1))
    max_x = float(np.percentile(xp, 99))
    min_y = float(np.percentile(yp, 1))
    max_y = float(np.percentile(yp, 99))

    min_x = min(min_x, sx) - margin
    max_x = max(max_x, sx) + margin
    min_y = min(min_y, sy) - margin
    max_y = max(max_y, sy) + margin

    span_x = max_x - min_x
    span_y = max_y - min_y

    gw = int(np.ceil(span_x / cfg.resolution_m))
    gh = int(np.ceil(span_y / cfg.resolution_m))

    # Safety cap (avoid accidental huge allocations)
    max_cells = 2500  # -> 2500x2500 is already massive
    gw = int(np.clip(gw, 50, max_cells))
    gh = int(np.clip(gh, 50, max_cells))

    origin_x = min_x
    origin_y = min_y

    # Allocate
    grid = np.full((gh, gw), -1, dtype=np.int8)

    # Convert to cell indices
    ix = np.floor((xp - origin_x) / cfg.resolution_m).astype(np.int32)
    iy = np.floor((yp - origin_y) / cfg.resolution_m).astype(np.int32)

    valid = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh) & np.isfinite(height)
    ixv = ix[valid]
    iyv = iy[valid]
    hv = height[valid]

    # Sensor cell
    s_ix = int(np.floor((sx - origin_x) / cfg.resolution_m))
    s_iy = int(np.floor((sy - origin_y) / cfg.resolution_m))

    if debug:
        print("MAP(ray2) dbg:")
        print(f"  grid: {gw} x {gh} res: {cfg.resolution_m}")
        print(f"  bounds x:[{origin_x:.3f},{(origin_x+gw*cfg.resolution_m):.3f}]  y:[{origin_y:.3f},{(origin_y+gh*cfg.resolution_m):.3f}]")
        print(f"  sensor (u,v): ({sx:.3f}, {sy:.3f}) -> cell ({s_ix},{s_iy}) in-bounds={0 <= s_ix < gw and 0 <= s_iy < gh}")
        print(f"  valid points: {ixv.size} / {ix.size}")
        print(f"  height(valid) p50/p90/p99: ({float(np.percentile(hv,50)):.3f}, {float(np.percentile(hv,90)):.3f}, {float(np.percentile(hv,99)):.3f})")

    if not (0 <= s_ix < gw and 0 <= s_iy < gh):
        # This should not happen with the bounds logic, but keep safe.
        if debug:
            print("  Sensor out of bounds even after bounds include it; check plane orientation.")
        return grid

    # ---- 1) Per-cell max height (this is the key improvement) ----
    maxH = np.full((gh, gw), -np.inf, dtype=np.float32)
    np.maximum.at(maxH, (iyv, ixv), hv)

    observed = np.isfinite(maxH)
    occ_cells = observed & (maxH >= cfg.occupied_height_m)
    free_cells = observed & (maxH <= cfg.free_height_m)

    # Seed floor-ish cells as free (optional but helpful)
    grid[free_cells] = 0

    # ---- 2) Raycast ONLY to occupied cells ----
    occ_y, occ_x = np.where(occ_cells)

    free_written = int((grid == 0).sum())
    occ_written = 0
    rays = 0

    for ex, ey in zip(occ_x.tolist(), occ_y.tolist()):
        rays += 1
        cells = list(bresenham(s_ix, s_iy, ex, ey))
        if not cells:
            continue

        # Free along ray except endpoint
        for cx, cy in cells[:-1]:
            if grid[cy, cx] == -1:
                grid[cy, cx] = 0
                free_written += 1

        # Endpoint occupied
        lx, ly = cells[-1]
        if grid[ly, lx] != 100:
            grid[ly, lx] = 100
            occ_written += 1

    if debug:
        uniq, cnt = np.unique(grid, return_counts=True)
        print(f"  occupied targets: {occ_x.size} rays:{rays} free_written:{free_written} occ_written:{occ_written}")
        print("  GRID unique:", list(zip(uniq.tolist(), cnt.tolist())))

    return grid


def build_occupancy_corner_range(pts: np.ndarray, n: np.ndarray, d: float, u: np.ndarray, v: np.ndarray, cfg: MapCfg):
    gw = int(cfg.grid_width_m / cfg.resolution_m)
    gh = int(cfg.grid_height_m / cfg.resolution_m)

    min_h = np.full((gh, gw), np.inf, dtype=np.float32)
    max_h = np.full((gh, gw), -np.inf, dtype=np.float32)
    cnt   = np.zeros((gh, gw), dtype=np.int32)

    signed = pts @ n + d
    # orient so "above plane" is positive
    if float(np.median(signed)) < 0.0:
        signed = -signed
    h = signed.copy()
    h[h < 0.0] = 0.0

    p_proj = pts - (signed[:, None] * n[None, :])
    x = p_proj @ u
    y = p_proj @ v

    margin = 2.0
    origin_x = float(np.percentile(x, 1)) - margin
    origin_y = float(np.percentile(y, 1)) - margin

    ix = np.floor((x - origin_x) / cfg.resolution_m).astype(np.int32)
    iy = np.floor((y - origin_y) / cfg.resolution_m).astype(np.int32)

    valid = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh)
    ixv, iyv, hv = ix[valid], iy[valid], h[valid]

    np.add.at(cnt, (iyv, ixv), 1)
    np.minimum.at(min_h, (iyv, iyv*0 + ixv), hv)  # see note below

    np.maximum.at(max_h, (iyv, ixv), hv)

    grid = np.full((gh, gw), -1, dtype=np.int8)
    observed = cnt >= 2

    h_range = max_h - min_h

    # free if the highest thing is still basically floor
    grid[observed & (max_h <= cfg.free_height_m)] = 0

    # occupied if there is a vertical structure in the cell
    grid[observed & (h_range >= cfg.occupied_height_m)] = 100

    dbg = {
        "gw": gw, "gh": gh,
        "origin_x": origin_x, "origin_y": origin_y,
        "x_p01_p99": (float(np.percentile(x, 1)), float(np.percentile(x, 99))),
        "y_p01_p99": (float(np.percentile(y, 1)), float(np.percentile(y, 99))),
        "cells_observed": int(observed.sum()),
        "cells_free": int((grid == 0).sum()),
        "cells_occ": int((grid == 100).sum()),
        "valid_points": int(valid.sum()),
        "total_points": int(pts.shape[0]),
    }
    print("cell h_range p50/p90/p99:",
          np.percentile(h_range[observed], [50, 90, 99]).tolist() if np.any(observed) else None)

    return grid, max_h, cnt, dbg


def occ_to_png(grid: np.ndarray) -> np.ndarray:
    img = np.full(grid.shape, 127, dtype=np.uint8)
    img[grid >= 50] = 0
    img[grid == 0] = 255
    return img


def depth_vis_u8(depth_m: np.ndarray) -> Tuple[np.ndarray, dict]:
    p01, p50, p99 = np.percentile(depth_m[np.isfinite(depth_m)], [1, 50, 99]).tolist()
    d = np.clip((depth_m - p01) / max(p99 - p01, 1e-6), 0.0, 1.0)
    vis = (d * 255.0).astype(np.uint8)
    dbg = {
        "raw_p01_p50_p99": (float(p01), float(p50), float(p99)),
        "raw_minmax": (float(np.min(depth_m)), float(np.max(depth_m))),
    }
    return vis, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    intr_full, cfg = load_yaml(args.config)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb_inf = cv2.resize(rgb, (cfg.inference_width, cfg.inference_height), interpolation=cv2.INTER_AREA)
    intr = resize_intrinsics(intr_full, cfg.inference_width, cfg.inference_height)

    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())
    depth_m = depth_model.infer_depth(rgb_inf).astype(np.float32)  # MUST be meters

    max_r = DepthAnythingV2Config().max_range_m
    min_r = DepthAnythingV2Config().min_range_m
    valid_depth = (
            np.isfinite(depth_m) &
            (depth_m > (min_r + 0.05)) &
            (depth_m < (max_r - 0.5))
    )

    sat_pct = float((depth_m >= (max_r - 0.5)).mean()) * 100.0
    print(f"DEPTH sat_pct_near_max: {sat_pct:.2f}%")
    print("DEPTH:",
          "shape", depth_m.shape,
          "dtype", depth_m.dtype,
          "min/max", float(np.min(depth_m)), float(np.max(depth_m)))

    vis, vis_dbg = depth_vis_u8(depth_m)
    print("DEPTH dbg:", vis_dbg)
    cv2.imwrite(os.path.join(args.out_dir, "depth_vis.png"), vis)

    pts, y_pix = depth_to_pointcloud_sparse(depth_m, intr, cfg.stride)
    print(f"PTS: total={pts.shape[0]} stride={cfg.stride} (inf={cfg.inference_width}x{cfg.inference_height})")

    y_thr = int((1.0 - cfg.floor_bottom_frac) * intr.height)
    cand = y_pix >= y_thr
    pts_floor = pts[cand]
    print(f"FLOOR cand: bottom_frac={cfg.floor_bottom_frac} y_thr={y_thr} pts_floor={pts_floor.shape[0]}")

    plane = fit_floor_plane_ransac(
        pts_floor,
        iters=cfg.floor_ransac_iters,
        dist_thresh=cfg.floor_dist_thresh,
        min_inliers=cfg.min_plane_inliers,
        normal_gate_abs_z=cfg.normal_gate_abs_z,
    )
    if plane is None:
        raise RuntimeError("Floor plane fit failed (no good hypothesis).")

    n0, d0, inliers0 = plane
    if n0[1] > 0:
        n0 = -n0
        d0 = -d0
    med0 = float(np.median(np.abs(pts_floor[inliers0] @ n0 + d0)))
    print(f"PLANE best: inliers={int(inliers0.sum())}/{pts_floor.shape[0]} med_dist={med0:.4f} n={n0} d={d0:.3f}")

    # refine with SVD on inliers
    n, d = refine_plane_svd(pts_floor[inliers0])
    if n[1] > 0:
        n = -n
        d = -d
    med = float(np.median(np.abs(pts_floor[inliers0] @ n + d)))
    print(f"PLANE refine: n={n} d={d:.3f} med_dist(inliers)={med:.4f}")

    u, v = plane_basis_from_camera(n)


    # grid, max_h, cnt, dbg = build_occupancy_corner_range(pts, n, d, u, v, cfg)
    occ = build_occupancy_raycast(pts, n, d, u, v, cfg, debug=True)

    # print("MAP dbg:")
    # print(f"  grid: {dbg['gw']} x {dbg['gh']} res: {cfg.resolution_m}")
    # print(f"  origin: ({dbg['origin_x']:.3f}, {dbg['origin_y']:.3f}) mode: corner")
    # print(f"  valid points: {dbg['valid_points']} / {dbg['total_points']}")
    # print(f"  x p01/p99: {dbg['x_p01_p99']}")
    # print(f"  y p01/p99: {dbg['y_p01_p99']}")
    # print(f"  cells observed/free/occ: {dbg['cells_observed']} {dbg['cells_free']} {dbg['cells_occ']}")
    #
    # uniq, cntu = np.unique(grid, return_counts=True)
    # print("GRID unique:", list(zip(uniq.tolist(), cntu.tolist())))

    cv2.imwrite(os.path.join(args.out_dir, "occ_map.png"), occ_to_png(occ))
    print("Saved outputs to:", args.out_dir)


if __name__ == "__main__":
    main()
