import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import yaml

from sparx_agency.core.mapping.costmap.probabilistic_grid_config import bresenham
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config
from sparx_agency.core.mapping.pipeline.mapping_pipeline import PinholeCloudGenerator
from sparx_agency.core.mapping.depth.depth_tiling import TileCfg, infer_depth_tiled
from sparx_agency.robots.common.spatial_math import rot_y_deg
from sparx_agency.tasks.mapping.common.helper import depth_compare_report, save_depth_diff_visuals


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
    up_axis: int = 2,
    up_gate: float = 0.75
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
        if abs(float(n[up_axis])) < up_gate:
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


def plane_basis_from_camera(n: np.ndarray, cam_forward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # cam_forward is in SAME coordinate frame as pts (for your base_xyz: [1,0,0])
    f = cam_forward.astype(np.float32)
    f = f / (np.linalg.norm(f) + 1e-9)

    # project forward onto the plane
    v = f - np.dot(f, n) * n
    v = v / (np.linalg.norm(v) + 1e-9)

    # u completes right-handed basis on the plane
    u = np.cross(v, n)
    u = u / (np.linalg.norm(u) + 1e-9)
    return u, v




def overlay_depth_grid_xyz_base(
    depth_m: np.ndarray,
    intr,
    grid_n: int = 10,
    out_path: str = "depth_grid_xyz.png",
    max_z: float = 35.0,
) -> np.ndarray:
    """
    Overlay per-cell XYZ in base_xyz convention:
      X = forward
      Y = left
      Z = up

    depth_m: HxW float32 depth in meters
    intr: has width,height,fx,fy,cx,cy matching depth_m resolution
    """
    H, W = depth_m.shape[:2]
    assert W == intr.width and H == intr.height, "Intrinsics must match depth resolution"

    d = depth_m.astype(np.float32)
    d_vis = np.clip(d, 0.0, max_z)

    # Robust normalization (ignore NaNs)
    finite = np.isfinite(d_vis)
    if not np.any(finite):
        img = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        dmin = float(np.min(d_vis[finite]))
        dmax = float(np.max(d_vis[finite]))
        d_norm = (d_vis - dmin) / (dmax - dmin + 1e-6)
        d_norm[~finite] = 0.0
        gray = (255.0 * (1.0 - d_norm)).astype(np.uint8)  # near bright
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Grid edges
    xs = np.linspace(0, W, grid_n + 1, dtype=np.int32)
    ys = np.linspace(0, H, grid_n + 1, dtype=np.int32)

    # Draw grid lines
    for x in xs:
        cv2.line(img, (int(x), 0), (int(x), H - 1), (80, 80, 80), 1, cv2.LINE_AA)
    for y in ys:
        cv2.line(img, (0, int(y)), (W - 1, int(y)), (80, 80, 80), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1

    for gy in range(grid_n):
        for gx in range(grid_n):
            x0, x1 = xs[gx], xs[gx + 1]
            y0, y1 = ys[gy], ys[gy + 1]
            if x1 <= x0 or y1 <= y0:
                continue

            cell = d[y0:y1, x0:x1]
            cell_f = cell[np.isfinite(cell) & (cell > 1e-3)]
            if cell_f.size == 0:
                z = np.nan
            else:
                z = float(np.median(cell_f))  # depth

            u = int((x0 + x1) // 2)
            v = int((y0 + y1) // 2)

            if not np.isfinite(z):
                text1 = "nan"
                text2 = ""
                color = (0, 0, 255)
            else:
                # base_xyz: X forward, Y left, Z up
                X = z
                Y = -(u - intr.cx) * z / intr.fx
                Z = -(v - intr.cy) * z / intr.fy

                text1 = f"X {X:4.1f}"
                text2 = f"Y {Y:+.1f} Z {Z:+.1f}"
                text1 = f"Z {Z:+.1f}"
                text2 = " "
                color = (0, 255, 255)

            # Compute background size for up to 2 lines
            (tw1, th1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            if text2:
                (tw2, th2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            else:
                tw2, th2 = 0, 0
            tw = max(tw1, tw2)
            th = th1 + (th2 + 4 if text2 else 0) + 4

            tx, ty = u - 35, v
            tx = max(0, min(W - tw - 4, tx))
            ty = max(th + 1, min(H - 2, ty))

            cv2.rectangle(img, (tx, ty - th - 2), (tx + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(img, text1, (tx + 2, ty - (th2 + 4 if text2 else 0)),
                        font, font_scale, color, thickness, cv2.LINE_AA)
            if text2:
                cv2.putText(img, text2, (tx + 2, ty),
                            font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    return img


def build_occupancy_raycast(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    u: np.ndarray,
    v: np.ndarray,
    cfg: MapCfg,
    sensor_origin: np.ndarray,
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
    if n[2] < 0:
        n = -n
        d = -d
    signed = (pts @ n + d).astype(np.float32)
    height = signed.copy()
    height[height < 0.0] = 0.0

    # Project points onto plane coords
    p_proj = pts - signed[:, None] * n[None, :]
    xp = (p_proj @ u).astype(np.float32)
    yp = (p_proj @ v).astype(np.float32)



    # Sensor origin projected onto plane
    o = sensor_origin.astype(np.float32)
    p0 = o - (float(o @ n) + d) * n  # projection of origin onto plane
    print("SENSOR origin:", o, "projected p0:", p0, "p0_z:", float(p0[2]))

    sx = float(p0 @ u)
    sy = float(p0 @ v)

    front = (yp - sy) > 0.0
    xp = xp[front]
    yp = yp[front]
    height = height[front]
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

    gw = int(np.ceil(cfg.grid_width_m / cfg.resolution_m))
    gh = int(np.ceil(cfg.grid_height_m / cfg.resolution_m))

    # Center the configured map around the dynamic data/sensor area
    cx_map = 0.5 * (min_x + max_x)
    cy_map = 0.5 * (min_y + max_y)

    min_x = cx_map - 0.5 * cfg.grid_width_m
    max_x = cx_map + 0.5 * cfg.grid_width_m
    min_y = cy_map - 0.5 * cfg.grid_height_m
    max_y = cy_map + 0.5 * cfg.grid_height_m

    # Safety cap (avoid accidental huge allocations)
    max_cells = 2500  # -> 2500x2500 is already massive
    gw = int(np.clip(gw, 50, max_cells))
    gh = int(np.clip(gh, 50, max_cells))

    print(f"BOUNDS: x [{min_x:.2f}, {max_x:.2f}] span={span_x:.2f}m -> gw={gw}")
    print(f"BOUNDS: y [{min_y:.2f}, {max_y:.2f}] span={span_y:.2f}m -> gh={gh}")
    print(f"MAP meters: {gw * cfg.resolution_m:.2f} x {gh * cfg.resolution_m:.2f}")

    origin_x = min_x
    origin_y = min_y

    # Allocate
    grid = np.full((gh, gw), -1, dtype=np.int8)
    cell_max_h = np.zeros((gh, gw), dtype=np.float32)
    cell_hits = np.zeros((gh, gw), dtype=np.uint16)
    cell_sum_h = np.zeros((gh, gw), dtype=np.float32)

    # Convert to cell indices
    ix = np.floor((xp - origin_x) / cfg.resolution_m).astype(np.int32)
    iy = np.floor((yp - origin_y) / cfg.resolution_m).astype(np.int32)

    valid = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh) & np.isfinite(height)
    ixv = ix[valid]
    iyv = iy[valid]
    hv = height[valid]
    np.add.at(cell_hits, (iyv, ixv), 1)
    np.add.at(cell_sum_h, (iyv, ixv), hv)

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

    min_hits_for_occ = 3  # try 2..6 (0.1m usually 3-5)
    min_hits_for_free = 10  # optional
    meanH = np.zeros_like(cell_sum_h)
    mask = cell_hits > 0
    meanH[mask] = cell_sum_h[mask] / cell_hits[mask]
    occ_cells = observed & (cell_hits >= min_hits_for_occ) & (meanH >= cfg.occupied_height_m)
    free_cells = observed & (cell_hits >= min_hits_for_free) & (maxH <= cfg.free_height_m)

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
        print("=========== Before filter ============")
        print(f"  occupied targets: {occ_x.size} rays:{rays} free_written:{free_written} occ_written:{occ_written}")
        print("  GRID unique:", list(zip(uniq.tolist(), cnt.tolist())))
        h = cell_hits[observed]
        print(
            f"  hits(observed) p50/p90/p99: ({int(np.percentile(h, 50))}, {int(np.percentile(h, 90))}, {int(np.percentile(h, 99))})")

    occ_img = (grid == 100).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    occ_img = cv2.morphologyEx(occ_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    grid[(occ_img > 0)] = 100

    if debug:
        print("=========== After filter ============")
        uniq, cnt = np.unique(grid, return_counts=True)
        print(f"  occupied targets: {occ_x.size} rays:{rays} free_written:{free_written} occ_written:{occ_written}")
        print("  GRID unique:", list(zip(uniq.tolist(), cnt.tolist())))
        h = cell_hits[observed]
        print(
            f"  hits(observed) p50/p90/p99: ({int(np.percentile(h, 50))}, {int(np.percentile(h, 90))}, {int(np.percentile(h, 99))})")

    return grid

def occ_to_png(grid: np.ndarray) -> np.ndarray:
    img = np.full(grid.shape, 127, dtype=np.uint8)
    img[grid >= 50] = 0
    img[grid == 0] = 255
    return img


def depth_vis_u8(depth_m: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    finite = np.isfinite(depth_m)
    if not finite.any():
        raise RuntimeError("Depth is all non-finite (NaN/Inf).")
    # else:
    #     print("DEPTH:",
    #          "shape", depth_m.shape,
    #          "dtype", depth_m.dtype,
    #          "finite_pct", float(finite.mean() * 100.0),
    #          "min/max",
    #          float(np.nanmin(depth_m)),
    #          float(np.nanmax(depth_m)))

    d = depth_m.copy()
    p01 = float(np.percentile(d[finite], 1))
    p99 = float(np.percentile(d[finite], 99))

    d_norm = np.zeros_like(d, dtype=np.float32)
    den = max(p99 - p01, 1e-6)
    d_norm[finite] = (d[finite] - p01) / den
    d_norm = np.clip(d_norm, 0.0, 1.0)

    vis = np.zeros(d.shape, dtype=np.uint8)
    vis[finite] = (255.0 * (1.0 - d_norm[finite])).astype(np.uint8)

    dmin = float(np.min(d[finite]))
    dmax = float(np.max(d[finite]))

    dbg = {
        "p01": p01,
        "p99": p99,
        "min": dmin,
        "max": dmax,
        "shape": d.shape,
        "dtype": str(d.dtype),
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
    cloud_generator = PinholeCloudGenerator(
        stride=cfg.stride,
        cam_rpy_deg=(0.0, 30.0, 0.0),  # roll, pitch-down, yaw
        t_base=(0.0, 0.0, 10.0),  # camera is 10m above ground
    )

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb_inf = cv2.resize(rgb, (cfg.inference_width, cfg.inference_height), interpolation=cv2.INTER_AREA)
    intr = resize_intrinsics(intr_full, cfg.inference_width, cfg.inference_height)

    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())

    # ---- Full depth ----
    depth_full_m = depth_model.infer_depth(rgb_inf).astype(np.float32)

    # ---- Tiled depth ----
    tile_cfg = TileCfg(
        tile_w=640,
        tile_h=384,
        overlap_div=4,  # 25% overlap
        model_w=518,
        model_h=518,
        global_norm_from_full_pass=True,
    )
    print(tile_cfg.tile_w, tile_cfg.tile_h, tile_cfg.overlap_x, tile_cfg.overlap_y, tile_cfg.step_x, tile_cfg.step_y)

    depth_tiled_m, tiled_dbg = infer_depth_tiled(
        rgb_full=rgb_inf,
        intr_full=intr,  # only needed if your tiler does intrinsics per tile; ok to pass
        depth_model=depth_model,
        cfg=tile_cfg,
    )

    # Choose which depth you use downstream for occupancy:
    depth_m = depth_tiled_m  # or depth_full_m if you want baseline

    # ---- Comparison report (before masking) ----
    # optional: compare only in valid range to avoid near-max clipped zones dominating
    max_r = DepthAnythingV2Config().max_range_m
    min_r = DepthAnythingV2Config().min_range_m
    valid_for_compare = (
            np.isfinite(depth_full_m) & np.isfinite(depth_tiled_m) &
            (depth_full_m > (min_r + 0.05)) & (depth_full_m < (max_r - 0.5)) &
            (depth_tiled_m > (min_r + 0.05)) & (depth_tiled_m < (max_r - 0.5))
    )
    depth_compare_report(depth_full_m, depth_tiled_m, valid_mask=valid_for_compare, name_a="full", name_b="tiled")

    # save visuals for both depth maps
    vis_full, _ = depth_vis_u8(depth_full_m)
    vis_tiled, _ = depth_vis_u8(depth_tiled_m)
    cv2.imwrite(os.path.join(args.out_dir, "depth_full_vis.png"), vis_full)
    cv2.imwrite(os.path.join(args.out_dir, "depth_tiled_vis.png"), vis_tiled)
    save_depth_diff_visuals(args.out_dir, depth_full_m, depth_tiled_m, max_abs_m=2.0)

    max_r = DepthAnythingV2Config().max_range_m
    min_r = DepthAnythingV2Config().min_range_m

    valid_depth = (
            np.isfinite(depth_m) &
            (depth_m > (min_r + 0.05)) &
            (depth_m < (max_r - 0.5))
    )

    raw_invalid_pct = float((~valid_depth).mean()) * 100.0

    depth_m = depth_m.copy()
    depth_m[~valid_depth] = np.nan

    H = depth_m.shape[0]
    depth_m[: int(0.25 * H), :] = np.nan

    finite = np.isfinite(depth_m)
    final_invalid_pct = float((~finite).mean()) * 100.0

    print(f"DEPTH raw_invalid_or_sat_pct: {raw_invalid_pct:.2f}%")
    print(f"DEPTH final_invalid_pct: {final_invalid_pct:.2f}%")

    if finite.any():
        print(
            "DEPTH:",
            "shape", depth_m.shape,
            "dtype", depth_m.dtype,
            "finite_pct", float(finite.mean() * 100.0),
            "min/max",
            float(np.nanmin(depth_m)),
            float(np.nanmax(depth_m)),
        )
    else:
        print("DEPTH: all values are NaN/invalid")

    vis, vis_dbg = depth_vis_u8(depth_m)
    print("DEPTH dbg:", vis_dbg)
    cv2.imwrite(os.path.join(args.out_dir, "depth_vis.png"), vis)
    overlay_depth_grid_xyz_base(depth_m, intr, grid_n=15, out_path=os.path.join(args.out_dir,"depth_grid_xyz.png"), max_z=35.0)

    pts = cloud_generator.depth_to_cloud_to_base_xyz(depth_m, intr)
    R = rot_y_deg(-30.0)  # 30 deg down
    t = np.array([0.0, 0.0, 10.0], np.float32)  # camera at +10m in Z-up world

    pts_w = cloud_generator.transform_points(pts, R, t)
    cam_o_w = t.copy()  # camera origin in world
    print(f"PTS: total={pts.shape[0]} stride={cfg.stride} (inf={cfg.inference_width}x{cfg.inference_height})")

    # Assume pts are in base_xyz: x forward, y left, z up.
    # Floor should be the "lowest" points => most negative z.
    z = pts[:, 2].astype(np.float32)

    # keep finite
    m = np.isfinite(z)

    # choose bottom_frac of points by z (lowest)
    bottom_frac = float(cfg.floor_bottom_frac)
    bottom_frac = float(np.clip(bottom_frac, 0.05, 0.9))

    z_thr = np.quantile(z[m], bottom_frac)  # e.g. 0.45 => 45% most-negative (lowest) points
    cand = m & (z <= z_thr)

    pts_floor = pts[cand]
    print(
        f"FLOOR cand (z-quantile): floor_bottom_frac={bottom_frac} "
        f"z_thr={float(z_thr):.3f} pts_floor={pts_floor.shape[0]} / {pts.shape[0]}"
    )

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

    cam_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # x-forward in your base_xyz
    u, v = plane_basis_from_camera(n, cam_forward)

    fwd_p = cam_forward - (cam_forward @ n) * n
    nf = np.linalg.norm(fwd_p)
    if nf > 1e-6:
        fwd_p = fwd_p / nf
        if float(v @ fwd_p) < 0.0:
            v = -v
            u = np.cross(n, v)
            u = u / (np.linalg.norm(u) + 1e-9)

    signed_all = pts @ n + d
    print("SANITY signed_all p01/p50/p99:",
          [float(np.percentile(signed_all, 1)),
           float(np.percentile(signed_all, 50)),
           float(np.percentile(signed_all, 99))])

    cx_pix = intr.width // 2
    cy_pix = intr.height // 2
    z = float(depth_m[cy_pix, cx_pix])

    if np.isfinite(z) and z > 1e-6:
        X = z
        Y = -(cx_pix - intr.cx) * z / intr.fx
        Z = -(cy_pix - intr.cy) * z / intr.fy
        p = np.array([X, Y, Z], dtype=np.float32)

        signed_c = float(p @ n + d)
        p_proj_c = p - signed_c * n
        xp_c = float(p_proj_c @ u)
        yp_c = float(p_proj_c @ v)

        print(f"SANITY center depth z: {z} finite: True")
        print(f"SANITY center pixel -> p={p}  (u,v)=({xp_c:.3f},{yp_c:.3f})  signed={signed_c:.3f}")
        print("SANITY yp>0 means forward:", yp_c > 0.0)
    else:
        print("SANITY center depth invalid:", z)
    print("PTS base xyz min:", np.nanmin(pts, axis=0), "max:", np.nanmax(pts, axis=0))
    print("PTS world xyz min:", np.nanmin(pts_w, axis=0), "max:", np.nanmax(pts_w, axis=0))
    print("cam_o_w:", cam_o_w)
    cam_o_base = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    occ = build_occupancy_raycast(
        pts,
        n,
        d,
        u,
        v,
        cfg,
        sensor_origin=cam_o_base,
        debug=True,
    )

    png = occ_to_png(occ)
    png = np.flipud(png)
    cv2.imwrite(os.path.join(args.out_dir, "occ_map.png"), png)
    print("Saved outputs to:", args.out_dir)


if __name__ == "__main__":
    main()
