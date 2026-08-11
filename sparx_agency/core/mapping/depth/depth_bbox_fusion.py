from __future__ import annotations  # PEP 563: stringize the tuple[...] annotations for Python 3.8

import numpy as np
from typing import Optional

from sparx_agency.core.common.math.bbox import rescale_xyxy


def valid_depth_mask(depth: np.ndarray, min_depth: float = 0.01,
                     max_depth: float = float("inf")) -> np.ndarray:
    """Boolean mask of finite depths within ``[min_depth, max_depth]``.

    Inlined (rather than imported from ``robots.common.helpers``) to keep this
    core geometry module ROS-free and torch-free, so it can be used by the
    Python-3.8 Noetic adapters without dragging ``sensor_msgs``.
    """
    return np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)

def bbox_to_xyz_cam_from_depth(
    depth_m: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    fx: float, fy: float, cx: float, cy: float,
    sample_grid: int = 9,
    low_quantile: float = 0.2,
    min_depth: float = 0.15,
    max_depth: float = 20.0,
) -> Optional[tuple[float, float, float]]:
    """
    Returns (X,Y,Z) in camera frame (camera_link optical-like convention
    assuming depth is Z-forward).
    Robust Z: median of lowest quantile depths inside bbox.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim != 2:
        return None

    H, W = d.shape
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(x1, 0, W - 1)); x2 = int(np.clip(x2, 0, W - 1))
    y1 = int(np.clip(y1, 0, H - 1)); y2 = int(np.clip(y2, 0, H - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    xs = np.linspace(x1 + 1, x2 - 1, sample_grid).astype(np.int32)
    ys = np.linspace(y1 + 1, y2 - 1, sample_grid).astype(np.int32)
    xv, yv = np.meshgrid(xs, ys)
    z = d[yv, xv].reshape(-1)

    m = valid_depth_mask(z, min_depth=min_depth, max_depth=max_depth)
    z = z[m]
    if z.size < 8:
        return None

    z_sorted = np.sort(z)
    k = max(1, int(np.ceil(low_quantile * z_sorted.size)))
    Z = float(np.median(z_sorted[:k]))

    u = int(np.clip(0.5 * (x1 + x2), 0, W - 1))
    v = int(np.clip(0.5 * (y1 + y2), 0, H - 1))

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return (float(X), float(Y), float(Z))


def rescale_bbox_to_depth(
    bbox_xyxy: tuple[float, float, float, float],
    fx: float, fy: float, cx: float, cy: float,
    intr_w: int, intr_h: int, depth_w: int, depth_h: int,
) -> tuple[tuple[int, int, int, int], float, float, float, float]:
    """Rescale a bbox + intrinsics from RGB/tracker space to depth-array space.

    Use when a depth source's native resolution differs from the bbox's own
    intrinsics (e.g. a fixed-resolution TensorRT engine) instead of requiring
    an exact shape match before sampling.
    """
    if (depth_h, depth_w) == (intr_h, intr_w):
        return tuple(int(v) for v in bbox_xyxy), fx, fy, cx, cy
    x1, y1, x2, y2 = rescale_xyxy(bbox_xyxy, intr_w, intr_h, depth_w, depth_h)
    sx, sy = depth_w / intr_w, depth_h / intr_h
    return (int(x1), int(y1), int(x2), int(y2)), fx * sx, fy * sy, cx * sx, cy * sy


def transform_point(T_world_cam: np.ndarray, xyz_cam: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    T_world_cam: 4x4 homogeneous transform.
    """
    p = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0], dtype=np.float64)
    pw = T_world_cam @ p
    return (float(pw[0]), float(pw[1]), float(pw[2]))


def xyz_from_pointcloud_bbox(
    cloud_xyz: np.ndarray,   # shape (H, W, 3) or (N, 3) projected
    bbox,
    min_z: float = 0.2,
    max_z: float = 10.0,
):
    """
    cloud_xyz assumed aligned with image (organized pointcloud).
    bbox = (x1,y1,x2,y2) in image coords.
    """
    x1, y1, x2, y2 = bbox
    patch = cloud_xyz[y1:y2, x1:x2, :]   # (h,w,3)

    Z = patch[..., 2]
    mask = np.isfinite(Z) & (Z > min_z) & (Z < max_z)
    if mask.sum() < 20:
        return None

    pts = patch[mask]
    return np.median(pts, axis=0)  # robust center


def pointcloud2_to_xyz_image(msg) -> np.ndarray:
    """
    Convert organized PointCloud2 to (H, W, 3) float32 array.
    Requires fields: x,y,z as float32.
    """
    H = int(msg.height)
    W = int(msg.width)
    if H <= 1 or W <= 1:
        raise ValueError(f"PointCloud2 not organized: height={H} width={W}")

    # Validate fields exist
    offsets = {f.name: f.offset for f in msg.fields}
    if not all(k in offsets for k in ("x", "y", "z")):
        raise ValueError(f"PointCloud2 missing x/y/z fields: {list(offsets.keys())}")

    point_step = int(msg.point_step)
    row_step = int(msg.row_step)

    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(H, row_step)

    def read_field(off: int) -> np.ndarray:
        # Read float32 field at byte offset 'off' for each point in each row
        # Slice rows, then view as float32
        # We take bytes for each point: start at off, step by point_step, for W points
        bytes_ = np.zeros((H, W, 4), dtype=np.uint8)
        for i in range(4):
            bytes_[:, :, i] = data[:, off + i : off + i + W * point_step : point_step]
        return bytes_.view(np.float32).reshape(H, W)

    X = read_field(offsets["x"])
    Y = read_field(offsets["y"])
    Z = read_field(offsets["z"])

    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return xyz

def xyz_from_unorganized_cloud_bbox(
    points_xyz: np.ndarray,  # (N,3)
    fx, fy, cx, cy,
    bbox,
    min_z=0.2,
    max_z=10.0,
):
    """
    Select points whose projection falls inside bbox.
    """
    x1, y1, x2, y2 = bbox

    X = points_xyz[:, 0]
    Y = points_xyz[:, 1]
    Z = points_xyz[:, 2]

    valid = np.isfinite(Z) & (Z > min_z) & (Z < max_z)
    if valid.sum() < 50:
        return None

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    # project to image
    u = (fx * X / Z + cx).astype(np.int32)
    v = (fy * Y / Z + cy).astype(np.int32)

    inside = (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
    if inside.sum() < 20:
        return None

    pts = np.stack([X[inside], Y[inside], Z[inside]], axis=1)
    return np.median(pts, axis=0)


def pointcloud2_to_xyz_array(msg):
    import struct
    pts = []
    for i in range(0, len(msg.data), msg.point_step):
        x = struct.unpack_from("f", msg.data, i + msg.fields[0].offset)[0]
        y = struct.unpack_from("f", msg.data, i + msg.fields[1].offset)[0]
        z = struct.unpack_from("f", msg.data, i + msg.fields[2].offset)[0]
        pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)






