# =========================
# File: interactive_rrtstar/tube.py
# =========================
from __future__ import annotations

from typing import Tuple
import numpy as np
import open3d as o3d


def make_tube_from_polyline(
    pts: np.ndarray,
    radius: float,
    rgb: Tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    """Create a thick "tube" mesh along a polyline by stitching cylinders segment-by-segment."""
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points for a tube")

    tube = o3d.geometry.TriangleMesh()
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i in range(len(pts) - 1):
        p0 = pts[i].astype(np.float64)
        p1 = pts[i + 1].astype(np.float64)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            continue

        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=float(radius),
            height=L,
            resolution=24,
            split=4,
        )
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(list(rgb))

        dir_vec = v / L
        axis = np.cross(z_axis, dir_vec)
        axis_norm = float(np.linalg.norm(axis))

        if axis_norm < 1e-9:
            if dir_vec[2] < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
                cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))
        else:
            axis = axis / axis_norm
            angle = float(np.arccos(np.clip(np.dot(z_axis, dir_vec), -1.0, 1.0)))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))

        mid = (p0 + p1) * 0.5
        cyl.translate(mid)
        tube += cyl

    tube.merge_close_vertices(1e-6)
    tube.compute_vertex_normals()
    return tube
