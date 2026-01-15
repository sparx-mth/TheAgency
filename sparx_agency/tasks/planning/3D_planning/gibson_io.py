# =========================
# File: interactive_rrtstar/gibson_io.py
# =========================
from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d

from logging_utils import pinfo, pok


def load_gibson_mesh(root: Path, scene: str) -> o3d.geometry.TriangleMesh:
    scene_dir = root / scene
    mesh_path = scene_dir / "mesh_z_up.obj"
    if not mesh_path.exists():
        mesh_path = scene_dir / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"No mesh.obj or mesh_z_up.obj in {scene_dir}")

    pinfo(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed loading mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    pok(f"Mesh loaded: vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
    return mesh


def sample_point_cloud(mesh: o3d.geometry.TriangleMesh, n: int) -> o3d.geometry.PointCloud:
    pinfo(f"Sampling point cloud: n={n}")
    pcd = mesh.sample_points_uniformly(number_of_points=int(n))
    if len(pcd.points) == 0:
        raise ValueError("Sampled point cloud is empty")

    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    colors = np.stack([t, 0.3 * np.ones_like(t), 1.0 - t], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pok(f"Point cloud ready: points={len(pcd.points)} z_range=[{z.min():.2f}, {z.max():.2f}]")
    return pcd
