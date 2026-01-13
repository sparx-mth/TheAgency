#!/usr/bin/env python3
"""
View a Gibson scene as a SIMPLE point cloud.

Goal:
- Understand what a point cloud looks like
- See rooms, walls, corridors clearly
- No voxels, no tricks

Controls in window:
- Left mouse: rotate
- Right mouse: pan
- Scroll: zoom
"""

from pathlib import Path
import numpy as np
import open3d as o3d


# === CONFIG ===
SCENE = "Benevolence"
ROOT = Path("gibson/extracted/gibson_tiny")
POINTS = 6_000_000   #
# =====================================


def colorize_by_height(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    zmin, zmax = z.min(), z.max()
    t = (z - zmin) / max(zmax - zmin, 1e-6)
    colors = np.stack([t, 0.3*np.ones_like(t), 1.0 - t], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def main():
    scene_dir = ROOT / SCENE
    mesh_path = scene_dir / "mesh_z_up.obj"
    if not mesh_path.exists():
        mesh_path = scene_dir / "mesh.obj"

    print("[INFO] Loading mesh:", mesh_path)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    print("[INFO] Sampling point cloud...")
    pcd = mesh.sample_points_uniformly(number_of_points=POINTS)

    colorize_by_height(pcd)

    print("[INFO] Points:", len(pcd.points))
    print("[INFO] Rotate / zoom to explore the house")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Gibson Point Cloud",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()
