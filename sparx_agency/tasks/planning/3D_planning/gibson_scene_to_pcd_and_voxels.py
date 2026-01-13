#!/usr/bin/env python3
"""
Gibson tiny: One scene -> PointCloud (.ply) + Voxel map (Open3D).

PyCharm-friendly defaults:
- root: ./extracted/gibson_tiny
- scene: Benevolence
- mesh: mesh_z_up.obj (Z is up)
- outputs saved under: ./outputs/

Usage:
  python gibson_scene_to_pcd_and_voxels.py
  python gibson_scene_to_pcd_and_voxels.py --scene Shelbyville --voxel 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import open3d as o3d


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_root() -> Path:
    return script_dir() / "extracted" / "gibson_tiny"


def default_out_dir() -> Path:
    return script_dir() / "outputs"


def colorize_by_z(pcd: o3d.geometry.PointCloud) -> None:
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return
    z = pts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    denom = (zmax - zmin) if (zmax - zmin) > 1e-12 else 1.0
    t = (z - zmin) / denom
    colors = np.stack([t, 0.15 * np.ones_like(t), 1.0 - t], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)


def voxel_centers_as_pointcloud(vg: o3d.geometry.VoxelGrid) -> o3d.geometry.PointCloud:
    centers = []
    for v in vg.get_voxels():
        centers.append(vg.get_voxel_center_coordinate(v.grid_index))
    out = o3d.geometry.PointCloud()
    if centers:
        out.points = o3d.utility.Vector3dVector(np.asarray(centers, dtype=np.float64))
        colorize_by_z(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(default_root()), help="Root folder of extracted gibson_tiny")
    ap.add_argument("--scene", default="Benevolence", help="Scene name under root (e.g. Benevolence)")
    ap.add_argument("--points", type=int, default=1_500_000, help="Points to sample from the mesh")
    ap.add_argument("--voxel", type=float, default=0.10, help="Voxel size in meters (default: 0.10)")
    ap.add_argument("--out-dir", default=str(default_out_dir()), help="Output folder for generated files")
    ap.add_argument("--export-voxel-centers", action="store_true", help="Export voxel centers as .ply")
    ap.add_argument("--show-pointcloud", action="store_true", help="Show sampled point cloud too (can be heavy)")
    args = ap.parse_args()

    try:
        root = Path(args.root).expanduser().resolve()
        scene_dir = (root / args.scene).resolve()
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

        # Prefer z-up mesh for correct orientation
        mesh_path = scene_dir / "mesh_z_up.obj"
        if not mesh_path.exists():
            mesh_path = scene_dir / "mesh.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"No mesh.obj or mesh_z_up.obj found in: {scene_dir}")

        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Root:  {root}")
        print(f"[INFO] Scene: {args.scene}")
        print(f"[INFO] Mesh:  {mesh_path}")

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh or empty mesh: {mesh_path}")
        mesh.compute_vertex_normals()

        # Sample a point cloud from the mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=int(args.points))
        if len(pcd.points) == 0:
            raise ValueError("Sampled point cloud is empty")

        colorize_by_z(pcd)

        out_pcd = out_dir / f"{args.scene}_pointcloud.ply"
        ok = o3d.io.write_point_cloud(str(out_pcd), pcd)
        if not ok:
            raise RuntimeError(f"Failed to write point cloud: {out_pcd}")

        print(f"[OK] Point cloud: {out_pcd}")
        print(f"[INFO] Points: {len(pcd.points)}")

        # Build voxel map
        if args.voxel <= 0:
            raise ValueError("--voxel must be > 0")
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(args.voxel))
        print(f"[OK] Voxels: {len(vg.get_voxels())}  voxel_size={args.voxel}")

        geoms = [vg]

        if args.export_voxel_centers:
            centers = voxel_centers_as_pointcloud(vg)
            out_centers = out_dir / f"{args.scene}_voxel_centers.ply"
            ok = o3d.io.write_point_cloud(str(out_centers), centers)
            if not ok:
                raise RuntimeError(f"Failed to write voxel centers: {out_centers}")
            print(f"[OK] Voxel centers: {out_centers}  points={len(centers.points)}")
            geoms.append(centers)

        if args.show_pointcloud:
            geoms.append(pcd)

        # Visualize (voxels are usually the clearest)
        o3d.visualization.draw_geometries(geoms)

        return 0

    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
