"""Write a voxel map as a PLY point cloud, so a human can open it and look.

The map itself lives in a ``.npz`` -- compact, exact, and what the planners
read. This is the other half: a file you can drag into Open3D, MeshLab or
CloudCompare and *see*, which is the only way to catch a survey that came out
subtly wrong.

Open3D is deliberately **not** a dependency. It is not installed on the machine
that produces these files, and requiring it there to write a format defined by a
70-byte header would be silly. The PLY written here is the minimal binary
little-endian coloured point cloud that ``o3d.io.read_point_cloud`` accepts:
``float x/y/z`` + ``uchar red/green/blue``, 15 bytes per point, no padding.

One point per occupied voxel centre, coloured by height. Together with the voxel
size (in the file's name and in the ``.npz`` beside it) that is a lossless
description of the occupied set -- Open3D's
``VoxelGrid.create_from_point_cloud(pcd, voxel_size)`` reconstructs the cubes
exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_HEADER = """ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

# Blue at the floor through to warm yellow at the ceiling: height is the one
# thing a plan-view map cannot show, so it is what the colour is spent on.
_LOW_COLOUR = np.array([60, 90, 200], dtype=np.float32)
_HIGH_COLOUR = np.array([250, 220, 120], dtype=np.float32)


def height_colours(points: np.ndarray) -> np.ndarray:
    """Colour an ``(N, 3)`` point cloud by z, low to high.

    Args:
        points: World-frame points, metres.

    Returns:
        ``(N, 3)`` uint8 RGB.
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z = points[:, 2].astype(np.float32)
    span = float(z.max() - z.min())
    t = ((z - z.min()) / span if span > 1e-6 else np.zeros_like(z))[:, None]
    return (_LOW_COLOUR * (1.0 - t) + _HIGH_COLOUR * t).astype(np.uint8)


def write_ply(path, points: np.ndarray, colours: np.ndarray = None) -> Path:
    """Write a binary PLY point cloud.

    Vectorised through a structured array rather than a per-point pack: at the
    million-point scale a Python loop costs the better part of a second and this
    costs a few tens of milliseconds.

    Args:
        path: Destination ``.ply``. Parent directories are created.
        points: ``(N, 3)`` world-frame points, metres.
        colours: ``(N, 3)`` uint8 RGB. Defaults to :func:`height_colours`.

    Returns:
        The path written.

    Raises:
        ValueError: If ``colours`` does not match ``points``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colours = height_colours(points) if colours is None else np.asarray(colours, np.uint8)
    if colours.shape != points.shape:
        raise ValueError(
            f"colours {colours.shape} does not match points {points.shape}")

    record = np.zeros(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    record["x"], record["y"], record["z"] = points.T
    record["red"], record["green"], record["blue"] = colours.T

    with path.open("wb") as handle:
        handle.write(PLY_HEADER.format(count=len(points)).encode("ascii"))
        record.tofile(handle)
    return path


def export_voxel_grid(grid, path) -> Path:
    """Write a :class:`VoxelGrid3D`'s occupied voxels as a viewable point cloud.

    Args:
        grid: The surveyed voxel grid.
        path: Destination ``.ply``.

    Returns:
        The path written.
    """
    return write_ply(path, grid.occupied_points())


def viewer_snippet(ply_path, voxel_size: float) -> str:
    """The Open3D incantation to view the exported file, as text.

    Printed next to the file rather than shipped as a script, because Open3D
    lives on whatever machine the user is looking at this from and not on the
    one that produced it.
    """
    return (
        f"# On a machine with open3d installed (pip install open3d):\n"
        f"import open3d as o3d\n"
        f"pcd = o3d.io.read_point_cloud({str(ply_path)!r})\n"
        f"voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, {voxel_size})\n"
        f"o3d.visualization.draw_geometries([voxels])\n"
        f"# ...or just the points, which is faster to spin around:\n"
        f"# o3d.visualization.draw_geometries([pcd])"
    )
