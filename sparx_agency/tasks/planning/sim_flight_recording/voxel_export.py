"""Write a voxel map as a PLY point cloud, so a human can open it and look.

Run it directly to turn a committed map into something viewable::

    .venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/voxel_export.py \
        --scene office --preview

Needs neither Isaac Sim nor Open3D: the ``.npz`` map holds everything, and the
``--preview`` render is numpy and cv2.


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


def clip_height(points: np.ndarray, min_z: float = None, max_z: float = None) -> np.ndarray:
    """Drop points outside a height band.

    Indispensable for looking at an indoor map: the ceiling is the largest
    surface in the building and it is between you and everything you wanted to
    see. Clipping just below it turns a picture of a roof into a floor plan
    with furniture standing on it.

    Args:
        points: ``(N, 3)`` world-frame points, metres.
        min_z: Drop anything below this. None keeps the floor.
        max_z: Drop anything above this. None keeps the ceiling.

    Returns:
        The surviving points.
    """
    keep = np.ones(len(points), dtype=bool)
    if min_z is not None:
        keep &= points[:, 2] >= min_z
    if max_z is not None:
        keep &= points[:, 2] <= max_z
    return points[keep]


def export_voxel_grid(grid, path, min_z: float = None, max_z: float = None) -> Path:
    """Write a :class:`VoxelGrid3D`'s occupied voxels as a viewable point cloud.

    Args:
        grid: The surveyed voxel grid.
        path: Destination ``.ply``.
        min_z: Optional floor of the exported height band, metres.
        max_z: Optional ceiling, metres. Clipping just under the real one is
            what makes an indoor map legible -- see :func:`clip_height`.

    Returns:
        The path written.
    """
    return write_ply(path, clip_height(grid.occupied_points(), min_z, max_z))


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


def render_isometric(grid, width: int = 1400, azimuth_deg: float = 35.0,
                     elevation_deg: float = 28.0, min_z: float = None,
                     max_z: float = None) -> "np.ndarray":
    """Draw the voxel map as a shaded isometric view, using only numpy and cv2.

    A stand-in for the interactive viewer, so the map can be checked on the
    machine that produced it -- where Open3D is not installed and a 450 MB wheel
    to look at one file is a poor trade. Rotates the occupied voxel centres,
    z-buffers them into an image, and shades by height.

    Args:
        grid: The surveyed :class:`VoxelGrid3D`.
        width: Output image width, pixels.
        azimuth_deg: Rotation about the world z axis.
        elevation_deg: Tilt above the horizon. 90 would be a plan view.
        min_z: Optional floor of the rendered height band, metres.
        max_z: Optional ceiling, metres. Without one you are looking at a roof.

    Returns:
        A BGR image.
    """
    import cv2
    import numpy as np

    points = clip_height(grid.occupied_points(), min_z, max_z)
    if len(points) == 0:
        return np.zeros((width // 2, width, 3), np.uint8)

    heights = points[:, 2].copy()
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    centred = points - points.mean(axis=0)
    # Yaw, then tilt: screen x is the rotated world x, screen y mixes world y
    # and z so the map is seen from above and to the side.
    rotated_x = centred[:, 0] * np.cos(azimuth) - centred[:, 1] * np.sin(azimuth)
    rotated_y = centred[:, 0] * np.sin(azimuth) + centred[:, 1] * np.cos(azimuth)
    screen_x = rotated_x
    screen_y = rotated_y * np.sin(elevation) - centred[:, 2] * np.cos(elevation)
    depth = rotated_y * np.cos(elevation) + centred[:, 2] * np.sin(elevation)

    margin = 20
    scale = (width - 2 * margin) / max(np.ptp(screen_x), 1e-6)
    height = int(np.ptp(screen_y) * scale) + 2 * margin
    column = np.clip(((screen_x - screen_x.min()) * scale + margin).astype(int),
                     0, width - 1)
    row = np.clip(((screen_y - screen_y.min()) * scale + margin).astype(int),
                  0, height - 1)

    # Painter's algorithm: sort back to front so nearer voxels overwrite.
    order = np.argsort(depth)
    canvas = np.full((height, width, 3), 245, np.uint8)
    colours = height_colours(np.stack([points[:, 0], points[:, 1], heights], axis=1))
    canvas[row[order], column[order]] = colours[order][:, ::-1]   # RGB -> BGR
    return np.flipud(canvas)


def main() -> int:
    """Turn a surveyed ``.npz`` map into a PLY, and optionally a preview image."""
    import argparse
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from sparx_agency.core.planning.environment import load_voxel_grid
    from sparx_agency.robots.PEGASUS.adapters.scene_map import voxel_map_path

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--scene", default="office", help="a surveyed scene name")
    ap.add_argument("--map", type=Path, default=None,
                    help="a voxel .npz to read instead of the scene's own")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the .ply (default: beside the map)")
    ap.add_argument("--preview", action="store_true",
                    help="also render a shaded isometric PNG, which needs no Open3D")
    ap.add_argument("--max-z", type=float, default=None,
                    help="drop voxels above this height, metres. Clipping just under "
                         "the ceiling is what makes an indoor map legible -- without "
                         "it you are looking at a roof")
    ap.add_argument("--min-z", type=float, default=None,
                    help="drop voxels below this height, metres, e.g. to lose the floor")
    args = ap.parse_args()

    path = args.map or voxel_map_path(args.scene)
    grid, metadata = load_voxel_grid(path)
    print(f"{path.name}: {grid}", flush=True)
    print(f"   {metadata.get('occupied', 0)} occupied voxels, surveyed at "
          f"{metadata.get('resolution_m')} m", flush=True)

    ply = args.out or path.with_suffix(".ply")
    export_voxel_grid(grid, ply, min_z=args.min_z, max_z=args.max_z)
    print(f"wrote {ply} ({ply.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.preview:
        import cv2

        image = render_isometric(grid, min_z=args.min_z, max_z=args.max_z)
        preview = ply.with_name(ply.stem + "_iso.png")
        cv2.imwrite(str(preview), image)
        print(f"wrote {preview} ({image.shape[1]}x{image.shape[0]})", flush=True)

    print()
    print(viewer_snippet(ply, grid.resolution), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
