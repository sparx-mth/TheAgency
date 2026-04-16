from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


def make_rgbd_image(color: np.ndarray, depth: np.ndarray) -> o3d.geometry.RGBDImage:
    """
    Convert NumPy RGB/depth arrays into an Open3D RGBDImage.

    Assumptions:
    - color: HxWx3 uint8 RGB
    - depth: HxW float32 depth in meters
    """
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {color.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected depth image with shape (H, W), got {depth.shape}")

    color = np.ascontiguousarray(color.astype(np.uint8))
    depth = np.ascontiguousarray(depth.astype(np.float32))

    color_img = o3d.geometry.Image(color)
    depth_img = o3d.geometry.Image(depth)

    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img,
        depth_img,
        depth_scale=1.0,              # depth is already in meters
        depth_trunc=10.0,             # adjust if needed
        convert_rgb_to_intensity=False,
    )


def read_frames(path: Path) -> List[Tuple[Path, o3d.geometry.RGBDImage]]:
    """
    Reads and processes RGBD frames from .npz files located in the specified directory.

    The function loads and parses .npz files containing RGB and depth images, converts them
    to RGBDImage objects, and returns a list of tuples containing the file path and the
    corresponding RGBDImage. If there are no `.npz` files available or less than two valid
    RGBD images are generated, it raises appropriate exceptions.

    Parameters:
    path (Path): The path to the directory containing the .npz files.

    Returns:
    List[Tuple[Path, o3d.geometry.RGBDImage]]: A list of tuples where each tuple consists of
    the file path and the corresponding RGBD image created from the file data.

    Raises:
    FileNotFoundError: If no .npz files are found in the provided path.
    RuntimeError: If less than two valid RGBD images are created from the .npz files.
    """
    items: List[Tuple[Path, o3d.geometry.RGBDImage]] = []

    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {path}")

    for p in files:
        with np.load(p) as data:
            if "rgb" not in data or "depth" not in data:
                print(f"[skip] {p.name}: missing 'rgb' or 'depth'")
                continue

            rgb = data["rgb"]
            depth = data["depth"]
            rgbd = make_rgbd_image(color=rgb, depth=depth)

            print(
                f"[load] {p.name} | "
                f"color={np.asarray(rgbd.color).shape} depth={np.asarray(rgbd.depth).shape}"
            )
            items.append((p, rgbd))

    if len(items) < 2:
        raise RuntimeError("Need at least two valid RGBD frames to compute odometry.")

    return items
