from pathlib import Path
from typing import List, Tuple
import open3d as o3d
from open3d.cuda.pybind.camera import PinholeCameraIntrinsic
from scipy.spatial.transform import RigidTransform as RT

import numpy as np


def print_transformation(t: np.ndarray):
    rt = RT.from_matrix(t)
    yaw, pitch, roll = rt.rotation.as_euler('xyz', degrees=True)
    x, y, z = rt.translation
    print(f"x={x}, y={y}, z={z}")
    print(f"yaw={yaw}, pitch={pitch}, roll={roll}")


def visualize_rgbd_list(rgbd_list: List[o3d.cuda.pybind.geometry.RGBDImage], intrinsic: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), transforms: List[np.ndarray] = None):
    assert len(rgbd_list) == len(transforms), "The number of RGBD images and transformations must be the same."
    # o3d.visualization.draw_geometries(rgbd_list)
    for t in transforms:
        t = np.linalg.inv(t)
        t[1,1] = -t[1,1]
        t[2,2] = -t[2,2]
    pcd_list = [o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic=np.array(t))
                for rgbd, t in zip(rgbd_list, transforms)]

    o3d.visualization.draw_geometries(pcd_list)

def visualize_rgbd(rgbd_image: o3d.cuda.pybind.geometry.RGBDImage,
                   intrinsic: o3d.camera.PinholeCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), transformation: np.ndarray = np.identity(4)):
    print(rgbd_image)

    # o3d.visualization.draw_geometries([rgbd_image])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, transformation)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

def create_pointcloud(rgbd: o3d.cuda.pybind.geometry.RGBDImage, intrinsic: PinholeCameraIntrinsic):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return pcd

def make_rgbd_image(color: np.ndarray, depth: np.ndarray) -> o3d.cuda.pybind.geometry.RGBDImage:
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

    color_img = o3d.cuda.pybind.geometry.Image(color)
    depth_img = o3d.cuda.pybind.geometry.Image(depth)

    return o3d.cuda.pybind.geometry.RGBDImage.create_from_color_and_depth(
        color_img,
        depth_img,
        depth_scale=1,              # depth is already in meters
        depth_trunc=15.0,             # adjust if needed
        convert_rgb_to_intensity=False,
    )

def print_depth_stats(depth: List[np.ndarray]):
    stats = [np.percentile(d, [10 * i for i in range(9)]) for d in depth]
    print(f"Depth stats: {stats}")

def read_frames(path: Path, max_imgs: int=int(1e6), start_from: int=0) -> List[Tuple[Path, o3d.geometry.RGBDImage]]:
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
    max_imgs (int): The maximum number of .npz files to read.
    start_from (int): The index of the first .npz file to read.
    Raises:
    FileNotFoundError: If no .npz files are found in the provided path.
    RuntimeError: If less than two valid RGBD images are created from the .npz files.
    """
    items: List[Tuple[Path, o3d.geometry.RGBDImage]] = []

    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {path}")
    end_at = min(len(files), max_imgs+start_from)

    files = files[start_from:end_at]
    for p in files:
        with np.load(p) as data:
            if "rgb" not in data or "depth" not in data:
                print(f"[skip] {p.name}: missing 'rgb' or 'depth'")
                continue
            rgb = data["rgb"]
            depth = data["depth"]
            rgb = rgb[:, 100:540]
            depth = depth[:, 100:540]
            # rgb = rgb[20:100, 100:540]
            # depth = depth[20:100, 100:540]
            rgbd = make_rgbd_image(color=rgb, depth=depth)

            print(
                f"[load] {p.name} | "
                f"color={np.asarray(rgbd.color).shape} depth={np.asarray(rgbd.depth).shape}"
            )
            items.append((p, rgbd))

    if len(items) < 2:
        raise RuntimeError("Need at least two valid RGBD frames to compute odometry.")

    return items
