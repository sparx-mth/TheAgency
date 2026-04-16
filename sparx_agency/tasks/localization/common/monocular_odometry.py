
"""Find camera movement between consecutive RGBD frames stored as .npz files."""
import argparse
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


def make_intrinsic_from_image(
    width: int,
    height: int,
    hfov_deg: float = 90.0,
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Create a PinholeCameraIntrinsic object based on image properties.

    This function computes the intrinsic camera parameters for a pinhole camera
    model using the specified image dimensions and horizontal field of view (HFOV).
    The resulting intrinsic parameters are then used to create and return an
    o3d.camera.PinholeCameraIntrinsic object. The focal lengths are derived from
    the HFOV, and the principal point is assumed to be at the image center.

    Parameters:
    width : int
        The width of the image in pixels.
    height : int
        The height of the image in pixels.
    hfov_deg : float, optional
        The horizontal field of view in degrees. Defaults to 90.0.

    Returns:
    o3d.camera.PinholeCameraIntrinsic
        A PinholeCameraIntrinsic object with computed parameters for the given
        image dimensions and field of view.
    """
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic


def compute_pairwise_odometry(
    source_rgbd: o3d.geometry.RGBDImage,
    target_rgbd: o3d.geometry.RGBDImage,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
) -> Tuple[bool, np.ndarray]:
    """
    Compute the relative pose transformation and odometry success status between two RGB-D frames.

    This function calculates the pairwise odometry transformation between a source RGB-D
    image and a target RGB-D image using the RGB-D odometry pipeline provided by Open3D.
    The computation employs a hybrid Jacobian term for optimization and uses the default
    odometry options.

    Parameters:
    source_rgbd (o3d.geometry.RGBDImage): The source RGB-D image used in the odometry computation.
    target_rgbd (o3d.geometry.RGBDImage): The target RGB-D image used in the odometry computation.
    intrinsic (o3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters used for transformation
        calculations.

    Returns:
    Tuple[bool, np.ndarray]: A tuple where the first element is a boolean indicating the success
        of the odometry computation, and the second element is a 4x4 numpy array representing
        the computed transformation matrix.
    """
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)

    success, transformation, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd,
        target_rgbd,
        intrinsic,
        odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

    return success, transformation


def main(args):
    in_dir = Path(args.in_dir).expanduser().resolve()
    frames = read_frames(in_dir)

    first_color = np.asarray(frames[0][1].color)
    height, width = first_color.shape[:2]

    intrinsic = make_intrinsic_from_image(
        width=width,
        height=height,
        hfov_deg=args.hfov_deg,
    )

    print(f"[intrinsic] width={width}, height={height}, hfov_deg={args.hfov_deg}")
    print(intrinsic)

    transforms = []

    for i in range(len(frames) - 1):
        src_path, src_rgbd = frames[i]
        dst_path, dst_rgbd = frames[i + 1]

        success, T = compute_pairwise_odometry(src_rgbd, dst_rgbd, intrinsic)

        print(f"\n[pair {i:04d}] {src_path.name} -> {dst_path.name}")
        if success:
            print("[odometry] success")
            print(T)
            transforms.append((src_path.name, dst_path.name, T))
        else:
            print("[odometry] failed")

    if args.out_file:
        out_file = Path(args.out_file).expanduser().resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            for src_name, dst_name, T in transforms:
                f.write(f"{src_name} -> {dst_name}\n")
                np.savetxt(f, T, fmt="%.8f")
                f.write("\n")

        print(f"\n[saved] odometry results written to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    work_dir = Path(__file__).parent.resolve()
    parser.add_argument("--in-dir", default=Path.home() / "Pictures/rgbd_exports")
    parser.add_argument("--out-file", default=(work_dir / "outputs" / "odometry.txt").as_posix())
    parser.add_argument("--hfov-deg", type=float, default=130.0)
    args = parser.parse_args()
    main(args)