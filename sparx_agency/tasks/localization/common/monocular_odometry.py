"""Find camera movement between consecutive RGBD frames stored as .npz files."""
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import open3d as o3d

from sparx_agency.tasks.localization.common.camera_model import make_intrinsic_from_image
from sparx_agency.tasks.localization.common.rgbd_utils import read_frames


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
        vfov_deg=args.vfov_deg
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
    parser.add_argument("--vfov-deg", type=float, default=90.0)
    args = parser.parse_args()
    main(args)