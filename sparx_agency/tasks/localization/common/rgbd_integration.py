# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
import argparse
import copy
from pathlib import Path

import open3d as o3d
import numpy as np

import os, sys

from open3d.cuda.pybind.camera import PinholeCameraIntrinsic
from open3d.cuda.pybind.geometry import RGBDImage

from sparx_agency.tasks.localization.common.camera_model import make_intrinsic_from_image
from sparx_agency.tasks.localization.common.monocular_odometry import compute_pairwise_odometry
from sparx_agency.tasks.localization.common.rgbd_utils import read_frames

# pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(pyexample_path)
#
# from open3d_example import read_trajectory


def compute_trajectory(frames: list[tuple[Path, RGBDImage]], intrinsic: PinholeCameraIntrinsic):
    waypoints = [o3d.geometry.TriangleMesh.create_coordinate_frame()]

    for i in range(len(frames) - 1):
        src_path, src_rgbd = frames[i]
        dst_path, dst_rgbd = frames[i + 1]

        success, T = compute_pairwise_odometry(src_rgbd, dst_rgbd, intrinsic)

        if success:
            print("[odometry] success")
            print(T)
            wp = copy.deepcopy(waypoints[-1])
            wp = wp.transform(T)
            waypoints.append(wp)
        else:
            print("[odometry] failed")
    return waypoints


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

    trajectory = compute_trajectory(frames, intrinsic)

    o3d.visualization.draw_geometries(trajectory)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    work_dir = Path(__file__).parent.resolve()
    parser.add_argument("--in-dir", default=Path.home() / "Pictures/rgbd_exports")
    parser.add_argument("--hfov-deg", type=float, default=130.0)
    args = parser.parse_args()
    main(args)
    # rgbd_data = o3d.data.SampleRedwoodRGBDImages()
    # camera_poses = read_trajectory(rgbd_data.odometry_log_path)
    # camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    # volume = o3d.pipelines.integration.UniformTSDFVolume(
    #     length=4.0,
    #     resolution=512,
    #     sdf_trunc=0.04,
    #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    # )
    #
    # for i in range(len(camera_poses)):
    #     print("Integrate {:d}-th image into the volume.".format(i))
    #     color = o3d.io.read_image(rgbd_data.color_paths[i])
    #     depth = o3d.io.read_image(rgbd_data.depth_paths[i])
    #
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    #     volume.integrate(
    #         rgbd,
    #         camera_intrinsics,
    #         np.linalg.inv(camera_poses[i].pose),
    #     )
    #
    # print("Extract triangle mesh")
    # mesh = volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    #
    # print("Extract voxel-aligned debugging point cloud")
    # voxel_pcd = volume.extract_voxel_point_cloud()
    # o3d.visualization.draw_geometries([voxel_pcd])
    #
    # print("Extract voxel-aligned debugging voxel grid")
    # voxel_grid = volume.extract_voxel_grid()
    # # o3d.visualization.draw_geometries([voxel_grid])
    #
    # # print("Extract point cloud")
    # # pcd = volume.extract_point_cloud()
    # # o3d.visualization.draw_geometries([pcd])