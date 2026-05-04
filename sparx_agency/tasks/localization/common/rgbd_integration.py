import argparse
import copy
from pathlib import Path
from typing import List
import open3d as o3d
import numpy as np
from open3d.cuda.pybind.camera import PinholeCameraIntrinsic
from sparx_agency.tasks.localization.common.camera_model import make_intrinsic_from_image
from sparx_agency.tasks.localization.common.rgbd_utils import read_frames, visualize_rgbd, visualize_rgbd_list
from scipy.spatial.transform import Rotation as R


def compute_trajectory(frames: list[tuple[Path, o3d.cuda.pybind.geometry.RGBDImage]], intrinsic: PinholeCameraIntrinsic):

    wp = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
    waypoints = [copy.deepcopy(wp)]

    option = o3d.pipelines.odometry.OdometryOption(depth_min=0.5, depth_max=2.3,)
    odo_init = np.identity(4)
    transformations = [odo_init]

    for i in range(len(frames) - 1):
        src_path, src_rgbd = frames[i]
        dst_path, dst_rgbd = frames[i + 1]

        success, transformation, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            src_rgbd,
            dst_rgbd,
            intrinsic,
            # transformations[i],
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        if success:
            transformation = np.linalg.inv(transformation)
            rot = R.from_matrix(transformation[:3, :3])
            yaw, pitch, roll = rot.as_euler('zyx', degrees=True)
            transformations.append(transformation)
            print("[odometry] success", transformation)
            print(
                f"[{i:04d}] {src_path.name} -> {dst_path.name} | "
                f"translation: {transformation[:3, 3]} | "
                f"rotation: {transformation[:3, :3]} | yaw={yaw}, pitch={pitch}, roll={roll}"
            )

            # new_wp = copy.deepcopy(wp).transform(transformation)
            new_wp = copy.deepcopy(waypoints[-1]).transform(transformation)
            print("new waypoint", new_wp)
            waypoints.append(new_wp)
        else:
            print("[odometry] failed")
    return waypoints, transformations


def integrate(frames: list[tuple[Path, o3d.cuda.pybind.geometry.RGBDImage]],
              camera_poses: List,
              camera_intrinsics: PinholeCameraIntrinsic):
    assert len(frames) == len(camera_poses), "The number of frames and camera poses must be the same."
    rgbd_data = [f[1] for f in frames]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        length=4.0,
        resolution=512,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for pose, rgbd in zip(camera_poses, rgbd_data):
        volume.integrate(rgbd, camera_intrinsics, pose)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

def main(args):
    in_dir = Path(args.in_dir).expanduser().resolve()
    frames = read_frames(in_dir, max_imgs=5)

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

    # o3d.visualization.draw_geometries([frames[0][1]])
    # o3d.visualization.draw_geometries([frames[1][1]])
    # o3d.visualization.draw_geometries([frames[2][1]])
    # o3d.visualization.draw_geometries([frames[3][1]])
    # o3d.visualization.draw_geometries([frames[4][1]])
    # o3d.visualization.draw_geometries([frames[5][1]])
    # visualize_rgbd(frames[0][1], intrinsic)


    rgbd_list = [f[1] for f in frames]

    trajectory, transformations = compute_trajectory(frames, intrinsic)

    visualize_rgbd_list(rgbd_list, intrinsic, transformations)
    # o3d.visualization.draw_geometries(trajectory)
    integrate(frames, transformations, intrinsic)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    work_dir = Path(__file__).parent.resolve()
    parser.add_argument("--in-dir", default=Path.home() / "Pictures/rgbd_exports")
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--vfov-deg", type=float, default=20.0)
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