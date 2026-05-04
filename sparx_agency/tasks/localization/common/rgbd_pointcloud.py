import argparse
from pathlib import Path
import open3d as o3d
from open3d.cuda.pybind.camera import PinholeCameraIntrinsic
import numpy as np
from sparx_agency.tasks.localization.common.camera_model import make_intrinsic_from_image
from sparx_agency.tasks.localization.common.rgbd_utils import read_frames, print_transformation, print_depth_stats, \
    make_rgbd_image


def compute_pointcloud(frames: list[o3d.cuda.pybind.geometry.RGBDImage], intrinsic: PinholeCameraIntrinsic):
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    transformation_list = [odo_init]
    pcd_list = [o3d.geometry.PointCloud.create_from_rgbd_image(frames[i], intrinsic).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) for i in range(len(frames))]
    for i in range(len(frames) - 1):
        src_rgbd = frames[i]
        dst_rgbd = frames[i + 1]

        success, transformation, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            src_rgbd,
            dst_rgbd,
            intrinsic,
            # transformation_list[i],
            odo_init,
            # o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        print_transformation(transformation)
        pcd_list[i].transform(transformation)


    return pcd_list

def compute_redwood_pointcloud(source_rgbd_image, target_rgbd_image, intrinsic: PinholeCameraIntrinsic):
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, intrinsic)
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    [success_color_term, trans_color_term,
     info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term,
     info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        print_transformation(trans_color_term)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term],
                                          zoom=0.48,
                                          front=[0.0999, -0.1787, -0.9788],
                                          lookat=[0.0345, -0.0937, 1.8033],
                                          up=[-0.0067, -0.9838, 0.1790])
    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        print_transformation(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term],
                                          zoom=0.48,
                                          front=[0.0999, -0.1787, -0.9788],
                                          lookat=[0.0345, -0.0937, 1.8033],
                                          up=[-0.0067, -0.9838, 0.1790])
def read_frames_redwood():
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    source_color = o3d.io.read_image(redwood_rgbd.color_paths[0])
    source_depth = o3d.io.read_image(redwood_rgbd.depth_paths[0])
    target_color = o3d.io.read_image(redwood_rgbd.color_paths[3])
    target_depth = o3d.io.read_image(redwood_rgbd.depth_paths[3])
    # source_rgbd_image = make_rgbd_image(np.asarray(source_color), np.asarray(source_depth))
    # target_rgbd_image = make_rgbd_image(np.asarray(target_color), np.asarray(target_depth))
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
    # source_rgbd_image = o3d.cuda.pybind.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
    # target_rgbd_image = o3d.cuda.pybind.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
    intrinsics = o3d.io.read_pinhole_camera_intrinsic(
        redwood_rgbd.camera_intrinsic_path)
    return source_rgbd_image, target_rgbd_image, intrinsics

def intrinsic_from_redwood():
    return o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

def main(args):
    in_dir = Path(args.in_dir).expanduser().resolve()
    frames = read_frames(in_dir, max_imgs=2, start_from=3)


    # depths = [f[1].depth for f in frames]
    # depths = [f.depth for f in frames]
    # print_depth_stats(depths)
    # return
    # first_color = np.asarray(frames[0][1].color)
    # height, width = first_color.shape[:2]
    #
    # intrinsic = make_intrinsic_from_image(
    #     width=width,
    #     height=height,
    #     hfov_deg=args.hfov_deg,
    #     vfov_deg=args.vfov_deg
    # )

    # print(f"[intrinsic] width={width}, height={height}, hfov_deg={args.hfov_deg}, vfov_deg={args.vfov_deg}")
    # print(intrinsic)

    # rgbd_list = [f[1] for f in frames]

    # pcd_list = compute_pointcloud(rgbd_list, intrinsic)
    # o3d.visualization.draw_geometries(pcd_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    work_dir = Path(__file__).parent.resolve()
    parser.add_argument("--in-dir", default=Path.home() / "Pictures/rgbd_exports")
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--vfov-deg", type=float, default=90.0)
    args = parser.parse_args()
    main(args)
