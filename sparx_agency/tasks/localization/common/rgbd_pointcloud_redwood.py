import copy
import open3d as o3d
from open3d.cuda.pybind.camera import PinholeCameraIntrinsic
import numpy as np
from sparx_agency.tasks.localization.common.rgbd_utils import print_transformation


def compute_odometry_color(source_rgbd_image, target_rgbd_image, intrinsic: PinholeCameraIntrinsic):
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    [success_color_term, trans_color_term,
     info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)

    print("compute_odometry_color",  "Success", success_color_term)
    print_transformation(trans_color_term)

    return (
        success_color_term,
        trans_color_term,
    )

def compute_odometry_hybrid(source_rgbd_image, target_rgbd_image, intrinsic: PinholeCameraIntrinsic):
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    [success_hybrid_term, trans_hybrid_term,
     info] = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    print("compute_odometry_hybrid",  "Success", success_hybrid_term)
    print_transformation(trans_hybrid_term)

    return (
        success_hybrid_term,
        trans_hybrid_term,
    )

def visualize_pointcloud(pcd):
    o3d.visualization.draw_geometries(pcd,
                                      zoom=0.48,
                                      front=[0.0999, -0.1787, -0.9788],
                                      lookat=[0.0345, -0.0937, 1.8033],
                                      up=[-0.0067, -0.9838, 0.1790])


def compute_redwood_pointcloud(source_rgbd_image, target_rgbd_image, intrinsic: PinholeCameraIntrinsic):
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, intrinsic)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, intrinsic)


    success_color_term, trans_color_term = compute_odometry_color(source_rgbd_image, target_rgbd_image, intrinsic)
    success_hybrid_term, trans_hybrid_term = compute_odometry_hybrid(source_rgbd_image, target_rgbd_image, intrinsic)

    if success_color_term:
        source_pcd_color_term = copy.deepcopy(source_pcd)
        source_pcd_color_term.transform(trans_color_term)
        visualize_pointcloud([target_pcd, source_pcd_color_term])
    if success_hybrid_term:
        source_pcd_hybrid_term = copy.deepcopy(source_pcd)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        visualize_pointcloud([target_pcd, source_pcd_hybrid_term])

def read_frames_redwood():
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    source_color = o3d.io.read_image(redwood_rgbd.color_paths[0])
    source_depth = o3d.io.read_image(redwood_rgbd.depth_paths[0])
    target_color = o3d.io.read_image(redwood_rgbd.color_paths[3])
    target_depth = o3d.io.read_image(redwood_rgbd.depth_paths[3])

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)

    intrinsics = o3d.io.read_pinhole_camera_intrinsic(
        redwood_rgbd.camera_intrinsic_path)
    return source_rgbd_image, target_rgbd_image, intrinsics


def main():
    source_rgbd_image, target_rgbd_image, intrinsics = read_frames_redwood()
    compute_redwood_pointcloud(source_rgbd_image, target_rgbd_image, intrinsics)


    # pcd_list = compute_pointcloud(rgbd_list, intrinsic)
    # o3d.visualization.draw_geometries(pcd_list)



if __name__ == "__main__":
    main()
