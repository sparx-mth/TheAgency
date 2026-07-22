import time
import csv
import os
from datetime import datetime
import rclpy
from rclpy.node import Node
import numpy as np
import json
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from sparx_agency.core.common.utils import XYZAccumulator
from sparx_agency.core.mapping.depth.depth_bbox_fusion import pointcloud2_to_xyz_array, xyz_from_unorganized_cloud_bbox
from sparx_agency.robots.common.image_utils import (
    robust_depth_from_bbox_hist,
    uvz_to_xyz_camera,
)
from sparx_agency.core.common.spatial_math import pose_xyz_yaw_to_T


class DepthBBoxWorldTest(Node):
    def __init__(self, json_path):
        super().__init__("depth_bbox_world_test")

        self.last_process_time = 0
        self.depth_stamp = None
        self.last_depth_stamp = 0
        self.cloud_xyz = None
        self.bridge = CvBridge()
        self.depth = None
        self.caminfo = None
        self.rgb = None

        self.acc_hist = XYZAccumulator(max_len=10)
        self.acc_cloud = XYZAccumulator(max_len=10)

        with open(json_path) as f:
            self.data = json.load(f)

        self.sub_depth = self.create_subscription(
            Image, "/video/depth", self.cb_depth, 10)
        self.sub_rgb = self.create_subscription(
            Image, "/video/image", self.cb_rgb, 10)
        self.sub_cam = self.create_subscription(
            CameraInfo, "/video/camera_info", self.cb_cam, 10)
        self.sub_cloud = self.create_subscription(
            PointCloud2, "/video/point_cloud", self.cb_cloud, 10
        )

        self.timer = self.create_timer(0.5, self.try_process)

    def cb_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_stamp = msg.header.stamp

    def cb_rgb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def cb_cam(self, msg):
        self.caminfo = msg

    def cb_cloud(self, msg):
        self.cloud_xyz = pointcloud2_to_xyz_array(msg)

    def try_process(self):
        if self.depth is None or self.caminfo is None:
            return

        if time.time() - self.last_process_time < 1:
            return
        self.last_process_time = time.time()

        self.last_depth_stamp = self.depth_stamp

        fx, fy = self.caminfo.k[0], self.caminfo.k[4]
        cx, cy = self.caminfo.k[2], self.caminfo.k[5]


        dets = self.data["nanoowl"]["result"]["detections"]

        print("\n=== OBJECTS ===")
        pose = self.data["pose"]
        T_wc = pose_xyz_yaw_to_T(
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            yaw=pose["yaw"],
        )

        for d in dets:
            x1, y1, x2, y2 = map(int, d["bbox"])

            z_obj = robust_depth_from_bbox_hist(
                self.depth,
                (x1, y1, x2, y2),
                min_depth=0.2,
                max_depth=10.0,
            )
            if z_obj is None:
                continue

            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            # --- Method A: depth histogram ---
            z_obj = robust_depth_from_bbox_hist(self.depth, (x1, y1, x2, y2))
            if z_obj is not None:
                xyz_cam = uvz_to_xyz_camera(u, v, z_obj, fx, fy, cx, cy)
                xyz_world = (T_wc @ np.append(xyz_cam, 1.0))[:3]
                self.acc_hist.add(xyz_world)
                print("acc_hist size:", len(self.acc_hist.samples))

            # --- Method B: pointcloud ---
            # Method B: pointcloud
            if self.cloud_xyz is not None:
                xyz_pc_cam = xyz_from_unorganized_cloud_bbox(
                    self.cloud_xyz,
                    fx, fy, cx, cy,
                    (x1, y1, x2, y2),
                )
                if xyz_pc_cam is not None:
                    xyz_pc_world = (T_wc @ np.append(xyz_pc_cam, 1.0))[:3]
                    self.acc_cloud.add(xyz_pc_world)

            # print(f"{d['label']} -> world XYZ = {xyz_world}")

            if self.acc_hist.is_full() and self.acc_cloud.is_full():
                print_comparison_table(self.acc_hist, self.acc_cloud, d['label'])
                save_comparison_csv(path="comparison_table", label=d["label"], hist_acc=self.acc_hist, cloud_acc=self.acc_cloud )
                self.get_logger().info("Accumulator full, stopping timer.")
                self.destroy_timer(self.timer)



def save_comparison_csv(
    path,
    label,
    hist_acc,
    cloud_acc,
):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    mean_h = hist_acc.mean()
    std_h = hist_acc.std()
    mean_c = cloud_acc.mean()
    std_c = cloud_acc.std()
    bias = mean_c - mean_h

    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "label",
                "method",
                "mean_x", "mean_y", "mean_z",
                "std_x", "std_y", "std_z",
            ])

        ts = datetime.utcnow().isoformat()

        writer.writerow([ts, label, "depth_hist",
                         *mean_h, *std_h])
        writer.writerow([ts, label, "pointcloud",
                         *mean_c, *std_c])
        writer.writerow([ts, label, "bias_pc_minus_hist",
                         *bias, 0, 0, 0])


def print_comparison_table(acc_a, acc_b, label="object"):
    A = acc_a.as_array()
    B = acc_b.as_array()
    if A is None or B is None:
        return

    mean_a, std_a = acc_a.mean(), acc_a.std()
    mean_b, std_b = acc_b.mean(), acc_b.std()
    diff = mean_b - mean_a

    print("\n=== Comparison:", label, "===")
    print("Method           Mean XYZ                 Std XYZ")
    print(f"Depth-Hist   {mean_a.round(3)}   {std_a.round(3)}")
    print(f"PointCloud   {mean_b.round(3)}   {std_b.round(3)}")
    print("Bias (PC - Hist):", diff.round(3))


def main():
    path_to_json_file = "/home/daphnaa/GIT/NanoLLM_VILA_and_OWL/room_mapping/OneDrive_1_10-20-2025/x0000y0900z0000yaw2268928__2025_10_22___14_09_31.json"
    rclpy.init()
    node = DepthBBoxWorldTest(path_to_json_file)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
