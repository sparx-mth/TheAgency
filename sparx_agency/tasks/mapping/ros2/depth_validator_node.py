import os
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import math
import csv
from datetime import datetime

# Importing your specific math class
from sparx_agency.robots.common.spatial_math import get_euler


class DepthValidatorNode(Node):
    def __init__(self):
        super().__init__('depth_validator_node')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('da3_topic', '/sparx/depth/da3_raw')
        self.declare_parameter('show_viz', True)

        self.da3_topic = self.get_parameter('da3_topic').value
        self.show_viz = self.get_parameter('show_viz').value

        self.log_file = f"{Path.home() / 'Documents/IAI/depth_validator_csv'}{os.sep}da3_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Open file once and keep it open for performance
        self.csv_fp = open(self.log_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_fp)
        self.csv_writer.writerow(['ts', 'roll', 'pitch', 'yaw', 'mae', 'rmse', 'temporal_drift'])

        # Subscribers
        self.da3_sub = message_filters.Subscriber(self, Image, self.da3_topic)
        # Ensure this matches your Gazebo topic exactly
        self.gt_sub = message_filters.Subscriber(self, Image, '/simple_drone/front_depth/depth/image_raw')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/simple_drone/odom')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.da3_sub, self.gt_sub, self.odom_sub],
            queue_size=20, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f"Validator Started. Logging to {self.log_file}")

    def sync_callback(self, da3_msg, gt_msg, odom_msg):
        try:
            # 1. Convert to float32
            cv_da3 = self.bridge.imgmsg_to_cv2(da3_msg, desired_encoding='passthrough').astype(np.float32)
            cv_gt_raw = self.bridge.imgmsg_to_cv2(gt_msg, desired_encoding='passthrough').astype(np.float32)

            # 2. Safety Check: If Gazebo is still sending 0-255 instead of 0-10m
            # We check the max value. If it's > 11, it's definitely not meters.
            if cv_gt_raw.max() > 11.0:
                cv_gt_raw = (cv_gt_raw / 255.0) * 10.0

            # 3. Handle RGB vs Single Channel
            if len(cv_gt_raw.shape) == 3:
                cv_gt = cv_gt_raw[:, :, 0]
            else:
                cv_gt = cv_gt_raw

            # 4. Resize to match DA3 (540 -> 504)
            h, w = cv_da3.shape
            cv_gt = cv2.resize(cv_gt, (w, h), interpolation=cv2.INTER_NEAREST)

            # 5. Grid Sampling
            y_coords = np.linspace(h * 0.3, h * 0.7, 3).astype(int)
            x_coords = np.linspace(w * 0.2, w * 0.8, 5).astype(int)
            yy, xx = np.meshgrid(y_coords, x_coords)
            idx_y, idx_x = yy.flatten(), xx.flatten()

            da3_pts = cv_da3[idx_y, idx_x]
            gt_pts = cv_gt[idx_y, idx_x]

            # 6. Mask and Calculate
            valid_mask = np.isfinite(gt_pts) & (gt_pts > 0.1)
            if not np.any(valid_mask): return

            errors = np.abs(da3_pts[valid_mask] - gt_pts[valid_mask])
            mae = np.mean(errors)
            rmse = np.sqrt(np.mean(errors ** 2))

            # 7. Attitude and Logging
            q = odom_msg.pose.pose.orientation
            r, p, y = get_euler(q)

            # Log to CSV
            ts = da3_msg.header.stamp.sec + da3_msg.header.stamp.nanosec * 1e-9
            self.csv_writer.writerow([ts, r, p, y, mae, rmse, 0.0])
            self.csv_fp.flush()

            if self.show_viz:
                self.visualize(cv_da3, cv_gt, idx_x, idx_y, valid_mask, mae, (r, p, y))

        except Exception as e:
            self.get_logger().error(f"Sync error: {e}")

    def visualize(self, da3_img, gt_img, idx_x, idx_y, valid_mask, mae, rpy):
        def colorize(img):
            # Clipping to 10m provides a consistent color scale for comparison
            norm = np.clip(img / 10.0, 0, 1)
            return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

        vis_da3 = colorize(da3_img)
        vis_gt = colorize(gt_img)

        # Loop through the 1D index arrays
        for i in range(len(idx_x)):
            pixel_x = idx_x[i]
            pixel_y = idx_y[i]
            color = (0, 255, 0) if valid_mask[i] else (0, 0, 255)

            # Draw on both images for comparison
            cv2.circle(vis_da3, (pixel_x, pixel_y), 4, color, -1)
            cv2.circle(vis_gt, (pixel_x, pixel_y), 4, color, -1)

        combined = np.hstack((vis_da3, vis_gt))

        # Add metrics overlay
        info_str = f"MAE: {mae:.3f}m | Pitch: {rpy[1]:.1f}deg"
        cv2.putText(combined, info_str, (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Validation: Left(DA3) vs Right(GT)", combined)
        cv2.waitKey(1)

    def destroy_node(self):
        self.csv_fp.close()  # Close file gracefully
        super().destroy_node()


def main():
    rclpy.init()
    node = DepthValidatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()