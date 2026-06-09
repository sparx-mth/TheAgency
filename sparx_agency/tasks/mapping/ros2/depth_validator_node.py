import os
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import csv
from datetime import datetime

# Importing your specific math class
from sparx_agency.core.common.spatial_math import get_euler


class DepthValidatorNode(Node):
    def __init__(self):
        super().__init__('depth_validator_node')

        self.bridge = CvBridge()
        self.prev_da3_pts = None
        self.n_points = 16
        self.idx_x = None
        self.idx_y = None
        self.temp_sub = None
        # Parameters
        self.declare_parameter('da3_topic', '/sparx/depth/da3_raw')
        self.declare_parameter('show_viz', True)

        self.da3_topic = self.get_parameter('da3_topic').value
        self.show_viz = self.get_parameter('show_viz').value

        log_dir =  f"{Path.home() / 'Documents/depth_validator_csv'}"
        log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f"{log_dir}{os.sep}da3_val_{log_time}.csv"
        self.frame_saved = False
        self.image_save_path = os.path.join(log_dir, f"{log_time}.jpg")

        # Open file once and keep it open for performance
        self.csv_fp = open(self.log_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_fp)
        # Create headers like pt0_err, pt1_err, etc.
        pt_err_cols = [f"pt{i}_err" for i in range(16)]
        pt_jitter_cols = [f"pt{i}_jitter" for i in range(16)]
        gt_cols = [f"gt_depth_{i}" for i in range(16)]

        header = ["ts", "roll", "pitch", "yaw", "mae", "rmse"] + pt_err_cols + pt_jitter_cols + gt_cols
        self.csv_writer.writerow(header)

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
            cv_da3 = self.bridge.imgmsg_to_cv2(da3_msg, 'passthrough').astype(np.float32)
            cv_gt_raw = self.bridge.imgmsg_to_cv2(gt_msg, 'passthrough').astype(np.float32)

            # 1. Spatial Alignment
            h, w = cv_da3.shape
            cv_gt = cv2.resize(cv_gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            if len(cv_gt.shape) == 3: cv_gt = cv_gt[:, :, 0]
            if cv_gt.max() > 11.0: cv_gt = (cv_gt / 255.0) * 10.0

            # 2. Initialize Fixed Random Points (Only once)
            if self.idx_x is None:
                self.get_logger().info(f"Locking {self.n_points} random points for stability testing.")
                self.idx_y = np.random.randint(int(h * 0.45), int(h * 0.90), self.n_points)
                self.idx_x = np.random.randint(int(w * 0.15), int(w * 0.85), self.n_points)

            # 3. Sample values at the FIXED locations
            da3_pts = cv_da3[self.idx_y, self.idx_x]
            gt_pts = cv_gt[self.idx_y, self.idx_x]

            point_errors = np.abs(da3_pts - gt_pts)
            point_jitters = np.abs(da3_pts - self.prev_da3_pts) if self.prev_da3_pts is not None else np.zeros_like(
                da3_pts)

            # 4. Stability (Temporal Jitter) calculation
            # This measures how much the prediction for these EXACT pixels changed since last frame
            jitter = 0.0
            if self.prev_da3_pts is not None:
                jitter = np.mean(np.abs(da3_pts - self.prev_da3_pts))

            self.prev_da3_pts = da3_pts.copy()

            # 5. Accuracy Metrics
            mask = np.isfinite(gt_pts) & (gt_pts > 0.1)
            if not np.any(mask): return

            mae = np.mean(np.abs(da3_pts[mask] - gt_pts[mask]))
            rmse = np.sqrt(np.mean((da3_pts[mask] - gt_pts[mask]) ** 2))

            # 6. Logging for Report
            q = odom_msg.pose.pose.orientation
            r, p, y = get_euler(q)
            ts = da3_msg.header.stamp.sec + da3_msg.header.stamp.nanosec * 1e-9

            # Logging the new 'jitter' metric
            # Construct row: [metadata] + [list of errors]
            row = [ts, r, p, y, mae, rmse] + \
                  point_errors.tolist() + \
                  point_jitters.tolist() + \
                  gt_pts.tolist()
            self.csv_writer.writerow(row)
            self.csv_fp.flush()

            # 1. Capture ONLY ONE RGB frame for the whole run
            if not self.frame_saved:
                def one_off_rgb_cb(rgb_msg):
                    if self.frame_saved: return
                    try:
                        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                        # Draw your sampling points on this ONE image for the dashboard
                        for x, y in zip(self.idx_x, self.idx_y):
                            cv2.circle(cv_rgb, (x, y), 5, (0, 255, 0), -1)

                        cv2.imwrite(self.image_save_path, cv_rgb)
                        self.get_logger().info(f"Saved reference RGB frame to {self.image_save_path}")
                        self.frame_saved = True
                    except Exception as exp:
                        self.get_logger().error(f"Failed to save ref frame: {exp}")

                # Create a temporary one-shot subscriber
                self.temp_sub = self.create_subscription(
                    Image, '/simple_drone/front/image_raw', one_off_rgb_cb, 10)

            if self.show_viz:
                self.visualize(cv_da3, cv_gt, mask, mae, (r, p, y))

        except Exception as e:
            self.get_logger().error(f"Sync error: {e}")

    def visualize(self, da3_img, gt_img, valid_mask, mae, rpy):
        def colorize(img):
            # 1. Clip the range to your max sensor distance (10m)
            # This prevents 'Inf' values from ruining the scale
            depth_clipped = np.clip(img, 0.1, 10.0)
            # 2. Normalize to 0.0 - 1.0 range
            depth_norm = depth_clipped / 10.0

            # 3. Scale to 0 - 255 and convert to 8-bit
            depth_8bit = (depth_norm * 255).astype(np.uint8)

            # 4. Apply a colormap so you can actually see the gradients
            return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)

        vis_da3 = colorize(da3_img)
        vis_gt = colorize(gt_img)

        # Loop through the 1D index arrays
        for i in range(len(self.idx_x)):
            pixel_x = self.idx_x[i]
            pixel_y = self.idx_y[i]
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
    except (KeyboardInterrupt,  ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()