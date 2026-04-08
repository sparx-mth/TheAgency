
import os
from pathlib import Path

import csv
from datetime import datetime
import math

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from sparx_agency.robots.common.spatial_math import get_euler, quat_to_rot, euler_to_rot_zyx
from sparx_agency.tasks.mapping.common.validator_landmarks import LANDMARK_OBJECTS


class DepthValidatorNode(Node):
    def __init__(self):
        super().__init__('depth_validator_node')

        self.bridge = CvBridge()
        self.prev_depth_by_landmark = {}
        self.frame_saved = False
        self.camera_info_msg = None
        self.landmarks_world = self._build_world_landmarks(LANDMARK_OBJECTS)
        self.visible_landmarks_last = []

        # Parameters
        self.declare_parameter('da3_topic', '/sparx/depth/da3_raw')
        self.declare_parameter('gt_topic', '/simple_drone/front_depth/depth/image_raw')
        self.declare_parameter('rgb_topic', '/simple_drone/front/image_raw')
        self.declare_parameter('camera_info_topic', '/simple_drone/front/camera_info')
        self.declare_parameter('odom_topic', '/simple_drone/odom')
        self.declare_parameter('show_viz', True)
        self.declare_parameter('patch_radius', 2)
        self.declare_parameter('min_gt_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 10.0)
        self.declare_parameter('camera_frame_mode', 'front_cam_x_forward')

        self.da3_topic = self.get_parameter('da3_topic').value
        self.gt_topic = self.get_parameter('gt_topic').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.show_viz = bool(self.get_parameter('show_viz').value)
        self.patch_radius = int(self.get_parameter('patch_radius').value)
        self.min_gt_depth_m = float(self.get_parameter('min_gt_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.camera_frame_mode = str(self.get_parameter('camera_frame_mode').value)

        log_dir = Path.home() / 'Documents' / 'depth_validator_csv'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = str(log_dir / f'da3_landmarks_{log_time}.csv')
        self.image_save_path = str(log_dir / f'{log_time}_rgb_landmarks.jpg')

        self.csv_fp = open(self.log_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_fp)
        self.csv_writer.writerow([
            'ts', 'object_id', 'landmark_id',
            'px', 'py',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'gt_depth_m', 'da3_depth_m', 'abs_err_m', 'jitter_m',
            'gt_x_cam_m', 'gt_y_cam_m', 'gt_z_cam_m'
        ])

        # Subscribers
        self.da3_sub = message_filters.Subscriber(self, Image, self.da3_topic)
        self.gt_sub = message_filters.Subscriber(self, Image, self.gt_topic)
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.odom_sub = message_filters.Subscriber(self, Odometry, self.odom_topic)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, 10
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.da3_sub, self.gt_sub, self.rgb_sub, self.odom_sub],
            queue_size=20,
            slop=0.1,
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f'Validator Started. Logging to {self.log_file}')
        self.get_logger().info(f'Loaded {len(self.landmarks_world)} world landmarks.')

    def camera_info_cb(self, msg: CameraInfo):
        self.camera_info_msg = msg

    def _build_world_landmarks(self, objects_cfg):
        landmarks = []
        for obj in objects_cfg:
            pos = obj['pose_world']['position'].astype(np.float32)
            rpy = obj['pose_world']['rpy'].astype(np.float32)
            R_obj = euler_to_rot_zyx(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            for landmark_id, p_local in obj['points_local'].items():
                p_world = (R_obj @ p_local.reshape(3, 1)).reshape(3) + pos
                landmarks.append({
                    'object_id': obj['id'],
                    'color': obj['color'],
                    'landmark_id': landmark_id,
                    'p_world': p_world.astype(np.float32),
                })
        return landmarks

    def _camera_pose_from_odom(self, odom_msg: Odometry):
        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation

        t_world_base = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        R_world_base = quat_to_rot(quat.x, quat.y, quat.z, quat.w)

        # Fallback: assume front camera is collocated with base and looks forward.
        # If later you want, we can add exact extrinsics here.
        t_world_cam = t_world_base.copy()
        R_world_cam = R_world_base.copy()
        return R_world_cam, t_world_cam

    def _project_world_point(self, p_world, R_world_cam, t_world_cam, fx, fy, cx, cy, width, height):
        p_cam = R_world_cam.T @ (p_world - t_world_cam)

        # camera frame expected by projection: X forward, Y left, Z up
        x_fwd = float(p_cam[0])
        y_left = float(p_cam[1])
        z_up = float(p_cam[2])

        if x_fwd <= 1e-4:
            return None

        u = cx - fx * (y_left / x_fwd)
        v = cy - fy * (z_up / x_fwd)

        if not (0 <= u < width and 0 <= v < height):
            return None

        return {
            'u': int(round(u)),
            'v': int(round(v)),
            'gt_depth_m': x_fwd,
            'p_cam': np.array([x_fwd, y_left, z_up], dtype=np.float32),
        }

    def _sample_patch_median(self, img, u, v):
        h, w = img.shape[:2]
        r = self.patch_radius
        x0 = max(0, u - r)
        x1 = min(w, u + r + 1)
        y0 = max(0, v - r)
        y1 = min(h, v + r + 1)
        patch = img[y0:y1, x0:x1]
        vals = patch[np.isfinite(patch)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals))

    def sync_callback(self, da3_msg, gt_msg, rgb_msg, odom_msg):
        if self.camera_info_msg is None:
            return

        try:
            cv_da3 = self.bridge.imgmsg_to_cv2(da3_msg, 'passthrough').astype(np.float32)
            cv_gt_raw = self.bridge.imgmsg_to_cv2(gt_msg, 'passthrough').astype(np.float32)
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')

            h, w = cv_da3.shape
            cv_gt = cv2.resize(cv_gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            if len(cv_gt.shape) == 3:
                cv_gt = cv_gt[:, :, 0]
            if cv_gt.max() > 11.0:
                cv_gt = (cv_gt / 255.0) * 10.0

            # Scale intrinsics from RGB camera to DA3 image resolution if needed
            k = self.camera_info_msg.k
            rgb_w = int(self.camera_info_msg.width)
            rgb_h = int(self.camera_info_msg.height)
            sx = float(w) / float(rgb_w)
            sy = float(h) / float(rgb_h)
            fx = float(k[0]) * sx
            fy = float(k[4]) * sy
            cx = float(k[2]) * sx
            cy = float(k[5]) * sy

            R_world_cam, t_world_cam = self._camera_pose_from_odom(odom_msg)

            q = odom_msg.pose.pose.orientation
            roll_deg, pitch_deg, yaw_deg = get_euler(q)
            ts = da3_msg.header.stamp.sec + da3_msg.header.stamp.nanosec * 1e-9

            vis_rows = []
            abs_errs = []

            for lm in self.landmarks_world:
                proj = self._project_world_point(
                    lm['p_world'], R_world_cam, t_world_cam,
                    fx, fy, cx, cy, w, h
                )
                if proj is None:
                    continue

                u = proj['u']
                v = proj['v']
                gt_depth_geom = float(proj['gt_depth_m'])
                gt_depth_img = self._sample_patch_median(cv_gt, u, v)
                da3_depth = self._sample_patch_median(cv_da3, u, v)

                if not np.isfinite(da3_depth):
                    continue

                # Prefer geometry GT; keep depth image only as a sanity source.
                gt_depth = gt_depth_geom
                if gt_depth <= self.min_gt_depth_m or gt_depth > self.max_depth_m:
                    continue

                abs_err = abs(da3_depth - gt_depth)
                key = (lm['object_id'], lm['landmark_id'])
                prev_depth = self.prev_depth_by_landmark.get(key)
                jitter = abs(da3_depth - prev_depth) if prev_depth is not None else 0.0
                self.prev_depth_by_landmark[key] = da3_depth

                self.csv_writer.writerow([
                    ts,
                    lm['object_id'], lm['landmark_id'],
                    u, v,
                    roll_deg, pitch_deg, yaw_deg,
                    gt_depth, da3_depth, abs_err, jitter,
                    float(proj['p_cam'][0]), float(proj['p_cam'][1]), float(proj['p_cam'][2]),
                ])

                vis_rows.append({
                    'u': u, 'v': v,
                    'object_id': lm['object_id'],
                    'landmark_id': lm['landmark_id'],
                    'gt_depth': gt_depth,
                    'gt_depth_img': gt_depth_img,
                    'da3_depth': da3_depth,
                    'abs_err': abs_err,
                })
                abs_errs.append(abs_err)

            self.csv_fp.flush()
            self.visible_landmarks_last = vis_rows

            if not self.frame_saved and vis_rows:
                rgb_save = cv_rgb.copy()
                for row in vis_rows:
                    cv2.circle(rgb_save, (row['u'], row['v']), 5, (0, 255, 0), -1)
                    cv2.putText(
                        rgb_save,
                        f"{row['object_id']}:{row['landmark_id']}",
                        (row['u'] + 4, row['v'] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 0),
                        1,
                    )
                cv2.imwrite(self.image_save_path, rgb_save)
                self.get_logger().info(f'Saved reference RGB frame to {self.image_save_path}')
                self.frame_saved = True

            if self.show_viz:
                mae = float(np.mean(abs_errs)) if abs_errs else float('nan')
                self.visualize(cv_da3, cv_gt, vis_rows, mae, (roll_deg, pitch_deg, yaw_deg))

        except Exception as e:
            self.get_logger().error(f'Sync error: {e}')

    def visualize(self, da3_img, gt_img, landmarks, mae, rpy):
        def colorize(img):
            depth_clipped = np.clip(img, 0.1, self.max_depth_m)
            depth_norm = depth_clipped / self.max_depth_m
            depth_8bit = (depth_norm * 255).astype(np.uint8)
            return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)

        vis_da3 = colorize(da3_img)
        vis_gt = colorize(gt_img)

        for row in landmarks:
            px = row['u']
            py = row['v']
            color = (0, 255, 0)
            cv2.circle(vis_da3, (px, py), 4, color, -1)
            cv2.circle(vis_gt, (px, py), 4, color, -1)
            label = row['landmark_id']
            cv2.putText(vis_da3, label, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(vis_gt, label, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        combined = np.hstack((vis_da3, vis_gt))
        info_str = f'Landmarks: {len(landmarks)} | MAE: {mae:.3f}m | Pitch: {rpy[1]:.1f}deg'
        cv2.putText(combined, info_str, (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Validation: Left(DA3) vs Right(GT)', combined)
        cv2.waitKey(1)

    def destroy_node(self):
        self.csv_fp.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = DepthValidatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
