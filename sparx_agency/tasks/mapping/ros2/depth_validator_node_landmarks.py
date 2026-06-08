
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

from sparx_agency.robots.common.spatial_math import quat_msg_to_rpy_deg, quat_to_rot, euler_to_rot_zyx
from sparx_agency.tasks.mapping.common.validator_markers import MARKER_OBJECTS

def wrap_hue_interval(hsv_h: np.ndarray, low: int, high: int) -> np.ndarray:
    if low <= high:
        return (hsv_h >= low) & (hsv_h <= high)
    return (hsv_h >= low) | (hsv_h <= high)


class DepthValidatorNode(Node):
    def __init__(self):
        super().__init__('depth_validator_node')

        self.bridge = CvBridge()
        self.prev_depth_by_marker = {}
        self.frame_saved = False
        self.camera_info_msg = None
        self.markers = MARKER_OBJECTS
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
        self.declare_parameter('min_blob_area_px', 30)
        self.declare_parameter('max_blob_area_px', 20000)
        self.declare_parameter('association_max_px', 90.0)
        self.declare_parameter('morph_kernel_px', 5)



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
        self.min_blob_area_px = int(self.get_parameter('min_blob_area_px').value)
        self.max_blob_area_px = int(self.get_parameter('max_blob_area_px').value)
        self.association_max_px = float(self.get_parameter('association_max_px').value)
        self.morph_kernel_px = int(self.get_parameter('morph_kernel_px').value)

        # ===== Calibration (from CSV) =====
        self.lin_m = 0.5005
        self.lin_b = 0.6114

        self.quad_a = 0.05296
        self.quad_b = 0.1069
        self.quad_c = 1.1834

        log_dir = Path.home() / 'Documents' / 'depth_validator_csv'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = str(log_dir / f'da3_markers_{log_time}.csv')
        self.image_save_path = str(log_dir / f'{log_time}_rgb_landmarks.jpg')

        self.csv_fp = open(self.log_file, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_fp)
        self.csv_writer.writerow([
            'ts', 'marker_id', 'color',
            'det_u', 'det_v', 'gt_u', 'gt_v', 'pixel_err_px',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            'gt_depth_geom_m', 'gt_depth_img_m',
            'da3_raw_m', 'da3_lin_m', 'da3_quad_m',
            'err_raw_m', 'err_lin_m', 'err_quad_m',
            'jitter_m',
            'gt_x_cam_m', 'gt_y_cam_m', 'gt_z_cam_m',
            'blob_area_px', 'detected'
        ])

        # Subscribers
        self.da3_sub = message_filters.Subscriber(self, Image, self.da3_topic)
        self.gt_sub = message_filters.Subscriber(self, Image, self.gt_topic)
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self.odom_sub = message_filters.Subscriber(self, Odometry, self.odom_topic)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_cb, 10
        )

        self.pub_raw = self.create_publisher(Image, '/sparx/depth/da3_raw_dbg', 10)
        self.pub_lin = self.create_publisher(Image, '/sparx/depth/da3_linear', 10)
        self.pub_quad = self.create_publisher(Image, '/sparx/depth/da3_quadratic', 10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.da3_sub, self.gt_sub, self.rgb_sub, self.odom_sub],
            queue_size=20,
            slop=0.1,
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(f'Validator Started. Logging to {self.log_file}')
        self.get_logger().info(f'Loaded {len(self.markers)} markers.')

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

        # front camera mounted 0.2m forward of base
        t_base_cam = np.array([0.20, 0.0, 0.0], dtype=np.float32)

        t_world_cam = t_world_base + (R_world_base @ t_base_cam.reshape(3, 1)).reshape(3)
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
        # self.get_logger().info(f"sample u={u} v={v} types=({type(u)}, {type(v)})")

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
        r = int(self.patch_radius)

        uc = int(round(u))
        vc = int(round(v))

        x0 = max(0, uc - r)
        x1 = min(w, uc + r + 1)
        y0 = max(0, vc - r)
        y1 = min(h, vc + r + 1)

        patch = img[y0:y1, x0:x1]
        vals = patch[np.isfinite(patch)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals))

    def _detect_color_blob(self, hsv_img, marker, proj_u_rgb, proj_v_rgb):
        cfg = marker['hsv']
        h = hsv_img[:, :, 0]
        s = hsv_img[:, :, 1]
        v = hsv_img[:, :, 2]

        hue_ok = wrap_hue_interval(h, cfg['h_low'], cfg['h_high'])
        mask = hue_ok & (s >= cfg['s_min']) & (v >= cfg['v_min'])
        mask = (mask.astype(np.uint8) * 255)

        k = max(1, int(self.morph_kernel_px))
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        best = None
        best_score = None

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_blob_area_px or area > self.max_blob_area_px:
                continue

            cx, cy = centroids[label]
            dist = float(np.hypot(cx - proj_u_rgb, cy - proj_v_rgb))
            if dist > self.association_max_px:
                continue

            score = dist
            if best is None or score < best_score:
                best = {
                    'u_rgb': float(cx),
                    'v_rgb': float(cy),
                    'area': area,
                    'dist_to_gt_px': dist,
                }
                best_score = score

        return best

    def scale_linear(self, d):
        return self.lin_m * d + self.lin_b

    def scale_quadratic(self, d):
        return self.quad_a * d * d + self.quad_b * d + self.quad_c

    def sync_callback(self, da3_msg, gt_msg, rgb_msg, odom_msg):
        if self.camera_info_msg is None:
            return

        try:
            cv_da3 = self.bridge.imgmsg_to_cv2(da3_msg, 'passthrough').astype(np.float32)
            cv_gt_raw = self.bridge.imgmsg_to_cv2(gt_msg, 'passthrough').astype(np.float32)
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')

            hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2HSV)

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
            roll_deg, pitch_deg, yaw_deg = quat_msg_to_rpy_deg(q)
            ts = da3_msg.header.stamp.sec + da3_msg.header.stamp.nanosec * 1e-9

            vis_rows = []
            abs_errs = []

            for marker in self.markers:
                proj = self._project_world_point(
                    marker['center_world'], R_world_cam, t_world_cam,
                    fx, fy, cx, cy, w, h
                )
                if proj is None:
                    continue

                gt_u = proj['u']
                gt_v = proj['v']
                gt_u_rgb = gt_u / sx
                gt_v_rgb = gt_v / sy

                gt_depth_geom = float(proj['gt_depth_m'])
                gt_depth_img = self._sample_patch_median(cv_gt, gt_u, gt_v)

                det = self._detect_color_blob(hsv, marker, gt_u_rgb, gt_v_rgb)
                if det is None:
                    self.csv_writer.writerow([
                        ts, marker['id'], marker['color'],
                        np.nan, np.nan, gt_u, gt_v, np.nan,
                        roll_deg, pitch_deg, yaw_deg,
                        gt_depth_geom, gt_depth_img, np.nan,
                        np.nan, np.nan,
                        float(proj['p_cam'][0]), float(proj['p_cam'][1]), float(proj['p_cam'][2]),
                        0, 0
                    ])
                    continue

                det_u = det['u_rgb'] * sx
                det_v = det['v_rgb'] * sy

                da3_raw = self._sample_patch_median(cv_da3, det_u, det_v)
                if not np.isfinite(da3_raw):
                    continue

                da3_lin = self.scale_linear(da3_raw)
                da3_quad = self.scale_quadratic(da3_raw)
                if not np.isfinite(da3_raw):
                    continue

                if gt_depth_geom <= self.min_gt_depth_m or gt_depth_geom > self.max_depth_m:
                    continue

                err_raw = abs(da3_raw - gt_depth_geom)
                err_lin = abs(da3_lin - gt_depth_geom)
                err_quad = abs(da3_quad - gt_depth_geom)

                key = marker['id']
                prev_depth = self.prev_depth_by_marker.get(key)
                jitter = abs(da3_raw - prev_depth) if prev_depth is not None else 0.0
                self.prev_depth_by_marker[key] = da3_raw

                pixel_err = float(np.hypot(det_u - gt_u, det_v - gt_v))

                self.csv_writer.writerow([
                    ts, marker['id'], marker['color'],
                    det_u, det_v, gt_u, gt_v, pixel_err,
                    roll_deg, pitch_deg, yaw_deg,
                    gt_depth_geom, gt_depth_img,
                    da3_raw, da3_lin, da3_quad,
                    err_raw, err_lin, err_quad,
                    jitter,
                    float(proj['p_cam'][0]), float(proj['p_cam'][1]), float(proj['p_cam'][2]),
                    det['area'], 1
                ])

                vis_rows.append({
                    'det_u': det_u,
                    'det_v': det_v,
                    'gt_u': gt_u,
                    'gt_v': gt_v,
                    'marker_id': marker['id'],
                    'color': marker['color'],
                    'gt_depth': gt_depth_geom,
                    'gt_depth_img': gt_depth_img,
                    'da3_raw': da3_raw,
                    'da3_quad': da3_quad,
                    'err_raw': err_raw,
                    'err_quad': err_quad,
                    'pixel_err': pixel_err,
                })
                abs_errs.append(abs_errs)

            self.csv_fp.flush()
            self.visible_landmarks_last = vis_rows

            if not self.frame_saved and vis_rows:
                rgb_save = cv_rgb.copy()
                for row in vis_rows:
                    cv2.circle(rgb_save, (int(round(row['det_u'] / sx)), int(round(row['det_v'] / sy))), 6, (0, 255, 0),
                               -1)
                    cv2.circle(rgb_save, (int(round(row['gt_u'] / sx)), int(round(row['gt_v'] / sy))), 6, (0, 0, 255),
                               2)
                    cv2.putText(
                        rgb_save,
                        row['marker_id'],
                        (int(round(row['det_u'] / sx)) + 6, int(round(row['det_v'] / sy)) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )
                cv2.imwrite(self.image_save_path, rgb_save)
                self.get_logger().info(f'Saved reference RGB frame to {self.image_save_path}')
                self.frame_saved = True

            if self.show_viz:
                mae_raw = float(np.mean([r['err_raw'] for r in vis_rows])) if vis_rows else float('nan')
                mae_quad = float(np.mean([r['err_quad'] for r in vis_rows])) if vis_rows else float('nan')
                self.visualize(cv_da3, cv_gt, vis_rows, mae_raw, mae_quad, (roll_deg, pitch_deg, yaw_deg))

        except Exception as e:
            self.get_logger().error(f'Sync error: {e}')

    def visualize(self, da3_img, gt_img, markers, mae_raw, mae_quad, rpy):
        def colorize(img):
            depth_clipped = np.clip(img, 0.1, self.max_depth_m)
            depth_norm = depth_clipped / self.max_depth_m
            depth_8bit = (depth_norm * 255).astype(np.uint8)
            return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)

        vis_da3 = colorize(da3_img)
        vis_gt = colorize(gt_img)

        for row in markers:
            du = int(round(row['det_u']))
            dv = int(round(row['det_v']))
            gu = int(round(row['gt_u']))
            gv = int(round(row['gt_v']))

            cv2.circle(vis_da3, (du, dv), 4, (0, 255, 0), -1)
            cv2.circle(vis_gt, (du, dv), 4, (0, 255, 0), -1)

            cv2.circle(vis_da3, (gu, gv), 4, (0, 0, 255), 1)
            cv2.circle(vis_gt, (gu, gv), 4, (0, 0, 255), 1)

            cv2.putText(vis_da3, row['marker_id'], (du + 4, dv - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(vis_gt, row['marker_id'], (du + 4, dv - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        combined = np.hstack((vis_da3, vis_gt))
        info_str = f'Markers: {len(markers)} | MAE raw: {mae_raw:.3f} | MAE quad: {mae_quad:.3f}'

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
