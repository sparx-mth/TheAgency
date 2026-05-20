#!/usr/bin/env python3
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray, dtype, float64

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Import Matplotlib for the live visualization
import matplotlib.pyplot as plt

# ==============================================================================
# Global Configuration & Hyperparameters
# ==============================================================================
SENSOR_MAX_RANGE_METERS = 10.0
NUM_ANGLES = 32
NUM_BEAMS = 64
SHOW_MAP = True
LUT_FILE_PATH = Path('sparx_agency/tasks/localization/data/saved_lut_small_map.npy')
GRID_FILE_PATH = 'sparx_agency/tasks/localization/data/cropped_occ_grid_int8.npy'
SIGMA = 8.0  

# --- Motion Model Hyperparameters ---
ROBOT_SPEED_MPS = 3.0 / 7.5  # ~0.465 m/s
SENSOR_FREQ_HZ = 10.0        
DT = 1.0 / SENSOR_FREQ_HZ    # 0.1 seconds per frame

# ==============================================================================
# Visualization Helper Function
# ==============================================================================
def show_world_map(
    world_binary,
    loc_pred,
    loc_amcl,
    orientation,
    odom_trajectory=None,
    amcl_trajectory=None,
    title='AMCL Tracking Debug'
):
    plt.clf()
    plt.imshow(world_binary, cmap='gray_r', origin='lower')

    # -----------------------------
    # Odometry trajectory
    # -----------------------------
    if odom_trajectory is not None and len(odom_trajectory) > 1:
        odom_traj = np.array(odom_trajectory)

        plt.plot(
            odom_traj[:, 1],
            odom_traj[:, 0],
            'b-',
            linewidth=2,
            label='Odometry Trajectory'
        )

        plt.plot(
            odom_traj[-1, 1],
            odom_traj[-1, 0],
            'bo',
            markersize=6,
            label='Current Odom'
        )

    # -----------------------------
    # AMCL corrected trajectory
    # -----------------------------
    if amcl_trajectory is not None and len(amcl_trajectory) > 1:
        amcl_traj = np.array(amcl_trajectory)

        plt.plot(
            amcl_traj[:, 1],
            amcl_traj[:, 0],
            'r-',
            linewidth=2,
            label='AMCL Trajectory'
        )

        plt.plot(
            amcl_traj[-1, 1],
            amcl_traj[-1, 0],
            'rx',
            markersize=8,
            markeredgewidth=2,
            label='Current AMCL'
        )

    # Current prediction point
    plt.plot(
        loc_pred[1],
        loc_pred[0],
        'c.',
        markersize=8,
        label='Kinematic Pred'
    )

    # Current AMCL corrected point
    plt.plot(
        loc_amcl[1],
        loc_amcl[0],
        'mx',
        markersize=10,
        markeredgewidth=2,
        label='AMCL Corrected Now'
    )

    plt.legend()
    plt.title(title)
    plt.draw()
    plt.pause(0.001)

# ==============================================================================
# Core AMCL Mathematical Functions
# ==============================================================================
def init_belief(map_shape: tuple[int, int],
                orientations: ndarray,
                robot_loc_pred: ndarray,
                robot_orientation_pred: float,
                loc_uncertainty: tuple[float, float]) -> ndarray:
    map_lat, map_long = map_shape
    num_angles = len(orientations)
    belief = np.zeros((map_lat, map_long, num_angles))

    orientation_diffs = np.abs(orientations - robot_orientation_pred)
    orientation_pred_idx = np.argmin(orientation_diffs)

    sigma_spatial = np.array(loc_uncertainty)
    sigma_spatial = np.maximum(sigma_spatial, 1e-3) 
    sigma_angular = 1.0

    for i in range(map_lat):
        for j in range(map_long):
            # Optimizing: only calculate inside a 3-sigma bounding box
            if abs(i - robot_loc_pred[0]) > 3 * sigma_spatial[0] or abs(j - robot_loc_pred[1]) > 3 * sigma_spatial[1]:
                continue
                
            for k in range(num_angles):
                offset = np.array([i - robot_loc_pred[0], j - robot_loc_pred[1]])
                angular_dist = min(abs(k - orientation_pred_idx),
                                   num_angles - abs(k - orientation_pred_idx))

                spatial_weight = math.exp(-0.5 * (offset[0] / sigma_spatial[0]) ** 2) * \
                                 math.exp(-0.5 * (offset[1] / sigma_spatial[1]) ** 2)
                                 
                angular_weight = math.exp(-0.5 * (angular_dist / sigma_angular) ** 2)

                belief[i, j, k] = spatial_weight * angular_weight

    belief /= (belief.sum() + 1e-9)
    return belief


def ray_cast_lut_pose(grid, orientations, beam_angles, max_range, step=0.1):
    m, n = grid.shape
    lut = np.ones((m, n, len(orientations), len(beam_angles)), dtype=np.float32) * np.inf

    for i in range(m):
        for j in range(n):
            if grid[i, j] == 1:
                continue

            for k, theta in enumerate(orientations):
                for b, rel in enumerate(beam_angles):
                    angle = theta + rel
                    dist = 0.0

                    while dist < max_range:
                        x = int(round(i + dist * math.cos(angle)))
                        y = int(round(j + dist * math.sin(angle)))

                        if x < 0 or y < 0 or x >= m or y >= n:
                            dist += max_range
                            break
                        if grid[x, y] == 1:
                            break

                        dist += step

                    lut[i, j, k, b] = dist
    return lut


def range_likelihood_lut(z, lut, sigma):
    err = lut - z[None, None, None, :]
    return np.exp(-0.5 * np.mean((err / sigma) ** 2, axis=3))


def measurement_update_pose(bel, lut, z, sigma, occupancy):
    likelihood = range_likelihood_lut(z, lut, sigma)
    likelihood[occupancy == 1] = 0.0

    bel_new = bel * likelihood
    s = bel_new.sum()
    if s > 0:
        bel_new /= s
    return bel_new


def amcl_estimator(lut: ndarray,
                   orientations: ndarray,
                   robot_loc_prediction: ndarray,
                   robot_orientation_prediction: float,
                   world: ndarray,
                   z_measured_pose: ndarray,
                   prediction_uncertainty: tuple[float, float]):
    
    belief = init_belief(map_shape=world.shape, orientations=orientations, robot_loc_pred=robot_loc_prediction,
                         robot_orientation_pred=robot_orientation_prediction, loc_uncertainty=prediction_uncertainty)

    belief = measurement_update_pose(belief, lut, z_measured_pose, sigma=SIGMA, occupancy=world)

    idx = np.unravel_index(np.argmax(belief), belief.shape)
    robot_loc_estimate = np.array(idx[:2])
    robot_orientation_estimate = orientations[idx[2]]
    return robot_loc_estimate, robot_orientation_estimate

# ==============================================================================
# ROS 2 AMCL Node
# ==============================================================================
class XtendAMCLNode(Node):
    def __init__(self):
        super().__init__("xtend_amcl_node")
        
        self.bridge = CvBridge()
        self.map_resolution = 0.05
        self.fov_rad = math.radians(75.98)
        
        if SHOW_MAP:
            plt.ion()
            plt.figure(figsize=(8, 6))

        # 1. Load Map
        self.get_logger().info("Loading Map and Metadata...")
        raw_grid = np.load(GRID_FILE_PATH)
        self.world_binary = np.where(raw_grid == 100, 1, 0).astype(np.int8)

        # 2. Map Origin Configuration
        # -2 * 0.05 = -0.1 and -21 *0.05 = -1.05
        self.map_origin_x = -1.0
        self.map_origin_y = -1.0  

        # --- INITIAL POSE SETUP (CONSTANT Y AND YAW) ---
        self.start_x_meters = 0.0
        self.constant_y_meters = 0.0
        self.constant_yaw_rad = math.pi / 2.0

        self.odom_pose_meters = np.array([
            self.start_x_meters,
            self.constant_y_meters
        ], dtype=float)

        self.amcl_pose_meters = np.array([
            self.start_x_meters,
            self.constant_y_meters
        ], dtype=float)

        self.odom_trajectory_cells = []
        self.amcl_trajectory_cells = []

        self.current_yaw_gt = self.constant_yaw_rad

        # 3. Load or Generate LUT
        self.orientations = np.linspace(-math.pi, math.pi, NUM_ANGLES)
        self.beam_angles = np.linspace(-self.fov_rad/2, self.fov_rad/2, NUM_BEAMS)
        max_range_cells = int(SENSOR_MAX_RANGE_METERS / self.map_resolution)
        
        if LUT_FILE_PATH.exists():
            self.get_logger().info(f"Found pre-computed LUT at {LUT_FILE_PATH}. Loading...")
            self.lut = np.load(LUT_FILE_PATH)
            self.get_logger().info("LUT Loaded successfully!")
        else:
            self.get_logger().info("Generating LUT...")
            self.lut = ray_cast_lut_pose(self.world_binary, self.orientations, self.beam_angles, max_range=max_range_cells)
            np.save(LUT_FILE_PATH, self.lut)
            self.get_logger().info("LUT saved successfully.")

        # 4. ROS Subscriptions Setup
        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.depth_sub = self.create_subscription(Image, "/xtend/depth_m", self.depth_callback, image_qos)


    def depth_callback(self, msg: Image):
        # ------------------------------------------------------------------
        # KINEMATIC PREDICTION STEP (Constant Velocity on X-Axis)
        # ------------------------------------------------------------------
        distance_moved_m = ROBOT_SPEED_MPS * DT

        # ============================================================
        # Pure odometry prediction - never corrected by AMCL
        # ============================================================
        self.odom_pose_meters[0] += distance_moved_m
        self.odom_pose_meters[1] = self.constant_y_meters

        odom_x_m = self.odom_pose_meters[0]

        odom_map_x_m = self.odom_pose_meters[0] - self.map_origin_x
        odom_map_y_m = self.odom_pose_meters[1] - self.map_origin_y

        odom_x_cells = int(odom_map_x_m / self.map_resolution)
        odom_y_cells = int(odom_map_y_m / self.map_resolution)

        odom_cells = np.array([odom_x_cells, odom_y_cells])

        self.odom_trajectory_cells.append(odom_cells.copy())

        # ============================================================
        # AMCL prediction - starts from previous AMCL corrected pose
        # ============================================================
        amcl_pred_pose_meters = self.amcl_pose_meters.copy()
        amcl_pred_pose_meters[0] += distance_moved_m
        amcl_pred_pose_meters[1] = self.constant_y_meters

        amcl_pred_x_m = amcl_pred_pose_meters[0]

        map_x_m = amcl_pred_pose_meters[0] - self.map_origin_x
        map_y_m = amcl_pred_pose_meters[1] - self.map_origin_y

        pred_x_cells = int(map_x_m / self.map_resolution)
        pred_y_cells = int(map_y_m / self.map_resolution)

        prediction_cells = np.array([pred_x_cells, pred_y_cells])
        
        #predicted_x_m = self.current_pose_gt_meters[0] + distance_moved_m
        
        #self.current_pose_gt_meters[0] = predicted_x_m
        
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        height = depth_img.shape[0]
        middle_row = depth_img[height // 2, :]
        indices = np.linspace(0, len(middle_row) - 1, NUM_BEAMS, dtype=int)
        z_measured_meters = middle_row[indices]
        
        z_measured_meters = np.where(z_measured_meters <= 0.0, SENSOR_MAX_RANGE_METERS, z_measured_meters)
        z_measured_meters = np.nan_to_num(z_measured_meters, nan=SENSOR_MAX_RANGE_METERS, posinf=SENSOR_MAX_RANGE_METERS)
        z_measured_cells = z_measured_meters / self.map_resolution

    
        sigma_x_m = 0.4   # X almost fixed
        sigma_y_m = 0.05   # Y has more uncertainty

        sigma_x_cells = sigma_x_m / self.map_resolution
        sigma_y_cells = sigma_y_m / self.map_resolution

        self.get_logger().info(
            f"[AMCL INPUT] "
            f"odom_cells={odom_cells}, "
            f"amcl_pred_cells={prediction_cells}, "
            f"sigma_cells=({sigma_x_cells:.2f}, {sigma_y_cells:.2f}), "
            f"search_window_x=±{3*sigma_x_cells:.1f} cells, "
            f"search_window_y=±{3*sigma_y_cells:.1f} cells"
        )
                
        # --- Execute AMCL with 1D Constraint ---
        robot_loc_estimate_cells, _ = amcl_estimator(
            self.lut, 
            self.orientations, 
            prediction_cells,
            self.constant_yaw_rad, 
            self.world_binary, 
            z_measured_cells,
            prediction_uncertainty=(sigma_x_cells, sigma_y_cells)
        )

        dx_cells = robot_loc_estimate_cells[0] - prediction_cells[0]
        dy_cells = robot_loc_estimate_cells[1] - prediction_cells[1]

        self.get_logger().info(
            f"[AMCL OUTPUT] "
            f"corrected={robot_loc_estimate_cells}, "
            f"delta_cells=({dx_cells}, {dy_cells}), "
            f"delta_m=({dx_cells*self.map_resolution:.2f}, {dy_cells*self.map_resolution:.2f})"
        )

        if robot_loc_estimate_cells[0] == 0 and robot_loc_estimate_cells[1] == 0:
            self.get_logger().warn("AMCL failed. Keeping AMCL prediction.")

            robot_loc_estimate_cells = prediction_cells.copy()
            self.amcl_pose_meters = amcl_pred_pose_meters.copy()

        else:
            est_map_x_m = float(robot_loc_estimate_cells[0] * self.map_resolution)
            est_world_x_m = est_map_x_m + self.map_origin_x

            est_world_y_m = self.constant_y_meters

            self.amcl_pose_meters = np.array([
                est_world_x_m,
                est_world_y_m
            ], dtype=float)

        self.amcl_trajectory_cells.append(robot_loc_estimate_cells.copy())

    

        # --- Visualization ---
        if SHOW_MAP:
            show_world_map(
                self.world_binary,
                loc_pred=odom_cells,
                loc_amcl=robot_loc_estimate_cells,
                orientation=self.constant_yaw_rad,
                odom_trajectory=self.odom_trajectory_cells,
                amcl_trajectory=self.amcl_trajectory_cells,
                title=f'Odom={odom_cells}, AMCL={robot_loc_estimate_cells}'
            )

def main():
    rclpy.init()
    node = XtendAMCLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down AMCL Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()