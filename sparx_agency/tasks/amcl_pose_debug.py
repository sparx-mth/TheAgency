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
SIGMA = 5.0  

# --- Motion Model Hyperparameters ---
ROBOT_SPEED_MPS = 4.0 / 6.5  # ~0.615 m/s
SENSOR_FREQ_HZ = 10.0        
DT = 1.0 / SENSOR_FREQ_HZ    # 0.1 seconds per frame

# ==============================================================================
# Visualization Helper Function
# ==============================================================================
def show_world_map(world_binary: np.ndarray, location: tuple, orientation: float, title='AMCL 1D Tracking w/ Velocity'):
    plt.clf()
    
    plt.imshow(world_binary, cmap='gray_r', origin='lower')
    
    if location is not None and location[0] > 0 and location[1] > 0:
        x, y = location
        plt.plot(x, y, 'rx', markersize=10, markeredgewidth=2)
        
        if orientation is not None:
            arrow_length = 8.0 
            dx = math.cos(orientation) * arrow_length
            dy = math.sin(orientation) * arrow_length
            plt.arrow(x, y, dx, dy, head_width=3, head_length=4, fc='blue', ec='blue')
            
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
    return np.exp(-0.5 * np.sum((err / sigma) ** 2, axis=3))


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
        self.map_origin_x = -57.2  
        self.map_origin_y = -4.4   

        # --- INITIAL POSE SETUP (CONSTANT Y AND YAW) ---
        self.start_x_meters = -55.0  
        self.constant_y_meters = -2.0 
        self.constant_yaw_rad = 0.0   
        
        self.current_pose_gt_meters = np.array([self.start_x_meters, self.constant_y_meters])
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
        distance_moved_m = ROBOT_SPEED_MPS * DT  # ~0.0615 meters per frame
        
        predicted_x_m = self.current_pose_gt_meters[0] + distance_moved_m
        
        self.current_pose_gt_meters[0] = predicted_x_m
        
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

        # --- Convert Prediction to Map Cells ---
        map_x_m = self.current_pose_gt_meters[0] - self.map_origin_x
        map_y_m = self.current_pose_gt_meters[1] - self.map_origin_y
        
        pred_x_cells = int(map_x_m / self.map_resolution)
        pred_y_cells = int(map_y_m / self.map_resolution)
        prediction_cells = np.array([pred_x_cells, pred_y_cells])
        
        # --- Execute AMCL with 1D Constraint ---
        robot_loc_estimate_cells, _ = amcl_estimator(
            self.lut, 
            self.orientations, 
            prediction_cells,
            self.constant_yaw_rad, 
            self.world_binary, 
            z_measured_cells,
            prediction_uncertainty=(8.0, 0.1) 
        )

        if robot_loc_estimate_cells[0] == 0 and robot_loc_estimate_cells[1] == 0:
            self.get_logger().error("AMCL failed to converge. Trusting prediction only.")
            return

        # --- Convert Correction to World Meters ---
        est_map_x_m = float(robot_loc_estimate_cells[0] * self.map_resolution)
        est_world_x_m = est_map_x_m + self.map_origin_x

        # FORCE Y AND YAW TO REMAIN CONSTANT
        est_world_y_m = self.constant_y_meters
        robot_orientation_estimate = self.constant_yaw_rad

        self.get_logger().info(f"[AMCL] Est X: {est_world_x_m:.2f}m (Predicted: {predicted_x_m:.2f}m)")
        
        # Update state for the next frame with the AMCL corrected pose
        self.current_pose_gt_meters = np.array([est_world_x_m, est_world_y_m])

        # --- Visualization ---
        if SHOW_MAP:
            show_world_map(
                self.world_binary, 
                location=(robot_loc_estimate_cells[0], robot_loc_estimate_cells[1]), 
                orientation=robot_orientation_estimate, 
                title='AMCL 1D Tracking w/ Velocity'
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