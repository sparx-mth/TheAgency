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
LUT_FILE_PATH = Path('sparx_agency/tasks/localization/data/saved_lut.npy')
GRID_FILE_PATH = 'sparx_agency/tasks/localization/data/occ_grid_int8.npy'
SIGMA = 5.0  # Increased for realistic sensor noise tolerance to prevent math crashes
MAX_HISTORY = 1000  # Max number of past poses to keep for visualization (if needed)

# ==============================================================================
# Visualization Helper Function (UPDATED WITH TRAJECTORY)
# ==============================================================================
def show_world_map(world_binary: np.ndarray, 
                   odom_history: list, 
                   amcl_history: list, 
                   current_loc: tuple, 
                   orientation: float, 
                   title='AMCL Live Correction'):
    """
    Displays the grid map, the robot's trajectories (Odom vs AMCL),
    and its current orientation.
    """
    plt.clf()  
    plt.imshow(world_binary, cmap='gray_r', origin='lower')
    
    # 1. Plot the Raw Odometry Trajectory (Green dashed line)
    if len(odom_history) > 0:
        # Unzip the list of (x,y) tuples into separate x and y lists
        ox, oy = zip(*odom_history)
        plt.plot(ox, oy, 'g--', label='Raw Odom (Optical Flow)', linewidth=1.5, alpha=0.7)

    # 2. Plot the Corrected AMCL Trajectory (Red solid line)
    if len(amcl_history) > 0:
        ax, ay = zip(*amcl_history)
        plt.plot(ax, ay, 'r-', label='AMCL Corrected Path', linewidth=2.0)
    
    # 3. Plot Current Position and Orientation
    if current_loc is not None and current_loc[0] > 0 and current_loc[1] > 0:
        x, y = current_loc
        plt.plot(x, y, 'rx', markersize=10, markeredgewidth=2, label='Current Pos')
        
        if orientation is not None:
            arrow_length = 8.0 
            dx = math.cos(orientation) * arrow_length
            dy = math.sin(orientation) * arrow_length
            plt.arrow(x, y, dx, dy, head_width=3, head_length=4, fc='blue', ec='blue')
            
    plt.title(title)
    
    # Add a legend to explain the lines
    plt.legend(loc='upper right', fontsize=8)
    
    plt.draw()
    plt.pause(0.001)

# ==============================================================================
# Core AMCL Mathematical Functions
# ==============================================================================
def init_belief(map_shape: tuple[int, int],
                orientations: ndarray,
                robot_loc_pred: ndarray,
                robot_orientation_pred: float,
                loc_uncertainty: tuple[int, int]) -> ndarray:
    """Initialize belief map using a Gaussian distribution around the prediction."""
    map_lat, map_long = map_shape
    num_angles = len(orientations)
    belief = np.zeros((map_lat, map_long, num_angles))

    orientation_diffs = np.abs(orientations - robot_orientation_pred)
    orientation_pred_idx = np.argmin(orientation_diffs)

    sigma_spatial = np.array(loc_uncertainty)
    sigma_angular = 1.0

    for i in range(map_lat):
        for j in range(map_long):
            for k in range(num_angles):
                offset = np.array([i - robot_loc_pred[0], j - robot_loc_pred[1]])
                angular_dist = min(abs(k - orientation_pred_idx),
                                   num_angles - abs(k - orientation_pred_idx))

                spatial_weight = np.linalg.norm(np.exp(-0.5 * (offset / sigma_spatial) ** 2))
                angular_weight = np.exp(-0.5 * (angular_dist / sigma_angular) ** 2)

                belief[i, j, k] = spatial_weight * angular_weight

    # Add small epsilon to prevent division by zero
    belief /= (belief.sum() + 1e-9)
    return belief


def ray_cast_lut_pose(grid, orientations, beam_angles, max_range, step=0.1):
    """Generate a 4D Ray-Casting Look-Up Table (LUT)."""
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
    """Compute Gaussian likelihood over the pre-computed LUT."""
    err = lut - z[None, None, None, :]
    return np.exp(-0.5 * np.sum((err / sigma) ** 2, axis=3))


def measurement_update_pose(bel, lut, z, sigma, occupancy):
    """Update pose belief distribution based on laser/depth measurements."""
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
                   prediction_uncertainty: tuple[int, int]):
    """Execute full AMCL cycle: Predict -> Update -> Extract Max Probability."""
    belief = init_belief(map_shape=world.shape, orientations=orientations, robot_loc_pred=robot_loc_prediction,
                         robot_orientation_pred=robot_orientation_prediction, loc_uncertainty=prediction_uncertainty)

    belief = measurement_update_pose(belief, lut, z_measured_pose, sigma=SIGMA, occupancy=world)

    idx = np.unravel_index(np.argmax(belief), belief.shape)
    robot_loc_estimate = np.array(idx[:2])
    robot_orientation_estimate = orientations[idx[2]]
    return robot_loc_estimate, robot_orientation_estimate


def get_yaw_from_quaternion(q):
    """Convert a ROS geometry_msgs Quaternion message to yaw angle in radians."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ==============================================================================
# ROS 2 AMCL Node
# ==============================================================================
class XtendAMCLNode(Node):
    def __init__(self):
        super().__init__("xtend_amcl_node")
        
        self.bridge = CvBridge()
        self.map_resolution = 0.05  # 5cm per cell
        self.fov_rad = math.radians(75.98) # Based on camera calibration
        
        # Enable interactive mode for matplotlib if SHOW_MAP is true
        if SHOW_MAP:
            plt.ion()
            plt.figure(figsize=(8, 6))

        # ------------------------------------------------------------------
        # 1. Load Map and Format it for AMCL
        # ------------------------------------------------------------------
        self.get_logger().info("Loading Map and Metadata...")
        raw_grid = np.load(GRID_FILE_PATH)
        # AMCL expects 1 for walls/occupied, 0 for free space.
        # In occ_grid_int8.npy: 100 is occupied, 0 is free, -1 is unknown.
        self.world_binary = np.where(raw_grid == 100, 1, 0).astype(np.int8)

        # ------------------------------------------------------------------
        # 2. Map Origin Configuration
        # ------------------------------------------------------------------
        # Calculated perfectly for the tip at physical world (-55.0, -2.0) 
        # mapping to matrix cell (44, 48).
        # map_origin = World_Position - (Cell_Index * Resolution)
        self.map_origin_x = -2.2  
        self.map_origin_y = -2.4   

        # Internal State Variables (Living in real-world meters)
        self.current_pose_gt_meters = np.array([0.0, 0.0])
        self.current_yaw_gt = 0.0
        self.has_odom = False


        # --- NEW: AMCL Statistics ---
        self.amcl_success_count = 0
        self.amcl_fail_count = 0

        # History Lists for Visualization ---
        self.odom_history_cells = []
        self.amcl_history_cells = []

        # ------------------------------------------------------------------
        # 3. Load or Generate LUT
        # ------------------------------------------------------------------
        self.orientations = np.linspace(-math.pi, math.pi, NUM_ANGLES)
        self.beam_angles = np.linspace(-self.fov_rad/2, self.fov_rad/2, NUM_BEAMS)
        max_range_cells = int(SENSOR_MAX_RANGE_METERS / self.map_resolution)
        
        if LUT_FILE_PATH.exists():
            self.get_logger().info(f"Found pre-computed LUT at {LUT_FILE_PATH}. Loading it instantly...")
            self.lut = np.load(LUT_FILE_PATH)
            self.get_logger().info("LUT Loaded successfully from disk! Ready to go.")
        else:
            self.get_logger().info("No saved LUT found. Generating a new Ray-Cast LUT...")
            self.get_logger().warn("⚠️ THIS MIGHT TAKE 10-20 MINUTES. PLEASE DO NOT CLOSE THE PROGRAM. ⚠️")
            self.lut = ray_cast_lut_pose(
                self.world_binary, 
                self.orientations, 
                self.beam_angles, 
                max_range=max_range_cells
            )
            np.save(LUT_FILE_PATH, self.lut)
            self.get_logger().info("LUT saved successfully. Ready to go!")

        # ------------------------------------------------------------------
        # 4. ROS Subscriptions Setup
        # ------------------------------------------------------------------
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        
        self.odom_sub = self.create_subscription(
            PoseStamped, "/flow_depth/pose_est", self.odom_callback, image_qos
        )
        self.depth_sub = self.create_subscription(
            Image, "/xtend/depth_m", self.depth_callback, image_qos
        )


    def odom_callback(self, msg: PoseStamped):
        """Callback to store the current raw world position from the tracking node."""
        self.current_pose_gt_meters[0] = -msg.pose.position.y
        self.current_pose_gt_meters[1] = msg.pose.position.x

        raw_yaw = get_yaw_from_quaternion(msg.pose.orientation)
        self.current_yaw_gt = raw_yaw + (math.pi / 2.0)
        self.has_odom = True


    def depth_callback(self, msg: Image):
        """Main AMCL loop triggered by new depth frames."""
        if not self.has_odom:
            return 
            
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Extract 1D array from the middle row of the depth image
        height = depth_img.shape[0]
        middle_row = depth_img[height // 2, :]
        indices = np.linspace(0, len(middle_row) - 1, NUM_BEAMS, dtype=int)
        z_measured_meters = middle_row[indices]
        
        # Aggressive cleaning of depth data to replace invalid/noisy 0.0, NaN, Inf values
        z_measured_meters = np.where(z_measured_meters <= 0.0, SENSOR_MAX_RANGE_METERS, z_measured_meters)
        z_measured_meters = np.nan_to_num(z_measured_meters, nan=SENSOR_MAX_RANGE_METERS, posinf=SENSOR_MAX_RANGE_METERS)
        z_measured_cells = z_measured_meters / self.map_resolution

        self.get_logger().info(f"[DEBUG] Sensor Depths (m) -> Min: {np.min(z_measured_meters):.2f}, Max: {np.max(z_measured_meters):.2f}, Mean: {np.mean(z_measured_meters):.2f}")

        # --- Convert Prediction from World Meters to Map Cells ---
        map_x_m = self.current_pose_gt_meters[0] - self.map_origin_x
        map_y_m = self.current_pose_gt_meters[1] - self.map_origin_y
        
        pred_x_cells = int(map_x_m / self.map_resolution)
        pred_y_cells = int(map_y_m / self.map_resolution)
        prediction_cells = np.array([pred_x_cells, pred_y_cells])

        self.get_logger().info(f"[DEBUG] Feeding to AMCL -> Cells: X={pred_x_cells}, Y={pred_y_cells}, Yaw={math.degrees(self.current_yaw_gt):.1f}")


        # Save Raw Odom to History ---
        self.odom_history_cells.append((pred_x_cells, pred_y_cells))
        if len(self.odom_history_cells) > MAX_HISTORY:
            self.odom_history_cells.pop(0) # Remove oldest point

        # Execute AMCL ---
        robot_loc_estimate_cells, robot_orientation_estimate = amcl_estimator(
            self.lut, self.orientations, prediction_cells, self.current_yaw_gt, 
            self.world_binary, z_measured_cells, prediction_uncertainty=(8, 8) 
        )

        # Safety Check: If correlation failed
        if robot_loc_estimate_cells[0] == 0 and robot_loc_estimate_cells[1] == 0:
            self.amcl_fail_count += 1
            self.get_logger().warn(f"❌ [AMCL FAILED] Correlation lost! Total Fails: {self.amcl_fail_count}")
            return

        # Save AMCL Corrected Pose to History ---
        self.amcl_history_cells.append((robot_loc_estimate_cells[0], robot_loc_estimate_cells[1]))
        if len(self.amcl_history_cells) > MAX_HISTORY:
            self.amcl_history_cells.pop(0)

        # Update State ---
        est_world_x_m = (robot_loc_estimate_cells[0] * self.map_resolution) + self.map_origin_x
        est_world_y_m = (robot_loc_estimate_cells[1] * self.map_resolution) + self.map_origin_y
 
         # --- NEW: Print Success and Stats ---
        self.amcl_success_count += 1
        self.get_logger().info(f"✅ [AMCL SUCCESS] Total Successes: {self.amcl_success_count}")
        self.get_logger().info(f"   -> Odom Pred: x={self.current_pose_gt_meters[0]:.2f}, y={self.current_pose_gt_meters[1]:.2f}")
        self.get_logger().info(f"   -> AMCL Est:  x={est_world_x_m:.2f}, y={est_world_y_m:.2f}")

 
        self.current_pose_gt_meters = np.array([est_world_x_m, est_world_y_m])
        self.current_yaw_gt = robot_orientation_estimate

        # --- Visualization ---
        if SHOW_MAP:
            show_world_map(
                self.world_binary, 
                odom_history=self.odom_history_cells,
                amcl_history=self.amcl_history_cells,
                current_loc=(robot_loc_estimate_cells[0], robot_loc_estimate_cells[1]), 
                orientation=robot_orientation_estimate, 
                title='AMCL Live Correction & Trajectory'
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