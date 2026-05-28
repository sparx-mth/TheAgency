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
#NUM_ANGLES = 32
#NUM_BEAMS = 64
NUM_ANGLES = 32
NUM_BEAMS = 64
SHOW_MAP = True
#LUT_FILE_PATH = Path('sparx_agency/tasks/localization/data/saved_lut.npy')
#GRID_FILE_PATH = 'sparx_agency/tasks/localization/data/occ_grid_int8.npy'
LUT_FILE_PATH = Path('sparx_agency/tasks/localization/data/saved_lut_cropped.npy')
GRID_FILE_PATH = 'sparx_agency/tasks/localization/data/cropped_occ_grid_int8.npy'
#LUT_FILE_PATH = Path('sparx_agency/tasks/localization/data/saved_lut_cropped_res_0_1.npy')
#GRID_FILE_PATH = 'sparx_agency/tasks/localization/data/cropped_occ_grid_int8_res_0_1.npy'
SIGMA = 2.0  # Increased for realistic sensor noise tolerance to prevent math crashes
MAX_HISTORY = 1000  # Max number of past poses to keep for visualization (if needed)

# ==============================================================================
# Visualization Helper Function (UPDATED WITH TRAJECTORY)
# ==============================================================================
def show_world_map(world_binary: np.ndarray, 
                   odom_history: list, 
                   amcl_history: list, 
                   current_loc: tuple, 
                   orientation: float,
                   success_count: int = 0,
                   fail_count: int = 0,
                   title='AMCL Live Correction'):
    """
    Displays the grid map, the robot's trajectories (Odom vs AMCL),
    and its current orientation.
    """
    plt.clf()  
    plt.imshow(world_binary.T, cmap='gray_r', origin='lower')
    
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

    if current_loc is not None:
        debug_text = (
            f"Pos (cells): x={current_loc[0]:.1f}, y={current_loc[1]:.1f}\n"
            f"Yaw: {math.degrees(orientation):.1f}°\n"
            f"Odom pts: {len(odom_history)}\n"
            f"AMCL pts: {len(amcl_history)}\n"
            f"✅ Success: {success_count}  ❌ Fail: {fail_count}"
        )
    else:
        debug_text = f"AMCL: FAILED / Waiting...\n✅ {success_count}  ❌ {fail_count}"

    plt.gca().text(
        0.02, 0.02, debug_text,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    plt.draw()
    plt.pause(0.01)


# ==============================================================================
# Core AMCL Mathematical Functions
# ==============================================================================

def get_local_window_bounds(pred_loc, map_shape, window_size_cells=50):

    half_w = window_size_cells // 2
    
    x_min = max(0, pred_loc[0] - half_w)
    x_max = min(map_shape[0], pred_loc[0] + half_w)
    
    y_min = max(0, pred_loc[1] - half_w)
    y_max = min(map_shape[1], pred_loc[1] + half_w)
    
    return x_min, x_max, y_min, y_max


def init_belief_vectorized(local_map_shape, orientations, local_robot_pred, robot_orientation_pred, loc_uncertainty):
    # Vectorized initialization of the belief distribution over the local window.
    shape_x, shape_y = local_map_shape 
    num_angles = len(orientations)
    
    # Create 3D grids for x, y, and theta indices
    x_idx = np.arange(shape_x)
    y_idx = np.arange(shape_y)
    theta_idx = np.arange(num_angles)
    
    X, Y, Theta = np.meshgrid(x_idx, y_idx, theta_idx, indexing='ij')
    
    dx = X - local_robot_pred[0]
    dy = Y - local_robot_pred[1]
    sigma_x, sigma_y = loc_uncertainty
    
    spatial_weight = np.exp(-0.5 * ((dx / sigma_x)**2 + (dy / sigma_y)**2))
    
    angles_rad = orientations[Theta]
    
    d_theta = np.abs(angles_rad - robot_orientation_pred)
    d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)
    
    sigma_angular_rad = math.radians(15.0) 
    angular_weight = np.exp(-0.5 * (d_theta / sigma_angular_rad)**2)
    
    belief = spatial_weight * angular_weight
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
    return np.exp(-0.5 * np.mean((err / sigma) ** 2, axis=3))


def measurement_update_pose(bel, lut, z, sigma, occupancy):
    """Update pose belief distribution based on laser/depth measurements."""
    likelihood = range_likelihood_lut(z, lut, sigma)
    likelihood[occupancy == 1] = 0.0

    bel_new = bel * likelihood
    s = bel_new.sum()
    if s > 0:
        bel_new /= s
    return bel_new


def amcl_estimator_optimized(lut: ndarray,
                             orientations: ndarray,
                             robot_loc_prediction: ndarray,
                             robot_orientation_prediction: float,
                             world: ndarray,
                             z_measured_pose: ndarray,
                             prediction_uncertainty: tuple[int, int],
                             window_size: int = 100):

    x_min, x_max, y_min, y_max = get_local_window_bounds(robot_loc_prediction, world.shape, window_size)
    
    local_world = world[x_min:x_max, y_min:y_max]
    local_lut = lut[x_min:x_max, y_min:y_max, :, :]
    
    local_pred_x = robot_loc_prediction[0] - x_min
    local_pred_y = robot_loc_prediction[1] - y_min
    local_robot_prediction = np.array([local_pred_x, local_pred_y])
    
    local_belief = init_belief_vectorized(
        local_map_shape=local_world.shape, 
        orientations=orientations, 
        local_robot_pred=local_robot_prediction,
        robot_orientation_pred=robot_orientation_prediction, 
        loc_uncertainty=prediction_uncertainty
    )

    local_belief = measurement_update_pose(local_belief, local_lut, z_measured_pose, sigma=SIGMA, occupancy=local_world)

    idx = np.unravel_index(np.argmax(local_belief), local_belief.shape)
    best_local_x, best_local_y = idx[0], idx[1]
    best_orientation = orientations[idx[2]]
    
    best_global_x = best_local_x + x_min
    best_global_y = best_local_y + y_min
    
    return np.array([best_global_x, best_global_y]), best_orientation


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
        #self.map_resolution = 0.05  # 5cm per cell
        self.map_resolution = 0.1  # 10cm per cell

        self.fov_rad = math.radians(75.98) # Based on camera calibration
        
        # Enable interactive mode for matplotlib if SHOW_MAP is true
        if SHOW_MAP:
            plt.ion()
            plt.figure(figsize=(8, 6))
            plt.show(block=False)


        # ------------------------------------------------------------------
        # 1. Load Map and Format it for AMCL
        # ------------------------------------------------------------------
        self.get_logger().info("Loading Map and Metadata...")
        raw_grid = np.load(GRID_FILE_PATH)
        # AMCL expects 1 for walls/occupied, 0 for free space.
        # In occ_grid_int8.npy: 100 is occupied, 0 is free, -1 is unknown.
        self.world_binary = np.where(raw_grid == 100, 1, 0).astype(np.int8)

        self.world_binary = self.world_binary.T

        # ------------------------------------------------------------------
        # 2. Map Origin Configuration
        # ------------------------------------------------------------------
        # mapping to matrix cell (44, 48). 44*0.05 and 48*0.05 = -2.2 , -2.4 
        #self.map_origin_x = -2.2   
        #self.map_origin_y = -2.4   
        # for cropped map (2,21) == -0.1, -1.05 and flip x and y 
        # map_origin = World_Position - (Cell_Index * Resolution)
        self.map_origin_x = -1.050
        self.map_origin_y = -0.100  

        # Internal State Variables (Living in real-world meters)

        self.odom_pose_meters = np.array([0.0, 0.0])  
        self.odom_yaw = 0.0
        self.amcl_pose_meters = np.array([0.0, 0.0])  
        self.amcl_yaw = 0.0
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

            self.get_logger().info(f"world_binary shape: {self.world_binary.shape}")
            self.get_logger().info(f"loaded LUT shape before fix: {self.lut.shape}")

            if self.lut.shape[:2] == self.world_binary.shape:
                self.get_logger().info("LUT shape already matches world_binary. No transpose needed.")

            elif self.lut.transpose((1, 0, 2, 3)).shape[:2] == self.world_binary.shape:
                self.get_logger().warn("LUT shape is reversed. Applying transpose.")
                self.lut = self.lut.transpose((1, 0, 2, 3))

            else:
                raise ValueError(
                    f"LUT/map shape mismatch: LUT spatial={self.lut.shape[:2]}, "
                    f"world={self.world_binary.shape}. Delete the LUT and regenerate it."
                )

            self.get_logger().info(f"LUT final shape: {self.lut.shape}")
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
        self.odom_pose_meters[0] = -msg.pose.position.y
        self.odom_pose_meters[1] = msg.pose.position.x


        map_w = self.world_binary.shape[0] * self.map_resolution
        map_h = self.world_binary.shape[1] * self.map_resolution
        self.odom_pose_meters[0] = np.clip(self.odom_pose_meters[0], self.map_origin_x, self.map_origin_x + map_w)
        self.odom_pose_meters[1] = np.clip(self.odom_pose_meters[1], self.map_origin_y, self.map_origin_y + map_h)

        raw_yaw = get_yaw_from_quaternion(msg.pose.orientation)
        self.odom_yaw = raw_yaw + (math.pi / 2.0)
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
        map_x_m = self.odom_pose_meters[0] - self.map_origin_x
        map_y_m = self.odom_pose_meters[1] - self.map_origin_y
        
        pred_x_cells = int(map_x_m / self.map_resolution)
        pred_y_cells = int(map_y_m / self.map_resolution)

        prediction_cells = np.array([pred_x_cells, pred_y_cells])

        self.get_logger().info(f"[DEBUG] Feeding to AMCL -> Cells: X={pred_x_cells}, Y={pred_y_cells}, Yaw={math.degrees(self.odom_yaw):.1f}")


        # Save Raw Odom to History ---
        self.odom_history_cells.append((pred_x_cells, pred_y_cells))
        if len(self.odom_history_cells) > MAX_HISTORY:
            self.odom_history_cells.pop(0) # Remove oldest point

        # Execute AMCL ---
        robot_loc_estimate_cells, robot_orientation_estimate = amcl_estimator_optimized(
            self.lut, self.orientations, prediction_cells, self.odom_yaw, 
            self.world_binary, z_measured_cells, prediction_uncertainty=(3, 3) 
        )

        # Safety Check
        x_est, y_est = robot_loc_estimate_cells[0], robot_loc_estimate_cells[1]
        amcl_failed = (
            x_est <= 0 or y_est <= 0 or
            x_est >= self.world_binary.shape[0] or y_est >= self.world_binary.shape[1] or
            self.world_binary[x_est, y_est] == 1
        )

        if not amcl_failed:
            # Save AMCL to history only on success
            self.amcl_history_cells.append((x_est, y_est))
            if len(self.amcl_history_cells) > MAX_HISTORY:
                self.amcl_history_cells.pop(0)

            est_world_x_m = (x_est * self.map_resolution) + self.map_origin_x
            est_world_y_m = (y_est * self.map_resolution) + self.map_origin_y

            self.amcl_success_count += 1
            self.get_logger().info(f"✅ [AMCL SUCCESS] Total Successes: {self.amcl_success_count}")
            self.get_logger().info(f"   -> Odom Pred: x={self.odom_pose_meters[0]:.2f}, y={self.odom_pose_meters[1]:.2f}")
            self.get_logger().info(f"   -> AMCL Est:  x={est_world_x_m:.2f}, y={est_world_y_m:.2f}")

            self.amcl_pose_meters = np.array([est_world_x_m, est_world_y_m])
            self.amcl_yaw = robot_orientation_estimate
        else:
            self.amcl_fail_count += 1
            self.get_logger().warn(f"❌ [AMCL FAILED] Estimate invalid! Total Fails: {self.amcl_fail_count}")

        if SHOW_MAP:
            current_loc = (x_est, y_est) if not amcl_failed else None
            orientation = robot_orientation_estimate if not amcl_failed else self.odom_yaw
            show_world_map(
                self.world_binary,
                odom_history=self.odom_history_cells,
                amcl_history=self.amcl_history_cells,
                current_loc=current_loc,
                orientation=orientation,
                success_count=self.amcl_success_count,
                fail_count=self.amcl_fail_count,
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