import argparse
import logging
import math
import time
import json
import yaml
from pathlib import Path
from typing import Any
import numpy as np
from numpy import dtype, ndarray, float64

# ROS 2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

# Custom project imports
from sparx_agency.tasks.localization.amcl import ray_cast_lut_pose, amcl_estimator
from sparx_agency.tasks.sim.grid import show_world_map

# Global configurations
SENSOR_MAX_RANGE_METERS = 10.0
NUM_ANGLES = 32
NUM_BEAMS = 64
SHOW_MAP = True

def get_yaw_from_quaternion(q) -> float:
    """
    Converts a quaternion into a Euler Yaw angle (rotation around the Z-axis).
    Required for 2D map localization.
    """
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def make_orientations(num_angles: int) -> ndarray[tuple[Any, ...], dtype[float64]]:
    """
    Generates an array of evenly spaced orientation angles from -pi to pi.
    """
    if num_angles <= 0:
        raise ValueError("Number of angles must be positive.")
    return np.linspace(-math.pi, math.pi, num_angles, endpoint=False)

def make_fov_angles(fov_rad: float, num_beams: int) -> ndarray[tuple[Any, ...], dtype[float64]]:
    """
    Generates an array of evenly spaced angles representing the camera's Field of View.
    """
    if num_beams <= 0:
        raise ValueError("Number of beams must be positive.")
    return np.linspace(-fov_rad / 2, fov_rad / 2, num_beams)


class XtendAMCLNode(Node):
    def __init__(self, map_npy_path: str, map_json_path: str, camera_yaml_path: str):
        super().__init__('xtend_amcl_node')
        self.bridge = CvBridge()
        
        # 1. Load Camera Configuration & Calculate FOV
        self.get_logger().info(f"Loading camera config from {camera_yaml_path}")
        with open(camera_yaml_path, 'r') as f:
            cam_config = yaml.safe_load(f)
            
        # Extract fx (focal length in x) and image width to compute horizontal FOV
        cam_matrix = cam_config['camera_matrix']['data']
        fx = cam_matrix[0]
        image_width = cam_config['image_width']
        self.fov_rad = 2 * math.atan(image_width / (2 * fx))
        self.get_logger().info(f"Calculated Horizontal FOV: {math.degrees(self.fov_rad):.2f} degrees")

        # 2. Load Real Office Map Data
        self.get_logger().info("Loading Map and Metadata...")
        with open(map_json_path, 'r') as f:
            metadata = json.load(f)
        
        self.map_resolution = metadata["resolution_m_per_cell"]
        grid_int8 = np.load(map_npy_path)
        
        # Convert the int8 occupancy grid (0=free, 100=occupied, -1=unknown) 
        # into a binary map for the raycaster (0=free space, 1=obstacles/unknown)
        self.world_binary = np.where(grid_int8 == 0, 0, 1).astype(np.int8)
        
        # 3. Pre-compute Ray-Casting Look-Up Table (LUT)
        self.orientations = make_orientations(NUM_ANGLES)
        self.beam_angles = make_fov_angles(fov_rad=self.fov_rad, num_beams=NUM_BEAMS)
        
        self.get_logger().info("Generating Ray-Cast LUT... Please wait.")
        max_range_cells = int(SENSOR_MAX_RANGE_METERS / self.map_resolution)
        self.lut = ray_cast_lut_pose(
            self.world_binary, 
            self.orientations, 
            self.beam_angles, 
            max_range=max_range_cells
        )
        self.get_logger().info("LUT Generated successfully.")

        # 4. Initialize State Variables
        self.current_pose_gt_meters = np.array([0.0, 0.0])
        self.current_yaw_gt = 0.0
        self.has_odom = False

        # 5. Setup ROS 2 Subscriptions
        # Subscribes to the optical flow pose estimate
        self.odom_sub = self.create_subscription(
            PoseStamped, '/flow_depth/pose_est', self.odom_callback, 10)
            
        # Subscribes to the depth map
        self.depth_sub = self.create_subscription(
            Image, '/xtend/depth_m', self.depth_callback, 10)


    def odom_callback(self, msg: PoseStamped):
        """
        Callback triggered by the optical flow odometry.
        Extracts 2D position and Yaw angle from the PoseStamped message.
        """
        # Extract X and Y position coordinates
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        # Extract the orientation quaternion and convert to Yaw
        q = msg.pose.orientation
        yaw = get_yaw_from_quaternion(q)
        
        # Update current prediction state
        self.current_pose_gt_meters = np.array([x, y])
        self.current_yaw_gt = yaw
        
        # Flag to indicate we have an initial guess, allowing depth processing
        self.has_odom = True

    def depth_callback(self, msg: Image):
        """
        Callback triggered by the depth camera.
        Converts the depth image into a 1D pseudo-laserscan and executes AMCL.
        """
        if not self.has_odom:
            # Wait until we receive at least one odometry reading
            return 
            
        # 1. Convert ROS Image message to Numpy Array
        try:
            # Assuming the depth image is published in meters (e.g., 32FC1 encoding)
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 2. Generate a 1D Pseudo-Laserscan from the depth map
        height = depth_img.shape[0]
        # Extract the middle horizontal row of the depth image
        middle_row = depth_img[height // 2, :]
        
        # Downsample the row to match the desired number of AMCL beams
        indices = np.linspace(0, len(middle_row) - 1, NUM_BEAMS, dtype=int)
        z_measured_meters = middle_row[indices]
        
        # Clean invalid data (NaN or Infinity) by setting them to MAX_RANGE
        z_measured_meters = np.nan_to_num(z_measured_meters, nan=SENSOR_MAX_RANGE_METERS, posinf=SENSOR_MAX_RANGE_METERS)
        
        # Convert real-world measurements (meters) to map scale (cells)
        z_measured_cells = z_measured_meters / self.map_resolution

        # 3. Convert Odometry Prediction from meters to cells
        pred_x_cells = int(self.current_pose_gt_meters[0] / self.map_resolution)
        pred_y_cells = int(self.current_pose_gt_meters[1] / self.map_resolution)
        prediction_cells = np.array([pred_x_cells, pred_y_cells])

        # 4. Execute the AMCL Localization step
        # prediction_uncertainty controls the search window size around the initial guess (in cells)
        robot_loc_estimate_cells, robot_orientation_estimate = amcl_estimator(
            self.lut, 
            self.orientations, 
            prediction_cells,
            self.current_yaw_gt, 
            self.world_binary, 
            z_measured_cells,
            prediction_uncertainty=(8, 8) 
        )

        # 5. Convert AMCL results back to real-world coordinates (meters)
        est_x_m = robot_loc_estimate_cells[0] * self.map_resolution
        est_y_m = robot_loc_estimate_cells[1] * self.map_resolution

        # Log the comparison between Odometry guess and AMCL correction
        self.get_logger().info(f"Odom Pred [m]: x={self.current_pose_gt_meters[0]:.2f}, y={self.current_pose_gt_meters[1]:.2f}")
        self.get_logger().info(f"AMCL Est  [m]: x={est_x_m:.2f}, y={est_y_m:.2f}, yaw_deg={math.degrees(robot_orientation_estimate):.1f}")
        
        # Prevent drift by overriding the odometry with the corrected AMCL pose
        self.current_pose_gt_meters = np.array([est_x_m, est_y_m])
        self.current_yaw_gt = robot_orientation_estimate

        # 6. Visualization
        if SHOW_MAP:
            show_world_map(
                self.world_binary, 
                location=(robot_loc_estimate_cells[0], robot_loc_estimate_cells[1]), 
                orientation=robot_orientation_estimate, 
                title='AMCL Live Correction'
            )


def main(args=None):
    """
    ROS 2 Entry Point.
    Initializes the node with specific configuration paths.
    """
    rclpy.init(args=args)
    
    # Define absolute or relative paths to your configuration and map files
    # Note: In a production environment, these should be passed as ROS 2 parameters via a launch file.
    node = XtendAMCLNode(
        map_npy_path='/home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/data/occ_grid_int8.npy',
        map_json_path='/home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/data/occ_metadata.json',
        camera_yaml_path='/home/shirb/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml'
    )
    
    try:
        # Keep the node alive and listening to topics
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down AMCL Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()