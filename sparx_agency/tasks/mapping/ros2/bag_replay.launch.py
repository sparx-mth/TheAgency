"""
bag_replay.launch.py

Minimal pipeline for checking occupancy publishing from a recorded ROS 2 bag.
Skips TRT inference and AprilTag detection — both are already recorded in the bag.

Pipeline:
  1. depth_to_pointcloud_node        — /xtend/depth_m + /xtend/camera_info → /xtend/pointcloud
  2. pose_to_tf_node                 — /xtend/april_tag_pose → TF map→xtend_camera
  3. static_transform_publisher      — fallback TF map→xtend_camera at z=1.0 m (no tags visible)
  4. occupancy_from_pointcloud_node  — /xtend/pointcloud + pose → /xtend/occupancy_grid
  5. rviz2                           — visualisation

Usage:
  # Terminal 1 — launch the pipeline
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=$PYTHONPATH:/home/daphnaa/GIT/TheAgency
  python -m launch.ros2 bag_replay.launch.py

  # Terminal 2 — play the bag (use --loop to repeat)
  ros2 bag play /path/to/walk_into_0 --loop

Launch args:
  sensor_z    (default 1.0)   Drone hover height for static TF fallback
  clip_max_m  (default 10.0)  Depth clip distance fed to pointcloud backprojection
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PY = os.path.expanduser("~/venvs/ros_py310/bin/python")
_RVIZ = os.path.join(os.path.dirname(__file__), "rgbd_mapping.rviz")


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "sensor_z",   default_value="1.0",
            description="Fallback drone height (m) for static TF when no AprilTag is visible",
        ),
        DeclareLaunchArgument(
            "clip_max_m", default_value="10.0",
            description="Max depth (m) for pointcloud backprojection",
        ),
    ]

    # ── 1. Depth image → PointCloud2 ─────────────────────────────────────────
    depth_to_cloud = Node(
        executable=_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.depth_to_pointcloud_node"],
        parameters=[{
            "depth_topic":       "/xtend/depth_m",
            "camera_info_topic": "/xtend/camera_info",
            "pointcloud_topic":  "/xtend/pointcloud",
            "clip_max_m":        LaunchConfiguration("clip_max_m"),
        }],
        output="screen",
    )

    # ── 2. AprilTag pose → dynamic TF map→xtend_camera ───────────────────────
    pose_to_tf = Node(
        executable=_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.pose_to_tf_node"],
        parameters=[{"pose_topic": "/xtend/april_tag_pose"}],
        output="screen",
    )

    # ── 3. Static TF fallback (used when no tag in view) ─────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0", "--y", "0", "--z", "1.0",
            "--roll", "-1.5708", "--pitch", "0", "--yaw", "-1.5708",
            "--frame-id", "map",
            "--child-frame-id", "xtend_camera",
        ],
    )

    # ── 4. Occupancy grid from pointcloud ─────────────────────────────────────
    occupancy = Node(
        executable=_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.occupancy_from_pointcloud_node"],
        parameters=[{
            "pointcloud_topic": "/xtend/pointcloud",
            "pose_topic":       "/xtend/april_tag_pose",
            "cloud_out_topic":  "/xtend/pointcloud_world",
            "occupancy_topic":  "/xtend/occupancy_grid",
            "sensor_z":         LaunchConfiguration("sensor_z"),
        }],
        output="screen",
    )

    # ── 5. RViz ───────────────────────────────────────────────────────────────
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", _RVIZ],
        output="screen",
    )

    return LaunchDescription(args + [depth_to_cloud, pose_to_tf, static_tf, occupancy, rviz])