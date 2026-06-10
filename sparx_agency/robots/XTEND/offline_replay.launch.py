"""
offline_replay.launch.py

Full offline replay pipeline for an XTEND take session
(recorded by take_xtend_da3_frames.py).

Replaces the live drone with offline_frame_dir_publisher, which reads
metadata.csv to replay RGB paths + bearing + optional depth at original timing.

depth_mode
  npy   (default) — read pre-computed .npy depth from the session.
                    No TRT inference; works on any laptop/PC.
  infer            — run DA3 TRT inference live during replay.
                    Requires engine_path. Slower but identical to Jetson output.

Pipeline:
  1. offline_frame_dir_publisher  → /xtend/rgb_frame_path, /xtend/bearing
                                  → /xtend/depth_frame_path  (npy mode only)
  2a. depth_to_pointcloud_node    → (npy mode)   depth_frame_path → /xtend/pointcloud
  2b. depth_processor_node        → (infer mode) rgb_frame_path   → /xtend/pointcloud
  3. apriltag_triangulation_node  → /xtend/rgb_frame_path → /xtend/april_tag_pose
  4. pose_to_tf_node              → april_tag_pose → map→xtend_camera TF
  5. static_transform_publisher   → fallback TF when no tag visible
  6. occupancy_from_pointcloud_node → /xtend/pointcloud + pose → /xtend/occupancy_grid
  7. rviz2

Usage:
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=$PYTHONPATH:/home/daphnaa/GIT/TheAgency
  ros2 launch sparx_agency/robots/XTEND/offline_replay.launch.py \\
      session_dir:=/path/to/xtend_da3_take_YYYYMMDD_HHMMSS

  # With TRT inference instead of saved depth:
  ros2 launch sparx_agency/robots/XTEND/offline_replay.launch.py \\
      session_dir:=... depth_mode:=infer engine_path:=/path/to/DA3.engine
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

_PY = os.path.expanduser("~/venvs/ros_py310/bin/python")
_RVIZ = os.path.join(os.path.dirname(__file__),
                     "../../tasks/mapping/ros2/rgbd_mapping.rviz")

_DEFAULT_CONFIG = os.path.expanduser(
    "~/GIT/TheAgency/sparx_agency/robots/XTEND/config"
    "/camera_xtend_ros_calib_720_420.yaml"
)
_DEFAULT_TAG_MAP = os.path.expanduser(
    "~/GIT/TheAgency/sparx_agency/tasks/localization/config/new_map.yaml"
)
_DEFAULT_ENGINE = os.path.expanduser(
    "~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx"
    "/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16.v2.engine"
)


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "session_dir", default_value="",
            description="Path to xtend_da3_take_* session directory"),
        DeclareLaunchArgument(
            "depth_mode", default_value="npy",
            description="npy = use saved .npy files | infer = run DA3 TRT live"),
        DeclareLaunchArgument(
            "config_yaml", default_value=_DEFAULT_CONFIG,
            description="Camera calibration YAML"),
        DeclareLaunchArgument(
            "tag_map_path", default_value=_DEFAULT_TAG_MAP,
            description="AprilTag world-coordinate map YAML"),
        DeclareLaunchArgument(
            "tag_size_m", default_value="0.13",
            description="AprilTag side length in metres"),
        DeclareLaunchArgument(
            "engine_path", default_value=_DEFAULT_ENGINE,
            description="(infer mode) DA3 TRT .engine path"),
        DeclareLaunchArgument(
            "max_depth_m", default_value="10.0",
            description="Far depth clip in metres"),
        DeclareLaunchArgument(
            "sensor_z", default_value="1.0",
            description="Fallback drone height (m) when no AprilTag visible"),
    ]

    _is_npy   = IfCondition(PythonExpression(
        ["'", LaunchConfiguration("depth_mode"), "' == 'npy'"]))
    _is_infer = IfCondition(PythonExpression(
        ["'", LaunchConfiguration("depth_mode"), "' == 'infer'"]))

    # offline publisher depth-mode:
    #   launch npy   → publisher sends npy files via /xtend/depth_frame_path
    #   launch infer → publisher sends rgb only; depth_processor runs inference
    _pub_depth_mode = PythonExpression(
        ["'npy' if '", LaunchConfiguration("depth_mode"), "' == 'npy' else 'none'"])

    # ── 1. Data source ────────────────────────────────────────────────────────
    offline_pub = ExecuteProcess(
        cmd=[
            _PY, "-m", "sparx_agency.robots.XTEND.offline_frame_dir_publisher",
            "--session-dir",      LaunchConfiguration("session_dir"),
            "--depth-mode",       _pub_depth_mode,
            "--bearing-topic",    "/xtend/bearing",
            "--depth-path-topic", "/xtend/depth_frame_path",
            "--use-original-timing",
            "--loop",
        ],
        output="screen",
    )

    # ── 2a. depth_frame_path → PointCloud2  (npy mode) ───────────────────────
    depth_to_cloud = Node(
        condition=_is_npy,
        executable=_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.depth_to_pointcloud_node"],
        parameters=[{
            "mode":             "file_path",
            "depth_path_topic": "/xtend/depth_frame_path",
            "config_yaml":      LaunchConfiguration("config_yaml"),
            "pointcloud_topic": "/xtend/pointcloud",
            "clip_max_m":       LaunchConfiguration("max_depth_m"),
        }],
        output="screen",
    )

    # ── 2b. rgb_frame_path → TRT → PointCloud2  (infer mode) ─────────────────
    # Delayed 3 s to let TRT engine warm up before frames arrive.
    depth_processor = TimerAction(period=3.0, actions=[
        Node(
            condition=_is_infer,
            executable=_PY,
            arguments=["-m", "sparx_agency.tasks.mapping.ros2.depth_processor_node"],
            parameters=[{
                "engine_path":       LaunchConfiguration("engine_path"),
                "config_yaml":       LaunchConfiguration("config_yaml"),
                "frame_path_topic":  "/xtend/rgb_frame_path",
                "depth_path_topic":  "/xtend/depth_frame_path",
                "pointcloud_topic":  "/xtend/pointcloud",
                "clip_max_m":        LaunchConfiguration("max_depth_m"),
                "publish_cloud":     True,
            }],
            output="screen",
        )
    ])

    # ── 3. AprilTag detection → /xtend/april_tag_pose ────────────────────────
    # Delayed 5 s so RGB frames are already flowing before detection starts.
    apriltag = TimerAction(period=5.0, actions=[
        ExecuteProcess(
            cmd=[
                _PY, "-m",
                "sparx_agency.tasks.localization.apriltag_triangulation_node",
                "--tag_map_path",      LaunchConfiguration("tag_map_path"),
                "--camera_calib_path", LaunchConfiguration("config_yaml"),
                "--tag_size_m",        LaunchConfiguration("tag_size_m"),
                "--frame_path_topic",  "/xtend/rgb_frame_path",
                "--no_vis",
            ],
            output="screen",
        )
    ])

    # ── 4. april_tag_pose → map→xtend_camera TF ──────────────────────────────
    pose_to_tf = Node(
        executable=_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.pose_to_tf_node"],
        parameters=[{"pose_topic": "/xtend/april_tag_pose"}],
        output="screen",
    )

    # ── 5. Static TF fallback ─────────────────────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x", "0", "--y", "0", "--z", LaunchConfiguration("sensor_z"),
            "--roll", "-1.5708", "--pitch", "0", "--yaw", "-1.5708",
            "--frame-id", "map", "--child-frame-id", "xtend_camera",
        ],
    )

    # ── 6. Occupancy grid ─────────────────────────────────────────────────────
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

    # ── 7. RViz ───────────────────────────────────────────────────────────────
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", _RVIZ],
        output="screen",
    )

    return LaunchDescription(args + [
        offline_pub,
        depth_to_cloud, depth_processor,
        apriltag, pose_to_tf, static_tf,
        occupancy, rviz,
    ])