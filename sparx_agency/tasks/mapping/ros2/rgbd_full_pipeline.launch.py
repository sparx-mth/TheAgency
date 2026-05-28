"""
rgbd_full_pipeline.launch.py

Launches the full RGBD mapping pipeline (per-frame, TRT DA3METRIC-LARGE):
  1. FrameSourceNode            — RGB frames from directory (mock) or live drone topic
  2. DepthProcessorNode         — DA3 TRT depth (32FC1) + PointCloud2
  3. octomap_server_node        — 3D occupancy (log-odds Bayesian) + 2D projected OccupancyGrid
  4. static_transform_publisher — fallback map → xtend_camera at z=1.0 m (drone initial height)
  5. apriltag_triangulation_node — estimates camera pose in map frame from detected tags
  6. pose_to_tf_node            — converts april_tag_pose → dynamic map→xtend_camera TF

Namespace
  xtend_ns (default: xtend) controls the /xtend/ topic prefix for multi-drone support.
  All drone-side topics are /{xtend_ns}/…  Pipeline-internal topics stay on /rgbd/…

Octomap log-odds thresholds (tunable via launch args):
  sensor_model/hit   = 0.70  → P(occupied | ray_hit),   adds  +0.85 log-odds
  sensor_model/miss  = 0.40  → P(occupied | ray_pass),  adds  -0.41 log-odds
  occupancy_thres    = 0.50  → voxel is OCCUPIED when P > this value
  prob_hit / miss ratio ≈ 2:1, so ~2 consistent hits flip a voxel from unknown to occupied

Requires:  sudo apt install ros-humble-octomap-server
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

_ROS_PY = os.path.expanduser("~/venvs/ros_py310/bin/python")
_DA3_PY = os.path.expanduser(
    "~/depth_anything_ws/src/ros2-depth-anything-v3-trt/da3_venv/bin/python3"
)
_DEFAULT_ENGINE = os.path.expanduser(
    "~/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16.v2.engine"
)
_DEFAULT_YAML = os.path.expanduser(
    "~/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml"
)
_DEFAULT_TAG_MAP = os.path.expanduser(
    "~/GIT/TheAgency/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml"
)
_DEFAULT_RGB_DIR = os.path.expanduser(
    "~/Documents/xtend_da3_takes/xtend_da3_take_20260527_124147/rgb"
)


def generate_launch_description():
    # ── Launch arguments ──────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("xtend_ns", default_value="xtend",
                              description="Drone topic namespace prefix (e.g. xtend, drone2)"),
        DeclareLaunchArgument("mode", default_value="mock",
                              description="Frame source mode: mock | live"),
        DeclareLaunchArgument("rgb_dir", default_value=_DEFAULT_RGB_DIR,
                              description="(mock) Directory of RGB images"),
        DeclareLaunchArgument("publish_hz", default_value="3.0",
                              description="(mock) Frame publish rate — keep ≤ TRT throughput"),
        DeclareLaunchArgument("engine_path", default_value=_DEFAULT_ENGINE,
                              description="TensorRT engine for DepthProcessorNode"),
        DeclareLaunchArgument("config_yaml", default_value=_DEFAULT_YAML,
                              description="Camera calibration YAML"),
        DeclareLaunchArgument("max_depth_m", default_value="8.0",
                              description="Max depth for cloud clipping and octomap ray range"),
        # Octomap sensor model — tune these for environment:
        #   indoor tight space: hit=0.75 miss=0.45 → faster occupancy buildup
        #   outdoor / noisy:    hit=0.65 miss=0.35 → more conservative
        DeclareLaunchArgument("octomap_resolution", default_value="0.05",
                              description="Octomap voxel size in metres (0.05 = 5 cm)"),
        DeclareLaunchArgument("prob_hit",  default_value="0.70",
                              description="P(occ | ray hit endpoint)"),
        DeclareLaunchArgument("prob_miss", default_value="0.40",
                              description="P(occ | ray passes through)"),
        DeclareLaunchArgument("occupancy_thres", default_value="0.50",
                              description="P threshold above which voxel is OCCUPIED"),
        DeclareLaunchArgument("min_z", default_value="0.1",
                              description="Min Z for octomap 2D projection (above floor)"),
        DeclareLaunchArgument("max_z", default_value="5.0",
                              description="Max Z for octomap occupied voxels"),
    ]

    # Derived topic names from namespace
    ns = LaunchConfiguration("xtend_ns")
    source_topic    = PythonExpression(["'/' + '", ns, "' + '/rgb'"])
    april_pose_topic = PythonExpression(["'/' + '", ns, "' + '/april_tag_pose'"])

    # ── Node 1: Frame Source ──────────────────────────────────────────────────
    frame_source = Node(
        executable=_ROS_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.frame_source_node"],
        parameters=[{
            "mode":               LaunchConfiguration("mode"),
            "rgb_dir":            LaunchConfiguration("rgb_dir"),
            "source_topic":       source_topic,
            "camera_config_yaml": LaunchConfiguration("config_yaml"),
            "publish_hz":         LaunchConfiguration("publish_hz"),
            "rgb_topic":          "/rgbd/rgb",
            "camera_info_topic":  "/rgbd/camera_info",
            "frame_id":           "xtend_camera",
            "loop":               False,
        }],
        output="screen",
    )

    # ── Node 2: Depth Processor — TRT DA3METRIC-LARGE ─────────────────────────
    # Uses ros_py310 (TRT 10.15.1.29, SM86 support) to match the fp16.v2.engine
    # built with system trtexec 10.14.1.48. da3_venv (10.16.1.11) lacks SM86.
    # Delayed 3 s to let frame source initialise first.
    depth_processor = TimerAction(
        period=3.0,
        actions=[Node(
            executable=_ROS_PY,
            arguments=["-m", "sparx_agency.tasks.mapping.ros2.depth_processor_node"],
            parameters=[{
                "engine_path":      LaunchConfiguration("engine_path"),
                "config_yaml":      LaunchConfiguration("config_yaml"),
                "image_topic":      "/rgbd/rgb",
                "depth_topic":      "/rgbd/depth_m",
                "depth_encoding":   "32FC1",
                "camera_info_mode": "crop_resize",
                "clip_max_m":       LaunchConfiguration("max_depth_m"),
                "publish_cloud":    True,
                "pointcloud_topic": "/rgbd/pointcloud",
            }],
            output="screen",
        )],
    )

    # ── Node 3: Octomap Server ────────────────────────────────────────────────
    # Subscribes to /rgbd/pointcloud (xtend_camera frame).
    # Raycasts from sensor origin (resolved via TF) to each point:
    #   - marks voxels along ray as FREE  (log-odds -= 0.41)
    #   - marks endpoint voxel as HIT     (log-odds += 0.85)
    # Voxel is OCCUPIED when accumulated P(occ) > occupancy_thres.
    octomap = Node(
        package="octomap_server",
        executable="octomap_server_node",
        parameters=[{
            "resolution":             LaunchConfiguration("octomap_resolution"),
            "frame_id":               "map",
            "sensor_model/max_range": LaunchConfiguration("max_depth_m"),
            "sensor_model/hit":       LaunchConfiguration("prob_hit"),
            "sensor_model/miss":      LaunchConfiguration("prob_miss"),
            "occupancy_thres":        LaunchConfiguration("occupancy_thres"),
            "occupancy_min_z":        LaunchConfiguration("min_z"),
            "occupancy_max_z":        LaunchConfiguration("max_z"),
            "filter_ground":          False,
        }],
        remappings=[("cloud_in", "/rgbd/pointcloud")],
        output="screen",
    )

    # ── Node 4: Static TF  map → xtend_camera ────────────────────────────────
    # Fallback pose (z=1.0m = drone hover height) used when no AprilTag is visible.
    # Dynamic TF from pose_to_tf_node overrides this while tags are detected.
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

    # ── Node 5: AprilTag triangulation ───────────────────────────────────────
    # Uses argparse (not ROS2 params) → ExecuteProcess.
    # mock mode: subscribes to /rgbd/rgb (relay from frame_source_node)
    # live mode: subscribes to /{xtend_ns}/rgb directly (set image_topic arg)
    # Publishes camera pose to /{xtend_ns}/april_tag_pose
    apriltag = TimerAction(
        period=5.0,
        actions=[ExecuteProcess(
            cmd=[
                _ROS_PY, "-m",
                "sparx_agency.tasks.localization.apriltag_triangulation_node",
                "--tag_map_path",      _DEFAULT_TAG_MAP,
                "--camera_calib_path", _DEFAULT_YAML,
                "--tag_size_m",        "0.13",
                "--source",            "ros",
                "--image_topic",       "/rgbd/rgb",
                "--pose_topic",        april_pose_topic,
                "--no_vis",
            ],
            output="screen",
        )],
    )

    # ── Node 6: Pose → TF bridge ──────────────────────────────────────────────
    # Converts /{xtend_ns}/april_tag_pose → dynamic map→xtend_camera TF.
    # Corrects body→optical frame convention (q_opt = q_body ⊗ q_body_T_optical).
    # Overrides the static fallback while tags are visible.
    pose_to_tf = Node(
        executable=_ROS_PY,
        arguments=["-m", "sparx_agency.tasks.mapping.ros2.pose_to_tf_node"],
        parameters=[{"pose_topic": april_pose_topic}],
        output="screen",
    )

    return LaunchDescription(args + [frame_source, depth_processor, octomap, static_tf, apriltag, pose_to_tf])