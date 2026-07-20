"""The drone side of the demo: bridge, depth, localization, TF and the map.

The perception chain, in the order it has to come up:

    XTEND WebSocket -> frames on disk -> /xtend/rgb_frame_path
      -> depth (DA3 TRT) -> /xtend/depth_m + /xtend/pointcloud
      -> AprilTag localization -> /xtend/localization
      -> pose_to_tf -> TF map->xtend_camera
      -> octomap_server -> /projected_map (what FALCON plans on)

Each item names the file that declares its parameters, so the parameter screen
is the node's real parameter list -- the localization node declares thirty-seven,
of which the command below spells out six -- rather than only what someone
thought to put on the command line.

The planner commands live in :mod:`falcon_items`; :data:`LAUNCH_ITEMS` is the two
lists joined, which is the order the demo is brought up in.
"""
from __future__ import annotations

from sparx_agency.tasks.common.launch_params.discovery import Source
from sparx_agency.tasks.common.launch_params.spec import SLOT, ParamSpec

from .environments import JETSON_REPO, PC_REPO
from .falcon_items import FALCON_ITEMS
from .item import LaunchItem

XTEND_ITEMS: list[LaunchItem] = [
    LaunchItem(
        name="1. XTEND online bridge + frame dir publisher",
        machine="jetson",
        tmux_name="xtend_bridge",
        description=(
            "Owns the XTEND WebSocket. Saves 504x294 resized frames to /tmp/xtend_frames "
            "and publishes each path on /xtend/rgb_frame_path (std_msgs/String). Also "
            "publishes /xtend/bearing and /xtend/local_telemetry.\n\n"
            "Everything downstream reads those paths, so this comes up first."
        ),
        command=f"""
python3 {JETSON_REPO}/sparx_agency/robots/XTEND/online_nav_bridge_dir_publisher.py \\
  --frequency 10.0 \\
  --out-dir /tmp/xtend_frames \\
  --path-topic /xtend/rgb_frame_path \\
  --preprocess-mode resize \\
  --output-width 504 \\
  --output-height 294
""",
        param_sources=(Source("argparse",
                              "sparx_agency/robots/XTEND/online_nav_bridge_dir_publisher.py"),),
    ),
    LaunchItem(
        name="2. DA3 Large Metric 504x294 depth + point cloud",
        machine="jetson",
        tmux_name="xtend_depth",
        description=(
            "Reads frames from /xtend/rgb_frame_path, runs DA3METRIC-LARGE FP16 TensorRT, "
            "and publishes /xtend/depth_m (32FC1 metres) and /xtend/pointcloud "
            "(PointCloud2 in the xtend_camera frame). Saves depth arrays to "
            "/tmp/xtend_depth and publishes their paths on /xtend/depth_frame_path.\n\n"
            "The engine path is device-specific and the calibration must match the "
            "publisher's output size (504x294), or the cloud comes out the wrong scale."
        ),
        command=f"""
python3 {JETSON_REPO}/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \\
  --ros-args \\
  -p frame_path_topic:=/xtend/rgb_frame_path \\
  -p depth_topic:=/xtend/depth_m \\
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-294x504.depth_only.engine \\
  -p config_yaml:={JETSON_REPO}/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \\
  -p camera_info_mode:=base \\
  -p model_type:=large_metric \\
  -p apply_metric_focal_scaling:=true \\
  -p metric_scale_divisor:=300.0 \\
  -p clip_min_m:=0.2 \\
  -p clip_max_m:=5.0 \\
  -p depth_encoding:=32FC1 \\
  -p depth_path_topic:=/xtend/depth_frame_path \\
  -p depth_dir:=/tmp/xtend_depth \\
  -p max_depth_kept:=300 \\
  -p publish_cloud:=false \\
  -p pointcloud_topic:=/xtend/pointcloud
""",
        param_sources=(Source("ros2_node",
                              "sparx_agency/tasks/mapping/ros2/depth_processor_node.py"),),
    ),
    LaunchItem(
        name="3. XTEND demo mode manager",
        machine="jetson",
        tmux_name="xtend_demo_manager",
        description=(
            "Publishes /xtend/demo_mode from planner and UI requests, and turns a FINISH "
            "request into stop -> land -> disarm. The object mission ends by asking this "
            "node to finish, so it must be up before anything can land itself."
        ),
        command=f"""
python3 {JETSON_REPO}/sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_drone_demo_manager.py \\
  --request-topic /xtend/demo_mode_request \\
  --mode-topic /xtend/demo_mode \\
  --cmd-nav-topic /xtend/cmd_nav \\
  --cmd-nav-state-sub-topic /xtend/cmd_nav \\
  --reset-odom-topic /xtend/reset_odom \\
  --initial-mode idle \\
  --disarm-delay-sec 8.0
""",
        param_sources=(Source(
            "argparse",
            "sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_drone_demo_manager.py"),),
    ),
    LaunchItem(
        name="4. Localization node (AprilTag provider)",
        machine="jetson",
        tmux_name="xtend_apriltag",
        description=(
            "Reads frames from /xtend/rgb_frame_path, detects tag36h11 AprilTags and "
            "estimates a 6-DOF pose via solvePnP. Publishes /xtend/localization "
            "(PoseStamped) and /xtend/localization_source (String).\n\n"
            "This is the world pose the whole planner stack flies on, so its parameters "
            "are worth knowing: alpha is the filter gain at full confidence, coast_frames "
            "is how long it dead-reckons with no tag in view, and the cmd_* parameters "
            "control how much the last velocity command is trusted while blind."
        ),
        command=f"""
python3 -m sparx_agency.tasks.localization.ros2.localization_node \\
  --ros-args \\
  -p provider_type:=apriltag \\
  -p frame_path_topic:=/xtend/rgb_frame_path \\
  -p tag_map_path:={JETSON_REPO}/sparx_agency/tasks/localization/config/new_map.yaml \\
  -p camera_calib_path:={JETSON_REPO}/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \\
  -p tag_size_m:=0.13 \\
  -p alpha:=0.2
""",
        param_sources=(Source("ros2_node",
                              "sparx_agency/tasks/localization/ros2/localization_node.py"),),
    ),
    LaunchItem(
        name="5. Static TF fallback (map -> xtend_camera)",
        machine="jetson",
        tmux_name="xtend_static_tf",
        description=(
            "Publishes a static TF map->xtend_camera at z=1.0 m (hover height). The "
            "rotation roll=-pi/2 yaw=-pi/2 aligns the optical frame (X right, Y down, "
            "Z forward) with the map.\n\n"
            "Overridden dynamically by item 6 while a tag is visible; TF2 falls back to "
            "this after roughly 10 s without one."
        ),
        command="""
ros2 run tf2_ros static_transform_publisher \\
  --x 0 --y 0 --z 1.0 \\
  --roll -1.5708 --pitch 0 --yaw -1.5708 \\
  --frame-id map \\
  --child-frame-id xtend_camera
""",
    ),
    LaunchItem(
        name="6. Pose-to-TF bridge (localization -> TF)",
        machine="jetson",
        tmux_name="xtend_pose_to_tf",
        description=(
            "Subscribes to /xtend/localization, applies the body->optical frame "
            "correction, and broadcasts the dynamic TF map->xtend_camera that octomap "
            "raycasts against. Overrides the static fallback while tags are visible."
        ),
        command="""
python3 -m sparx_agency.tasks.mapping.ros2.pose_to_tf_node \\
  --ros-args \\
  -p pose_topic:=/xtend/localization
""",
        param_sources=(Source("ros2_node",
                              "sparx_agency/tasks/mapping/ros2/pose_to_tf_node.py"),),
    ),
    LaunchItem(
        name="7. Octomap server (3D voxels + 2D occupancy)",
        machine="jetson",
        tmux_name="xtend_octomap",
        description=(
            "Subscribes to /xtend/pointcloud (xtend_camera frame) and raycasts using the "
            "TF from item 6. Publishes /occupied_cells_vis_array (3D) and /projected_map "
            "(the 2D grid). occupancy_min_z clips the floor out of the 2D projection.\n\n"
            "A C++ node, so its full parameter list is not in this repo — the ones the "
            "command names are editable, and any other can be added by hand."
        ),
        command="""
ros2 run octomap_server octomap_server_node \\
  --ros-args \\
  -p resolution:=0.05 \\
  -p frame_id:=map \\
  -p sensor_model/max_range:=5.0 \\
  -p sensor_model/hit:=0.70 \\
  -p sensor_model/miss:=0.40 \\
  -p occupancy_thres:=0.50 \\
  -p occupancy_min_z:=0.1 \\
  -p occupancy_max_z:=5.0 \\
  -p filter_ground:=false \\
  --remap cloud_in:=/xtend/pointcloud
""",
    ),
    LaunchItem(
        name="8. Optional Twist replayer",
        machine="jetson",
        tmux_name="xtend_twist_replayer",
        description=(
            "Replays a JSONL Twist log onto /cmd_vel. For reproducing a flight without "
            "the planner in the loop; set log_path to the recording."
        ),
        enabled_by_default=False,
        command=f"""
python3 {JETSON_REPO}/sparx_agency/tasks/planning/twist_replayer.py \\
  --ros-args \\
  -p log_path:={JETSON_REPO}/cmd_log.jsonl \\
  -p topic:=/cmd_vel \\
  -p speed:=1.0 \\
  -p loop:=false
""",
        param_sources=(Source("ros2_node", "sparx_agency/tasks/planning/twist_replayer.py"),),
    ),
    LaunchItem(
        name="9. PC: RViz mapping view",
        machine="pc",
        tmux_name="xtend_pc_rviz",
        description=(
            "Opens RViz with the mapping config on the PC side. Needs the same "
            "ROS_DOMAIN_ID as the Jetson or no topic will appear."
        ),
        enabled_by_default=False,
        command="rviz2 -d {config}",
        template=True,
        params=(ParamSpec(
            name="config",
            default=f"{PC_REPO}/sparx_agency/tasks/mapping/ros2/rgbd_mapping.rviz",
            syntax=SLOT, section="What it opens",
            doc="RViz config with the demo's displays already wired up."),),
    ),
    LaunchItem(
        name="10. PC: manual flight UI",
        machine="pc",
        tmux_name="xtend_pc_ui",
        description=(
            "Manual ARM / TAKEOFF / LAND / DISARM / STOP UI, which can also publish Twist "
            "directly. The hand-flying counterpart to the mission."
        ),
        enabled_by_default=False,
        command=f"python3 {PC_REPO}/sparx_agency/robots/XTEND/ui.py",
    ),
]

#: The whole catalog, in bring-up order: perception first, then the planner.
LAUNCH_ITEMS: list[LaunchItem] = XTEND_ITEMS + FALCON_ITEMS
