#!/usr/bin/env python3
"""
XTEND pipeline launcher UI — RGBD Mapping variant.

Replaces optical-flow localization (flow_depth + velocity_integrator + static odom TF)
with AprilTag-based 6-DOF pose estimation driving a live octomap:

  AprilTag triangulation → /xtend/april_tag_pose
    → pose_to_tf_node    → TF map→xtend_camera (dynamic, overrides static fallback)
    → octomap_server     ← /xtend/pointcloud (from depth_processor_node)

Safety:
- AUTO sends ARM and TAKEOFF after bridge/depth are available.
- AUTO then waits 30 seconds and starts mapping nodes + planner.
- Closing this launcher does not stop tmux sessions.

Requires on Jetson:  sudo apt install ros-humble-octomap-server
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Literal

JETSON_SSH_DEFAULT = "user@192.0.0.89"
JETSON_REPO = "/home/user/agency_ws"

JETSON_ENV = """
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
""".replace("/home/user/GIT/TheAgency", JETSON_REPO)

PC_ENV = """
cd /home/user1/GIT/TheAgency
source /opt/ros/jazzy/setup.bash
source /home/user1/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/ros/jazzy/lib
export PYTHONPATH=/usr/lib/python3.12/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/home/user1/GIT/TheAgency:${PYTHONPATH}
"""

AUTO_PIPELINE_SCRIPT = r"""
set +e

echo "[AUTO] XTEND RGBD mapping pipeline auto launch started"

cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}

wait_for_topic_name() {
    local topic="$1"
    local timeout_sec="$2"
    echo "[AUTO] Waiting for topic name: ${topic} (${timeout_sec}s)"
    local start_t=$(date +%s)
    while true; do
        if ros2 topic list | grep -qx "${topic}"; then
            echo "[AUTO] Topic exists: ${topic}"
            return 0
        fi
        local now_t=$(date +%s)
        if [ $((now_t - start_t)) -ge "${timeout_sec}" ]; then
            echo "[AUTO][WARN] Timeout waiting for topic name: ${topic}"
            return 1
        fi
        sleep 1
    done
}

wait_for_topic_rate() {
    local topic="$1"
    local timeout_sec="$2"
    local reliability="${3:-best_effort}"
    echo "[AUTO] Waiting for messages on: ${topic} (${timeout_sec}s, qos=${reliability})"
    timeout "${timeout_sec}" ros2 topic hz "${topic}" >/tmp/xtend_wait_topic_hz.log 2>&1 || true
    if grep -q "average rate" /tmp/xtend_wait_topic_hz.log 2>/dev/null; then
        echo "[AUTO] Topic has messages: ${topic}"
        cat /tmp/xtend_wait_topic_hz.log || true
        return 0
    fi
    echo "[AUTO][WARN] No messages confirmed on: ${topic}"
    cat /tmp/xtend_wait_topic_hz.log || true
    return 1
}

send_xtend_cmd() {
    local action="$1"
    local value="${2:-0}"
    echo "[AUTO] Sending XTEND command: action=${action}, value=${value}"
    ros2 topic pub --once /xtend/cmd_nav std_msgs/msg/String "{data: '{\"action\":\"${action}\", \"value\":${value}}'}"
}

start_tmux() {
    local session="$1"
    local command="$2"
    echo "[AUTO] Starting tmux session: ${session}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    tmux new-session -d -s "${session}" "bash -lc $(printf '%q' "${command}")"
}

echo "[AUTO] Step 1: start online bridge + RGB"
start_tmux xtend_bridge '
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/online_nav_bridge_publisher.py \
  --frequency 10.0 \
  --image-topic /xtend/rgb \
  --camera-info-topic /xtend/camera_info \
  --camera-info-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  --preprocess-mode resize \
  --output-width 504 \
  --output-height 294
'

wait_for_topic_name /xtend/rgb 20
wait_for_topic_rate /xtend/bearing 20 best_effort || true
wait_for_topic_rate /xtend/rgb 20 best_effort || true

echo "[AUTO] Step 2: start DA3 Large Metric 504x294 depth + point cloud"
start_tmux xtend_depth '
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
python3 /home/user/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-294x504.depth_only.v2.engine \
  -p config_yaml:=/home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  -p camera_info_mode:=base \
  -p model_type:=large_metric \
  -p apply_metric_focal_scaling:=true \
  -p metric_scale_divisor:=300.0 \
  -p clip_min_m:=0.2 \
  -p clip_max_m:=5.0 \
  -p depth_encoding:=32FC1 \
  -p publish_cloud:=true \
  -p pointcloud_topic:=/xtend/pointcloud
'

wait_for_topic_name /xtend/depth_m 30
wait_for_topic_rate /xtend/depth_m 60 best_effort
wait_for_topic_name /xtend/pointcloud 15

echo "[AUTO] Step 3: start Twist converter"
start_tmux xtend_twist_converter '
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONPATH=/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/adapters/xtend_twist_to_cmd_nav.py \
  --cmd-vel-topic /cmd_vel \
  --cmd-nav-topic /xtend/cmd_nav \
  --timeout-sec 1.5
'

sleep 2

echo "[AUTO] Step 4: start XTEND demo mode manager"
start_tmux xtend_demo_manager '
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONPATH=/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
python3 /home/user/GIT/TheAgency/sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_drone_demo_manager.py \
  --request-topic /xtend/demo_mode_request \
  --mode-topic /xtend/demo_mode \
  --cmd-nav-topic /xtend/cmd_nav \
  --cmd-nav-state-sub-topic /xtend/cmd_nav \
  --reset-odom-topic /xtend/reset_odom \
  --initial-mode idle \
  --disarm-delay-sec 8.0
'

wait_for_topic_name /xtend/demo_mode 15 || true

echo "[AUTO] Step 5: arm and takeoff"
send_xtend_cmd arm 0
sleep 3
send_xtend_cmd takeoff 0

echo "[AUTO] Waiting 30 seconds for takeoff/stabilization"
sleep 30

echo "[AUTO] Re-check depth + pointcloud before mapping"
wait_for_topic_rate /xtend/depth_m 30 best_effort
wait_for_topic_rate /xtend/pointcloud 30 best_effort

echo "[AUTO] Step 6: start AprilTag triangulation (pose estimation)"
start_tmux xtend_apriltag '
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONPATH=/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
python3 -m sparx_agency.tasks.localization.apriltag_triangulation_node \
  --tag_map_path /home/user/GIT/TheAgency/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml \
  --camera_calib_path /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  --tag_size_m 0.13 \
  --source ros \
  --image_topic /xtend/rgb \
  --pose_topic /xtend/april_tag_pose \
  --no_vis
'

echo "[AUTO] Step 7: start static TF fallback (map -> xtend_camera, z=1.0m)"
start_tmux xtend_static_tf '
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=5
ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 1.0 \
  --roll -1.5708 --pitch 0 --yaw -1.5708 \
  --frame-id map \
  --child-frame-id xtend_camera
'

echo "[AUTO] Step 8: start pose-to-TF bridge (AprilTag pose -> TF)"
start_tmux xtend_pose_to_tf '
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONPATH=/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
python3 -m sparx_agency.tasks.mapping.ros2.pose_to_tf_node \
  --ros-args \
  -p pose_topic:=/xtend/april_tag_pose
'

echo "[AUTO] Step 9: start octomap server"
start_tmux xtend_octomap '
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=5
ros2 run octomap_server octomap_server_node \
  --ros-args \
  -p resolution:=0.05 \
  -p frame_id:=map \
  -p sensor_model/max_range:=5.0 \
  -p sensor_model/hit:=0.70 \
  -p sensor_model/miss:=0.40 \
  -p occupancy_thres:=0.50 \
  -p occupancy_min_z:=0.1 \
  -p occupancy_max_z:=5.0 \
  -p filter_ground:=false \
  --remap cloud_in:=/xtend/pointcloud
'

wait_for_topic_name /xtend/april_tag_pose 20 || true

echo "[AUTO] Step 10: check planner containers"
docker ps --format '{{.Names}}' | tee /tmp/xtend_docker_names.txt
grep -qx falcon /tmp/xtend_docker_names.txt || echo "[AUTO][WARN] falcon container is not running"
grep -qx ros1_bridge /tmp/xtend_docker_names.txt || echo "[AUTO][WARN] ros1_bridge container is not running"
grep -qx roscore /tmp/xtend_docker_names.txt || echo "[AUTO][WARN] roscore container is not running"

echo "[AUTO] Step 11: launch FALCON planner inside falcon container"
if docker ps --format '{{.Names}}' | grep -qx falcon; then
    tmux kill-session -t planner_falcon 2>/dev/null || true
    tmux new-session -d -s planner_falcon "docker exec -it falcon bash -lc 'roslaunch falcon_adapter real_drone.launch map_name:=office'"
else
    echo "[AUTO][WARN] Skipping planner launch because falcon container is not running"
fi

echo "[AUTO] Optional display commands:"
echo "  docker exec -it falcon bash -lc 'roslaunch exploration_manager rviz.launch'"
echo "  docker exec -it falcon bash -lc 'rosrun falcon_adapter bev_click_goal.py'"

echo "[AUTO] Done. Active tmux sessions:"
tmux ls || true
""".replace("/home/user/GIT/TheAgency", JETSON_REPO)


@dataclass(frozen=True)
class LaunchItem:
    name: str
    machine: Literal["jetson", "pc", "manual"]
    tmux_name: str
    description: str
    command: str
    enabled_by_default: bool = True


LAUNCH_ITEMS: list[LaunchItem] = [
    LaunchItem(
        name="1. XTEND online bridge + RGB publisher",
        machine="jetson",
        tmux_name="xtend_bridge",
        description="Owns XTEND WebSocket, publishes /xtend/rgb as 504x294 full-FOV resized frames, /xtend/camera_info, /xtend/bearing, /xtend/local_telemetry, subscribes to /xtend/cmd_nav.",
        command="""
python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/online_nav_bridge_publisher.py \
  --frequency 10.0 \
  --image-topic /xtend/rgb \
  --camera-info-topic /xtend/camera_info \
  --camera-info-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  --preprocess-mode resize \
  --output-width 504 \
  --output-height 294
""",
    ),
    LaunchItem(
        name="2. DA3 Large Metric 504x294 depth + point cloud",
        machine="jetson",
        tmux_name="xtend_depth",
        description=(
            "Subscribes to /xtend/rgb 504x294, runs DA3METRIC-LARGE FP16 TRT, "
            "publishes /xtend/depth_m (32FC1 meters) and /xtend/pointcloud (PointCloud2 in xtend_camera frame). "
            "32FC1 required for correct metric backprojection."
        ),
        command="""
python3 /home/user/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
  --ros-args \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-294x504.depth_only.v2.engine \
  -p config_yaml:=/home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  -p camera_info_mode:=base \
  -p model_type:=large_metric \
  -p apply_metric_focal_scaling:=true \
  -p metric_scale_divisor:=300.0 \
  -p clip_min_m:=0.2 \
  -p clip_max_m:=5.0 \
  -p depth_encoding:=32FC1 \
  -p publish_cloud:=true \
  -p pointcloud_topic:=/xtend/pointcloud
""",
    ),
    LaunchItem(
        name="3. Twist -> XTEND command converter",
        machine="jetson",
        tmux_name="xtend_twist_converter",
        description="Converts /cmd_vel Twist to /xtend/cmd_nav JSON. Calibrated: linear.x=0.3 m/s -> forward thrust 400, forward max 600.",
        command="""
python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/adapters/xtend_twist_to_cmd_nav.py \
  --cmd-vel-topic /cmd_vel \
  --cmd-nav-topic /xtend/cmd_nav \
  --timeout-sec 1.5
""",
    ),
    LaunchItem(
        name="4. XTEND demo mode manager",
        machine="jetson",
        tmux_name="xtend_demo_manager",
        description="Publishes /xtend/demo_mode from planner/UI requests and handles FINISH as stop -> land -> disarm.",
        command="""
python3 /home/user/GIT/TheAgency/sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_drone_demo_manager.py \
  --request-topic /xtend/demo_mode_request \
  --mode-topic /xtend/demo_mode \
  --cmd-nav-topic /xtend/cmd_nav \
  --cmd-nav-state-sub-topic /xtend/cmd_nav \
  --reset-odom-topic /xtend/reset_odom \
  --initial-mode idle \
  --disarm-delay-sec 8.0
""",
    ),
    LaunchItem(
        name="5. Optional Twist replayer",
        machine="jetson",
        tmux_name="xtend_twist_replayer",
        description="Optional: replays a JSONL Twist log onto /cmd_vel. Edit LOG_PATH before running.",
        enabled_by_default=False,
        command="""
LOG_PATH="/home/user/GIT/TheAgency/cmd_log.jsonl"
python3 /home/user/GIT/TheAgency/sparx_agency/tasks/planning/twist_replayer.py \
  --ros-args \
  -p log_path:="${LOG_PATH}" \
  -p topic:=/cmd_vel \
  -p speed:=1.0 \
  -p loop:=false
""",
    ),
    LaunchItem(
        name="6. AprilTag triangulation (pose estimation)",
        machine="jetson",
        tmux_name="xtend_apriltag",
        description=(
            "Detects tag36h11 AprilTags in /xtend/rgb, estimates 6-DOF camera pose in map frame "
            "via solvePnP + known tag world positions. "
            "Publishes /xtend/april_tag_pose (PoseStamped). "
            "Tag map: tag_map_path_ALL.yaml. Calibration: 504x294."
        ),
        command="""
python3 -m sparx_agency.tasks.localization.apriltag_triangulation_node \
  --tag_map_path /home/user/GIT/TheAgency/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml \
  --camera_calib_path /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
  --tag_size_m 0.13 \
  --source ros \
  --image_topic /xtend/rgb \
  --pose_topic /xtend/april_tag_pose \
  --no_vis
""",
    ),
    LaunchItem(
        name="7. Static TF fallback (map -> xtend_camera)",
        machine="jetson",
        tmux_name="xtend_static_tf",
        description=(
            "Publishes static TF map→xtend_camera at z=1.0m (drone hover height). "
            "Rotation roll=-π/2 yaw=-π/2 aligns optical frame (X=right, Y=down, Z=forward) with map. "
            "Overridden dynamically by pose_to_tf_node when AprilTag is visible."
        ),
        command="""
ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 1.0 \
  --roll -1.5708 --pitch 0 --yaw -1.5708 \
  --frame-id map \
  --child-frame-id xtend_camera
""",
    ),
    LaunchItem(
        name="8. Pose-to-TF bridge (april_tag_pose -> TF)",
        machine="jetson",
        tmux_name="xtend_pose_to_tf",
        description=(
            "Subscribes to /xtend/april_tag_pose, applies body→optical frame correction "
            "(q_opt = q_body ⊗ (-0.5, 0.5, -0.5, 0.5)), broadcasts dynamic TF map→xtend_camera. "
            "Overrides static fallback while tags are visible; TF2 falls back after ~10s without tags."
        ),
        command="""
python3 -m sparx_agency.tasks.mapping.ros2.pose_to_tf_node \
  --ros-args \
  -p pose_topic:=/xtend/april_tag_pose
""",
    ),
    LaunchItem(
        name="9. Octomap server (3D + 2D occupancy)",
        machine="jetson",
        tmux_name="xtend_octomap",
        description=(
            "Subscribes to /xtend/pointcloud (xtend_camera frame), raycasts using TF map→xtend_camera. "
            "Publishes /occupied_cells_vis_array (3D voxels) and /projected_map (2D occupancy grid). "
            "Resolution 5cm, min_z=0.1m clips floor from 2D projection."
        ),
        command="""
ros2 run octomap_server octomap_server_node \
  --ros-args \
  -p resolution:=0.05 \
  -p frame_id:=map \
  -p sensor_model/max_range:=5.0 \
  -p sensor_model/hit:=0.70 \
  -p sensor_model/miss:=0.40 \
  -p occupancy_thres:=0.50 \
  -p occupancy_min_z:=0.1 \
  -p occupancy_max_z:=5.0 \
  -p filter_ground:=false \
  --remap cloud_in:=/xtend/pointcloud
""",
    ),
    LaunchItem(
        name="10. PC: RViz mapping view",
        machine="pc",
        tmux_name="xtend_pc_rviz",
        description="Opens RViz with the mapping config on the PC side (requires ROS_DOMAIN_ID=5).",
        command="rviz2 -d /home/user1/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/rgbd_mapping.rviz",
    ),
    LaunchItem(
        name="11. PC manual UI",
        machine="pc",
        tmux_name="xtend_pc_ui",
        description="Manual ARM/TAKEOFF/LAND/DISARM/STOP UI. Movement can publish Twist.",
        command="python3 /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/ui.py",
    ),
    LaunchItem(
        name="12. Planner: hospital world",
        machine="manual",
        tmux_name="planner_hospital",
        description="Manual planner step on Jetson/container: starts hospital environment.",
        enabled_by_default=False,
        command="""
cd /home/user/GIT/sjtu_project/falcon_docker
./run_hospital.sh office
""",
    ),
    LaunchItem(
        name="13. Planner container: FALCON adapter",
        machine="manual",
        tmux_name="planner_falcon",
        description="Run inside the planner container.",
        enabled_by_default=False,
        command="roslaunch falcon_adapter real_drone.launch map_name:=office",
    ),
    LaunchItem(
        name="14. Planner ROS bridge docker",
        machine="manual",
        tmux_name="planner_ros_bridge",
        description="Manual ROS bridge step.",
        enabled_by_default=False,
        command="""
cd /home/user/GIT/sjtu_project/ros_bridge_docker
./run_bridge.sh
""",
    ),
    LaunchItem(
        name="15. FALCON RViz display",
        machine="manual",
        tmux_name="planner_rviz",
        description="Optional display command inside falcon container.",
        enabled_by_default=False,
        command="docker exec -it falcon bash -lc 'roslaunch exploration_manager rviz.launch'",
    ),
    LaunchItem(
        name="16. BEV click goal UI",
        machine="manual",
        tmux_name="planner_bev_goal",
        description="Optional click-goal command inside falcon container.",
        enabled_by_default=False,
        command="docker exec -it falcon bash -lc 'rosrun falcon_adapter bev_click_goal.py'",
    ),
]


def normalize_command(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def wrap_with_env(machine: str, command: str) -> str:
    env = JETSON_ENV if machine == "jetson" else PC_ENV
    if machine == "jetson":
        command = command.replace("/home/user/GIT/TheAgency", JETSON_REPO)
    return normalize_command(env) + "\n" + normalize_command(command)


class XtendPipelineLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XTEND Pipeline Launcher — RGBD Mapping")
        self.geometry("1260x780")
        self.jetson_ssh_var = tk.StringVar(value=JETSON_SSH_DEFAULT)
        self.status_var = tk.StringVar(value="Ready.")
        self.selected_item: LaunchItem | None = None
        self.item_vars: dict[str, tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Jetson SSH:").pack(side="left")
        ttk.Entry(top, textvariable=self.jetson_ssh_var, width=28).pack(side="left", padx=6)
        ttk.Button(top, text="Start selected", command=self.start_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Stop selected tmux", command=self.stop_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Start checked Jetson core", command=self.start_checked_jetson).pack(side="left", padx=4)
        ttk.Button(top, text="AUTO full mapping pipeline", command=self.start_auto_pipeline).pack(side="left", padx=4)
        ttk.Button(top, text="Stop all known tmux", command=self.stop_all_known).pack(side="left", padx=4)

        mode_row = ttk.Frame(self)
        mode_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(mode_row, text="Demo mode:").pack(side="left")
        ttk.Button(mode_row, text="IDLE", command=lambda: self.publish_demo_mode("idle")).pack(side="left", padx=4)
        ttk.Button(mode_row, text="FLY_STRAIGHT", command=lambda: self.publish_demo_mode("fly_straight")).pack(side="left", padx=4)
        ttk.Button(mode_row, text="TURNING", command=lambda: self.publish_demo_mode("turning")).pack(side="left", padx=4)
        ttk.Button(mode_row, text="VISUAL_SERVOING", command=lambda: self.publish_demo_mode("visual_servoing")).pack(side="left", padx=4)
        ttk.Button(mode_row, text="FINISH", command=lambda: self.publish_demo_mode("finish", confirm=True)).pack(side="left", padx=4)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=6)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        ttk.Label(left, text="Nodes / Commands", font=("Arial", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, height=30, exportselection=False)
        self.listbox.pack(fill="both", expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        for item in LAUNCH_ITEMS:
            self.listbox.insert("end", item.name)
            self.item_vars[item.tmux_name] = tk.BooleanVar(value=item.enabled_by_default)

        checks = ttk.LabelFrame(left, text="Checked for batch start")
        checks.pack(fill="x", pady=6)
        for item in LAUNCH_ITEMS:
            if item.machine == "jetson":
                ttk.Checkbutton(checks, text=item.name, variable=self.item_vars[item.tmux_name]).pack(anchor="w")

        self.desc_text = tk.Text(right, height=5, wrap="word")
        self.desc_text.pack(fill="x", pady=(0, 6))
        ttk.Label(right, text="Command", font=("Arial", 11, "bold")).pack(anchor="w")
        self.cmd_text = tk.Text(right, height=24, wrap="none")
        self.cmd_text.pack(fill="both", expand=True)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Copy command", command=self.copy_command).pack(side="left", padx=4)
        ttk.Button(btns, text="Copy env + command", command=self.copy_full_command).pack(side="left", padx=4)
        ttk.Button(btns, text="Run local terminal", command=self.run_local_terminal).pack(side="left", padx=4)
        ttk.Button(btns, text="Run Jetson tmux over SSH", command=self.run_jetson_tmux).pack(side="left", padx=4)
        ttk.Button(btns, text="Copy tmux attach command", command=self.copy_attach_command).pack(side="left", padx=4)

        ttk.Label(
            right,
            text="Safety: AUTO sends ARM and TAKEOFF. Verify drone state manually before using it.",
            foreground="darkred",
        ).pack(anchor="w", pady=4)
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", padx=10, pady=4)
        self.listbox.selection_set(0)
        self.on_select()

    def current_index(self) -> int | None:
        sel = self.listbox.curselection()
        return int(sel[0]) if sel else None

    def on_select(self, _event=None):
        idx = self.current_index()
        if idx is None:
            return
        item = LAUNCH_ITEMS[idx]
        self.selected_item = item
        self.desc_text.delete("1.0", "end")
        self.desc_text.insert("end", f"{item.name}\nMachine: {item.machine}\nTmux: {item.tmux_name}\n\n{item.description}")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", normalize_command(item.command))

    def get_command_text(self) -> str:
        return normalize_command(self.cmd_text.get("1.0", "end"))

    def copy_to_clipboard(self, text: str, label: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set(f"Copied {label}.")

    def copy_command(self):
        self.copy_to_clipboard(self.get_command_text(), "command")

    def copy_full_command(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        full = self.get_command_text() if item.machine == "manual" else wrap_with_env(item.machine, self.get_command_text())
        self.copy_to_clipboard(full, "env + command")

    def run_local_terminal(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if item.machine != "pc" and not messagebox.askyesno("Run locally?", "This item is not marked as PC/local. Run it locally anyway?"):
            return
        script = wrap_with_env("pc", self.get_command_text())
        self._spawn_terminal(script, title=item.tmux_name)
        self.status_var.set(f"Started local terminal for {item.name}")

    def start_selected(self):
        if self.selected_item is None:
            return
        if self.selected_item.machine == "jetson":
            self.run_jetson_tmux()
        elif self.selected_item.machine == "pc":
            self.run_local_terminal()
        else:
            self.copy_command()
            messagebox.showinfo("Manual command", "This command is manual. It was copied to the clipboard.")

    def run_jetson_tmux(self):
        if self.selected_item is not None:
            self._start_jetson_tmux(self.selected_item)

    def publish_demo_mode(self, mode: str, confirm: bool = False):
        if confirm and not messagebox.askyesno(
            "Confirm demo mode change",
            f"Publish demo mode request: {mode}?\n\nFINISH will trigger stop -> land -> disarm if the manager is running.",
        ):
            return

        ssh_target = self.jetson_ssh_var.get().strip()
        mode_json = json.dumps({"mode": mode, "source": "launcher_ui_manual", "reason": "manual mode button"})
        payload = f"{{data: '{mode_json}'}}"
        remote_cmd = (
            f"cd {JETSON_REPO} && "
            "source /opt/ros/humble/setup.bash && "
            f"source {JETSON_REPO}/venv/bin/activate && "
            "export ROS_DOMAIN_ID=5 && "
            f"ros2 topic pub --once /xtend/demo_mode_request std_msgs/msg/String {shlex.quote(payload)}"
        )
        result = subprocess.run(["ssh", ssh_target, remote_cmd], text=True, capture_output=True, check=False)
        if result.returncode != 0:
            messagebox.showerror("Mode publish failed", result.stderr.strip() or result.stdout.strip() or f"Failed to publish mode: {mode}")
            self.status_var.set(f"Failed to publish demo mode: {mode}")
            return
        self.status_var.set(f"Published demo mode request: {mode}")

    def start_auto_pipeline(self):
        if not messagebox.askyesno(
            "Start AUTO full mapping pipeline?",
            "This will start bridge/depth/cloud, send ARM and TAKEOFF, wait 30 seconds, "
            "then start AprilTag localization, octomap, and planner. Continue?",
        ):
            return
        ssh_target = self.jetson_ssh_var.get().strip()
        script = wrap_with_env("jetson", AUTO_PIPELINE_SCRIPT)
        debug_script = f"""
set +e
{script}
status=$?
echo
echo "[AUTO] exited with status $status"
echo "Press Enter to close this tmux session..."
read
"""
        tmux_cmd = f"bash -lc {shlex.quote(debug_script)}"
        remote_cmd = (
            "tmux kill-session -t xtend_auto_launch 2>/dev/null || true; "
            f"tmux new-session -d -s xtend_auto_launch {shlex.quote(tmux_cmd)}; "
            "tmux set-option -t xtend_auto_launch remain-on-exit on; "
            "tmux ls"
        )
        result = subprocess.run(["ssh", ssh_target, remote_cmd], text=True, capture_output=True, check=False)
        if result.returncode != 0:
            messagebox.showerror("AUTO launch failed", result.stderr.strip() or result.stdout.strip())
            self.status_var.set("AUTO mapping pipeline failed to start")
            return
        self.status_var.set("AUTO mapping pipeline started in tmux: xtend_auto_launch")

    def start_checked_jetson(self):
        count = 0
        for item in LAUNCH_ITEMS:
            if item.machine == "jetson" and self.item_vars[item.tmux_name].get():
                self._start_jetson_tmux(item, quiet=True)
                count += 1
        self.status_var.set(f"Started {count} Jetson tmux sessions.")

    def _start_jetson_tmux(self, item: LaunchItem, quiet: bool = False):
        if item.machine != "jetson":
            if not quiet:
                messagebox.showwarning("Not Jetson", "This item is not a Jetson command.")
            return
        ssh_target = self.jetson_ssh_var.get().strip()
        script = wrap_with_env("jetson", item.command)
        tmux_cmd = f"bash -lc {shlex.quote(script)}"
        remote_cmd = (
            f"tmux kill-session -t {shlex.quote(item.tmux_name)} 2>/dev/null || true; "
            f"tmux new-session -d -s {shlex.quote(item.tmux_name)} {shlex.quote(tmux_cmd)}; "
            "tmux ls"
        )
        result = subprocess.run(["ssh", ssh_target, remote_cmd], text=True, capture_output=True, check=False)
        if result.returncode != 0:
            messagebox.showerror("SSH/tmux failed", result.stderr.strip() or result.stdout.strip())
            self.status_var.set(f"Failed to start {item.tmux_name}")
            return
        if not quiet:
            self.status_var.set(f"Started Jetson tmux session: {item.tmux_name}")

    def stop_selected(self):
        if self.selected_item is None:
            return
        self._stop_tmux(self.selected_item.tmux_name)

    def stop_all_known(self):
        for item in LAUNCH_ITEMS:
            if item.machine == "jetson":
                self._stop_tmux(item.tmux_name, quiet=True)
        for extra in ("xtend_auto_launch", "planner_falcon"):
            self._stop_tmux(extra, quiet=True)
        self.status_var.set("Requested stop for all known Jetson tmux sessions.")

    def _stop_tmux(self, tmux_name: str, quiet: bool = False):
        ssh_target = self.jetson_ssh_var.get().strip()
        remote_cmd = (
            f"tmux has-session -t {shlex.quote(tmux_name)} 2>/dev/null && "
            f"tmux send-keys -t {shlex.quote(tmux_name)} C-c && sleep 1 && "
            f"tmux kill-session -t {shlex.quote(tmux_name)} 2>/dev/null || true"
        )
        subprocess.run(["ssh", ssh_target, remote_cmd], check=False)
        if not quiet:
            self.status_var.set(f"Stopped Jetson tmux session: {tmux_name}")

    def copy_attach_command(self):
        if self.selected_item is None:
            return
        ssh_target = self.jetson_ssh_var.get().strip()
        cmd = f"ssh -t {ssh_target} 'tmux attach -t {self.selected_item.tmux_name}'"
        self.copy_to_clipboard(cmd, "tmux attach command")

    def _spawn_terminal(self, script: str, title: str):
        candidates = [
            ["gnome-terminal", "--title", title, "--", "bash", "-lc", script],
            ["xterm", "-T", title, "-e", f"bash -lc {shlex.quote(script)}"],
            ["konsole", "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-lc", script],
        ]
        last_error = None
        for cmd in candidates:
            try:
                subprocess.Popen(cmd)
                return
            except FileNotFoundError as exc:
                last_error = exc
        raise RuntimeError(f"No supported terminal found. Last error: {last_error}")


def main():
    app = XtendPipelineLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
