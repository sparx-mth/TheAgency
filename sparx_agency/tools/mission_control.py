"""
Mission Control — Unified launcher & monitor for AGX1 (192.0.0.89)

Run:
    streamlit run sparx_agency/tools/mission_control.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JETSON_SSH   = "user@192.0.0.89"
JETSON_REPO  = "/home/user/agency_ws"
NANOOWL_REPO = "/home/user/GIT/NanoLLM_VILA_and_OWL"
JETSON_DATA  = "/home/user/jetson-containers/data"
VLLM_IP      = "192.0.0.89"

_ROS_ENV = f"""
source /opt/ros/humble/setup.bash
source {JETSON_REPO}/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
cd {JETSON_REPO}
"""

_DEPTH_WS_ENV = """
source /opt/ros/humble/setup.bash
source /home/user/depth_anything_ws/install/setup.bash
export PYTHONUNBUFFERED=1
"""

_NANOOWL_ENV = f"cd {NANOOWL_REPO}"

# ROBOTICAN container (docker compose service `it`, network_mode: host)
ROOSTER_CONTAINER   = "it"
ROOSTER_DOCKER_COMPOSE = "~/rqs_iai_ws/src"
_CONTAINER_ENV = """\
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/site-packages:/home/rooster
source /opt/ros/foxy/setup.bash
source /home/rooster/workspace/install/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml
export PYTHONUNBUFFERED=1"""

# Host-side env for talking to the ROBOTICAN container over network_mode: host.
# Must match _CONTAINER_ENV's ROS_DOMAIN_ID/RMW or the two sides never
# discover each other's topics (the plain "ros" env below is ROS_DOMAIN_ID=5,
# which is for the Jetson/XTEND stack, not ROBOTICAN).
_ROOSTER_PC_ENV = """
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1
"""

# Host-side env for XTEND PC tools (Manual UI, etc.). "ros" is Jetson-only
# (/opt/ros/humble + the Jetson venv), so PC-side XTEND services need this
# instead: this host runs ROS2 Jazzy and its own repo venv, but must still
# match the XTEND stack's ROS_DOMAIN_ID=5 to see its topics.
_XTEND_PC_ENV = """
source /opt/ros/jazzy/setup.bash
source /home/user1/GIT/TheAgency/.venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
"""

_ENVS = {
    "ros":       _ROS_ENV,
    "depth_ws":  _DEPTH_WS_ENV,
    "nanoowl":   _NANOOWL_ENV,
    "none":      "",
    "container": _CONTAINER_ENV,
    "rooster_pc": _ROOSTER_PC_ENV,
    "xtend_pc":  _XTEND_PC_ENV,
}


# ---------------------------------------------------------------------------
# Service definitions
# ---------------------------------------------------------------------------
@dataclass
class Service:
    name: str
    key: str           # tmux session name / unique id
    group: str
    description: str
    cmd: str
    env: str = "ros"   # "ros" | "depth_ws" | "nanoowl" | "none" | "docker" | "container"
    docker_container: str = ""   # for env="docker" — container name to inspect
    stop_extra: str = ""         # extra stop command if needed
    machine: str = "jetson"      # "jetson" | "pc" | "container"
    container_name: str = ""     # for machine="container" — docker container to exec into
    is_interactive: bool = False  # if True, opens a terminal window (needs TTY)
    proc_container: str = ""     # container to exec into for process-based status check
    proc_pattern: str = ""       # grep pattern inside proc_container to detect running

    def log_file(self) -> str:
        return f"/tmp/{self.key}.log"


XTEND_SERVICES: list[Service] = [
    # ── Core sensing ──────────────────────────────────────────────────────
    Service(
        name="XTEND Bridge",
        key="xtend_bridge",
        group="core",
        description="XTEND WebSocket → saves frames to /tmp/xtend_frames, publishes /xtend/rgb_frame_path, /xtend/bearing",
        cmd=f"""python3 {JETSON_REPO}/sparx_agency/robots/XTEND/online_nav_bridge_dir_publisher.py \\
  --frequency 10.0 --out-dir /tmp/xtend_frames \\
  --path-topic /xtend/rgb_frame_path \\
  --preprocess-mode resize --output-width 504 --output-height 294 \\
  --debug""",
    ),
    Service(
        name="Depth DA3",
        key="xtend_depth",
        group="core",
        description="DA3 Metric Large TRT, saves NPY to /tmp/xtend_depth, publishes only the file path on /xtend/depth_frame_path (no raw depth_m message, no pointcloud).",
        cmd=f"""python3 {JETSON_REPO}/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \\
  --ros-args \\
  -p frame_path_topic:=/xtend/rgb_frame_path \\
  -p depth_topic:=/xtend/depth_m \\
  -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-294x504.depth_only.v2.engine \\
  -p config_yaml:={JETSON_REPO}/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \\
  -p camera_info_mode:=base -p model_type:=large_metric \\
  -p apply_metric_focal_scaling:=true -p metric_scale_divisor:=300.0 \\
  -p clip_min_m:=0.2 -p clip_max_m:=5.0 -p depth_encoding:=32FC1 \\
  -p depth_path_topic:=/xtend/depth_frame_path \\
  -p depth_dir:=/tmp/xtend_depth -p max_depth_kept:=300 \\
  -p depth_smoothing_window:=1""",
    ),
    Service(
        name="Demo Mode Manager",
        key="xtend_demo_mgr",
        group="core",
        description="Publishes /xtend/demo_mode. FINISH → stop/land/disarm.",
        cmd=f"""python3 {JETSON_REPO}/sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_drone_demo_manager.py \\
  --request-topic /xtend/demo_mode_request --mode-topic /xtend/demo_mode \\
  --cmd-nav-topic /xtend/cmd_nav --reset-odom-topic /xtend/reset_odom \\
  --disarm-delay-sec 8.0""",
    ),
    # ── Localization & mapping ─────────────────────────────────────────────
    Service(
        name="Localization (AprilTag)",
        key="xtend_apriltag",
        group="localization",
        description="AprilTag solvePnP → /xtend/localization (PoseStamped), /xtend/localization_source",
        cmd=f"""python3 -m sparx_agency.tasks.localization.ros2.localization_node \\
  --ros-args -p provider_type:=apriltag \\
  -p frame_path_topic:=/xtend/rgb_frame_path \\
  -p tag_map_path:={JETSON_REPO}/sparx_agency/tasks/localization/config/new_map.yaml \\
  -p camera_calib_path:={JETSON_REPO}/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \\
  -p tag_size_m:=0.13""",
    ),
    Service(
        name="Octomap",
        key="xtend_octomap",
        group="localization",
        description="/xtend/pointcloud → 3D voxels + 2D /projected_map. Resolution 5cm.",
        cmd="""ros2 run octomap_server octomap_server_node \\
  --ros-args -p resolution:=0.05 -p frame_id:=map \\
  -p sensor_model/max_range:=5.0 -p sensor_model/hit:=0.70 -p sensor_model/miss:=0.40 \\
  -p occupancy_min_z:=0.1 -p occupancy_max_z:=5.0 -p filter_ground:=false \\
  --remap cloud_in:=/xtend/pointcloud""",
    ),
    # ── Dome capture ──────────────────────────────────────────────────────
    Service(
        name="Dome Main",
        key="xtend_dome",
        group="dome",
        description="360° sweep: arm → takeoff → rotate 90°×4 (AprilTag guided) → land. Saves RGB+depth+pose.",
        cmd=f"""python3 {JETSON_REPO}/sparx_agency/demos/Demo_No4_XTEND_MapRoom/xtend_dome_main.py \\
  --pose-topic /xtend/localization \\
  --depth-topic /xtend/depth_frame_path \\
  --out-dir {JETSON_DATA}/captures""",
    ),
    # ── Room Mapper (offline post-flight) ────────────────────────────────
    Service(
        name="Room Mapper",
        key="room_mapper",
        group="room_mapper",
        description="Post-flight: build occupancy map + place VLM-detected objects from latest session.",
        cmd=f"""cp {JETSON_DATA}/captures/latest_ann/*.json {JETSON_DATA}/captures/latest/ 2>/dev/null; \\
python3 -m sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.run_room_mapper \\
  --data-dir {JETSON_DATA}/captures/latest \\
  --tag-map {JETSON_REPO}/sparx_agency/tasks/localization/config/new_map.yaml \\
  --stride 1 \\
  --output-dir {JETSON_DATA}/captures/latest_room_map""",
    ),
    # ── Planner (Falcon) ──────────────────────────────────────────────────
    Service(
        name="Falcon Container",
        key="planner_hospital",
        group="planner",
        description="Start falcon-ros Docker container (office map). Must be running before Falcon Adapter.",
        cmd=f"cd {JETSON_REPO}/sparx_agency/tasks/planning/falcon && ./run_falcon.sh office",
        env="docker",
        docker_container="falcon",
        stop_extra="docker rm -f falcon 2>/dev/null || true",
    ),
    Service(
        name="Falcon Adapter",
        key="planner_falcon",
        group="planner",
        description="ROS1 Falcon planner inside the falcon container (requires falcon container running).",
        cmd="docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && roslaunch falcon_adapter real_drone.launch map_name:=office'",
        env="none",
        proc_container="falcon",
        proc_pattern="real_drone.launch",
        stop_extra="docker exec falcon bash -lc 'pkill -f real_drone.launch || true; pkill -f roslaunch || true' 2>/dev/null || true",
    ),
    Service(
        name="ROS1↔ROS2 Bridge",
        key="planner_ros_bridge",
        group="planner",
        description="ROS1↔ROS2 bridge container. Forwards /xtend/localization and cmd_vel between ROS versions.",
        cmd=f"cd {JETSON_REPO}/sparx_agency/tasks/planning/falcon/bridge && ./run_bridge.sh",
        env="docker",
        docker_container="ros1_bridge",
        stop_extra="docker rm -f ros1_bridge 2>/dev/null || true",
    ),
    Service(
        name="Falcon RViz",
        key="planner_rviz_falcon",
        group="planner",
        description="RViz inside the falcon container (optional, for visualisation).",
        cmd="docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && export DISPLAY=:0 && roslaunch exploration_manager rviz.launch'",
        env="none",
    ),
    Service(
        name="BEV Click Goal",
        key="planner_bev_goal",
        group="planner",
        description="Bird's-eye-view click-to-goal UI inside falcon container.",
        cmd="docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && export DISPLAY=:0 && rosrun falcon_adapter bev_click_goal_node.py'",
        env="none",
        proc_container="falcon",
        proc_pattern="bev_click_goal_node.py",
        stop_extra="docker exec falcon bash -lc 'pkill -f bev_click_goal_node.py || true' 2>/dev/null || true",
    ),
    # ── PC-side tools (run locally) ───────────────────────────────────────
    Service(
        name="RViz Mapping",
        key="pc_rviz",
        group="pc",
        description="Opens RViz with the RGBD mapping config. Requires ROS_DOMAIN_ID=5 locally.",
        cmd="rviz2 -d /home/user1/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/rgbd_mapping.rviz",
        env="ros",
        machine="pc",
        proc_pattern="rgbd_mapping.rviz",
    ),
    Service(
        name="Manual UI",
        key="pc_manual_ui",
        group="pc",
        description="ARM / TAKEOFF / LAND / DISARM / STOP UI + optional Twist publishing.",
        cmd="python3 /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/ui.py",
        env="xtend_pc",
        machine="pc",
        proc_pattern="robots/XTEND/ui.py",
    ),
]

NANOOWL_SERVICES: list[Service] = [
    Service(
        name="vLLM (Qwen3-VL-4B)",
        key="vllm_server",
        group="nanoowl",
        description="Vision-language API at :8080. Quantised Qwen3-VL-4B-Instruct.",
        env="docker",
        docker_container="vllm_server",
        cmd=(
            "docker rm -f vllm_server >/dev/null 2>&1 || true; "
            "docker run --name vllm_server "
            "--runtime nvidia --network host "
            "-v ~/my_models/qwen3-vl-4b:/app/model "
            "-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "
            "vllm_qwen3_vl_4b_instruct_aws_4bit:latest "
            f"vllm serve /app/model --host {VLLM_IP} --port 8080 "
            "--dtype float16 --gpu-memory-utilization 0.5 --max-model-len 2048 --enforce-eager"
        ),
        stop_extra="docker rm -f vllm_server >/dev/null 2>&1 || true",
    ),
    Service(
        name="NanoOWL",
        key="nanoowl_service",
        group="nanoowl",
        description="Open-vocabulary object detector at :5060. Min-score 0.2.",
        env="docker",
        docker_container="nanoowl_service",
        cmd=(
            "docker rm -f nanoowl_service >/dev/null 2>&1 || true; "
            "docker run --name nanoowl_service "
            "--runtime nvidia --network host --ipc=host "
            "-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all "
            "nanoowl_new:v1.5 "
            "python3 /opt/nanoowl/examples/jetson_server/nanoowl_service.py "
            f"--engine /opt/nanoowl/data/owl_image_encoder_patch32.engine "
            f"--host {VLLM_IP} --port 5060 --min-score 0.2"
        ),
        stop_extra="docker rm -f nanoowl_service >/dev/null 2>&1 || true",
    ),
    Service(
        name="Depth V3 HTTP",
        key="depth_v3_http",
        group="nanoowl_optional",
        description="On-demand DA3 depth at :5070 for comm_manager's per-frame --depth-endpoint. "
                     "One request = one frame run fresh through DA3 — no ROS topics, no timestamp "
                     "pairing, so no sync/staleness/smoothing-ghosting is possible.",
        env="ros",
        cmd=(
            f"python3 {JETSON_REPO}/sparx_agency/tasks/mapping/da3_depth_http_server.py "
            "--engine /home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE.fp16-294x504.depth_only.v2.engine "
            f"--calib {JETSON_REPO}/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml "
            f"--host 0.0.0.0 --port 5070"
        ),
        stop_extra="fuser -k 5070/tcp || true",
    ),
    Service(
        name="Data Server",
        key="data_server",
        group="nanoowl",
        description="HTTP file server at :9000 serving jetson-containers/data (images for vLLM URL refs).",
        env="none",
        cmd=f"cd {JETSON_DATA} && python3 -m http.server 9000 --bind {VLLM_IP}",
        stop_extra="fuser -k 9000/tcp || true",
    ),
    Service(
        name="Display Server",
        key="display_server",
        group="nanoowl",
        description="Web gallery at :8090. Browse latest captures with VLM captions.",
        env="nanoowl",
        cmd=f"python3 display_server.py --root {JETSON_DATA}/captures --host {VLLM_IP} --port 8090 --latest-only",
        stop_extra="fuser -k 8090/tcp || true",
    ),
    Service(
        name="Comm Manager",
        key="comm_manager",
        group="nanoowl",
        description="Watches captures root, calls vLLM + OWL per new image, writes results.",
        env="nanoowl",
        cmd=(
            f"python3 comm_manager_vllm.py --profile xtend "
            f"--vllm-model espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16 "
            f"--captures-root {JETSON_DATA}/captures/ "
            f"--endpoint http://{VLLM_IP}:8080 --host {VLLM_IP} --force"
        ),
        stop_extra="fuser -k 5050/tcp || true",
    ),
]

_ROOSTER_CTRL = (
    "/home/rooster/sparx_agency/robots/ROBOTICAN/examples/src/position_fly_controller.py"
)

ROBOTICAN_SERVICES: list[Service] = [
    # ── Core controller ───────────────────────────────────────────────────────
    Service(
        name="Position Fly Controller",
        key="rooster_position_ctrl",
        group="rooster_core",
        description=(
            "Keyboard controller — POSITION mode (flight_mode=3).\n"
            "f=arm+takeoff  w/s=fwd/back  j/l=strafe  i/k=up/dn  "
            "a/d=yaw  h=hover-lock  e=disarm  p=path  q=quit"
        ),
        cmd=(
            f"python3 {_ROOSTER_CTRL} \\\n"
            "  --ros-args \\\n"
            "  -p rooster_ids:=R1 \\\n"
            "  -p step:=50.0 \\\n"
            "  -p climb_z:=700.0 \\\n"
            "  -p hover_z:=700.0 \\\n"
            "  -p climb_duration_sec:=5.0 \\\n"
            "  -p log_dir:=/tmp"
        ),
        env="container",
        machine="container",
        container_name=ROOSTER_CONTAINER,
        is_interactive=True,
        proc_pattern="position_fly_controller",
    ),
    # ── Command unit + UI ────────────────────────────────────────────────────
    # RoosterCommandUnit is the single gateway that talks to the FCU for one
    # drone: it owns arm/disarm/takeoff/land/manual-move and listens on
    # /<id>/cmd_nav (JSON). The manual UI below is just one publisher on that
    # topic — a planner node can publish the same JSON commands to drive the
    # same drone through the same gateway.
    Service(
        name="Rooster Command Unit (R1)",
        key="rooster_command_unit_R1",
        group="rooster_core",
        description=(
            "Single command gateway for R1 — owns arm/disarm/takeoff/land/move.\n"
            "Listens on /R1/cmd_nav (String JSON), publishes /R1/rooster_status.\n"
            "Both the manual UI and a future planner talk to this one node."
        ),
        cmd=(
            "python3 /home/rooster/sparx_agency/robots/ROBOTICAN/adapters/rooster_command_unit.py \\\n"
            "  --ros-args \\\n"
            "  -p rooster_id:=R1 \\\n"
            "  -p climb_z:=700.0 \\\n"
            "  -p hover_z:=700.0 \\\n"
            "  -p climb_duration_sec:=5.0"
        ),
        env="container",
        machine="container",
        container_name=ROOSTER_CONTAINER,
        proc_pattern="rooster_command_unit",
    ),
    Service(
        name="Rooster Manual UI (R1)",
        key="rooster_manual_ui_R1",
        group="rooster_core",
        description=(
            "ARM / TAKEOFF / LAND / DISARM + movement d-pad for R1.\n"
            "Publishes /R1/cmd_nav — requires Rooster Command Unit (R1) running."
        ),
        cmd="python3 /home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/ui.py --ros-args -p rooster_id:=R1",
        env="rooster_pc",
        machine="pc",
        proc_pattern="robots/ROBOTICAN/ui.py",
    ),
    # ── Vision (host, domain 9) ──────────────────────────────────────────────
    # Unlike XTEND (Jetson-hosted), ROBOTICAN's whole vision stack runs on
    # this PC — frame capture needs the container's network_mode: host UDP
    # port, and this PC also has its own DA3-TRT engine, so depth doesn't
    # need the Jetson either. Both scripts already set ROS_DOMAIN_ID=9 +
    # CycloneDDS + CYCLONEDDS_URI themselves (see the scripts) — don't
    # replicate that here via `env=`, just call them directly.
    Service(
        name="Rooster Frame Capture (R1)",
        key="rooster_frame_capture_R1",
        group="rooster_vision",
        description="Decodes R1's UDP/RTP-H264 stream, saves JPEGs to /tmp/rooster_frames, publishes /R1/rgb_frame_path.",
        cmd="bash /home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/run_rooster_frame_dir_publisher.sh",
        env="none",
        machine="pc",
        proc_pattern="rooster_frame_dir_publisher",
    ),
    Service(
        name="Rooster Depth Processor (R1)",
        key="rooster_depth_R1",
        group="rooster_vision",
        description="DA3-TRT depth from /R1/rgb_frame_path, publishes /R1/depth_frame_path. Requires Rooster Frame Capture (R1) running.",
        cmd="bash /home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/run_depth_processor.sh",
        env="none",
        machine="pc",
        proc_pattern="rooster_depth_processor",
    ),
    Service(
        name="Rooster Video Trigger (R1)",
        key="rooster_video_trigger_R1",
        group="rooster_vision",
        description=(
            "Calls /R1/video_handler/set_video_mode (playing=True) and holds "
            "gcs_keep_alive so R1 actually streams its camera to this PC:5001. "
            "Without this, video_handler never sends a single frame and "
            "Rooster Frame Capture (R1) silently sits empty — must run before it."
        ),
        cmd=(
            "bash /home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/run_video_trigger.sh "
            "--drone-id R1 --host-ip 127.0.0.1 --port 5001 --width 540 --height 360"
        ),
        env="none",
        machine="pc",
        proc_container="it",
        proc_pattern="video_trigger.py",
        stop_extra="docker exec it bash -lc 'pkill -f video_trigger.py || true' 2>/dev/null || true",
    ),
    # ── Planner (Falcon), host, domain 9 — mirrors XTEND's "planner" group ──
    Service(
        name="Rooster Falcon Container",
        key="rooster_planner_falcon",
        group="rooster_planner",
        description="Start falcon-ros Docker container (sphera_jail map) for ROBOTICAN. Must be running before Rooster Falcon Adapter.",
        # run_falcon_sphera.sh is a Rooster-only duplicate of run_falcon.sh
        # (which XTEND/colleagues also edit) -- keeps this repo's own RViz-
        # view mount addition out of a file shared with other maps/users.
        cmd="cd /home/user1/GIT/TheAgency/sparx_agency/tasks/planning/falcon && ./run_falcon_sphera.sh sphera_jail",
        env="docker",
        machine="pc",
        docker_container="falcon",
        stop_extra="docker rm -f falcon 2>/dev/null || true",
    ),
    Service(
        name="Rooster Falcon Adapter",
        key="rooster_planner_adapter",
        group="rooster_planner",
        description="ROS1 Falcon planner inside the falcon container, wired to R1's topics/camera intrinsics (requires falcon container + Rooster ROS1<->ROS2 Bridge running).",
        cmd=(
            "docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
            "source /catkin_ws/devel/setup.bash && roslaunch falcon_adapter sphera_drone.launch "
            "map_name:=sphera_jail "
            "real_pose_topic:=/R1/localization "
            "real_depth_path_topic:=/R1/depth_frame_path "
            "real_rgb_path_topic:=/R1/rgb_frame_path "
            "cam_fx:=125.903068 cam_fy:=180.0 cam_cx:=269.5 cam_cy:=179.5 "
            "cam_width:=540 cam_height:=360 "
            # Rooster's ground-truth pose is stamped independently from depth
            # (unlike XTEND, where both share one capture event) — the
            # launch default is 0.0/0.0 (exact-timestamp-only, correct for
            # XTEND) which mapping_sync can never satisfy for Rooster, so
            # every depth frame gets dropped as "dropout" despite a pose
            # arriving within ~0.5ms. These match mapping_sync_node.py's own
            # documented defaults for this exact case.
            "sync_tolerance:=0.05 max_interp_gap:=0.12 "
            # goal_x/goal_y default to (0.0, -3.0) -- correct for XTEND,
            # where the drone starts near the origin, but /gt_pose is a
            # direct passthrough of Rooster's raw Sphera-world pose
            # (confirmed live: no re-zeroing), which starts at roughly
            # (-54.75, 14.66). The unmodified default goal sat ~57m away,
            # entirely outside sphera_jail.yaml's own box bounds, so
            # astar_planner could never find a reachable target. This is
            # the same 3m-south-of-spawn offset the default intended,
            # just expressed in Rooster's actual world coordinates.
            "goal_x:=-54.75 goal_y:=11.66 "
            # bev_xmin/ymin/xmax/ymax default to a 12x12m box at world
            # origin (0,0) -- correct for XTEND's near-origin office/hospital
            # maps, but Rooster's real spawn is ~55m away at (-54.75, 14.66),
            # so the BEV click map (and every click on it) was pointed at an
            # empty patch of world far from the drone. These match
            # sphera_jail.yaml's doubled box bounds, centered on the real
            # spawn -- this is why the adapter launches sphera_drone.launch
            # (a Sphera-only fork of real_drone.launch) instead: only that
            # file forwards bev_* to nav_stack.launch.
            "bev_xmin:=-70.75 bev_ymin:=-1.34 bev_xmax:=-38.75 bev_ymax:=30.66'"
        ),
        env="none",
        machine="pc",
        proc_container="falcon",
        proc_pattern="real_drone.launch",
        stop_extra="docker exec falcon bash -lc 'pkill -f real_drone.launch || true; pkill -f roslaunch || true' 2>/dev/null || true",
    ),
    Service(
        name="Rooster ROS1<->ROS2 Bridge",
        key="rooster_planner_bridge",
        group="rooster_planner",
        description="ROS1<->ROS2 bridge container for R1 — CycloneDDS/domain 9 (Rooster is Jazzy, not XTEND's Foxy/FastRTPS). Forwards /R1/localization, /R1/rgb_frame_path, /R1/depth_frame_path, cmd_vel.",
        cmd=(
            "cd /home/user1/GIT/TheAgency/sparx_agency/tasks/planning/falcon/bridge && "
            "ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            "CYCLONEDDS_URI=file:///home/user1/rqs_iai_ws/src/cyclonedds.xml ./run_bridge.sh"
        ),
        env="docker",
        machine="pc",
        docker_container="ros1_bridge",
        stop_extra="docker rm -f ros1_bridge 2>/dev/null || true",
    ),
    # RViz/BEV-click below mirror the XTEND "planner" group's identically
    # named services, but XTEND's are Jetson-hosted (machine defaults to
    # "jetson", checked/started over SSH) because XTEND's whole vision stack
    # lives on the Jetson. Rooster's falcon container runs on this PC (see
    # the vision-group note above), so these must be machine="pc" +
    # proc_container="falcon" like Rooster Falcon Adapter, not SSH'd.
    Service(
        name="Rooster Falcon RViz",
        key="rooster_planner_rviz",
        group="rooster_planner",
        description="RViz inside the falcon container (host-side), visualising R1's occupancy map/plan.",
        # DISPLAY/XAUTHORITY are already set correctly in the container's own
        # environment (run_falcon_sphera.sh passes through the host's actual
        # $DISPLAY at `docker run` time) -- do NOT override with :0 here;
        # confirmed live that :0 is wrong on this PC (the container's real
        # DISPLAY was :1) and silently fails with "couldn't connect to
        # display" instead of erroring loudly.
        cmd=(
            "docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
            "source /catkin_ws/devel/setup.bash && "
            "roslaunch exploration_manager rviz.launch'"
        ),
        env="none",
        machine="pc",
        proc_container="falcon",
        proc_pattern="rviz.launch",
        stop_extra=(
            "docker exec falcon bash -lc 'pkill -f rviz.launch || true; "
            "pkill -f lib/rviz/rviz || true' 2>/dev/null || true"
        ),
    ),
    Service(
        name="Rooster BEV Click Goal",
        key="rooster_planner_bev_goal",
        group="rooster_planner",
        description="Bird's-eye-view click-to-goal UI inside the falcon container (host-side) for R1.",
        # See the DISPLAY note on Rooster Falcon RViz above -- same reasoning.
        cmd=(
            "docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && "
            "source /catkin_ws/devel/setup.bash && "
            "rosrun falcon_adapter bev_click_goal_node.py'"
        ),
        env="none",
        machine="pc",
        proc_container="falcon",
        proc_pattern="bev_click_goal_node.py",
        stop_extra="docker exec falcon bash -lc 'pkill -f bev_click_goal_node.py || true' 2>/dev/null || true",
    ),
    # ── Monitors ──────────────────────────────────────────────────────────────
    Service(
        name="State Monitor (R1)",
        key="rooster_state_R1",
        group="rooster_monitor",
        description="Streams /R1/state — armed, flight_mode, airborne, roll/pitch/azimuth.",
        cmd="ros2 topic echo /R1/state",
        env="container",
        machine="container",
        container_name=ROOSTER_CONTAINER,
        proc_pattern="topic echo /R1/state",
    ),
    Service(
        name="KeepAlive Hz (R1)",
        key="rooster_keepalive_hz",
        group="rooster_monitor",
        description="Publish rate of /R1/keep_alive. Expected ~1 Hz.",
        cmd="ros2 topic hz /R1/keep_alive",
        env="container",
        machine="container",
        container_name=ROOSTER_CONTAINER,
        proc_pattern="topic hz /R1/keep_alive",
    ),
    Service(
        name="ManualControl Hz (R1)",
        key="rooster_manual_hz",
        group="rooster_monitor",
        description="Publish rate of /R1/manual_control. Expected ~40 Hz.",
        cmd="ros2 topic hz /R1/manual_control",
        env="container",
        machine="container",
        container_name=ROOSTER_CONTAINER,
        proc_pattern="topic hz /R1/manual_control",
    ),
]

ALL_SERVICES = XTEND_SERVICES + NANOOWL_SERVICES + ROBOTICAN_SERVICES


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------
def _ssh(cmd: str, timeout: int = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=4", JETSON_SSH, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def fetch_remote_file(remote_path: str, timeout: int = 15) -> str | None:
    """scp a file from the Jetson to a local temp path. Returns the local path, or None on failure."""
    local_path = os.path.join(tempfile.gettempdir(), "mc_" + os.path.basename(remote_path))
    try:
        r = subprocess.run(
            ["scp", "-o", "ConnectTimeout=5", f"{JETSON_SSH}:{remote_path}", local_path],
            capture_output=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None
        return local_path
    except Exception:
        return None


def get_all_states() -> dict[str, bool]:
    """Returns {service_key: is_running} for all services."""
    states: dict[str, bool] = {}

    # ── Jetson services — single SSH call ────────────────────────────────────
    # machine="pc" is excluded here even when it has proc_container/env=="docker"
    # set (e.g. Rooster Falcon Container/Adapter, which docker-run/exec on THIS
    # PC, not the Jetson) — those are checked locally below instead. Before this
    # split, any proc_container+proc_pattern service was checked via SSH to the
    # Jetson unconditionally, so a PC-hosted "falcon" container always read as
    # not-running even while genuinely up (confirmed live: real_drone.launch and
    # all its child nodes running fine inside the container, dashboard still red).
    jetson_proc_containers = {
        svc.proc_container for svc in ALL_SERVICES
        if svc.proc_pattern and svc.proc_container and svc.machine != "pc"
    }
    proc_exec_cmds = "".join(
        f"echo '=PROCS:{c}='; docker exec {c} ps -eo args 2>/dev/null || true; "
        for c in sorted(jetson_proc_containers)
    )
    cmd = (
        "echo '=TMUX='; tmux ls 2>/dev/null | cut -d: -f1; "
        "echo '=DOCKER='; docker ps --format '{{.Names}}' 2>/dev/null; "
        + proc_exec_cmds
    )
    try:
        result = _ssh(cmd)
        tmux_sessions: set[str] = set()
        docker_containers: set[str] = set()
        container_procs: dict[str, list[str]] = {c: [] for c in jetson_proc_containers}
        section = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line == "=TMUX=":
                section = "tmux"
            elif line == "=DOCKER=":
                section = "docker"
            elif line.startswith("=PROCS:") and line.endswith("="):
                section = f"procs:{line[7:-1]}"
            elif section == "tmux" and line:
                tmux_sessions.add(line)
            elif section == "docker" and line:
                docker_containers.add(line)
            elif section and section.startswith("procs:") and line:
                container_procs.setdefault(section[6:], []).append(line)
    except Exception:
        tmux_sessions = set()
        docker_containers = set()
        container_procs = {}

    # ── PC services — local docker ps / docker exec (no SSH) ─────────────────
    pc_proc_containers = {
        svc.proc_container for svc in ALL_SERVICES
        if svc.proc_pattern and svc.proc_container and svc.machine == "pc"
    }
    pc_docker_containers: set[str] = set()
    pc_container_procs: dict[str, list[str]] = {c: [] for c in pc_proc_containers}
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        pc_docker_containers = {line.strip() for line in r.stdout.splitlines() if line.strip()}
    except Exception:
        pass
    for c in pc_proc_containers:
        try:
            r = subprocess.run(
                ["docker", "exec", c, "ps", "-eo", "args"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            pc_container_procs[c] = r.stdout.splitlines()
        except Exception:
            pc_container_procs[c] = []

    for svc in ALL_SERVICES:
        if svc.machine == "container":
            continue  # handled below
        if svc.machine == "pc":
            if svc.proc_container and svc.proc_pattern:
                procs = pc_container_procs.get(svc.proc_container, [])
                states[svc.key] = any(svc.proc_pattern in p for p in procs)
            elif svc.env == "docker":
                states[svc.key] = svc.docker_container in pc_docker_containers
            elif svc.proc_pattern:
                try:
                    r = subprocess.run(
                        ["pgrep", "-f", svc.proc_pattern],
                        capture_output=True, check=False, timeout=2,
                    )
                    states[svc.key] = r.returncode == 0
                except Exception:
                    states[svc.key] = False
            else:
                states[svc.key] = False
        elif svc.proc_pattern and svc.proc_container:
            procs = container_procs.get(svc.proc_container, [])
            states[svc.key] = any(svc.proc_pattern in p for p in procs)
        elif svc.env == "docker":
            states[svc.key] = svc.docker_container in docker_containers
        else:
            states[svc.key] = svc.key in tmux_sessions

    # ── Container services — local docker exec pgrep ─────────────────────────
    for svc in ALL_SERVICES:
        if svc.machine != "container":
            continue
        if not svc.proc_pattern:
            states[svc.key] = False
            continue
        try:
            r = subprocess.run(
                ["docker", "exec", svc.container_name, "pgrep", "-f", svc.proc_pattern],
                capture_output=True, check=False, timeout=3,
            )
            states[svc.key] = r.returncode == 0
        except Exception:
            states[svc.key] = False

    return states


def _spawn_terminal_window(cmd: str, title: str) -> None:
    """Open a new host terminal window running cmd."""
    candidates = [
        ["gnome-terminal", "--title", title, "--", "bash", "-c", cmd],
        ["xterm", "-T", title, "-e", f"bash -c {shlex.quote(cmd)}"],
        ["konsole", "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-c", cmd],
    ]
    last_exc: Exception | None = None
    for args in candidates:
        try:
            subprocess.Popen(args)
            return
        except FileNotFoundError as exc:
            last_exc = exc
    raise RuntimeError(f"No supported terminal emulator found. Last error: {last_exc}")


def start_service(svc: Service) -> str | None:
    """Start service. Returns error string or None on success."""
    if svc.machine == "pc":
        env = _ENVS.get(svc.env, "")
        if svc.proc_pattern:
            subprocess.run(
                ["pkill", "-f", svc.proc_pattern],
                capture_output=True, check=False, timeout=5,
            )
        script = f"{env}\n{svc.cmd} 2>&1 | tee {svc.log_file()}"
        # Wrapped in `script` to allocate a real PTY: some commands (e.g.
        # docker run -it, used by run_falcon.sh/run_bridge.sh) refuse to
        # start at all without one ("cannot attach stdin to a TTY-enabled
        # container because stdin is not a terminal") when launched from a
        # plain detached subprocess, which has no controlling terminal.
        # tmux would do the same job (like the Jetson/SSH path below does)
        # but isn't installed on this PC; `script` (util-linux, already
        # present) is a dependency-free stand-in.
        pty_cmd = f"script -qc {shlex.quote(script)} /dev/null"
        try:
            subprocess.Popen(
                ["bash", "-lc", pty_cmd],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            return str(exc)
        return None

    if svc.machine == "container":
        env_block = _ENVS.get(svc.env, _CONTAINER_ENV)
        full_script = f"{env_block}\n{svc.cmd} 2>&1 | tee /tmp/{svc.key}.log"
        docker_cmd = (
            f"docker exec -it {shlex.quote(svc.container_name)} "
            f"bash -c {shlex.quote(full_script)}"
        )
        try:
            _spawn_terminal_window(docker_cmd, title=svc.name)
        except Exception as exc:
            return str(exc)
        return None

    env_prefix = _ENVS.get(svc.env, "")
    full_cmd = f"{env_prefix}\n{svc.cmd} 2>&1 | tee {svc.log_file()}"
    tmux_cmd = f"bash -lc {shlex.quote(full_cmd)}"
    remote = (
        f"tmux kill-session -t {shlex.quote(svc.key)} 2>/dev/null || true; "
        f"tmux new-session -d -s {shlex.quote(svc.key)} {shlex.quote(tmux_cmd)}"
    )
    try:
        r = _ssh(remote, timeout=12)
        return r.stderr.strip() if r.returncode != 0 else None
    except Exception as exc:
        return str(exc)


def stop_service(svc: Service) -> str | None:
    """Stop service. Returns error string or None on success."""
    if svc.machine == "pc":
        # A proc_container here means the actual process runs inside a
        # docker container on this PC (e.g. Rooster Falcon Adapter) — the
        # container has its own PID namespace (no --pid=host), so a bare
        # host-level pkill can never see it; has to be pkill'd from inside
        # via docker exec, same as the machine=="container" branch below.
        try:
            if svc.proc_container and svc.proc_pattern:
                kill_cmd = f"pkill -f {shlex.quote(svc.proc_pattern)} 2>/dev/null || true"
                subprocess.run(
                    ["docker", "exec", svc.proc_container, "bash", "-c", kill_cmd],
                    capture_output=True, check=False, timeout=5,
                )
            elif svc.proc_pattern:
                subprocess.run(
                    ["pkill", "-f", svc.proc_pattern],
                    capture_output=True, check=False, timeout=5,
                )
            if svc.stop_extra:
                subprocess.run(
                    ["bash", "-lc", svc.stop_extra],
                    capture_output=True, check=False, timeout=10,
                )
        except Exception as exc:
            return str(exc)
        return None

    if svc.machine == "container":
        if not svc.proc_pattern:
            return None
        kill_cmd = f"pkill -f {shlex.quote(svc.proc_pattern)} 2>/dev/null || true"
        try:
            subprocess.run(
                ["docker", "exec", svc.container_name, "bash", "-c", kill_cmd],
                capture_output=True, check=False, timeout=5,
            )
        except Exception as exc:
            return str(exc)
        return None

    cmds = [f"tmux kill-session -t {shlex.quote(svc.key)} 2>/dev/null || true"]
    if svc.stop_extra:
        cmds.append(svc.stop_extra)
    try:
        r = _ssh("; ".join(cmds), timeout=10)
        return r.stderr.strip() if r.returncode != 0 else None
    except Exception as exc:
        return str(exc)


def get_logs(svc: Service, lines: int = 120) -> str:
    """Fetch last N lines from the service log."""
    if svc.machine == "container":
        log_cmd = f"tail -n {lines} /tmp/{svc.key}.log 2>/dev/null || echo 'No log yet: /tmp/{svc.key}.log'"
        try:
            r = subprocess.run(
                ["docker", "exec", svc.container_name, "bash", "-c", log_cmd],
                capture_output=True, text=True, timeout=8,
            )
            return r.stdout or r.stderr or "(empty)"
        except Exception as exc:
            return f"docker exec error: {exc}"

    if svc.machine == "pc":
        cmd = f"tail -n {lines} {svc.log_file()} 2>/dev/null || echo 'No log yet: {svc.log_file()}'"
        try:
            r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=5)
            return r.stdout or r.stderr or "(empty)"
        except Exception as exc:
            return f"Local exec error: {exc}"

    if svc.env == "docker":
        cmd = f"docker logs --tail {lines} {svc.docker_container} 2>&1 || echo 'No container logs'"
    else:
        cmd = f"tail -n {lines} {svc.log_file()} 2>/dev/null || echo 'No log yet: {svc.log_file()}'"
    try:
        r = _ssh(cmd, timeout=8)
        return r.stdout or r.stderr or "(empty)"
    except Exception as exc:
        return f"SSH error: {exc}"


def force_arm_rooster(drone_id: str, arm: bool) -> str | None:
    """Send force_arm service call to a Rooster drone via the container."""
    action_str = "true" if arm else "false"
    srv_cmd = (
        "source /opt/ros/foxy/setup.bash && "
        "source /home/rooster/workspace/install/setup.bash && "
        "export ROS_DOMAIN_ID=9 && "
        "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
        "export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml && "
        f"ros2 service call /{drone_id}/fcu/command/force_arm "
        f"std_srvs/srv/SetBool '{{data: {action_str}}}'"
    )
    try:
        r = subprocess.run(
            ["docker", "exec", ROOSTER_CONTAINER, "bash", "-c", srv_cmd],
            capture_output=True, text=True, timeout=10,
        )
        return r.stderr.strip() if r.returncode != 0 else None
    except subprocess.TimeoutExpired:
        return "Service call timed out (10 s)"
    except Exception as exc:
        return str(exc)


def publish_demo_mode(mode: str) -> str | None:
    payload = f"{{data: '{{\"mode\": \"{mode}\", \"source\": \"mission_control\"}}'}}"
    cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source {JETSON_REPO}/venv/bin/activate && "
        f"export ROS_DOMAIN_ID=5 && "
        f"ros2 topic pub --once /xtend/demo_mode_request std_msgs/msg/String {shlex.quote(payload)}"
    )
    try:
        r = _ssh(cmd, timeout=12)
        return r.stderr.strip() if r.returncode != 0 else None
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _service_cards(services: list[Service], states: dict[str, bool]):
    cols = st.columns(3)
    for i, svc in enumerate(services):
        running = states.get(svc.key, False)
        with cols[i % 3]:
            with st.container(border=True):
                dot = "🟢" if running else "🔴"
                st.markdown(f"**{dot} {svc.name}**")
                st.caption(svc.description)
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("▶ Start", key=f"s_{svc.key}", disabled=running, use_container_width=True):
                        with st.spinner("Starting…"):
                            err = start_service(svc)
                        if err:
                            st.error(err)
                        else:
                            time.sleep(1)
                            st.rerun()
                with b2:
                    if st.button("■ Stop", key=f"x_{svc.key}", disabled=not running, use_container_width=True):
                        stop_service(svc)
                        time.sleep(0.5)
                        st.rerun()
                with b3:
                    if st.button("📋 Logs", key=f"l_{svc.key}", use_container_width=True):
                        st.session_state.log_key = svc.key
                        st.session_state.log_text = get_logs(svc)
                        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Mission Control", page_icon="🚀", layout="wide")
st.title("🚀 Mission Control — AGX1 (192.0.0.89)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=False)
    if st.button("🔄 Refresh now"):
        st.rerun()
    st.divider()
    log_lines = st.slider("Log lines", 40, 500, 120, step=20)
    st.divider()
    st.caption(f"SSH: `{JETSON_SSH}`")

# Status summary
with st.spinner("Checking service states…"):
    states = get_all_states()

running_count = sum(states.values())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Services running", f"{running_count} / {len(ALL_SERVICES)}")
m2.metric("XTEND", f"{sum(states.get(s.key, False) for s in XTEND_SERVICES)} / {len(XTEND_SERVICES)}")
m3.metric("NanoLLM", f"{sum(states.get(s.key, False) for s in NANOOWL_SERVICES)} / {len(NANOOWL_SERVICES)}")
m4.metric("ROBOTICAN", f"{sum(states.get(s.key, False) for s in ROBOTICAN_SERVICES)} / {len(ROBOTICAN_SERVICES)}")

st.divider()

tab_xtend, tab_nanoowl, tab_rooster = st.tabs(["🚁  XTEND", "🤖  NanoLLM / OWL", "🐓  ROBOTICAN"])

# ── XTEND tab ──────────────────────────────────────────────────────────────
with tab_xtend:

    # Demo mode quick actions
    st.markdown("#### Demo Mode")
    dm_cols = st.columns(6)
    MODES = [ "fly_straight", "turning", "visual_servoing", "finish"]
    LABELS = [ "➡ FLY", "🔄 TURNING", "👁 SERVO", "🛑 FINISH"]
    for col, mode, label in zip(dm_cols, MODES, LABELS):
        with col:
            if st.button(label, use_container_width=True, key=f"dm_{mode}"):
                if mode == "finish":
                    if not st.session_state.get("finish_confirmed", False):
                        st.session_state.finish_confirmed = True
                        st.warning("Click FINISH again to confirm land + disarm.")
                        st.stop()
                    st.session_state.finish_confirmed = False
                err = publish_demo_mode(mode)
                if err:
                    st.error(f"Mode publish failed: {err}")
                else:
                    st.success(f"Published: {mode}")

    st.divider()

    st.markdown("#### Core Sensing")
    core = [s for s in XTEND_SERVICES if s.group == "core"]
    _service_cards(core, states)

    st.markdown("#### Dome Capture")
    dome = [s for s in XTEND_SERVICES if s.group == "dome"]
    _service_cards(dome, states)

    st.markdown("#### Localization & Mapping")
    loc = [s for s in XTEND_SERVICES if s.group == "localization"]
    _service_cards(loc, states)

    st.markdown("#### Room Mapper (offline)")
    room_mapper = [s for s in XTEND_SERVICES if s.group == "room_mapper"]
    _service_cards(room_mapper, states)

    map_col, objs_col = st.columns([2, 1])
    with map_col:
        if st.button("🔄 Load latest room map", key="load_room_map"):
            with st.spinner("Fetching from Jetson…"):
                st.session_state["room_map_img"] = fetch_remote_file(
                    f"{JETSON_DATA}/captures/latest_room_map/map_with_objects.png")
                st.session_state["room_map_objs"] = fetch_remote_file(
                    f"{JETSON_DATA}/captures/latest_room_map/objects.json")
        img_path = st.session_state.get("room_map_img")
        if img_path and os.path.exists(img_path):
            st.image(img_path, caption="map_with_objects.png", use_container_width=True)
        else:
            st.caption("No map loaded yet — run Room Mapper above, then click Load.")
    with objs_col:
        st.markdown("**Detected objects**")
        objs_path = st.session_state.get("room_map_objs")
        if objs_path and os.path.exists(objs_path):
            try:
                with open(objs_path) as f:
                    objects = json.load(f)
                st.dataframe(
                    [{"label": o["label"],
                      "x": round(o["position_m"]["x"], 2),
                      "y": round(o["position_m"]["y"], 2),
                      "z": round(o["position_m"]["z"], 2),
                      "tags": o.get("tag_ids", [])} for o in objects],
                    use_container_width=True, hide_index=True,
                )
            except Exception as exc:
                st.error(f"Failed to parse objects.json: {exc}")
        else:
            st.caption("—")

    with st.expander("🗺️  Planner (Falcon)", expanded=True):
        planner = [s for s in XTEND_SERVICES if s.group == "planner"]
        _service_cards(planner, states)

    with st.expander("💻  PC Tools (run locally)", expanded=False):
        pc = [s for s in XTEND_SERVICES if s.group == "pc"]
        _service_cards(pc, states)

# ── NanoLLM tab ────────────────────────────────────────────────────────────
with tab_nanoowl:
    nanoowl_running = sum(states.get(s.key, False) for s in NANOOWL_SERVICES)
    cols_top = st.columns([3, 1])
    with cols_top[1]:
        if st.button("▶▶ Start All", use_container_width=True):
            with st.spinner("Starting all NanoLLM services in order…"):
                for svc in NANOOWL_SERVICES:
                    if not states.get(svc.key, False):
                        start_service(svc)
                        time.sleep(2)
            st.rerun()

    _service_cards([s for s in NANOOWL_SERVICES if s.group == "nanoowl"], states)

    with st.expander("⚙️  Optional", expanded=False):
        _service_cards([s for s in NANOOWL_SERVICES if s.group == "nanoowl_optional"], states)

    st.caption(f"📷 Display: http://{VLLM_IP}:8090   |   📁 Data: http://{VLLM_IP}:9000")# ── ROBOTICAN tab ──────────────────────────────────────────────────────────
with tab_rooster:

    # Container status banner
    try:
        ctr_result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{ROOSTER_CONTAINER}$", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=3,
        )
        ctr_running = ROOSTER_CONTAINER in ctr_result.stdout
    except Exception:
        ctr_running = False

    if ctr_running:
        st.success(f"Container `{ROOSTER_CONTAINER}` is running.")
    else:
        st.error(
            f"Container `{ROOSTER_CONTAINER}` is **not running**. "
            f"Start it with:  `cd {ROOSTER_DOCKER_COMPOSE} && docker compose up -d it`"
        )

    # ARM / DISARM quick actions
    st.markdown("#### Drone Quick Actions")
    qa_cols = st.columns(6)
    for col, drone_id, arm in zip(
        qa_cols,
        ["R1", "R1", "R2", "R2"],
        [True, False, True, False],
    ):
        label = ("ARM" if arm else "DISARM") + f" {drone_id}"
        btn_type = "primary" if arm else "secondary"
        with col:
            if st.button(label, key=f"arm_{drone_id}_{arm}", use_container_width=True, type=btn_type):
                if arm:
                    st.session_state[f"arm_confirm_{drone_id}"] = True
                    st.warning(f"Click **ARM {drone_id}** again to confirm — drone will arm!")
                    st.stop()
                if st.session_state.pop(f"arm_confirm_{drone_id}", False) or not arm:
                    err = force_arm_rooster(drone_id, arm)
                    if err:
                        st.error(f"{label} failed: {err}")
                    else:
                        st.success(f"{label} sent.")

    st.divider()

    st.markdown("#### Core")
    _service_cards([s for s in ROBOTICAN_SERVICES if s.group == "rooster_core"], states)

    st.markdown("#### Vision")
    _service_cards([s for s in ROBOTICAN_SERVICES if s.group == "rooster_vision"], states)

    st.markdown("#### Planner (Falcon)")
    _service_cards([s for s in ROBOTICAN_SERVICES if s.group == "rooster_planner"], states)

    with st.expander("📊  Monitors", expanded=False):
        _service_cards([s for s in ROBOTICAN_SERVICES if s.group == "rooster_monitor"], states)

    with st.expander("ℹ️  Container commands", expanded=False):
        st.code(
            f"# Start container\ncd {ROOSTER_DOCKER_COMPOSE} && docker compose up -d it\n\n"
            f"# Attach interactive shell\ndocker exec -it {ROOSTER_CONTAINER} bash\n\n"
            f"# Stop container\ndocker compose -f {ROOSTER_DOCKER_COMPOSE}/docker-compose.yml stop it",
            language="bash",
        )

# ── Log viewer ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📋 Logs")

log_key: str | None = st.session_state.get("log_key")

if log_key is None:
    st.info("Click **📋 Logs** on any service card to inspect it here.")
else:
    svc_map = {s.key: s for s in ALL_SERVICES}
    svc = svc_map.get(log_key)
    if svc:
        c1, c2, c3, _ = st.columns([2, 1, 1, 4])
        with c1:
            st.caption(f"**{svc.name}** — `{svc.log_file()}`")
        with c2:
            if st.button("🔄 Refresh logs"):
                st.session_state.log_text = get_logs(svc, log_lines)
                st.rerun()
        with c3:
            if st.button("✕ Close"):
                st.session_state.log_key = None
                st.session_state.log_text = ""
                st.rerun()
        text = st.session_state.get("log_text", "")
        st.code(text[-20_000:] if text else "No log loaded.", language="bash")

# ── Auto-refresh ───────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()