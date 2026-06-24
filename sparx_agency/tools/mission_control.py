"""
Mission Control — Unified launcher & monitor for AGX1 (192.0.0.89)

Run:
    streamlit run sparx_agency/tools/mission_control.py
"""

from __future__ import annotations

import shlex
import subprocess
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
cd {JETSON_REPO}
"""

_DEPTH_WS_ENV = """
source /opt/ros/humble/setup.bash
source /home/user/depth_anything_ws/install/setup.bash
"""

_NANOOWL_ENV = f"cd {NANOOWL_REPO}"

_ENVS = {
    "ros":      _ROS_ENV,
    "depth_ws": _DEPTH_WS_ENV,
    "nanoowl":  _NANOOWL_ENV,
    "none":     "",
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
    env: str = "ros"   # "ros" | "depth_ws" | "nanoowl" | "none" | "docker"
    docker_container: str = ""   # for env="docker" — container name to inspect
    stop_extra: str = ""         # extra stop command if needed
    machine: str = "jetson"      # "jetson" | "pc"

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
  --preprocess-mode resize --output-width 504 --output-height 294""",
    ),
    Service(
        name="Depth DA3",
        key="xtend_depth",
        group="core",
        description="DA3 Metric Large TRT → /xtend/depth_m (32FC1), saves NPY to /tmp/xtend_depth, publishes /xtend/depth_frame_path",
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
  -p publish_cloud:=true -p pointcloud_topic:=/xtend/pointcloud""",
    ),
    Service(
        name="Twist Converter",
        key="xtend_twist",
        group="core",
        description="/cmd_vel Twist → /xtend/cmd_nav JSON. linear.x=0.3 m/s → thrust 400",
        cmd=f"""python3 {JETSON_REPO}/sparx_agency/robots/XTEND/adapters/xtend_twist_to_cmd_nav.py \\
  --cmd-vel-topic /cmd_vel --cmd-nav-topic /xtend/cmd_nav --timeout-sec 1.5""",
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
  -p tag_map_path:={JETSON_REPO}/sparx_agency/tasks/localization/config/tag_map_path_ALL.yaml \\
  -p camera_calib_path:={JETSON_REPO}/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \\
  -p tag_size_m:=0.13""",
    ),
    Service(
        name="Static TF",
        key="xtend_static_tf",
        group="localization",
        description="Static fallback TF map→xtend_camera at z=1.0m. Overridden when AprilTags visible.",
        cmd="""ros2 run tf2_ros static_transform_publisher \\
  --x 0 --y 0 --z 1.0 --roll -1.5708 --pitch 0 --yaw -1.5708 \\
  --frame-id map --child-frame-id xtend_camera""",
    ),
    Service(
        name="Pose → TF",
        key="xtend_pose_to_tf",
        group="localization",
        description="/xtend/localization PoseStamped → dynamic TF map→xtend_camera",
        cmd=f"""python3 -m sparx_agency.tasks.mapping.ros2.pose_to_tf_node \\
  --ros-args -p pose_topic:=/xtend/localization""",
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
        cmd=f"""python3 -m sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.run_room_mapper \\
  --data-dir {JETSON_DATA}/captures/latest \\
  --tag-map {JETSON_REPO}/sparx_agency/tasks/localization/config/new_map.yaml \\
  --stride 1 --no-scale-correction \\
  --output-dir {JETSON_DATA}/captures/latest_room_map""",
    ),
    # ── Planner (Falcon) ──────────────────────────────────────────────────
    Service(
        name="Hospital World",
        key="planner_hospital",
        group="planner",
        description="Starts hospital/office Docker environment for Falcon planner.",
        cmd="cd /home/user/GIT/sjtu_project/falcon_docker && ./run_hospital.sh office",
        env="none",
    ),
    Service(
        name="Falcon Adapter",
        key="planner_falcon",
        group="planner",
        description="ROS1 Falcon planner inside the falcon container.",
        cmd="docker exec falcon bash -lc 'roslaunch falcon_adapter real_drone.launch map_name:=office'",
        env="none",
    ),
    Service(
        name="ROS Bridge",
        key="planner_ros_bridge",
        group="planner",
        description="ROS1↔ROS2 bridge container.",
        cmd="cd /home/user/GIT/sjtu_project/ros_bridge_docker && ./run_bridge.sh",
        env="none",
    ),
    Service(
        name="Falcon RViz",
        key="planner_rviz_falcon",
        group="planner",
        description="RViz inside the falcon container.",
        cmd="docker exec falcon bash -lc 'roslaunch exploration_manager rviz.launch'",
        env="none",
    ),
    Service(
        name="BEV Click Goal",
        key="planner_bev_goal",
        group="planner",
        description="Bird's-eye-view click-to-goal UI inside falcon container.",
        cmd="docker exec falcon bash -lc 'rosrun falcon_adapter bev_click_goal.py'",
        env="none",
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
    ),
    Service(
        name="Manual UI",
        key="pc_manual_ui",
        group="pc",
        description="ARM / TAKEOFF / LAND / DISARM / STOP UI + optional Twist publishing.",
        cmd="python3 /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/ui.py",
        env="ros",
        machine="pc",
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
        description="DA3-SMALL HTTP server at :5070. Optional — only needed if comm_manager uses --depth-endpoint.",
        env="depth_ws",
        cmd=(
            "ros2 run depth_anything_v3 depth_anything_http_server "
            "--model /home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3-SMALL/DA3-SMALL.fp16-batch1.engine "
            f"--camera-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml "
            f"--max-depth 7.0 --host {VLLM_IP} --port 5070 --save-depth --tiling 1"
        ),
        stop_extra=f"fuser -k 5070/tcp || true",
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

ALL_SERVICES = XTEND_SERVICES + NANOOWL_SERVICES


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------
def _ssh(cmd: str, timeout: int = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=4", JETSON_SSH, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def get_all_states() -> dict[str, bool]:
    """Single SSH call: returns {service_key: is_running} for all services."""
    cmd = (
        "echo '=TMUX='; tmux ls 2>/dev/null | cut -d: -f1; "
        "echo '=DOCKER='; docker ps --format '{{.Names}}' 2>/dev/null"
    )
    try:
        result = _ssh(cmd)
    except Exception:
        return {svc.key: False for svc in ALL_SERVICES}

    tmux_sessions: set[str] = set()
    docker_containers: set[str] = set()
    section = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line == "=TMUX=":
            section = "tmux"
        elif line == "=DOCKER=":
            section = "docker"
        elif section == "tmux" and line:
            tmux_sessions.add(line)
        elif section == "docker" and line:
            docker_containers.add(line)

    states: dict[str, bool] = {}
    for svc in ALL_SERVICES:
        if svc.env == "docker":
            states[svc.key] = svc.docker_container in docker_containers
        elif svc.machine == "pc":
            states[svc.key] = False  # not tracked remotely
        else:
            states[svc.key] = svc.key in tmux_sessions
    return states


def start_service(svc: Service) -> str | None:
    """Start service. Returns error string or None on success."""
    if svc.machine == "pc":
        env = _ENVS.get(svc.env, "")
        script = f"{env}\n{svc.cmd}"
        try:
            subprocess.Popen(
                ["bash", "-lc", script],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
        return None  # user closes the window manually

    cmds = [f"tmux kill-session -t {shlex.quote(svc.key)} 2>/dev/null || true"]
    if svc.stop_extra:
        cmds.append(svc.stop_extra)
    try:
        r = _ssh("; ".join(cmds), timeout=10)
        return r.stderr.strip() if r.returncode != 0 else None
    except Exception as exc:
        return str(exc)


def get_logs(svc: Service, lines: int = 120) -> str:
    """Fetch last N lines from the service log on the Jetson."""
    if svc.env == "docker":
        cmd = f"docker logs --tail {lines} {svc.docker_container} 2>&1 || echo 'No container logs'"
    else:
        cmd = f"tail -n {lines} {svc.log_file()} 2>/dev/null || echo 'No log yet: {svc.log_file()}'"
    try:
        r = _ssh(cmd, timeout=8)
        return r.stdout or r.stderr or "(empty)"
    except Exception as exc:
        return f"SSH error: {exc}"


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
m1, m2, m3 = st.columns(3)
m1.metric("Services running", f"{running_count} / {len(ALL_SERVICES)}")
m2.metric("XTEND", f"{sum(states.get(s.key, False) for s in XTEND_SERVICES)} / {len(XTEND_SERVICES)}")
m3.metric("NanoLLM", f"{sum(states.get(s.key, False) for s in NANOOWL_SERVICES)} / {len(NANOOWL_SERVICES)}")

st.divider()

tab_xtend, tab_nanoowl = st.tabs(["🚁  XTEND", "🤖  NanoLLM / OWL"])

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

    with st.expander("🗺️  Planner (Falcon)", expanded=False):
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

    st.caption(f"📷 Display: http://{VLLM_IP}:8090   |   📁 Data: http://{VLLM_IP}:9000")

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