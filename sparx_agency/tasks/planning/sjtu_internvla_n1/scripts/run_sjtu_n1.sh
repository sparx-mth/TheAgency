#!/usr/bin/env bash
# ============================================================
# run_sjtu_n1.sh — fly the SJTU Gazebo drone under InternVLA-N1.
#
#   ./run_sjtu_n1.sh [world] [instruction]
#   ./run_sjtu_n1.sh no_roof_small_warehouse "go to the far shelf and stop"
#
# The whole point of this deployment: EVERYTHING runs on the CPU except the
# InternVLA-N1 model server, which owns the GPU (~8 GB) alone. This script
# enforces that -- it refuses to start until the card is empty, gives it to N1,
# and pins every other process (Gazebo, both ROS2 nodes) off the GPU.
#
# The chain it wires, which is NavDP's shape exactly:
#
#   Gazebo (SJTU warehouse, CPU) --RGB/depth/odom-->
#     n1_policy_node (CPU) --HTTP--> InternVLA-N1 server (GPU) --trajectory-->
#       /simple_drone/n1/trajectory (nav_msgs/Path, world) -->
#         trajectory_follower_node (CPU, pure pursuit) --> /simple_drone/cmd_vel
#
# It does NOT vendor Gazebo or the model server. Point SJTU_PROJECT_DIR at the
# sim checkout; have the InternVLA-N1 server runnable (conda env `internnav`) or
# set N1_SERVER_CMD to start it.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

WORLD="${1:-no_roof_small_warehouse}"
INSTRUCTION="${2:-${INSTRUCTION:-explore the warehouse and avoid the shelves}}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/sparx_agency/robots/SJTU/config/vla/internvla_n1.yaml}"

# ── environment the sim and the nodes must share ──────────────────────────
# The SJTU sim runs in Docker on CycloneDDS, domain 20 (robots/SJTU/README.md).
# The host nodes must join the same domain with the same RMW or they see nothing
# -- silently, exactly like a dead stack.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
export DISPLAY="${DISPLAY:-:1}"   # Gazebo Classic disables its cameras with no X
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

N1_HOST="${N1_HOST:-127.0.0.1}"
N1_PORT="${N1_PORT:-8087}"
LOG_DIR="${SJTU_N1_LOG_DIR:-/tmp/sjtu_n1}"
mkdir -p "${LOG_DIR}"

# Recording: RECORD=1 also writes an MP4 (drone camera + N1 route + S1/S2 FPS)
# and a rosbag. RECORD_SECONDS>0 flies for that long then tears down and reports;
# 0 flies until Ctrl-C.
RECORD="${RECORD:-0}"
RECORD_SECONDS="${RECORD_SECONDS:-0}"
RECORD_OUTPUT="${RECORD_OUTPUT:-${LOG_DIR}/run.mp4}"
BAG_DIR="${BAG_DIR:-${LOG_DIR}/bag_$(date +%H%M%S)}"

say() { echo "[sjtu_n1] $*"; }
die() { echo "[sjtu_n1] ERROR: $*" >&2; exit 1; }

# ── 1. GPU preflight: the card must be free for N1 ────────────────────────
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    say "checking the GPU is empty (N1 needs the whole card)..."
    python3 "${SCRIPT_DIR}/check_gpu_free.py" --require-empty \
        || die "GPU is not free. Free it (or SKIP_GPU_CHECK=1 to override), then rerun."
fi

# ── 2. Gazebo warehouse on the CPU ────────────────────────────────────────
# SJTU has no GPU physics; the world comes up on the CPU. Started only if a
# sim container is not already running, so a world left up between runs is
# reused. Set START_SIM=0 to manage the world yourself in another terminal.
sim_up() { docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' 2>/dev/null | grep -q .; }
if sim_up; then
    say "SJTU sim already running: $(docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' | head -n1)"
elif [[ "${START_SIM:-1}" == "1" ]]; then
    [[ -d "${SJTU_PROJECT_DIR}" ]] || die "SJTU_PROJECT_DIR=${SJTU_PROJECT_DIR} not found. Point it at the sim checkout."
    say "bringing up Gazebo world '${WORLD}' on the CPU (log: ${LOG_DIR}/gazebo.log)..."
    nohup bash "${REPO_ROOT}/sparx_agency/robots/SJTU/setup/bringup_world.sh" \
        --skip-build --domain "${ROS_DOMAIN_ID}" "${WORLD}" \
        > "${LOG_DIR}/gazebo.log" 2>&1 &
    say "waiting for the world to publish (odom + camera)..."
    for _ in $(seq 1 60); do sim_up && break; sleep 2; done
    sim_up || die "the SJTU sim did not come up. See ${LOG_DIR}/gazebo.log"
else
    die "no SJTU sim running and START_SIM=0. Bring it up first: bringup_world.sh ${WORLD}"
fi

# ── 3. InternVLA-N1 model server on the GPU ───────────────────────────────
n1_healthy() { curl -sf -m 3 "http://${N1_HOST}:${N1_PORT}/openapi.json" >/dev/null 2>&1; }
if n1_healthy; then
    say "InternVLA-N1 server healthy at ${N1_HOST}:${N1_PORT}"
elif [[ -n "${N1_SERVER_CMD:-}" ]]; then
    say "starting the InternVLA-N1 server: ${N1_SERVER_CMD} (log: ${LOG_DIR}/n1_server.log)"
    nohup bash -lc "${N1_SERVER_CMD}" > "${LOG_DIR}/n1_server.log" 2>&1 &
else
    say "InternVLA-N1 server is not up at ${N1_HOST}:${N1_PORT}."
    say "Start it on the GPU (conda env 'internnav'), e.g.:"
    say "    conda activate internnav && python <InternNav>/scripts/eval/start_server.py \\"
    say "        --config <...h1_internvla_n1_async_cfg.py> --host 0.0.0.0 --port ${N1_PORT}"
    say "or export N1_SERVER_CMD='<that command>' and rerun. Waiting for it..."
fi
say "waiting for the InternVLA-N1 server to answer..."
for _ in $(seq 1 "${N1_WAIT_TRIES:-120}"); do n1_healthy && break; sleep 2; done
n1_healthy || die "InternVLA-N1 server never became healthy at ${N1_HOST}:${N1_PORT}."
say "InternVLA-N1 server is up."

# ── 4. verify the GPU is the server's alone ───────────────────────────────
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    python3 "${SCRIPT_DIR}/check_gpu_free.py" --allow internnav --allow python --allow start_server \
        || say "WARNING: a process other than the N1 server is on the GPU (see above)."
fi

# ── 5. the two CPU nodes: policy trajectory + follower ────────────────────
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash" 2>/dev/null || die "cannot source /opt/ros/${ROS_DISTRO}/setup.bash"
export CUDA_VISIBLE_DEVICES=""            # both nodes: CPU only, no GPU
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
say "launching the N1 policy + follower nodes (CPU, domain ${ROS_DOMAIN_ID}, ${RMW_IMPLEMENTATION})..."
ros2 launch "${PKG_DIR}/launch/sjtu_internvla_n1.launch.py" \
    config_file:="${CONFIG_FILE}" \
    record:="$([[ "${RECORD}" == "1" ]] && echo true || echo false)" \
    record_output:="${RECORD_OUTPUT}" > "${LOG_DIR}/nodes.log" 2>&1 &
NODES_PID=$!
BAG_PID=""
cleanup() {
    say "tearing down nodes"
    [[ -n "${BAG_PID}" ]] && kill -INT "${BAG_PID}" 2>/dev/null || true
    kill "${NODES_PID}" 2>/dev/null || true
}
trap cleanup INT TERM
sleep 4
kill -0 "${NODES_PID}" 2>/dev/null || die "the ROS2 nodes exited at startup. See ${LOG_DIR}/nodes.log"

report() {
    echo
    say "=================== run summary ==================="
    if [[ "${RECORD}" == "1" ]]; then
        say "video : ${RECORD_OUTPUT}"
        say "rosbag: ${BAG_DIR}"
    fi
    local fps_line
    fps_line="$(grep -a 'N1 FPS' "${LOG_DIR}/nodes.log" 2>/dev/null | tail -n1)"
    if [[ -n "${fps_line}" ]]; then
        say "measured: ${fps_line#*N1 FPS}"
    else
        say "measured FPS: none yet (no N1 step logged — check the server & camera)."
    fi
    say "reference (this machine, ~/trt/internnav/REPORT.md):"
    say "  System 1: 6.77 Hz torch -> 22.99 Hz TensorRT   System 2: ~1.4 Hz (dual-system 1.41 Hz)"
    say "==================================================="
}

# ── 6. take off, then hand the instruction to N1 ──────────────────────────
if [[ "${RECORD}" == "1" ]]; then
    say "recording rosbag -> ${BAG_DIR}"
    ros2 bag record -o "${BAG_DIR}" \
        /simple_drone/front/image_raw /simple_drone/front_depth/depth/image_raw \
        /simple_drone/odom /simple_drone/cmd_vel /simple_drone/state \
        /simple_drone/n1/trajectory /simple_drone/n1/trajectory_full /simple_drone/n1/info \
        /simple_drone/navigation/instruction \
        > "${LOG_DIR}/bag.log" 2>&1 &
    BAG_PID=$!
fi

say "commanding takeoff..."
for _ in 1 2 3; do
    ros2 topic pub --once /simple_drone/takeoff std_msgs/msg/Empty "{}" >/dev/null 2>&1 || true
    sleep 1
done
sleep "${TAKEOFF_SETTLE_S:-5}"
say "sending instruction: '${INSTRUCTION}'"
ros2 topic pub --once /simple_drone/navigation/instruction std_msgs/msg/String \
    "{data: '${INSTRUCTION}'}" >/dev/null 2>&1 || say "WARNING: could not publish the instruction"

say "flying. N1 route -> /simple_drone/n1/trajectory ; cmd -> /simple_drone/cmd_vel"
[[ "${RECORD}" == "1" ]] && say "camera+route video -> ${RECORD_OUTPUT}"
say "logs: ${LOG_DIR}/nodes.log  (gazebo: ${LOG_DIR}/gazebo.log)"
if [[ "${RECORD_SECONDS}" -gt 0 ]]; then
    say "recording for ${RECORD_SECONDS}s, then tearing down..."
    sleep "${RECORD_SECONDS}"
    cleanup
else
    say "Ctrl-C to stop the nodes (the world and the server are left up)."
    wait "${NODES_PID}"
fi
sleep 2   # let the recorder flush and close the MP4
report

