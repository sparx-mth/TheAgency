#!/bin/bash
# ============================================================
# run_falcon.sh — run the FALCON container for ANY environment.
#
# Usage:  ./run_falcon.sh <env> [extra docker CMD ...]
#   <env> selects maps/<env>.yaml (e.g. office, hospital, bookstore).
#   Works for both the Gazebo sim and the real drone — the same
#   container serves both; the launch file you run inside picks which.
#
#   - Auto-detects arch and uses the right NVIDIA flag:
#       x86_64  → --gpus all  (nvidia-container-toolkit)
#       aarch64 → --runtime nvidia + NVIDIA_VISIBLE_DEVICES=all
#                 (works on JetPack 4.x and 5.x+ alike)
#   - Falls back to no-GPU if neither is available, with a warning.
#   - Picks the image tag built by docker-compose.yml based on arch
#     (falcon-ros:noetic on x86 vs falcon-ros:jetson on Jetson) so you
#     don't accidentally launch the x86 image on Jetson.
# ============================================================

CONTAINER="falcon"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH=$(uname -m)
if [ "${ARCH}" = "aarch64" ]; then
  # Change to your custom Jetson image tag
  IMAGE="${IMAGE:-falcon-ros-custom:v2}"
else
  # Change to your custom x86 image tag
  IMAGE="${IMAGE:-falcon-ros-custom:v1}"
fi
echo "[INFO] Arch: ${ARCH}   Image: ${IMAGE}"

# ── GPU flag selection ────────────────────────────────────────
# On Jetson, --gpus all only works on JetPack 5.x+ with
# nvidia-container-toolkit installed. The legacy --runtime nvidia
# is universally supported on Jetson (and is what the L4T docs
# recommend). On x86 we keep --gpus all.
GPU_ARGS=""
if [ "${ARCH}" = "aarch64" ]; then
  # Verify the nvidia runtime is actually registered with docker.
  if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    GPU_ARGS="--runtime nvidia \
              --env NVIDIA_VISIBLE_DEVICES=all \
              --env NVIDIA_DRIVER_CAPABILITIES=all"
    echo "[INFO] GPU: --runtime nvidia (Jetson)"
  else
    echo "[WARN] nvidia runtime not registered with docker.        "
    echo "[WARN] Edit /etc/docker/daemon.json so it contains:      "
    echo "[WARN]   { \"runtimes\": { \"nvidia\": {                 "
    echo "[WARN]       \"path\": \"nvidia-container-runtime\",     "
    echo "[WARN]       \"runtimeArgs\": [] } } }                   "
    echo "[WARN] then 'sudo systemctl restart docker'. Running    "
    echo "[WARN] CPU-only for now (RViz/Gazebo will be very slow)."
  fi
else
  # x86_64: prefer modern --gpus all; warn if missing.
  if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    GPU_ARGS="--gpus all \
              --env NVIDIA_DRIVER_CAPABILITIES=all \
              --env NVIDIA_VISIBLE_DEVICES=all"
    echo "[INFO] GPU: --gpus all"
  else
    echo "[WARN] No nvidia runtime detected; running CPU-only."
  fi
fi

# ── Map config ────────────────────────────────────────────────
ENV_NAME="${1:-hospital}"
if [[ $# -ge 1 ]]; then shift; fi

if [[ ! -f "${SCRIPT_DIR}/maps/${ENV_NAME}.yaml" ]]; then
  echo "[ERROR] Map config not found: ${SCRIPT_DIR}/maps/${ENV_NAME}.yaml"
  echo "        Available configs:"
  ls -1 "${SCRIPT_DIR}"/maps/*.yaml 2>/dev/null | xargs -n1 basename || true
  exit 1
fi
echo "[INFO] FALCON env: ${ENV_NAME}  (config: ${SCRIPT_DIR}/maps/${ENV_NAME}.yaml)"

# Auto-chmod +x on host so we don't lose 10 min wondering why nodes
# aren't found. Harmless if they were already executable.
chmod +x "${SCRIPT_DIR}"/adapter/scripts/*.py 2>/dev/null || true

# ── X display ─────────────────────────────────────────────────
# RViz and the mission director's object-list window are X clients inside the
# container, so DISPLAY must be set. Over a bare `ssh jetson` (no -X) it is unset,
# and an empty --env DISPLAY= makes every GUI die on "cannot open display". Default
# to the Jetson's local screen; an existing DISPLAY (e.g. ssh -X, or a desktop
# session) is respected. Export so xhost below and the docker --env both see it.
export DISPLAY="${DISPLAY:-:0}"
echo "[INFO] DISPLAY: ${DISPLAY}"

xhost +local:docker 2>/dev/null || true

# ── sparx_agency repo (ROS-free algorithms used by the nodes) ──
# The adapter nodes here import core.mapping.bev / core.localization from the
# repo. Mount it read-only and put its parent on PYTHONPATH so those imports
# resolve inside the container. SPARX_PARENT can be overridden; it defaults to
# the repo root four levels up (sparx_agency/tasks/planning/falcon -> repo root).
SPARX_PARENT="${SPARX_PARENT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
if [ ! -d "${SPARX_PARENT}/sparx_agency" ]; then
  echo "[ERROR] sparx_agency not found under SPARX_PARENT=${SPARX_PARENT}"
  echo "        Set SPARX_PARENT to the directory that CONTAINS sparx_agency."
  exit 1
fi
echo "[INFO] sparx_agency repo: ${SPARX_PARENT}/sparx_agency (mounted at /opt/sparx_agency)"

# ── Volume mounts ─────────────────────────────────────────────
# Mount each adapter node (+ the cloud_utils / pure_pursuit_follower / thinking
# helpers) so host edits take effect without a rebuild. Loop so a missing file is
# skipped with a log line instead of docker silently creating an empty dir.
# NOTE: helper modules the nodes IMPORT must be listed here too -- an unmounted
# helper is not a missing feature, it is an ImportError that takes the node down.
SCRIPTS_HOST="${SCRIPT_DIR}/adapter/scripts"
SCRIPTS_TARGET="/catkin_ws/src/falcon_adapter/scripts"
SCRIPT_MOUNTS=()
for f in falcon_adapter_node.py sensor_gate_node.py bev_publisher_node.py mapping_sync_node.py bev_click_goal_node.py astar_planner_node.py navdp_click_node.py flownav_node.py combination_planner_node.py astar_navdp_fallback_node.py hybrid_planner_node.py path_corrector_node.py trajectory_simplifier_node.py waypoint_follower_node.py pose_adapter_node.py sim_adapter_node.py cloud_utils.py pure_pursuit_follower.py drift_pid_follower.py localization_quality.py thinking.py thought_journal.py certainty_log.py object_approach_node.py target_lock_viewer_node.py mission_director_node.py cmd_vel_gate_node.py lost_localization_node.py depth_debug.py; do
  if [ -f "${SCRIPTS_HOST}/${f}" ]; then
    SCRIPT_MOUNTS+=( --volume "${SCRIPTS_HOST}/${f}:${SCRIPTS_TARGET}/${f}" )
  else
    echo "[INFO] Skipping missing script: ${f}"
  fi
done

LAUNCH_HOST="${SCRIPT_DIR}/adapter/launch"
LAUNCH_TARGET="/catkin_ws/src/falcon_adapter/launch"
LAUNCH_MOUNTS=()
for f in nav_stack.launch real_drone.launch object_approach.launch real_drone_object_approach.launch object_mission.launch ; do
  if [ -f "${LAUNCH_HOST}/${f}" ]; then
    LAUNCH_MOUNTS+=( --volume "${LAUNCH_HOST}/${f}:${LAUNCH_TARGET}/${f}" )
  fi
done

# ── Frame directories (frame-path transport) ──────────────────
# Depth/RGB now arrive as std_msgs/String "<path> <sec> <nsec>" messages whose
# paths point at files the drone-side publisher writes on the HOST (e.g.
# /tmp/xtend_depth/*.npy). The ROS1 consumers (mapping_sync, navdp_click) run
# INSIDE this container, so those host dirs MUST be bind-mounted at the SAME path
# for the paths to resolve. The mount is live: files the host writes after launch
# appear in the container, so producer/container start order does not matter.
# We mkdir -p first so the mount is created even if the publisher has not run yet
# (a missing source would otherwise be skipped and reads would fail until a
# restart). Override the set with XTEND_FRAME_DIRS (space-separated).
XTEND_FRAME_DIRS="${XTEND_FRAME_DIRS:-/tmp/xtend_frames /tmp/xtend_depth}"
FRAME_MOUNTS=()
for d in ${XTEND_FRAME_DIRS}; do
  mkdir -p "${d}" 2>/dev/null || true
  if [ -d "${d}" ]; then
    FRAME_MOUNTS+=( --volume "${d}:${d}:ro" )
    echo "[INFO] Frame dir mounted (ro): ${d}"
  else
    echo "[WARN] Could not create/mount frame dir: ${d}"
  fi
done

# ── Object catalog directory (the room map's objects.json) ────
# The mission director runs INSIDE this container but the catalog is written on
# the HOST by the room mapper, so its dir is bind-mounted at the SAME path for
# the default objects_file to resolve (same trick as the frame dirs above).
# We mount the DIRECTORY, not the file: a bind-mounted file pins one inode, so a
# re-run of the mapper -- which REPLACES objects.json -- would leave the container
# reading the stale catalog. Mounting the dir keeps it live.
# Unlike the frame dirs we do NOT mkdir -p: an empty dir would mask a missing map
# and the director would fly to nothing. Override with OBJECTS_DIR.
OBJECTS_DIR="${OBJECTS_DIR:-/home/user/jetson-containers/data/captures/latest_room_map}"
OBJECTS_MOUNT=()
if [ -d "${OBJECTS_DIR}" ]; then
  OBJECTS_MOUNT=( --volume "${OBJECTS_DIR}:${OBJECTS_DIR}:ro" )
  echo "[INFO] Object catalog dir mounted (ro): ${OBJECTS_DIR}"
else
  echo "[WARN] Object catalog dir not found: ${OBJECTS_DIR}"
  echo "[WARN]   Only matters for the object mission (mission_director)."
  echo "[WARN]   Set OBJECTS_DIR=<host dir holding objects.json> to relocate it."
fi

# docker.sock is only needed when respawn_drone.py is in play
# (sim-only). On Jetson it's harmless to mount but pointless.
DOCKER_SOCK_MOUNT=()
if [ "${ARCH}" != "aarch64" ] && [ -S /var/run/docker.sock ]; then
  DOCKER_SOCK_MOUNT=( --volume /var/run/docker.sock:/var/run/docker.sock )
fi

# ── Flight logs (thought journal + certainty log) ─────────────
# The nodes write INSIDE the container, which runs --rm, so without a host
# bind-mount the log of the flight you just did is deleted the moment it lands.
# Mount a host directory read-write and name ONE file per log for this run:
# every narrating node appends to the SAME thought journal, so the filename
# cannot be decided per node (each would stamp its own). Defaults to /tmp/falcon
# so a Jetson run needs no extra setup; override with FALCON_LOG_DIR. Only
# populated when the stack is launched with thinking_log:=true /
# certainty_log:=true; see adapter/scripts/thought_journal.py and
# adapter/scripts/certainty_log.py.
LOG_HOST="${FALCON_LOG_DIR:-/tmp/falcon}"
mkdir -p "${LOG_HOST}"
STAMP="$(date +%Y%m%d_%H%M%S)"
THOUGHT_LOG="/falcon_logs/thoughts_${STAMP}.log"
CERTAINTY_LOG="/falcon_logs/certainty_${STAMP}.csv"
echo "[INFO] Thought log (with thinking_log:=true) -> ${LOG_HOST}/$(basename "${THOUGHT_LOG}")"
echo "[INFO] Certainty log (with certainty_log:=true) -> ${LOG_HOST}/$(basename "${CERTAINTY_LOG}")"

# ── Run ───────────────────────────────────────────────────────
docker run -it --rm \
    --volume "${LOG_HOST}:/falcon_logs" \
    --env FALCON_LOG_DIR=/falcon_logs \
    --env FALCON_THOUGHT_LOG="${THOUGHT_LOG}" \
    --env FALCON_CERTAINTY_LOG="${CERTAINTY_LOG}" \
    --name "${CONTAINER}" \
    ${GPU_ARGS} \
    --env DISPLAY="${DISPLAY}" \
    --env XAUTHORITY=/tmp/.docker.xauth \
    --volume /tmp/.docker.xauth:/tmp/.docker.xauth \
    --env QT_X11_NO_MITSHM=1 \
    --shm-size=2g \
    --ulimit nofile=65536:65536 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume "${SPARX_PARENT}/sparx_agency:/opt/sparx_agency:ro" \
    --env PYTHONPATH=/opt \
    "${SCRIPT_MOUNTS[@]}" \
    "${LAUNCH_MOUNTS[@]}" \
    "${FRAME_MOUNTS[@]}" \
    "${OBJECTS_MOUNT[@]}" \
    "${DOCKER_SOCK_MOUNT[@]}" \
    --volume "${SCRIPT_DIR}/maps/${ENV_NAME}.yaml:/catkin_ws/src/FALCON/falcon_planner/exploration_manager/config/map/${ENV_NAME}.yaml" \
    --network host \
    "${IMAGE}" \
    "${@:-bash}"