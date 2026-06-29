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
# Mount each adapter node (+ the cloud_utils helper) so host edits take effect
# without a rebuild. Loop so a missing file is skipped with a log line instead
# of docker silently creating an empty dir.
SCRIPTS_HOST="${SCRIPT_DIR}/adapter/scripts"
SCRIPTS_TARGET="/catkin_ws/src/falcon_adapter/scripts"
SCRIPT_MOUNTS=()
for f in falcon_adapter_node.py sensor_gate_node.py bev_publisher_node.py mapping_sync_node.py bev_click_goal_node.py astar_planner_node.py navdp_click_node.py path_corrector_node.py trajectory_simplifier_node.py waypoint_follower_node.py pose_adapter_node.py sim_adapter_node.py cloud_utils.py depth_debug.py; do
  if [ -f "${SCRIPTS_HOST}/${f}" ]; then
    SCRIPT_MOUNTS+=( --volume "${SCRIPTS_HOST}/${f}:${SCRIPTS_TARGET}/${f}" )
  else
    echo "[INFO] Skipping missing script: ${f}"
  fi
done

LAUNCH_HOST="${SCRIPT_DIR}/adapter/launch"
LAUNCH_TARGET="/catkin_ws/src/falcon_adapter/launch"
LAUNCH_MOUNTS=()
for f in nav_stack.launch real_drone.launch ; do
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

# docker.sock is only needed when respawn_drone.py is in play
# (sim-only). On Jetson it's harmless to mount but pointless.
DOCKER_SOCK_MOUNT=()
if [ "${ARCH}" != "aarch64" ] && [ -S /var/run/docker.sock ]; then
  DOCKER_SOCK_MOUNT=( --volume /var/run/docker.sock:/var/run/docker.sock )
fi

# ── Run ───────────────────────────────────────────────────────
docker run -it --rm \
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
    "${DOCKER_SOCK_MOUNT[@]}" \
    --volume "${SCRIPT_DIR}/maps/${ENV_NAME}.yaml:/catkin_ws/src/FALCON/falcon_planner/exploration_manager/config/map/${ENV_NAME}.yaml" \
    --network host \
    "${IMAGE}" \
    "${@:-bash}"