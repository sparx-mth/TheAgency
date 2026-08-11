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

# ============== ROBOTICAN FIX ==============
# General bugfix, not Rooster-specific: these were stale placeholder tags
# (falcon-ros-custom:v1/v2) that never matched any image actually built by
# docker-compose.yml, on EITHER arch -- so this was already broken for
# Jetson/XTEND too, not something working that could regress.
# ============================================
ARCH=$(uname -m)
if [ "${ARCH}" = "aarch64" ]; then
  # Built by `docker compose build falcon-jetson` (docker-compose.yml).
  IMAGE="${IMAGE:-falcon-ros:jetson}"
else
  # Built by `docker compose build falcon-pc` (docker-compose.yml).
  IMAGE="${IMAGE:-falcon-ros:noetic}"
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

# ── Exploration area: derived and costed before anything starts ───
# The map file gives the area in a handful of numbers under `map_config.area`;
# FALCON wants the eighteen of `map_config.map_size`. Deriving them here rather
# than in the container means a bad area fails in a tenth of a second with a
# sentence instead of inside roslaunch with a glog CHECK and a stack trace, and
# the size of the voxel grid gets printed while there is still time to change
# it -- it is allocated in full on the first tick and never grows.
#
# The expanded copy is what gets mounted, so the file the planner reads always
# matches the file you edited.
#
# SPARX_PARENT is not resolved until further down, so work it out here the same
# way that line does; exporting it still wins.
MAPSIZE_ROOT="${SPARX_PARENT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
MAP_EXPANDED_DIR="$(mktemp -d)"
trap 'rm -rf "${MAP_EXPANDED_DIR}"' EXIT
MAP_EXPANDED="${MAP_EXPANDED_DIR}/${ENV_NAME}.yaml"
PYTHON="${PYTHON:-${MAPSIZE_ROOT}/.venv/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="python3"
if ! PYTHONPATH="${MAPSIZE_ROOT}" "${PYTHON}" \
        -m sparx_agency.tasks.planning.falcon_pegasus.mapsize \
        "${SCRIPT_DIR}/maps/${ENV_NAME}.yaml" --out "${MAP_EXPANDED}"; then
  echo "[ERROR] this environment's exploration area is unusable. Fix" >&2
  echo "        map_config.area in maps/${ENV_NAME}.yaml -- see" >&2
  echo "        ../falcon_pegasus/mapsize/README.md." >&2
  exit 2
fi

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
# ============== ROBOTICAN FIX ==============
# Mount the whole scripts/ and launch/ DIRECTORIES instead of one --volume
# per individual file. A single-file bind mount is bound to that file's
# INODE at container-creation time: any host edit made via write-temp-then-
# rename (which most editors, including Claude Code's, do for atomic
# writes) replaces the inode at that path, so the container keeps viewing
# the orphaned old inode forever -- a host edit is invisible inside an
# already-running container until the container itself is recreated. Hit
# this repeatedly during the ROBOTICAN/Sphera bridging session (real_drone.
# launch, bev_publisher_node.py) before recognizing the pattern (see
# project_falcon_robotican_bridging memory). A DIRECTORY bind mount doesn't
# have this problem -- the mount point is the directory's dentry, not each
# file's inode, so edits/renames underneath it are visible immediately.
# This also means every script/launch file is now live-editable (not just
# the ones previously listed by name), and a newly added script needs no
# corresponding line here. Behavior for XTEND/Jetson is unchanged: every
# file that was previously mounted individually is still present (just via
# the parent directory), nothing was removed or renamed.
# ============================================
SCRIPTS_HOST="${SCRIPT_DIR}/adapter/scripts"
SCRIPTS_TARGET="/catkin_ws/src/falcon_adapter/scripts"
SCRIPT_MOUNTS=( --volume "${SCRIPTS_HOST}:${SCRIPTS_TARGET}" )

LAUNCH_HOST="${SCRIPT_DIR}/adapter/launch"
LAUNCH_TARGET="/catkin_ws/src/falcon_adapter/launch"
LAUNCH_MOUNTS=( --volume "${LAUNCH_HOST}:${LAUNCH_TARGET}" )

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
# ============== ROBOTICAN FIX ==============
# Added /tmp/rooster_frames /tmp/rooster_depth to the default alongside
# XTEND's two dirs. Purely additive -- XTEND's dirs are still mounted
# exactly as before, so any existing XTEND/Jetson run is unaffected. The
# extra two are harmless no-ops on a pure-XTEND run (mkdir -p creates an
# always-empty dir, nothing ever reads/writes it). Saves forgetting the
# override for a Rooster run: a missing bind mount here is NOT visible as
# an error until mapping_sync's depth reads start failing with ENOENT,
# which looks identical to a timing/rotation bug (cost real debugging time
# this session before the missing mount was found).
# ============================================
XTEND_FRAME_DIRS="${XTEND_FRAME_DIRS:-/tmp/xtend_frames /tmp/xtend_depth /tmp/rooster_frames /tmp/rooster_depth}"
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
# NOTE: the map YAML below is intentionally still a single-file mount, NOT
# a directory mount like scripts/launch above -- FALCON's own
# config/map/ dir in the image ships several upstream example maps
# (darpa_tunnel.yaml, octa_maze.yaml, complex_office.yaml, etc.) that do
# NOT exist in this repo's maps/ dir. A directory mount would shadow the
# whole target dir with only what's in maps/ here, making those upstream
# maps inaccessible. So editing the CURRENT env's map YAML still needs a
# container restart to take effect (single-file bind-mount inode gotcha,
# see project_falcon_robotican_bridging memory) -- only scripts/launch
# got the directory-mount fix.
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
    --volume "${MAP_EXPANDED}:/catkin_ws/src/FALCON/falcon_planner/exploration_manager/config/map/${ENV_NAME}.yaml" \
    --network host \
    "${IMAGE}" \
    "${@:-bash}"