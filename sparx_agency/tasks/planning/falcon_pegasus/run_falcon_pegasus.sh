#!/usr/bin/env bash
# ============================================================
# run_falcon_pegasus.sh — start the FALCON side of an exploration run.
#
#   ./run_falcon_pegasus.sh <run> [extra roslaunch args ...]
#
# <run> names one of runs/*.yaml without the extension, e.g. 3_open_plan.
#
# Pass --rviz to watch it live in FALCON's own RViz: the occupancy voxels, the
# ESDF, the frontier clusters, the sampled viewpoints, the hierarchical grid and
# the executed trajectory. That needs an X display, which this script forwards
# into the container.
#
# Start this FIRST, then the Isaac Sim side (run_isaac_side.sh). The bridge
# binds the two localhost sockets and the aircraft connects to them: the ROS
# stack is up in seconds, Kit takes minutes to load a stage, so whoever takes
# longer should be the one that retries.
#
# The container runs with --network host, which is not optional: it is what puts
# both containers on the same loopback device so 127.0.0.1:5599 means the same
# socket in each. On a bridge network the aircraft's connect() fails with
# ECONNREFUSED and looks exactly like a start-up ordering bug.
#
# Without --rviz no X server is used or needed: FALCON's own visualisation is an
# RViz config, and this stack writes an MP4 instead (map_recorder_node), so an
# unattended run works over a bare ssh session.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PARENT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IMAGE="${FALCON_PEGASUS_IMAGE:-falcon-pegasus:noetic}"
CONTAINER="${FALCON_PEGASUS_CONTAINER:-falcon-pegasus}"
LOG_HOST="${FALCON_LOG_DIR:-/tmp/falcon_pegasus}"

WANT_RVIZ=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--rviz" ]]; then WANT_RVIZ=1; else ARGS+=("$arg"); fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

RUN_NAME="${1:-3_open_plan}"
if [[ $# -ge 1 ]]; then shift; fi

if [[ ! -f "${SCRIPT_DIR}/runs/${RUN_NAME}.yaml" ]]; then
    echo "[ERROR] no such run: ${RUN_NAME}" >&2
    echo "        available:" >&2
    ls -1 "${SCRIPT_DIR}"/runs/*.yaml | xargs -n1 basename | sed 's/\.yaml$//;s/^/          /' >&2
    exit 2
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[ERROR] image ${IMAGE} not found. Build it with:" >&2
    echo "          cd ${SCRIPT_DIR} && docker build -t ${IMAGE} ." >&2
    exit 1
fi

# ── X display, only when RViz is wanted ───────────────────────────────
# The container reaches the host's X server through the mounted socket, with
# host-based access control opened for local connections. That is what
# run_falcon.sh does too, minus the XAUTHORITY file: on this machine
# ~/.Xauthority is a root-owned directory rather than a cookie file, so
# `xhost +local:` is what actually grants access.
X11_ARGS=()
if (( WANT_RVIZ )); then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "[ERROR] --rviz needs a DISPLAY and none is set." >&2
        echo "        Over ssh use 'ssh -X'; otherwise run from a desktop session." >&2
        exit 2
    fi
    if [[ ! -S "/tmp/.X11-unix/X${DISPLAY##*:}" ]]; then
        echo "[WARN] no X socket for DISPLAY=${DISPLAY}; RViz will fail to open a window."
    fi
    xhost +local:docker >/dev/null 2>&1 \
        || echo "[WARN] xhost failed; the X server may refuse RViz"
    X11_ARGS=(--env "DISPLAY=${DISPLAY}"
              --env QT_X11_NO_MITSHM=1
              --volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
    # Send RViz's GL to the NVIDIA card. Without this the container's Mesa tries
    # the laptop's Intel iGPU, fails to load the i915 DRI driver (it is not in
    # the image), floods the console with libGL errors and falls back to
    # software rendering -- which does work, at about 11 fps on a building-sized
    # voxel cloud. Set FALCON_PEGASUS_SOFTWARE_GL=1 to go back to that if the
    # offload misbehaves on another machine.
    if [[ -z "${FALCON_PEGASUS_SOFTWARE_GL:-}" ]]; then
        X11_ARGS+=(--env __NV_PRIME_RENDER_OFFLOAD=1
                   --env __GLX_VENDOR_LIBRARY_NAME=nvidia)
    else
        X11_ARGS+=(--env LIBGL_ALWAYS_SOFTWARE=1)
    fi
    echo "[INFO] RViz on, DISPLAY=${DISPLAY}"
fi

if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "[WARN] no nvidia container runtime; running CPU-only (FALCON needs no GPU)."
    GPU_ARGS=()
else
    GPU_ARGS=(--gpus all --env NVIDIA_DRIVER_CAPABILITIES=all)
fi

# One folder per run: the map video, the ROS log and anything else this run
# produces land together under a single timestamp.
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_HOST="${LOG_HOST}/${STAMP}_${RUN_NAME}"
mkdir -p "${RUN_DIR_HOST}"
echo "[INFO] run:        ${RUN_NAME}"
echo "[INFO] image:      ${IMAGE}"
echo "[INFO] output dir: ${RUN_DIR_HOST}"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

chmod +x "${SCRIPT_DIR}"/adapter/scripts/*.py 2>/dev/null || true

# Allocate a TTY only when there is one. A campaign runs this backgrounded with
# its output piped to a log, and `docker run -t` without a terminal fails.
TTY_ARGS=(-i)
if [ -t 1 ]; then TTY_ARGS=(-it); fi

# Extra roslaunch arguments, quoted for the shell inside the container. Built
# conditionally: `printf '%q ' "$@"` with no arguments still runs its format
# once and emits a literal '', which roslaunch reads as an empty filename and
# refuses to start on.
EXTRA_ARGS=""
if (( WANT_RVIZ )); then EXTRA_ARGS="rviz:=true "; fi
if [ $# -gt 0 ]; then EXTRA_ARGS="${EXTRA_ARGS}$(printf '%q ' "$@")"; fi

# Mount the scripts over BOTH the source tree and catkin's devel bin directory.
# catkin_install_python copies them into devel/lib/<pkg> at build time, and which
# of the two roslaunch resolves `type=` against is not worth depending on --
# covering both means a host edit is always what runs.
docker run "${TTY_ARGS[@]}" --rm \
    --name "${CONTAINER}" \
    --network host \
    "${GPU_ARGS[@]}" \
    ${X11_ARGS[@]+"${X11_ARGS[@]}"} \
    --shm-size=2g \
    --ulimit nofile=65536:65536 \
    --volume "${REPO_PARENT}/sparx_agency:/opt/sparx_agency:ro" \
    --env PYTHONPATH=/opt \
    --volume "${SCRIPT_DIR}/adapter/scripts:/catkin_ws/src/falcon_pegasus/scripts:ro" \
    --volume "${SCRIPT_DIR}/adapter/scripts:/catkin_ws/devel/lib/falcon_pegasus:ro" \
    --volume "${SCRIPT_DIR}/adapter/launch:/catkin_ws/src/falcon_pegasus/launch:ro" \
    --volume "${SCRIPT_DIR}/runs:/catkin_ws/src/falcon_pegasus/runs:ro" \
    --volume "${RUN_DIR_HOST}:/falcon_logs" \
    --env FALCON_RUN_NAME="${RUN_NAME}" \
    "${IMAGE}" \
    bash -lc "source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && \
              roslaunch falcon_pegasus falcon_pegasus.launch run:=${RUN_NAME} ${EXTRA_ARGS}"

echo "[INFO] finished. Output in ${RUN_DIR_HOST}"
