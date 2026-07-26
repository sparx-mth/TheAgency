#!/bin/bash
set -e

IMAGE="ros1_bridge:noetic-foxy"
CONTAINER="ros1_bridge"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build image if missing
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "[INFO] Image '${IMAGE}' not found - building..."
    docker build -t "${IMAGE}" "${SCRIPT_DIR}"
fi

# Which bridge config to mount. Default bridge.yaml = frame-path transport
# (RGB/depth as std_msgs/String "<path> <sec> <nsec>", the real drone). Set
# BRIDGE_CFG=bridge_topic.yaml for the TOPIC transport variant (raw
# sensor_msgs/Image, a sim or bag replay). Always launch the ROS1 consumers with
# the matching image_transport (frame_path | topic).
BRIDGE_CFG="${BRIDGE_CFG:-bridge.yaml}"

# Sanity-check the chosen config is here (entrypoint will hard-fail
# otherwise, but failing here gives a more helpful message)
if [ ! -f "${SCRIPT_DIR}/${BRIDGE_CFG}" ]; then
    echo "[ERROR] ${BRIDGE_CFG} not found at ${SCRIPT_DIR}/${BRIDGE_CFG}"
    echo "        parameter_bridge needs it to know which topics + QoS"
    exit 1
fi

# Prepare host log file
LOGFILE_HOST="${SCRIPT_DIR}/bridge.log"
touch "${LOGFILE_HOST}"
chmod 666 "${LOGFILE_HOST}"

docker rm -f "${CONTAINER}" 2>/dev/null || true

# Optional rosbag mount for playback. Set BAG_DIR=/path/to/bags to mount it at
# /bag inside the container (no hardcoded host paths).
BAG_MOUNT=()
if [ -n "${BAG_DIR:-}" ]; then
    BAG_MOUNT=( -v "${BAG_DIR}:/bag:ro" )
    echo "[INFO] Mounting rosbag dir: ${BAG_DIR} -> /bag"
fi

# ============== ROBOTICAN FIX ==============
# CYCLONEDDS_URI (needed for ROBOTICAN/Sphera, unset by default for XTEND's
# fastrtps profile). A file:// URI only resolves if the file exists at that
# same path INSIDE the container, so bind-mount it there read-only. Example
# for Sphera (matches run_depth_processor.sh / run_rooster_frame_dir_publisher.sh):
#   ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
#   CYCLONEDDS_URI=file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml ./run_bridge.sh
# CYCLONEDDS_ENV stays empty (no -e flag added at all) when CYCLONEDDS_URI is
# unset, so a plain XTEND run's container environment is byte-identical to
# before this change -- not even an empty-string CYCLONEDDS_URI is injected.
CYCLONEDDS_MOUNT=()
CYCLONEDDS_ENV=()
if [ -n "${CYCLONEDDS_URI:-}" ] && [[ "${CYCLONEDDS_URI}" == file://* ]]; then
    CDDS_HOST_PATH="${CYCLONEDDS_URI#file://}"
    if [ -f "${CDDS_HOST_PATH}" ]; then
        CYCLONEDDS_MOUNT=( -v "${CDDS_HOST_PATH}:${CDDS_HOST_PATH}:ro" )
        CYCLONEDDS_ENV=( -e "CYCLONEDDS_URI=${CYCLONEDDS_URI}" )
    else
        echo "[WARN] CYCLONEDDS_URI file not found on host: ${CDDS_HOST_PATH}"
    fi
fi
# ============================================

echo "================================================"
echo "  Launching ${CONTAINER}"
echo "  Domain   : ${ROS_DOMAIN_ID:-5}"
echo "  RMW      : ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "  CycloneDDS URI : ${CYCLONEDDS_URI:-(unset)}"
echo "  Config   : ${SCRIPT_DIR}/${BRIDGE_CFG}  (editable; restart container)"
echo "  Log      : ${LOGFILE_HOST}"
echo "================================================"

# -i/-t ONLY when stdin and stdout really are terminals. run_object_mission.sh starts
# this script in the BACKGROUND with its output redirected to a log file, so stdin is
# not a terminal and `docker run -it` hard-fails with "cannot attach stdin to a
# TTY-enabled container because stdin is not a terminal": the container never starts
# and NO topics cross the bridge, while the mission carries on looking healthy.
# Interactive runs still get -it (colours, Ctrl+C).
DOCKER_TTY=()
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_TTY=( -it )
fi

docker run "${DOCKER_TTY[@]}" --rm \
    --net=host \
    --ipc=host \
    --name="${CONTAINER}" \
    -e ROS_MASTER_URI="http://localhost:11311" \
    -e ROS_HOSTNAME="localhost" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-5}" \
    -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
    -e FASTRTPS_DEFAULT_PROFILES_FILE="/fastdds_no_shm.xml" \
    -e LOGFILE="/tmp/bridge.log" \
    -e BRIDGE_YAML="/bridge.yaml" \
    -v "${SCRIPT_DIR}/entrypoint.sh:/entrypoint.sh:ro" \
    -v "${SCRIPT_DIR}/${BRIDGE_CFG}:/bridge.yaml:ro" \
    -v "${LOGFILE_HOST}:/tmp/bridge.log:rw" \
    -v "${SCRIPT_DIR}/fastdds_no_shm.xml:/fastdds_no_shm.xml:ro" \
    -v /dev/shm:/dev/shm \
    -v ~/Downloads/OneDrive_1_6-1-2026:/bag_1:ro \
    -v ~/Documents/record:/bag:ro \
    "${BAG_MOUNT[@]}" \
    "${CYCLONEDDS_MOUNT[@]}" \
    "${CYCLONEDDS_ENV[@]}" \
    --entrypoint /entrypoint.sh \
    "${IMAGE}"
