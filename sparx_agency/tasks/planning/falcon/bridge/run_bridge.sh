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

# Sanity-check the bridge.yaml is here (entrypoint will hard-fail
# otherwise, but failing here gives a more helpful message)
if [ ! -f "${SCRIPT_DIR}/bridge.yaml" ]; then
    echo "[ERROR] bridge.yaml not found at ${SCRIPT_DIR}/bridge.yaml"
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

echo "================================================"
echo "  Launching ${CONTAINER}"
echo "  Domain   : ${ROS_DOMAIN_ID:-5}"
echo "  RMW      : ${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "  Config   : ${SCRIPT_DIR}/bridge.yaml  (editable; restart container)"
echo "  Log      : ${LOGFILE_HOST}"
echo "================================================"

docker run -it --rm \
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
    -v "${SCRIPT_DIR}/bridge.yaml:/bridge.yaml:ro" \
    -v "${LOGFILE_HOST}:/tmp/bridge.log:rw" \
    -v "${SCRIPT_DIR}/fastdds_no_shm.xml:/fastdds_no_shm.xml:ro" \
    -v /dev/shm:/dev/shm \
    -v ~/Downloads/OneDrive_1_6-1-2026:/bag_1:ro \
    -v ~/Documents/record:/bag:ro \
    "${BAG_MOUNT[@]}" \
    --entrypoint /entrypoint.sh \
    "${IMAGE}"
