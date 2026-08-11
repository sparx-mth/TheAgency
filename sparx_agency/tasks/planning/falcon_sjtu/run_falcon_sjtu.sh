#!/usr/bin/env bash
# ============================================================
# run_falcon_sjtu.sh — the full FALCON exploration on the SJTU Gazebo drone.
#
#   ./run_falcon_sjtu.sh [map_name]        # default small_house
#
# Expects the Gazebo sim ALREADY RUNNING (robots/SJTU/setup/bringup_world.sh);
# this brings up the other two containers and wires them:
#
#   sjtu sim (ROS2 Humble, domain 20)   ── already up, not touched here
#   ros1_bridge:noetic-foxy             ── 4 topics, config/bridge.yaml
#   falcon-ros-custom:v1                ── FALCON + our adapter nodes
#
# The mount pattern is the old stack's (and the XTEND stack's): scripts are
# overlaid file-by-file into the image's falcon_adapter package, so an
# unmounted file silently runs the version BAKED INTO THE IMAGE. If an edit
# appears to do nothing, check it is in the mount list below.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MAP_NAME="${1:-hospital}"

FALCON_IMAGE="${FALCON_IMAGE:-falcon-ros-custom:v1}"
BRIDGE_IMAGE="${BRIDGE_IMAGE:-ros1_bridge:noetic-foxy}"
# The sim container runs ROS_DOMAIN_ID=20 (checked live, not assumed). A
# mismatch is not an error -- it is silently zero data on every topic.
SIM_DOMAIN_ID="${SIM_DOMAIN_ID:-20}"
LOG_DIR="${FALCON_SJTU_LOG_DIR:-/tmp/falcon_sjtu}"
mkdir -p "${LOG_DIR}"
# FOLLOW=1 (default when interactive) tails the FALCON log and tears the stack
# down on Ctrl-C. FOLLOW=0 starts everything detached and returns, leaving the
# containers up for a monitor/soak to watch -- which is how the iterate loop
# drives it.
FOLLOW="${FOLLOW:-1}"

if [[ ! -f "${SCRIPT_DIR}/config/${MAP_NAME}.yaml" ]]; then
    echo "[ERROR] no map config: ${SCRIPT_DIR}/config/${MAP_NAME}.yaml" >&2
    exit 2
fi

chmod +x "${SCRIPT_DIR}"/adapter/scripts/*.py 2>/dev/null || true

cleanup() {
    docker rm -f falcon-sjtu sjtu-ros1-bridge > /dev/null 2>&1 || true
}
# Only tear down on an explicit interrupt, not on normal completion: FOLLOW=0
# returns with the stack still up on purpose.
trap cleanup INT TERM
cleanup

# ── FALCON container: roscore + planner + our nodes ────────────────────────
SCRIPTS_TARGET="/catkin_ws/src/falcon_adapter/scripts"
LAUNCH_TARGET="/catkin_ws/src/falcon_adapter/launch"
docker run -d --name falcon-sjtu \
    --network host \
    --env PYTHONPATH="/opt/sparx_agency" \
    --volume "${REPO_ROOT}:/opt/sparx_agency:ro" \
    --volume "${SCRIPT_DIR}/adapter/scripts/bspline_follower_node.py:${SCRIPTS_TARGET}/bspline_follower_node.py" \
    --volume "${SCRIPT_DIR}/adapter/scripts/sensor_pose_node.py:${SCRIPTS_TARGET}/sensor_pose_node.py" \
    --volume "${SCRIPT_DIR}/adapter/launch/exploration.launch:${LAUNCH_TARGET}/exploration.launch" \
    --volume "${SCRIPT_DIR}/adapter/launch/bspline_follower.launch:${LAUNCH_TARGET}/bspline_follower.launch" \
    --volume "${SCRIPT_DIR}/config/${MAP_NAME}.yaml:/catkin_ws/src/FALCON/falcon_planner/exploration_manager/config/map/${MAP_NAME}.yaml" \
    "${FALCON_IMAGE}" \
    roslaunch falcon_adapter exploration.launch map_name:="${MAP_NAME}" \
    > /dev/null

echo -n "[falcon_sjtu] waiting for roscore"
for _ in $(seq 1 30); do
    docker exec falcon-sjtu bash -lc 'source /opt/ros/noetic/setup.bash >/dev/null 2>&1; rostopic list' > /dev/null 2>&1 && break
    echo -n "."; sleep 2
done
echo " -- up"

# ── the bridge: ROS1 <-> ROS2 ──────────────────────────────────────────────
# The image bakes an entrypoint that defaults to dynamic_bridge (bridges every
# topic with default QoS) unless BRIDGE_MODE=static AND the yaml sits at
# /bridge_ws/bridge.yaml in the ROS2 --params-file schema. We want the explicit,
# QoS-per-topic parameter_bridge instead, so we override the entrypoint and run
# the classic mechanic: rosparam-load the topic list onto the ROS1 master, then
# `parameter_bridge` (which reads it from there). This is the shape the previous
# stack's ros_bridge_docker used and is known to work, and our config/bridge.yaml
# is already in that (rosparam) format.
#
# CycloneDDS, not Fast DDS: Foxy's Fast DDS 2.1 and Humble's 2.6 are not
# wire-compatible, so a Fast-DDS bridge would discover the sim and receive
# nothing. Both sides run CycloneDDS with SharedMemory off on the sim side
# (Humble hands SHM samples to iceoryx, which Foxy predates and never sees).
docker run -d --name sjtu-ros1-bridge \
    --network host --ipc=host \
    --env ROS_DOMAIN_ID="${SIM_DOMAIN_ID}" \
    --env ROS_MASTER_URI="http://localhost:11311" \
    --env ROS_HOSTNAME="localhost" \
    --env ROS_IP="127.0.0.1" \
    --env RMW_IMPLEMENTATION="rmw_cyclonedds_cpp" \
    --env CYCLONEDDS_URI="file:///cyclonedds_localhost.xml" \
    --volume "${SCRIPT_DIR}/config/bridge.yaml:/bridge.yaml:ro" \
    --volume /dev/shm:/dev/shm \
    --entrypoint bash \
    "${BRIDGE_IMAGE}" -lc '
        set -o pipefail
        source /opt/ros/noetic/setup.bash
        echo "[bridge] waiting for the FALCON roscore..."
        until timeout 2 rostopic list >/dev/null 2>&1; do sleep 1; done
        echo "[bridge] roscore up; loading topic list from /bridge.yaml"
        rosparam load /bridge.yaml
        source /opt/ros/foxy/setup.bash
        source /bridge_ws/install/setup.bash
        while true; do
            echo "[bridge] starting parameter_bridge (RMW=${RMW_IMPLEMENTATION:-default})"
            ros2 run ros1_bridge parameter_bridge
            echo "[bridge] parameter_bridge exited; restarting in 3s"
            sleep 3
        done
    ' > /dev/null

echo "[falcon_sjtu] up (map=${MAP_NAME})."
if [[ "${FOLLOW}" == "0" ]]; then
    echo "[falcon_sjtu] detached; containers left running. Follow with:"
    echo "    docker logs -f falcon-sjtu   |   docker logs -f sjtu-ros1-bridge"
    echo "    tear down: docker rm -f falcon-sjtu sjtu-ros1-bridge"
    exit 0
fi
echo "[falcon_sjtu] following falcon-sjtu; Ctrl-C tears both containers down."
echo "Logs mirrored to ${LOG_DIR}/falcon.log."
docker logs -f falcon-sjtu 2>&1 | tee "${LOG_DIR}/falcon.log"
