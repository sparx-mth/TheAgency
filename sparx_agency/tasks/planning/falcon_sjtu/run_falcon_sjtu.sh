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
MAP_NAME="${1:-small_house}"

FALCON_IMAGE="${FALCON_IMAGE:-falcon-ros-custom:v1}"
BRIDGE_IMAGE="${BRIDGE_IMAGE:-ros1_bridge:noetic-foxy}"
# The sim container runs ROS_DOMAIN_ID=20 (checked live, not assumed). A
# mismatch is not an error -- it is silently zero data on every topic.
SIM_DOMAIN_ID="${SIM_DOMAIN_ID:-20}"
LOG_DIR="${FALCON_SJTU_LOG_DIR:-/tmp/falcon_sjtu}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${SCRIPT_DIR}/config/${MAP_NAME}.yaml" ]]; then
    echo "[ERROR] no map config: ${SCRIPT_DIR}/config/${MAP_NAME}.yaml" >&2
    exit 2
fi

chmod +x "${SCRIPT_DIR}"/adapter/scripts/*.py 2>/dev/null || true

cleanup() {
    docker rm -f falcon-sjtu sjtu-ros1-bridge > /dev/null 2>&1 || true
}
trap cleanup EXIT
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
    docker exec falcon-sjtu bash -lc 'rostopic list' > /dev/null 2>&1 && break
    echo -n "."; sleep 2
done
echo " -- up"

# ── the bridge: ROS1 <-> ROS2, four topics ─────────────────────────────────
docker run -d --name sjtu-ros1-bridge \
    --network host \
    --env ROS_DOMAIN_ID="${SIM_DOMAIN_ID}" \
    --env ROS_MASTER_URI="http://localhost:11311" \
    --env ROS_HOSTNAME="localhost" \
    --volume "${SCRIPT_DIR}/config/bridge.yaml:/bridge.yaml:ro" \
    "${BRIDGE_IMAGE}" \
    > /dev/null

echo "[falcon_sjtu] up. Follow with:"
echo "    docker logs -f falcon-sjtu"
echo "Logs mirrored to ${LOG_DIR}/falcon.log; Ctrl-C tears both containers down."
docker logs -f falcon-sjtu 2>&1 | tee "${LOG_DIR}/falcon.log"
