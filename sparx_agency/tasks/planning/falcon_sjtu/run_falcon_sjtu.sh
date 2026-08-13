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
# The domain the bridge must join: the one the SIM CONTAINER IS ACTUALLY ON.
# A mismatch is not an error -- it is silently zero data on every topic, which
# reads as a dead stack for as long as you care to look.
#
# Read it off the running container rather than guessing, because every guess
# is wrong in some real case: the sim follows $ROS_DOMAIN_ID via
# robots/SJTU/setup/env.sh, EXCEPT when it was pinned with `--domain N`, and a
# hardcoded 20 here splits from both. Order: an explicit SIM_DOMAIN_ID, then
# the live container, then $ROS_DOMAIN_ID, then 20.
SIM_ENV_DOMAIN=""
SIM_CONTAINER="$(docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' 2>/dev/null | head -n1)"
if [[ -n "${SIM_CONTAINER}" ]]; then
    SIM_ENV_DOMAIN="$(docker inspect "${SIM_CONTAINER}" \
        --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
        | sed -n 's/^ROS_DOMAIN_ID=//p' | head -n1)"
fi
SIM_DOMAIN_ID="${SIM_DOMAIN_ID:-${SIM_ENV_DOMAIN:-${ROS_DOMAIN_ID:-20}}}"
if [[ -z "${SIM_CONTAINER}" ]]; then
    echo "[falcon_sjtu] WARNING: no sjtu_drone_* container is running, so the bridge's"
    echo "  domain (${SIM_DOMAIN_ID}) is a guess. Start the world FIRST, or the bridge"
    echo "  joins an empty domain and FALCON never receives a depth frame."
else
    echo "[falcon_sjtu] sim ${SIM_CONTAINER} is on ROS_DOMAIN_ID=${SIM_DOMAIN_ID}; bridging there"
fi

# A domain SHARED with other ROS 2 work is not safe here: Foxy's
# parameter_bridge segfaults during discovery against participants it cannot
# map (measured on this machine -- domain 5, the everyday working domain, dies
# instantly and restarts forever; domain 20 brings up all 12 bridges). The
# post-start check below turns that crash loop into a message instead of a
# mystery.
LOG_DIR="${FALCON_SJTU_LOG_DIR:-/tmp/falcon_sjtu}"
mkdir -p "${LOG_DIR}"
# FOLLOW=1 (default when interactive) tails the FALCON log and tears the stack
# down on Ctrl-C. FOLLOW=0 starts everything detached and returns, leaving the
# containers up for a monitor/soak to watch -- which is how the iterate loop
# drives it.
FOLLOW="${FOLLOW:-1}"
# RVIZ=1 (default when there is a display) opens FALCON's own RViz view (map,
# frontiers, trajectories, the drone) in a sibling container on the same
# roscore. It is started BEFORE the bridge, so the view is up before the first
# depth frame is mapped. RVIZ=0 for headless/soak runs.
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    RVIZ="${RVIZ:-1}"
else
    RVIZ="${RVIZ:-0}"
fi

if [[ ! -f "${SCRIPT_DIR}/config/${MAP_NAME}.yaml" ]]; then
    echo "[ERROR] no map config: ${SCRIPT_DIR}/config/${MAP_NAME}.yaml" >&2
    exit 2
fi

chmod +x "${SCRIPT_DIR}"/adapter/scripts/*.py 2>/dev/null || true

# ── per-world planner clearance ───────────────────────────────────────────
# safe_distance is the ONLY clearance FALCON's optimiser respects, and the
# right value is a property of the WORLD, not of this stack: the rule is
# safe_distance <= narrowest_half_width, or the optimiser is asked for margin
# the corridor cannot provide and trades it away unpredictably.
#
#   warehouse  aisles ~1.4 m wide  -> half-width 0.70. Flown at 0.15 / 0.55 /
#              0.70 / 0.85: retreats 15 / 9 / 7 / 3 and bubble breaches
#              many / - / 2 / 1, so more clearance keeps monotonically buying
#              fewer conflicts with the follower's gates. 0.85 is chosen over
#              upstream's 0.70 for that, and the cost is visible and bounded:
#              it EXCEEDS the aisle half-width, so A* intermittently has no
#              legal route and logs "No path to next viewpoint" (70 in one
#              run, 0 at 0.70). FALCON replans through it and the mission
#              still finished 154.2 m3 in 100 s. Drop to 0.70 if those
#              failures ever stop being transient.
#   hospital   doorways 0.9 m      -> half-width 0.45. 0.70 makes a doorway
#              unplannable; keep it under 0.45.
#
# An explicit safe_distance in FALCON_LAUNCH_ARGS always wins.
if [[ "${FALCON_LAUNCH_ARGS:-}" != *safe_distance* ]]; then
    case "${MAP_NAME}" in
        warehouse)              MAP_SAFE_DISTANCE="0.85" ;;
        hospital|small_house)   MAP_SAFE_DISTANCE="0.40" ;;
        *)                      MAP_SAFE_DISTANCE="" ;;
    esac
    if [[ -n "${MAP_SAFE_DISTANCE}" ]]; then
        FALCON_LAUNCH_ARGS="safe_distance:=${MAP_SAFE_DISTANCE} ${FALCON_LAUNCH_ARGS:-}"
        echo "[falcon_sjtu] ${MAP_NAME}: planner clearance safe_distance:=${MAP_SAFE_DISTANCE}"
    fi
fi

cleanup() {
    docker rm -f falcon-sjtu sjtu-ros1-bridge falcon-rviz > /dev/null 2>&1 || true
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
    --cap-add=SYS_PTRACE \
    --env PYTHONPATH="/opt/sparx_agency" \
    --env PYTHONUNBUFFERED=1 \
    --volume "${REPO_ROOT}:/opt/sparx_agency:ro" \
    --volume "${SCRIPT_DIR}/adapter/scripts/bspline_follower_node.py:${SCRIPTS_TARGET}/bspline_follower_node.py" \
    --volume "${SCRIPT_DIR}/adapter/scripts/sensor_pose_node.py:${SCRIPTS_TARGET}/sensor_pose_node.py" \
    --volume "${SCRIPT_DIR}/adapter/launch/exploration.launch:${LAUNCH_TARGET}/exploration.launch" \
    --volume "${SCRIPT_DIR}/adapter/launch/bspline_follower.launch:${LAUNCH_TARGET}/bspline_follower.launch" \
    --volume "${SCRIPT_DIR}/config/${MAP_NAME}.yaml:/catkin_ws/src/FALCON/falcon_planner/exploration_manager/config/map/${MAP_NAME}.yaml" \
    "${FALCON_IMAGE}" \
    roslaunch falcon_adapter exploration.launch map_name:="${MAP_NAME}" ${FALCON_LAUNCH_ARGS:-} \
    > /dev/null

echo -n "[falcon_sjtu] waiting for roscore"
for _ in $(seq 1 30); do
    docker exec falcon-sjtu bash -lc 'source /opt/ros/noetic/setup.bash >/dev/null 2>&1; rostopic list' > /dev/null 2>&1 && break
    echo -n "."; sleep 2
done
echo " -- up"

# ── RViz: FALCON's own exploration view, on the same roscore ───────────────
# A sibling container from the FALCON image (the config and the rviz binary are
# both in there), sharing the host X socket. Software GL: the image has mesa but
# no GPU bindings, and rviz's displays are cheap enough without one. The drone
# model itself comes from the odom_visualization node that exploration.launch
# starts, fed by /odom_world.
if [[ "${RVIZ}" == "1" ]]; then
    command -v xhost > /dev/null && xhost +local: > /dev/null 2>&1 || true
    docker run -d --name falcon-rviz \
        --network host \
        --env DISPLAY="${DISPLAY}" \
        --env QT_X11_NO_MITSHM=1 \
        --env LIBGL_ALWAYS_SOFTWARE=1 \
        --env ROS_MASTER_URI="http://localhost:11311" \
        --env ROS_HOSTNAME="localhost" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${FALCON_IMAGE}" \
        roslaunch exploration_manager rviz.launch \
        > /dev/null
    echo "[falcon_sjtu] rviz up (RVIZ=0 to disable)"
fi

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

# ── did the bridge actually bridge anything? ──────────────────────────────
# parameter_bridge announces one "create bidirectional bridge" line per topic.
# None of them means it died before reading the list -- and because the
# entrypoint restarts it forever, the container stays "Up" while carrying zero
# data. That silence is the single most expensive failure in this stack, so it
# is checked rather than left to be discovered from an empty map.
BRIDGED=0
for _ in $(seq 1 15); do
    BRIDGED=$(docker logs sjtu-ros1-bridge 2>&1 | grep -c 'create bidirectional bridge' || true)
    [[ "${BRIDGED}" -gt 0 ]] && break
    sleep 2
done
if [[ "${BRIDGED}" -gt 0 ]]; then
    echo "[falcon_sjtu] bridge up: ${BRIDGED} topic bridges on domain ${SIM_DOMAIN_ID}"
else
    echo "[falcon_sjtu] ERROR: the bridge created NO topic bridges on domain ${SIM_DOMAIN_ID}." >&2
    echo "  It is restarting in a loop and FALCON will receive nothing -- no depth, no" >&2
    echo "  odometry, no /clock -- while every container still reads as Up." >&2
    echo "  Most likely: that domain is shared with other ROS 2 participants, and Foxy's" >&2
    echo "  parameter_bridge segfaults during discovery against them. Bring the world up" >&2
    echo "  on a domain of its own:" >&2
    echo "    bringup_world.sh --domain 20 <world>   (then rerun this script)" >&2
    echo "  Diagnose with: docker logs sjtu-ros1-bridge | tail" >&2
fi

echo "[falcon_sjtu] up (map=${MAP_NAME})."
if [[ "${FOLLOW}" == "0" ]]; then
    echo "[falcon_sjtu] detached; containers left running. Follow with:"
    echo "    docker logs -f falcon-sjtu   |   docker logs -f sjtu-ros1-bridge"
    echo "    tear down: docker rm -f falcon-sjtu sjtu-ros1-bridge falcon-rviz"
    exit 0
fi
echo "[falcon_sjtu] following falcon-sjtu; Ctrl-C tears both containers down."
echo "Logs mirrored to ${LOG_DIR}/falcon.log."
docker logs -f falcon-sjtu 2>&1 | tee "${LOG_DIR}/falcon.log"
