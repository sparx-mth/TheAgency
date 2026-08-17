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

# ── the launch files must be well-formed XML, checked HERE ────────────────
# roslaunch parses with a strict XML parser, so a malformed launch file kills
# the container at startup with a single RLException line -- and from outside,
# a container that exits in two seconds is indistinguishable from any other
# infrastructure failure. campaign_run.sh then spends its sim bring-up (about
# 100 s) before returning a bare "falcon container died".
#
# The trap is specific and easy to walk into while documenting a parameter:
# an XML comment may not contain a double hyphen, and may not contain a bare
# '<'. Both read as perfectly ordinary English prose. Two seconds of checking
# here turns a wasted run into a message naming the file, the line and the
# column.
for lf in "${SCRIPT_DIR}"/adapter/launch/*.launch; do
    if ! python3 -c "import sys,xml.dom.minidom as m; m.parse(sys.argv[1])" "${lf}" 2>/dev/null; then
        echo "[ERROR] ${lf} is not well-formed XML:" >&2
        python3 -c "import sys,xml.dom.minidom as m; m.parse(sys.argv[1])" "${lf}" 2>&1 \
            | tail -1 | sed 's/^/    /' >&2
        echo "  roslaunch will refuse it and the FALCON container will exit at startup." >&2
        echo "  Most likely a '--' or a bare '<' inside an XML comment: neither is legal," >&2
        echo "  and both are natural things to write in a prose explanation." >&2
        exit 2
    fi
done

chmod +x "${SCRIPT_DIR}"/adapter/scripts/*.py 2>/dev/null || true

# ── planner clearance: ONE value, both worlds ─────────────────────────────
# This used to be set per world -- 0.85 in the warehouse, 0.45 in the hospital
# -- on the belief that warehouse aisles are ~1.4 m and hospital doorways
# 0.9 m. The measurement says otherwise. The warehouse's six shelf aisles are
# 0.909, 0.916, 0.942, 1.035, 1.044 and 1.216 m (the "1.4 m" came from a shelf
# mesh read without its 90 degree node transform, which also rotated the whole
# rack; see config/warehouse.yaml), and its tightest clutter slots are 0.81 and
# 0.95 m. The hospital's 18 small doorways are 0.930 m. The two buildings ask
# for the SAME clearance, and the reason a single value looked impossible was a
# geometry error, not a conflict between the worlds.
#
# 0.45 is the doorway HALF-WIDTH, and it is chosen at the half-width rather
# than under it on purpose. The cost is
#     if (dist < safe_distance) cost += pow(dist - safe_distance, 2)
# which is strictly ONE-SIDED (verified in bspline_optimizer.cpp:271-289): at
# 0.40 in a 0.90 m opening the penalty is exactly zero anywhere within +-0.05 m
# of centre, so nothing pulls the curve to the middle and the follower starts
# its pass having already spent that budget. At the half-width the cost becomes
# delta^2 with its minimum on the centreline and a restoring gradient
# everywhere off it -- the optimiser threads the exact middle of the opening,
# which is what a 0.52 m airframe in a 0.90 m door needs.
#
# The old file also claimed 0.85 caused the warehouse's 70 "No path to next
# viewpoint" lines. It cannot have: safe_distance is read by the B-spline
# optimiser and by NOTHING ELSE in FALCON -- not astar.cpp, not
# path_cost_evaluator.cpp, not frontier_finder.cpp. What it really did at 0.85
# was leave a permanent non-zero distance residual (weight 50) fighting START
# (100), END (50) and SMOOTHNESS (20) on every curve, so the optimiser bought
# clearance by distorting routes it had no room to distort.
#
# ── 0.70 MEASURED BETTER IN THE WAREHOUSE. NOT SHIPPED. 2026-08-17. ────────
# The best safety result this campaign has produced, and it is left OFF because
# only half the evidence exists. Interleaved warehouse A/B, n=6 per arm, the
# live rosparam checked on every single leg:
#
#              objects/leg   CLEAN legs   finished   coverage   elapsed
#   0.45 (now)     3.33         0/6          5/6      378 m3     431 s
#   0.70           0.67         4/6          5/6      366 m3     295 s
#
# Objects touched -80%, bumper reports -87% (158 -> 20), the warehouse goes from
# NEVER producing a contact-free leg to four in six, finishes TIE, and runs are
# 32% faster, for ~3% less coverage. Both arms aborted exactly once, which is
# what showed the treatment's single abort was the world and not the change.
#
# Why it is not shipped anyway. This value was deliberately unified across both
# worlds (see above) because the warehouse's aisles (0.909-1.216 m) and the
# hospital's doorways (0.930 m) are the SAME size, with 0.45 chosen as the
# half-width so the one-sided cost has its minimum on the centreline. Raising it
# to 0.70 makes the constraint unsatisfiable in every opening in BOTH worlds --
# which is precisely the 0.85 failure described above, where a permanent cost
# residual bought clearance by distorting routes. The warehouse result
# CONTRADICTS that theory rather than confirming it, so the theory cannot be
# used to predict the hospital in either direction. The hospital A/B was flying
# when it was stopped to open RViz, so the hospital has NOT been measured.
#
# To finish this: SAFE_DISTANCE=0.70 vs 0.45, interleaved, 2+ rounds, hospital,
# cap 4200 s, checking `live=` per leg. Keep ONE value if the hospital is neutral
# or better; only split per-map if it regresses, and say plainly that doing so
# overturns a deliberate design decision.
#
# An explicit safe_distance in FALCON_LAUNCH_ARGS always wins.
if [[ "${FALCON_LAUNCH_ARGS:-}" != *safe_distance* ]]; then
    MAP_SAFE_DISTANCE="${SAFE_DISTANCE:-0.45}"
    FALCON_LAUNCH_ARGS="safe_distance:=${MAP_SAFE_DISTANCE} ${FALCON_LAUNCH_ARGS:-}"
    echo "[falcon_sjtu] ${MAP_NAME}: planner clearance safe_distance:=${MAP_SAFE_DISTANCE}"
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
    --volume "${SCRIPT_DIR}/adapter/scripts/mission_watchdog_node.py:${SCRIPTS_TARGET}/mission_watchdog_node.py" \
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
    # ── and did each of them SURVIVE? ──────────────────────────────────────
    # Counting the announcements is not enough, and the gap is not academic.
    # parameter_bridge prints "create bidirectional bridge for topic X" BEFORE
    # it looks for a conversion pair, and only then prints "failed ... No
    # template specialization for the pair". So a topic whose message type was
    # not compiled into the bridge image is counted as bridged, the check above
    # passes, every container reads as Up, and the topic carries nothing for the
    # whole run.
    #
    # /simple_drone/bumper_states is the one this costs. It is the follower's
    # only GROUND TRUTH that the airframe has touched something -- the retreat,
    # the three-strikes hold and the hand-off to FALCON's dead-end guard are all
    # keyed on it -- and gazebo_msgs is exactly the pair a bridge image gets
    # built without, because omitting it breaks no build. Measured on
    # LP-Boston-17093: a hospital run logging 157 bumper contacts against a
    # cubicle curtain and ZERO retreats, the aircraft grinding on geometry it had
    # no way to know it was touching, with the whole stack reporting health.
    #
    # Reported per topic rather than only for the bumper: the same silence would
    # follow any type this image was not built for, and the fix is always the
    # same one line in ros_bridge_docker/Dockerfile.
    BRIDGE_FAILED=$(docker logs sjtu-ros1-bridge 2>&1 \
        | sed -n "s/.*failed to create bidirectional bridge for topic '\([^']*\)'.*/\1/p" \
        | sort -u | tr '\n' ' ')
    if [[ -n "${BRIDGE_FAILED// /}" ]]; then
        echo "[falcon_sjtu] ERROR: these topics announced a bridge and then FAILED to build one:" >&2
        echo "    ${BRIDGE_FAILED}" >&2
        echo "  They will carry no data for the entire run while every container reads as Up." >&2
        docker logs sjtu-ros1-bridge 2>&1 | grep 'failed to create bidirectional bridge' \
            | sed 's/^/    /' >&2
        case "${BRIDGE_FAILED}" in
            *bumper_states*)
                echo "  bumper_states is SAFETY-CRITICAL: without it the follower cannot tell that" >&2
                echo "  it has hit anything, so its contact retreat, its three-strikes hold and" >&2
                echo "  FALCON's dead-end guard are all inference-only. Any contact number measured" >&2
                echo "  on this configuration is measuring a stack flying with that reflex removed." >&2
                echo "  Fix: add ros-noetic-gazebo-msgs and ros-foxy-gazebo-msgs to" >&2
                echo "  sjtu_project/ros_bridge_docker/Dockerfile and rebuild ${BRIDGE_IMAGE}." >&2
                ;;
        esac
    fi
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
