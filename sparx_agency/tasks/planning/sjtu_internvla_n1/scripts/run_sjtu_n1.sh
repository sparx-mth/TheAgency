#!/usr/bin/env bash
# ============================================================
# run_sjtu_n1.sh — fly the SJTU Gazebo drone under InternVLA-N1.
#
#   ./run_sjtu_n1.sh [world] [instruction]
#   ./run_sjtu_n1.sh no_roof_small_warehouse "go to the far shelf and stop"
#
# The whole point of this deployment: EVERYTHING runs on the CPU except the
# InternVLA-N1 model server, which owns the GPU (~8 GB) alone. This script
# enforces that -- it refuses to start until the card is empty, gives it to N1,
# and pins every other process (Gazebo, both ROS2 nodes) off the GPU.
#
# The chain it wires, which is NavDP's shape exactly:
#
#   Gazebo (SJTU warehouse, CPU) --RGB/depth/odom-->
#     n1_policy_node (CPU) --HTTP--> InternVLA-N1 server (GPU) --trajectory-->
#       /simple_drone/n1/trajectory (nav_msgs/Path, world) -->
#         trajectory_follower_node (CPU, pure pursuit) --> /simple_drone/cmd_vel
#
# It does NOT vendor Gazebo or the model server. Point SJTU_PROJECT_DIR at the
# sim checkout; have the InternVLA-N1 server runnable (conda env `internnav`) or
# set N1_SERVER_CMD to start it.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

say() { echo "[sjtu_n1] $*"; }
die() { echo "[sjtu_n1] ERROR: $*" >&2; exit 1; }

WORLD="${1:-no_roof_small_warehouse}"
# Falls through to the binding YAML's `default_instruction` when neither the
# command line nor $INSTRUCTION names one, rather than inventing a warehouse
# order for a hospital.
INSTRUCTION="${2:-${INSTRUCTION:-}}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/sparx_agency/robots/SJTU/config/vla/internvla_n1.yaml}"

# ── environment the sim and the nodes must share ──────────────────────────
# The SJTU sim runs in Docker; the host nodes must join the SAME domain with the
# SAME middleware or they see nothing -- silently, exactly like a dead stack.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
export DISPLAY="${DISPLAY:-:1}"   # Gazebo Classic disables its cameras with no X
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
SETUP_DIR="${REPO_ROOT}/sparx_agency/robots/SJTU/setup"

# The middleware is CHOSEN FROM WHAT IS INSTALLED, not assumed. CycloneDDS is
# the better answer (it is what the sim image prefers and the only one the ROS 1
# bridge can reach) but ros-${ROS_DISTRO}-rmw-cyclonedds-cpp is not installed
# everywhere, and asking for an absent RMW aborts rclpy with a dlopen error
# three screens into a redirected log. Fast DDS is the fallback and works fine
# for anything that stays inside ROS 2.
if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
    if [[ -f "/opt/ros/${ROS_DISTRO}/lib/librmw_cyclonedds_cpp.so" ]]; then
        export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
    else
        export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
    fi
fi

# Shared memory does not cross the container boundary between the sim's Fast DDS
# and the host's: discovery succeeds over multicast so `ros2 topic list` is full,
# then every sample is dropped. Both profiles below switch shared memory off.
# See the header of each file.
case "${RMW_IMPLEMENTATION}" in
    rmw_cyclonedds_cpp)
        export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${SETUP_DIR}/cyclonedds_no_shm.xml}"
        SIM_RMW="cyclonedds" ;;
    rmw_fastrtps_cpp)
        export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${SETUP_DIR}/fastdds_udp_only.xml}"
        export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"
        SIM_RMW="fastrtps" ;;
    *)
        die "RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION}' is not one this stack knows how to configure." ;;
esac

N1_HOST="${N1_HOST:-127.0.0.1}"
N1_PORT="${N1_PORT:-8087}"
LOG_DIR="${SJTU_N1_LOG_DIR:-/tmp/sjtu_n1}"
mkdir -p "${LOG_DIR}"

# Recording: RECORD=1 also writes an MP4 (drone camera + N1 route + S1/S2 FPS)
# and a rosbag. RECORD_SECONDS>0 flies for that long then tears down and reports;
# 0 flies until Ctrl-C.
RECORD="${RECORD:-0}"
RECORD_SECONDS="${RECORD_SECONDS:-0}"
RECORD_OUTPUT="${RECORD_OUTPUT:-${LOG_DIR}/run.mp4}"
BAG_DIR="${BAG_DIR:-${LOG_DIR}/bag_$(date +%H%M%S)}"
# The recorder starts before takeoff and must not stop before the flight does,
# so its own deadline is the flight plus the bring-up it sits through. It is a
# BACKSTOP, not the normal path -- cleanup's SIGINT closes the file first when
# the shutdown goes the way it is supposed to.
# The trailing ".0" is load-bearing: the recorder declares `record_seconds` as a
# DOUBLE, and `record_seconds:=90` is parsed as an INTEGER, which rclpy refuses
# with InvalidParameterTypeException before the node has done anything at all.
# SIXTY, not thirty. The recorder's clock starts when the NODE is constructed;
# the flight clock only starts after `sleep 4`, the bag, ensure_flying (up to 21 s
# by default) and 3 s of instruction publishing -- about 28 s of bring-up against
# a 30 s lead. A slow takeoff then closes the video BEFORE the flight ends, and
# the launch file's `on_exit=Shutdown()` on the recorder takes the policy and the
# follower down with it. Overshooting costs nothing: cleanup's SIGINT closes the
# file first on every normal teardown.
if [[ "${RECORD_SECONDS}" -gt 0 ]]; then
    RECORDER_SECONDS="$(( RECORD_SECONDS + ${RECORDER_LEAD_S:-60} )).0"
else
    RECORDER_SECONDS="0.0"
fi


# Validate before the first `[[ x -gt 0 ]]`. Under `set -u`, arithmetic on a
# non-numeric string does not fail the test -- it aborts the shell with
# "bash: atrium: unbound variable" and exit 127, which for `_await` would mean
# aborting in the middle of the teardown, before the aircraft has been stopped.
for _var in RECORD_SECONDS SHUTDOWN_WAIT_S TAKEOFF_TRIES RECORDER_LEAD_S N1_WAIT_TRIES; do
    _val="$(eval "printf '%s' \"\${${_var}:-0}\"")"
    [[ "${_val}" =~ ^[0-9]+$ ]] || die "${_var}='${_val}' is not a whole number of seconds/tries."
done
unset _var _val

say "domain ${ROS_DOMAIN_ID}  rmw ${RMW_IMPLEMENTATION}  display ${DISPLAY}"

# A follower left over from an earlier run is the worst possible starting state:
# two publishers on `/simple_drone/cmd_vel`, each convinced it is flying the
# aircraft, and every result after that is noise. Refuse rather than join in.
# Scoped to THIS domain: the pattern alone matches a run in another terminal on
# another ROS_DOMAIN_ID, and killing that one leaves ITS aircraft flying.
EXISTING="$(pgrep -f 'sparx_agency.tasks.planning.sjtu_internvla_n1.ros2' 2>/dev/null \
    | while read -r _p; do
          tr '\0' '\n' < "/proc/${_p}/environ" 2>/dev/null \
              | grep -qx "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" && printf '%s ' "${_p}"
      done)"
if [[ -n "${EXISTING// /}" ]]; then
    if [[ "${KILL_STALE_NODES:-1}" == "1" ]]; then
        say "killing stale nodes from a previous run: ${EXISTING}"
        # shellcheck disable=SC2086
        kill -9 ${EXISTING} 2>/dev/null || true
        sleep 2
    else
        die "nodes from a previous run are still up (${EXISTING}). Two followers on
  /simple_drone/cmd_vel make every result meaningless. Kill them, or rerun with
  KILL_STALE_NODES=1."
    fi
fi

# ── 1. GPU preflight: the card must be free for N1 ────────────────────────
# ...but only when N1 is not already holding it. The server keeps ~6 GB resident
# between runs and takes a couple of minutes to reload, so demanding an empty
# card in front of a *healthy* server would force a pointless restart -- and
# then fail, because the check cannot tell the server it wants from the server
# that is there.
# The server is single-worker uvicorn and the agent holds the GIL through a
# System-2 forward pass, so a healthy server routinely takes seconds to answer
# anything at all. A single short probe therefore reports a *busy* server as a
# dead one -- which, one line later, aborts the run for "the GPU is not free"
# while pointing at the very server it just failed to reach.
n1_healthy() {
    local i
    for i in 1 2 3; do
        curl -sf -m 8 "http://${N1_HOST}:${N1_PORT}/openapi.json" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}
if n1_healthy; then
    say "InternVLA-N1 server already healthy at ${N1_HOST}:${N1_PORT}; leaving the card to it"
elif [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    say "checking the GPU is empty (N1 needs the whole card)..."
    python3 "${SCRIPT_DIR}/check_gpu_free.py" --require-empty \
        || die "GPU is not free. Free it (or SKIP_GPU_CHECK=1 to override), then rerun."
fi

# ── 2. Gazebo warehouse on the CPU ────────────────────────────────────────
# SJTU has no GPU physics; the world comes up on the CPU. Started only if a
# sim container is not already running, so a world left up between runs is
# reused. Set START_SIM=0 to manage the world yourself in another terminal.
sim_up() { docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' 2>/dev/null | grep -q .; }
# A container in `docker ps` is not a simulator. gzserver, the spawn and the
# plugin take tens of seconds after that, and handing an empty graph to
# ensure_flying makes a healthy world look like a capsized aircraft.
world_ready() { grep -qa "drone plugin finished loading" "${LOG_DIR}/gazebo.log" 2>/dev/null; }
if sim_up; then
    say "SJTU sim already running: $(docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' | head -n1)"
elif [[ "${START_SIM:-1}" == "1" ]]; then
    [[ -d "${SJTU_PROJECT_DIR}" ]] || die "SJTU_PROJECT_DIR=${SJTU_PROJECT_DIR} not found. Point it at the sim checkout."
    say "bringing up Gazebo world '${WORLD}' on the CPU (log: ${LOG_DIR}/gazebo.log)..."
    nohup bash "${REPO_ROOT}/sparx_agency/robots/SJTU/setup/bringup_world.sh" \
        --skip-build --domain "${ROS_DOMAIN_ID}" --rmw "${SIM_RMW}" "${WORLD}" \
        > "${LOG_DIR}/gazebo.log" 2>&1 &
    say "waiting for the world to publish (odom + camera)..."
    for _ in $(seq 1 90); do world_ready && break; sleep 2; done
    world_ready || die "the SJTU sim did not come up. See ${LOG_DIR}/gazebo.log"
    sleep 5
else
    die "no SJTU sim running and START_SIM=0. Bring it up first: bringup_world.sh ${WORLD}"
fi

# ── 3. InternVLA-N1 model server on the GPU ───────────────────────────────
if n1_healthy; then
    say "InternVLA-N1 server healthy at ${N1_HOST}:${N1_PORT}"
elif [[ -n "${N1_SERVER_CMD:-}" ]]; then
    say "starting the InternVLA-N1 server: ${N1_SERVER_CMD} (log: ${LOG_DIR}/n1_server.log)"
    nohup bash -lc "${N1_SERVER_CMD}" > "${LOG_DIR}/n1_server.log" 2>&1 &
else
    say "InternVLA-N1 server is not up at ${N1_HOST}:${N1_PORT}."
    say "Start it on the GPU (conda env 'internnav'), e.g.:"
    say "    cd <InternNav>/code   # start_server.py does sys.path.append('.')"
    say "    INTERNVLA_N1_4BIT=1 CUDA_VISIBLE_DEVICES=0 conda run -n internnav \\"
    say "        python scripts/eval/start_server.py --host 127.0.0.1 --port ${N1_PORT}"
    # NO --config, and INTERNVLA_N1_4BIT is not optional on an 8 GB card. This
    # hint used to say `--config h1_internvla_n1_async_cfg.py --port 8087`, which
    # is wrong twice over and cost a whole flight: start_server.py OVERWRITES
    # --port with eval_cfg.agent.server_port (8023), so the stack waits at 8087
    # for a server that is listening elsewhere; and without the 4-bit env the
    # loader takes the stock bf16 branch, which asks for flash_attn and wants
    # 16.8 GB. Both fail as `/agent/init` 500s while GET /openapi.json answers
    # happily -- "the server is up" proves nothing. See
    # tasks/planning/vlas/internvla_n1/upstream/README.md.
    say "or export N1_SERVER_CMD='<that command>' and rerun. Waiting for it..."
fi
say "waiting for the InternVLA-N1 server to answer..."
for _ in $(seq 1 "${N1_WAIT_TRIES:-120}"); do n1_healthy && break; sleep 2; done
n1_healthy || die "InternVLA-N1 server never became healthy at ${N1_HOST}:${N1_PORT}."
say "InternVLA-N1 server is up."

# ── 4. verify the GPU is the server's alone ───────────────────────────────
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
    python3 "${SCRIPT_DIR}/check_gpu_free.py" --allow internnav --allow python --allow start_server \
        || say "WARNING: a process other than the N1 server is on the GPU (see above)."
fi

# ── 5. the two CPU nodes: policy trajectory + follower ────────────────────
# `set -u` and ROS 2's setup.bash are incompatible: ament's shell hooks read
# unset variables, and under -u an unbound reference EXITS the shell outright
# rather than returning non-zero -- so `|| die` never runs and the script
# vanishes after its last successful line with no error at all. Take -u off for
# the source and put it straight back.
[[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]] || die "no /opt/ros/${ROS_DISTRO}/setup.bash"
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
command -v ros2 >/dev/null || die "ros2 is not on PATH after sourcing /opt/ros/${ROS_DISTRO}/setup.bash"
export CUDA_VISIBLE_DEVICES=""            # both nodes: CPU only, no GPU
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# `setsid` puts the launch and every node under it in their OWN process group,
# so the teardown below can signal the GROUP rather than one process. This is
# not tidiness. Signalling only `ros2 launch` orphans its children: they keep
# running, keep publishing `/simple_drone/cmd_vel`, and the NEXT run starts with
# two followers fighting over the aircraft's only control input. Observed here
# after five runs -- five live followers, one drone, and a capsize nobody could
# explain because each run looked correct in isolation.
say "launching the N1 policy + follower nodes (CPU, domain ${ROS_DOMAIN_ID}, ${RMW_IMPLEMENTATION})..."
# The launch runs in its OWN session, so the teardown can signal the whole
# group and no node is ever orphaned into the next run. `setsid` FORKS when its
# caller is already a group leader, so `$!` is not reliably the launch -- it can
# be a wrapper that exits immediately, after which `kill -0 $!` reads as "the
# nodes died at startup" while they are running perfectly well. So the session
# leader records its own pid, and `exec` hands that pid straight to ros2 launch.
NODES_PIDFILE="${LOG_DIR}/nodes.pid"
rm -f "${NODES_PIDFILE}"
setsid bash -c 'trap - INT; echo $$ > "$1"; shift; exec "$@"' _ "${NODES_PIDFILE}" \
    ros2 launch "${PKG_DIR}/launch/sjtu_internvla_n1.launch.py" \
    config_file:="${CONFIG_FILE}" \
    record:="$([[ "${RECORD}" == "1" ]] && echo true || echo false)" \
    record_output:="${RECORD_OUTPUT}" \
    record_seconds:="${RECORDER_SECONDS}" > "${LOG_DIR}/nodes.log" 2>&1 &
for _ in 1 2 3 4 5 6 7 8 9 10; do [[ -s "${NODES_PIDFILE}" ]] && break; sleep 0.3; done
NODES_PID="$(cat "${NODES_PIDFILE}" 2>/dev/null || echo "$!")"
NODES_PGID="${NODES_PID}"   # setsid made it a session and group leader
OWN_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
if [[ -z "${NODES_PGID}" || "${NODES_PGID}" == "${OWN_PGID}" ]]; then
    NODES_PGID=""
    say "NOTE: the nodes share this script's process group; falling back to per-process teardown."
fi
BAG_PID=""
CLEANED=0
cleanup() {
    [[ "${CLEANED}" == "1" ]] && return 0
    CLEANED=1
    say "tearing down nodes"
    # ORDER MATTERS. Kill the follower first, then stop the aircraft, then land.
    # The SJTU plugin holds the last twist it was given for ever -- there is no
    # failsafe reachable from outside it -- so tearing the nodes down without
    # this leaves the drone flying its final command into a wall, and the next
    # run starts from wherever that ended.
    # SIGINT, never SIGTERM, and then WAIT. `ros2 launch` turns SIGINT into an
    # orderly shutdown of its children; the recorder needs that to release its
    # VideoWriter (an mp4 with no moov atom plays nowhere) and `ros2 bag` needs
    # it to write metadata.yaml (without which the bag cannot even be opened).
    # Both failures produce a plausibly-sized file and no usable recording.
    # The process GROUP: the launch and every node it started.
    if [[ -n "${NODES_PGID}" ]]; then
        kill -INT -- "-${NODES_PGID}" 2>/dev/null || true
    fi
    kill -INT "${NODES_PID}" 2>/dev/null || true
    # The bag stops with the flight, not after the landing descent -- otherwise
    # every recording ends with the aircraft sinking to the floor.
    if [[ -n "${BAG_PID}" ]]; then
        kill -INT "${BAG_PID}" 2>/dev/null || true
    fi
    _await() {  # $1 pid, $2 seconds
        local i=0
        while [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && (( i < $2 * 4 )); do
            sleep 0.25; i=$((i + 1))
        done
        kill -0 "$1" 2>/dev/null && { say "forcing down pid $1"; kill -9 "$1" 2>/dev/null; }
    }
    _await "${NODES_PID}" "${SHUTDOWN_WAIT_S:-15}"
    _await "${BAG_PID}" "${SHUTDOWN_WAIT_S:-15}"
    BAG_PID=""
    # Nothing from this run may outlive it. A node that survives here is not a
    # leak, it is a second publisher on the aircraft's only control input.
    if [[ -n "${NODES_PGID}" ]]; then
        kill -9 -- "-${NODES_PGID}" 2>/dev/null || true
    fi
    local strays
    strays="$(pgrep -f 'sparx_agency.tasks.planning.sjtu_internvla_n1.ros2' 2>/dev/null \
        | while read -r p; do
              tr '\0' '\n' < "/proc/${p}/environ" 2>/dev/null \
                  | grep -qx "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" && printf '%s ' "${p}"
          done)"
    if [[ -n "${strays// /}" ]]; then
        say "killing stray nodes from this run: ${strays}"
        # shellcheck disable=SC2086
        kill -9 ${strays} 2>/dev/null || true
    fi
    sleep 1
    for _ in 1 2 3 4 5; do
        timeout 5 ros2 topic pub --once -w 0 /simple_drone/cmd_vel geometry_msgs/msg/Twist "{}" \
            >/dev/null 2>&1 || true
        sleep 0.2
    done
    if [[ "${LAND_ON_EXIT:-1}" == "1" ]]; then
        say "landing"
        timeout 5 ros2 topic pub --once -w 0 /simple_drone/land std_msgs/msg/Empty "{}" >/dev/null 2>&1 || true
        sleep "${LAND_SETTLE_S:-4}"
    fi
    sleep 2   # let the recorder flush and close the MP4
}
# EXIT and HUP as well as INT/TERM. The nodes run in their own session
# (`setsid`), so a closed terminal or a dropped ssh does NOT reach them: without
# EXIT/HUP here the script dies and the follower keeps flying the aircraft with
# nobody left to stop it. `CLEANED` makes the handler re-entrant.
trap cleanup EXIT INT TERM HUP
sleep 4
kill -0 "${NODES_PID}" 2>/dev/null || die "the ROS2 nodes exited at startup. See ${LOG_DIR}/nodes.log"

report() {
    echo
    say "=================== run summary ==================="
    if [[ "${RECORD}" == "1" ]]; then
        if [[ -s "${RECORD_OUTPUT}" ]]; then
            say "video : ${RECORD_OUTPUT}  ($(du -h "${RECORD_OUTPUT}" | cut -f1))"
        else
            say "video : MISSING (${RECORD_OUTPUT}) -- the recorder node never wrote one."
        fi
        say "rosbag: ${BAG_DIR}"
    fi
    # `grep -c` prints its count AND exits 1 on zero, so `|| echo 0` yields the
    # two-line string "0\n0" -- and the `== "0"` test below is then FALSE on
    # exactly the run this warning exists for.
    local commits
    commits="$(grep -ac 'committed #' "${LOG_DIR}/nodes.log" 2>/dev/null | head -n1)"
    commits="${commits:-0}"
    say "N1 route commitments: ${commits}"
    if [[ "${commits}" == "0" ]]; then
        say "  NO TRAJECTORY WAS EVER COMMITTED -- the flight is not a result."
        say "  Check ${LOG_DIR}/nodes.log for the policy node, and ${LOG_DIR}/n1_server.log"
        say "  for HTTP 500s (a System-1 step that fails still answers 200 on the"
        say "  queued discrete actions, so 'the server is up' proves nothing)."
    fi
    # WHAT KIND of routes, not just how many. A run of 0.25 m stubs and a run
    # of 2 m curves both report "22 commitments" and are not the same result;
    # this is the line that separates them, and it is why `sys1_continuous_only`
    # and the rotation mode exist at all.
    local curves actions turns escapes blocked metres
    curves="$(grep -ac '\[curve\]' "${LOG_DIR}/nodes.log" 2>/dev/null | head -n1)"; curves="${curves:-0}"
    actions="$(grep -ac '\[action\]' "${LOG_DIR}/nodes.log" 2>/dev/null | head -n1)"; actions="${actions:-0}"
    turns="$(grep -ac 'turn #' "${LOG_DIR}/nodes.log" 2>/dev/null | head -n1)"; turns="${turns:-0}"
    escapes="$(grep -ac 'BLOCKED ESCAPE' "${LOG_DIR}/nodes.log" 2>/dev/null | head -n1)"; escapes="${escapes:-0}"
    blocked="$(grep -ac 'HARD BLOCKED' "${LOG_DIR}/nodes.log" 2>/dev/null | head -n1)"; blocked="${blocked:-0}"
    metres="$(grep -ao 'committed #[0-9]*: [0-9]* pts, [0-9.]* m' "${LOG_DIR}/nodes.log" 2>/dev/null \
        | awk '{s += $(NF-1)} END {printf "%.1f", s+0}')"
    say "  routes flown        ${curves} curves + ${actions} action steps, ${metres:-0} m of route"
    say "  rotations flown     ${turns}  (blocked-forward escapes: ${escapes}, hard blocks: ${blocked})"
    # HOW MUCH OF THE BUILDING THE FLIGHT LOOKED AT. Under an exploration order
    # this is the only line here that answers the instruction: the others say
    # what the aircraft did, this says what it achieved. The recorder writes
    # FINAL when it closes the video, which is the whole flight; the periodic
    # line is the fallback for a run whose recorder was killed outright.
    local seen
    seen="$(grep -ao 'N1 COVERAGE FINAL .*' "${LOG_DIR}/nodes.log" 2>/dev/null | tail -n1)"
    [[ -n "${seen}" ]] || seen="$(grep -ao 'N1 COVERAGE .*' "${LOG_DIR}/nodes.log" 2>/dev/null | tail -n1)"
    if [[ -n "${seen}" ]]; then
        say "  hospital seen       ${seen#N1 COVERAGE }"
    else
        say "  hospital seen       not measured (no map backdrop, or recorder.coverage off)"
    fi
    local fps_line
    fps_line="$(grep -a 'N1 FPS' "${LOG_DIR}/nodes.log" 2>/dev/null | tail -n1)"
    if [[ -n "${fps_line}" ]]; then
        say "measured: ${fps_line#*N1 FPS}"
    else
        say "measured FPS: none yet (no N1 step logged — check the server & camera)."
    fi
    say "reference (this machine, ~/trt/internnav/REPORT.md):"
    say "  System 1: 6.77 Hz torch -> 22.99 Hz TensorRT   System 2: ~1.4 Hz (dual-system 1.41 Hz)"
    say "==================================================="
}

# ── 6. take off, then hand the instruction to N1 ──────────────────────────
if [[ "${RECORD}" == "1" ]]; then
    # The raw camera topics are NOT bagged by default. 600x600 RGB plus 600x600
    # float32 depth is ~11 GB per minute here, which fills a disk in an
    # afternoon of five-run campaigns -- and the MP4 already carries the camera.
    # Everything below is kilobytes: the decisions, the route and the pose, which
    # is what an offline analysis of a flight actually needs. BAG_IMAGES=1 puts
    # the images back for a run you intend to re-infer from.
    BAG_TOPICS=(
        /simple_drone/odom /simple_drone/cmd_vel /simple_drone/state
        /simple_drone/n1/trajectory /simple_drone/n1/trajectory_full
        /simple_drone/n1/info /simple_drone/navigation/instruction
    )
    if [[ "${BAG_IMAGES:-0}" == "1" ]]; then
        BAG_TOPICS+=(/simple_drone/front/image_raw /simple_drone/front_depth/depth/image_raw)
        say "BAG_IMAGES=1: bagging the raw camera too (~11 GB/min)"
    fi
    say "recording rosbag -> ${BAG_DIR}"
    # `( trap - INT; exec ... ) &` -- the subshell and the trap reset are both
    # load-bearing. A NON-INTERACTIVE shell starts every background job with
    # SIGINT set to IGNORE (POSIX; it is what stops a Ctrl-C in a script from
    # killing its own background children), so `kill -INT` on this pid is a
    # silent no-op. rosbag2 stops on SIGINT and finalises: it writes the
    # remaining cache, the mcap footer and `metadata.yaml`. Without that it is
    # eventually SIGKILLed, and what is left on disk is a bag with a truncated
    # final record and NO metadata -- `ros2 bag info` refuses to open it, and
    # every run of a campaign is unreadable. Measured: 4 of 5 runs unusable.
    ( trap - INT; exec ros2 bag record -o "${BAG_DIR}" "${BAG_TOPICS[@]}" ) \
        > "${LOG_DIR}/bag.log" 2>&1 &
    BAG_PID=$!
fi

# Takeoff is CONFIRMED, not assumed. The SJTU plugin silently drops the command
# from the wrong state, and nothing says so: the aircraft sits on the floor at
# state 0 while the policy commits route after route to it and the follower
# publishes a twist the plugin ignores while landed. A whole 90 s recording has
# been lost to exactly this. /simple_drone/state is 0 landed, 1 flying.
say "commanding takeoff..."
python3 "${SCRIPT_DIR}/ensure_flying.py" --tries "${TAKEOFF_TRIES:-8}" \
        --settle "${TAKEOFF_SETTLE_S:-5}" 2>&1 | sed 's/^/[sjtu_n1] /'
TAKEOFF_STATUS=${PIPESTATUS[0]}
# 2 and 1 are different problems with the same symptom, and sending an operator
# to restart a perfectly healthy world over a DDS mismatch wastes an evening.
if [[ "${TAKEOFF_STATUS}" == "2" ]]; then
    cleanup
    die "no telemetry from the drone at all. The world may be up but unreachable: check \
ROS_DOMAIN_ID (${ROS_DOMAIN_ID}), RMW_IMPLEMENTATION (${RMW_IMPLEMENTATION}) and the DDS \
profile against the container, and see ${LOG_DIR}/gazebo.log."
elif [[ "${TAKEOFF_STATUS}" != "0" ]]; then
    cleanup
    die "the drone never got airborne. A capsized aircraft from a previous run cannot \
take off and /simple_drone/reset does NOT right it -- restart the world."
fi
# The policy node subscribes VOLATILE, depth 1: an instruction published before
# it is listening is simply gone, and the node then flies the config file's
# default -- a warehouse order, in a hospital. Publish it several times, and
# never with the default `-w 1`, which in Jazzy waits for a matching subscriber
# with NO timeout and hangs the whole script when the node has died.
if [[ -z "${INSTRUCTION}" ]]; then
    say "no instruction given; the nodes fly the config's default_instruction"
else
say "sending instruction: '${INSTRUCTION}'"
for _ in 1 2 3; do
    timeout 10 ros2 topic pub --once -w 0 /simple_drone/navigation/instruction std_msgs/msg/String \
        "{data: '${INSTRUCTION}'}" >/dev/null 2>&1 || say "WARNING: could not publish the instruction"
    sleep 1
done
fi

say "flying. N1 route -> /simple_drone/n1/trajectory ; cmd -> /simple_drone/cmd_vel"
[[ "${RECORD}" == "1" ]] && say "camera+route video -> ${RECORD_OUTPUT}"
say "logs: ${LOG_DIR}/nodes.log  (gazebo: ${LOG_DIR}/gazebo.log)"
if [[ "${RECORD_SECONDS}" -gt 0 ]]; then
    say "recording for ${RECORD_SECONDS}s, then tearing down..."
    sleep "${RECORD_SECONDS}"
    cleanup
else
    say "Ctrl-C to stop the nodes (the world and the server are left up)."
    wait "${NODES_PID}"
    cleanup
fi
report

