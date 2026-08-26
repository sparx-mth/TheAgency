#!/usr/bin/env bash
# ============================================================
# record_campaign.sh — a series of InternVLA-N1 recordings, one per hospital area.
#
#   ./record_campaign.sh                       # all five areas, 60 s each
#   ./record_campaign.sh 90 atrium south_hall  # 90 s each, two named areas
#   REPEAT=4 ./record_campaign.sh              # four passes over the five areas
#   REPEAT=0 ./record_campaign.sh              # keep cycling until stopped
#   REPEAT=5 ./record_campaign.sh 180 office_door   # the SAME prompt, five times
#
# That last form is the repeatability experiment: one area, one instruction,
# five runs. Each is hermetic -- the world is restarted and the aircraft
# re-ferried -- so the only thing that differs between them is the policy's own
# non-determinism, which is the thing being looked at.
#
# Each recording is HERMETIC: the world is restarted, the aircraft is ferried
# above the walls to the area, and only then does the policy get the instruction.
# That matters more here than it looks. The SJTU plugin has no failsafe and no
# way to right a capsized airframe -- /simple_drone/reset does NOT restore its
# attitude, measured -- so one bad contact would otherwise poison every
# subsequent recording with an aircraft lying on its side, reporting healthy
# odometry and ignoring every command.
#
# Outputs, one directory per run under $CAMPAIGN_DIR:
#   <NN>_<area>/run.mp4     camera + N1's route on the hospital map + S1/S2 FPS
#   <NN>_<area>/bag/        the decisions, the route and the pose
#   <NN>_<area>/run.log     the bring-up log, ending in the run summary
# plus campaign.log with one verdict line per run.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

SECONDS_PER_RUN="${1:-60}"
shift 2>/dev/null || true
AREAS=("$@")
if [[ ${#AREAS[@]} -eq 0 ]]; then
    AREAS=(atrium north_wing reception east_wards south_hall)
fi

CAMPAIGN_DIR="${CAMPAIGN_DIR:-${HOME}/sjtu_n1_recordings/$(date +%Y%m%d_%H%M%S)}"
WORLD="${WORLD:-hospital}"
# One room, one table -- an instruction with a state at which it is satisfied.
# "Explore the entire hospital" has none, so no flight can be evidence about it.
INSTRUCTION="${INSTRUCTION:-There is a room to your right. Enter it, go to the center of the room, find the table and stop near the table.}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
export DISPLAY="${DISPLAY:-:1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
SETUP_DIR="${REPO_ROOT}/sparx_agency/robots/SJTU/setup"

# SOURCE ROS 2 HERE, not in the operator's shell. This script runs `ros2 topic
# echo` to decide the world is really up and `goto_area.py` (which imports
# rclpy) to ferry the aircraft, and `/opt/ros/<distro>/bin/ros2` is on the
# default PATH *without* the environment that makes it work -- so from a plain
# shell it fails on an importlib traceback, the odom wait silently burns its
# sixty seconds, and every run is written off as "could not reach the area"
# with nothing anywhere pointing at the cause. It only ever worked because the
# person running it happened to have sourced ROS first.
#
# `set -u` and ament's shell hooks are incompatible: the hooks read unset
# variables, and under -u an unbound reference EXITS the shell outright rather
# than returning non-zero. Take -u off for the source and put it straight back.
if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    set -u
else
    echo "[campaign] ERROR: no /opt/ros/${ROS_DISTRO}/setup.bash" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""   # the ferry and the verdicts stay off the card

# The ferry runs BEFORE run_sjtu_n1.sh and therefore inherits none of its
# environment. Without the shared-memory-free profile it sees an empty graph,
# `wait_for_odom` raises after 20 s, and every run reports "could not reach the
# area" with nothing pointing at the cause. Pick the middleware here, once, and
# hand the same choice to the world and to the run script.
if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
    if [[ -f "/opt/ros/${ROS_DISTRO}/lib/librmw_cyclonedds_cpp.so" ]]; then
        export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
    else
        export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
    fi
fi
case "${RMW_IMPLEMENTATION}" in
    rmw_cyclonedds_cpp)
        export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${SETUP_DIR}/cyclonedds_no_shm.xml}"
        SIM_RMW="cyclonedds" ;;
    rmw_fastrtps_cpp)
        export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${SETUP_DIR}/fastdds_udp_only.xml}"
        export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"
        SIM_RMW="fastrtps" ;;
    *)
        echo "[campaign] ERROR: RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION}' is not one this stack configures." >&2
        exit 2 ;;
esac

[[ "${SECONDS_PER_RUN}" =~ ^[0-9]+$ ]] || {
    echo "[campaign] ERROR: the first argument is SECONDS PER RUN, got '${SECONDS_PER_RUN}'." >&2
    echo "  Usage: record_campaign.sh [seconds] [area ...]" >&2
    exit 2
}

mkdir -p "${CAMPAIGN_DIR}"
LOG="${CAMPAIGN_DIR}/campaign.log"
say() { echo "[campaign] $*" | tee -a "${LOG}"; }

say "output      ${CAMPAIGN_DIR}"
say "areas       ${AREAS[*]}"
say "seconds     ${SECONDS_PER_RUN} per run"
say "instruction ${INSTRUCTION}"

restart_world() {
    say "restarting the world (a fresh, level aircraft for every run)"
    docker rm -f "sjtu_drone_${WORLD}" >/dev/null 2>&1 || true
    sleep 3
    nohup bash "${REPO_ROOT}/sparx_agency/robots/SJTU/setup/bringup_world.sh" \
        --skip-build --domain "${ROS_DOMAIN_ID}" --rmw "${SIM_RMW}" "${WORLD}" \
        > "${CAMPAIGN_DIR}/gazebo.log" 2>&1 &
    local i
    for i in $(seq 1 60); do
        grep -qa "drone plugin finished loading" "${CAMPAIGN_DIR}/gazebo.log" 2>/dev/null && break
        sleep 2
    done
    grep -qa "drone plugin finished loading" "${CAMPAIGN_DIR}/gazebo.log" \
        || { say "the world did not come up; see ${CAMPAIGN_DIR}/gazebo.log"; return 1; }
    # The log line means the plugin is loaded, not that a host subscriber can
    # hear it: DDS discovery to a process that did not exist when the container
    # started takes seconds more. Wait for an actual message rather than
    # guessing, or the ferry opens on an empty graph and the run is written off
    # as "could not reach the area".
    local i
    for i in $(seq 1 30); do
        timeout 4 ros2 topic echo --once /simple_drone/odom >/dev/null 2>&1 && break
        sleep 2
    done
    sleep 2
}

# REPEAT: how many passes over the area list. 0 means "keep going" -- one
# recording ends and the next begins, which is what an unattended overnight
# session wants. Ctrl-C, or kill the script, to stop between runs.
REPEAT="${REPEAT:-1}"
say "passes      ${REPEAT} ($([[ "${REPEAT}" == "0" ]] && echo "until stopped" || echo "then stop"))"

n=0
pass=0
while :; do
  pass=$((pass + 1))
  [[ "${REPEAT}" != "0" && "${pass}" -gt "${REPEAT}" ]] && break
  [[ "${REPEAT}" == "0" || "${REPEAT}" -gt 1 ]] && say "=== pass ${pass} ==="
  for area in "${AREAS[@]}"; do
    n=$((n + 1))
    run_dir="${CAMPAIGN_DIR}/$(printf '%02d' "${n}")_${area}"
    mkdir -p "${run_dir}"
    say "--- run ${n} (pass ${pass}, area ${area}) -> ${run_dir}"

    restart_world || { say "run ${n} (${area}): INFRA -- world failed to start"; continue; }

    say "ferrying to ${area}"
    if ! python3 "${SCRIPT_DIR}/goto_area.py" "${area}" > "${run_dir}/ferry.log" 2>&1; then
        say "run ${n} (${area}): INFRA -- could not reach the area, see ferry.log"
        continue
    fi
    tail -n1 "${run_dir}/ferry.log" | sed 's/^/[campaign]   /' | tee -a "${LOG}"

    START_SIM=0 RECORD=1 RECORD_SECONDS="${SECONDS_PER_RUN}" \
    RECORD_OUTPUT="${run_dir}/run.mp4" BAG_DIR="${run_dir}/bag" \
    SJTU_N1_LOG_DIR="${run_dir}" \
        bash "${SCRIPT_DIR}/run_sjtu_n1.sh" "${WORLD}" "${INSTRUCTION}" \
        > "${run_dir}/run.log" 2>&1
    run_status=$?
    if [[ "${run_status}" != "0" ]]; then
        # A run that never launched a node and a run that flew and found nothing
        # both produce zero commitments. Say which happened.
        say "run ${n} (${area}): FAILED (exit ${run_status}) -- $(tail -n1 "${run_dir}/run.log")"
        continue
    fi

    # A verdict, not a shrug. The three things that make a recording a result:
    # a playable video, at least one committed route, and an aircraft that was
    # still the right way up at the end.
    # `grep -c` prints its count AND exits 1 when the count is zero, so the
    # obvious `|| echo 0` appends a second line and every later arithmetic test
    # dies on "0\n0". Count without grep's exit status instead.
    count_in() { grep -ac "$1" "${run_dir}/nodes.log" 2>/dev/null | head -n1; }
    commits="$(count_in 'committed #')"; commits="${commits:-0}"
    capsized="$(count_in 'CAPSIZED')"; capsized="${capsized:-0}"
    frames="$(grep -a 'wrote .* frames' "${run_dir}/nodes.log" 2>/dev/null | tail -n1)"
    verdict="OK"
    [[ -s "${run_dir}/run.mp4" ]] || verdict="NO VIDEO"
    [[ "${commits}" -gt 0 ]] || verdict="NO ROUTE"
    [[ "${capsized}" -eq 0 ]] || verdict="CAPSIZED"
    grep -qa "CAPSIZED" "${run_dir}/run.log" 2>/dev/null && verdict="CAPSIZED"
    # The shape of what was flown, not just the count. Five runs of the same
    # prompt are only comparable if the line says whether they were curves.
    curves="$(count_in '\[curve\]')"; curves="${curves:-0}"
    actions="$(count_in '\[action\]')"; actions="${actions:-0}"
    turns="$(count_in 'turn #')"; turns="${turns:-0}"
    escapes="$(count_in 'BLOCKED ESCAPE')"; escapes="${escapes:-0}"
    stops="$(count_in 'N1 STOP')"; stops="${stops:-0}"
    metres="$(grep -ao 'committed #[0-9]*: [0-9]* pts, [0-9.]* m' "${run_dir}/nodes.log" 2>/dev/null \
        | awk '{s += $(NF-1)} END {printf "%.1f", s+0}')"
    say "run ${n} (${area}): ${verdict}  routes=${commits} (${curves} curve/${actions} action, ${metres:-0} m)  turns=${turns}  escapes=${escapes}  stops=${stops}  ${frames:-no frame count}"
  done
done

say "=== campaign done ==="
for d in "${CAMPAIGN_DIR}"/*/; do
    [[ -f "${d}/run.mp4" ]] && say "$(basename "${d}")  $(du -h "${d}/run.mp4" | cut -f1)  ${d}run.mp4"
done
