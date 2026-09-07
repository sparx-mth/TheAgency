#!/usr/bin/env bash
# ============================================================
# record_medium_horizon.sh — fly the medium-horizon ladder, network alone.
#
#   ./record_medium_horizon.sh                 # the whole ladder, 3 passes
#   ./record_medium_horizon.sh t1 t3           # just these rungs
#   PASSES=1 ./record_medium_horizon.sh        # one pass each, for a smoke test
#   LATCH=5 ./record_medium_horizon.sh t3      # re-fly a rung WITH the STOP restart
#
# THE QUESTION THIS ANSWERS. A concrete short order works: "there is a room to
# your right, enter it, go to the center, find the table and stop near the
# table" put the aircraft inside a room and stopped it there five times out of
# five -- though scoring those tapes properly shows it was the SAME room only
# twice; the other three passes crossed the atrium into a different room
# entirely. The order does not name a room, so that is a pass, but "5/5" was
# always a looser claim than it sounded. The abstract long one does not: "explore
# the entire hospital" never cleared 28% of the floor. Everything in between is
# untested, and "the supervisor is not good enough yet" and "the policy cannot
# hold a two-step goal" predict the same failure, so neither is refuted by
# another exploration run.
#
# So: five rungs, each one step further along ONE axis -- how far ahead of the
# aircraft the last sub-goal sits -- with the supervisor OFF and the policy node
# in exactly the configuration that flew the known-good order. Rung t0 IS that
# order, re-flown, so the ladder is anchored to a measured point rather than to
# a memory of one.
#
# THE BASELINE IS FLOWN WITH THE STOP RESTART DISABLED, on purpose. It is new
# and unflown, and leaving it on would mean a failure could be the horizon or
# could be the restart, which is the ambiguity this whole exercise exists to
# remove. The knob is not edited in the tracked YAML: a derived copy is written
# into the campaign directory and CONFIG_FILE points at it, so what flew is
# recorded beside the flight. Set LATCH=5 to re-fly a rung with it on -- do that
# per rung, after a failure, to ask whether the restart rescues that rung.
#
# Every referent every instruction names has been checked against the map with
# check_referents.py, which this script runs first and refuses to fly without.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
# TWO INTERPRETERS, AND THE SPLIT IS NOT OPTIONAL. The offline tools are numpy
# work and run in the venv; the ferry is a ROS 2 node and cannot, because the
# venv shadows the system numpy and OpenCV that cv_bridge was built against.
# Getting this the wrong way round is a ModuleNotFoundError at best.
PY="${REPO_ROOT}/.venv/bin/python"
ROS_PY="${ROS_PY:-python3}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${ROS_DISTRO:-}" ]]; then
    # `set -u` OFF across the source, and back on straight after. ROS's own
    # setup.bash reads unbound variables, so under -u it aborts the whole
    # script at the first one -- silently, with an empty log and no campaign
    # directory, which looks exactly like the script was never run.
    set +u
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO_HINT:-jazzy}/setup.bash" 2>/dev/null
    set -u
    [[ -n "${ROS_DISTRO:-}" ]] || die "could not source ROS 2; the ferry needs rclpy"
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
# bringup_world.sh is invoked directly here, so it inherits nothing from
# run_sjtu_n1.sh. Both of these have to be set or the world does not come up:
# setup/env.sh hard-fails without SJTU_PROJECT_DIR, and Gazebo Classic silently
# disables its cameras with no X, which produces a running simulator that sends
# no images at all.
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
export DISPLAY="${DISPLAY:-:1}"

# THE FERRY RUNS BEFORE run_sjtu_n1.sh AND INHERITS NONE OF ITS ENVIRONMENT.
# Shared memory does not cross the sim container's boundary: discovery succeeds
# over multicast so the topic list looks full, and then every sample is dropped.
# Without the no-shm profile `goto_area.py` sees no odometry, times out after
# 20 s, and every rung reports "could not reach the area" with nothing pointing
# at why. Identical to the block in record_campaign.sh, which says the same
# thing at more length; both should end up in setup/env.sh beside ROS_DOMAIN_ID.
SETUP_DIR="${REPO_ROOT}/sparx_agency/robots/SJTU/setup"
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
    *) die "RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION}' is not one this stack configures." ;;
esac

say() { echo "[ladder] $*"; }
die() { echo "[ladder] ERROR: $*" >&2; exit 1; }

PASSES="${PASSES:-3}"
LATCH="${LATCH:-0}"
OUT_ROOT="${OUT_ROOT:-${HOME}/sjtu_n1_recordings}"
N1_HOST="${N1_HOST:-127.0.0.1}"
N1_PORT="${N1_PORT:-8087}"

# name | start x | y | yaw | seconds | instruction | scoring argument
#
# The clocks are sized off measured throughput, not off route length: this
# aircraft covers about 2.7 m of ground per minute once turns and thinking are
# counted, so a 9 m route is a six-minute job and anything tighter scores a
# null that means "ran out of clock", not "could not do it".
TASKS=(
# name | start x | y | yaw | seconds | instruction | scoring
#
# NO DIRECTION WORD IN ANY OF THESE, and that is the whole design. Measured
# over the first ladder: every order containing "turn right" or "on your right"
# made the aircraft turn right on the spot, immediately, whatever the sentence
# said about doing it later -- 7 runs out of 7, about fifty seconds of spinning
# each, after which it was facing the wrong way and flew off confidently in it.
# The single order with no direction word was the only one that worked, 3 for 3.
# So these name destinations and things to look for, and nothing else.
#
# The clocks come from measured throughput -- about 2.7 m of ground per minute
# once turning and thinking are counted -- so a null means "could not do it",
# not "ran out of clock".
"m1_which_room|-3.90|-24.98|180|360|There are two rooms in front of you. You will find a refrigerator in one of the rooms. Go into the room that has the refrigerator and stop inside it.|--enter|the south-west room (2)"
"m2_find_and_look|-3.90|-24.98|180|420|There are two rooms in front of you. One of them has a refrigerator in it. Find the room with the refrigerator, go inside it, and look around the whole room.|--enter|the south-west room (2)"
"m3_both_rooms|-3.90|-24.98|180|660|There are two rooms in front of you. Go into the first room and look around it, then leave that room and go into the second room and look around it too.|--enter|the south-west room (2)"
"m4_down_the_hall|-5.00|-16.50|-90|480|Fly along the corridor in front of you and go through the wide doorway at the far end of it, then stop inside the room.|--enter|the south-west room (1)"
)

WANTED=("$@")
matches() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    local name="$1" w
    for w in "${WANTED[@]}"; do [[ "${name}" == "${w}"* ]] && return 0; done
    return 1
}

# ── preflight ─────────────────────────────────────────────────────────────
# Three checks, and each one has cost a run here before: an instruction naming
# something off-camera, a cold server answering nothing, and a second process
# on the card.
say "checking that every instruction names something the camera can see..."
"${PY}" "${SCRIPT_DIR}/check_referents.py" >/dev/null \
    || die "a referent failed its visibility check; run check_referents.py"

curl -sf -m 5 "http://${N1_HOST}:${N1_PORT}/openapi.json" -o /dev/null \
    || die "no InternVLA-N1 server at ${N1_HOST}:${N1_PORT}. Start it, then pre-warm with prewarm_server.py."

"${PY}" "${SCRIPT_DIR}/check_gpu_free.py" \
    || die "the card is not clear for the model server"

CAMPAIGN="${OUT_ROOT}/ladder_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CAMPAIGN}"
LOG="${CAMPAIGN}/ladder.log"
exec > >(tee -a "${LOG}") 2>&1

# The derived config, so the knob that flew is recorded beside the flight.
BASE_CFG="${REPO_ROOT}/sparx_agency/robots/SJTU/config/vla/internvla_n1.yaml"
CFG="${CAMPAIGN}/internvla_n1.flown.yaml"
"${PY}" - "${BASE_CFG}" "${CFG}" "${LATCH}" <<'PYEOF'
import re, sys
src, dst, latch = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(src).read()
new, n = re.subn(r"^(\s*)stop_restart_after:\s*\d+\s*$",
                 r"\g<1>stop_restart_after: %d" % latch, text, count=1, flags=re.M)
if n != 1:
    # REFUSE rather than fly an unknown configuration. An absent key means the
    # node falls back to its own default, which is not necessarily the value
    # this campaign says it flew -- and a campaign that misreports its own
    # configuration is worse than one that did not run.
    raise SystemExit("could not find stop_restart_after in %s; refusing to guess" % src)
open(dst, "w").write(new)
print("[ladder] flying with stop_restart_after: %d" % latch)
PYEOF
[[ -s "${CFG}" ]] || die "could not write the derived config"

# A FRESH, LEVEL AIRCRAFT FOR EVERY PASS. Not hygiene -- a correctness
# requirement, and this ladder learnt it the hard way. Left flying between
# rungs the aircraft accumulates attitude and altitude error, and the third
# pass of the day refused to ferry at all: "never reached the ferry altitude
# (4.5 m); refusing to cross the building below the walls". A rung scored
# against an aircraft that could not climb measures the aircraft, not the
# horizon. record_campaign.sh does the same thing for the same reason.
restart_world() {
    say "    restarting the world (a fresh, level aircraft)"
    docker rm -f "sjtu_drone_hospital" >/dev/null 2>&1 || true
    sleep 3
    nohup bash "${REPO_ROOT}/sparx_agency/robots/SJTU/setup/bringup_world.sh" \
        --skip-build --domain "${ROS_DOMAIN_ID}" --rmw "${SIM_RMW}" hospital \
        > "${CAMPAIGN}/gazebo.log" 2>&1 &
    local i
    for i in $(seq 1 60); do
        grep -qa "drone plugin finished loading" "${CAMPAIGN}/gazebo.log" 2>/dev/null && break
        sleep 2
    done
    grep -qa "drone plugin finished loading" "${CAMPAIGN}/gazebo.log" \
        || { say "    the world did not come up; see ${CAMPAIGN}/gazebo.log"; return 1; }
    # The plugin being loaded is not the same as a host subscriber being able
    # to hear it: discovery to a process that did not exist when the container
    # started takes seconds more, and without this wait the ferry opens on an
    # empty graph and the pass is written off as "could not reach the area".
    for i in $(seq 1 30); do
        timeout 4 ros2 topic echo --once /simple_drone/odom >/dev/null 2>&1 && break
        sleep 2
    done
    sleep 2
}

say "campaign  ${CAMPAIGN}"
say "passes    ${PASSES}"
say "supervisor OFF; STOP restart $( [[ "${LATCH}" == "0" ]] && echo disabled || echo "at ${LATCH}")"

for row in "${TASKS[@]}"; do
    IFS='|' read -r name sx sy syaw secs instruction score_flag score_val <<< "${row}"
    matches "${name}" || continue
    for pass in $(seq 1 "${PASSES}"); do
        run_dir="${CAMPAIGN}/${name}/pass${pass}"
        mkdir -p "${run_dir}"
        say "--- ${name} pass ${pass}/${PASSES} -> ${run_dir}"
        say "    ${instruction}"

        if ! restart_world; then
            # STOP THE CAMPAIGN, do not skip. Every cause of this is systemic --
            # a missing SJTU_PROJECT_DIR, no display, a docker image that is not
            # there -- so pass two fails for the reason pass one did. The first
            # version of this said "skipping" and burned all fifteen passes in
            # half an hour, writing a log that had to be read to the bottom to
            # discover that nothing had flown.
            die "the world will not come up; see ${CAMPAIGN}/gazebo.log"
        fi
        if ! "${ROS_PY}" "${SCRIPT_DIR}/goto_area.py" --x "${sx}" --y "${sy}" \
                --yaw-deg "${syaw}" > "${run_dir}/ferry.log" 2>&1; then
            say "    ferry to (${sx}, ${sy}) FAILED; see ferry.log -- skipping this pass"
            continue
        fi

        SJTU_N1_LOG_DIR="${run_dir}" CONFIG_FILE="${CFG}" \
        SUPERVISE=0 START_SIM=0 RECORD=1 RECORD_SECONDS="${secs}" \
        RECORD_OUTPUT="${run_dir}/run.mp4" \
            "${SCRIPT_DIR}/run_sjtu_n1.sh" hospital "${instruction}" \
            > "${run_dir}/run.log" 2>&1

        # A run that committed no route flew nothing, whatever else it wrote.
        commits=$(grep -c 'committed #' "${run_dir}/nodes.log" 2>/dev/null | head -n1)
        say "    ${commits:-0} route(s) committed"
    done
    if [[ -d "${CAMPAIGN}/${name}" ]]; then
        say "    scoring ${name}:"
        # t0 names no particular room, so it is scored on the region trace
        # alone. Passing it a --enter would mark three of the five original
        # passes as failures for entering a DIFFERENT room, which that order
        # never forbade -- and mis-scoring the control is the one error that
        # would invalidate every rung above it.
        if [[ -n "${score_flag}" ]]; then
            "${PY}" "${SCRIPT_DIR}/campaign_report.py" "${CAMPAIGN}/${name}" \
                --regions "${score_flag}" "${score_val}" 2>&1 | sed 's/^/      /'
        else
            "${PY}" "${SCRIPT_DIR}/campaign_report.py" "${CAMPAIGN}/${name}" \
                --regions 2>&1 | sed 's/^/      /'
        fi
    fi
done

flown=$(grep -rl 'committed #' "${CAMPAIGN}"/*/*/nodes.log 2>/dev/null | wc -l)
say "done. ${CAMPAIGN}"
if [[ "${flown}" -eq 0 ]]; then
    say "NOTHING FLEW. No run committed a single route -- treat every table"
    say "above as absent data, not as a result."
else
    say "${flown} run(s) committed at least one route."
fi
say "the ladder answers one question: which rung it falls off. A rung that"
say "fails by STOPping is a latch, and worth re-flying with LATCH=5."
