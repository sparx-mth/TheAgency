#!/usr/bin/env bash
# ============================================================
# soak.sh — fly ONE run over and over, and say how long the streak lasted.
#
#   ./soak.sh [run] [attempts] [-- extra run_exploration.py args]
#
# `run_campaign.sh` flies the six configurations once each: that is the
# deliverable. This asks the other question -- is the stack *reliable* -- by
# flying the same configuration repeatedly and counting consecutive clean
# flights.
#
# It exists because almost every failure on this stack has been intermittent.
# FALCON's vendored LKH solver segfaulted on some coverage-tour instances and
# not others; its hierarchical grid crashed only once the aircraft left the
# exploration box; the aircraft wedges in a doorway on one seed and flies past
# it on the next. A single green run proves very little, and the only way to
# tell a fix from a lucky seed is a streak.
#
# A flight counts as CLEAN when all three hold:
#   * the outcome is one of the good ones (`explored`, `flight_timeout`,
#     `planner_stopped` -- see the README on why a timeout is a success),
#   * `exploration_node` did not crash, which the aircraft CANNOT see:
#     traj_server outlives the planner and keeps republishing, so this is
#     checked in FALCON's own log rather than in result.json,
#   * coverage reached MIN_COVERAGE_M3, so a flight that ended tidily after
#     mapping one room is not counted as success.
#
# Stops at the first dirty flight, on purpose: the next thing to do is read that
# run's logs, and burning another hour of GPU first helps nobody. Everything is
# kept per attempt under the output directory.
#
# Environment:
#   MIN_COVERAGE_M3   coverage a flight must reach to count      (default 2200)
#   RUN_TIMEOUT_S     wall-clock ceiling on one flight           (default 3600)
#   FALCON_PEGASUS_OUT  where recordings go   (default ~/data/sim/falcon_pegasus)
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"
OUT_ROOT="${FALCON_PEGASUS_OUT:-$HOME/data/sim/falcon_pegasus}/soak"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3600}"
# The exploration box is 28.1 x 71.9 x 1.2 m = 2424 m3. Full coverage is never
# literally 100%: some of that volume is furniture and wall. 2200 is ~91%, which
# is "the office is mapped" without demanding FALCON chase the last frontier
# behind a desk leg.
MIN_COVERAGE_M3="${MIN_COVERAGE_M3:-2200}"

RUN_NAME="${1:-6_whole_office}"
ATTEMPTS="${2:-10}"
shift $(( $# < 2 ? $# : 2 ))
[ "${1:-}" = "--" ] && shift
EXTRA=("$@")

if [[ ! -f "${SCRIPT_DIR}/runs/${RUN_NAME}.yaml" ]]; then
    echo "[ERROR] no such run: ${RUN_NAME}" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
LEDGER="${OUT_ROOT}/ledger.jsonl"
echo "soak: ${RUN_NAME} x ${ATTEMPTS}, need coverage >= ${MIN_COVERAGE_M3} m3"
echo "      output ${OUT_ROOT}, ledger ${LEDGER}"
echo

streak=0
for attempt in $(seq 1 "${ATTEMPTS}"); do
    stamp="$(date +%Y%m%d_%H%M%S)"
    dir="${OUT_ROOT}/${attempt}_${stamp}"
    mkdir -p "${dir}"
    echo "============================================================"
    echo "ATTEMPT ${attempt}/${ATTEMPTS}  (streak ${streak})  -> ${dir}"
    echo "============================================================"

    # Leftovers from a previous attempt hold the GPU, the MAVLink ports or the
    # localhost sockets, and the failure that causes looks like a new bug.
    docker exec "${CONTAINER}" pkill -9 -f run_exploration.py > /dev/null 2>&1 || true
    docker rm -f falcon-pegasus > /dev/null 2>&1 || true
    sleep 3

    FALCON_LOG_DIR="${dir}" "${SCRIPT_DIR}/run_falcon_pegasus.sh" "${RUN_NAME}" \
        > "${dir}/falcon.log" 2>&1 &
    falcon_pid=$!

    echo -n "  waiting for the bridge"
    bound=0
    for _ in $(seq 1 150); do
        if ss -ltn 2>/dev/null | grep -q '127.0.0.1:5599' \
           && ss -ltn 2>/dev/null | grep -q '127.0.0.1:5600'; then
            bound=1; echo " -- up"; break
        fi
        echo -n "."; sleep 2
    done
    if [ "${bound}" -eq 0 ]; then
        echo " -- NEVER BOUND"
        docker rm -f falcon-pegasus > /dev/null 2>&1 || true
        echo "{\"attempt\":${attempt},\"clean\":false,\"why\":\"bridge never bound\"}" >> "${LEDGER}"
        break
    fi

    timeout "${RUN_TIMEOUT_S}" "${SCRIPT_DIR}/run_isaac_side.sh" "${RUN_NAME}" --video \
        "${EXTRA[@]+"${EXTRA[@]}"}" > "${dir}/isaac.log" 2>&1
    status=$?

    # Stopping the container is what finalises the map video: OpenCV writes an
    # MP4's index on release, and the recorder releases on ROS shutdown. Killed,
    # it leaves a file with no moov atom that nothing will play.
    docker stop -t 30 falcon-pegasus > /dev/null 2>&1 || true
    wait "${falcon_pid}" 2>/dev/null || true
    docker cp "${CONTAINER}:${DEV_ROOT}/falcon_pegasus/${RUN_NAME}/." "${dir}/" \
        > /dev/null 2>&1 || true

    outcome="$(sed -n 's/.*"outcome": "\([a-z_]*\)".*/\1/p' "${dir}/result.json" 2>/dev/null | head -1)"
    coverage="$(grep -ao 'Coverage: [0-9.]*' "${dir}/falcon.log" 2>/dev/null | tail -1 | awk '{print $2}')"
    # No `|| echo 0` here: `grep -c` PRINTS 0 and then exits 1 when it matches
    # nothing, so the fallback appends a second line and the ledger entry
    # becomes invalid JSON -- which it silently did for two rounds.
    crashes="$(grep -ac 'Stack trace' "${dir}/falcon.log" 2>/dev/null)"
    lkh="$(grep -ac 'the LKH solver died on' "${dir}/falcon.log" 2>/dev/null)"
    crashes="${crashes:-0}"
    lkh="${lkh:-0}"
    coverage="${coverage:-0}"
    outcome="${outcome:-unknown}"

    why=""
    case "${outcome}" in
        explored|flight_timeout|planner_stopped) ;;
        *) why="outcome ${outcome}" ;;
    esac
    [ "${crashes}" -gt 0 ] && why="${why:+${why}; }exploration_node crashed ${crashes}x"
    awk "BEGIN{exit !(${coverage} < ${MIN_COVERAGE_M3})}" \
        && why="${why:+${why}; }coverage ${coverage} < ${MIN_COVERAGE_M3} m3"

    if [ -z "${why}" ]; then
        streak=$(( streak + 1 ))
        echo "  CLEAN  outcome=${outcome} coverage=${coverage} m3 (LKH recoveries ${lkh})  streak=${streak}"
    else
        echo "  DIRTY  ${why}   (outcome=${outcome} coverage=${coverage} crashes=${crashes})"
    fi
    printf '{"attempt":%d,"clean":%s,"outcome":"%s","coverage_m3":%s,"crashes":%s,"lkh_recoveries":%s,"exit":%d,"why":"%s","dir":"%s"}\n' \
        "${attempt}" "$([ -z "${why}" ] && echo true || echo false)" "${outcome}" \
        "${coverage}" "${crashes}" "${lkh}" "${status}" "${why}" "${dir}" >> "${LEDGER}"

    if [ -n "${why}" ]; then
        echo
        echo "stopping so this one can be read: ${dir}"
        break
    fi
    echo
done

echo "============================================================"
echo "SOAK DONE — longest clean streak this session: ${streak}/${ATTEMPTS}"
echo "ledger: ${LEDGER}"
