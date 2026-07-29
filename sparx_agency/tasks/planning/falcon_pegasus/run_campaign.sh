#!/usr/bin/env bash
# ============================================================
# run_campaign.sh — fly every exploration run, one after another, unattended.
#
#   ./run_campaign.sh [run ...]
#
# With no arguments it flies all six configurations in runs/, which is the
# deliverable: six recordings of FALCON exploring six structurally different
# parts of the same building.
#
# Each run is a pair of processes that have to start in order -- the FALCON
# stack binds the sockets, then the aircraft connects -- and each produces two
# videos: the map being built (FALCON side) and the flight itself (Isaac side).
#
# Runs are sequential, not parallel, and not because of PX4 ports. Kit's
# start-up is the heaviest moment of a worker's life and this is an 8 GB laptop
# GPU; two overlapping boots crash the RTX shader compiler.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"
OUT_ROOT="${FALCON_PEGASUS_OUT:-$HOME/data/sim/falcon_pegasus}"
# How long to allow one run before giving up on it, wall-clock seconds. Kit's
# boot and PX4's warm-up alone are about five minutes before the aircraft is
# even armable.
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-3600}"

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
    mapfile -t RUNS < <(ls -1 "${SCRIPT_DIR}"/runs/*.yaml | xargs -n1 basename | sed 's/\.yaml$//')
fi

mkdir -p "${OUT_ROOT}"
echo "campaign: ${#RUNS[@]} run(s) -> ${OUT_ROOT}"
printf '  %s\n' "${RUNS[@]}"
echo

summary=()
for run in "${RUNS[@]}"; do
    echo "============================================================"
    echo "RUN ${run}"
    echo "============================================================"
    run_dir="${OUT_ROOT}/${run}"
    mkdir -p "${run_dir}"

    # Anything left over from a previous run holds the GPU, the MAVLink ports or
    # the localhost sockets, and the failure it causes looks like a new bug.
    docker exec "${CONTAINER}" pkill -9 -f run_exploration.py > /dev/null 2>&1 || true
    docker rm -f falcon-pegasus > /dev/null 2>&1 || true
    sleep 3

    FALCON_LOG_DIR="${run_dir}" "${SCRIPT_DIR}/run_falcon_pegasus.sh" "${run}" \
        > "${run_dir}/falcon.log" 2>&1 &
    falcon_pid=$!

    # Wait for both sockets, checked with `ss` rather than by connecting: the
    # bridge accepts exactly one downlink connection, and a probe would take it.
    echo -n "waiting for the FALCON bridge"
    for _ in $(seq 1 150); do
        if ss -ltn 2>/dev/null | grep -q '127.0.0.1:5599' \
           && ss -ltn 2>/dev/null | grep -q '127.0.0.1:5600'; then
            echo " -- up"; break
        fi
        echo -n "."; sleep 2
    done

    timeout "${RUN_TIMEOUT_S}" "${SCRIPT_DIR}/run_isaac_side.sh" "${run}" --video \
        > "${run_dir}/isaac.log" 2>&1
    status=$?

    # Stopping the container is what finalises the map video: OpenCV only writes
    # an MP4's index on release, and the recorder releases on ROS shutdown. A
    # killed container leaves a file with no moov atom that nothing will play.
    docker stop -t 30 falcon-pegasus > /dev/null 2>&1 || true
    wait "${falcon_pid}" 2>/dev/null || true

    docker cp "${CONTAINER}:${DEV_ROOT}/falcon_pegasus/${run}/." "${run_dir}/" \
        > /dev/null 2>&1 || echo "WARNING: nothing to copy back for ${run}"

    outcome="$(sed -n 's/.*"outcome": "\([a-z_]*\)".*/\1/p' "${run_dir}/result.json" 2>/dev/null | head -1)"
    summary+=("${run}: exit ${status}, outcome ${outcome:-unknown}")
    echo "-> ${summary[-1]}"
    echo
done

echo "============================================================"
echo "CAMPAIGN DONE"
printf '  %s\n' "${summary[@]}"
echo
echo "Recordings in ${OUT_ROOT}:"
find "${OUT_ROOT}" -name '*.mp4' -printf '  %p (%sB)\n' 2>/dev/null | sort
