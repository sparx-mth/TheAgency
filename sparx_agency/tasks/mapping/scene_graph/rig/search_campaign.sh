#!/usr/bin/env bash
# Time-to-find, measured. One trial per (target, start), each a full flight.
#
# The method's whole claim is that commonsense priors find an object FASTER
# than uniform exploration, and nothing in this repo measured that. FALCON's
# own campaign_run.sh scores COVERAGE -- the exact behaviour this search exists
# to replace -- so this is its sibling, scoring the only number that matters
# here: seconds of sim time from takeoff to a confirmed detection.
#
# Every trial is a cold stack. The alternative -- reusing a world between
# trials -- carries the previous trial's voxel map, room ids and LLM labels
# into the next one, so trial N+1 starts with a map trial N paid for. That is
# not a shorter search, it is a different experiment.
#
# CENSORING IS REPORTED, NOT DROPPED. A trial that hits the time cap without
# finding the target is a real outcome and is recorded as outcome=timeout with
# the cap as a lower bound on its time-to-find. Averaging only the successes
# is how a search that fails half the time reports a good mean.
#
# Usage:
#   search_campaign.sh [--targets "a,b"] [--starts "x,y,z,yaw;..."]
#                      [--cap-s 600] [--repeats 1] [--out DIR]
#                      [--backend falcon|host_sweep] [--world hospital]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/../scripts/run_scene_graph.sh"
STOP_SCRIPT="${SCRIPT_DIR}/../scripts/stop_scene_graph.sh"

TARGETS="wheelchair,x-ray machine,hospital bed"
# Four corners of the hospital's explorable box, all inside it and clear of
# furniture. The default spawn (1,1) is the first: a campaign that only ever
# starts there measures the route out of that corner more than it measures the
# search.
STARTS="1.0,1.0,2.0,0.0;-6.0,-12.0,2.0,1.57;5.0,10.0,2.0,3.14;-8.0,8.0,2.0,-1.57"
CAP_S=600
REPEATS=1
BACKEND="falcon"
WORLD="hospital"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targets) TARGETS="${2:?}"; shift 2 ;;
        --starts)  STARTS="${2:?}";  shift 2 ;;
        --cap-s)   CAP_S="${2:?}";   shift 2 ;;
        --repeats) REPEATS="${2:?}"; shift 2 ;;
        --backend) BACKEND="${2:?}"; shift 2 ;;
        --world)   WORLD="${2:?}";   shift 2 ;;
        --out)     OUT_DIR="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown flag '$1'" >&2; exit 2 ;;
    esac
done

OUT_DIR="${OUT_DIR:-/tmp/search_campaign/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"
TRIALS="${OUT_DIR}/trials.jsonl"
: > "${TRIALS}"

say() { echo "[campaign] $*"; }

# --flag=value throughout: a start pose in a real building has NEGATIVE
# coordinates, and argparse reads a value beginning with "-" as a flag. And
# never fatal: one unrecordable trial must not end a campaign that has hours
# of flights left in it.
record_trial() {
    local tag="$1" target="$2" start="$3" repeat="$4" outcome="$5" dir="$6" wall="$7"
    python3 "${SCRIPT_DIR}/record_trial.py" \
        --out="${TRIALS}" --tag="${tag}" --target="${target}" \
        --start="${start}" --repeat="${repeat}" --backend="${BACKEND}" \
        --outcome="${outcome}" --cap-s="${CAP_S}" \
        --trial-dir="${dir}" --wall-s="${wall}" \
        || say "   WARNING: could not record ${tag}"
}

IFS=',' read -r -a TARGET_LIST <<< "${TARGETS}"
IFS=';' read -r -a START_LIST  <<< "${STARTS}"

say "out         ${OUT_DIR}"
say "targets     ${#TARGET_LIST[@]}  (${TARGETS})"
say "starts      ${#START_LIST[@]}"
say "repeats     ${REPEATS}   cap ${CAP_S}s   backend ${BACKEND}   world ${WORLD}"
say "trials      $(( ${#TARGET_LIST[@]} * ${#START_LIST[@]} * REPEATS ))"

trial_index=0
for repeat in $(seq 1 "${REPEATS}"); do
for target in "${TARGET_LIST[@]}"; do
for start in "${START_LIST[@]}"; do
    trial_index=$(( trial_index + 1 ))
    tag="t$(printf '%03d' "${trial_index}")"
    trial_out="${OUT_DIR}/${tag}"
    mkdir -p "${trial_out}"
    say "── ${tag}  target='${target}'  start=(${start})  repeat=${repeat}"

    # Cold every time: the previous trial's map is the previous trial's work.
    bash "${STOP_SCRIPT}" --out "${trial_out}" --all >/dev/null 2>&1 || true
    docker rm -f falcon-sjtu sjtu-ros1-bridge falcon-rviz sjtu_drone_"${WORLD}" >/dev/null 2>&1 || true
    sleep 3

    started_at="$(date -u +%s)"
    set +e
    SJTU_DRONE_SPAWN="${start}" \
    SEARCH_BACKEND="${BACKEND}" \
    KILL_STALE=0 \
    timeout 900 bash "${RUN_SCRIPT}" \
        --world "${WORLD}" --target "${target}" --object-search --fly \
        --out "${trial_out}" > "${trial_out}/bringup.log" 2>&1
    bringup_rc=$?
    set -e

    if [[ ${bringup_rc} -ne 0 ]]; then
        say "   bring-up FAILED (rc=${bringup_rc}); recording and moving on"
        record_trial "${tag}" "${target}" "${start}" "${repeat}" \
            bringup_failed "${trial_out}" "$(( $(date -u +%s) - started_at ))"
        continue
    fi

    # Watch for the latch. /target_seen/info carries the sim stamp at which the
    # watcher latched, which is the number we want -- wall time is meaningless
    # here because this world runs well below real time and the ratio varies
    # with what else is on the GPU.
    outcome="timeout"
    set +e
    timeout "${CAP_S}" bash -c '
      out="$1"
      while true; do
        if grep -q "TARGET FOUND" "$out"/target_watcher_node.log 2>/dev/null; then exit 0; fi
        if grep -q "MISSION ABORT" "$out"/*.log 2>/dev/null; then exit 3; fi
        sleep 2
      done' _ "${trial_out}"
    watch_rc=$?
    set -e
    case "${watch_rc}" in
        0) outcome="found" ;;
        3) outcome="aborted" ;;
        *) outcome="timeout" ;;
    esac
    say "   outcome=${outcome}"

    record_trial "${tag}" "${target}" "${start}" "${repeat}" \
        "${outcome}" "${trial_out}" "$(( $(date -u +%s) - started_at ))"

    bash "${STOP_SCRIPT}" --out "${trial_out}" --all >/dev/null 2>&1 || true
    sleep 2
done
done
done

say "══ campaign complete ══"
python3 "${SCRIPT_DIR}/analyze_campaign.py" "${TRIALS}" | tee "${OUT_DIR}/summary.txt"
say "trials  ${TRIALS}"
say "summary ${OUT_DIR}/summary.txt"
