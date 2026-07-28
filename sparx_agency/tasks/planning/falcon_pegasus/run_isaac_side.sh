#!/usr/bin/env bash
# ============================================================
# run_isaac_side.sh — fly one exploration run on Isaac Sim.
#
#   ./run_isaac_side.sh <run> [extra run_exploration.py args ...]
#
# Run this on the HOST, AFTER run_falcon_pegasus.sh is up: the bridge binds the
# sockets and the aircraft connects to them, because Kit takes minutes to load a
# stage and the ROS stack takes seconds.
#
# It syncs the repo into the isaac-sim container (which holds a COPY, not a bind
# mount -- host edits do not appear there until this runs), clears the PX4 lock
# files a killed run leaves behind, and starts the flight under Isaac Sim's own
# Python.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"
WORKER="${WORKER:-0}"

RUN_NAME="${1:-3_open_plan}"
if [[ $# -ge 1 ]]; then shift; fi

if [[ ! -f "${SCRIPT_DIR}/runs/${RUN_NAME}.yaml" ]]; then
    echo "[ERROR] no such run: ${RUN_NAME}" >&2
    ls -1 "${SCRIPT_DIR}"/runs/*.yaml | xargs -n1 basename | sed 's/\.yaml$//;s/^/          /' >&2
    exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "[ERROR] container '${CONTAINER}' is not running." >&2
    exit 1
fi

# A previous run's Isaac Sim or PX4 still holding the GPU and the ports is the
# most common cause of an unexplained crash on the next one.
if docker exec "${CONTAINER}" pgrep -f "run_exploration.py" > /dev/null 2>&1; then
    echo "[ERROR] an exploration run is already in progress. Stop it with:" >&2
    echo "  docker exec ${CONTAINER} pkill -9 -f run_exploration.py" >&2
    exit 1
fi

echo "[INFO] syncing the repo into ${CONTAINER} ..."
docker cp "${REPO_ROOT}/sparx_agency" "${CONTAINER}:${DEV_ROOT}/repo/" > /dev/null

# pymavlink lives in Isaac Sim's own Python, which is in the container's writable
# layer -- so it is gone after any container recreation, and the failure lands
# minutes into a boot rather than at the start. Check for it here instead.
if ! docker exec "${CONTAINER}" /isaac-sim/python.sh -c "import pymavlink" > /dev/null 2>&1; then
    echo "[INFO] installing pymavlink into Isaac Sim's Python (lost on container recreate)"
    docker exec "${CONTAINER}" /isaac-sim/python.sh -m pip install --quiet pymavlink
fi

docker exec "${CONTAINER}" rm -f "/tmp/px4_lock-${WORKER}" "/tmp/px4-sock-${WORKER}"

OUT_DIR="${DEV_ROOT}/falcon_pegasus/${RUN_NAME}"
echo "[INFO] flying ${RUN_NAME}; output goes to ${CONTAINER}:${OUT_DIR}"

EXTRA_ARGS=""
if [ $# -gt 0 ]; then EXTRA_ARGS="$(printf '%q ' "$@")"; fi

docker exec "${CONTAINER}" bash -c \
    "cd ${DEV_ROOT}/repo && /isaac-sim/python.sh \
     sparx_agency/tasks/planning/falcon_pegasus/isaac/run_exploration.py \
     --run ${RUN_NAME} --worker ${WORKER} --out-dir ${OUT_DIR} ${EXTRA_ARGS}"
status=$?

echo "[INFO] run finished (exit ${status}). Copy the output out with:"
echo "  docker cp ${CONTAINER}:${OUT_DIR} ."
exit "${status}"
