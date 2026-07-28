#!/usr/bin/env bash
# Run one stub flight end to end: bring the FALCON side up, fly the stand-in
# aircraft against it, tear it down, and report what happened.
#
#   stub/check.sh [run] [max_flight_s]
#
# This is the fast loop. It exercises the bridge, FALCON's configuration, the
# exploration box, the camera contract and the outer-loop tracker against the
# real surveyed building, without Isaac Sim, a GPU, or PX4's three-minute
# warm-up. Use it after changing a run config or anything in adapter/, and only
# then spend a real flight on it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PKG_DIR}/../../../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"

RUN="${1:-3_open_plan}"
BUDGET="${2:-240}"
LOG_DIR="${STUB_LOG_DIR:-/tmp/falcon_pegasus_stub}"
mkdir -p "${LOG_DIR}"
FALCON_LOG="${LOG_DIR}/${RUN}_falcon.log"

cleanup() { docker stop -t 20 falcon-pegasus >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[stub] starting the FALCON side (log: ${FALCON_LOG})"
FALCON_LOG_DIR="${LOG_DIR}" "${PKG_DIR}/run_falcon_pegasus.sh" "${RUN}" \
    > "${FALCON_LOG}" 2>&1 &

# Wait for both sockets to be LISTENing. Deliberately checked with `ss` rather
# than by connecting: the bridge accepts exactly one downlink connection, so a
# probe that connects would consume it and the aircraft would never get through.
echo -n "[stub] waiting for the bridge to bind 5599/5600"
for _ in $(seq 1 120); do
    if ss -ltn 2>/dev/null | grep -q '127.0.0.1:5599' \
       && ss -ltn 2>/dev/null | grep -q '127.0.0.1:5600'; then
        echo " -- up"; break
    fi
    echo -n "."; sleep 2
done

"${PYTHON}" "${SCRIPT_DIR}/run_stub.py" --run "${RUN}" --max-flight-s "${BUDGET}" \
    --out "${LOG_DIR}/${RUN}_stub.json"
status=$?

echo
echo "[stub] FALCON's side, last words:"
grep -E "Transit state|Plan fail|No path|Finish exploration|\[bridge\]|\[recorder\]" \
    "${FALCON_LOG}" | tail -20 || true
exit "${status}"
