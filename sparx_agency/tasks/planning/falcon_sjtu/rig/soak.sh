#!/usr/bin/env bash
# ============================================================
# soak.sh — consecutive-clean-run soak until a streak target.
#
#   WORLD=no_roof_small_warehouse ./soak.sh warehouse 10 [cap_s]
#
# Runs campaign_run.sh back to back. A CLEAN run extends the streak and the
# soak continues on its own; anything else (dirty, fatal, timeout, infra)
# STOPS the soak immediately with the failing run's artifacts on disk --
# the point of the campaign is that a human or the improving agent diagnoses
# every failure, so the loop never buries one under later runs.
#
# State in <base>/streak.txt; run dirs are <base>/NNN_<map>/.
# Exit 0 when the streak target is reached, 1 when a run broke the streak.
# ============================================================
set -uo pipefail

MAP="${1:?map name}"
TARGET="${2:?consecutive clean runs required}"
CAP_S="${3:-2400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SOAK_BASE:-/tmp/falcon_sjtu/soak_${MAP}}"
mkdir -p "${BASE}"

streak=0
n=0
# resume: continue numbering after existing runs, streak restarts at 0 --
# a soak interrupted by a fix must re-earn the whole streak.
while [[ -d "${BASE}/$(printf '%03d' $((n + 1)))_${MAP}" ]]; do n=$((n + 1)); done

echo "[soak] target ${TARGET} consecutive CLEAN on ${MAP} (cap ${CAP_S}s/run), starting at run $((n + 1))"
while (( streak < TARGET )); do
    n=$((n + 1))
    dir="${BASE}/$(printf '%03d' ${n})_${MAP}"
    echo "[soak] run ${n} (streak ${streak}/${TARGET}) -> ${dir}"
    "${SCRIPT_DIR}/campaign_run.sh" "${MAP}" "${dir}" "${CAP_S}"
    rc=$?
    if [[ ${rc} -eq 0 ]]; then
        streak=$((streak + 1))
        echo "[soak] run ${n} CLEAN; streak ${streak}/${TARGET}"
        echo "${streak}" > "${BASE}/streak.txt"
    else
        echo "[soak] run ${n} broke the streak (exit ${rc}); stopping for diagnosis"
        echo "0" > "${BASE}/streak.txt"
        exit 1
    fi
done
echo "[soak] TARGET REACHED: ${TARGET}/${TARGET} consecutive clean runs on ${MAP}"
exit 0
