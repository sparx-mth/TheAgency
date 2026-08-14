#!/usr/bin/env bash
# ============================================================
# both_worlds.sh — fly the hospital AND the warehouse on ONE configuration.
#
#   ./both_worlds.sh [rounds] [cap_s]        # default 1 round, 3600 s per run
#
# The pair is the point. This package spent a long time with a per-world tuning
# (safe_distance 0.85 in the warehouse against 0.45 in the hospital) because
# every improvement to one world was measured only against that world, and a
# regression in the other was found weeks later by somebody flying it. A change
# that fixes the hospital by breaking the warehouse has to FAIL, and it can only
# fail here if the two are flown by the same binary with the same parameters in
# the same session, back to back, with no editing in between.
#
# So this script takes no per-world arguments and passes none. The only thing
# that differs between the two runs is the map config naming the building's own
# geometry -- its walls, its flight box, its spawn -- which is a description of
# the world and not a tuning of the stack.
#
# The world file and the map config are named differently for the warehouse
# (`no_roof_small_warehouse` is the Gazebo world, `warehouse` is the FALCON map
# config), which campaign_run.sh handles via WORLD=.
#
# THE BAR IS THAT THE MISSION FINISHES. FALCON declaring its frontiers
# exhausted is the only signal that a world has actually been mapped, and it is
# what every other verdict here is a failure to reach.
#
# Contacts are counted and printed but do not on their own fail a run, and that
# is a deliberate judgement rather than a loosened bar. Two reasons. First,
# Gazebo's bumper emits a fresh entry per contact point per physics step, so one
# five-second graze along a crate reads as eight "contacts" - count objects, not
# reports. Second, the residual contacts in these worlds are all first
# approaches to geometry that is not yet in the map: the depth camera cannot see
# its last 0.95 m, and nothing in this stack yet slows the aircraft for flying
# into cells it has never observed (see "Still genuinely open" in the README).
# Failing the campaign on that would fail it on a known, characterised sensor
# limit rather than on anything the run did wrong. A DIRTY finish is reported
# loudly, in the summary and in the exit line, so it cannot be mistaken for a
# clean one.
#
# Exit 0 when every run in every round FINISHED. Anything that did not finish
# stops the campaign immediately with its artifacts on disk, because the point
# is that a human or an improving agent diagnoses each failure rather than
# burying it under later runs.
# ============================================================
set -uo pipefail

ROUNDS="${1:-1}"
CAP_S="${2:-3600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BOTH_WORLDS_BASE:-/tmp/falcon_sjtu/both_worlds}"
mkdir -p "${BASE}"

# map:world:min_coverage_m3
#
# The map config is ours; the world is the Gazebo file's own name; the coverage
# floor is what that world AFFORDS, and it is here rather than in campaign_run.sh
# because only the caller knows the world.
#
# Without it a premature finish passes as a success, and premature finishes are
# real: FALCON stops the moment its frontier finder comes up empty, which can
# happen with two thirds of a building unvisited. Measured, a hospital run that
# "FINISHED" at 260.8 m3 having never gone south of y = -2.3.
#
#   warehouse 170: explorable is 192.8 m3 of a 204.1 m3 box, measured off the
#             collision meshes, and complete runs land at 201.5 (the figure
#             exceeds explorable because coverage counts observed obstacle
#             SURFACES as well as free space).
#   hospital  600: the box is 24.0 x 55.0 x 0.7 = 924 m3 and the best measured
#             complete run reached 760 m3 while touching every corner of the
#             building. 600 is comfortably above any partial finish seen (the
#             worst was 260.8, the next 304.9) and below a real one.
WORLDS=("hospital:hospital:600" "warehouse:no_roof_small_warehouse:170")

# Keep the in-container watchdog's cap just under the harness cap so a run that
# runs out of time gets the watchdog's specific verdict (with its coverage and
# confinement numbers) rather than a bare TIMEOUT from the outside.
WATCHDOG_CAP=$(awk -v c="${CAP_S}" 'BEGIN{printf "%.1f", (c > 300) ? c - 300 : c}')
export FALCON_LAUNCH_ARGS="${FALCON_LAUNCH_ARGS:-} watchdog_time_cap_s:=${WATCHDOG_CAP}"

echo "[both] ${ROUNDS} round(s), ${CAP_S}s cap, watchdog cap ${WATCHDOG_CAP}s -> ${BASE}"
failed=0
dirty=0
failed_legs=0
KEEP_GOING="${KEEP_GOING:-0}"
for round in $(seq 1 "${ROUNDS}"); do
    for entry in "${WORLDS[@]}"; do
        IFS=':' read -r map world floor <<< "${entry}"
        dir="${BASE}/r$(printf '%02d' "${round}")_${map}"
        echo "[both] round ${round}: ${map} (world ${world}, needs >= ${floor} m3) -> ${dir}"
        WORLD="${world}" FINISH_MIN_COVERAGE_M3="${floor}" \
            "${SCRIPT_DIR}/campaign_run.sh" "${map}" "${dir}" "${CAP_S}"
        rc=$?
        field() { sed -n "s/.*\"$1\": *\"\\?\\([A-Za-z_0-9.]*\\).*/\\1/p" "${dir}/verdict.json" 2>/dev/null | head -1; }
        verdict=$(field verdict)
        coverage=$(field coverage_m3)
        contacts=$(field contacts)
        objects=$(grep -oE 'began: [A-Za-z_0-9]+' "${dir}/monitor.log" 2>/dev/null | sort -u | wc -l)
        echo "[both] round ${round} ${map}: ${verdict:-?} (exit ${rc}), coverage ${coverage:-?} m3, ${contacts:-?} bumper report(s) on ${objects} object(s)"
        case "${verdict}" in
            CLEAN)
                ;;
            FINISHED_DIRTY)
                dirty=$((dirty + 1))
                echo "[both]   FINISHED but touched ${objects} object(s) -- see ${dir}/monitor.log"
                ;;
            *)
                failed=1
                failed_legs=$((failed_legs + 1))
                echo "[both] ${map} did NOT finish."
                echo "[both]   artifacts: ${dir}"
                echo "[both]   start with: ${dir}/progress.jsonl (coverage, confinement, plan-origin gap)"
                if [[ "${KEEP_GOING}" == "1" ]]; then
                    # Measuring, not gating: one round is one sample of a
                    # process whose coverage has ranged 96 to 736 m3 on an
                    # unchanged configuration, so stopping at the first failure
                    # spends a whole campaign to learn a single bit. Carry on and
                    # report a rate.
                    echo "[both]   KEEP_GOING=1: continuing to the next leg."
                    continue
                fi
                echo "[both] stopping for diagnosis (set KEEP_GOING=1 to measure a rate instead)."
                break 2
                ;;
        esac
    done
done

echo
echo "[both] ==== summary ===="
for d in "${BASE}"/r*_*/; do
    [[ -f "${d}/verdict.json" ]] || continue
    printf '  %-28s %s\n' "$(basename "${d}")" \
        "$(tr -d '\n' < "${d}/verdict.json" | sed 's/  */ /g' | cut -c1-190)"
done
if [[ ${failed} -eq 0 ]]; then
    if [[ ${dirty} -eq 0 ]]; then
        echo "[both] BOTH WORLDS FINISHED CLEAN on one configuration, ${ROUNDS} round(s)."
    else
        echo "[both] BOTH WORLDS FINISHED on one configuration, ${ROUNDS} round(s)"
        echo "[both]   -- but ${dirty} run(s) touched something. Count OBJECTS, not"
        echo "[both]      bumper reports, and check whether each was a first approach"
        echo "[both]      to unmapped geometry (the known near-clip limit) or a real"
        echo "[both]      tracking failure."
    fi
    exit 0
fi
if [[ "${KEEP_GOING}" == "1" ]]; then
    total=0
    for d in "${BASE}"/r*_*/; do
        [[ -f "${d}/verdict.json" ]] && total=$((total + 1))
    done
    echo "[both] MEASURED: ${failed_legs} of ${total} leg(s) did not finish."
    echo "[both]   Per-world tally:"
    for entry in "${WORLDS[@]}"; do
        IFS=':' read -r map _ _ <<< "${entry}"
        ok=0; bad=0
        for d in "${BASE}"/r*_"${map}"/; do
            [[ -f "${d}/verdict.json" ]] || continue
            if grep -qE '"verdict": *"(CLEAN|FINISHED_DIRTY)"' "${d}/verdict.json"; then
                ok=$((ok + 1))
            else
                bad=$((bad + 1))
            fi
        done
        echo "[both]     ${map}: ${ok} finished, ${bad} did not"
    done
fi
exit 1
