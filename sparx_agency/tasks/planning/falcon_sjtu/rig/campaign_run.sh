#!/usr/bin/env bash
# ============================================================
# campaign_run.sh — one hermetic FALCON exploration run, with a verdict.
#
#   ./campaign_run.sh <map> <run_dir> [time_cap_s]
#
# The engine of the reliability campaign: restart the world (drone back at
# spawn), fly one full exploration headless, watch the PHYSICS (flight_monitor
# inside the sim container) and the PLANNER (falcon log) simultaneously, and
# leave behind a run directory a human or a later analysis can trust:
#
#   monitor.log     flight_monitor stdout: status lines + CONTACT/CAPSIZE/...
#   trace.jsonl     5 Hz pose/attitude/contact trace (physics ground truth)
#   tracking.csv    the follower's ~tracking diagnostics (control-side view)
#   rtf.log         periodic Gazebo real-time-factor samples
#   falcon.log      the FALCON container's full log
#   bridge.log      the ros1_bridge container's log
#   verdict.json    machine-readable outcome
#
# Exit codes: 0 CLEAN (finished, zero contacts), 1 FINISHED_DIRTY (finished but
# touched something), 2 FATAL (capsize/grounded), 3 TIMEOUT, 4 INFRA.
# ============================================================
set -uo pipefail

MAP="${1:?map name (config/<map>.yaml)}"
RUN_DIR="${2:?run directory}"
TIME_CAP_S="${3:-2100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export DISPLAY="${DISPLAY:-:1}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
# Gazebo world name and therefore sim container name; defaults to the map name
# (hospital -> hospital) but e.g. small_warehouse's world file is
# no_roof_small_warehouse.
WORLD="${WORLD:-${MAP}}"
SIM_CONTAINER="sjtu_drone_${WORLD}"
# The domain every `docker exec ... ros2` probe below must join. Same rule as
# run_falcon_sjtu.sh: follow the shell's ROS_DOMAIN_ID (which is also what
# bringup_world.sh -> env.sh gives the sim), then 20. Hardcoding 20 while the
# sim followed the shell made every probe query an empty domain, which is
# indistinguishable from a simulator that never started.
SIM_DOMAIN_ID="${SIM_DOMAIN_ID:-${ROS_DOMAIN_ID:-20}}"
POLL_S=10
# Early-abort watchdogs: a run that stops DISCOVERING or stops MOVING is over,
# whatever the planner believes -- waiting out the full cap on it is wasted
# wall-clock. Healthy warehouse missions finish in ~4 min; five silent minutes
# is diagnosis material, not patience material.
NO_GROWTH_CAP_S="${NO_GROWTH_CAP_S:-300}"   # no new occupied voxels for this long -> stop
NO_MOVE_CAP_S="${NO_MOVE_CAP_S:-300}"       # no net movement (0.5 m) for this long -> stop

mkdir -p "${RUN_DIR}"
say() { echo "[campaign $(date +%H:%M:%S)] $*"; }

finish() {
    local verdict="$1" detail="$2" rc="$3"
    local elapsed=$(( $(date +%s) - T0 ))
    # ── collect artifacts (best effort, the verdict must survive anything) ──
    docker logs falcon-sjtu > "${RUN_DIR}/falcon.log" 2>&1 || true
    docker logs sjtu-ros1-bridge > "${RUN_DIR}/bridge.log" 2>&1 || true
    docker cp "${SIM_CONTAINER}:/tmp/flight_trace.jsonl" "${RUN_DIR}/trace.jsonl" 2>/dev/null || true
    kill "${MON_PID:-0}" "${TRK_PID:-0}" 2>/dev/null || true
    docker rm -f falcon-sjtu sjtu-ros1-bridge > /dev/null 2>&1 || true

    local contacts wedges respawns finish_line
    contacts=$(grep -c '\[CONTACT\]' "${RUN_DIR}/monitor.log" 2>/dev/null); contacts=${contacts:-0}
    wedges=$(grep -c '\[WEDGED\]' "${RUN_DIR}/monitor.log" 2>/dev/null); wedges=${wedges:-0}
    respawns=$(grep -c 'process has died' "${RUN_DIR}/falcon.log" 2>/dev/null); respawns=${respawns:-0}
    finish_line=$(grep -m1 'Exploration finished. Start' "${RUN_DIR}/falcon.log" 2>/dev/null | sed 's/.*\[FSM\]/[FSM]/' || true)
    cat > "${RUN_DIR}/verdict.json" <<EOF
{"map": "${MAP}", "verdict": "${verdict}", "detail": "${detail}",
 "elapsed_s": ${elapsed}, "contacts": ${contacts}, "wedges": ${wedges},
 "planner_respawns": ${respawns},
 "finish_line": "${finish_line//\"/\\\"}"}
EOF
    say "VERDICT ${verdict} (${detail}) after ${elapsed}s: contacts=${contacts} wedges=${wedges} respawns=${respawns}"
    exit "${rc}"
}

# ── 1. fresh world: drone back at spawn, physics from scratch ──────────────
# Kill EVERY sjtu_drone_* container, not just this world's: a survivor from a
# different world keeps gzserver's port, the new world's server dies silently,
# and DDS being host-wide means every topic check then passes against the
# WRONG world -- two runs flew the hospital wearing the warehouse config
# before this was caught.
say "restarting sim (${SIM_CONTAINER})"
docker ps -a --format '{{.Names}}' | grep '^sjtu_drone_' | xargs -r docker rm -f > /dev/null 2>&1 || true
docker rm -f falcon-sjtu sjtu-ros1-bridge falcon-rviz "${SIM_CONTAINER}" > /dev/null 2>&1 || true
sleep 2
nohup bash "${PKG_DIR}/../../../robots/SJTU/setup/bringup_world.sh" --skip-build "${WORLD}" \
    > "${RUN_DIR}/sim_bringup.log" 2>&1 &
SIM_UP=0
for _ in $(seq 1 40); do
    if docker exec "${SIM_CONTAINER}" bash -c '
            source /opt/ros/humble/setup.bash
            export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
            export CYCLONEDDS_URI=file:///etc/cyclonedds/no_shm.xml
            export ROS_DOMAIN_ID='"${SIM_DOMAIN_ID}"'
            ros2 topic list 2>/dev/null' 2>/dev/null | grep -q front_depth; then
        SIM_UP=1; break
    fi
    sleep 5
done
[[ "${SIM_UP}" == "1" ]] || { echo '{"verdict":"INFRA","detail":"sim never published depth"}' > "${RUN_DIR}/verdict.json"; exit 4; }
if docker logs "${SIM_CONTAINER}" 2>&1 | grep -q "Unable to create CameraSensor"; then
    echo '{"verdict":"INFRA","detail":"cameras disabled (no display)"}' > "${RUN_DIR}/verdict.json"; exit 4
fi
# Positive world identification: the topics existing proves a drone is alive
# SOMEWHERE on the domain, not that it is in the right world. gzserver logs
# the world file it actually loaded; demand it is ours.
if ! docker logs "${SIM_CONTAINER}" 2>&1 | grep "Loading world file" | grep -q "${WORLD}.world"; then
    echo "{\"verdict\":\"INFRA\",\"detail\":\"wrong world: gzserver did not load ${WORLD}.world\"}" > "${RUN_DIR}/verdict.json"
    say "WRONG WORLD (gzserver did not load ${WORLD}.world)"; exit 4
fi
say "sim up (world verified: ${WORLD})"

# ── 2. the physics witness, before anything can move ───────────────────────
docker cp "${SCRIPT_DIR}/flight_monitor.py" "${SIM_CONTAINER}:/tmp/flight_monitor.py"
docker exec "${SIM_CONTAINER}" bash -c '
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file:///etc/cyclonedds/no_shm.xml
    export ROS_DOMAIN_ID='"${SIM_DOMAIN_ID}"'
    exec python3 /tmp/flight_monitor.py --trace /tmp/flight_trace.jsonl' \
    > "${RUN_DIR}/monitor.log" 2>&1 &
MON_PID=$!

# ── 3. FALCON + bridge, headless ───────────────────────────────────────────
T0=$(date +%s)
# RVIZ=1 to WATCH a campaign run (map, frontiers, tour, drone) while it still
# produces the full monitor/verdict artifacts; default off for soaks.
RVIZ="${RVIZ:-0}" FOLLOW=0 bash "${PKG_DIR}/run_falcon_sjtu.sh" "${MAP}" \
    > "${RUN_DIR}/stack_up.log" 2>&1 \
    || finish INFRA "run_falcon_sjtu failed" 4
# the follower's control-side view of the same flight
docker exec falcon-sjtu bash -c '
    source /opt/ros/noetic/setup.bash
    exec rostopic echo -p /bspline_follower/tracking' \
    > "${RUN_DIR}/tracking.csv" 2>/dev/null &
TRK_PID=$!

# ── 4. watch until someone declares an ending ──────────────────────────────
LAST_RTF_T=0
LAST_OCC=0; LAST_OCC_T=$(date +%s); LAST_OCC_SAMPLE_T=0
ANCHOR_X=""; ANCHOR_Y=""; ANCHOR_T=$(date +%s)
while true; do
    sleep "${POLL_S}"
    NOW=$(date +%s); ELAPSED=$(( NOW - T0 ))

    # position watchdog: net movement from the anchor, monitor's own numbers
    POSLINE=$(tail -1 "${RUN_DIR}/monitor.log" 2>/dev/null | grep -oE 'pos=\(\s*[-0-9.]+,\s*[-0-9.]+' | tr -d 'pos=( ')
    if [[ -n "${POSLINE}" ]]; then
        PX="${POSLINE%%,*}"; PY="${POSLINE##*,}"
        if [[ -z "${ANCHOR_X}" ]]; then ANCHOR_X="${PX}"; ANCHOR_Y="${PY}"; ANCHOR_T=${NOW}; fi
        MOVED=$(awk -v ax="${ANCHOR_X}" -v ay="${ANCHOR_Y}" -v x="${PX}" -v y="${PY}" \
                'BEGIN{print (((x-ax)^2+(y-ay)^2) > 0.25) ? 1 : 0}')
        if [[ "${MOVED}" == "1" ]]; then ANCHOR_X="${PX}"; ANCHOR_Y="${PY}"; ANCHOR_T=${NOW}; fi
        if (( NOW - ANCHOR_T > NO_MOVE_CAP_S )); then
            finish STALLED_POSITION "no net movement for ${NO_MOVE_CAP_S}s" 3
        fi
    fi

    # discovery watchdog: total occupied voxels from the mapper's full sweeps
    if (( NOW - LAST_OCC_SAMPLE_T >= 60 )); then
        LAST_OCC_SAMPLE_T=${NOW}
        OCC=$(docker exec falcon-sjtu bash -c 'source /opt/ros/noetic/setup.bash; timeout 8 rostopic echo -n1 --noarr /voxel_mapping/occupancy_grid_occupied 2>/dev/null | awk "/^width/{print \$2}"' 2>/dev/null); OCC=${OCC:-0}
        if [[ "${OCC}" =~ ^[0-9]+$ ]] && (( OCC > LAST_OCC + 25 )); then
            LAST_OCC=${OCC}; LAST_OCC_T=${NOW}
        fi
        if (( LAST_OCC > 0 )) && (( NOW - LAST_OCC_T > NO_GROWTH_CAP_S )); then
            finish STALLED_COVERAGE "no new voxels for ${NO_GROWTH_CAP_S}s (occupied=${LAST_OCC})" 3
        fi
    fi

    # the physics witness died => it printed a terminal verdict
    if ! kill -0 "${MON_PID}" 2>/dev/null; then
        V=$(grep -oE 'flight ended: [A-Z]+' "${RUN_DIR}/monitor.log" | tail -1 | awk '{print $3}')
        finish FATAL "${V:-monitor died}" 2
    fi
    # infra died under us
    docker ps --format '{{.Names}}' | grep -q "^${SIM_CONTAINER}$" || finish INFRA "sim container died" 4
    docker ps --format '{{.Names}}' | grep -q '^falcon-sjtu$' || finish INFRA "falcon container died" 4

    # FALCON says it is done. Match the FOLLOWER's reaction line (printed
    # every tick once replan=2 arrives) rather than only the FSM's own line,
    # whose ANSI-coloured single print slipped past this grep and turned the
    # campaign's first genuine finish into a TIMEOUT verdict.
    # grep -c, not -q: with pipefail, -q's early exit SIGPIPEs docker logs and
    # the pipeline reads as NO MATCH once the log is large -- a finish that
    # spams thousands of holding-station lines is exactly when it matters.
    FIN=$(docker logs falcon-sjtu 2>&1 | grep -cE '\[follower\] exploration finished|Exploration finished'); FIN=${FIN:-0}
    if [[ "${FIN}" -gt 0 ]]; then
        sleep 15   # grace: a capsize on the final hover still counts
        if ! kill -0 "${MON_PID}" 2>/dev/null; then
            V=$(grep -oE 'flight ended: [A-Z]+' "${RUN_DIR}/monitor.log" | tail -1 | awk '{print $3}')
            finish FATAL "${V:-late fatal}" 2
        fi
        # NB grep -c PRINTS 0 and exits 1 on no match: `|| echo 0` would emit
        # "0\n0" and poison the -eq below (it cost run 031 its CLEAN verdict).
        C=$(grep -c '\[CONTACT\]' "${RUN_DIR}/monitor.log" 2>/dev/null); C=${C:-0}
        # A finish is only a MAPPING if the aircraft actually went somewhere.
        # FALCON declares "exploration finished" whenever its frontier finder
        # comes up empty -- including on a near-empty map -- and certifying
        # that as CLEAN is how a campaign fools itself: runs 026/031 and three
        # soak runs "finished untouched" having flown 0.0-10.8 m total.
        PATH_M=$(docker cp "${SIM_CONTAINER}:/tmp/flight_trace.jsonl" - 2>/dev/null | tar -xO 2>/dev/null | awk -F'[,:]' '
            /"x"/ { for (i=1;i<=NF;i++) { if ($i ~ /"x"/) x=$(i+1)+0; if ($i ~ /"y"/) y=$(i+1)+0 }
                    if (seen) { d=sqrt((x-px)^2+(y-py)^2); s+=d } px=x; py=y; seen=1 }
            END { printf "%.1f", s }')
        PATH_M=${PATH_M:-0}
        if awk -v p="${PATH_M}" -v m="${MIN_PATH_M:-40}" 'BEGIN{exit !(p < m)}'; then
            finish TRIVIAL_FINISH "finished after only ${PATH_M} m of flight (< ${MIN_PATH_M:-40} m): empty-map false finish" 1
        fi
        if [[ "${C}" -eq 0 ]]; then finish CLEAN "finished untouched after ${PATH_M} m" 0
        else finish FINISHED_DIRTY "finished with ${C} contact(s) after ${PATH_M} m" 1; fi
    fi

    # periodic real-time factor + per-container CPU/mem samples (cheap;
    # diagnose clock skew and any compute cost that grows with map size)
    if (( NOW - LAST_RTF_T >= 60 )); then
        LAST_RTF_T=${NOW}
        docker exec "${SIM_CONTAINER}" bash -c 'timeout 3 gz stats -p 2>/dev/null | tail -1' \
            >> "${RUN_DIR}/rtf.log" 2>/dev/null || true
        docker stats --no-stream --format "${ELAPSED}s {{.Name}} {{.CPUPerc}} {{.MemUsage}}" \
            >> "${RUN_DIR}/stats.log" 2>/dev/null || true
    fi

    (( ELAPSED < TIME_CAP_S )) || finish TIMEOUT "cap ${TIME_CAP_S}s" 3
done
