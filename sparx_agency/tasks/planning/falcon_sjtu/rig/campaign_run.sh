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
# Early-abort watchdogs. These are now BACKSTOPS, not the primary judgement:
# mission_watchdog_node runs inside the FALCON container with the mapper's own
# coverage figure and the aircraft's pose on one clock, and rules on
# confinement (orbiting a mapped region) that neither of the two below can
# see -- each of them is individually satisfied by exactly that failure. They
# remain because the node can be disabled (watchdog:=false), can fail to start,
# and cannot notice its own container dying.
#
# Their caps are deliberately LONGER than the node's so the node rules first
# and the run gets the specific diagnosis rather than "no new voxels".
NO_GROWTH_CAP_S="${NO_GROWTH_CAP_S:-420}"   # no new occupied voxels for this long -> stop
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
    # The mission watchdog's own 1 Hz record: coverage, confinement radius,
    # growth rate and the plan-origin gap. It is the only artifact that says
    # WHY a run went nowhere rather than merely that it did.
    docker cp "falcon-sjtu:/tmp/mission_progress.jsonl" "${RUN_DIR}/progress.jsonl" 2>/dev/null || true
    kill "${MON_PID:-0}" "${TRK_PID:-0}" 2>/dev/null || true
    docker rm -f falcon-sjtu sjtu-ros1-bridge > /dev/null 2>&1 || true

    local contacts wedges respawns finish_line coverage retreats drifts
    contacts=$(grep -c '\[CONTACT\]' "${RUN_DIR}/monitor.log" 2>/dev/null); contacts=${contacts:-0}
    # Distinct objects touched, which is the metric this package judges on.
    # Gazebo emits a fresh contacts entry per contact point per physics step, so
    # one five-second graze along a crate reads as eight "contacts" -- and a
    # single IV stand grazed repeatedly has read as 186 reports on one object.
    # Reporting only the raw count has already produced one wrong verdict, so
    # the object count belongs in the machine-readable artifact and not just in
    # both_worlds.sh's console line.
    contact_objects=$(grep -oE 'began: [A-Za-z_0-9]+' "${RUN_DIR}/monitor.log" 2>/dev/null | sort -u | wc -l)
    contact_objects=${contact_objects:-0}
    wedges=$(grep -c '\[WEDGED\]' "${RUN_DIR}/monitor.log" 2>/dev/null); wedges=${wedges:-0}
    respawns=$(grep -c 'process has died' "${RUN_DIR}/falcon.log" 2>/dev/null); respawns=${respawns:-0}
    retreats=$(grep -c 'contact/wedge (retreat #' "${RUN_DIR}/falcon.log" 2>/dev/null); retreats=${retreats:-0}
    drifts=$(grep -c 'replanning from the real pose' "${RUN_DIR}/falcon.log" 2>/dev/null); drifts=${drifts:-0}
    # Explored volume is the mission's only honest success signal, so it
    # belongs in the verdict rather than in a log somebody has to go and read.
    coverage=$(tail -1 "${RUN_DIR}/progress.jsonl" 2>/dev/null \
        | sed -n 's/.*"coverage_m3": *\([0-9.]*\).*/\1/p'); coverage=${coverage:-0}
    finish_line=$(grep -m1 'Exploration finished. Start' "${RUN_DIR}/falcon.log" 2>/dev/null | sed 's/.*\[FSM\]/[FSM]/' || true)
    # Did the aircraft fly with its ground-truth contact sense connected?
    # parameter_bridge announces a bridge before it looks for a conversion pair
    # and only then reports the failure, so a bridge image built without
    # gazebo_msgs carries no bumper for the whole run while every check upstream
    # of this one passes. That removes the follower's contact retreat, its
    # three-strikes hold and FALCON's dead-end-guard hand-off, so a contacts
    # count from such a run is not comparable with one from a healthy stack.
    # Recorded rather than judged, exactly as world_models is: a bad sample must
    # be VISIBLE in the artifact afterwards, not silently averaged in.
    local bumper_failed bumper_bridged
    bumper_failed=$(grep -c "failed to create bidirectional bridge for topic '/simple_drone/bumper_states'" \
        "${RUN_DIR}/bridge.log" 2>/dev/null); bumper_failed=${bumper_failed:-0}
    if [[ "${bumper_failed}" -gt 0 ]]; then bumper_bridged=false; else bumper_bridged=true; fi
    cat > "${RUN_DIR}/verdict.json" <<EOF
{"map": "${MAP}", "verdict": "${verdict}", "detail": "${detail}",
 "elapsed_s": ${elapsed}, "contacts": ${contacts},
 "contact_objects": ${contact_objects}, "wedges": ${wedges},
 "planner_respawns": ${respawns}, "coverage_m3": ${coverage},
 "retreats": ${retreats}, "plan_origin_corrections": ${drifts},
 "world_models": ${SPARX_MODEL_COUNT:-0}, "bumper_bridged": ${bumper_bridged},
 "finish_line": "${finish_line//\"/\\\"}"}
EOF
    [[ "${bumper_bridged}" == "true" ]] || \
        say "WARNING: /simple_drone/bumper_states never bridged -- this run flew with NO ground-truth contact sense"
    say "VERDICT ${verdict} (${detail}) after ${elapsed}s: coverage=${coverage} m3 contacts=${contacts} on ${contact_objects} object(s) retreats=${retreats} respawns=${respawns}"
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

# ── 1b. wait for the world to finish LOADING, not just to start ────────────
# The checks above prove gzserver began loading our world file and that a depth
# topic exists. Neither proves the furniture is in it. An AWS world is a few
# dozen separate model directories of meshes, and a mission that starts early
# maps free space where a shelf is about to appear, plans through it, and spends
# the run recovering from collisions with geometry its map says is not there.
#
# No per-world model count is needed, and none is wanted: the property that
# matters is that the count has STOPPED changing. Poll until it is stable across
# three consecutive reads, and record it either way, so a run flown into a
# half-built world is visible in the verdict afterwards rather than silently
# producing a bad sample.
MODEL_COUNT=0
model_count() {
    docker exec "${SIM_CONTAINER}" bash -c '
        source /opt/ros/humble/setup.bash
        export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
        export CYCLONEDDS_URI=file:///etc/cyclonedds/no_shm.xml
        export ROS_DOMAIN_ID='"${SIM_DOMAIN_ID}"'
        timeout 10 ros2 service call /get_model_list gazebo_msgs/srv/GetModelList 2>/dev/null' 2>/dev/null \
        | tr ',' '\n' | grep -c "aws_\|ground_plane\|simple_drone" || true
}
stable=0
prev_count=-1
for _ in $(seq 1 30); do
    MODEL_COUNT="$(model_count)"
    if [[ "${MODEL_COUNT}" -gt 0 && "${MODEL_COUNT}" == "${prev_count}" ]]; then
        stable=$((stable + 1))
        [[ ${stable} -ge 2 ]] && break
    else
        stable=0
    fi
    prev_count="${MODEL_COUNT}"
    sleep 2
done
if [[ ${stable} -ge 2 ]]; then
    say "world settled: ${MODEL_COUNT} models"
else
    say "WARNING: world model count never settled (last ${MODEL_COUNT}); flying anyway"
fi
export SPARX_MODEL_COUNT="${MODEL_COUNT}"

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
        elif [[ "${OCC}" =~ ^[0-9]+$ ]] && (( OCC > 0 )) && (( OCC < LAST_OCC / 2 )); then
            # The count COLLAPSED: FALCON's exploration_node respawns after its
            # in-process LKH solver aborts, and it comes back with an empty map
            # that then rebuilds from scratch. Against a high-water mark from
            # before the crash, a mission re-earning ground at 60 m3/min looks
            # frozen, and this watchdog kills it: measured, a healthy hospital
            # rebuild was cut at 420 s while its own coverage was climbing, the
            # mark stuck at 472,587 occupied voxels from the previous
            # incarnation. Re-baseline instead, exactly as the in-container
            # progress monitor does for the same event.
            say "occupied count collapsed ${LAST_OCC} -> ${OCC}: planner respawn, re-baselining the discovery watchdog"
            LAST_OCC=${OCC}; LAST_OCC_T=${NOW}
        fi
        if (( LAST_OCC > 0 )) && (( NOW - LAST_OCC_T > NO_GROWTH_CAP_S )); then
            finish STALLED_COVERAGE "no new voxels for ${NO_GROWTH_CAP_S}s (occupied=${LAST_OCC})" 3
        fi
    fi

    # ── the mission watchdog's verdict, which outranks everything below ──
    # It runs INSIDE the FALCON container with the mapper's own coverage
    # number and the aircraft's pose on the same clock, so it sees "orbiting
    # a room it has already mapped" -- which every watchdog in this file is
    # blind to, because each of them is individually satisfied by it. The
    # harness stays the executioner; the node only ever declares.
    ABORT_LINE=$(grep -m1 -o '\[watchdog\] MISSION ABORT (\([a-z_]*\)): [^-]*' \
        <(docker logs falcon-sjtu 2>&1) 2>/dev/null | head -1)
    if [[ -n "${ABORT_LINE}" ]]; then
        AB_STATE=$(sed -n 's/.*MISSION ABORT (\([a-z_]*\)).*/\1/p' <<< "${ABORT_LINE}")
        AB_WHY=$(sed -n 's/.*): *//p' <<< "${ABORT_LINE}")
        sleep 5   # let the follower park and the trace flush
        case "${AB_STATE}" in
            abort_time_cap) finish TIMEOUT "watchdog: ${AB_WHY}" 3 ;;
            *)              finish "$(tr 'a-z' 'A-Z' <<< "${AB_STATE}")" "watchdog: ${AB_WHY}" 3 ;;
        esac
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
        # A finish is only a MAPPING if the aircraft actually mapped something.
        # FALCON declares "exploration finished" whenever its frontier finder
        # comes up empty -- including on a near-empty map -- and certifying
        # that as CLEAN is how a campaign fools itself: runs 026/031 and three
        # soak runs "finished untouched" having flown 0.0-10.8 m total.
        #
        # DISTANCE FLOWN IS THE WRONG TEST, and it was only ever a proxy for the
        # right one, chosen before the mapper's own coverage figure was
        # available here. An exploration's job is to observe the volume, not to
        # visit it: the depth camera reaches 5 m, so in an open world the
        # aircraft legitimately finishes having flown a fraction of the floor it
        # mapped. Measured: a warehouse run covering 201.9 m3 of a 204.1 m3 box
        # -- 98.9%, a complete map -- was failed by this guard for flying 30.1 m
        # against a 40 m bar.
        #
        # So the guard now needs BOTH signals to agree before it calls a finish
        # trivial: a short flight AND a map that is nearly empty. Either one
        # alone is a proxy, and the coverage one is the proxy for nothing -- it
        # is the thing itself.
        PATH_M=$(docker cp "${SIM_CONTAINER}:/tmp/flight_trace.jsonl" - 2>/dev/null | tar -xO 2>/dev/null | awk -F'[,:]' '
            /"x"/ { for (i=1;i<=NF;i++) { if ($i ~ /"x"/) x=$(i+1)+0; if ($i ~ /"y"/) y=$(i+1)+0 }
                    if (seen) { d=sqrt((x-px)^2+(y-py)^2); s+=d } px=x; py=y; seen=1 }
            END { printf "%.1f", s }')
        PATH_M=${PATH_M:-0}
        COV_M3=$(docker exec falcon-sjtu bash -lc 'tail -1 /tmp/mission_progress.jsonl' 2>/dev/null \
            | sed -n 's/.*"coverage_m3": *\([0-9.]*\).*/\1/p'); COV_M3=${COV_M3:-0}
        if awk -v p="${PATH_M}" -v m="${MIN_PATH_M:-40}" \
               -v c="${COV_M3}" -v k="${MIN_COVERAGE_M3:-50}" \
               'BEGIN{exit !(p < m && c < k)}'; then
            finish TRIVIAL_FINISH "finished after only ${PATH_M} m of flight and ${COV_M3} m3 of map (< ${MIN_PATH_M:-40} m and < ${MIN_COVERAGE_M3:-50} m3): empty-map false finish" 1
        fi
        # A finish that leaves most of the world unmapped is a FAILURE, and it
        # is the one this harness was still certifying as success. FALCON
        # declares "exploration finished" the moment its frontier finder comes
        # up empty, and that happens early for reasons that have nothing to do
        # with the world being explored: viewpoints retired by the dead-end
        # guard, a coverage tour whose connectivity model is one z slice, an
        # unlucky sequence of blocked regions. Measured: a hospital run that
        # "FINISHED" having covered 260.8 m3 and never left y > -2.3, i.e. the
        # north third of a 56 m building, while the same configuration had
        # reached y = -33.6 and 760 m3 an hour earlier.
        #
        # The floor is per world and belongs to the CALLER, because only the
        # caller knows what the world affords -- both_worlds.sh sets it from the
        # measured explorable volume of each box. Unset means no check, which is
        # what an ad-hoc single run wants.
        if [[ -n "${FINISH_MIN_COVERAGE_M3:-}" ]] \
           && awk -v c="${COV_M3}" -v k="${FINISH_MIN_COVERAGE_M3}" 'BEGIN{exit !(c < k)}'; then
            finish PARTIAL_FINISH "finished having mapped only ${COV_M3} m3 (< ${FINISH_MIN_COVERAGE_M3} m3 expected for this world) after ${PATH_M} m" 1
        fi
        if [[ "${C}" -eq 0 ]]; then finish CLEAN "finished untouched after ${PATH_M} m, ${COV_M3} m3" 0
        else finish FINISHED_DIRTY "finished with ${C} contact(s) after ${PATH_M} m, ${COV_M3} m3" 1; fi
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
