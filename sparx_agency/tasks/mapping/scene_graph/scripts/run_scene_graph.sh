#!/usr/bin/env bash
# ============================================================
# run_scene_graph.sh — the ONE command that brings the whole
# scene-graph mission up: Gazebo world, ollama LLM, YOLO-World
# detection server, FALCON exploration (+ ROS1 bridge + BEV),
# and the eight host ROS2 nodes.
#
#   ./run_scene_graph.sh                          # hospital, target=wheelchair
#   ./run_scene_graph.sh --target "hospital bed"
#   ./run_scene_graph.sh --no-sim --no-falcon     # world+FALCON already up
#
# Flags:
#   --world <name>    Gazebo world (default hospital). Anything else gets a
#                     warning: robots/SJTU/maps/hospital_doors.yaml AND the viz
#                     backdrop robots/SJTU/maps/hospital.yaml are hospital-only,
#                     so the run continues with NO surveyed doors and the trail
#                     drawn on the wrong floor plan.
#   --target <str>    object to search for (default wheelchair)
#   --no-sim          the world is already up; do not start it
#   --no-falcon       do not start FALCON/bridge/rviz (no BEV -> no rooms)
#   --no-llm          skip ollama; classifier logs failures and keeps stale
#                     labels, the oracle emits source=uniform_fallback
#   --out <dir>       run directory (default /tmp/scene_graph/<UTC stamp>)
#   --attach          after bring-up, tail -f the viz node's log (Ctrl-C
#                     detaches; the stack stays up). Default: exit 0, leaving
#                     everything running.
#   --rviz            also open rviz2 on config/scene_graph.rviz: the room
#                     graph itself -- rooms in their own colours, the Voronoi
#                     spine, and the gold room->door->room edges. Needs
#                     $DISPLAY. This is a SEPARATE rviz2 from FALCON's own
#                     container RViz (that one is ROS1 and cannot see these).
#   --no-search       do not start room_search_node (the LLM ranking then only
#                     describes the building instead of choosing where to look)
#   --no-approach     do not start target_approach_node, so finding the target
#                     ends the mission at a log line and FALCON keeps
#                     exploring. With it started (the default) the node is
#                     still completely inert until /target_seen latches: it
#                     publishes nothing and holds no camera, and only then
#                     mutes the FALCON follower and flies the last leg itself.
#   --fly             arm the search loop. NOT wired end to end yet: the
#                     two-gate cmd_vel arbiter is still missing, so this
#                     publishes a route and a handover flag nothing consumes.
#
# Env overrides: DETECT_PORT, DETECT_MODEL, LLM_MODEL, RVIZ, VIZ_WINDOW,
#   KILL_STALE, SKIP_GPU_CHECK, FALCON_LAUNCH_ARGS.
#
# GPU discipline (the 8 GB card is EXCLUSIVE): YOLO-World owns it, alone.
# Gazebo renders on the CPU (llvmpipe, bringup_world.sh does this), every host
# node runs with CUDA_VISIBLE_DEVICES="", and ollama runs CPU-only by design.
# The gate below refuses to start while anything else holds the card.
#
# Everything long-running is nohup'd with its log and pid under --out.
# Tear down with:  scripts/stop_scene_graph.sh --out <dir> [--all]
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

say() { echo "[scene-graph] $*"; }
die() { echo "[scene-graph] ERROR: $*" >&2; exit 1; }

# Exported HERE and not further down, because the LLM smoke check below is a
# `python -m sparx_agency....` and runs long before the ROS section. Setting it
# late made the whole mission refuse to start from any CWD but the repo root,
# with a bare ModuleNotFoundError from a step called "LLM smoke check".
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── flags ─────────────────────────────────────────────────────────────────
WORLD="hospital"
TARGET="wheelchair"
NO_SIM=0
NO_FALCON=0
NO_LLM=0
ATTACH=0
RVIZ_SCENE=0
NO_SEARCH=0
SEARCH_FLY=0
OBJECT_SEARCH=0
NO_APPROACH=0
OUT_DIR=""
MAP_NAME=""     # set in the FALCON section; referenced by the status block
while [[ $# -gt 0 ]]; do
    case "$1" in
        --world)  WORLD="${2:?--world needs a value}"; shift 2 ;;
        --target) TARGET="${2:?--target needs a value}"; shift 2 ;;
        --no-sim)    NO_SIM=1; shift ;;
        --no-falcon) NO_FALCON=1; shift ;;
        --no-llm)    NO_LLM=1; shift ;;
        --out)    OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
        --attach) ATTACH=1; shift ;;
        # The scene-graph RViz view (config/scene_graph.rviz) is the room graph
        # itself -- coloured rooms, the Voronoi spine, and the gold room->door->
        # room edges. It is a SEPARATE rviz2 from FALCON's own container RViz,
        # which runs ROS1 on the ROS1 master and cannot see these markers.
        --rviz)   RVIZ_SCENE=1; shift ;;
        # room_search_node is what makes the LLM ranking steer the mission
        # instead of merely describing it. It never commands the aircraft
        # unless armed (see --fly), so it is on by default.
        --no-search) NO_SEARCH=1; shift ;;
        # target_approach_node is the ONLY node in this stack that ever takes
        # the aircraft off FALCON, and it does so only after /target_seen has
        # latched. Before that it publishes nothing at all, so leaving it out
        # buys the exploration flight nothing -- this flag is here for the run
        # where you want the target FOUND and the mapping to carry on anyway.
        --no-approach) NO_APPROACH=1; shift ;;
        # object_search_node: the full find-an-object loop -- arc weights
        # between room centres, a solver order over them, our own A* transit
        # with FALCON muted, then a frontier sweep bounded to the chosen
        # room's mask under a budget. Replaces room_search_node for the run
        # (both may be launched, but only one may be armed, so this flag
        # implies --no-search).
        --object-search) OBJECT_SEARCH=1; NO_SEARCH=1; shift ;;
        # Arming the search loop. OFF by default. With --object-search this
        # is now wired end to end: cmd_vel_arbiter_node holds the FALCON mute
        # and gates our follower's twists behind it, so exactly one process
        # writes cmd_vel at every instant.
        --fly)    SEARCH_FLY=1; shift ;;
        # Print the header banner to its real end rather than to a line number
        # that drifts every time the banner is edited.
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown flag '$1' (see --help)" ;;
    esac
done

# room_confine starts the ROS1 shim that turns /scene_graph/confine into
# FALCON's leased keep-in boxes. Only meaningful with the object-search loop
# and an image carrying falcon_room_confine.patch; harmless otherwise (the
# node simply never receives a request).
FALCON_CONFINE_ARG=""
if [[ "${OBJECT_SEARCH}" == "1" && "${SEARCH_BACKEND:-falcon}" == "falcon" ]]; then
    FALCON_CONFINE_ARG="room_confine:=true"
fi

OUT_DIR="${OUT_DIR:-/tmp/scene_graph/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}/pids" "${OUT_DIR}/viz" \
    || die "could not create the run directory ${OUT_DIR}"
# ABSOLUTE from here on. The detection server is started from a subshell that
# cd's to $HOME (ultralytics litters the CWD), so a relative --out would put its
# log and pid file under $HOME while every other step used the CWD-relative
# path -- and the liveness check would then report a healthy server as dead.
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

if [[ "${WORLD}" != "hospital" ]]; then
    say "WARNING: --world ${WORLD}: robots/SJTU/maps/hospital_doors.yaml is"
    say "  surveyed for the HOSPITAL only. Continuing with NO doors: rooms come"
    say "  purely from the online Voronoi-skeleton cuts, and door discovery"
    say "  stays empty. The viz backdrop (maps/hospital.yaml) is hospital-only"
    say "  too, so the trail will be drawn on the wrong floor plan."
fi

# ── environment every ROS2 participant must share ─────────────────────────
# (copied from sjtu_internvla_n1/scripts/run_sjtu_n1.sh — the sim runs in
# Docker, and a domain or middleware mismatch is silently zero data)
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
SETUP_DIR="${REPO_ROOT}/sparx_agency/robots/SJTU/setup"
[[ -x "${SETUP_DIR}/bringup_world.sh" ]] \
    || die "missing ${SETUP_DIR}/bringup_world.sh — is REPO_ROOT (${REPO_ROOT}) right?"

# The middleware is chosen from what is installed, not assumed: CycloneDDS is
# the only one the ROS1 bridge can reach; Fast DDS is the fallback for a box
# without it. Both profiles turn shared memory OFF — SHM does not cross the
# container boundary, and with it on discovery succeeds while every sample is
# dropped.
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
        SIM_RMW="cyclonedds"
        # An unreadable CycloneDDS config is not a warning: the participant
        # aborts at creation and every node dies with a message about the
        # config, not about the path that produced it.
        [[ -f "${SETUP_DIR}/cyclonedds_no_shm.xml" ]] \
            || die "CYCLONEDDS_URI points into ${SETUP_DIR}, but cyclonedds_no_shm.xml is not there." ;;
    rmw_fastrtps_cpp)
        export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${SETUP_DIR}/fastdds_udp_only.xml}"
        export FASTDDS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE}"
        SIM_RMW="fastrtps"
        [[ -f "${SETUP_DIR}/fastdds_udp_only.xml" ]] \
            || die "FASTRTPS_DEFAULT_PROFILES_FILE points into ${SETUP_DIR}, but fastdds_udp_only.xml is not there." ;;
    *)
        die "RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION}' is not one this stack knows." ;;
esac

# ── the WORLD's middleware is not the HOST's ──────────────────────────────
# These are two independent participants and conflating them cost a run.
#
# What is genuinely non-negotiable: the ROS1 bridge carries depth and odom INTO
# FALCON and the BEV grid back out, and run_falcon_sjtu.sh starts it with
# RMW_IMPLEMENTATION=rmw_cyclonedds_cpp hardcoded. So the WORLD must speak
# CycloneDDS or the bridge bridges nothing — no depth, no odom, no
# /falcon/bev_2d, no rooms — for the whole run, while every container reads Up.
#
# What is NOT true: that the world's middleware follows the host's. The world
# runs inside $SJTU_CYCLONE_IMAGE, which ships its own CycloneDDS; whether the
# HOST has ros-${ROS_DISTRO}-rmw-cyclonedds-cpp installed says nothing about
# what the container can speak. This script used to derive the world's --rmw
# from the host's RMW_IMPLEMENTATION and then refuse the Fast-DDS world it had
# just asked for — a self-inflicted abort on a box that runs the mission fine.
#
# So: pin the WORLD to CycloneDDS whenever the bridge is in the graph, and let
# the host nodes keep whatever they have. Cross-vendor RTPS is what carries the
# host half, and it is measured, not assumed — on this box (host Fast DDS, sim
# and bridge CycloneDDS, domain 20) RGB arrives ~20 Hz, depth ~4.6 Hz, and
# /falcon/bev_2d crosses with its TRANSIENT_LOCAL latch intact.
if [[ "${NO_FALCON}" == "0" ]]; then
    SIM_RMW="cyclonedds"
    if [[ "${RMW_IMPLEMENTATION}" != "rmw_cyclonedds_cpp" ]]; then
        say "host RMW is ${RMW_IMPLEMENTATION}; the world and bridge are pinned to"
        say "  CycloneDDS (the bridge speaks nothing else). Host nodes reach the"
        say "  sim cross-vendor. If the host half is ever silent while the"
        say "  containers are Up, that pairing is the first thing to re-measure:"
        say "      sudo apt install ros-${ROS_DISTRO}-rmw-cyclonedds-cpp"
    fi
fi

# LLM contract, matching core/mapping/topology/llm_client.py's from_env().
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}"
# 30 s (llm_client's own default) is not enough here, and the way it fails is
# misleading: the oracle prompt grows with the ROOM COUNT, and this model runs
# on the CPU because the GPU belongs to YOLO. Measured on this box, a 10-room
# ranking takes over 30 s, times out, retries once, and only then degrades --
# so the panel reads source=uniform_fallback with every room at 1/N, which
# looks like the LLM is down rather than merely slow. It got WORSE as the
# segmentation improved, because better rooms mean more of them.
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}"

VENV_PY="${REPO_ROOT}/.venv/bin/python"
NAVDP_PY="${HOME}/miniconda3/envs/navdp/bin/python"
DETECT_PORT="${DETECT_PORT:-8092}"
DETECT_MODEL="${DETECT_MODEL:-${REPO_ROOT}/yolov8s-worldv2.pt}"
OLLAMA_NAME="ollama-scene-graph"

say "world=${WORLD}  target='${TARGET}'  out=${OUT_DIR}"
say "domain ${ROS_DOMAIN_ID}  rmw ${RMW_IMPLEMENTATION}  llm ${LLM_BACKEND}/${LLM_MODEL}"

# ── (a) preflight ─────────────────────────────────────────────────────────
[[ -x "${VENV_PY}" ]] || die ".venv missing at ${REPO_ROOT}/.venv — the host nodes run on it."
[[ -x "${NAVDP_PY}" ]] || die "conda env 'navdp' missing at ${NAVDP_PY} — the detection server needs torch."
command -v curl >/dev/null || die "curl is not on PATH; every health poll below uses it."
command -v docker >/dev/null || die "docker is not on PATH; the world, the bridge and ollama are all containers."
if [[ "${NO_SIM}" == "0" ]]; then
    [[ -n "${SJTU_PROJECT_DIR:-}" && -d "${SJTU_PROJECT_DIR:-}" ]] \
        || die "SJTU_PROJECT_DIR is unset or not a directory. Run:
    export SJTU_PROJECT_DIR=\$HOME/GIT/sjtu_project
  (it must be the external sim checkout, on branch xtend_integration_nadav)"
    # Gazebo Classic silently disables EVERY camera sensor without an X display
    # — the drone still flies and publishes odom, so the failure reads as a
    # camera bug, not a display one.
    [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]] \
        || die "DISPLAY is unset (or /tmp/.X11-unix missing). Gazebo's cameras die
  silently headless. On this machine: export DISPLAY=:1"
fi

# A node left over from a previous run is a second writer on every latched
# topic. Refuse rather than join in (KILL_STALE=1 auto-kills, like run_sjtu_n1).
STALE="$(pgrep -f 'sparx_agency\.tasks\.mapping\.scene_graph\.ros2\.' 2>/dev/null | tr '\n' ' ')"
if [[ -n "${STALE// /}" ]]; then
    if [[ "${KILL_STALE:-1}" == "1" ]]; then
        say "killing stale scene-graph nodes from a previous run: ${STALE}"
        # shellcheck disable=SC2086
        kill -9 ${STALE} 2>/dev/null || true
        sleep 1
    else
        die "scene-graph nodes from a previous run are still up (${STALE}).
  Run scripts/stop_scene_graph.sh first, or rerun with KILL_STALE=1."
    fi
fi

# ── (b) GPU gate: the card must be YOLO-World's alone ─────────────────────
detect_healthy() {
    curl -sf -m 5 "http://127.0.0.1:${DETECT_PORT}/health" 2>/dev/null | grep -q '"ok": *true'
}
if detect_healthy; then
    say "detection server already healthy on :${DETECT_PORT}; leaving the card to it"
elif [[ "${SKIP_GPU_CHECK:-0}" == "1" ]]; then
    # Expert-only bypass: you are asserting you know what holds the card. A
    # second CUDA context in the leftovers of an 8 GB card has HARD-LOCKED this
    # host before.
    say "SKIP_GPU_CHECK=1 — skipping the GPU gate (on your head)"
else
    say "checking the GPU is empty (YOLO-World needs the card)..."
    "${VENV_PY}" "${REPO_ROOT}/sparx_agency/tasks/planning/sjtu_internvla_n1/scripts/check_gpu_free.py" \
        --require-empty
    GPU_RC=$?
    case "${GPU_RC}" in
        0) ;;
        3) die "no nvidia-smi on PATH, so the GPU gate could not run. The detection
  server needs a CUDA device; install the driver tools, or SKIP_GPU_CHECK=1." ;;
        *) die "GPU is not free (check_gpu_free exit ${GPU_RC}). Free it (or
  SKIP_GPU_CHECK=1, expert-only), then rerun." ;;
    esac
fi

# ── (c) LLM: ollama in docker, CPU-only ───────────────────────────────────
# The smoke check runs against the TARGET this run is about to fly, not against
# llm_check's own placeholder: a model that ranks 'car keys' fine and falls
# apart on 'hospital bed' is exactly what this step exists to catch.
run_llm_check() {
    say "running the LLM smoke check (llm_check --target '${TARGET}')..."
    "${VENV_PY}" -m sparx_agency.tasks.mapping.scene_graph.scripts.llm_check \
        --target "${TARGET}" 2>&1 | tee "${OUT_DIR}/llm_check.log"
    [[ "${PIPESTATUS[0]}" == "0" ]] \
        || die "LLM smoke check FAILED against ${LLM_BASE_URL} — the classifier and
  oracle would fail all run long. See ${OUT_DIR}/llm_check.log (or rerun with
  --no-llm to accept uniform-fallback probabilities)."
}
if [[ "${NO_LLM}" == "1" ]]; then
    say "--no-llm: skipping ollama. room_classifier will log LLM failures and"
    say "  keep stale labels; llm_oracle will publish source=uniform_fallback."
elif [[ "${LLM_BACKEND}" != "ollama" ]]; then
    # An openai-compat backend is somebody else's server; starting a local
    # ollama and pulling a 2 GB model for it would be pure waste. The smoke
    # check still runs, against whatever LLM_BASE_URL names.
    say "LLM_BACKEND=${LLM_BACKEND} (not ollama): no container to manage."
    run_llm_check
else
    # Idempotent against EVERY state the container can be in, read from docker
    # rather than inferred from `docker ps` output: a paused container is listed
    # by `docker ps` as running and answers nothing, and `docker start` on it
    # fails. This is the common case in practice -- the container is normally
    # already up from the last run with the model still in its volume.
    OLLAMA_STATE="$(docker inspect -f '{{.State.Status}}' "${OLLAMA_NAME}" 2>/dev/null || true)"
    case "${OLLAMA_STATE}" in
        running)
            say "ollama container '${OLLAMA_NAME}' already running" ;;
        paused)
            say "unpausing ollama container '${OLLAMA_NAME}'"
            docker unpause "${OLLAMA_NAME}" >/dev/null || die "could not unpause ${OLLAMA_NAME}" ;;
        exited|created|dead)
            say "starting existing ollama container '${OLLAMA_NAME}' (state=${OLLAMA_STATE})"
            docker start "${OLLAMA_NAME}" >/dev/null || die "could not start ${OLLAMA_NAME}" ;;
        restarting)
            say "ollama container '${OLLAMA_NAME}' is restarting; waiting for it below" ;;
        "")
            # NO --gpus: CPU-only BY DESIGN. The card belongs to YOLO-World, and
            # a second CUDA context on this 8 GB card has hard-locked the host. A
            # 3B-parameter model answers in a few seconds on 32 CPU threads,
            # which is fast enough for a classifier that runs per room, not per
            # frame. The named volume is what keeps the pulled model between
            # runs, so it must not change.
            say "starting ollama (CPU-only, port 127.0.0.1:11434)..."
            docker run -d --name "${OLLAMA_NAME}" \
                -p 127.0.0.1:11434:11434 \
                -v "${OLLAMA_NAME}:/root/.ollama" \
                ollama/ollama:latest >/dev/null \
                || die "docker run failed for ${OLLAMA_NAME}" ;;
        *)
            say "ollama container '${OLLAMA_NAME}' is in state '${OLLAMA_STATE}'; trying start"
            docker start "${OLLAMA_NAME}" >/dev/null 2>&1 || true ;;
    esac

    say "waiting for ollama to answer..."
    OLLAMA_UP=0
    OLLAMA_TAGS=""
    for _ in $(seq 1 30); do
        OLLAMA_TAGS="$(curl -sf -m 3 "http://127.0.0.1:11434/api/tags" 2>/dev/null || true)"
        [[ -n "${OLLAMA_TAGS}" ]] && { OLLAMA_UP=1; break; }
        sleep 2
    done
    [[ "${OLLAMA_UP}" == "1" ]] \
        || die "ollama never answered on 11434 after 60 s. docker logs ${OLLAMA_NAME}"

    # Skip the pull when the model is already in the volume: `ollama pull` on a
    # present model is idempotent but still contacts the registry, so it turns a
    # working offline box into a failed bring-up for no reason.
    if grep -qF "\"${LLM_MODEL}\"" <<<"${OLLAMA_TAGS}"; then
        say "${LLM_MODEL} is already pulled; skipping the download"
    else
        say "pulling ${LLM_MODEL} (first pull downloads ~2 GB)..."
        docker exec "${OLLAMA_NAME}" ollama pull "${LLM_MODEL}" \
            || die "ollama pull ${LLM_MODEL} failed. docker logs ${OLLAMA_NAME}"
    fi

    run_llm_check
fi

# ── ROS2 env for polling and for every host node ──────────────────────────
# `set -u` and ROS 2's setup.bash are incompatible (ament's hooks read unset
# variables and -u EXITS the shell), so take -u off for the source only.
[[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]] || die "no /opt/ros/${ROS_DISTRO}/setup.bash"
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u
command -v ros2 >/dev/null || die "ros2 not on PATH after sourcing /opt/ros/${ROS_DISTRO}/setup.bash"
# Re-prepended, not set: setup.bash puts ROS's own site-packages at the FRONT of
# PYTHONPATH, and the repo has to win for `sparx_agency` to resolve out of the
# working tree. The duplicate entry left behind is harmless.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── (d) world ─────────────────────────────────────────────────────────────
sim_up() { docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' 2>/dev/null | grep -q .; }
WORLD_PID=""
if [[ "${NO_SIM}" == "1" ]]; then
    # --no-sim is an assertion, so check it rather than warning and then
    # polling an empty domain for five minutes to say the same thing.
    sim_up || die "--no-sim, but no sjtu_drone_* container is running. Either drop
  --no-sim, or bring the world up first:
    bash ${SETUP_DIR}/bringup_world.sh --skip-build --domain ${ROS_DOMAIN_ID} --rmw ${SIM_RMW} ${WORLD}"
elif sim_up; then
    say "SJTU sim already running: $(docker ps --filter 'name=^/sjtu_drone_' --format '{{.Names}}' | head -n1)"
else
    say "bringing up Gazebo world '${WORLD}' on the CPU (log: ${OUT_DIR}/gazebo.log)..."
    # bringup_world.sh's own flags, verified against its parser: --skip-build,
    # --domain <N>, --rmw <cyclonedds|fastrtps>, and a BARE world name (a
    # trailing .world is stripped for you). Same invocation as run_sjtu_n1.sh.
    nohup bash "${SETUP_DIR}/bringup_world.sh" \
        --skip-build --domain "${ROS_DOMAIN_ID}" --rmw "${SIM_RMW}" "${WORLD}" \
        > "${OUT_DIR}/gazebo.log" 2>&1 &
    WORLD_PID=$!
    echo "${WORLD_PID}" > "${OUT_DIR}/pids/world_bringup.pid"
fi
say "waiting for the world to publish /simple_drone/odom (this proves domain,"
say "  middleware AND the spawned drone in one shot)..."
WORLD_OK=0
WORLD_TIMEOUT_S=300     # the hospital loads in ~100 s on this box, cold
WORLD_DEADLINE=$((SECONDS + WORLD_TIMEOUT_S))
while (( SECONDS < WORLD_DEADLINE )); do
    if timeout 8 ros2 topic echo --once /simple_drone/odom >/dev/null 2>&1; then
        WORLD_OK=1; break
    fi
    # bringup_world.sh refuses loudly and exits for a missing image, a missing
    # world file, no display, or --skip-build against an unbuilt workspace.
    # Noticing that takes two seconds; waiting out the whole ${WORLD_TIMEOUT_S} s
    # to say "no odometry" names the wrong problem five minutes later.
    if [[ -n "${WORLD_PID}" ]] && ! kill -0 "${WORLD_PID}" 2>/dev/null; then
        echo "---- tail ${OUT_DIR}/gazebo.log ----" >&2
        tail -n 20 "${OUT_DIR}/gazebo.log" >&2 || true
        die "bringup_world.sh exited before the drone published odom (see above)."
    fi
    sleep 2
done
if [[ "${WORLD_OK}" != "1" ]]; then
    echo "---- tail ${OUT_DIR}/gazebo.log ----" >&2
    tail -n 20 "${OUT_DIR}/gazebo.log" >&2 || true
    die "no odometry from the drone after ${WORLD_TIMEOUT_S} s. If the container is up,
  this is almost always domain/middleware: the sim must be on
  ROS_DOMAIN_ID=${ROS_DOMAIN_ID} with ${RMW_IMPLEMENTATION}. See ${OUT_DIR}/gazebo.log"
fi
say "world is up and publishing odom"

# ── (e) detection server: YOLO-World on the GPU (conda navdp) ─────────────
if detect_healthy; then
    say "reusing the healthy detection server on :${DETECT_PORT}"
else
    [[ -f "${DETECT_MODEL}" ]] || die "YOLO-World checkpoint missing: ${DETECT_MODEL}
  (gitignored, per-device — download yolov8s-worldv2.pt to the repo root, or
  set DETECT_MODEL=/path/to/it)"
    say "starting the detection server on :${DETECT_PORT} (log: ${OUT_DIR}/detection_server.log)..."
    # cd "$HOME", NEVER the repo: ultralytics silently downloads checkpoints
    # into the CWD on first use, and the repo must not collect stray weights.
    # OUT_DIR, DETECT_MODEL and PYTHONPATH are all absolute, so the cd is safe.
    (
        cd "${HOME}" || exit 1
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${REPO_ROOT}" \
            nohup "${NAVDP_PY}" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
            --model "${DETECT_MODEL}" --device cuda:0 --port "${DETECT_PORT}" \
            > "${OUT_DIR}/detection_server.log" 2>&1 &
        echo $! > "${OUT_DIR}/pids/detection_server.pid"
    )
    DETECT_PID="$(cat "${OUT_DIR}/pids/detection_server.pid" 2>/dev/null || true)"
    say "waiting for the model to warm-load..."
    DETECT_OK=0
    DETECT_TIMEOUT_S=240    # torch import + checkpoint warm-load, cold, is ~60 s
    DETECT_DEADLINE=$((SECONDS + DETECT_TIMEOUT_S))
    while (( SECONDS < DETECT_DEADLINE )); do
        detect_healthy && { DETECT_OK=1; break; }
        # It aborts at startup by design on a missing torch, a bad checkpoint or
        # a busy card; report that immediately instead of after three minutes.
        if [[ -n "${DETECT_PID}" ]] && ! kill -0 "${DETECT_PID}" 2>/dev/null; then
            echo "---- tail ${OUT_DIR}/detection_server.log ----" >&2
            tail -n 20 "${OUT_DIR}/detection_server.log" >&2 || true
            die "the detection server exited during startup (see above)."
        fi
        sleep 2
    done
    [[ "${DETECT_OK}" == "1" ]] || die "detection server never became healthy on
  :${DETECT_PORT} after ${DETECT_TIMEOUT_S} s. See ${OUT_DIR}/detection_server.log
  (torch import? CUDA busy? bad checkpoint?)"
    say "detection server is up"
fi

# The open vocabulary is what the detector can see AT ALL. A target outside it
# is a run that cannot succeed however long it flies, and nothing downstream
# says so -- target_watcher just never matches.
DETECT_HEALTH="$(curl -sf -m 5 "http://127.0.0.1:${DETECT_PORT}/health" 2>/dev/null || true)"
if [[ -n "${DETECT_HEALTH}" ]] && ! grep -qF "\"${TARGET}\"" <<<"${DETECT_HEALTH}"; then
    say "WARNING: '${TARGET}' is not a class in the detector's live vocabulary."
    say "  target_watcher matches semantically, so a synonym of a listed class is"
    say "  fine — but a genuinely new object is never detected. Add it with:"
    say "    curl -sX POST http://127.0.0.1:${DETECT_PORT}/set_classes \\"
    say "      -d '{\"classes\": [\"${TARGET}\", ...the rest...]}'"
fi

# ── (f) FALCON exploration + ROS1 bridge (+ BEV) ──────────────────────────
if [[ "${NO_FALCON}" == "1" ]]; then
    say "--no-falcon: skipping FALCON. Without /falcon/bev_2d there are NO rooms"
    say "  — the semantic mapper idles and the oracle has nothing to rank."
else
    # The world name and the FALCON map config are two namespaces, and the
    # warehouse is the pair that proves it: the world file is
    # no_roof_small_warehouse.world, the map config is config/warehouse.yaml.
    # (falcon_sjtu/config/ holds hospital, warehouse and small_house.)
    case "${WORLD}" in
        hospital)                MAP_NAME="hospital" ;;
        no_roof_small_warehouse) MAP_NAME="warehouse" ;;
        small_house)             MAP_NAME="small_house" ;;
        *)  MAP_NAME="${WORLD}"
            say "WARNING: no known world->map mapping for '${WORLD}'; assuming the"
            say "  map config is called '${MAP_NAME}' too. Checked next." ;;
    esac
    FALCON_SH="${REPO_ROOT}/sparx_agency/tasks/planning/falcon_sjtu/run_falcon_sjtu.sh"
    [[ -f "${FALCON_SH}" ]] || die "missing ${FALCON_SH}"
    [[ -f "${REPO_ROOT}/sparx_agency/tasks/planning/falcon_sjtu/config/${MAP_NAME}.yaml" ]] \
        || die "no FALCON map config for '${MAP_NAME}': expected
  sparx_agency/tasks/planning/falcon_sjtu/config/${MAP_NAME}.yaml
  (available: $(cd "${REPO_ROOT}/sparx_agency/tasks/planning/falcon_sjtu/config" \
        && ls *.yaml 2>/dev/null | grep -v '^bridge' | sed 's/\.yaml//' | tr '\n' ' '))"
    if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then RVIZ_DEFAULT=1; else RVIZ_DEFAULT=0; fi
    say "starting FALCON (map=${MAP_NAME}, enable_bev:=true, detached)..."
    # FOLLOW=0 makes it return with the containers up; RVIZ and
    # FALCON_LAUNCH_ARGS are read the same way (all three are `${VAR:-default}`
    # reads in that script). enable_bev is a real arg of
    # adapter/launch/exploration.launch and defaults to FALSE, so passing it is
    # what puts /falcon/bev_2d on the wire at all.
    FOLLOW=0 RVIZ="${RVIZ:-${RVIZ_DEFAULT}}" \
        FALCON_LAUNCH_ARGS="enable_bev:=true ${FALCON_CONFINE_ARG} ${FALCON_LAUNCH_ARGS:-}" \
        bash "${FALCON_SH}" "${MAP_NAME}" \
        2>&1 | tee "${OUT_DIR}/falcon_bringup.log"
    [[ "${PIPESTATUS[0]}" == "0" ]] || die "run_falcon_sjtu.sh failed. See ${OUT_DIR}/falcon_bringup.log"
    docker ps --format '{{.Names}}' | grep -qx falcon-sjtu \
        || die "falcon-sjtu container is not running. docker logs falcon-sjtu"
    # run_falcon_sjtu.sh checks the bridge itself but exits 0 either way; a
    # bridge that bridged nothing means zero BEV for the whole run, so re-check.
    grep -q 'bridge up:' "${OUT_DIR}/falcon_bringup.log" \
        || say "WARNING: the ROS1 bridge reported NO topic bridges — /falcon/bev_2d
  will stay silent. docker logs sjtu-ros1-bridge"
fi

# ── (g) the host nodes (CPU, .venv, domain ${ROS_DOMAIN_ID}) ──────────────
export CUDA_VISIBLE_DEVICES=""   # every host node: CPU only, the card is YOLO's
start_node() {
    # $1 = module basename under ...scene_graph.ros2, rest = extra -p args.
    local module="$1"; shift
    nohup "${VENV_PY}" -m "sparx_agency.tasks.mapping.scene_graph.ros2.${module}" \
        --ros-args -p use_sim_time:=true "$@" \
        > "${OUT_DIR}/${module}.log" 2>&1 &
    echo $! > "${OUT_DIR}/pids/${module}.pid"
    say "  ${module}  pid $(cat "${OUT_DIR}/pids/${module}.pid")"
}
say "starting host nodes (use_sim_time:=true, CUDA_VISIBLE_DEVICES='')..."
# server_url is forwarded rather than left at its default: the node's default is
# the literal :8092, so a DETECT_PORT override moved the server and left the
# client posting into a closed port for the whole run.
start_node detector_client_node -p "server_url:=http://127.0.0.1:${DETECT_PORT}"
start_node object_mapper_node
start_node semantic_mapper_node
start_node room_classifier_node
start_node llm_oracle_node    -p "target_object:=${TARGET}"
start_node target_watcher_node -p "target_object:=${TARGET}"
start_node scene_graph_viz_node -p "out_dir:=${OUT_DIR}/viz" \
                                -p "show_window:=${VIZ_WINDOW:-true}"
# The search loop. With fly:=false (the default) it plans and publishes the
# chosen room and the route to it without touching the aircraft, so FALCON keeps
# exploring and the ranking becomes visible instead of merely logged.
if [[ "${NO_SEARCH}" == "0" ]]; then
    start_node room_search_node -p "fly:=$([[ "${SEARCH_FLY}" == "1" ]] && echo true || echo false)"
fi

# The object-search loop, and the two processes it needs to fly.
#
# ORDER MATTERS. The arbiter goes up FIRST so that no raw Twist is ever
# forwarded by a gate that has not yet decided it is closed, then the
# follower, then the supervisor that drives both. The follower lives in a
# DIFFERENT package, so start_node (which only knows scene_graph.ros2
# modules) cannot launch it -- but its pid file still has to match
# stop_scene_graph.sh's *_node.pid glob or a stopped stack leaves an
# unattended 20 Hz writer on cmd_vel_raw.
if [[ "${OBJECT_SEARCH}" == "1" ]]; then
    start_node cmd_vel_arbiter_node
    if [[ "${SEARCH_FLY}" == "1" ]]; then
        FOLLOWER_CFG="${SCRIPT_DIR}/../config/room_search_follower.yaml"
        nohup "${VENV_PY}" -m sparx_agency.tasks.planning.sjtu_internvla_n1.ros2.trajectory_follower_node \
            --ros-args -p use_sim_time:=true -p "config_file:=${FOLLOWER_CFG}" \
            > "${OUT_DIR}/trajectory_follower_node.log" 2>&1 &
        echo $! > "${OUT_DIR}/pids/trajectory_follower_node.pid"
        say "  trajectory_follower_node  pid $(cat "${OUT_DIR}/pids/trajectory_follower_node.pid")"
    fi
    start_node object_search_node \
        -p "fly:=$([[ "${SEARCH_FLY}" == "1" ]] && echo true || echo false)" \
        -p "search_backend:=${SEARCH_BACKEND:-falcon}" \
        -p "search_timeout_s:=${ROOM_BUDGET_S:-90.0}"
fi

# The last leg: once target_watcher_node latches /target_seen, this is what
# flies onto the object and lands beside it. Until then it is inert -- no
# publication, no camera subscription, no control timer -- so today's
# exploration flight is unchanged by its presence. It takes the aircraft by
# muting the FALCON b-spline follower over the latched
# /scene_graph/external_ctrl (see tasks/planning/falcon_sjtu/README.md), and
# server_url is forwarded for the same reason detector_client_node's is: the
# node's default is the literal :8092, so a DETECT_PORT override would leave
# the approach posting into a closed port and never re-confirming the target.
if [[ "${NO_APPROACH}" == "0" ]]; then
    start_node target_approach_node -p "server_url:=http://127.0.0.1:${DETECT_PORT}"
fi

# The scene-graph RViz view. Started AFTER the nodes so their latched markers
# are already up: every scene-graph topic is TRANSIENT_LOCAL, so a late-joining
# rviz2 gets the last picture immediately instead of an empty screen until the
# next publish.
if [[ "${RVIZ_SCENE}" == "1" ]]; then
    RVIZ_CFG="${SCRIPT_DIR}/../config/scene_graph.rviz"
    if [[ ! -f "${RVIZ_CFG}" ]]; then
        say "WARNING: ${RVIZ_CFG} is missing; skipping --rviz."
    elif [[ -z "${DISPLAY:-}" ]]; then
        say "WARNING: --rviz needs \$DISPLAY and it is unset; skipping."
    else
        nohup rviz2 -d "${RVIZ_CFG}" > "${OUT_DIR}/rviz_scene.log" 2>&1 &
        echo $! > "${OUT_DIR}/pids/rviz_scene.pid"
        say "  rviz2 (scene graph)  pid $(cat "${OUT_DIR}/pids/rviz_scene.pid")"
    fi
fi

sleep 5
DEAD=""
for pf in "${OUT_DIR}"/pids/*_node.pid; do
    [[ -e "${pf}" ]] || continue
    pid="$(cat "${pf}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null \
        || DEAD="${DEAD} $(basename "${pf}" .pid)"
done
if [[ -n "${DEAD// /}" ]]; then
    for n in ${DEAD}; do
        echo "---- tail ${OUT_DIR}/${n}.log ----" >&2
        tail -n 15 "${OUT_DIR}/${n}.log" >&2 || true
    done
    die "these nodes died at startup:${DEAD}. Logs above / under ${OUT_DIR}."
fi

# ── (h) status ────────────────────────────────────────────────────────────
say "==================== scene-graph mission UP ===================="
say "world     ${WORLD}   target '${TARGET}'   map ${MAP_NAME:-(no falcon)}"
say "out dir   ${OUT_DIR}"
say "viz       ${OUT_DIR}/viz"
say "pids:"
for pf in "${OUT_DIR}"/pids/*.pid; do
    [[ -e "${pf}" ]] || continue
    printf '[scene-graph]   %-24s %s\n' "$(basename "${pf}" .pid)" "$(cat "${pf}")"
done
say "logs      ${OUT_DIR}/*.log"
say "health:"
say "  detection  curl http://127.0.0.1:${DETECT_PORT}/health"
say "  ollama     curl http://127.0.0.1:11434/api/tags"
say "  ollama ps  curl http://127.0.0.1:11434/api/ps   (which model is loaded)"
say "watch:"
say "  ros2 topic echo --once /scene_graph"
say "  ros2 topic echo /target_seen"
say "stop      bash ${SCRIPT_DIR}/stop_scene_graph.sh --out ${OUT_DIR} [--all]"
say "================================================================"
if [[ "${ATTACH}" == "1" ]]; then
    say "attaching to the viz log (Ctrl-C detaches; the stack stays up)"
    exec tail -f "${OUT_DIR}/scene_graph_viz_node.log"
fi
exit 0
