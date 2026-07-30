#!/usr/bin/env bash
# ============================================================
# run_object_mission_sphera.sh -- ROBOTICAN/Sphera-only fork of
# run_object_mission.sh. Does not touch run_object_mission.sh (XTEND's path);
# same relationship as run_falcon_sphera.sh to run_falcon.sh. Differences:
#   - CONFIG_FILE defaults to config/mission_sphera.yaml, not mission.yaml
#   - ENV_NAME defaults to sphera_jail, not office
#   - calls run_falcon_sphera.sh (not run_falcon.sh) and roslaunches
#     object_mission_sphera.launch (not object_mission.launch)
# See docs/progress/entries/005-yolo-object-navigation.md.
#
# run_object_mission.sh -- one command for the "pick an object, then fly to it
# and land" mission (the select-then-go flavour of the object-approach stack).
#
# Same three processes as run_object_approach_mission.sh, but the TARGET and GOAL
# are NOT given on the command line: the mission_director (launched inside the
# container) selects the object -- randomly, or from a click in its object-list
# window -- and only then arms the stack (publishes the YOLO prompt, the coordinate
# goal, and the enable). Until then nothing plans or flies.
#
#   1. the TensorRT YOLO-World detector, a ROS2 sidecar ON THE HOST (the FALCON
#      container has no CUDA/TensorRT/pycuda). It starts with a placeholder prompt
#      and is RE-PROMPTED by the director when the object is selected.
#   2. the ros1<->ros2 bridge (carries /object_approach/{detections,goal}).
#   3. FALCON nav + A*/NavDP + object-approach (DISABLED) + the mission director,
#      inside the ROS1 container via object_mission.launch. Runs in the foreground.
#
# Usage:
#   ./run_object_mission.sh [--falcon-only|--detector-only] [env] [SELECTION_MODE]
#     [env]            maps/<env>.yaml (e.g. office, hospital)  (default office).
#     SELECTION_MODE   gui (default: window with the object list) | random.
#
# RESTARTING JUST FALCON. The detector sidecar is the slow part of a start (it loads
# multi-hundred-MB TensorRT engines onto the GPU); FALCON is the part you actually
# iterate on. Split them across two terminals and you can relaunch FALCON as often as
# you like without paying for the engines again:
#
#   terminal A:  ./run_object_mission.sh --detector-only     # start it once, leave it
#   terminal B:  ./run_object_mission.sh --falcon-only       # relaunch as often as you like
#
#   --detector-only  Run ONLY the YOLO-World detector sidecar, in the foreground, and
#                    hold it until Ctrl+C. Nothing plans, nothing flies.
#   --falcon-only    Run ONLY the mission: the bridge + the FALCON container. Reuses
#                    the detector sidecar that is ALREADY running and never touches it
#                    (neither starting nor, on exit, killing it) -- so no engines are
#                    reloaded. Refuses to start if no sidecar is running, rather than
#                    flying a mission whose detector will never publish a detection.
#
#                    The BRIDGE is restarted along with FALCON, and cannot sensibly be
#                    kept: it is a ROS1 node against the roscore that roslaunch starts
#                    INSIDE the FALCON container, so that master dies with the container
#                    and takes the bridge's topic registrations with it. Restarting it
#                    costs a couple of seconds; the engines are what you are saving.
#                    (A fresh roscore per run is wanted anyway -- it is what stops a
#                    stale latched /waypoint_nav/goal from pre-arming the planners.)
#
#   EVERY parameter below has its default in ONE file: config/mission.yaml. Edit that
#   to change what a plain `./run_object_mission.sh` does. The env vars and the command
#   line still override it per run, so the precedence is:
#
#       command line  >  environment variable  >  config/mission.yaml  >  built-in
#
#   Env overrides:
#     CONFIG_FILE  path to the YAML holding all of the below (default config/mission.yaml).
#                  An unknown key in it is a hard error, never a silent no-op.
#     NAV_MODE     fallback | hybrid | astar | combination | navdp
#                  (default fallback: A*, with NavDP rescuing only a boxed-in A* --
#                  same default as real_drone.launch. Needs the NavDP server up.)
#     SEED         >=0 for a reproducible random pick       (default -1).
#     OBJECTS_DIR  HOST dir holding the room map's objects.json. run_falcon.sh
#                  bind-mounts it into the container at the SAME path, so host and
#                  container agree. Default: the room mapper's latest_room_map.
#     OBJECTS_FILE Path to the objects JSON (default: $OBJECTS_DIR/objects.json).
#                  Any path works as long as the container can see it -- i.e. it is
#                  under OBJECTS_DIR, or under the repo mounted at /opt/sparx_agency
#                  (e.g. the shipped objects.json).
#     MODEL        s | m | l | x  detector size             (default x).
#     ENGINES_DIR  host dir with the *.engine files         (default engines/<tag>).
#     WEIGHTS_DIR  host dir holding yolov8<MODEL>-worldv2.pt (default ~user/Downloads).
#     TEXT_WEIGHTS full host path to the .pt (default $WEIGHTS_DIR/yolov8<MODEL>-worldv2.pt).
#     INIT_TARGET  placeholder detector prompt before selection (default refrigerator).
#     Any object_mission.launch arg can be appended after SELECTION_MODE,
#     e.g. land_range_m:=0 arrive_radius_m:=0.8 viewer:=false
#
# Examples:
#   ./run_object_mission.sh                      # office, gui select, fallback nav
#   ./run_object_mission.sh hospital             # another map, still gui select
#   ./run_object_mission.sh office random        # random pick
#   NAV_MODE=hybrid ./run_object_mission.sh office gui land_range_m:=0
#   ./run_object_mission.sh --detector-only      # terminal A: engines up, stay up
#   ./run_object_mission.sh --falcon-only office # terminal B: relaunch the mission
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# falcon -> planning -> tasks -> sparx_agency -> repo root (the dir with sparx_agency/)
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"

# ── Config file: every parameter in one place ────────────────
# config/mission.yaml holds the defaults; each is still overridable per run by an
# env var or the command line (see the precedence note in that file). The reader
# validates it and prints MISSION_CFG_* assignments for us to eval.
# Capture FIRST, eval second: `eval "$(cmd)"` reports eval's own status, so a reader
# that exited nonzero would be swallowed and the mission would fly on with the bad
# parameter silently ignored -- exactly what validating it was meant to prevent.
PYTHON="${PYTHON:-python3}"
CONFIG_FILE="${CONFIG_FILE:-$HERE/config/mission_sphera.yaml}"
MISSION_CFG_LAUNCH_ARGS=()
if [[ -f "$CONFIG_FILE" ]]; then
  if ! _cfg_shell="$("$PYTHON" "$HERE/config/mission_config.py" "$CONFIG_FILE" \
                       --launch-file "$HERE/adapter/launch/object_mission_sphera.launch")"; then
    echo "[ERROR] invalid config file: $CONFIG_FILE  (see the error above)" >&2
    exit 1
  fi
  eval "$_cfg_shell"
elif [[ -n "${CONFIG_FILE:-}" && "$CONFIG_FILE" != "$HERE/config/mission_sphera.yaml" ]]; then
  # An explicitly-requested config that does not exist is a mistake, not a default.
  echo "[ERROR] config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# ── Arguments ────────────────────────────────────────────────
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0
fi

# Which of the three processes this invocation owns (see the header). Parsed before
# the positionals so the flag can lead: ./run_object_mission.sh --falcon-only office
RUN_MODE=all                       # all | falcon | detector
while [[ $# -ge 1 ]]; do
  case "$1" in
    --falcon-only|-f)   RUN_MODE=falcon;   shift ;;
    --detector-only|-d) RUN_MODE=detector; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
WANT_DETECTOR=0; WANT_BRIDGE=0; WANT_FALCON=0
case "$RUN_MODE" in
  all)      WANT_DETECTOR=1; WANT_BRIDGE=1; WANT_FALCON=1 ;;
  falcon)   WANT_BRIDGE=1;   WANT_FALCON=1 ;;   # bridge dies with FALCON's roscore
  detector) WANT_DETECTOR=1 ;;
esac
# The map is optional: office is the default mission environment.
ENV_NAME="${1:-${MISSION_CFG_MAP:-sphera_jail}}"; [[ $# -ge 1 ]] && shift || true
SELECTION_MODE="${1:-${MISSION_CFG_SELECTION_MODE:-gui}}"; [[ $# -ge 1 ]] && shift || true
EXTRA_ARGS=("$@")                          # any further launch args, verbatim

NAV_MODE="${NAV_MODE:-${MISSION_CFG_NAV_MODE:-fallback}}"
SEED="${SEED:-${MISSION_CFG_SEED:--1}}"

# ── Object catalog: the room map's objects.json by default ───
# The room mapper writes the catalog on the host; run_falcon.sh mounts OBJECTS_DIR
# read-only at the SAME path inside the container, so this one value resolves on
# both sides. Exported so run_falcon.sh mounts whatever we resolved here.
OBJECTS_DIR="${OBJECTS_DIR:-${MISSION_CFG_OBJECTS_DIR:-/home/user/jetson-containers/data/captures/latest_room_map}}"
OBJECTS_FILE="${OBJECTS_FILE:-${MISSION_CFG_OBJECTS_FILE:-$OBJECTS_DIR/objects.json}}"
export OBJECTS_DIR
INIT_TARGET="${INIT_TARGET:-${MISSION_CFG_INIT_TARGET:-refrigerator}}"  # re-prompted on selection
MODEL="${MODEL:-${MISSION_CFG_MODEL:-x}}"
CONF_THRESH="${CONF_THRESH:-${MISSION_CFG_CONF_THRESH:-0.1}}"
# The detector sidecar needs the tensorrt+pycuda(+torch/ultralytics) env. Run this
# script from that activated venv, or point PYTHON at its interpreter.

# ── Locate the GPU engines + text weights (fail loudly if missing) ──
TARGET_TAG="$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
  "from sparx_agency.tasks.mapping.yolo_world_trt.hardware import detect; print(detect().target_tag)" \
  2>/dev/null || echo orin_sm87)"
ENGINES_DIR="${ENGINES_DIR:-${MISSION_CFG_ENGINES_DIR:-$HERE/../../mapping/yolo_world_trt/engines/$TARGET_TAG}}"
BACKBONE="$ENGINES_DIR/yolo_world_${MODEL}.backbone.fp16.gpu.engine"
HEAD="$ENGINES_DIR/yolo_world_${MODEL}.head.fp16.gpu.engine"
# The .pt is a multi-hundred-MB download kept outside the repo, so default to the
# dir it is downloaded into rather than a repo-relative path that is never populated.
WEIGHTS_DIR="${WEIGHTS_DIR:-${MISSION_CFG_WEIGHTS_DIR:-/home/user/Downloads}}"
TEXT_WEIGHTS="${TEXT_WEIGHTS:-${MISSION_CFG_TEXT_WEIGHTS:-$WEIGHTS_DIR/yolov8${MODEL}-worldv2.pt}}"

# Only when we are the ones starting the detector: under --falcon-only the engines
# are already loaded into a running sidecar, and blocking a FALCON relaunch because a
# host path moved since would be a check working against its own purpose.
if [[ $WANT_DETECTOR -eq 1 ]]; then
  for f in "$BACKBONE" "$HEAD"; do
    if [[ ! -f "$f" ]]; then
      echo "[ERROR] engine not found: $f" >&2
      echo "        Build the GPU split first, e.g.:" >&2
      echo "          $HERE/../../mapping/yolo_world_trt/build_all.sh $MODEL" >&2
      echo "        or point ENGINES_DIR at the folder holding the *.engine files." >&2
      exit 1
    fi
  done
  if [[ ! -f "$TEXT_WEIGHTS" ]]; then
    echo "[ERROR] text weights (.pt) not found: $TEXT_WEIGHTS" >&2
    echo "        Set TEXT_WEIGHTS=/path/to/yolov8${MODEL}-worldv2.pt (host path)." >&2
    exit 1
  fi
fi

# ── --falcon-only: the sidecar we are about to rely on must exist ──
# A mission whose detector is not running looks completely healthy -- it plans, it
# flies, it just never confirms an object and lands "by A* alone" every time. That is
# far worse than refusing to start, so check for the process rather than assume it.
# Match an INTERPRETER running the node, not the bare filename: a bare-filename
# pattern is also matched by anything that merely mentions it -- an editor, a
# `tail -f` on the log, or the very shell command you typed to check -- and a guard
# satisfied by an open editor is worse than no guard.
# Sphera-only difference: the sidecar runs inside detector_dev (see below),
# not on the bare host -- so this check is a `docker exec` pgrep, not a
# host-level one.
DETECTOR_PATTERN="python.*yolo_detector_ros2_node\.py"
if [[ $WANT_FALCON -eq 1 && $WANT_DETECTOR -eq 0 ]]; then
  if ! docker exec detector_dev pgrep -f "$DETECTOR_PATTERN" >/dev/null 2>&1; then
    echo "[ERROR] --falcon-only: no detector sidecar is running (in detector_dev)." >&2
    echo "        Nothing would ever publish a detection, so the mission could only" >&2
    echo "        ever land by A* alone -- while looking perfectly healthy." >&2
    echo "        Start one first, in another terminal:" >&2
    echo "          $0 --detector-only" >&2
    echo "        or drop the flag to run the whole stack: $0" >&2
    exit 1
  fi
  echo "[mission] reusing the detector sidecar already running (pid $(docker exec detector_dev pgrep -f "$DETECTOR_PATTERN" | head -1))"
fi
# Fail here rather than inside the container: a missing catalog is a stale/absent
# room map, and the next-best candidate (the shipped objects.json) holds a DIFFERENT
# room's coordinates -- silently falling back to it would fly the drone to the wrong
# place. Check on the host; a container-only path under /opt/sparx_agency maps back
# to the repo, so translate it before testing.
CATALOG_ON_HOST="$OBJECTS_FILE"
if [[ "$OBJECTS_FILE" == /opt/sparx_agency/* ]]; then
  CATALOG_ON_HOST="$REPO_ROOT/sparx_agency/${OBJECTS_FILE#/opt/sparx_agency/}"
fi
if [[ $WANT_FALCON -eq 1 && ! -f "$CATALOG_ON_HOST" ]]; then
  echo "[ERROR] object catalog not found: $OBJECTS_FILE" >&2
  [[ "$CATALOG_ON_HOST" != "$OBJECTS_FILE" ]] && \
    echo "        (resolved on the host to: $CATALOG_ON_HOST)" >&2
  echo "        The room mapper writes it to \$OBJECTS_DIR/objects.json." >&2
  echo "        Set OBJECTS_DIR=<host dir with objects.json>, or OBJECTS_FILE=<path>," >&2
  echo "        e.g. OBJECTS_FILE=/opt/sparx_agency/tasks/planning/falcon/objects.json" >&2
  echo "        to use the catalog shipped with the repo." >&2
  exit 1
fi

if [[ $WANT_FALCON -eq 1 && ! -f "$HERE/maps/${ENV_NAME}.yaml" ]]; then
  echo "[ERROR] map not found: $HERE/maps/${ENV_NAME}.yaml" >&2
  echo "        Available: $(ls "$HERE/maps"/*.yaml 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

LOG_DIR="/tmp/object_mission"
mkdir -p "$LOG_DIR"

# ── Resolve the launch args before reporting them ────────────
# The values we resolved above, then the config's launch: section for whatever they
# and the command line did NOT set -- de-duplicated HERE rather than handing roslaunch
# the same arg twice and trusting its precedence. Done before the banner so what we
# print is what actually runs.
LAUNCH_ARGS=(
  map_name:="$ENV_NAME"
  selection_mode:="$SELECTION_MODE"
  seed:="$SEED"
  nav_mode:="$NAV_MODE"
  objects_file:="$OBJECTS_FILE"
  # The SAME placeholder the detector sidecar gets (-p target_object above).
  # There are two pre-selection targets -- what YOLO is prompted with, and what
  # object_approach filters detections for -- and they must be the same word. Left
  # unset, object_approach kept its own 'refrigerator' default while the detector
  # was prompted with init_target, so the detector published a label nothing was
  # listening for: no confirmation, no visual land, and the only clue was the two
  # heartbeats quietly disagreeing. The director overwrites both on selection.
  target_object:="$INIT_TARGET"
)
_arg_key() { printf '%s' "${1%%:=*}"; }
CFG_ARGS=()
for _cfg in ${MISSION_CFG_LAUNCH_ARGS[@]+"${MISSION_CFG_LAUNCH_ARGS[@]}"}; do
  _dup=0
  for _other in ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
    if [[ "$(_arg_key "$_other")" == "$(_arg_key "$_cfg")" ]]; then _dup=1; break; fi
  done
  [[ $_dup -eq 0 ]] && CFG_ARGS+=( "$_cfg" )
done

case "$RUN_MODE" in
  all)      echo "== Object mission (select -> fly -> land) ==" ;;
  falcon)   echo "== Object mission: FALCON ONLY (bridge + container; detector reused) ==" ;;
  detector) echo "== Object mission: DETECTOR ONLY (engines up; nothing plans or flies) ==" ;;
esac
if [[ $WANT_FALCON -eq 1 ]]; then
  echo "   env/map     : $ENV_NAME"
  echo "   selection   : $SELECTION_MODE   (seed: $SEED)"
  echo "   nav_mode    : $NAV_MODE"
  echo "   objects     : $OBJECTS_FILE"
fi
if [[ $WANT_DETECTOR -eq 1 ]]; then
  echo "   detector    : GPU $MODEL  ($ENGINES_DIR), conf>=$CONF_THRESH, initial prompt '$INIT_TARGET' (re-prompted on select)"
else
  echo "   detector    : REUSED -- the running sidecar keeps its engines; this run neither starts nor stops it"
fi
echo "   config      : ${CONFIG_FILE}$([[ -f "$CONFIG_FILE" ]] || echo '  (absent -- built-in defaults)')"
echo "   logs        : $LOG_DIR/{sidecar,bridge}.log"
if [[ $WANT_FALCON -eq 1 ]]; then
  [[ ${#CFG_ARGS[@]} -gt 0 ]] && echo "   from config : ${CFG_ARGS[*]}"
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "   overridden  : ${EXTRA_ARGS[*]}"
fi
echo

# ── Cleanup: stop ONLY what this invocation started ───────────
# Under --falcon-only the sidecar belongs to another terminal: killing it on exit
# would defeat the entire point of the flag (the next relaunch would reload the
# engines), so the guards below are load-bearing, not defensive.
DETECTOR_PROC_PATTERN="yolo_detector_ros2_node.py"
cleanup() {
  echo
  echo "[mission] shutting down what this run started ..."
  # docker exec pkill (not a bare `kill $SIDECAR_PID`): SIDECAR_PID is the
  # LOCAL `docker exec` client process. Killing only that does not reliably
  # stop the node actually running inside detector_dev (docker exec has no
  # ptrace/process-group tie to the client by default) -- confirmed against
  # the same pattern mission_control.py already uses for proc_container
  # services.
  if [[ $WANT_DETECTOR -eq 1 && -n "${SIDECAR_PID:-}" ]]; then
    kill "$SIDECAR_PID" 2>/dev/null || true
    docker exec detector_dev pkill -f "$DETECTOR_PROC_PATTERN" 2>/dev/null || true
  fi
  [[ $WANT_BRIDGE -eq 1 ]] && docker rm -f ros1_bridge 2>/dev/null || true
  [[ $WANT_FALCON -eq 1 ]] && docker rm -f falcon      2>/dev/null || true
  return 0
}
trap cleanup EXIT INT TERM

# ── ROS2 environment ─────────────────────────────────────────
# Sourced in EVERY mode, not just the one that starts the detector: the bridge picks
# up ROS_DOMAIN_ID / RMW_IMPLEMENTATION from this environment, so sourcing it only
# sometimes would make --falcon-only bridge on a different domain than a full run.
# ROS setup scripts reference unbound vars / return nonzero -- guard set -e/-u.
set +u +e; source /opt/ros/humble/setup.bash; set -u -e   # shellcheck disable=SC1091

# ── 1. Detector sidecar (container `detector_dev`, GPU) ───────
# Sphera-only difference from run_object_mission.sh: that one runs the
# sidecar on the bare host venv (torch/ultralytics installed there); this
# fork runs it inside the `detector` image instead (docker/Dockerfile.detector,
# started persistently via docker-compose.detector.yml as `detector_dev` --
# same long-running pattern as `robotican_dev`). --falcon-only's "is a
# sidecar already running" check below still greps by process pattern, now
# via `docker exec` instead of a bare host `pgrep`.
if [[ $WANT_DETECTOR -eq 1 ]]; then
  if ! docker ps --format '{{.Names}}' | grep -qx detector_dev; then
    echo "[ERROR] detector_dev container is not running." >&2
    echo "        Start it first:" >&2
    echo "          docker compose -f $REPO_ROOT/docker-compose.detector.yml run -d --rm --name detector_dev detector tail -f /dev/null" >&2
    exit 1
  fi
  echo "[mission] starting the YOLO-World detector sidecar (detector_dev container, GPU) ..."
  # NOT `docker exec -d` (detached): that returns immediately, so `$!` would
  # be the already-exited detach client, not something `kill -0`/`wait` can
  # track. Plain `docker exec` (no -d), backgrounded with shell `&` instead,
  # stays attached and streaming for as long as the remote process runs --
  # same shape the original host-PID tracking below expects. $LOG_DIR is
  # bind-mounted into detector_dev at the same path (docker-compose.detector.yml)
  # so this host-side script can still read/tail it directly.
  docker exec detector_dev bash -lc "
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI='file:///home/$USER/rqs_iai_ws/src/cyclonedds.xml'
    export PYTHONPATH=\"$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}\"
    python3 '$REPO_ROOT/sparx_agency/tasks/mapping/ros2/yolo_detector_ros2_node.py' \
      --ros-args \
        -p target_object:='$INIT_TARGET' \
        -p conf_thresh:='$CONF_THRESH' \
        -p backbone_engine:='$BACKBONE' \
        -p head_engine:='$HEAD' \
        -p text_weights:='$TEXT_WEIGHTS' \
        -p rgb_topic:=/R1/rgb_frame_path \
  " >"$LOG_DIR/sidecar.log" 2>&1 &
  SIDECAR_PID=$!
  sleep 3
  if ! docker exec detector_dev pgrep -f "$DETECTOR_PROC_PATTERN" >/dev/null 2>&1; then
    echo "[ERROR] detector sidecar died on startup -- see $LOG_DIR/sidecar.log" >&2
    tail -n 20 "$LOG_DIR/sidecar.log" >&2 || true
    exit 1
  fi
fi

# ── --detector-only: hold the engines and stop here ──────────
# Loading the engines is the expensive part of a start, so this mode exists purely to
# pay it ONCE and keep it paid while --falcon-only relaunches the mission next door.
if [[ $WANT_FALCON -eq 0 ]]; then
  echo
  echo "[mission] detector up (pid $SIDECAR_PID), log: $LOG_DIR/sidecar.log"
  echo "[mission] leave this terminal open; relaunch the mission next door with:"
  echo "            $0 --falcon-only ${ENV_NAME} ${SELECTION_MODE}"
  echo "[mission] Ctrl+C here stops the detector."
  # || true: a Ctrl+C makes wait report the signal, which set -e would turn into a
  # failed exit for what is the normal way to end this mode.
  wait "$SIDECAR_PID" || true
  exit 0
fi

# ── 2. ros1<->ros2 bridge (host) ─────────────────────────────
# Always restarted with FALCON: it is a ROS1 node against the roscore that roslaunch
# starts inside the FALCON container, so that master -- and every topic registration
# the bridge made against it -- dies when the container does.
echo "[mission] starting the ros1<->ros2 bridge ..."
"$HERE/bridge/run_bridge.sh" >"$LOG_DIR/bridge.log" 2>&1 &
sleep 3

# ── 3. FALCON nav + A*/NavDP + object-approach + director (container, foreground) ─
# Clear any container left behind by a killed run: run_falcon.sh names it, so a stale
# one makes docker refuse the name and the relaunch fails for a reason that has
# nothing to do with the mission.
docker rm -f falcon >/dev/null 2>&1 || true
echo "[mission] launching FALCON (nav + object-approach + mission director) ..."
# NOT exec: keep this shell alive so the EXIT/INT/TERM trap runs cleanup() (stop
# whatever this run started) when the launch ends -- on a node crash / normal
# teardown, not only on Ctrl+C. exec would discard the trap.
"$HERE/run_falcon_sphera.sh" "$ENV_NAME" \
  roslaunch falcon_adapter object_mission_sphera.launch \
    ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} \
    "${LAUNCH_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
