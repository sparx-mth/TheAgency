#!/usr/bin/env bash
# ============================================================
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
#   ./run_object_mission.sh [env] [SELECTION_MODE]
#     [env]            maps/<env>.yaml (e.g. office, hospital)  (default office).
#     SELECTION_MODE   gui (default: window with the object list) | random.
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
CONFIG_FILE="${CONFIG_FILE:-$HERE/config/mission.yaml}"
MISSION_CFG_LAUNCH_ARGS=()
if [[ -f "$CONFIG_FILE" ]]; then
  if ! _cfg_shell="$("$PYTHON" "$HERE/config/mission_config.py" "$CONFIG_FILE" \
                       --launch-file "$HERE/adapter/launch/object_mission.launch")"; then
    echo "[ERROR] invalid config file: $CONFIG_FILE  (see the error above)" >&2
    exit 1
  fi
  eval "$_cfg_shell"
elif [[ -n "${CONFIG_FILE:-}" && "$CONFIG_FILE" != "$HERE/config/mission.yaml" ]]; then
  # An explicitly-requested config that does not exist is a mistake, not a default.
  echo "[ERROR] config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# ── Arguments ────────────────────────────────────────────────
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0
fi
# The map is optional: office is the default mission environment.
ENV_NAME="${1:-${MISSION_CFG_MAP:-office}}"; [[ $# -ge 1 ]] && shift || true
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
# Fail here rather than inside the container: a missing catalog is a stale/absent
# room map, and the next-best candidate (the shipped objects.json) holds a DIFFERENT
# room's coordinates -- silently falling back to it would fly the drone to the wrong
# place. Check on the host; a container-only path under /opt/sparx_agency maps back
# to the repo, so translate it before testing.
CATALOG_ON_HOST="$OBJECTS_FILE"
if [[ "$OBJECTS_FILE" == /opt/sparx_agency/* ]]; then
  CATALOG_ON_HOST="$REPO_ROOT/sparx_agency/${OBJECTS_FILE#/opt/sparx_agency/}"
fi
if [[ ! -f "$CATALOG_ON_HOST" ]]; then
  echo "[ERROR] object catalog not found: $OBJECTS_FILE" >&2
  [[ "$CATALOG_ON_HOST" != "$OBJECTS_FILE" ]] && \
    echo "        (resolved on the host to: $CATALOG_ON_HOST)" >&2
  echo "        The room mapper writes it to \$OBJECTS_DIR/objects.json." >&2
  echo "        Set OBJECTS_DIR=<host dir with objects.json>, or OBJECTS_FILE=<path>," >&2
  echo "        e.g. OBJECTS_FILE=/opt/sparx_agency/tasks/planning/falcon/objects.json" >&2
  echo "        to use the catalog shipped with the repo." >&2
  exit 1
fi

if [[ ! -f "$HERE/maps/${ENV_NAME}.yaml" ]]; then
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

echo "== Object mission (select -> fly -> land) =="
echo "   env/map     : $ENV_NAME"
echo "   selection   : $SELECTION_MODE   (seed: $SEED)"
echo "   nav_mode    : $NAV_MODE"
echo "   objects     : $OBJECTS_FILE"
echo "   detector    : GPU $MODEL  ($ENGINES_DIR), conf>=$CONF_THRESH, initial prompt '$INIT_TARGET' (re-prompted on select)"
echo "   config      : ${CONFIG_FILE}$([[ -f "$CONFIG_FILE" ]] || echo '  (absent -- built-in defaults)')"
echo "   logs        : $LOG_DIR/{sidecar,bridge}.log"
[[ ${#CFG_ARGS[@]} -gt 0 ]] && echo "   from config : ${CFG_ARGS[*]}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "   overridden  : ${EXTRA_ARGS[*]}"
echo

# ── Cleanup: stop the two background host helpers on any exit ──
cleanup() {
  echo
  echo "[mission] shutting down the detector sidecar and bridge ..."
  [[ -n "${SIDECAR_PID:-}" ]] && kill "$SIDECAR_PID" 2>/dev/null || true
  docker rm -f ros1_bridge 2>/dev/null || true
  docker rm -f falcon      2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 1. Detector sidecar (host, ROS2, GPU) ────────────────────
echo "[mission] 1/3 starting the YOLO-World detector sidecar (host, GPU) ..."
# ROS setup scripts reference unbound vars / return nonzero -- guard set -e/-u.
set +u +e; source /opt/ros/humble/setup.bash; set -u -e   # shellcheck disable=SC1091
# PREPEND, never assign: the setup.bash above puts ROS's site-packages (rclpy et al)
# on PYTHONPATH, and a bare PYTHONPATH="$REPO_ROOT" prefix would drop it -- the node
# then dies on `import rclpy` even though the venv is fine.
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  "$REPO_ROOT/sparx_agency/tasks/mapping/ros2/yolo_detector_ros2_node.py" \
  --ros-args \
    -p target_object:="$INIT_TARGET" \
    -p conf_thresh:="$CONF_THRESH" \
    -p backbone_engine:="$BACKBONE" \
    -p head_engine:="$HEAD" \
    -p text_weights:="$TEXT_WEIGHTS" \
  >"$LOG_DIR/sidecar.log" 2>&1 &
SIDECAR_PID=$!
sleep 3
if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
  echo "[ERROR] detector sidecar died on startup -- see $LOG_DIR/sidecar.log" >&2
  tail -n 20 "$LOG_DIR/sidecar.log" >&2 || true
  exit 1
fi

# ── 2. ros1<->ros2 bridge (host) ─────────────────────────────
echo "[mission] 2/3 starting the ros1<->ros2 bridge ..."
"$HERE/bridge/run_bridge.sh" >"$LOG_DIR/bridge.log" 2>&1 &
sleep 3

# ── 3. FALCON nav + A*/NavDP + object-approach + director (container, foreground) ─
echo "[mission] 3/3 launching FALCON (nav + object-approach + mission director) ..."
# NOT exec: keep this shell alive so the EXIT/INT/TERM trap runs cleanup() (stop the
# host detector sidecar + remove the bridge/falcon containers) when the launch ends --
# on a node crash / normal teardown, not only on Ctrl+C. exec would discard the trap.
"$HERE/run_falcon.sh" "$ENV_NAME" \
  roslaunch falcon_adapter object_mission.launch \
    ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} \
    "${LAUNCH_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
