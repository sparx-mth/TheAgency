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
#   ./run_object_mission.sh <env> [SELECTION_MODE]
#     <env>            maps/<env>.yaml (e.g. office, hospital) -- REQUIRED.
#     SELECTION_MODE   gui (default: window with the object list) | random.
#
#   Env overrides:
#     NAV_MODE     hybrid | astar | combination | navdp   (default hybrid: A*+NavDP).
#     SEED         >=0 for a reproducible random pick       (default -1).
#     OBJECTS_FILE CONTAINER-visible path to the objects JSON (must be under the repo
#                  mounted at /opt/sparx_agency; default: the shipped objects.json,
#                  resolved automatically inside the container).
#     MODEL        s | m | l | x  detector size             (default s).
#     ENGINES_DIR  host dir with the *.engine files         (default engines/<tag>).
#     TEXT_WEIGHTS host path to yolov8<MODEL>-worldv2.pt.
#     INIT_TARGET  placeholder detector prompt before selection (default refrigerator).
#     Any object_mission.launch arg can be appended after SELECTION_MODE,
#     e.g. land_range_m:=0 arrive_radius_m:=0.8 viewer:=false
#
# Examples:
#   ./run_object_mission.sh office               # gui select, hybrid A*+NavDP
#   ./run_object_mission.sh office random        # random pick
#   NAV_MODE=astar ./run_object_mission.sh office gui land_range_m:=0
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# falcon -> planning -> tasks -> sparx_agency -> repo root (the dir with sparx_agency/)
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"

# ── Arguments ────────────────────────────────────────────────
if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0
fi
ENV_NAME="$1"; shift
SELECTION_MODE="${1:-gui}"; [[ $# -ge 1 ]] && shift || true
EXTRA_ARGS=("$@")                          # any further launch args, verbatim

NAV_MODE="${NAV_MODE:-hybrid}"
SEED="${SEED:--1}"
INIT_TARGET="${INIT_TARGET:-refrigerator}"   # re-prompted by the director on selection
MODEL="${MODEL:-s}"
# The detector sidecar needs the tensorrt+pycuda(+torch/ultralytics) env. Run this
# script from that activated venv, or point PYTHON at its interpreter.
PYTHON="${PYTHON:-python3}"

# ── Locate the GPU engines + text weights (fail loudly if missing) ──
TARGET_TAG="$(PYTHONPATH="$REPO_ROOT" "$PYTHON" -c \
  "from sparx_agency.tasks.mapping.yolo_world_trt.hardware import detect; print(detect().target_tag)" \
  2>/dev/null || echo orin_sm87)"
ENGINES_DIR="${ENGINES_DIR:-$HERE/../../mapping/yolo_world_trt/engines/$TARGET_TAG}"
BACKBONE="$ENGINES_DIR/yolo_world_${MODEL}.backbone.fp16.gpu.engine"
HEAD="$ENGINES_DIR/yolo_world_${MODEL}.head.fp16.gpu.engine"
TEXT_WEIGHTS="${TEXT_WEIGHTS:-$HERE/../../mapping/yolo_world_trt/weights/yolov8${MODEL}-worldv2.pt}"

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
if [[ ! -f "$HERE/maps/${ENV_NAME}.yaml" ]]; then
  echo "[ERROR] map not found: $HERE/maps/${ENV_NAME}.yaml" >&2
  echo "        Available: $(ls "$HERE/maps"/*.yaml 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi

LOG_DIR="/tmp/object_mission"
mkdir -p "$LOG_DIR"

echo "== Object mission (select -> fly -> land) =="
echo "   env/map     : $ENV_NAME"
echo "   selection   : $SELECTION_MODE   (seed: $SEED)"
echo "   nav_mode    : $NAV_MODE"
echo "   detector    : GPU $MODEL  ($ENGINES_DIR), initial prompt '$INIT_TARGET' (re-prompted on select)"
echo "   logs        : $LOG_DIR/{sidecar,bridge}.log"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "   extra args  : ${EXTRA_ARGS[*]}"
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
PYTHONPATH="$REPO_ROOT" "$PYTHON" \
  "$REPO_ROOT/sparx_agency/tasks/mapping/ros2/yolo_detector_ros2_node.py" \
  --ros-args \
    -p target_object:="$INIT_TARGET" \
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
LAUNCH_ARGS=(
  map_name:="$ENV_NAME"
  selection_mode:="$SELECTION_MODE"
  seed:="$SEED"
  nav_mode:="$NAV_MODE"
)
[[ -n "${OBJECTS_FILE:-}" ]] && LAUNCH_ARGS+=( objects_file:="$OBJECTS_FILE" )
# NOT exec: keep this shell alive so the EXIT/INT/TERM trap runs cleanup() (stop the
# host detector sidecar + remove the bridge/falcon containers) when the launch ends --
# on a node crash / normal teardown, not only on Ctrl+C. exec would discard the trap.
"$HERE/run_falcon.sh" "$ENV_NAME" \
  roslaunch falcon_adapter object_mission.launch \
    "${LAUNCH_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
