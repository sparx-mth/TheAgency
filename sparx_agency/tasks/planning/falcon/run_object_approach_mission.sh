#!/usr/bin/env bash
# ============================================================
# run_object_approach_mission.sh -- one command for the whole
# "fly the route while hunting a named object, then close on it" mission.
#
# The mission is inherently 3 processes (see README "Object approach"):
#   1. the TensorRT YOLO-World detector, a ROS2 sidecar ON THE HOST (the
#      FALCON container has no CUDA/TensorRT/pycuda, so it cannot run there);
#   2. the ros1<->ros2 bridge, which carries only two std_msgs/String topics
#      (/object_approach/detections and /object_approach/goal);
#   3. the FALCON nav stack + A* planner + visual-servo closure + HUD, inside
#      the ROS1 container via real_drone_object_approach.launch.
# This script starts all three in the documented order and tears the two
# background host helpers down on exit. The container runs in the foreground
# so you keep the target-lock HUD and can Ctrl-C the whole mission.
#
# The detector defaults to the GPU backbone engine: on-target Orin 15W
# benchmarks showed the GPU beats the DLA at every model size, so the GPU
# split is the mission default (build it with `build_all.sh <model>`).
#
# Usage:
#   ./run_object_approach_mission.sh <env> [TARGET] [GOAL_X GOAL_Y]
#     <env>       maps/<env>.yaml (e.g. office, hospital) -- REQUIRED.
#     TARGET      open-vocab prompt for the object to hunt (default: gun).
#     GOAL_X/Y    where the route flies while hunting (default: 0.0 -3.0).
#
#   Env overrides:
#     NAV_MODE    astar | combination | navdp   (default combination:
#                 A* global route + NavDP local legs; use astar for pure A*).
#     MODEL       s | m | l | x  detector size    (default s -- fastest;
#                 detect runs at ~2 Hz so a larger model is affordable).
#     ENGINES_DIR host dir holding the *.engine files
#                 (default engines/<target_tag>, resolved for this board).
#     TEXT_WEIGHTS host path to yolov8<MODEL>-worldv2.pt (the CLIP prompt
#                 encoder; torch runs once per retarget, never per frame).
#     Any real_drone_object_approach.launch arg can be appended after GOAL_Y,
#     e.g. closure_mode:=waypoint target_range_m:=1.0 viewer:=false
#     (detector knobs like conf_thresh are set on the sidecar, not here).
#
# Examples:
#   ./run_object_approach_mission.sh office gun 0.0 -3.0
#   NAV_MODE=astar MODEL=l ./run_object_approach_mission.sh office weapon 1.0 -4.0
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
TARGET="${1:-gun}";  [[ $# -ge 1 ]] && shift || true
GOAL_X="${1:-0.0}";  [[ $# -ge 1 ]] && shift || true
GOAL_Y="${1:--3.0}"; [[ $# -ge 1 ]] && shift || true
EXTRA_ARGS=("$@")                          # any further launch args, verbatim

NAV_MODE="${NAV_MODE:-combination}"
MODEL="${MODEL:-s}"
# The detector sidecar needs the tensorrt+pycuda(+torch/ultralytics) env. Run
# this script from that activated venv, or point PYTHON at its interpreter.
PYTHON="${PYTHON:-python3}"

# ── Locate the GPU engines + text weights (fail loudly if missing) ──
TARGET_TAG="$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
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

LOG_DIR="/tmp/object_approach_mission"
mkdir -p "$LOG_DIR"

echo "== Object-approach mission =="
echo "   env/map     : $ENV_NAME"
echo "   target      : $TARGET"
echo "   goal        : ($GOAL_X, $GOAL_Y)   nav_mode: $NAV_MODE"
echo "   detector    : GPU $MODEL  ($ENGINES_DIR)"
echo "   logs        : $LOG_DIR/{sidecar,bridge}.log"
echo "   retarget    : ./retarget_object.sh <object>   (switch the hunted object live)"
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
# PREPEND, never assign: the setup.bash above puts ROS's site-packages (rclpy et al)
# on PYTHONPATH, and a bare PYTHONPATH="$REPO_ROOT" prefix would drop it -- the node
# then dies on `import rclpy` even though the venv is fine.
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  "$REPO_ROOT/sparx_agency/tasks/mapping/ros2/yolo_detector_ros2_node.py" \
  --ros-args \
    -p target_object:="$TARGET" \
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

# ── 3. FALCON nav + A* + closure + HUD (container, foreground) ─
echo "[mission] 3/3 launching FALCON (nav + A* + visual-servo closure + HUD) ..."
exec "$HERE/run_falcon.sh" "$ENV_NAME" \
  roslaunch falcon_adapter real_drone_object_approach.launch \
    map_name:="$ENV_NAME" \
    target_object:="$TARGET" \
    goal_x:="$GOAL_X" goal_y:="$GOAL_Y" \
    nav_mode:="$NAV_MODE" \
    "${EXTRA_ARGS[@]}"
