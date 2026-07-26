#!/usr/bin/env bash
# Fly the PEGASUS Iris in an Isaac Sim indoor scene and watch it live over WebRTC.
#
# Run this on the HOST (not inside the container). It syncs the repo into the
# running isaac-sim container, clears anything left over from a previous run,
# and starts the flight.
#
# Usage:
#   run_flight.sh [--scene office|hospital|simple_room|warehouse|full_warehouse]
#                 [--mode px4|direct] [--altitude M] [--out-dir DIR] [--video]
#
# Once the log prints STREAMING_READY, open NVIDIA's "Isaac Sim WebRTC Streaming
# Client" and connect to  localhost  port  49100.
#
# See robots/PEGASUS/README.md for setup and for what each mode does.
set -euo pipefail

CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

SCENE="office"
MODE="px4"
ALTITUDE="1.5"
OUT_DIR=""
VIDEO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)    SCENE="$2"; shift 2 ;;
        --mode)     MODE="$2"; shift 2 ;;
        --altitude) ALTITUDE="$2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --video)    VIDEO="yes"; shift ;;
        -h|--help)  sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$OUT_DIR" ]] || OUT_DIR="$DEV_ROOT/recordings/${SCENE}_${MODE}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container '$CONTAINER' is not running." >&2
    echo "Start it first, then re-run this script." >&2
    exit 1
fi

# Only one flight at a time: they would fight over the GPU, over PX4's UDP
# ports, and over the livestream port.
if docker exec "$CONTAINER" pgrep -f "fly_px4.py|fly_direct.py" > /dev/null 2>&1; then
    echo "ERROR: a flight is already running. Stop it with:" >&2
    echo "  docker exec $CONTAINER pkill -9 -f 'fly_px4.py|fly_direct.py'" >&2
    exit 1
fi

echo "Syncing the repo into $CONTAINER..."
docker cp "$REPO_ROOT/sparx_agency" "$CONTAINER:$DEV_ROOT/repo/" > /dev/null

# PX4 leaves these behind if it was killed abruptly; a stale one makes the next
# PX4 exit instantly with no explanation.
docker exec "$CONTAINER" rm -f /tmp/px4_lock-0 /tmp/px4-sock-0

VIDEO_ARGS=()
[[ -n "$VIDEO" ]] && VIDEO_ARGS=(--video-out "$OUT_DIR.mp4" --video-source chase)

echo "Starting the $MODE flight in '$SCENE' (allow 1-5 minutes to reach STREAMING_READY)."
echo "When you see STREAMING_READY: open the Isaac Sim WebRTC Streaming Client -> localhost:49100"
echo

if [[ "$MODE" == "px4" ]]; then
    docker exec "$CONTAINER" bash -c "cd $DEV_ROOT/repo && /isaac-sim/python.sh \
        sparx_agency/tasks/planning/sim_flight_recording/fly_px4.py \
        --pegasus-root $DEV_ROOT/PegasusSimulator/extensions/pegasus.simulator \
        --px4-dir $DEV_ROOT/PX4-Autopilot \
        --scene '$SCENE' --out-dir '$OUT_DIR' --altitude '$ALTITUDE' \
        $(printf '%q ' "${VIDEO_ARGS[@]}")"
elif [[ "$MODE" == "direct" ]]; then
    docker exec "$CONTAINER" bash -c "cd $DEV_ROOT/repo && /isaac-sim/python.sh \
        sparx_agency/tasks/planning/sim_flight_recording/fly_direct.py \
        --pegasus-root $DEV_ROOT/PegasusSimulator/extensions/pegasus.simulator \
        --scene '$SCENE' --out-dir '$OUT_DIR' --altitude '$ALTITUDE' \
        $(printf '%q ' "${VIDEO_ARGS[@]}")"
else
    echo "ERROR: --mode must be 'px4' or 'direct', got '$MODE'" >&2
    exit 2
fi

echo
echo "Recording written to $OUT_DIR (inside the container)."
echo "Copy it out with:  docker cp $CONTAINER:$OUT_DIR ."
