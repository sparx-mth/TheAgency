#!/usr/bin/env bash
# Run a simulated flight-recording campaign, optionally with several workers at once.
#
# Run this on the HOST (not inside the container). It syncs the repo into the
# running isaac-sim container, clears anything a previous run left behind, and
# starts one collect.py per worker.
#
# Each worker is a whole Isaac Sim process with its own PX4 instance, its own
# UDP/TCP ports, its own PX4 working directory and its own RNG seed, so workers
# never interact -- see px4_launch.py for how that identity is derived. PX4
# itself caps this at 10.
#
# Usage:
#   run_collection.sh [--scene office] [--episodes N] [--workers N]
#                     [--altitude M] [--resolution WxH] [--rate-hz HZ]
#                     [--out-dir DIR] [--seed N] [--video] [--stream]
#                     [-- <extra collect.py args>]
#
# Survey a scene first, once per altitude:
#   docker exec isaac-sim bash -c "cd /tmp/dev/repo && /isaac-sim/python.sh \
#     sparx_agency/tasks/planning/sim_flight_recording/survey_scene.py \
#     --scene office --altitude 1.5 --preview"
#
# See tasks/planning/sim_flight_recording/README.md for the whole pipeline.
set -euo pipefail

CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

SCENE="office"
EPISODES=5
WORKERS=1
ALTITUDE="1.5"
RESOLUTION=""
RATE_HZ="10"
OUT_DIR=""
SEED=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)      SCENE="$2"; shift 2 ;;
        --episodes)   EPISODES="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --altitude)   ALTITUDE="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --rate-hz)    RATE_HZ="$2"; shift 2 ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --video)      EXTRA+=(--video); shift ;;
        --stream)     EXTRA+=(--stream); shift ;;
        --)           shift; EXTRA+=("$@"); break ;;
        -h|--help)    sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$OUT_DIR" ]] || OUT_DIR="$DEV_ROOT/recordings/${SCENE}"

if (( WORKERS < 1 || WORKERS > 10 )); then
    echo "ERROR: --workers must be 1..10 (PX4 gives instances >=10 the same UDP port)." >&2
    exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container '$CONTAINER' is not running." >&2
    exit 1
fi

# A previous campaign's Isaac Sim or PX4 still holding the GPU and the ports is
# the single most common cause of an unexplained crash on the next run.
if docker exec "$CONTAINER" pgrep -f "collect.py|survey_scene.py" > /dev/null 2>&1; then
    echo "ERROR: a collection run is already in progress. Stop it with:" >&2
    echo "  docker exec $CONTAINER pkill -9 -f 'collect.py|survey_scene.py'" >&2
    exit 1
fi

echo "Syncing the repo into $CONTAINER..."
docker cp "$REPO_ROOT/sparx_agency" "$CONTAINER:$DEV_ROOT/repo/" > /dev/null

echo "Starting $WORKERS worker(s): scene=$SCENE episodes=$EPISODES altitude=$ALTITUDE"
echo "Recordings go to $OUT_DIR (inside the container)."
echo

pids=()
for (( worker = 0; worker < WORKERS; worker++ )); do
    args=(--scene "$SCENE" --out-dir "$OUT_DIR" --episodes "$EPISODES"
          --altitude "$ALTITUDE" --rate-hz "$RATE_HZ" --worker "$worker"
          --pegasus-root "$DEV_ROOT/PegasusSimulator/extensions/pegasus.simulator"
          --px4-dir "$DEV_ROOT/PX4-Autopilot")
    [[ -n "$RESOLUTION" ]] && args+=(--resolution "$RESOLUTION")
    [[ -n "$SEED" ]] && args+=(--seed "$(( SEED + worker ))")
    args+=("${EXTRA[@]}")

    # Stale PX4 lock/socket files make the next instance exit instantly with no
    # explanation. collect.py clears its own, but a worker killed mid-run before
    # this point would not have.
    docker exec "$CONTAINER" rm -f "/tmp/px4_lock-$worker" "/tmp/px4-sock-$worker"

    log="$OUT_DIR/worker${worker}.log"
    docker exec "$CONTAINER" mkdir -p "$OUT_DIR"
    echo "  worker $worker -> $log"
    docker exec "$CONTAINER" bash -c \
        "cd $DEV_ROOT/repo && /isaac-sim/python.sh \
         sparx_agency/tasks/planning/sim_flight_recording/collect.py \
         $(printf '%q ' "${args[@]}") > '$log' 2>&1" &
    pids+=($!)

    # Kit's start-up is the heaviest moment of a worker's life; overlapping two
    # of them contends for the GPU hard enough to crash the RTX shader compiler.
    if (( worker + 1 < WORKERS )); then
        sleep 45
    fi
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo
echo "All workers finished (exit status $status). Manifests:"
docker exec "$CONTAINER" bash -c "ls -1 $OUT_DIR/campaign_w*.json 2>/dev/null" || true
echo
echo "Copy the recordings out with:  docker cp $CONTAINER:$OUT_DIR ."
exit "$status"
