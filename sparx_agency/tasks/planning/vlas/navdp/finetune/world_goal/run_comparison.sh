#!/usr/bin/env bash
# Fly the same missions twice -- pretrained NavDP, then the fine-tune -- and cut
# the side-by-side video.
#
#   bash .../world_goal/run_comparison.sh --missions-to-fly 0,1,2
#
# This is the closed-loop half of the comparison, and it is a *campaign*, not a
# command: two policies, N missions each, one Isaac Sim session per mission,
# with the inference server swapped between arms. Three things about that are
# easy to get wrong by hand and are the reason this script exists.
#
#   ONE MISSION PER SESSION. There is no teleport in the vehicle adapter, so
#   mission N+1 would start wherever mission N ended, and once one crashes the
#   aircraft is on the ground for good -- every later mission returns in three
#   seconds having flown 0.1 m. fly_navdp.py takes --mission-index for exactly
#   this; the seed still draws the whole set so both arms fly the same missions.
#
#   ONE SERVER AT A TIME. The README's recipe runs both arms' servers at once on
#   different ports. That needs a GPU with room for two NavDP policies *and*
#   Isaac Sim; on an 8 GB laptop it does not fit, and the failure is an
#   out-of-memory crash somewhere in Kit that looks nothing like its cause. The
#   arms are sequential anyway, so this starts the arm's server, flies the arm,
#   and stops it again. Set --both-servers to keep the original behaviour.
#
#   THE ARMS MUST DIFFER ONLY IN THEIR WEIGHTS. Same seed, same missions, same
#   settle time, same commitment settings. Anything else and the comparison
#   measures the difference in the harness.
#
# Everything policy-side runs in the `navdp` conda env on the HOST (Isaac's
# Python has torch but not diffusers); the flying runs inside the isaac-sim
# container and reaches the server over host networking.
#
# ffmpeg and ffprobe come from the `navdp` env, which is why every video step
# here goes through `conda run -n navdp`. Neither is on the bare host PATH and
# neither is in the container -- the container writes its chase-cam MP4 with
# cv2.VideoWriter, which needs no ffmpeg, and that is the only reason the
# flights themselves work without one. `conda install -n navdp -c conda-forge
# ffmpeg` if this env is ever rebuilt.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
WG=sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal
FLY=sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/fly_navdp.py

CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"

SCENE="${NAVDP_WG_SCENE:-office}"
SEED="${NAVDP_WG_SEED:-4242}"
MISSION_SET="${NAVDP_WG_MISSIONS:-6}"
TO_FLY="0,1,2"
OUT="${NAVDP_WG_FLIGHTS:-$HOME/navdp_world_goal/flights}"
BASE_CKPT="${NAVDP_CKPT:-$HOME/Downloads/navdp-cross-modal.ckpt}"
TUNED_CKPT="${NAVDP_WG_CKPT:-$HOME/navdp_world_goal/navdp-world-goal.ckpt}"
NAVDP_REPO="${NAVDP_REPO:-$HOME/PycharmProjects/NavDP/baselines/navdp}"
# The drone's OWN forward camera, not the external chase view: those are the
# exact frames NavDP ran inference on, so the panel shows what the policy saw
# rather than what a spectator saw. --camera chase for the outside view.
CAMERA="onboard"
ARMS="baseline trained"
BOTH_SERVERS=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)           SCENE="$2"; shift 2 ;;
        --seed)            SEED="$2"; shift 2 ;;
        --mission-set)     MISSION_SET="$2"; shift 2 ;;
        --missions-to-fly) TO_FLY="$2"; shift 2 ;;
        --out)             OUT="$2"; shift 2 ;;
        --base-ckpt)       BASE_CKPT="$2"; shift 2 ;;
        --tuned-ckpt)      TUNED_CKPT="$2"; shift 2 ;;
        --navdp-repo)      NAVDP_REPO="$2"; shift 2 ;;
        --arms)            ARMS="${2//,/ }"; shift 2 ;;
        --camera)          CAMERA="$2"; shift 2 ;;
        --both-servers)    BOTH_SERVERS=1; shift ;;
        --)                shift; EXTRA+=("$@"); break ;;
        -h|--help)         sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

LOGS="$OUT/logs"
mkdir -p "$LOGS"

# `printf '%q ' "${EXTRA[@]}"` on an EMPTY array still prints the format once,
# with an empty value -- so the flight command grows a bare '' argument and
# argparse rejects the whole run with "unrecognized arguments:" and nothing
# after the colon. Build the string only when there is something in it.
EXTRA_ARGS=""
(( ${#EXTRA[@]} )) && EXTRA_ARGS="$(printf '%q ' "${EXTRA[@]}")"

port_for()  { [[ "$1" == "baseline" ]] && echo 8888 || echo 8889; }
ckpt_for()  { [[ "$1" == "baseline" ]] && echo "$BASE_CKPT" || echo "$TUNED_CKPT"; }

server_pid=""

serve() {
    local arm="$1" port ckpt holder
    port="$(port_for "$arm")"; ckpt="$(ckpt_for "$arm")"
    [[ -f "$ckpt" ]] || { echo "ERROR: no checkpoint at $ckpt" >&2; exit 1; }
    # Refuse to start on a port something else already holds. This is not
    # fussiness: the server below would die with "Address already in use", the
    # readiness curl would be answered by the *stranger*, and the whole campaign
    # would fly against unknown weights and report the result as if it had not.
    # That happened, and only luck made the leftover the right checkpoint.
    holder="$(ss -ltnp "sport = :$port" 2>/dev/null | tail -n +2)"
    if [[ -n "$holder" ]]; then
        echo "ERROR: port $port is already in use, so the $arm arm cannot be" >&2
        echo "       served its own weights. Stop the holder first:" >&2
        echo "       $holder" >&2
        echo "       pkill -f navdp_trt_server" >&2
        exit 1
    fi
    echo "[compare] serving $arm on :$port  ($(basename "$ckpt"))"
    conda run --no-capture-output -n navdp python -m \
        sparx_agency.tasks.planning.vlas.navdp.serve.navdp_trt_server \
        --backend torch --port "$port" --ckpt "$ckpt" --navdp-repo "$NAVDP_REPO" \
        > "$LOGS/server_${arm}.log" 2>&1 &
    server_pid=$!
    # Flask answers 404 on / long before the policy is built (the agent is
    # constructed lazily inside /navigator_reset), so "answering at all" is the
    # only readiness signal there is -- and it is the one that matters, because
    # a refused connection is what the flight would hit. Safe only because the
    # port was proved free above: otherwise this cannot tell our server from
    # somebody else's.
    for _ in $(seq 1 60); do
        if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/")" != "000" ]]; then
            return 0
        fi
        kill -0 "$server_pid" 2>/dev/null || {
            echo "ERROR: $arm server died on startup; see $LOGS/server_${arm}.log" >&2
            tail -20 "$LOGS/server_${arm}.log" >&2
            exit 1
        }
        sleep 2
    done
    echo "ERROR: $arm server never answered on :$port" >&2
    exit 1
}

stop_server() {
    [[ -n "$server_pid" ]] || return 0
    # conda run spawns a child; kill the group so the Flask process goes too,
    # otherwise the next arm finds its port taken and its GPU memory gone.
    pkill -P "$server_pid" 2>/dev/null || true
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
}
trap stop_server EXIT

docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || {
    echo "ERROR: container '$CONTAINER' is not running (docker start $CONTAINER)" >&2
    exit 1
}

echo "[compare] syncing the repo into $CONTAINER"
docker cp "$REPO/sparx_agency" "$CONTAINER:$DEV_ROOT/repo/" > /dev/null

# Recordings live under the SCENE, and the copy back out takes whole arm
# directories. Two things go wrong when every scene shares one directory, and
# both did:
#
#   Frames. FlightRecorder writes rgb/NNNNNN.jpg from zero and never clears, so
#   a shorter flight into a directory a longer one used leaves the tail of the
#   OLD flight behind, and `ffmpeg -i %06d.jpg` cannot tell. A cluttered-office
#   recording ended with 201 frames of a different scene.
#
#   Whole missions. A four-mission warehouse run copied out a seven-mission
#   directory, because missions 4-6 of the previous office run were still
#   sitting there -- office flights drawn on the warehouse map and averaged
#   into the warehouse result.
#
# Keying the directory by scene makes both impossible rather than remembered,
# and the per-mission clear below still handles re-flying within one scene.
FLIGHT_DIR="$DEV_ROOT/navdp_flights/$SCENE"
for arm in $ARMS; do
    for index in ${TO_FLY//,/ }; do
        docker exec "$CONTAINER" rm -rf \
            "$FLIGHT_DIR/$arm/mission_$(printf '%02d' "$index")"
    done
done
docker exec "$CONTAINER" mkdir -p "$FLIGHT_DIR"

for arm in $ARMS; do
    (( BOTH_SERVERS )) || stop_server
    serve "$arm"
    for index in ${TO_FLY//,/ }; do
        echo "[compare] $arm mission $index"
        docker exec "$CONTAINER" rm -f /tmp/px4_lock-0 /tmp/px4-sock-0
        log="$FLIGHT_DIR/${arm}_m$(printf '%02d' "$index").log"
        docker exec "$CONTAINER" bash -c \
            "cd $DEV_ROOT/repo && /isaac-sim/python.sh $FLY \
             --scene '$SCENE' --missions '$MISSION_SET' --seed '$SEED' \
             --mission-index '$index' --arm '$arm' \
             --server 'http://127.0.0.1:$(port_for "$arm")' \
             --out '$FLIGHT_DIR' --video $EXTRA_ARGS \
             > '$log' 2>&1" || echo "[compare] WARNING: $arm mission $index exited non-zero"
        docker exec "$CONTAINER" grep -E '^\[fly\] mission' "$log" || true
    done
done
stop_server

echo "[compare] copying flights out to $OUT"
mkdir -p "$OUT"
for arm in $ARMS; do
    docker cp "$CONTAINER:$FLIGHT_DIR/$arm" "$OUT/" 2>/dev/null || \
        echo "[compare] WARNING: nothing to copy for $arm"
done

# aggregate_flights folds the per-mission results_NN.json files that one session
# each wrote into the summary.json compare_videos and report.py both read. A
# one-mission session writes results_NN.json, never summary.json's results list,
# so without this the comparison finds no missions in common and writes nothing.
conda run --no-capture-output -n navdp python -m "$WG".aggregate_flights \
    --flights "$OUT" || echo "[compare] WARNING: could not aggregate the flights"

echo "[compare] cutting the comparison video"
conda run --no-capture-output -n navdp python -m "$WG".compare_videos \
    --flights "$OUT" --out "$OUT/comparison" --left trained --right baseline \
    --layout quad --scene "$SCENE" --camera-source "$CAMERA"

echo "[compare] done -- $OUT/comparison"
