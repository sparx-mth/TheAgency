#!/usr/bin/env bash
# ============================================================
# stop_scene_graph.sh — tear the scene-graph mission down.
#
#   ./stop_scene_graph.sh [--out <dir>] [--all]
#
#   default : kill the host ROS2 nodes + the detection server, from the pid
#             files under <out>/pids (latest /tmp/scene_graph/* run when --out
#             is not given), falling back to pkill -f on the module names. The
#             world, FALCON and ollama are LEFT UP, so a rerun reuses them.
#   --all   : ALSO remove the FALCON containers (falcon-sjtu, sjtu-ros1-bridge,
#             falcon-rviz — names from run_falcon_sjtu.sh), the Gazebo world
#             container (sjtu_drone_* — name from bringup_world.sh) and its
#             host-side bringup wrapper, and STOP (not rm) ollama-scene-graph
#             so the pulled model volume stays warm.
#
# Prints every pid and container it killed.
# ============================================================
set -uo pipefail

say() { echo "[scene-graph-stop] $*"; }

OUT_DIR=""
ALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
        --all) ALL=1; shift ;;
        # Printed to the real end of the banner rather than to a line number
        # that drifts the moment the banner is edited.
        -h|--help)
            awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[scene-graph-stop] ERROR: unknown flag '$1'" >&2; exit 2 ;;
    esac
done

# No --out: the newest run directory, if any.
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="$(ls -1dt /tmp/scene_graph/*/ 2>/dev/null | head -n1 || true)"
    [[ -n "${OUT_DIR}" ]] && say "no --out given; using latest run ${OUT_DIR}"
fi

KILLED=""
kill_pidfile() {
    local pf="$1" name pid
    [[ -e "${pf}" ]] || return 0
    name="$(basename "${pf}" .pid)"
    pid="$(cat "${pf}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] || { rm -f "${pf}"; return 0; }
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8; do kill -0 "${pid}" 2>/dev/null || break; sleep 0.25; done
        kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null
        say "killed ${name} (pid ${pid})"
        KILLED="${KILLED} ${name}"
    fi
    rm -f "${pf}"
}

# The eight nodes and the detection server, and NOTHING ELSE. world_bringup.pid
# is deliberately excluded here: it is the host-side `docker run` client for the
# Gazebo container, and killing it detaches from a container that keeps running
# — the world would survive with no handle left to stop it by. It is handled
# under --all, next to the container removal that actually ends the world.
if [[ -n "${OUT_DIR}" && -d "${OUT_DIR}/pids" ]]; then
    for pf in "${OUT_DIR}"/pids/*_node.pid "${OUT_DIR}"/pids/detection_server.pid; do
        kill_pidfile "${pf}"
    done
else
    say "no pid directory found (${OUT_DIR:-<none>}); relying on pkill fallback"
fi

# Fallback for anything the pid files missed (a run started by hand, a pid file
# lost to a crash). Full module paths, so nothing else can match.
for pattern in \
    'sparx_agency\.tasks\.mapping\.scene_graph\.ros2\.' \
    'sparx_agency\.tasks\.mapping\.scene_graph\.serve\.detection_server'; do
    strays="$(pgrep -f "${pattern}" 2>/dev/null | tr '\n' ' ')"
    if [[ -n "${strays// /}" ]]; then
        # shellcheck disable=SC2086
        kill -9 ${strays} 2>/dev/null || true
        say "pkill fallback ${pattern}: killed pids ${strays}"
        KILLED="${KILLED} [${pattern}:${strays}]"
    fi
done

if [[ "${ALL}" == "1" ]]; then
    # FALCON stack. These three names are the literal `docker run --name`
    # arguments in run_falcon_sjtu.sh (and its own cleanup() removes exactly
    # this set) — a name that drifts here leaves a container holding the
    # roscore, and the next run's roslaunch fails in a way that reads as FALCON.
    for c in falcon-sjtu sjtu-ros1-bridge falcon-rviz; do
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${c}"; then
            docker rm -f "${c}" >/dev/null 2>&1 && say "removed container ${c}"
            KILLED="${KILLED} ${c}"
        fi
    done
    # The Gazebo world. bringup_world.sh names it sjtu_drone_<world> and runs it
    # with --rm, so removing the container IS the teardown; the wrapper process
    # then exits on its own, and its pid file goes with it.
    for c in $(docker ps -a --filter 'name=^/sjtu_drone_' --format '{{.Names}}' 2>/dev/null); do
        docker rm -f "${c}" >/dev/null 2>&1 && say "removed container ${c}"
        KILLED="${KILLED} ${c}"
    done
    kill_pidfile "${OUT_DIR:-/nonexistent}/pids/world_bringup.pid"
    # ollama is STOPPED, not removed: the named volume keeps the pulled model,
    # so the next run skips a ~2 GB download. Unpause first — `docker stop` on a
    # paused container is an error on some engine versions, and a paused
    # container is listed by `docker ps` as if it were running.
    OLLAMA_STATE="$(docker inspect -f '{{.State.Status}}' ollama-scene-graph 2>/dev/null || true)"
    case "${OLLAMA_STATE}" in
        paused)
            docker unpause ollama-scene-graph >/dev/null 2>&1 || true
            docker stop ollama-scene-graph >/dev/null 2>&1 && say "stopped container ollama-scene-graph (kept)"
            KILLED="${KILLED} ollama-scene-graph(stopped)" ;;
        running|restarting)
            docker stop ollama-scene-graph >/dev/null 2>&1 && say "stopped container ollama-scene-graph (kept)"
            KILLED="${KILLED} ollama-scene-graph(stopped)" ;;
        *) ;;   # absent or already exited: nothing to do
    esac
fi

if [[ -z "${KILLED// /}" ]]; then
    say "nothing was running."
else
    say "done. killed:${KILLED}"
fi
exit 0
