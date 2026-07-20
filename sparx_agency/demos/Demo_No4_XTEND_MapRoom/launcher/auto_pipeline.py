"""The scripted full bring-up, in the order the pipeline can actually start in.

One bash script run in a single tmux session on the Jetson. It exists because
the order matters and the waits between the steps matter more: depth cannot
start before frames are flowing, mapping cannot start before the drone is off
the ground and steady, and starting them anyway produces a stack that looks up
and is quietly useless.

Each node command is taken from the launcher's catalog rather than written out
again here, so a parameter edited on the Parameters tab is what AUTO launches
too. The waits, the arm/takeoff and the ordering are this module's own.

SAFETY: this arms the drone and takes off. Nothing in it asks twice.
"""
from __future__ import annotations

from typing import Callable

from .environments import JETSON_REPO, normalize_command

#: Session names AUTO creates, in the order it creates them.
AUTO_SESSIONS = ("xtend_bridge", "xtend_depth", "xtend_demo_manager",
                 "xtend_apriltag", "xtend_static_tf", "xtend_pose_to_tf",
                 "xtend_octomap")

#: The tmux session AUTO itself runs in.
AUTO_SESSION = "xtend_auto_launch"

_HELPERS = f"""
set +e

echo "[AUTO] XTEND RGBD mapping pipeline auto launch started"

cd {JETSON_REPO}
source /opt/ros/humble/setup.bash
source {JETSON_REPO}/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:${{LD_LIBRARY_PATH}}
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:{JETSON_REPO}:{JETSON_REPO}/sparx_agency:${{PYTHONPATH}}

wait_for_topic_name() {{
    local topic="$1"
    local timeout_sec="$2"
    echo "[AUTO] Waiting for topic name: ${{topic}} (${{timeout_sec}}s)"
    local start_t=$(date +%s)
    while true; do
        if ros2 topic list | grep -qx "${{topic}}"; then
            echo "[AUTO] Topic exists: ${{topic}}"
            return 0
        fi
        local now_t=$(date +%s)
        if [ $((now_t - start_t)) -ge "${{timeout_sec}}" ]; then
            echo "[AUTO][WARN] Timeout waiting for topic name: ${{topic}}"
            return 1
        fi
        sleep 1
    done
}}

wait_for_topic_rate() {{
    local topic="$1"
    local timeout_sec="$2"
    local reliability="${{3:-best_effort}}"
    echo "[AUTO] Waiting for messages on: ${{topic}} (${{timeout_sec}}s, qos=${{reliability}})"
    timeout "${{timeout_sec}}" ros2 topic hz "${{topic}}" >/tmp/xtend_wait_topic_hz.log 2>&1 || true
    if grep -q "average rate" /tmp/xtend_wait_topic_hz.log 2>/dev/null; then
        echo "[AUTO] Topic has messages: ${{topic}}"
        cat /tmp/xtend_wait_topic_hz.log || true
        return 0
    fi
    echo "[AUTO][WARN] No messages confirmed on: ${{topic}}"
    cat /tmp/xtend_wait_topic_hz.log || true
    return 1
}}

send_xtend_cmd() {{
    local action="$1"
    local value="${{2:-0}}"
    echo "[AUTO] Sending XTEND command: action=${{action}}, value=${{value}}"
    ros2 topic pub --once /xtend/cmd_nav std_msgs/msg/String "{{data: '{{\\"action\\":\\"${{action}}\\", \\"value\\":${{value}}}}'}}"
}}

start_tmux() {{
    local session="$1"
    local command="$2"
    echo "[AUTO] Starting tmux session: ${{session}}"
    tmux kill-session -t "${{session}}" 2>/dev/null || true
    tmux new-session -d -s "${{session}}" "bash -lc $(printf '%q' "${{command}}")"
}}
"""

#: The node environment each AUTO-started session sources for itself; the outer
#: script's exports do not survive into a fresh `tmux new-session` shell.
_NODE_ENV = f"""cd {JETSON_REPO}
source /opt/ros/humble/setup.bash
source {JETSON_REPO}/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export PYTHONPATH={JETSON_REPO}:{JETSON_REPO}/sparx_agency:${{PYTHONPATH}}"""


def _session(name: str, command: str, *, with_env: bool = True) -> str:
    """A ``start_tmux`` call carrying ``command``, safely single-quoted."""
    body = (_NODE_ENV + "\n" + normalize_command(command)) if with_env else \
        normalize_command(command)
    return "start_tmux %s '%s'\n" % (name, body.replace("'", "'\\''"))


def build(command_for: Callable[[str], str]) -> str:
    """Render the AUTO script against the launcher's current commands.

    Args:
        command_for: Maps a ``tmux_name`` to the command the launcher would
            start for it right now -- i.e. with the operator's parameter edits
            already in it. AUTO and the individual items therefore cannot
            disagree about how a node is started.

    Returns:
        The bash script, ready to run in one tmux session on the Jetson.
    """
    steps = [_HELPERS]
    add = steps.append

    add('echo "[AUTO] Step 1: start online bridge + frame dir publisher"\n')
    add(_session("xtend_bridge", command_for("xtend_bridge")))
    add("""
wait_for_topic_name /xtend/rgb_frame_path 20
wait_for_topic_rate /xtend/bearing 20 best_effort || true
wait_for_topic_rate /xtend/rgb_frame_path 20 best_effort || true
""")

    add('echo "[AUTO] Step 2: start DA3 Large Metric depth + point cloud"\n')
    add(_session("xtend_depth", command_for("xtend_depth")))
    add("""
wait_for_topic_name /xtend/depth_m 30
wait_for_topic_rate /xtend/depth_m 60 best_effort
wait_for_topic_name /xtend/pointcloud 15
""")

    add('echo "[AUTO] Step 3: start XTEND demo mode manager"\n')
    add(_session("xtend_demo_manager", command_for("xtend_demo_manager")))
    add("wait_for_topic_name /xtend/demo_mode 15 || true\n")

    add("""
echo "[AUTO] Step 4: arm and takeoff"
send_xtend_cmd arm 0
sleep 3
send_xtend_cmd takeoff 0

echo "[AUTO] Waiting 30 seconds for takeoff/stabilization"
sleep 30

echo "[AUTO] Re-check depth before mapping"
wait_for_topic_rate /xtend/depth_m 30 best_effort
""")

    add('echo "[AUTO] Step 5: start localization (AprilTag provider)"\n')
    add(_session("xtend_apriltag", command_for("xtend_apriltag")))

    add('echo "[AUTO] Step 6: start static TF fallback (map -> xtend_camera)"\n')
    add(_session("xtend_static_tf", command_for("xtend_static_tf")))

    add('echo "[AUTO] Step 7: start pose-to-TF bridge (AprilTag pose -> TF)"\n')
    add(_session("xtend_pose_to_tf", command_for("xtend_pose_to_tf")))

    add('echo "[AUTO] Step 8: start octomap server"\n')
    add(_session("xtend_octomap", command_for("xtend_octomap")))
    add("wait_for_topic_name /xtend/localization 20 || true\n")

    add("""
echo "[AUTO] Step 9: perception is up. The planner is NOT started here."
echo "[AUTO]   The object mission is a two-terminal workflow, and its terminals"
echo "[AUTO]   are meant to be watched. Start them from the launcher list:"
echo "[AUTO]     item 12  FALCON A: detector sidecar   (start once, leave up)"
echo "[AUTO]     item 13  FALCON B: bridge + container (relaunch freely)"
echo "[AUTO]   plus item 11 (NavDP server) unless nav_mode is astar."

docker ps --format '{{.Names}}' | tee /tmp/xtend_docker_names.txt
grep -qx falcon /tmp/xtend_docker_names.txt || echo "[AUTO][NOTE] falcon container not running (expected until item 13)"

echo "[AUTO] Done. Active tmux sessions:"
tmux ls || true
""")
    return "".join(steps)
