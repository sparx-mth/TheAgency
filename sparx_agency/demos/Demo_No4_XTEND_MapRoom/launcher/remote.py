"""Getting a command to the machine that will run it, and back out again.

Three ways, and the difference is where the process ends up living:

* :func:`start_tmux_over_ssh` -- a named tmux session on the Jetson. Detached,
  so it survives both the ssh connection and this launcher closing, which is the
  whole point: the drone's stack must not die because a laptop lid shut.
* :func:`spawn_local_terminal` -- a terminal window on this machine, for the
  PC-side viewers.
* :func:`publish_demo_mode` -- a one-shot topic publish, no session at all.

Every one of them reports failure by raising :class:`RemoteError` with what the
remote actually said, rather than returning a status nobody reads.
"""
from __future__ import annotations

import json
import shlex
import subprocess

from .environments import ROS_DOMAIN_ID


class RemoteError(RuntimeError):
    """A remote command failed, carrying whatever it printed."""


def _run(argv: list[str], what: str) -> str:
    """Run ``argv``, returning its stdout or raising with its stderr."""
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RemoteError("%s failed: %s" % (
            what, result.stderr.strip() or result.stdout.strip()
            or "exit status %d" % result.returncode))
    return result.stdout


def start_tmux_over_ssh(ssh_target: str, session: str, script: str,
                        *, hold_open: bool = False) -> str:
    """Start ``script`` in a fresh detached tmux session on ``ssh_target``.

    Any session of the same name is killed first, so starting a node twice
    restarts it rather than failing on the name.

    Args:
        ssh_target: ``user@host``.
        session: tmux session name.
        script: The bash to run.
        hold_open: Keep the session alive after the script exits, and wait for a
            keypress. For a scripted run whose output is the thing you wanted --
            without it, a failure closes the window that was about to explain it.

    Returns:
        The remote ``tmux ls`` output.

    Raises:
        RemoteError: If ssh or tmux failed.
    """
    if hold_open:
        script = ("set +e\n%s\nstatus=$?\necho\n"
                  'echo "[launcher] exited with status $status"\n'
                  'echo "Press Enter to close this tmux session..."\nread\n') % script

    inner = "bash -lc %s" % shlex.quote(script)
    remote = ("tmux kill-session -t %(s)s 2>/dev/null || true; "
              "tmux new-session -d -s %(s)s %(cmd)s; "
              % {"s": shlex.quote(session), "cmd": shlex.quote(inner)})
    if hold_open:
        remote += "tmux set-option -t %s remain-on-exit on; " % shlex.quote(session)
    return _run(["ssh", ssh_target, remote + "tmux ls"],
                "starting tmux session %r" % session)


def stop_tmux_over_ssh(ssh_target: str, session: str) -> None:
    """Interrupt then kill a tmux session, if it exists.

    The Ctrl+C comes first and is given a second to land: ROS nodes shut their
    publishers down on SIGINT, and killing the session outright leaves stale
    topic registrations behind for the next run to trip over.
    """
    remote = ("tmux has-session -t %(s)s 2>/dev/null && "
              "tmux send-keys -t %(s)s C-c && sleep 1 && "
              "tmux kill-session -t %(s)s 2>/dev/null || true"
              % {"s": shlex.quote(session)})
    subprocess.run(["ssh", ssh_target, remote], check=False)


def attach_command(ssh_target: str, session: str) -> str:
    """The shell command that attaches a terminal to a running session."""
    return "ssh -t %s 'tmux attach -t %s'" % (ssh_target, session)


def publish_demo_mode(ssh_target: str, repo: str, mode: str,
                      reason: str = "manual mode button") -> None:
    """Publish one demo-mode request on the Jetson.

    Args:
        ssh_target: ``user@host``.
        repo: The repo on the Jetson, whose venv holds rclpy.
        mode: The requested mode, e.g. ``"idle"`` or ``"finish"``.
        reason: Recorded in the message, so the manager's log says who asked.

    Raises:
        RemoteError: If the publish failed.
    """
    payload = json.dumps({"mode": mode, "source": "launcher_ui_manual",
                          "reason": reason})
    remote = (
        "cd %s && source /opt/ros/humble/setup.bash && "
        "source %s/venv/bin/activate && export ROS_DOMAIN_ID=%d && "
        "ros2 topic pub --once /xtend/demo_mode_request std_msgs/msg/String %s"
        % (repo, repo, ROS_DOMAIN_ID, shlex.quote("{data: '%s'}" % payload)))
    _run(["ssh", ssh_target, remote], "publishing demo mode %r" % mode)


def spawn_local_terminal(script: str, title: str) -> None:
    """Open a terminal window on this machine running ``script``.

    Raises:
        RemoteError: If no supported terminal emulator is installed.
    """
    candidates = [
        ["gnome-terminal", "--title", title, "--", "bash", "-lc", script],
        ["xterm", "-T", title, "-e", "bash -lc %s" % shlex.quote(script)],
        ["konsole", "--new-tab", "-p", "tabtitle=%s" % title, "-e", "bash", "-lc", script],
    ]
    for argv in candidates:
        try:
            subprocess.Popen(argv)
            return
        except FileNotFoundError:
            continue
    raise RemoteError("no supported terminal emulator found (tried %s)"
                      % ", ".join(argv[0] for argv in candidates))
