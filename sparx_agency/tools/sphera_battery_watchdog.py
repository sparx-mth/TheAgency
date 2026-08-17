"""Battery watchdog for the Sphera simulator.

Watches the Rooster drone's battery telemetry inside the `it` container and,
when it drops too low: force-removes `R1`, restarts Sphera the same way the
user's own desktop icon does (``~/.sphera/sphera-restart.sh`` — a plain
``docker compose down`` / ``up -d run_sphera`` cycle), then (by default)
drives Sphera's own GUI back into the scenario via `sphera_gui_automation`
(Continue -> Manager -> Start -> Standalone -> select Rooster_1 -> Play) so a
fresh, flyable `R1` comes back with no one at the keyboard. This is
deliberately narrow in one dimension only: it does not touch Falcon, the
ROS1 bridge, or any Rooster node — it only automates the "battery died, exit
and restart the sim, get back into it" cycle. Whatever needs re-launching
afterward on the ROS side (bridge, command unit, video trigger, ...) is a
separate concern, not this script's job.

The GUI re-entry step was added 2026-08-17 by explicit user request (they
will not be at the computer when this fires) — it assumes Sphera's window
isn't moved/resized and nothing else drives its UI concurrently. Pass
``--no-gui-reentry`` to fall back to the original behavior (stop at the
container restart, leave the manual walkthrough in
``.claude/skills/fly-rooster-sphera/SKILL.md``'s post-2026-08-13 update to
whoever is at the keyboard).

Why `R1` is force-removed, not just `drone_simulator`: confirmed live
(2026-08-17) that `docker compose down`/`up` on `run_sphera` alone does NOT
kill/recreate `R1` — it's a sibling container Sphera spawns via the host
Docker socket (bind-mounted in, not a child of `drone_simulator`'s own
cgroup), so it survives the engine bounce untouched, and its battery never
resets. `LESSONS.md`'s 2026-08-13 entry already noted the green Play button
only spawns `R1` "if R1 dies mid-scenario" — it does NOT auto-respawn a
still-alive one. Removing it here guarantees the next Play spawns a genuinely
fresh `R1` with reset battery, matching what a real quit-and-relaunch of the
whole Sphera app apparently already does.

Run standalone:
    python3 sparx_agency/tools/sphera_battery_watchdog.py

Or via ``mission_control.py``'s "Sphera Battery Watchdog" service card.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import sphera_gui_automation

ROOSTER_CONTAINER = "it"
R1_CONTAINER = "R1"
SPHERA_RESTART_SCRIPT = Path("~/.sphera/sphera-restart.sh").expanduser()
# How long to wait, after Play is (supposedly) clicked, for R1 to actually
# exist and report a readable battery before declaring the re-entry failed.
POST_PLAY_VERIFY_TIMEOUT_SEC = 30.0

# Same env `it`-exec commands need to see R1's ROS2 graph — matches the
# battery-check step in fly-rooster-sphera/SKILL.md.
_BATTERY_ENV = (
    "export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/site-packages:/home/rooster\n"
    "source /opt/ros/foxy/setup.bash && source /home/rooster/workspace/install/setup.bash\n"
    "export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n"
    "export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml\n"
)

# RoosterState.msg's `percentage` field is a flat float32 in [0, 1] (see
# rooster_manager_interfaces/msg/RoosterState.msg) — a plain regex over the
# echoed YAML is enough, no message-nesting to worry about.
_PERCENTAGE_RE = re.compile(r"percentage:\s*([0-9.]+)")


def read_battery_fraction(drone_id: str = "R1", timeout_sec: int = 5) -> float | None:
    """Reads `/<drone_id>/state` for up to `timeout_sec` and returns battery
    in [0, 1] from the first message seen.

    Returns None if the topic isn't publishing right now (Sphera down, `it`
    not up, or the topic not discovered yet) rather than raising — a missing
    reading just means "skip this poll", not a fatal error.

    No `--once` here (confirmed live: this container's ROS2 Foxy `ros2 topic
    echo` doesn't support that flag at all — it's a later-ROS2 addition, and
    passing it is a silent argparse failure, not a "topic not up" case).
    Matches fly-rooster-sphera/SKILL.md's own battery-check command: let it
    stream and cut it off with `timeout` instead.
    """
    cmd = f"{_BATTERY_ENV}timeout {timeout_sec} ros2 topic echo /{drone_id}/state 2>/dev/null"
    try:
        result = subprocess.run(
            ["docker", "exec", ROOSTER_CONTAINER, "bash", "-lc", cmd],
            capture_output=True, text=True, timeout=timeout_sec + 3, check=False,
        )
    except Exception:
        return None
    match = _PERCENTAGE_RE.search(result.stdout)
    return float(match.group(1)) if match else None


def notify(message: str) -> None:
    """Best-effort desktop notification — matches sphera-restart.sh's own style."""
    subprocess.run(["notify-send", "Sphera Battery Watchdog", message], check=False)


def restart_sphera() -> None:
    """Force-removes `R1`, then runs the exact script the desktop icon runs.

    `R1` first, and always before touching `drone_simulator`: it guarantees
    that whatever Play does next (inside the freshly-restarted Sphera) finds
    no existing `R1` to reconnect to, so it spawns a genuinely fresh one with
    reset battery, instead of silently reusing the still-alive old instance
    (see the module docstring for how this was confirmed live). sphera-
    restart.sh already has its own 5s de-dupe guard and its own notify-send
    messages, so this only needs to announce *why* it fired.
    """
    notify("Battery low — restarting Sphera")
    print(f"[watchdog] battery low, removing {R1_CONTAINER} and running sphera-restart.sh", flush=True)
    subprocess.run(["docker", "rm", "-f", R1_CONTAINER], capture_output=True, check=False)
    subprocess.run(["bash", str(SPHERA_RESTART_SCRIPT)], check=False)


def _r1_exists() -> bool:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{R1_CONTAINER}$", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() == R1_CONTAINER


def _wait_for_fresh_r1(timeout_sec: float, drone_id: str) -> bool:
    """Polls for `R1` to exist AND report a readable battery after Play is
    clicked. This is the structural fallback for the whole GUI sequence: if
    a click landed wrong (unexpected dialog, Sphera slower than usual to
    render a screen, ...), this fails loudly instead of the caller silently
    assuming success."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _r1_exists() and read_battery_fraction(drone_id) is not None:
            return True
        time.sleep(2)
    return False


def restart_and_reenter(drone_id: str, gui_reentry: bool) -> None:
    """`restart_sphera()`, then (by default) drive Sphera's GUI back into
    the scenario so a fresh, flyable drone comes back with no one at the
    keyboard. Never raises — a failure here is reported via `notify()` and
    logged, and the watchdog just keeps polling (it stays disarmed until a
    battery reading recovers, which won't happen on its own if re-entry
    failed — that's the honest state to be in, not a silent retry loop).

    Timed end-to-end (logged and included in the final notification) so
    real-world runs are self-documenting — confirmed live 2026-08-17 at
    ~38s per run, but that number is this one machine's, not a guarantee;
    worth watching for drift over time rather than re-measuring by hand.
    """
    start = time.time()
    restart_sphera()
    if not gui_reentry:
        print(f"[watchdog] restart done ({time.time() - start:.1f}s, no GUI re-entry requested)", flush=True)
        return

    print("[watchdog] restart done, driving Sphera's GUI back into the scenario...", flush=True)
    if not sphera_gui_automation.enter_scenario():
        elapsed = time.time() - start
        notify(f"Sphera restarted, but its window never reappeared ({elapsed:.0f}s) — check manually")
        print(f"[watchdog] GUI re-entry aborted after {elapsed:.1f}s: Sphera window not found", flush=True)
        return

    elapsed = time.time() - start
    if _wait_for_fresh_r1(POST_PLAY_VERIFY_TIMEOUT_SEC, drone_id):
        total = time.time() - start
        print(f"[watchdog] scenario re-entered, R1 fresh, battery restored — {total:.1f}s end-to-end", flush=True)
    else:
        total = time.time() - start
        notify(f"Sphera restarted, but R1/battery never came back ({total:.0f}s) — check manually")
        print(
            f"[watchdog] GUI re-entry finished after {elapsed:.1f}s but R1/battery didn't come up "
            f"within {POST_PLAY_VERIFY_TIMEOUT_SEC:.0f}s more ({total:.1f}s total) — a click likely "
            "landed wrong (unexpected dialog, moved/resized window, ...)",
            flush=True,
        )


def watch(
    threshold: float,
    recovery_threshold: float,
    poll_interval_sec: float,
    drone_id: str,
    gui_reentry: bool,
) -> None:
    """Poll forever; fire once per drain.

    After a restart fires, the trigger is disarmed until battery is seen at
    or above `recovery_threshold` again — a fresh Sphera restart resets it to
    ~99-100% (see fly-rooster-sphera/SKILL.md), so this simply prevents
    re-firing on every poll while Sphera is still coming back up.
    """
    armed = True
    print(
        f"[watchdog] watching /{drone_id}/state — restart at <= {threshold:.0%}, "
        f"re-arm at >= {recovery_threshold:.0%}, every {poll_interval_sec:.0f}s, "
        f"gui_reentry={gui_reentry}",
        flush=True,
    )
    while True:
        pct = read_battery_fraction(drone_id)
        if pct is None:
            print("[watchdog] battery unreadable (Sphera/it down, or topic not up yet)", flush=True)
        else:
            print(f"[watchdog] battery={pct:.0%} armed={armed}", flush=True)
            if armed and pct <= threshold:
                armed = False
                restart_and_reenter(drone_id, gui_reentry)
            elif not armed and pct >= recovery_threshold:
                armed = True
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-id", default="R1", help="Rooster drone id to watch (default: R1).")
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Restart Sphera once battery is at or below this fraction, 0-1 (default: 0.10).",
    )
    parser.add_argument(
        "--recovery-threshold", type=float, default=0.80,
        help="Re-arm the trigger once battery is seen at or above this fraction (default: 0.80).",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=10.0,
        help="Seconds between battery checks (default: 10).",
    )
    parser.add_argument(
        "--no-gui-reentry", action="store_true",
        help="Stop at the container restart; don't drive Sphera's GUI back into "
             "the scenario. Use this if you'll be at the keyboard to do it yourself.",
    )
    args = parser.parse_args()

    if not SPHERA_RESTART_SCRIPT.exists():
        sys.exit(f"Sphera restart script not found: {SPHERA_RESTART_SCRIPT}")
    if not (0.0 <= args.threshold < args.recovery_threshold <= 1.0):
        sys.exit("--threshold must be < --recovery-threshold, both within [0, 1].")
    if not args.no_gui_reentry and shutil.which("xdotool") is None:
        sys.exit("xdotool not found (required for GUI re-entry) — install it, or pass --no-gui-reentry.")

    try:
        watch(args.threshold, args.recovery_threshold, args.poll_interval, args.drone_id, not args.no_gui_reentry)
    except KeyboardInterrupt:
        print("\n[watchdog] stopped.")


if __name__ == "__main__":
    main()
