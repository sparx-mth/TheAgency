"""FALCON exploration bring-up sequence for the ROBOTICAN Rooster in Sphera.

Kept out of ``mission_control.py`` because this is a policy, not a service
list: a generic "launch everything in order" pass cannot fly this stack, since
two of its steps are gated on the aircraft's state rather than on a process
being up.

Two constraints shape the order:

* **The twist control adapter must not be running during takeoff.** Its
  watchdog publishes ``{"action": "stop"}`` at 20 Hz whenever ``/cmd_vel`` has
  been quiet, and ``RoosterUnit.stop()`` cancels an in-progress climb -- so
  launching it alongside everything else kills the takeoff within ~50 ms.
* **Exploration only works from a hover.** ``exploration_node`` picks
  viewpoints in free 3D space and has no route from a ground-level pose;
  started too early it spins at high CPU logging ``[FSM] Plan fail`` rather
  than failing loudly.

So the sequence is: bring up sensing and planning, arm, take off, wait for the
altitude hold to actually settle, and only then hand control to FALCON.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time

# Vendor backend container -- every Rooster ROS2 command execs into this.
ROOSTER_CONTAINER = "it"

# Environment every ros2 call inside the vendor container needs. Domain 9 is
# Sphera's; the CycloneDDS profile pins the NIC (see the interface-selection
# bug in LESSONS.md).
_IT_ENV = (
    "source /opt/ros/foxy/setup.bash && "
    "source /home/rooster/workspace/install/setup.bash && "
    "export ROS_DOMAIN_ID=9 && "
    "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
    "export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml && "
)

# Everything that must be up BEFORE the aircraft leaves the ground, in
# dependency order, as (service_key, wait_timeout_s). Deliberately excludes:
#   rooster_twist_control_R1      -- would cancel the takeoff (see module docs)
#   rooster_planner_object_approach -- publishes /cmd_vel_raw, the same topic
#                                      the exploration follower drives
#   rooster_position_ctrl         -- manual keyboard controller, fights everything
#   rooster_planner_detector      -- only needed for object approach
FALCON_PREFLIGHT_ORDER: list[tuple[str, float]] = [
    ("rooster_dev_container", 20),     # hosts frame capture / depth / twist adapter
    ("rooster_it_container", 15),      # vendor backend, everything execs into it
    ("rooster_planner_falcon", 20),    # falcon container must exist before roslaunch
    ("rooster_gtl_R1", 8),
    ("rooster_video_trigger_R1", 8),
    ("rooster_command_unit_R1", 8),
    ("rooster_frame_capture_R1", 10),  # needs the video trigger
    ("rooster_depth_R1", 10),          # needs frame capture
    ("rooster_planner_adapter", 15),   # roslaunch sphera_drone.launch
    ("rooster_planner_bridge", 15),    # after the adapter -- falcon's roscore is new
    ("rooster_planner_rviz", 8),
]

# Started only once the aircraft is confirmed hovering.
TWIST_ADAPTER_KEY = "rooster_twist_control_R1"


def send_cmd_nav(action: str, drone_id: str = "R1",
                 timeout: float = 12.0) -> str | None:
    """Publish one ``cmd_nav`` action to the command unit.

    Args:
        action: Action name understood by ``rooster_command_unit.py`` --
            ``arm``, ``takeoff``, ``land``, ``disarm``, ``stop``.
        drone_id: Rooster id, e.g. ``R1``.
        timeout: Seconds to allow the publish to complete.

    Returns:
        ``None`` on success, or an error string.
    """
    payload = json.dumps({"data": json.dumps({"action": action})})
    cmd = (
        f"{_IT_ENV} ros2 topic pub -1 /{drone_id}/cmd_nav "
        f"std_msgs/msg/String {shlex.quote(payload)}"
    )
    try:
        r = subprocess.run(
            ["docker", "exec", ROOSTER_CONTAINER, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stderr.strip() if r.returncode != 0 else None
    except subprocess.TimeoutExpired:
        return f"cmd_nav '{action}' timed out after {timeout:.0f}s"
    except Exception as exc:
        return str(exc)


# e.g. "... ranger=1.925m target=1.600m error=-0.325m vel=+0.0016m/s ..."
_HOVER_RE = re.compile(r"ranger=([0-9.]+)m.*?vel=([+-][0-9.]+)m/s")


def read_hover_samples(drone_id: str = "R1", lines: int = 40) -> list[tuple[float, float]]:
    """Read recent (ranger, vertical velocity) pairs from the command unit log.

    The command unit's altitude-hold tick is the only place that reports the
    ranger the hold loop is actually chasing, and it tees to this log inside
    the vendor container.

    Args:
        drone_id: Rooster id, e.g. ``R1``.
        lines: How many trailing log lines to scan.

    Returns:
        Oldest-first list of ``(ranger_m, vel_mps)``; empty if the log has no
        altitude-hold lines yet (which is normal before takeoff).
    """
    log_path = f"/tmp/rooster_command_unit_{drone_id}.log"
    try:
        r = subprocess.run(
            ["docker", "exec", ROOSTER_CONTAINER, "tail", "-n", str(lines), log_path],
            capture_output=True, text=True, timeout=8,
        )
    except Exception:
        return []
    if r.returncode != 0:
        return []
    return [(float(m.group(1)), float(m.group(2)))
            for m in (_HOVER_RE.search(ln) for ln in r.stdout.splitlines()) if m]


def wait_for_stable_hover(drone_id: str = "R1", timeout: float = 45.0,
                          min_ranger_m: float = 0.6, tolerance_m: float = 0.08,
                          max_vel_mps: float = 0.05,
                          samples: int = 6) -> tuple[bool, str]:
    """Block until the altitude hold has actually settled, or give up.

    "Settled" means airborne, with the last few ranger readings inside a small
    band and vertical velocity near zero. A drifting climb passes the velocity
    test on any single sample, which is why the band is checked across several.

    Args:
        drone_id: Rooster id, e.g. ``R1``.
        timeout: Seconds before giving up.
        min_ranger_m: Below this the aircraft is treated as still on the ground.
        tolerance_m: Allowed spread across the sampled ranger readings.
        max_vel_mps: Allowed mean absolute vertical velocity.
        samples: How many consecutive readings must satisfy the above.

    Returns:
        ``(ok, message)`` -- the message describes the final state either way,
        so it can go straight into the UI.
    """
    deadline = time.time() + timeout
    last = "no altitude-hold lines yet"
    while time.time() < deadline:
        window = read_hover_samples(drone_id)[-samples:]
        if len(window) >= samples:
            rangers = [r for r, _ in window]
            spread = max(rangers) - min(rangers)
            mean_vel = sum(abs(v) for _, v in window) / len(window)
            if min(rangers) < min_ranger_m:
                last = f"still on the ground (ranger {rangers[-1]:.2f} m)"
            elif spread <= tolerance_m and mean_vel <= max_vel_mps:
                return True, (f"hover settled at {rangers[-1]:.2f} m "
                              f"(spread {spread * 100:.0f} cm, |vel| {mean_vel:.3f} m/s)")
            else:
                last = (f"not settled yet -- {rangers[-1]:.2f} m, "
                        f"spread {spread * 100:.0f} cm, |vel| {mean_vel:.3f} m/s")
        time.sleep(1.0)
    return False, f"hover did not settle within {timeout:.0f}s ({last})"
