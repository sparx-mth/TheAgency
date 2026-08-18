"""Headless, health-checked bring-up of the whole FALCON/Rooster/Sphera stack.

``mission_control.py`` can do this from a browser; nothing could do it from a
script, which is what an unattended multi-day campaign needs. The ordering here
is not a preference -- every step encodes a failure this stack has actually hit:

* the ``ros1_bridge`` goes stale the moment ``falcon`` or ``R1`` is recreated,
  so it is always started **after** them, never before;
* ``falcon``'s voxel map has no decay, so any ``R1`` recreation means the whole
  container is rebuilt rather than its nodes restarted -- garbage fused during
  the disruption would otherwise never clear;
* ``video_trigger.py`` keeps writing frames with byte-identical content after
  ``R1`` is recreated, so a freshness watchdog runs for the whole campaign;
* the twist control adapter publishes ``stop`` at 20 Hz whenever ``/cmd_vel`` is
  quiet, which cancels a takeoff within ~50 ms -- it is deliberately NOT started
  here, only after the aircraft is confirmed hovering;
* more than one publisher on ``/R1/manual_control`` silently ruins a whole day
  of tests, so every step kills its own predecessors first and the result is
  verified by publisher count, not by convention.

Python 3.8-compatible.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time

from sparx_agency.tools.falcon_campaign import config as C


class BringupError(RuntimeError):
    """A step failed its health check. Never raised for a recoverable retry."""


# ── shell helpers ────────────────────────────────────────────────────────
def sh(cmd, timeout=60, check=False):
    """Run a shell command, capturing output.

    Args:
        cmd: Shell command line.
        timeout: Seconds before giving up.
        check: Raise :class:`BringupError` on a non-zero exit.

    Returns:
        ``subprocess.CompletedProcess``. A timeout is returned as returncode 124
        rather than raised, so callers can treat it as a failed health check.
    """
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        r = subprocess.CompletedProcess(cmd, 124, "", "timeout after %ss" % timeout)
    if check and r.returncode != 0:
        raise BringupError("%s -> rc=%s\n%s" % (cmd, r.returncode, r.stderr[:500]))
    return r


def spawn(cmd, log_path):
    """Start a long-running command detached, with a PTY, teeing to a log.

    ``run_falcon_sphera.sh`` and ``run_bridge.sh`` use ``docker run -it`` and
    refuse to start without a controlling terminal, so everything goes through
    ``script`` rather than a plain ``Popen``.

    Args:
        cmd: The command to run.
        log_path: Host path to tee combined output into.
    """
    script = "%s 2>&1 | tee %s" % (cmd, shlex.quote(str(log_path)))
    subprocess.Popen(
        ["bash", "-lc", "script -qc %s /dev/null" % shlex.quote(script)],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def docker_exec_d(container, env, cmd, log_name):
    """Start a command detached inside a container, teeing to a log in it."""
    full = "%s %s 2>&1 | tee /tmp/%s" % (env, cmd, log_name)
    subprocess.run(["docker", "exec", "-d", container, "bash", "-lc", full],
                   capture_output=True, timeout=30)


def container_up(name):
    """Whether a container exists and is running."""
    r = sh("docker inspect -f '{{.State.Running}}' %s 2>/dev/null" % shlex.quote(name), 20)
    return r.stdout.strip() == "true"


def kill_host(pattern):
    """Kill host-side processes matching a pattern, ignoring absence."""
    sh("pkill -f %s" % shlex.quote(pattern), 15)


def kill_in(container, pattern):
    """Kill processes inside a container.

    ``pgrep -f`` is deliberately avoided inside ``it`` -- it self-matches and has
    produced watchdogs that kill themselves.
    """
    sh("docker exec %s bash -lc \"ps -eo pid,args | grep %s | grep -v grep | "
       "awk '{print \\$1}' | xargs -r kill\" 2>/dev/null"
       % (shlex.quote(container), shlex.quote(pattern)), 25)


# ── safety ───────────────────────────────────────────────────────────────
def assert_simulator():
    """Refuse to continue unless the drone container is the Sphera simulator.

    The campaign arms and flies a drone unattended. This is the one check that
    stands between that and a real airframe.

    Raises:
        BringupError: If the drone container is missing or is not a Sphera image.
    """
    r = sh("docker inspect -f '{{.Config.Image}}' %s 2>/dev/null" % C.DRONE_CONTAINER, 20)
    image = r.stdout.strip()
    if not image:
        raise BringupError(
            "%s does not exist -- cannot verify this is the simulator. Refusing "
            "to fly." % C.DRONE_CONTAINER)
    if not image.startswith(C.SIM_IMAGE_PREFIX):
        raise BringupError(
            "%s runs image %r, which is not a %r image. Refusing to fly: this "
            "campaign may only ever command the simulator."
            % (C.DRONE_CONTAINER, image, C.SIM_IMAGE_PREFIX))
    return image


#: Marker strings that must be present in the compiled FALCON artifacts, mapped
#: to the artifact expected to contain them. Each corresponds to a patch in
#: ``tasks/planning/falcon/patches/`` whose rosparams ``nav_stack.launch`` sets.
_PATCH_MARKERS = {
    "/catkin_ws/devel/lib/exploration_manager/exploration_node": [
        "finish_grace",             # fix_falcon_finish_grace.sh
        "publish_fail_blacklist",   # falcon_publish_fail_blacklist.patch
        "replan_from_pose",         # falcon_replan_from_pose.patch
    ],
    "/catkin_ws/devel/lib/libexploration_preprocessing.so": [
        "visib_unknown",            # fix_falcon_frontier_visibility.sh
        "amnesty",                  # falcon_finish_amnesty_gate.patch
    ],
}


def assert_falcon_patches():
    """Refuse to fly a falcon image that predates its own patches.

    This exact failure cost a full day: the image was built at 11:17 and the
    patch that stops exploration quitting after 26 s landed at 14:36, so
    ``nav_stack.launch`` spent the day setting rosparams that nothing in the
    binary read. Nothing anywhere said so -- the launch echoed the parameters
    and the FSM quietly sat in its terminal FINISH state.

    Checked at bring-up rather than at build time on purpose: a build-time gate
    only protects the build that runs it, while this catches any stale image
    that later gets started.

    Raises:
        BringupError: If any expected marker is missing from its artifact.
    """
    missing = []
    for artifact, markers in _PATCH_MARKERS.items():
        for marker in markers:
            r = sh("docker exec %s bash -lc %s" % (
                C.FALCON_CONTAINER,
                shlex.quote("strings %s 2>/dev/null | grep -c %s"
                            % (shlex.quote(artifact), shlex.quote(marker)))), 40)
            try:
                found = int(r.stdout.strip().splitlines()[0])
            except (ValueError, IndexError):
                found = 0
            if found == 0:
                missing.append("%s in %s" % (marker, artifact.rsplit("/", 1)[-1]))
    if missing:
        raise BringupError(
            "The falcon image is STALE -- these patch markers are absent from the "
            "compiled binaries, so nav_stack.launch's matching rosparams are "
            "silently inert:\n  %s\nRebuild first:\n  docker build -t falcon-ros:"
            "noetic %s/sparx_agency/tasks/planning/falcon/"
            % ("\n  ".join(missing), C.REPO_ROOT))
    return True


def battery_fraction():
    """Battery as a 0-1 fraction from ``/R1/state``, or None if unreadable."""
    # No `--once`: this container is ROS 2 Foxy, where that flag does not exist.
    # `timeout` + `grep -m1` is the form the runbook has always used here.
    cmd = ("docker exec %s bash -lc %s" % (
        C.IT_CONTAINER,
        shlex.quote(C.IT_ENV + "timeout 6 ros2 topic echo %s 2>/dev/null "
                    "| grep -m1 percentage" % C.ROS2_TOPICS["state"])))
    r = sh(cmd, 25)
    for token in r.stdout.replace(":", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        return value if value <= 1.0 else value / 100.0
    return None


# ── steps ────────────────────────────────────────────────────────────────
def restart_sphera(reentry_attempts=4):
    """Restart Sphera and drive its GUI back into the scenario.

    The GUI re-entry is retried rather than trusted once. An X window exists
    well before Sphera is actually rendering and accepting input, so the first
    click after a cold start is regularly swallowed -- observed live: the
    watchdog's single attempt left Sphera sitting on its Welcome screen, and the
    identical click sequence worked first time a few minutes later. Retrying is
    safe because every step is idempotent from the caller's point of view: the
    only success criterion is a fresh drone container appearing.

    Args:
        reentry_attempts: How many times to drive the click sequence before
            giving up.

    Returns:
        True if a fresh drone container came back.
    """
    sh(C.SPHERA_RESTART_CMD, timeout=420)
    if container_up(C.DRONE_CONTAINER):
        return True

    from sparx_agency.tools import sphera_gui_automation

    for attempt in range(1, reentry_attempts + 1):
        # Let Sphera finish becoming interactive before clicking at it.
        time.sleep(15)
        try:
            sphera_gui_automation.enter_scenario()
        except Exception:                          # noqa: BLE001 -- GUI is flaky
            pass
        if wait_for(lambda: container_up(C.DRONE_CONTAINER), 60,
                    "fresh %s (re-entry attempt %d)" % (C.DRONE_CONTAINER, attempt)):
            return True
    return False


def ensure_sphera(min_battery=0.30):
    """Make sure a fresh, flyable ``R1`` exists with usable battery.

    Args:
        min_battery: Restart Sphera if the battery is below this fraction.
            Calibration and speed measurements taken below ~25% are corrupted
            by loss of thrust authority, so this is a data-quality gate as much
            as an endurance one.

    Returns:
        True if the simulator is ready to fly.
    """
    if not container_up(C.DRONE_CONTAINER):
        return restart_sphera()
    level = battery_fraction()
    if level is not None and level < min_battery:
        return restart_sphera()
    return True


def start_containers():
    """Bring up the two long-lived support containers if they are down."""
    if not container_up(C.DEV_CONTAINER):
        sh("cd %s && docker compose -f docker-compose.robotican.yml up -d robotican"
           % C.REPO_ROOT, 180)
    if not container_up(C.IT_CONTAINER):
        raise BringupError(
            "%s is down and this campaign does not know how to recreate it "
            "(it is created alongside Sphera). Restart Sphera." % C.IT_CONTAINER)


def start_falcon(follower=None, extra=""):
    """Recreate the falcon container and launch the adapter inside it.

    The container is always recreated rather than reused: ``exploration_node``'s
    voxel map is long-lived with no decay, so a map polluted during a previous
    run's disruption would silently persist into this one.
    """
    sh("docker rm -f %s" % C.FALCON_CONTAINER, 60)
    spawn(C.FALCON_CONTAINER_CMD, "/tmp/falcon_container.log")
    if not wait_for(lambda: container_up(C.FALCON_CONTAINER), 90, "falcon container"):
        raise BringupError("falcon container never came up; see /tmp/falcon_container.log")
    time.sleep(8)
    assert_falcon_patches()
    # Stopgap until falcon-ros is rebuilt with these: combination_planner_node
    # crashes on `import requests` without them.
    sh("docker exec %s bash -c 'apt-get update -qq && apt-get install -y "
       "python3-requests python3-pil' " % C.FALCON_CONTAINER, 300)
    spawn(C.adapter_launch_cmd(follower, extra), "/tmp/falcon_roslaunch.log")
    if not wait_for(lambda: rosnode_exists("/exploration_node"), 120, "exploration_node"):
        raise BringupError("exploration_node never started; see /tmp/falcon_roslaunch.log")


def start_bridge():
    """Recreate the ROS1<->ROS2 bridge. Always AFTER falcon and R1."""
    sh("docker rm -f %s" % C.BRIDGE_CONTAINER, 60)
    spawn(C.BRIDGE_CMD, "/tmp/bridge.log")
    wait_for(lambda: container_up(C.BRIDGE_CONTAINER), 90, "ros1_bridge")
    time.sleep(5)


def start_rooster_nodes():
    """Ground truth, video, command unit, frame capture and depth.

    Each kills its own predecessors first: a second ``rooster_command_unit`` or
    a stale ``video_trigger`` are both silent, day-destroying failures.
    """
    kill_in(C.IT_CONTAINER, "rooster_ground_truth_localization")
    docker_exec_d(C.IT_CONTAINER, C.IT_ENV, C.GTL_CMD, "rooster_gtl.log")

    kill_in(C.IT_CONTAINER, "video_trigger.py")
    kill_host("video_trigger.py")
    spawn(C.VIDEO_TRIGGER_CMD, "/tmp/rooster_video_trigger.log")

    kill_in(C.IT_CONTAINER, "rooster_command_unit")
    docker_exec_d(C.IT_CONTAINER, C.IT_ENV, C.COMMAND_UNIT_CMD,
                  "rooster_command_unit_%s.log" % C.DRONE_ID)

    kill_host("rooster_frame_dir_publisher")
    spawn(C.FRAME_CAPTURE_CMD, "/tmp/rooster_frame_capture.log")

    kill_host("rooster_depth_processor")
    spawn(C.DEPTH_CMD, "/tmp/rooster_depth_processor.log")
    time.sleep(10)


def start_twist_adapter():
    """Start the Twist -> cmd_nav adapter. ONLY once the aircraft is hovering."""
    kill_host("run_twist_control_adapter")
    kill_host("rooster_twist_control_adapter")
    spawn(C.TWIST_ADAPTER_CMD, "/tmp/rooster_twist_control_%s.log" % C.DRONE_ID)
    time.sleep(3)


def stop_twist_adapter():
    """Stop the adapter so its stop-watchdog cannot cancel a takeoff or land."""
    kill_host("run_twist_control_adapter")
    kill_host("rooster_twist_control_adapter")
    time.sleep(1)


# ── health ───────────────────────────────────────────────────────────────
def wait_for(predicate, timeout_s, what, poll_s=2.0):
    """Poll a predicate until it is true or the timeout expires."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def rosnode_exists(name):
    """Whether a ROS1 node is registered with falcon's master."""
    r = sh("docker exec %s bash -lc %s" % (
        C.FALCON_CONTAINER,
        shlex.quote(C.FALCON_ENV + "timeout 8 rosnode list 2>/dev/null")), 25)
    return name in r.stdout.split()


def ros1_publisher_count(topic):
    """How many ROS1 publishers a topic has, or -1 if it cannot be read."""
    r = sh("docker exec %s bash -lc %s" % (
        C.FALCON_CONTAINER,
        shlex.quote(C.FALCON_ENV + "timeout 8 rostopic info %s 2>/dev/null" % topic)), 25)
    if r.returncode != 0 or "Publishers" not in r.stdout:
        return -1
    block = r.stdout.split("Publishers:", 1)[1].split("Subscribers:", 1)[0]
    if "None" in block:
        return 0
    return sum(1 for line in block.splitlines() if line.strip().startswith("*"))


def ros2_publisher_count(topic):
    """How many ROS2 publishers a topic has, or -1 if it cannot be read."""
    r = sh("docker exec %s bash -lc %s" % (
        C.IT_CONTAINER,
        shlex.quote(C.IT_ENV + "timeout 8 ros2 topic info %s 2>/dev/null" % topic)), 25)
    for line in r.stdout.splitlines():
        if "Publisher count" in line:
            try:
                return int(line.split(":")[1])
            except (IndexError, ValueError):
                return -1
    return -1


def ros2_publisher_names(topic):
    """Node names publishing a ROS2 topic, as a list (may contain duplicates).

    Counting publishers is not enough to detect the failure that matters. The
    vendor's own ``rooster_manager`` legitimately publishes on
    ``/R1/manual_control`` alongside ours, so the healthy count is two -- while a
    stray ``position_fly_controller`` also makes it two and silently fights us
    for the throttle axis. Only the names distinguish them.
    """
    r = sh("docker exec %s bash -lc %s" % (
        C.IT_CONTAINER,
        shlex.quote(C.IT_ENV +
                    "timeout 10 ros2 topic info %s --verbose 2>/dev/null" % topic)), 30)
    names = []
    for line in r.stdout.splitlines():
        if line.strip().startswith("Node name:"):
            names.append(line.split(":", 1)[1].strip())
    return names


#: Node names allowed to publish /<drone>/manual_control. Anything else is a
#: second controller flying the same aircraft.
_ALLOWED_MANUAL_PUBLISHERS = {
    "rooster_command_unit",              # ours, the sole legitimate commander
    "rooster_manager",                   # the vendor's own relay
    "_CREATED_BY_BARE_DDS_APP_",         # how CycloneDDS reports a non-rclpy peer
}


def manual_control_authority():
    """Check that exactly one of *our* commanders owns the throttle axis.

    Returns:
        ``(ok, detail)``. Not ok means either nothing of ours is publishing, or
        something unrecognised is -- both of which invalidate a flight.
    """
    names = ros2_publisher_names(C.ROS2_TOPICS["manual"])
    if not names:
        return False, "could not read publishers on %s" % C.ROS2_TOPICS["manual"]
    ours = [n for n in names if n == "rooster_command_unit"]
    strangers = [n for n in names if n not in _ALLOWED_MANUAL_PUBLISHERS]
    if len(ours) != 1:
        return False, "expected exactly 1 rooster_command_unit, found %d (%s)" % (
            len(ours), ", ".join(names))
    if strangers:
        return False, "unrecognised publisher(s) on the throttle axis: %s" % (
            ", ".join(sorted(set(strangers))))
    return True, "rooster_command_unit + %d expected peer(s)" % (len(names) - 1)


def frames_fresh(max_age_s=10):
    """Whether new camera frames are landing AND their content is changing.

    Both halves matter: ``video_trigger.py`` keeps writing new filenames forever
    after ``R1`` is recreated while the image bytes stay frozen on the last
    frame it decoded, and that frozen frame gets fused into the map as real
    geometry.
    """
    r = sh("ls -t /tmp/rooster_frames/*.jpg 2>/dev/null | head -2", 15)
    files = [f for f in r.stdout.split() if f]
    if len(files) < 2:
        return False
    age = sh("echo $(( $(date +%%s) - $(stat -c %%Y %s) ))" % shlex.quote(files[0]), 15)
    try:
        if int(age.stdout.strip()) > max_age_s:
            return False
    except ValueError:
        return False
    sums = sh("md5sum %s %s | awk '{print $1}'"
              % (shlex.quote(files[0]), shlex.quote(files[1])), 20)
    hashes = sums.stdout.split()
    return len(hashes) == 2 and hashes[0] != hashes[1]


def start_video_watchdog():
    """Run the frame-freshness watchdog for the whole campaign, once."""
    r = sh("pgrep -f video_freshness_watchdog >/dev/null && echo up || echo down", 15)
    if "up" in r.stdout:
        return
    spawn("bash %s/sparx_agency/robots/ROBOTICAN/video_freshness_watchdog.sh"
          % C.REPO_ROOT, "/tmp/video_freshness_watchdog.log")


def health_report():
    """Snapshot every check the campaign cares about, as a dict.

    Returns:
        A dict of check name -> value. ``ok`` is True only when nothing that
        would invalidate a flight is wrong.
    """
    report = {
        "drone_image": None,
        "battery": battery_fraction(),
        "falcon_up": container_up(C.FALCON_CONTAINER),
        "bridge_up": container_up(C.BRIDGE_CONTAINER),
        "exploration_node": rosnode_exists("/exploration_node"),
        "frames_fresh": frames_fresh(),
        # Advisory only: the active follower and lost_localization both
        # legitimately advertise cmd_vel_raw (the latter never publishes on the
        # Rooster stack -- it watches an XTEND pose topic that is never fed).
        "cmd_vel_raw_publishers": ros1_publisher_count(C.ROS1_TOPICS["cmd_vel_raw"]),
    }
    report["manual_authority_ok"], report["manual_authority"] = manual_control_authority()
    try:
        report["drone_image"] = assert_simulator()
    except BringupError as exc:
        report["drone_image"] = "ERROR: %s" % exc

    report["ok"] = bool(
        report["falcon_up"] and report["bridge_up"] and report["exploration_node"]
        and report["frames_fresh"] and report["manual_authority_ok"]
        and str(report["drone_image"]).startswith(C.SIM_IMAGE_PREFIX))
    return report


def full_bringup(follower=None, extra="", min_battery=0.30):
    """Everything, in order, ending with a verified-healthy stack on the ground.

    Deliberately stops short of arming: the twist adapter is not started and
    nothing is commanded. :mod:`campaign` owns the flight itself.

    Returns:
        The final :func:`health_report`.

    Raises:
        BringupError: If a step's health check fails irrecoverably.
    """
    stop_twist_adapter()
    if not ensure_sphera(min_battery):
        raise BringupError("Sphera restart / GUI re-entry failed; no fresh R1")
    assert_simulator()
    start_containers()
    start_rooster_nodes()
    start_falcon(follower, extra)
    start_bridge()
    start_video_watchdog()
    if not wait_for(frames_fresh, 90, "fresh camera frames"):
        raise BringupError("camera frames never went fresh -- video_trigger is stuck")
    return health_report()


if __name__ == "__main__":
    print(json.dumps(full_bringup(), indent=2, default=str))
