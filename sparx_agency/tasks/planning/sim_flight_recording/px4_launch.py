"""Launch and tear down PX4 SITL instances, one per simulated aircraft.

Pegasus's own ``PX4LaunchTool`` cannot be used for this PX4 version: it runs
PX4 with ``cwd`` set to a fresh empty ``tempfile.TemporaryDirectory()``, but
PX4 sources ``$PWD/etc/init.d/rc.vehicle_setup`` at boot, and that only exists
under the real build output. Every run failed with ``rc.vehicle_setup: No such
file`` until PX4 was launched with the right working directory -- which is what
this module does. Vehicles are therefore always constructed with
``px4_autolaunch=False``.

**Every instance gets its own working directory**, and that is not optional.
PX4's storage paths are all relative to its cwd -- ``parameters.bson``,
``dataman``, ``log/`` -- and ``param_save_default`` writes the parameter file in
place with ``O_TRUNC`` under a *process-local* semaphore. Two instances sharing
one directory interleave truncate-and-write on the same file every time either
receives a ``PARAM_SET``, which is on every flight. They would silently corrupt
each other's configuration. This is the single thing that has to be right before
a collection farm can run more than one aircraft.

Per-instance identity, all derived from the ``-i`` argument:

===========================  ============================================
resource                     instance ``N``
===========================  ============================================
companion/offboard UDP       ``14540 + N`` (PX4 sends here)
simulator HIL TCP            ``4560 + N`` (Pegasus listens)
lock file                    ``/tmp/px4_lock-N``
command socket               ``/tmp/px4-sock-N``
working directory            ``build/px4_sitl_default/instance_N``
===========================  ============================================

PX4 clamps the offboard port to 14549 for instances 10 and above, so **10 is the
hard ceiling** on concurrent instances without patching PX4 itself.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

PX4_VEHICLE_MODEL = "gazebo-classic_iris"  # must match PegasusIrisVehicle's backend config
MAX_INSTANCES = 10
"""PX4's own ceiling: ``px4-rc.mavlink`` sends instances >= 10 to port 14549."""


def _check_instance(instance: int) -> None:
    if not 0 <= instance < MAX_INSTANCES:
        raise ValueError(
            f"PX4 instance must be in [0, {MAX_INSTANCES}), got {instance}. "
            f"PX4 sends every instance from {MAX_INSTANCES} up to the same "
            f"offboard port (see ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink), "
            f"so they could not be told apart."
        )


def offboard_port(instance: int) -> int:
    """UDP port PX4 instance ``instance`` streams to a companion computer on."""
    _check_instance(instance)
    return 14540 + instance


def hil_port(instance: int) -> int:
    """TCP port the simulator listens on for instance ``instance``'s HIL link."""
    _check_instance(instance)
    return 4560 + instance


def working_dir(px4_dir: Path, instance: int) -> Path:
    """Private storage directory for one PX4 instance.

    Args:
        px4_dir: A built ``PX4-Autopilot`` checkout.
        instance: PX4 instance id.

    Returns:
        The directory (not created).
    """
    return Path(px4_dir) / "build" / "px4_sitl_default" / f"instance_{instance}"


def clear_stale_locks(instance: int = 0) -> None:
    """Remove lock/socket files left by an abruptly-killed previous PX4.

    A stale one makes the next instance exit immediately (``PX4 Exiting...``)
    with no other explanation.
    """
    for path in (f"/tmp/px4_lock-{instance}", f"/tmp/px4-sock-{instance}"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def kill_stale_px4(px4_dir: Path, instance: int = 0) -> None:
    """Kill a PX4 daemon left behind by an earlier run of the same instance.

    :func:`launch_px4` starts PX4 with ``-d``, which daemonises it: the process
    :class:`subprocess.Popen` holds is the launcher, while the ``bin/px4`` that
    actually binds TCP ``4560 + N`` detaches and outlives it. Terminating the
    launcher therefore leaves the real PX4 running, and the *next* run on that
    instance boots into a port clash which PX4 reports only as its sensors going
    ``STALE!`` and never sending a heartbeat -- an hour of flights lost to a
    message that names neither the port nor the cause.

    Instances are told apart by **working directory**, which :func:`working_dir`
    guarantees is unique. Matching on ``-i N`` in a command line instead would
    also match a live sibling instance and kill another worker's aircraft.

    Args:
        px4_dir: A built ``PX4-Autopilot`` checkout.
        instance: PX4 instance id.
    """
    target = working_dir(px4_dir, instance)
    try:
        target = target.resolve()
    except OSError:
        return
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.joinpath("cwd").resolve() != target:
                continue
            command = entry.joinpath("cmdline").read_bytes()
        except OSError:          # the process exited, or is not ours to read
            continue
        if b"px4" not in command:
            continue
        try:
            os.kill(int(entry.name), signal.SIGKILL)
        except OSError:
            pass


def clear_saved_parameters(px4_dir: Path, instance: int = 0) -> None:
    """Delete one instance's persisted parameter store.

    PX4 saves every parameter it is sent over MAVLink and reloads it on the next
    boot, so a run that experiments with (say) the estimator's aiding source
    silently leaves those settings in place for every later run. Enabling
    ``--vision`` once made *every* subsequent flight fail pre-flight with
    ``ekf2 missing data``, long after the flag was dropped, because
    ``EKF2_GPS_CTRL=0`` persisted.

    A collection campaign calls this at startup so its configuration is exactly
    what :mod:`px4_params` says and nothing else.

    Args:
        px4_dir: A built ``PX4-Autopilot`` checkout.
        instance: PX4 instance id.
    """
    directory = working_dir(px4_dir, instance)
    for name in ("parameters.bson", "parameters_backup.bson"):
        try:
            (directory / name).unlink()
        except FileNotFoundError:
            pass


def launch_px4(px4_dir: Path, instance: int = 0,
               log_path: Optional[Path] = None) -> subprocess.Popen:
    """Start one PX4 SITL instance in its own working directory.

    Args:
        px4_dir: A ``PX4-Autopilot`` checkout built with
            ``make px4_sitl_default none``.
        instance: PX4 instance id, ``0 <= instance < 10``. Selects every port
            and lock file listed in this module's docstring.
        log_path: Redirect PX4's console output here. Strongly recommended when
            running several instances -- interleaved PX4 consoles are unreadable,
            and the pre-flight check that refused an arming is only ever visible
            in this output.

    Returns:
        The running process, to be passed to :func:`terminate_px4`.

    Raises:
        FileNotFoundError: If the PX4 binary has not been built.
        ValueError: If ``instance`` is out of range.
    """
    _check_instance(instance)
    px4_dir = Path(px4_dir)
    rootfs_dir = px4_dir / "build" / "px4_sitl_default"  # holds etc/, created by the build
    binary = rootfs_dir / "bin" / "px4"
    if not binary.exists():
        raise FileNotFoundError(
            f"PX4 SITL binary not found at {binary} -- build it with "
            f"'make px4_sitl_default none' (see robots/PEGASUS/setup/install.sh)"
        )

    # Before the locks, because a *live* orphan from the last run will simply
    # recreate them -- and it, not the lock file, is what holds the HIL port.
    kill_stale_px4(px4_dir, instance)
    clear_stale_locks(instance)
    cwd = working_dir(px4_dir, instance)
    cwd.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, PX4_SIM_MODEL=PX4_VEHICLE_MODEL)
    romfs = px4_dir / "ROMFS" / "px4fmu_common"
    command = [str(binary), str(romfs) + "/", "-s", str(romfs / "init.d-posix" / "rcS"),
               "-i", str(instance), "-d"]

    if log_path is None:
        process = subprocess.Popen(command, cwd=str(cwd), env=env)
        process._px4_dir = px4_dir  # so terminate_px4 can reap the daemon
        return process

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    process = subprocess.Popen(command, cwd=str(cwd), env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    process._px4_log_handle = handle  # keep it open as long as the process lives
    process._px4_dir = px4_dir
    return process


def terminate_px4(process: subprocess.Popen, instance: int = 0,
                  timeout: float = 5.0) -> None:
    """Stop a PX4 instance and release its lock files.

    Args:
        process: The process :func:`launch_px4` returned. None is a no-op.
        instance: The instance id it was launched with.
        timeout: Seconds to wait for a clean exit before killing it.
    """
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
    # PX4 runs with -d, so the process above was only the launcher; the daemon
    # holding TCP 4560+N is still alive and would poison the next run.
    px4_dir = getattr(process, "_px4_dir", None)
    if px4_dir is not None:
        kill_stale_px4(px4_dir, instance)
    handle = getattr(process, "_px4_log_handle", None)
    if handle is not None:
        handle.close()
    clear_stale_locks(instance)
