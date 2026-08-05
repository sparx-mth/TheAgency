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

BOOT_PARAM_SCRIPT = "px4-rc.params"
"""PX4's own hook for pre-boot parameters, and the only way to set a
``@reboot_required`` one.

``rcS`` runs ``. px4-rc.params`` -- resolved through ``PATH`` -- after the
airframe file has applied its defaults and *before* ``rc.vehicle_setup`` starts
``ekf2``. Upstream's copy in ``ROMFS/px4fmu_common/init.d-posix/`` is entirely
commented out and exists for exactly this purpose. ``rcS`` appends the ROMFS
directory to ``PATH``, so a copy in a directory placed *earlier* on ``PATH``
shadows it, which is how :func:`write_boot_parameters` keeps one per instance
without touching the PX4 checkout.
"""


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
    for name in ("parameters.bson", "parameters_backup.bson", BOOT_PARAM_SCRIPT):
        try:
            (directory / name).unlink()
        except FileNotFoundError:
            pass


def format_param_value(value) -> str:
    """Render one parameter value for a ``param set`` line.

    PX4's ``param set`` infers the type from how the value is written, exactly as
    ``PARAM_SET`` over MAVLink infers it from the declared type -- so an INT32
    parameter written ``1.0`` is refused, and a REAL32 one written ``0`` becomes
    an integer PX4 then rejects against the parameter's declared type. The
    int/float split in :mod:`px4_params` therefore has to survive being turned
    into text.

    Args:
        value: An ``int`` for a PX4 INT32 parameter, a ``float`` for a REAL32
            one. ``bool`` is refused: it is an ``int`` to Python but never what
            a PX4 parameter is, and accepting it would write ``True``.

    Returns:
        The value as PX4's shell should see it.

    Raises:
        TypeError: If ``value`` is not an int or a float.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"PX4 has no boolean parameter type; write {int(value)} (INT32) or "
            f"{float(value)} (REAL32) so the intent is explicit"
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if value != int(value) else f"{value:.1f}"
    raise TypeError(f"cannot write {value!r} ({type(value).__name__}) as a PX4 parameter")


def write_boot_parameters(px4_dir: Path, instance: int, params: dict) -> Optional[Path]:
    """Write one instance's pre-boot ``param set`` script.

    This is the channel for every ``@reboot_required`` parameter -- see
    :data:`BOOT_PARAM_SCRIPT` for why it lands in the right place, and
    :data:`px4_params.REBOOT_REQUIRED` for which ones need it.

    ``param set`` is used rather than ``param set-default`` deliberately: the
    airframe file has already run by this point and set its own defaults, and a
    default does not override a value that is already there.

    Args:
        px4_dir: A built ``PX4-Autopilot`` checkout.
        instance: PX4 instance id. The script is private to it, like every other
            file in its working directory.
        params: Name to value. Empty removes any script left by an earlier run
            rather than leaving it to apply silently -- the same trap as PX4's
            persisted ``parameters.bson``.

    Returns:
        The script's path, or None if there was nothing to write.

    Raises:
        TypeError: If a value is not an int or a float.
    """
    directory = working_dir(px4_dir, instance)
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / BOOT_PARAM_SCRIPT
    if not params:
        try:
            script.unlink()
        except FileNotFoundError:
            pass
        return None

    lines = [
        "#!/bin/sh",
        "# Generated by sim_flight_recording.px4_launch -- do not edit.",
        "# Sourced by PX4's rcS before rc.vehicle_setup starts ekf2, which is the",
        "# only point at which a @reboot_required parameter can be set.",
    ]
    lines += [f"param set {name} {format_param_value(value)}"
              for name, value in sorted(params.items())]
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return script


def launch_px4(px4_dir: Path, instance: int = 0,
               log_path: Optional[Path] = None,
               boot_params: Optional[dict] = None) -> subprocess.Popen:
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
        boot_params: Parameters to apply before ``ekf2`` starts, from
            :func:`px4_params.boot_params`. Written to
            :data:`BOOT_PARAM_SCRIPT` in the instance's working directory, which
            is prepended to ``PATH`` so PX4's ``rcS`` finds it there instead of
            the empty upstream copy. None or empty removes a stale script from a
            previous run, so an instance is never configured by a flag that is
            no longer set.

    Returns:
        The running process, to be passed to :func:`terminate_px4`.

    Raises:
        FileNotFoundError: If the PX4 binary has not been built.
        ValueError: If ``instance`` is out of range.
        TypeError: If a boot parameter's value is not an int or a float.
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
    write_boot_parameters(px4_dir, instance, boot_params or {})

    # cwd first on PATH so `. px4-rc.params` in rcS resolves to the script above
    # rather than the empty upstream one, which rcS appends to PATH.
    env = dict(os.environ, PX4_SIM_MODEL=PX4_VEHICLE_MODEL,
               PATH=f"{cwd}{os.pathsep}{os.environ.get('PATH', '')}")
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
