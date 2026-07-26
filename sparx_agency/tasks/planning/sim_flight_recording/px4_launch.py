"""Launch and tear down a PX4 SITL instance for Pegasus to talk to.

Pegasus's own ``PX4LaunchTool`` cannot be used for this PX4 version: it runs
PX4 with ``cwd`` set to a fresh empty ``tempfile.TemporaryDirectory()``, but
PX4 sources ``$PWD/etc/init.d/rc.vehicle_setup`` at boot, and that only exists
under the real build output (``build/px4_sitl_default``). Every run failed with
``rc.vehicle_setup: No such file`` until PX4 was launched with the right
working directory -- which is what this module does. Vehicles are therefore
always constructed with ``px4_autolaunch=False``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PX4_VEHICLE_MODEL = "gazebo-classic_iris"  # must match PegasusIrisVehicle's backend config

# PX4 leaves these behind when killed abruptly. A stale one makes the next PX4
# instance exit immediately ("PX4 Exiting...") with no other explanation.
_LOCK_FILES = ("/tmp/px4_lock-0", "/tmp/px4-sock-0")


def clear_stale_locks() -> None:
    """Remove lock/socket files left by an abruptly-killed previous PX4."""
    for path in _LOCK_FILES:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def clear_saved_parameters(px4_dir: Path) -> None:
    """Delete PX4's persisted parameter store. **Recovery tool, not routine.**

    PX4 SITL saves every parameter set over MAVLink and reloads it on the next
    boot, so a run that experiments with (say) the EKF aiding source silently
    leaves those settings in place for every later run. Enabling ``--vision``
    once made *every* subsequent flight fail pre-flight with ``ekf2 missing
    data``, long after the flag was dropped, because ``EKF2_GPS_CTRL=0``
    persisted. Use this to recover from that.

    The file lives in PX4's **working directory** --
    ``build/px4_sitl_default/parameters.bson`` -- not under ``rootfs/``, which
    also contains a stale copy from a differently-configured run. PX4 says
    which one it read: ``INFO [param] importing from 'parameters.bson'``.

    Not called on every launch, so a deliberate parameter change survives if
    you want it to.
    """
    working_dir = px4_dir / "build" / "px4_sitl_default"
    for name in ("parameters.bson", "parameters_backup.bson"):
        try:
            (working_dir / name).unlink()
        except FileNotFoundError:
            pass


def launch_px4(px4_dir: Path, instance: int = 0) -> subprocess.Popen:
    """Start PX4 SITL from its build output directory.

    Args:
        px4_dir: A ``PX4-Autopilot`` checkout built with
            ``make px4_sitl_default none``.
        instance: PX4 instance id. Instance 0 opens the companion link on UDP
            14540, which is what :class:`px4_offboard.PX4Offboard` binds.

    Returns:
        The running process, to be passed to :func:`terminate_px4`.

    Raises:
        FileNotFoundError: If the PX4 binary has not been built.
    """
    rootfs_dir = px4_dir / "build" / "px4_sitl_default"  # holds etc/, created by the build
    binary = rootfs_dir / "bin" / "px4"
    if not binary.exists():
        raise FileNotFoundError(
            f"PX4 SITL binary not found at {binary} -- build it with "
            f"'make px4_sitl_default none' (see robots/PEGASUS/setup/install.sh)"
        )

    clear_stale_locks()
    env = dict(os.environ, PX4_SIM_MODEL=PX4_VEHICLE_MODEL)
    romfs = px4_dir / "ROMFS" / "px4fmu_common"
    return subprocess.Popen(
        [str(binary), str(romfs) + "/", "-s", str(romfs / "init.d-posix" / "rcS"),
         "-i", str(instance), "-d"],
        cwd=str(rootfs_dir), env=env,
    )


def terminate_px4(process: subprocess.Popen, timeout: float = 5.0) -> None:
    """Stop PX4 and release its lock files."""
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
    clear_stale_locks()
