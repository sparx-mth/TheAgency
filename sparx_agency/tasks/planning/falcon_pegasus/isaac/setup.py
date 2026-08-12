"""Get one aircraft from "nothing running" to "ready to hand over to FALCON".

The order below is not stylistic. Every step of it was learned by something
failing silently, and the reasoning is recorded in
``sim_flight_recording/campaign_setup.py``, whose public helpers this reuses
rather than reimplements. What is different here, and why this is not simply a
call to ``campaign_setup.bring_up``:

* **The camera is a different camera.** A collection flight renders the XTEND's
  calibration so its frames are interchangeable with real ones. An exploration
  flight renders FALCON's own reference sensor instead, because FALCON's
  frontier model assumes a symmetric field of view about the body boresight and
  the XTEND crop is not one -- see
  ``robots/PEGASUS/config/camera_falcon_explorer_640x480.yaml``.
* **There is no route to plan**, so no surveyed map has to be loaded. FALCON
  builds its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from sparx_agency.robots.PEGASUS.adapters.vehicle import camera_intrinsics
from sparx_agency.tasks.planning.falcon_pegasus.isaac import px4_exploration_params
from sparx_agency.tasks.planning.sim_flight_recording import campaign_setup, flight_session
from sparx_agency.tasks.planning.sim_flight_recording.px4_offboard import PX4Offboard
from sparx_agency.tasks.planning.sim_flight_recording.sim_loop import SimLoop

HEARTBEAT_TIMEOUT_S = 120.0
PARAM_SETTLE_S = 3.0
ARMABLE_TIMEOUT_S = 420.0
"""Simulated seconds to let PX4 become armable. See :func:`bring_up`."""


def bring_up(simulation_app, scene: str, pegasus_root: Path, px4_dir: Path,
             spawn_xyz, spawn_yaw: float, camera_config: str, rate_hz: float,
             worker: int = 0, want_chase_camera: bool = False,
             settle_s: float = 30.0, control_mode: str = "attitude"):
    """Build the world, spawn the aircraft, and get PX4 to the point of arming.

    Args:
        simulation_app: The app :func:`flight_session.boot_isaac` returned.
        scene: Isaac indoor scene key.
        pegasus_root: The patched ``pegasus.simulator`` extension.
        px4_dir: A built ``PX4-Autopilot`` checkout.
        spawn_xyz: Where to put the aircraft, world frame.
        spawn_yaw: Its initial heading, radians.
        camera_config: Which ``robots/PEGASUS/config/*.yaml`` to render with.
        rate_hz: Render (and therefore depth) rate.
        worker: PX4 instance id. Selects every port and lock file.
        want_chase_camera: Create the external camera used for the flight video.
        settle_s: Simulated seconds to let PX4's estimator converge before the
            first arming attempt.
        control_mode: Which cut into PX4 the flight will use. It selects the
            parameter set, because it decides which of PX4's control loops are
            in the chain at all.

    Returns:
        ``(loop, adapter, px4, chase_camera)``; ``chase_camera`` is None unless
        it was asked for.

    Raises:
        RuntimeError: If PX4 never sent a heartbeat (it did not boot) or would
            not arm during the warm-up (a broken aircraft, not a bad run).
    """
    intrinsics = camera_intrinsics(name=camera_config)
    world = flight_session.build_world(simulation_app, scene)
    adapter = flight_session.spawn_vehicle(
        simulation_app, pegasus_root, spawn_xyz, spawn_yaw=spawn_yaw,
        use_px4=True, px4_dir=px4_dir, vehicle_id=worker,
        intrinsics=intrinsics, camera_rate_hz=rate_hz)

    chase_camera = None
    if want_chase_camera:
        from sparx_agency.tasks.planning.sim_flight_recording.chase_camera import (
            make_chase_camera,
        )
        chase_camera = make_chase_camera()

    world.reset()
    flight_session.verify_timestep(world)

    px4 = PX4Offboard(instance=worker)
    # The warm-up runs FREE, the flight runs PACED, and both halves of that
    # matter.
    #
    # Paced flight is not a preference: FALCON walks its B-spline at
    # `ros::Time::now() - start_time` on a 100 Hz timer, on the wall clock, and
    # cannot be told about simulated time -- its exploration node aborts outright
    # on `use_sim_time`. A simulation running at any rate other than 1x therefore
    # slides the commanded position along the trajectory at a speed the airframe
    # is not flying at, and the outer loop saturates chasing a reference that is
    # not where the planner believes it is.
    #
    # But PX4's warm-up is several minutes of *simulated* seconds in which
    # nothing is being planned and nothing is watching, so pacing them only
    # spends real time. The switch happens once, below, right before the flight.
    loop = SimLoop(world, adapter.vehicle, px4, rate_hz=rate_hz, realtime=False)
    loop.start()

    _wait_for_heartbeat(loop, px4, HEARTBEAT_TIMEOUT_S)
    loop.warmup_camera()
    # The parameter set depends on where the flight will cut into PX4: an
    # attitude cut bypasses every MPC_* gain and leans on the MC_* ones instead.
    campaign_setup.configure_px4(loop, px4,
                                 px4_exploration_params.all_params(control_mode),
                                 PARAM_SETTLE_S)
    campaign_setup.settle_estimator(loop, px4, settle_s)
    # PX4 refuses to arm for a while after boot, and how long is neither
    # documented nor constant -- 150 to 180 simulated seconds is typical and one
    # measured run wanted more than 240, which is upstream's default ceiling and
    # is why this passes its own.
    _wait_until_armable(loop, px4, adapter)

    loop.set_realtime(True)
    return loop, adapter, px4, chase_camera


def _wait_until_armable(loop, px4, adapter, timeout_s: float = ARMABLE_TIMEOUT_S,
                        retry_s: float = 2.0, report_s: float = 20.0) -> None:
    """Arm once and disarm again, saying out loud what is happening while it waits.

    Same job as ``campaign_setup.wait_until_armable`` -- PX4 refuses to arm for
    the first two or three minutes after boot and nothing shortens it -- but a
    campaign that runs unattended needs the wait to be legible. Upstream's
    version reports only at the end, and its report is usually "PX4 said:
    (nothing)", because a pre-flight check that has not been *asked* about does
    not announce itself.

    So this prints, every ``report_s``: the aircraft's attitude and height
    (a multirotor resting on furniture is refused on attitude and looks
    identical to one that is merely still warming up), PX4's flight mode, and
    anything PX4 has said since the last line.

    Raises:
        RuntimeError: If PX4 never armed. That is a broken aircraft, not a bad
            run, so it stops the flight rather than being flown around.
    """
    started = loop.sim_time
    last_request = -retry_s
    last_report = loop.sim_time
    while not px4.armed:
        elapsed = loop.sim_time - started
        if elapsed > timeout_s:
            raise RuntimeError(
                "PX4 would not arm in %.0f simulated seconds. Last words: %s. The "
                "aircraft was at %s. Check px4.log for a refused pre-flight check."
                % (timeout_s, "; ".join(px4.drain_status_texts()[-4:]) or "(nothing)",
                   _attitude_report(adapter, px4)))
        px4.send_velocity_world(0.0, 0.0, 0.0, 0.0)
        if loop.sim_time - last_request >= retry_s:
            px4.set_offboard_mode()
            px4.arm()
            last_request = loop.sim_time
        loop.step()
        if loop.sim_time - last_report >= report_s:
            last_report = loop.sim_time
            said = "; ".join(px4.drain_status_texts()[-3:])
            print("  waiting for PX4 to arm (%3.0f s): %s%s"
                  % (elapsed, _attitude_report(adapter, px4),
                     ("  PX4: " + said) if said else ""), flush=True)

    px4.disarm()
    loop.run_for(3.0)
    print("PX4 armed for the first time after %.0f simulated seconds of warm-up; "
          "disarmed and ready to fly" % (loop.sim_time - started), flush=True)


def _attitude_report(adapter, px4) -> str:
    """One line of "is this aircraft in a state anything could arm from"."""
    from scipy.spatial.transform import Rotation

    state = adapter.vehicle.state
    qx, qy, qz, qw = state.attitude
    roll, pitch, _yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=True)
    estimate = "none" if px4.local_ned is None else (
        "(%.2f, %.2f, %.2f)" % px4.local_ned)
    return ("z=%.2f m roll=%+.1f pitch=%+.1f deg, PX4 mode %s, its local position %s"
            % (state.position[2], roll, pitch, px4.main_mode, estimate))


def _wait_for_heartbeat(loop, px4, timeout_s: float) -> float:
    """Step the simulation, feeding PX4 sensor data, until it answers.

    PX4 SITL is built with lockstep: its clock only advances when the simulator
    sends it sensor data, so this cannot be a blocking wait -- the loop that
    produces the answer is the loop that would be stopped.

    Returns:
        The simulated time the first heartbeat arrived.

    Raises:
        RuntimeError: If PX4 never answered.
    """
    started = loop.sim_time
    while loop.sim_time - started < timeout_s:
        loop.step()
        if px4.heartbeat_seen:
            print("PX4 heartbeat after %.1f s of simulated time"
                  % (loop.sim_time - started), flush=True)
            return loop.sim_time
    raise RuntimeError(
        "no PX4 heartbeat after %.0f simulated seconds -- PX4 SITL did not boot"
        % timeout_s)


def resolve_paths(dev_root: Path) -> Tuple[Path, Path]:
    """The Pegasus extension and PX4 checkout under a dev root.

    Args:
        dev_root: The directory ``robots/PEGASUS/setup/install.sh`` populated.

    Returns:
        ``(pegasus_root, px4_dir)``.

    Raises:
        FileNotFoundError: If either is missing, naming what to run to get it.
    """
    pegasus_root = Path(dev_root) / "PegasusSimulator" / "extensions" / "pegasus.simulator"
    px4_dir = Path(dev_root) / "PX4-Autopilot"
    for path, what in ((pegasus_root, "the patched Pegasus extension"),
                       (px4_dir, "the PX4-Autopilot checkout")):
        if not path.exists():
            raise FileNotFoundError(
                "%s is not at %s -- run robots/PEGASUS/setup/install.sh %s"
                % (what, path, dev_root))
    return pegasus_root, px4_dir


def load_run(path) -> dict:
    """Read one ``runs/*.yaml``.

    Args:
        path: Path to the run config.

    Returns:
        The parsed YAML.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If it has no ``run`` block, which means it is not one of
            these files.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("no run config at %s" % path)
    config = yaml.safe_load(path.read_text())
    if "run" not in config or "map_config" not in config:
        raise ValueError(
            "%s is not a falcon_pegasus run config: it must carry both a `run` "
            "block (for the aircraft) and a `map_config` block (for FALCON)" % path)
    return config


def find_run(name: str, runs_dir: Optional[Path] = None) -> Path:
    """Locate a run config by name, with or without its numeric prefix.

    Args:
        name: e.g. ``3_open_plan``, ``open_plan``, or a path to a YAML.
        runs_dir: Where the run configs live. Defaults to the package's own.

    Returns:
        The path to the config.

    Raises:
        FileNotFoundError: If nothing matches, listing what does exist.
    """
    candidate = Path(name)
    if candidate.suffix == ".yaml" and candidate.exists():
        return candidate
    runs_dir = Path(runs_dir) if runs_dir else Path(__file__).resolve().parent.parent / "runs"
    exact = runs_dir / ("%s.yaml" % name)
    if exact.exists():
        return exact
    matches = sorted(p for p in runs_dir.glob("*.yaml")
                     if p.stem == name or p.stem.split("_", 1)[-1] == name)
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(sorted(p.stem for p in runs_dir.glob("*.yaml")))
    raise FileNotFoundError(
        "no run config matching %r in %s. Available: %s" % (name, runs_dir, available))
