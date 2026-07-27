"""Get a collection worker from "nothing running" to "ready to fly".

The bring-up sequence for one worker, kept out of :mod:`collect` because it is
all order-sensitive detail and none of it is the campaign's logic. Two parts:

* :class:`WorldMap` -- the surveyed map plus everything derived from it that is
  expensive and constant (the traversable region, the planner and its cost
  cache). Built before the simulator starts, so a missing or unusable map fails
  in a second rather than after a multi-minute Kit boot.
* :func:`bring_up` / :func:`configure_px4` -- the simulator and autopilot, in the
  one order that works.

The ordering constraints, all learned the hard way and none of them obvious:

1. **PX4 starts before the scene.** It dials into the simulator's HIL port as a
   TCP client and retries until the vehicle's backend is listening, so starting
   it first only costs it a few reconnects. The reverse -- a vehicle waiting for
   a PX4 that has not been started -- has no such recovery.
2. **The chase camera is aimed after ``world.reset()``.** A reset puts the
   streamed viewport back on Kit's default camera, so aiming it earlier looks
   like it worked and shows an empty room.
3. **Parameters are pushed after the first heartbeat.** PX4 rejects parameter
   traffic until it has booted, silently, and the only symptom is a setting that
   never applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.mission import (
    largest_region, sample_start_goal, snap_to_region,
)
from sparx_agency.tasks.planning.sim_flight_recording import flight_session
from sparx_agency.tasks.planning.sim_flight_recording.episode_plan import (
    EpisodeSpec, make_planner,
)


@dataclass
class WorldMap:
    """A surveyed scene, ready to plan against.

    Two regions, and the difference between them is what keeps a campaign
    alive. ``region`` is where the aircraft may *fly*: clear at cruise altitude.
    ``landing_region`` is the subset it may also be *put down* on: clear all the
    way to the floor. A goal drawn from the first but not the second lands the
    aircraft on a desk, where it tips, and every subsequent episode is refused
    with "Preflight Fail: Attitude failure (roll)".

    Attributes:
        grid: The surveyed occupancy grid.
        region: The largest connected block of clear space at cruise altitude.
        landing_region: The part of ``region`` that is also clear to the floor.
        planner: A weighted A* planner, shared across the campaign so its cost
            map is built once rather than per episode.
        metadata: The survey's own provenance.
        spec: The episode parameters the regions were computed for.
    """

    grid: OccupancyGrid2D
    region: np.ndarray
    landing_region: np.ndarray
    planner: object
    metadata: dict
    spec: EpisodeSpec

    @property
    def summary(self) -> str:
        """One line describing the usable airspace."""
        area = self.grid.resolution ** 2
        return (f"{self.metadata.get('scene')} at "
                f"{self.metadata.get('altitude_m')} m -- {self.grid.width}x"
                f"{self.grid.height} cells @ {self.grid.resolution} m, "
                f"{int(self.region.sum() * area)} m^2 flyable of which "
                f"{int(self.landing_region.sum() * area)} m^2 landable, at "
                f"{self.spec.clearance_m:.2f} m clearance")

    def random_pose(self, rng: np.random.Generator) -> Pose2D:
        """A random landable pose to spawn at, facing a random direction."""
        mission = sample_start_goal(
            self.grid, rng, clearance_m=self.spec.clearance_m,
            min_separation_m=self.spec.min_separation_m,
            max_separation_m=self.spec.max_separation_m,
            start_yaw_jitter_rad=np.pi, region=self.landing_region,
        )
        return mission.start

    def snap(self, x: float, y: float, yaw: float = 0.0) -> Pose2D:
        """Pull a real vehicle position onto the nearest cell a route can start from."""
        snapped = snap_to_region(self.grid, self.region, x, y)
        return Pose2D(snapped.x, snapped.y, yaw)


def load_map(scene: str, altitude_m: float, spec: EpisodeSpec,
             map_dir: Optional[Path] = None) -> WorldMap:
    """Read a surveyed map and derive everything a campaign needs from it.

    Args:
        scene: Scene key.
        altitude_m: Cruise altitude. Must match a surveyed map.
        spec: Episode parameters.
        map_dir: Override where maps are read from.

    Returns:
        The :class:`WorldMap`.

    Raises:
        FileNotFoundError: If the scene has not been surveyed at this altitude.
            The message names the command that would produce it.
        ValueError: If no part of the map has the requested clearance, or no
            part of it can be landed on.
    """
    from sparx_agency.robots.PEGASUS.adapters.scene_map import LANDABLE_LAYER, load_scene_map

    grid, metadata, layers = load_scene_map(scene, altitude_m, map_dir)
    region = largest_region(grid, spec.clearance_m)

    landable = layers.get(LANDABLE_LAYER)
    if landable is None:
        # A map surveyed before landability was measured. Fall back to treating
        # the whole flyable region as landable and say so, rather than silently
        # flying a campaign that will end on a desk.
        print("WARNING: this map has no 'landable' layer, so goals may be chosen "
              "on top of furniture. Re-run survey_scene.py to fix it.", flush=True)
        landing_region = region
    else:
        landing_region = region & landable
        if not landing_region.any():
            raise ValueError(
                f"no cell in {scene!r} is both flyable at {altitude_m:.2f} m and "
                f"clear to the floor -- nothing could take off or land"
            )

    return WorldMap(grid=grid, region=region, landing_region=landing_region,
                    planner=make_planner(spec), metadata=metadata, spec=spec)


def bring_up(simulation_app, args, spawn: Pose2D, heartbeat_timeout_s: float) -> Tuple:
    """Build the world, spawn the aircraft, and wait for PX4 to boot.

    Args:
        simulation_app: The app :func:`flight_session.boot_isaac` returned.
        args: The parsed :mod:`collect` arguments.
        spawn: Where and which way to spawn the aircraft.
        heartbeat_timeout_s: Simulated seconds to wait for PX4's first heartbeat.

    Returns:
        ``(loop, adapter, px4, chase_camera)``. ``chase_camera`` is None when
        nothing needs it.

    Raises:
        RuntimeError: If PX4 never answered -- which means it did not boot, and
            no flight is possible.
    """
    from sparx_agency.robots.PEGASUS.adapters.scene import SPAWN_HEIGHT_M
    from sparx_agency.tasks.planning.sim_flight_recording.px4_offboard import PX4Offboard
    from sparx_agency.tasks.planning.sim_flight_recording.sim_loop import SimLoop
    from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import (
        parse_resolution,
    )

    resolution = parse_resolution(args.resolution) if args.resolution else None
    world = flight_session.build_world(simulation_app, args.scene)
    adapter = flight_session.spawn_vehicle(
        simulation_app, args.pegasus_root,
        (spawn.x, spawn.y, SPAWN_HEIGHT_M), spawn_yaw=spawn.yaw,
        use_px4=True, px4_dir=args.px4_dir, vehicle_id=args.worker,
        resolution=resolution, camera_rate_hz=args.rate_hz,
    )

    chase_camera = None
    if args.video or args.stream:
        from sparx_agency.tasks.planning.sim_flight_recording.chase_camera import (
            make_chase_camera,
        )
        chase_camera = make_chase_camera()

    world.reset()
    flight_session.verify_timestep(world)
    if chase_camera is not None and args.stream:
        _aim_live_viewport()

    px4 = PX4Offboard(instance=args.worker)
    loop = SimLoop(world, adapter.vehicle, px4, rate_hz=args.rate_hz,
                   realtime=args.realtime)
    loop.start()

    _wait_for_heartbeat(loop, px4, heartbeat_timeout_s)
    loop.warmup_camera()
    if args.stream:
        print("STREAMING_READY -- connect the Isaac Sim WebRTC Streaming Client to "
              "this machine, port 49100", flush=True)
    return loop, adapter, px4, chase_camera


def _wait_for_heartbeat(loop, px4, timeout_s: float) -> float:
    """Step the sim, feeding PX4 sensor data, until it answers with a heartbeat.

    Returns:
        The simulated time the heartbeat arrived at.

    Raises:
        RuntimeError: If PX4 never responded.
    """
    started = loop.sim_time
    while loop.sim_time - started < timeout_s:
        loop.step()
        if px4.heartbeat_seen:
            print(f"PX4 heartbeat after {loop.sim_time - started:.1f} s of "
                  f"simulated time", flush=True)
            return loop.sim_time
    raise RuntimeError(
        f"no PX4 heartbeat after {timeout_s:.0f} simulated seconds -- PX4 SITL did "
        f"not boot. Check the px4_worker*.log next to the recordings."
    )


def configure_px4(loop, px4, params: dict, settle_s: float) -> None:
    """Push the simulation's parameter set and confirm PX4 took it.

    PX4 echoes a ``PARAM_VALUE`` for every parameter it accepts and says nothing
    at all about one it rejects (a type mismatch, a name that does not exist in
    this build), so the acknowledgements are the only way to find out from the
    companion side. A rejection is reported rather than raised: most of these
    are comfort settings, and a campaign that refuses to start because one
    parameter name moved between PX4 versions would be worse than one that says
    so and flies.

    Args:
        loop: The simulation loop to step while PX4 answers.
        px4: The autopilot link.
        params: Parameter name to value.
        settle_s: Simulated seconds to wait for the acknowledgements.
    """
    px4.set_params(params)
    px4.request_data_streams()
    loop.run_for(settle_s)

    missing = sorted(set(params) - px4.acknowledged_params)
    print(f"PX4 accepted {len(params) - len(missing)}/{len(params)} parameters",
          flush=True)
    if missing:
        print(f"WARNING: PX4 did not acknowledge {', '.join(missing)} -- they are "
              f"not applied. Check the PX4 log for 'param types mismatch'.", flush=True)


def settle_estimator(loop, px4, seconds: float) -> None:
    """Sit still until PX4's estimator has converged, before flying anything.

    EKF2 needs time on the ground with the aircraft stationary before its
    covariances settle, and the first flight of a campaign is otherwise flown on
    a half-converged solution. Measured: with no settle the first episode spent
    100 seconds failing to reach three waypoints while every later one took 20;
    the difference was entirely how long the estimator had been running.

    Args:
        loop: The simulation loop to step.
        px4: The autopilot link, polled for the drift readout.
        seconds: Simulated seconds to wait.
    """
    if seconds <= 0.0:
        return
    print(f"letting PX4's estimator settle for {seconds:.0f} simulated seconds...",
          flush=True)
    loop.run_for(seconds)
    position = loop.vehicle.state.position
    offset = px4.measure_frame_offset(position)
    estimated_yaw = px4.estimated_yaw_enu
    heading = ("unknown" if estimated_yaw is None
               else f"{math.degrees(estimated_yaw):+.1f} deg")
    print(f"estimator settled; PX4's frame is offset from the world by "
          f"({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}) m and it believes "
          f"it is heading {heading}", flush=True)


def wait_until_armable(loop, px4, timeout_s: float = 240.0, retry_s: float = 2.0,
                       verbose: bool = True) -> bool:
    """Arm once and disarm again, before any episode is planned.

    PX4 will not arm for a while after boot, and how long is neither documented
    nor constant -- measured here, two consecutive 60-second attempts were
    refused with no stated reason and the third succeeded, so the aircraft
    needed about 150 simulated seconds of sitting still. Nothing in the
    parameter set shortens it.

    Absorbing that here rather than inside the first episode matters for two
    reasons: those attempts would otherwise be recorded as failed episodes, and
    three failures in a row is what stops a worker -- so a campaign could be
    killed by its own warm-up before it ever flew.

    Args:
        loop: The simulation loop to step.
        px4: The autopilot link.
        timeout_s: Simulated seconds to keep trying.
        retry_s: How often to re-request arming.
        verbose: Announce the outcome.

    Returns:
        True once PX4 has armed (and been disarmed again), False on timeout --
        which the caller should treat as a broken worker, not a bad episode.
    """
    started = loop.sim_time
    last_request = -retry_s
    while not px4.armed:
        if loop.sim_time - started > timeout_s:
            if verbose:
                print(f"WARNING: PX4 would not arm within {timeout_s:.0f} simulated "
                      f"seconds of warm-up; PX4 said: "
                      f"{'; '.join(px4.drain_status_texts()[-4:]) or '(nothing)'}",
                      flush=True)
            return False
        px4.send_velocity_world(0.0, 0.0, 0.0, 0.0)
        if loop.sim_time - last_request >= retry_s:
            px4.set_offboard_mode()
            px4.arm()
            last_request = loop.sim_time
        loop.step()

    px4.disarm()
    loop.run_for(3.0)
    if verbose:
        print(f"PX4 armed for the first time after {loop.sim_time - started:.0f} "
              f"simulated seconds of warm-up; disarmed and ready to fly", flush=True)
    return True


def _aim_live_viewport(camera_path: str = "/World/ChaseCamera") -> None:
    """Point the streamed viewport at the chase camera.

    Without this a live viewer sees Kit's unaimed default camera rather than the
    aircraft. Must run *after* ``world.reset()``, which resets the viewport back
    to its default camera. ``get_active_viewport()`` can return nothing useful
    in a ``--no-window`` headless app, so the primary viewport is looked up by
    its known window name first.
    """
    from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name

    viewport = get_viewport_from_window_name("Viewport") or get_active_viewport()
    if viewport is None:
        print("WARNING: no viewport found to point at the chase camera -- the live "
              "stream will show the default unaimed camera", flush=True)
        return
    viewport.camera_path = camera_path
