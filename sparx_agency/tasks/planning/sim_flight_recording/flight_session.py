"""Boot Isaac Sim, build the scene + vehicle, and record what the camera sees.

The parts every flight script needs and none of them should own: bringing up
the Kit app with the right extensions, loading an indoor scene, spawning the
Iris, warming the camera up, and streaming frames into a recording.

The flight *control* differs per script and is deliberately not here --
:mod:`fly_direct` applies forces itself, :mod:`episode` lets a PX4 autopilot do
it.

## The timestep, which used to be wrong and mattered enormously

Pegasus's PX4 world defaults to ``physics_dt = 1/250`` and ``rendering_dt =
1/60``. Isaac Sim turns that into ``substeps = int(rendering_dt / physics_dt) =
4``, which means **``world.step()`` advances a different amount of time
depending on whether it rendered**: 4 ms without a render, 16 ms with one. A
caller that steps the vehicle by one fixed ``dt`` is therefore wrong on every
step, in one of two different directions.

Everything downstream is derived from that ``dt``. PX4's lockstep clock is
integrated from it (``_current_utime += dt * 1e6``), so its clock ran about
twice as fast as the world. The simulated accelerometer is
``(v - v_prev) / dt``, so specific force alternated between 0.4x and 1.6x of
truth at 25 Hz -- a square wave straight into the attitude estimator, which is
what the earlier "compass/accelerometer-bias estimator divergence" almost
certainly was. And every recorded timestamp was a frame index over a nominal
rate that the simulation was not actually running at.

The fix is one line, :func:`build_scene`'s ``set_world_settings``: make
``rendering_dt`` equal ``physics_dt`` so ``substeps == 1`` and **every**
``world.step()`` advances exactly ``PHYSICS_DT``, rendered or not. Rendering
stays occasional -- the caller chooses when, via
:func:`render_every_n_steps` -- it just no longer changes how much time passes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# The MonocularCamera ignores its first ~100 render callbacks before producing
# a frame (see MonocularCamera.update), so the sim must be rendered at least
# that many times before capture_frame() can succeed.
CAMERA_WARMUP_RENDER_TICKS = 110
# Async USD reference loads need a few app updates to compose before prim
# queries (masses, bounding boxes) see the real geometry.
STAGE_SETTLE_STEPS = 20
PHYSICS_HZ = 250.0
PHYSICS_DT = 1.0 / PHYSICS_HZ
"""Physics timestep. Also the rate PX4's sensors and lockstep clock run at.

250 Hz is Pegasus's own default for the PX4 world and matches what PX4 SITL
expects from a simulator, so the airframe dynamics and the autopilot are both
being driven at the rate they were tuned for.
"""


def render_every_n_steps(rate_hz: float, physics_hz: float = PHYSICS_HZ) -> int:
    """How many physics steps to take between renders for a given capture rate.

    Args:
        rate_hz: Desired camera capture rate.
        physics_hz: Physics rate, see :data:`PHYSICS_HZ`.

    Returns:
        Steps per render, at least 1.

    Raises:
        ValueError: If ``rate_hz`` is not positive.
    """
    if rate_hz <= 0.0:
        raise ValueError(f"capture rate must be positive, got {rate_hz}")
    return max(int(round(physics_hz / rate_hz)), 1)


def boot_isaac(stream: bool = False):
    """Start a headless Kit app with the Pegasus (and optionally livestream) extensions.

    Args:
        stream: Enable ``omni.kit.livestream.app``, which binds
            ``0.0.0.0:49100`` for NVIDIA's Isaac Sim WebRTC Streaming Client.
            The container uses host networking, so that port is reachable as
            ``localhost:49100`` on the host with no port publishing. Only one
            process can bind it, so a multi-worker campaign leaves it off.

    Returns:
        The live ``SimulationApp``. Must be ``.close()``d by the caller.
    """
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("pegasus.simulator")
    simulation_app.update()
    if stream:
        enable_extension("omni.kit.livestream.app")
        simulation_app.update()
    return simulation_app


def add_collision_ground(z: float = 0.0, prim_path: str = "/World/CollisionGround") -> None:
    """Add an invisible collision plane under the scene.

    The stock environments' own floors do not reliably stop a rigid body: in
    ``simple_room`` the Iris falls straight through the floor and comes to rest
    upside down under it (``z=-0.72``, ``roll=180``), which PX4 refuses to arm
    on ("Preflight Fail: Attitude failure (roll)"). Their *walls* do collide --
    the raycast survey hits them -- so this only needs to supply the missing
    floor.

    It is invisible so it cannot z-fight with the scene's own textured floor or
    show up in the recorded camera images.

    Args:
        z: Height to place the plane at, metres.
        prim_path: Stage path for the plane.
    """
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.utils.prims import get_prim_at_path

    GroundPlane(prim_path=prim_path, name="collision_ground", z_position=z)
    get_prim_at_path(prim_path).GetAttribute("visibility").Set("invisible")


def build_world(simulation_app, scene: str, ground_plane: bool = True):
    """Bring up the physics world and load an indoor scene into it.

    Args:
        simulation_app: The app returned by :func:`boot_isaac`.
        scene: A key of
            :data:`~sparx_agency.robots.PEGASUS.adapters.scene.INDOOR_SCENES`.
        ground_plane: Add an invisible collision floor, see
            :func:`add_collision_ground`.

    Returns:
        The Pegasus world.
    """
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    from sparx_agency.robots.PEGASUS.adapters.scene import load_indoor_scene

    pg = PegasusInterface()
    # Equal timesteps => one physics substep per world.step(), so a step advances
    # exactly PHYSICS_DT whether or not it rendered. See this module's docstring;
    # this must happen before initialize_world(), which is what constructs World.
    pg.set_world_settings(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
    pg.initialize_world()
    world = pg.world

    load_indoor_scene(scene)
    if ground_plane:
        add_collision_ground()
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()
    return world


def spawn_vehicle(simulation_app, pegasus_root: Path, spawn_xyz, spawn_yaw: float = 0.0,
                  use_px4: bool = True, px4_dir=None, vehicle_id: int = 0,
                  resolution: Optional[Tuple[int, int]] = None,
                  camera_rate_hz: float = 10.0):
    """Spawn the Iris into an already-built world.

    Args:
        simulation_app: The app returned by :func:`boot_isaac`.
        pegasus_root: Path to the patched ``pegasus.simulator`` extension.
        spawn_xyz: World-frame ``(x, y, z)`` to spawn at. Must be free space.
        spawn_yaw: Initial heading, radians CCW from +X.
        use_px4: Attach a ``PX4MavlinkBackend`` to the vehicle.
        px4_dir: PX4-Autopilot checkout, required when ``use_px4`` is True.
            PX4 is never auto-launched (Pegasus's ``PX4LaunchTool`` uses a
            broken working directory) -- the caller launches it, see
            :mod:`px4_launch`.
        vehicle_id: PX4 instance this aircraft is bound to. Selects its HIL
            port; must match what PX4 was launched with.
        resolution: Camera ``(width, height)``. None uses the platform
            calibration's own resolution.
        camera_rate_hz: Camera capture rate.

    Returns:
        The :class:`PegasusIrisVehicle` adapter.
    """
    from sparx_agency.robots.PEGASUS.adapters.vehicle import PegasusIrisVehicle

    adapter = PegasusIrisVehicle(
        pegasus_extension_root=pegasus_root,
        init_pos=tuple(spawn_xyz),
        init_yaw=spawn_yaw,
        resolution=resolution,
        camera_frequency=camera_rate_hz,
        use_px4=use_px4,
        px4_dir=str(px4_dir) if px4_dir is not None else None,
        px4_autolaunch=False,
        vehicle_id=vehicle_id,
    )
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()
    return adapter


def build_scene(simulation_app, scene: str, pegasus_root: Path, spawn_xyz, use_px4: bool,
                px4_dir=None, ground_plane: bool = True, spawn_yaw: float = 0.0,
                vehicle_id: int = 0, resolution: Optional[Tuple[int, int]] = None,
                camera_rate_hz: float = 10.0):
    """Load an indoor scene and spawn the Iris in it.

    :func:`build_world` followed by :func:`spawn_vehicle`, for callers that do
    both at once. A campaign wants them separate: the world is built once and
    the aircraft may be respawned into it many times.

    Returns:
        ``(world, adapter)``.
    """
    world = build_world(simulation_app, scene, ground_plane=ground_plane)
    adapter = spawn_vehicle(
        simulation_app, pegasus_root, spawn_xyz, spawn_yaw=spawn_yaw,
        use_px4=use_px4, px4_dir=px4_dir, vehicle_id=vehicle_id,
        resolution=resolution, camera_rate_hz=camera_rate_hz,
    )
    return world, adapter


def verify_timestep(world, dt: float = PHYSICS_DT, tolerance: float = 1e-6) -> None:
    """Check that one ``world.step()`` really advances ``dt``, rendered or not.

    This is the assumption everything else rests on -- PX4's lockstep clock, the
    simulated accelerometer, every recorded timestamp -- and it is silently
    violated by the default world settings (see this module's docstring). It
    costs four steps to confirm, so it is confirmed rather than assumed.

    Args:
        world: The Pegasus world, already reset.
        dt: The timestep each step is expected to advance.
        tolerance: Allowed error, seconds.

    Raises:
        RuntimeError: If a step advances something other than ``dt``, or if a
            rendered step and an unrendered one advance different amounts.
    """
    def measure(render: bool) -> float:
        before = world.current_time
        world.step(render=render)
        return world.current_time - before

    quiet = [measure(False) for _ in range(2)]
    rendered = [measure(True) for _ in range(2)]
    measured = quiet + rendered
    if any(abs(step - dt) > tolerance for step in measured):
        raise RuntimeError(
            f"world.step() advances {measured} s, expected {dt} s each. Physics and "
            f"rendering timesteps must be equal for a step to be one physics step "
            f"(see flight_session's docstring); PegasusInterface.set_world_settings "
            f"is where that is configured."
        )
    print(f"timestep verified: every world.step() advances {dt * 1000:.1f} ms, "
          f"rendered or not", flush=True)


def warmup_camera(world, ticks: int = CAMERA_WARMUP_RENDER_TICKS) -> int:
    """Render until the onboard camera starts producing frames.

    Returns:
        The number of steps taken, so the caller can keep a continuous count.
    """
    for _ in range(ticks):
        world.step(render=True)
    return ticks


class FlightRecorder:
    """Streams onboard RGBD frames into a recording, and optionally an MP4.

    The on-disk output is the same schema real rosbag extractions produce (see
    ``tasks/planning/vlas/common/finetune/datasets/recording.py``), so a
    simulated flight is a drop-in ``data.recording`` source for VLA fine-tuning
    -- no parallel dataset format.

    Frames go to disk as they arrive rather than being buffered: a
    three-minute flight is over 2 GB of imagery, and a collection farm runs
    several aircraft at once.

    Args:
        adapter: The :class:`PegasusIrisVehicle` to capture from.
        out_dir: Directory to write the recording to.
        rate_hz: Capture rate, recorded in the output metadata and used to stamp
            any frame that arrives without a simulation time.
        camera_height_m: Height of the camera above the floor during cruise.
            The camera sits at the vehicle's body origin, so this is the
            flight altitude.
        depth_format: ``"png"`` (uint16 millimetres, ~4x smaller) or ``"npy"``
            (float32 metres).
        video_out: Optional MP4 path.
        video_source: ``"chase"`` writes the external chase camera's view,
            ``"onboard"`` the drone's own forward-facing camera.
        chase_camera: The chase camera, required for ``video_source="chase"``.
    """

    def __init__(self, adapter, out_dir: Path, rate_hz: float = 10.0,
                 camera_height_m: float = 1.5, depth_format: str = "png",
                 video_out: Path = None, video_source: str = "chase",
                 chase_camera=None):
        from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import (
            FlightWriter,
        )

        if video_out is not None and video_source == "chase" and chase_camera is None:
            raise ValueError("video_source='chase' needs a chase_camera")

        self._adapter = adapter
        self._video_source = video_source
        self._chase_camera = chase_camera
        self._writer = FlightWriter(
            Path(out_dir), adapter.intrinsics, rate_hz=rate_hz,
            camera_height_m=camera_height_m, pitch_deg=0.0,
            depth_format=depth_format,
        )
        self._video = None

        if video_out is not None:
            import cv2

            video_out = Path(video_out)
            video_out.parent.mkdir(parents=True, exist_ok=True)
            size = ((960, 540) if video_source == "chase"
                    else (adapter.intrinsics.width, adapter.intrinsics.height))
            self._video = cv2.VideoWriter(
                str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), rate_hz, size,
            )
            self._video_out = video_out
            print(f"VIDEO_RECORDING ({video_source}) -- writing to {video_out}", flush=True)

    @property
    def frames(self) -> int:
        """How many frames have been recorded so far."""
        return self._writer.frames

    def capture(self, stamp_s: float = None) -> None:
        """Grab one onboard frame (and one video frame, if recording video).

        Args:
            stamp_s: Simulation time of this frame.
        """
        frame = self._adapter.capture_frame(stamp_s=stamp_s)
        self._writer.append(frame)

        if self._chase_camera is not None:
            # Re-aim before the next render. It chases at a fixed world-frame
            # offset rather than sitting still, so it cannot clip into room
            # geometry: wherever the aircraft is, is by definition clear air.
            from sparx_agency.tasks.planning.sim_flight_recording.chase_camera import (
                aim_chase_camera,
            )
            aim_chase_camera(self._chase_camera, self._adapter.vehicle.state.position)

        if self._video is None:
            return
        rgb = frame.rgb if self._video_source == "onboard" else self._chase_camera.get_rgb()
        if rgb is not None:
            self._video.write(rgb[:, :, ::-1])  # RGB -> BGR for cv2

    def finish(self, extra_meta: dict = None) -> dict:
        """Close the video and write the recording's metadata.

        Args:
            extra_meta: Provenance to merge into ``meta.json`` (scene, seed,
                start, goal, outcome, ...).

        Returns:
            The stats dict.
        """
        if self._video is not None:
            self._video.release()
            print(f"wrote video to {self._video_out}", flush=True)

        stats = self._writer.close(extra_meta)
        print(f"wrote {stats['frames']} frames to {self._writer.out_dir}", flush=True)
        return stats
