"""Boot Isaac Sim, build the scene + vehicle, and record a flight.

The parts every flight script needs and none of them should own: bringing up
the Kit app with the right extensions, loading an indoor scene, spawning the
Iris at a pose that scene is known to be flyable from, warming the camera up,
and collecting frames into a recording (plus an optional MP4).

The flight *control* differs per script and is deliberately not here --
:mod:`fly_direct` applies forces itself, :mod:`fly_px4` lets a PX4 autopilot do
it.
"""
from __future__ import annotations

from pathlib import Path

# The MonocularCamera ignores its first ~100 render callbacks before producing
# a frame (see MonocularCamera.update), so the sim must be rendered at least
# that many times before capture_frame() can succeed.
CAMERA_WARMUP_RENDER_TICKS = 110
# Async USD reference loads need a few app updates to compose before prim
# queries (masses, bounding boxes) see the real geometry.
STAGE_SETTLE_STEPS = 20
PHYSICS_DT = 1.0 / 100.0  # matches Pegasus's default SimulationCfg physics dt


def boot_isaac(stream: bool = False):
    """Start a headless Kit app with the Pegasus (and optionally livestream) extensions.

    Args:
        stream: Enable ``omni.kit.livestream.app``, which binds
            ``0.0.0.0:49100`` for NVIDIA's Isaac Sim WebRTC Streaming Client.
            The container uses host networking, so that port is reachable as
            ``localhost:49100`` on the host with no port publishing.

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


def build_scene(simulation_app, scene: str, pegasus_root: Path, spawn_xyz, use_px4: bool,
                px4_dir=None, ground_plane: bool = True):
    """Load an indoor scene and spawn the Iris in it.

    Args:
        simulation_app: The app returned by :func:`boot_isaac`.
        scene: A key of
            :data:`~sparx_agency.robots.PEGASUS.adapters.scene.INDOOR_SCENES`.
        pegasus_root: Path to the patched ``pegasus.simulator`` extension.
        spawn_xyz: World-frame ``(x, y, z)`` to spawn at. Must be free space --
            see ``probe_scene.py`` and the surveyed poses in ``scene.py``.
        use_px4: Attach a ``PX4MavlinkBackend`` to the vehicle.
        px4_dir: PX4-Autopilot checkout, required when ``use_px4`` is True.
            PX4 is never auto-launched (Pegasus's ``PX4LaunchTool`` uses a
            broken working directory) -- the caller launches it, see
            :mod:`px4_launch`.
        ground_plane: Add an invisible collision floor, see
            :func:`add_collision_ground`. Required for the PX4 path.

    Returns:
        ``(world, adapter)`` -- the Pegasus world and the
        :class:`PegasusIrisVehicle` adapter.
    """
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    from sparx_agency.robots.PEGASUS.adapters.scene import load_indoor_scene
    from sparx_agency.robots.PEGASUS.adapters.vehicle import PegasusIrisVehicle

    pg = PegasusInterface()
    pg.initialize_world()
    world = pg.world

    load_indoor_scene(scene)
    if ground_plane:
        add_collision_ground()
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()

    adapter = PegasusIrisVehicle(
        pegasus_extension_root=pegasus_root,
        init_pos=tuple(spawn_xyz),
        use_px4=use_px4,
        px4_dir=str(px4_dir) if px4_dir is not None else None,
        px4_autolaunch=False,
    )
    for _ in range(STAGE_SETTLE_STEPS):
        simulation_app.update()
    return world, adapter


def warmup_camera(world, ticks: int = CAMERA_WARMUP_RENDER_TICKS) -> int:
    """Render until the onboard camera starts producing frames.

    Returns:
        The number of steps taken, so the caller can keep a continuous count.
    """
    for _ in range(ticks):
        world.step(render=True)
    return ticks


class FlightRecorder:
    """Collects onboard RGBD frames into a recording, and optionally an MP4.

    The on-disk output is the same schema real rosbag extractions produce (see
    ``tasks/planning/vlas/common/finetune/datasets/recording.py``), so a
    simulated flight is a drop-in ``data.recording`` source for NavDP
    fine-tuning -- no parallel dataset format.

    Args:
        adapter: The :class:`PegasusIrisVehicle` to capture from.
        out_dir: Directory to write the recording to.
        rate_hz: Capture rate recorded in the output metadata.
        video_out: Optional MP4 path.
        video_source: ``"chase"`` writes the external chase camera's view,
            ``"onboard"`` the drone's own forward-facing camera.
        chase_camera: The chase camera, required for ``video_source="chase"``.
    """

    def __init__(self, adapter, out_dir: Path, rate_hz: float = 10.0,
                 video_out: Path = None, video_source: str = "chase", chase_camera=None):
        if video_out is not None and video_source == "chase" and chase_camera is None:
            raise ValueError("video_source='chase' needs a chase_camera")

        self._adapter = adapter
        self._out_dir = out_dir
        self._rate_hz = rate_hz
        self._video_source = video_source
        self._chase_camera = chase_camera
        self._frames = []
        self._writer = None

        if video_out is not None:
            import cv2

            video_out.parent.mkdir(parents=True, exist_ok=True)
            size = ((960, 540) if video_source == "chase"
                    else (adapter.intrinsics.width, adapter.intrinsics.height))
            self._writer = cv2.VideoWriter(
                str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), rate_hz, size,
            )
            self._video_out = video_out
            print(f"VIDEO_RECORDING ({video_source}) -- writing to {video_out}", flush=True)

    def capture(self) -> None:
        """Grab one onboard frame (and one video frame, if recording video)."""
        frame = self._adapter.capture_frame()
        self._frames.append(frame)
        if self._writer is None:
            return
        rgb = frame.rgb if self._video_source == "onboard" else self._chase_camera.get_rgb()
        if rgb is not None:
            self._writer.write(rgb[:, :, ::-1])  # RGB -> BGR for cv2

    def finish(self) -> dict:
        """Close the video and write the recording to disk.

        Returns:
            The ``export_flight`` stats dict.
        """
        from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import export_flight

        if self._writer is not None:
            self._writer.release()
            print(f"wrote video to {self._video_out}", flush=True)

        stats = export_flight(
            self._frames, self._out_dir, self._adapter.intrinsics,
            rate_hz=self._rate_hz, camera_height_m=1.0, pitch_deg=0.0,
        )
        print(f"wrote {stats['frames']} frames to {self._out_dir}", flush=True)
        return stats
