"""Spawn a PX4-backed Pegasus Iris multirotor with an RGBD camera.

This is the "robot" layer for the PEGASUS platform (see
``tasks/planning/vlas/README.md``): topics/intrinsics/actuation for a
simulated drone, translated into the same shared vocabulary real platforms use
(:class:`~sparx_agency.core.common.types.Intrinsics`,
:class:`~sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract.SimFrame`).
It names no policy -- see ``robots/PEGASUS/README.md``.

Must run inside a live Isaac Sim process (``isaac_run`` / ``python.sh``) with the
patched Pegasus extension enabled -- see ``robots/PEGASUS/setup/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import SimFrame

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_IRIS_USD = "pegasus/simulator/assets/Robots/Iris/iris.usd"  # relative to the pegasus.simulator extension root


def load_camera_intrinsics(name: str = "camera_pegasus_iris_504x392.yaml") -> Intrinsics:
    """Load one of the ``robots/PEGASUS/config/*.yaml`` camera intrinsics files."""
    data = yaml.safe_load((_CONFIG_DIR / name).read_text())
    return Intrinsics(
        width=data["image_width"], height=data["image_height"],
        fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"],
    )


class PegasusIrisVehicle:
    """A PX4-backed Pegasus Iris multirotor, with a forward-facing RGBD camera.

    Ground-truth pose is read directly from the simulation (no localization
    stack needed) -- this is what makes simulated recordings the first source
    to carry a real ``poses.npy`` (see :mod:`sim_extract`).
    """

    def __init__(
        self,
        pegasus_extension_root: Path,
        stage_prefix: str = "/World/quadrotor",
        init_pos=(0.0, 0.0, 0.5),
        intrinsics: Intrinsics = None,
        camera_frequency: float = 10.0,
        px4_dir: str = None,
        px4_autolaunch: bool = True,
        use_px4: bool = True,
    ):
        """Spawn the vehicle.

        Args:
            pegasus_extension_root: Path to the patched
                ``PegasusSimulator/extensions/pegasus.simulator`` checkout (see
                ``robots/PEGASUS/setup/install.sh``).
            stage_prefix: Stage path to spawn the vehicle at.
            init_pos: Initial ``(x, y, z)`` spawn position, world frame.
            intrinsics: Camera intrinsics; defaults to
                :func:`load_camera_intrinsics`'s default (matches XTEND's
                392x504 depth-engine calibration).
            camera_frequency: Camera capture rate, Hz.
            px4_dir: Path to a ``PX4-Autopilot`` checkout built with
                ``make px4_sitl_default none``. Defaults to
                ``PegasusInterface().px4_path``. Ignored if ``use_px4=False``.
            px4_autolaunch: If True (default), Pegasus's own ``PX4LaunchTool``
                spawns PX4 from a **fresh empty temp directory**, which fails
                (``rc.vehicle_setup: No such file``) for this PX4 version --
                its rootfs (``etc/init.d/...``) only exists under the real
                build output, ``px4_dir/build/px4_sitl_default``. Set False
                and launch PX4 yourself with that as its cwd (see
                ``tasks/planning/sim_flight_recording/fly_and_watch.py``).
            use_px4: If False, no ``PX4MavlinkBackend`` is attached at all --
                ``config.backends`` stays empty and nothing drives the vehicle
                automatically. Use this with
                ``tasks/planning/sim_flight_recording/fly_direct.py``, which
                applies forces directly instead of going through PX4/MAVLink.
        """
        from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

        self.intrinsics = intrinsics or load_camera_intrinsics()
        self.stage_prefix = stage_prefix

        backends = []
        if use_px4:
            from pegasus.simulator.logic.backends.px4_mavlink_backend import (
                PX4MavlinkBackend, PX4MavlinkBackendConfig,
            )
            backend_config = PX4MavlinkBackendConfig({
                "vehicle_id": 0,
                "px4_autolaunch": px4_autolaunch,
                "px4_dir": px4_dir or PegasusInterface().px4_path,
                "px4_vehicle_model": "gazebo-classic_iris",
            })
            backends = [PX4MavlinkBackend(config=backend_config)]

        camera = MonocularCamera("front_camera", config={
            "depth": True,
            # The Iris body mesh extends to x=+0.156 (checked via UsdGeom.BBoxCache);
            # this clears it without reaching the rotor arms/props (x=+0.267).
            "position": np.array([0.2, 0.0, 0.0]),   # clear of the body, FLU
            # MonocularCamera's own default is yaw=180 -- yaw=0 (what this used to
            # say) strips that flip and points the camera BACKWARD at the vehicle's
            # own body/arms instead of forward and away from it.
            "orientation": np.array([0.0, 0.0, 180.0]),  # level, forward-facing
            "resolution": (self.intrinsics.width, self.intrinsics.height),
            "frequency": camera_frequency,
            "intrinsics": np.array([
                [self.intrinsics.fx, 0.0, self.intrinsics.cx],
                [0.0, self.intrinsics.fy, self.intrinsics.cy],
                [0.0, 0.0, 1.0],
            ]),
        })

        config = MultirotorConfig()
        config.graphical_sensors = [camera]
        config.backends = backends

        self._camera = camera
        self.vehicle = Multirotor(
            stage_prefix,
            str(pegasus_extension_root / _IRIS_USD),
            0,
            list(init_pos),
            [0.0, 0.0, 0.0, 1.0],
            config,
        )

    def pose_flu(self) -> tuple:
        """Current ``(x, y, yaw)`` in the world frame, FLU convention."""
        x, y, _z = self.vehicle.state.position
        qx, qy, qz, qw = self.vehicle.state.attitude
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return float(x), float(y), float(yaw)

    def capture_frame(self) -> SimFrame:
        """Read the current RGB+depth frame and ground-truth pose from the sim.

        Returns:
            A :class:`SimFrame` ready for :func:`sim_extract.export_flight`.

        Raises:
            RuntimeError: If the camera has not produced a frame yet (it warms
                up for the first ~100 physics steps -- see
                ``MonocularCamera.update``).
        """
        depth = self._camera._camera.get_depth()
        if depth is None:
            raise RuntimeError("camera has not produced a depth frame yet")
        rgb = self._camera._camera.get_rgb()
        return SimFrame(depth=np.asarray(depth, dtype=np.float32), pose=self.pose_flu(), rgb=rgb)
