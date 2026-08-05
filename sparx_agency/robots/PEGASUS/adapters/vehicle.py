"""Spawn a PX4-backed Pegasus Iris multirotor with an RGBD camera.

This is the "robot" layer for the PEGASUS platform (see
``tasks/planning/vlas/README.md``): topics/intrinsics/actuation for a
simulated drone, translated into the same shared vocabulary real platforms use
(:class:`~sparx_agency.core.common.types.Intrinsics`,
:class:`~sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract.SimFrame`).
It names no policy -- see ``robots/PEGASUS/README.md``.

Two things here exist specifically to serve autonomous data collection:

* **The camera resolution is a parameter**, not a fixture. The calibration YAML
  describes the real XTEND's camera; rendering it at another resolution is a
  pure rescale of the pinhole intrinsics, so a campaign can trade image size
  against throughput and disk without recalibrating anything.
* **Every frame carries the full 6-DoF state**, not just ``(x, y, yaw)``. A
  simulator knows the aircraft's exact position, attitude and velocity; a real
  flight has to estimate them and loses most of it. Throwing that away in the
  recorder would be discarding the one advantage simulated data has.

Must run inside a live Isaac Sim process (``isaac_run`` / ``python.sh``) with the
patched Pegasus extension enabled -- see ``robots/PEGASUS/setup/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.sim_extract import (
    SimFrame, resolution_of,
)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_IRIS_USD = "pegasus/simulator/assets/Robots/Iris/iris.usd"  # relative to the pegasus.simulator extension root

# The Iris body mesh extends to x=+0.156 (checked via UsdGeom.BBoxCache); this
# clears it without reaching the rotor arms/props (x=+0.267).
CAMERA_OFFSET_FLU = (0.2, 0.0, 0.0)
# MonocularCamera's own default is yaw=180. yaw=0 strips that flip and points the
# camera BACKWARD at the vehicle's own body/arms instead of forward and away.
CAMERA_ORIENTATION_DEG = (0.0, 0.0, 180.0)
# Half the rotor-tip span plus a little. What the planner must keep clear.
AIRFRAME_RADIUS_M = 0.35

DEFAULT_CAMERA_CONFIG = "camera_pegasus_iris_720x420.yaml"
"""The calibration a data-collection campaign renders with: the camera's native size.

Recording native and downsampling at training time means a change of camera or
of model input size never requires re-flying a campaign, which is the expensive
half of this pipeline. It is free for NavDP, because 720x420 and the deployed
504x294 share an aspect ratio and so arrive as the same 224x131 image inside
NavDP's 224x224 input.

Do **not** render with ``camera_pegasus_iris_504x392.yaml``. It descends from
the same native calibration by a *crop* rather than a resize, so it discards 20
degrees of horizontal field of view and leaves what remains lopsided about the
optical axis (36.9 deg left, 18.9 deg right); a policy trained on it meets
obstacles from the right 20 degrees later than obstacles from the left.
"""


def load_camera_intrinsics(name: str = DEFAULT_CAMERA_CONFIG) -> Intrinsics:
    """Load one of the ``robots/PEGASUS/config/*.yaml`` camera intrinsics files."""
    data = yaml.safe_load((_CONFIG_DIR / name).read_text())
    return Intrinsics(
        width=data["image_width"], height=data["image_height"],
        fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"],
    )


def camera_intrinsics(resolution: Optional[Tuple[int, int]] = None,
                      name: str = DEFAULT_CAMERA_CONFIG) -> Intrinsics:
    """The platform camera's calibration, optionally rendered at another size.

    Args:
        resolution: ``(width, height)`` to render at. None keeps the
            calibration's own resolution.
        name: Which calibration file under ``robots/PEGASUS/config/``.

    Returns:
        Intrinsics valid for the requested resolution.
    """
    intrinsics = load_camera_intrinsics(name)
    if resolution is None:
        return intrinsics
    return resolution_of(intrinsics, resolution[0], resolution[1])


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
        init_yaw: float = 0.0,
        intrinsics: Intrinsics = None,
        resolution: Optional[Tuple[int, int]] = None,
        camera_frequency: float = 10.0,
        px4_dir: str = None,
        px4_autolaunch: bool = False,
        use_px4: bool = True,
        vehicle_id: int = 0,
        noiseless: bool = True,
    ):
        """Spawn the vehicle.

        Args:
            pegasus_extension_root: Path to the patched
                ``PegasusSimulator/extensions/pegasus.simulator`` checkout (see
                ``robots/PEGASUS/setup/install.sh``).
            stage_prefix: Stage path to spawn the vehicle at.
            init_pos: Initial ``(x, y, z)`` spawn position, world frame.
            init_yaw: Initial heading, radians CCW from +X (FLU).
            intrinsics: Camera intrinsics. Defaults to the platform calibration
                (which matches XTEND's 392x504 depth-engine calibration),
                rescaled to ``resolution``.
            resolution: ``(width, height)`` to render at. Ignored when
                ``intrinsics`` is given explicitly.
            camera_frequency: Camera capture rate, Hz.
            px4_dir: Path to a ``PX4-Autopilot`` checkout built with
                ``make px4_sitl_default none``. Defaults to
                ``PegasusInterface().px4_path``. Ignored if ``use_px4=False``.
            px4_autolaunch: If True, Pegasus's own ``PX4LaunchTool`` spawns PX4
                from a **fresh empty temp directory**, which fails
                (``rc.vehicle_setup: No such file``) for this PX4 version --
                its rootfs only exists under the real build output. Leave False
                and launch PX4 yourself, see :mod:`px4_launch`.
            use_px4: If False, no ``PX4MavlinkBackend`` is attached at all --
                nothing drives the vehicle automatically. Used by
                ``tasks/planning/sim_flight_recording/fly_direct.py``, which
                applies forces directly.
            vehicle_id: PX4 instance this vehicle talks to. Selects the HIL
                TCP port (``4560 + vehicle_id``) and must match the ``-i``
                argument PX4 was launched with -- see :mod:`px4_launch`. This is
                what makes several simulators on one machine possible.
            noiseless: Replace Pegasus's noisy sensor suite with the exact one
                from :mod:`sensors`. On by default: simulated sensor noise costs
                metres of position hold indoors and buys nothing for expert
                demonstrations.
        """
        from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

        from sparx_agency.robots.PEGASUS.adapters.sensors import noiseless_sensors

        self.intrinsics = intrinsics or camera_intrinsics(resolution)
        self.stage_prefix = stage_prefix
        self.vehicle_id = vehicle_id

        backends = []
        if use_px4:
            from pegasus.simulator.logic.backends.px4_mavlink_backend import (
                PX4MavlinkBackend, PX4MavlinkBackendConfig,
            )
            backend_config = PX4MavlinkBackendConfig({
                "vehicle_id": vehicle_id,
                "px4_autolaunch": px4_autolaunch,
                "px4_dir": px4_dir or PegasusInterface().px4_path,
                "px4_vehicle_model": "gazebo-classic_iris",
            })
            backends = [PX4MavlinkBackend(config=backend_config)]

        camera = MonocularCamera("front_camera", config={
            "depth": True,
            "position": np.array(CAMERA_OFFSET_FLU),
            "orientation": np.array(CAMERA_ORIENTATION_DEG),
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
        if noiseless:
            config.sensors = noiseless_sensors()

        self._camera = camera
        self.vehicle = Multirotor(
            stage_prefix,
            str(pegasus_extension_root / _IRIS_USD),
            vehicle_id,
            list(init_pos),
            _yaw_to_quaternion(init_yaw),
            config,
        )

    def pose_flu(self) -> tuple:
        """Current ``(x, y, yaw)`` in the world frame, FLU convention."""
        x, y, _z = self.vehicle.state.position
        return float(x), float(y), self.yaw()

    def yaw(self) -> float:
        """Current heading, radians CCW from +X."""
        qx, qy, qz, qw = self.vehicle.state.attitude
        return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def capture_frame(self, stamp_s: float = None) -> SimFrame:
        """Read the current RGB+depth frame and full ground-truth state.

        The pose is read *after* the render that produced the images, so the two
        describe the same instant to within one physics step -- which is why the
        stamp is the caller's simulation clock rather than a frame counter.

        Args:
            stamp_s: Simulation time to label the frame with. None falls back to
                the exporter's nominal rate.

        Returns:
            A :class:`SimFrame` ready for :func:`sim_extract.export_flight`.

        Raises:
            RuntimeError: If the camera has not produced a frame yet (it warms
                up for the first ~100 render ticks -- see
                ``MonocularCamera.update``).
        """
        depth = self._camera._camera.get_depth()
        if depth is None:
            raise RuntimeError("camera has not produced a depth frame yet")
        rgb = self._camera._camera.get_rgb()
        state = self.vehicle.state
        x, y, z = state.position
        return SimFrame(
            depth=np.asarray(depth, dtype=np.float32),
            pose=(float(x), float(y), self.yaw()),
            rgb=rgb,
            stamp_s=stamp_s,
            z=float(z),
            quaternion=tuple(float(v) for v in state.attitude),
            linear_velocity=tuple(float(v) for v in state.linear_velocity),
            angular_velocity=tuple(float(v) for v in state.angular_velocity),
            # Free here and expensive later: recovering acceleration from a
            # recording means differentiating a velocity sampled at the render
            # rate, which is noisy exactly where the transients are.
            linear_acceleration=tuple(float(v) for v in state.linear_acceleration),
            body_velocity=tuple(float(v) for v in state.linear_body_velocity),
        )


def _yaw_to_quaternion(yaw: float) -> list:
    """``[qx, qy, qz, qw]`` for a rotation of ``yaw`` about world +Z.

    Scalar-**last**, which is what Pegasus's ``Vehicle`` and ``State`` use.
    ``isaacsim.core.prims`` uses scalar-first for the same quantity -- check
    which convention you are in before copying a quaternion between them.
    """
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]
