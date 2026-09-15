"""Versioned MP3D ObjectNav evaluation settings, outside the lightweight harness.

Every number here is transcribed from the published configuration, not tuned:
habitat-lab ``configs/tasks/objectnav_mp3d.yaml`` (v0.2.2) and
``habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml``
(v0.2.4), which agree with habitat-challenge 2021's
``configs/challenge_objectnav2021.local.rgbd.yaml``. Changing a field is a
different experiment, not a tuning knob.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import ALL_ACTIONS, DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import KinematicTolerance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import (
    NAVMESH_AGENT_RECOMPUTED)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import EvaluationSettings

#: The published validation split, as the comparison papers describe it
#: (ApexNav, arXiv:2504.14478v1 Sec. V-A: "MP3D (Matterport3D from 2021
#: Habitat Challenge, 2195 episodes, 11 scenes, 21 goal categories)").
#: The scene *names* are read from the installed release, never assumed here.
PUBLISHED_VAL_SCENES = 11
PUBLISHED_VAL_EPISODES = 2195


@dataclass(frozen=True)
class MP3DProtocol:
    """Habitat ObjectNav MP3D v1 validation protocol.

    Unlike the SemExp Gibson evaluator, this one is the habitat-lab evaluator:
    STOP is required, the distance is the geodesic distance to the nearest
    published goal *view point*, the path is accumulated in 3-D from the reset
    pose with a zero initial accumulator, and sliding is disabled. Those are
    the shared harness defaults, so :meth:`evaluation` overrides nothing but
    the motion tolerances.
    """

    protocol_id: str = "mp3d-objectnav-v1-val/1"
    benchmark: str = "mp3d"
    split: str = "val"
    max_steps: int = 500
    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0
    camera_height_m: float = 0.88
    agent_height_m: float = 0.88
    agent_radius_m: float = 0.18
    min_depth_m: float = 0.5
    max_depth_m: float = 5.0
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    tilt_angle_deg: float = 30.0
    success_distance_m: float = 0.1
    distance_to: str = "view_points"
    allow_sliding: bool = False
    #: habitat-lab copies AGENT_0.RADIUS/HEIGHT into habitat_sim's agent
    #: config, and habitat-sim then recomputes <scene>.navmesh for that
    #: embodiment. Every published MP3D distance and collision test therefore
    #: comes from the recomputed navmesh, not the shipped 0.1 m / 1.5 m one.
    #: See the measured difference in habitat/simulator.py's docstring.
    navmesh_source: str = NAVMESH_AGENT_RECOMPUTED
    require_stop_for_success: bool = True
    path_length_dimension: str = "3d"
    path_length_epsilon_m: float = 0.0
    reference_sim_version: str = "0.2.4"

    def camera(self) -> CameraSpec:
        """The registered RGB-D pinhole camera, mounted at [0, 0.88, 0]."""
        return CameraSpec(
            intrinsics_from_hfov(self.width, self.height, self.hfov_deg),
            self.camera_height_m, self.min_depth_m, self.max_depth_m)

    def actions(self) -> DiscreteActionSpec:
        """All six published actions, LOOK_UP/LOOK_DOWN included.

        The search method does not tilt, but the action space an episode
        advertises is the benchmark's, not the subset one policy happens to
        use. Habitat does not clamp LOOK, so the pitch limits stay None.
        """
        return DiscreteActionSpec(
            forward_step_m=self.forward_step_m,
            turn_angle_deg=self.turn_angle_deg,
            tilt_angle_deg=self.tilt_angle_deg, actions=ALL_ACTIONS)

    def kinematics(self) -> KinematicTolerance:
        """Shared defaults: sliding is off, so a blocked step simply does not move."""
        return KinematicTolerance()

    def evaluation(self) -> EvaluationSettings:
        """The harness options this protocol requires; all are shared defaults."""
        return EvaluationSettings(
            require_stop_for_success=self.require_stop_for_success,
            path_length_dimension=self.path_length_dimension,
            path_length_epsilon_m=self.path_length_epsilon_m,
            kinematics=self.kinematics())


PROTOCOL = MP3DProtocol()
