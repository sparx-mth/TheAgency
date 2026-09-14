"""Versioned Gibson evaluation settings, outside the lightweight harness."""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteActionSpec, NAVIGATION_ACTIONS)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import KinematicTolerance

SCENES = ("Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco")


@dataclass(frozen=True)
class GibsonProtocol:
    """SemExp v1.1 validation protocol; changing a field is a different experiment.

    See README.md for source revisions and deviations of the public PONI code.
    Geometry is metric RGB-D and GT base pose. Semantics remain predicted.
    STOP's upstream dummy right turn is omitted: it affects no score, and our
    shared STOP contract requires an unchanged pose. Sliding is enabled in the
    reference, unlike the HM3D/MP3D configurations used by many later papers.
    """

    protocol_id: str = "gibson-semexp-v1.1-val/1"
    benchmark: str = "gibson"
    split: str = "val"
    max_steps: int = 500
    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0
    camera_height_m: float = 0.88
    agent_radius_m: float = 0.18
    min_depth_m: float = 0.5
    max_depth_m: float = 5.0
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    success_radius_m: float = 1.0
    map_resolution_m: float = 0.05
    allow_sliding: bool = True
    require_stop_for_success: bool = False
    path_length_dimension: str = "xz-planar"
    path_length_epsilon_m: float = 1e-5
    reference_sim_version: str = "0.1.5"

    def camera(self) -> CameraSpec:
        """The registered RGB-D pinhole camera."""
        return CameraSpec(
            intrinsics_from_hfov(self.width, self.height, self.hfov_deg),
            self.camera_height_m, self.min_depth_m, self.max_depth_m)

    def actions(self) -> DiscreteActionSpec:
        """The four actions used by Gibson baselines; no extra camera sweeps."""
        return DiscreteActionSpec(
            forward_step_m=self.forward_step_m,
            turn_angle_deg=self.turn_angle_deg, tilt_angle_deg=30.0,
            actions=NAVIGATION_ACTIONS)

    def kinematics(self) -> KinematicTolerance:
        """Permit sliding and one 5 cm navmesh cell of positional correction.

        The requested action stays 0.25 m. Habitat 0.2.4 can realise 0.285 m
        between two valid navmesh points near a sliding boundary (observed in
        Wiconisco). Keep a finite bound, all turn/heading checks, and the
        independently recomputed path-length check rather than disabling them.
        """
        return KinematicTolerance(heading_deg=90.0, forward_overshoot_m=0.05)


PROTOCOL = GibsonProtocol()

