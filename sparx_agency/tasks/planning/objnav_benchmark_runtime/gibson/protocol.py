"""Versioned Gibson evaluation settings, outside the lightweight harness."""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS, DiscreteActionSpec, NAVIGATION_ACTIONS)
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

    **The camera tilts** (``camera_tilt``, protocol ``/2`` since 2026-10-13).
    The published baselines drive the four body actions alone; this protocol
    adds Habitat's LOOK_UP / LOOK_DOWN pair (30 degrees a step, the camera
    held between 30 degrees up and 60 degrees down), the action space of
    habitat-lab's ObjectNav task. It is a deviation from the SemExp action
    set, recorded in the frozen configuration: the scoring is untouched (a
    LOOK is an action against the same 500-action budget and moves nothing),
    and it is what lets the target closing keep a low toilet or a high
    television whole in the frame on the way in -- the box-driven pitch
    servo of ``methods/bbox_pitch`` never fired in the 15-episode toilet
    campaign of 2026-10-13 because the protocol had no LOOK to fire.
    ``camera_tilt=False`` is the published four-action set.
    """

    protocol_id: str = "gibson-semexp-v1.1-val/2"
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
    camera_tilt: bool = True
    tilt_angle_deg: float = 30.0
    min_pitch_deg: float = -30.0
    max_pitch_deg: float = 60.0
    success_radius_m: float = 1.0
    map_resolution_m: float = 0.05
    allow_sliding: bool = True
    require_stop_for_success: bool = False
    path_length_dimension: str = "xz-planar"
    path_length_epsilon_m: float = 1e-5
    reference_sim_version: str = "0.1.5"
    # Depth holes (since 2026-10-08): a Gibson scan has no geometry where the scanner saw glass,
    # a mirror or a glossy screen, and Habitat renders those pixels as 0. Every zero-shot
    # baseline drops them (SemExp/L3MVN/SG-Nav push them to 100 m, home-robot zero-weights
    # them, ApexNav culls them; VLFM inpaints and then excludes them from obstacles), which
    # leaves a mirror as unknown -- a frontier. Here a hole that is enclosed by valid depth on
    # at least ``depth_hole_min_rim`` of its rim and no larger than ``depth_hole_max_fraction``
    # of the frame is filled from its nearest valid pixels (``core.mapping.depth.depth_holes``)
    # and projects as the surface around it; larger or open voids are dropped as before. A
    # deviation from the baselines' sensor handling, recorded here in the frozen configuration;
    # ``depth_hole_fill=False`` restores theirs.
    depth_hole_fill: bool = True
    depth_hole_max_fraction: float = 0.5
    depth_hole_min_rim: float = 0.5

    def depth_holes(self) -> "DepthHoleFill":
        """The rule the simulator bridge fills no-return pixels by."""
        from sparx_agency.core.mapping.depth.depth_holes import DepthHoleFill
        return DepthHoleFill(self.depth_hole_fill, self.depth_hole_max_fraction, self.depth_hole_min_rim)

    def camera(self) -> CameraSpec:
        """The registered RGB-D pinhole camera."""
        return CameraSpec(
            intrinsics_from_hfov(self.width, self.height, self.hfov_deg),
            self.camera_height_m, self.min_depth_m, self.max_depth_m)

    def actions(self) -> DiscreteActionSpec:
        """The body actions of the Gibson baselines, plus the LOOK pair when ``camera_tilt`` is on."""
        if not self.camera_tilt:
            return DiscreteActionSpec(
                forward_step_m=self.forward_step_m,
                turn_angle_deg=self.turn_angle_deg, tilt_angle_deg=self.tilt_angle_deg,
                actions=NAVIGATION_ACTIONS)
        return DiscreteActionSpec(
            forward_step_m=self.forward_step_m,
            turn_angle_deg=self.turn_angle_deg, tilt_angle_deg=self.tilt_angle_deg,
            min_pitch_deg=self.min_pitch_deg, max_pitch_deg=self.max_pitch_deg,
            actions=ALL_ACTIONS)

    def kinematics(self) -> KinematicTolerance:
        """Permit sliding and one 5 cm navmesh cell of positional correction.

        The requested action stays 0.25 m. Habitat 0.2.4 can realise 0.285 m
        between two valid navmesh points near a sliding boundary (observed in
        Wiconisco). Keep a finite bound, all turn/heading checks, and the
        independently recomputed path-length check rather than disabling them.

        ``settle_m`` and ``climb_m`` (since 2026-10-07): some published v1.1
        starts lie off the shipped navmesh surface -- up to 0.28 m below it
        (Wiconisco/000003; Darden/000000 is 0.10 m below), thirty of them
        by more than 0.15 m -- and habitat-sim 0.2.4 settles the agent onto
        the mesh on its first body action (a turn included: ``turn_left`` is
        registered as a body action, so the step filter's
        ``try_step(start, start)`` places the agent; measured 2026-10-11),
        a vertical move the agent did not make. Up to 0.30 m of it is taken
        off that one action before the bounds are judged. And the evaluated
        storey itself is not flat: steps measured on the shipped navmeshes
        climb up to 0.50 m on one 0.25 m stride (Collierville, Wiconisco:
        treads and split levels inside the storey band) in about one step
        in three hundred, where the default 0.20 m bound -- Habitat's one
        stair riser -- refused them. Without both the contract refused the
        episode and, through the runner, the whole run, and a resume
        replayed the refusal. Starts are still never snapped by us, and
        the horizontal bounds (step 0.30 m, heading 90 degrees; 100k
        sampled steps gave 0.287 m / 77 degrees at most) stand.

        ``settle_xy_m`` (since 2026-10-11): one published start,
        Wiconisco/000118, lies 0.033 m *outside* the shipped navmesh
        horizontally (the only one of the 1,000 that ``is_navigable``
        rejects; every other start is on the mesh to the millimetre), and
        the same first-action placement pulled it onto the mesh edge
        sideways: the warm-up TURN_LEFT "moved" 0.033 m, the 0.020 m
        turn-in-place bound refused it, and the full validation run died at
        episode 919 of 1,000 -- twice, deterministically, on 2026-10-10.
        One map cell (0.05 m, the same width ``forward_overshoot_m`` allows
        a step) of horizontal placement is now taken off that first action
        too, measured against all 1,000 published starts: 0.033 m is the
        only non-zero value. Like ``settle_m`` it changes no score and no
        decision: the contract only raises.
        """
        return KinematicTolerance(heading_deg=90.0, forward_overshoot_m=0.05, settle_m=0.30, climb_m=0.60,
                                  settle_xy_m=0.05)


PROTOCOL = GibsonProtocol()

