"""Versioned HM3D ObjectNav evaluation settings, outside the lightweight harness.

Every number here is read off habitat-lab's own ``objectnav_hm3d`` benchmark
configuration, not inferred from a benchmark name. Changing any field is a
different experiment, so the whole dataclass is serialised into the run
manifest and into the configuration lock.

Two published protocols are defined:

``HM3D_V1``
    The 2022 Habitat ObjectNav challenge: episode dataset
    ``objectnav/hm3d/v1`` over the HM3D-Semantics **v0.1** scene release.
``HM3D_V2``
    The later episode dataset ``objectnav/hm3d/v2`` over HM3D-Semantics
    **v0.2**. Same six categories, same agent, same action geometry; only the
    episodes and the scenes differ.

The 2023 Habitat challenge itself moved to a Stretch embodiment with velocity
control, which is **not** what the papers we compare against ran; "HM3D-v2"
throughout this package means the v2 *episode dataset* under the v1 agent
configuration, which is what ApexNav's released configs do.

One deliberate deviation is offered explicitly rather than applied silently:
:data:`APEXNAV_SUCCESS_DISTANCE_M`. ApexNav's released configs set the success
radius to 0.2 m where habitat-lab's ``objectnav_hm3d`` uses 0.1 m. We evaluate
under the official 0.1 m and re-score the *same* episodes at 0.2 m for a
like-for-like row under their table; see ``report.py``.

**Revision ``/2`` (2026-10-14)** carries the two sensor/actuation choices the
Gibson protocol made for the shared method, so that the *same* algorithm runs
on both benchmarks: the camera pitch the method keeps itself to (level to
30 degrees up, 60 degrees down -- habitat-sim itself never clamps a LOOK, and
the bound is a method choice the target closing reads from the action spec)
and the enclosed depth-hole fill (``depth_hole_*``; see
``GibsonProtocol`` for the measurement behind it). Neither touches the
scoring: a LOOK is an action against the same 500-action budget, and the
depth fill changes what the policy sees, not what the evaluator measures.
``camera_tilt`` cannot be turned off here -- the HM3D action space *is* the
six actions -- it exists so the two protocols read alike.

Python 3.8 syntax; no simulator import at module scope.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS, DiscreteActionSpec)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import KinematicTolerance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import EvaluationSettings

#: The success radius ApexNav's released HM3D configs use, in metres. It is not
#: the official one; it exists here only so the deviation can be named, scored
#: separately and disclosed.
APEXNAV_SUCCESS_DISTANCE_M = 0.2

#: Dataset version -> the HM3D scene release its episodes were generated on.
SCENE_RELEASES = {"v1": "hm3d-v0.1", "v2": "hm3d-v0.2"}

#: Dataset version -> the first path component every episode's ``scene_id``
#: begins with, and therefore the directory name the scenes must be installed
#: under. Measured over all 2000 v1 and all 1000 v2 validation rows: v1 says
#: ``hm3d/val/...`` and v2 says ``hm3d_v0.2/val/...``. They must be able to
#: coexist, and ``datasets_download`` creates only the first, so the v0.2 link
#: has to be made by hand -- see ``assets.py``.
SCENE_ROOT_NAMES = {"v1": "hm3d", "v2": "hm3d_v0.2"}

#: Dataset version -> the habitat-sim ``datasets_download`` uid for those scenes.
SCENE_DOWNLOAD_UIDS = {"v1": "hm3d_val_v0.1", "v2": "hm3d_val_v0.2"}


@dataclass(frozen=True)
class HM3DProtocol:
    """One published HM3D ObjectNav configuration.

    Attributes:
        dataset_version: ``"v1"`` or ``"v2"``; selects the episode dataset and
            the scene release, nothing else.
        benchmark: The benchmark key results are filed under
            (``"hm3d_v1"`` / ``"hm3d_v2"``). Rows from the two are never mixed.
        success_distance_m: Geodesic distance to the nearest goal **view
            point** below which a STOP succeeds. habitat-lab's
            ``objectnav_hm3d`` uses 0.1 m.
        distance_to: What the distance-to-goal is measured to. habitat-lab's
            ObjectNav measures to ``view_points``, never to the object centre.
        camera_height_m: Where the RGB-D sensor sits above the floor.
        agent_height_m: The body's physical height, which is *not* the sensor
            mounting height in general even though HM3D happens to use 0.88 for
            both.
        navmesh: Which navmesh the episode is executed and measured on.
            ``"agent"`` is the reference behaviour and the default:
            habitat-sim's ``Simulator._config_pathfinder`` loads the shipped
            ``<scene>.basis.navmesh`` and then **recomputes** it whenever its
            recorded settings differ from the agent's radius and height -- which
            they always do here, because the shipped meshes are built at the
            library defaults (radius 0.10, height 1.50) and ObjectNav's agent is
            0.18 / 0.88. Measured on an ObjectNav episode with a published
            ``info.geodesic_distance``, the recomputed mesh reproduces the
            publisher's number bit-exactly while the shipped mesh is 5.5% short,
            so this is not a detail: it sets ``l``, and therefore every SPL.
            ``"published"`` forces the shipped mesh instead, for a deliberate
            comparison. Either way the choice is recorded, never inferred.
        reference_habitat_lab_version: The habitat-lab release the numbers were
            copied from. habitat-lab is not installed here; this adapter reads
            the published episode files and measures geodesics with habitat-sim
            directly, so the reference is recorded rather than imported.
    """

    dataset_version: str
    protocol_id: str
    benchmark: str
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
    camera_tilt: bool = True
    tilt_angle_deg: float = 30.0
    min_pitch_deg: float = -30.0
    max_pitch_deg: float = 60.0
    success_distance_m: float = 0.1
    distance_to: str = "view_points"
    allow_sliding: bool = False
    navmesh: str = "agent"
    require_stop_for_success: bool = True
    path_length_dimension: str = "3d"
    path_length_epsilon_m: float = 0.0
    reference_habitat_lab_version: str = "0.2.1"
    reference_habitat_sim_version: str = "0.2.4"
    # Depth holes, as in ``GibsonProtocol`` (where the rule is documented): a
    # no-return pixel enclosed by valid depth on at least ``depth_hole_min_rim``
    # of its rim and no larger than ``depth_hole_max_fraction`` of the frame is
    # filled from its nearest valid pixels; larger or open voids are dropped as
    # every baseline drops them. ``depth_hole_fill=False`` is the baselines'
    # sensor handling.
    depth_hole_fill: bool = True
    depth_hole_max_fraction: float = 0.5
    depth_hole_min_rim: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset_version not in SCENE_RELEASES:
            raise ValueError("HM3D dataset version must be one of %r, got %r"
                             % (sorted(SCENE_RELEASES), self.dataset_version))
        if not self.camera_tilt:
            raise ValueError("HM3D ObjectNav's action space is the six actions; camera_tilt cannot be off")
        if self.distance_to != "view_points":
            raise ValueError("habitat-lab ObjectNav measures to goal view points")
        if not 0 < self.success_distance_m <= 1.0:
            raise ValueError("Success radius must be a small positive distance")
        if self.navmesh not in ("agent", "published"):
            raise ValueError("navmesh must be 'agent' or 'published', got %r"
                             % (self.navmesh,))

    @property
    def scene_root_name(self) -> str:
        """The directory name this version's ``scene_id`` values start with."""
        return SCENE_ROOT_NAMES[self.dataset_version]

    @property
    def scene_release(self) -> str:
        """The HM3D scene release these episodes were generated against."""
        return SCENE_RELEASES[self.dataset_version]

    @property
    def scene_download_uid(self) -> str:
        """The ``habitat_sim.utils.datasets_download`` uid for those scenes."""
        return SCENE_DOWNLOAD_UIDS[self.dataset_version]

    def camera(self) -> CameraSpec:
        """The registered RGB-D pinhole camera, from the published HFOV."""
        return CameraSpec(
            intrinsics_from_hfov(self.width, self.height, self.hfov_deg),
            self.camera_height_m, self.min_depth_m, self.max_depth_m)

    def depth_holes(self) -> "DepthHoleFill":
        """The rule the simulator bridge fills no-return pixels by."""
        from sparx_agency.core.mapping.depth.depth_holes import DepthHoleFill
        return DepthHoleFill(self.depth_hole_fill, self.depth_hole_max_fraction, self.depth_hole_min_rim)

    def actions(self) -> DiscreteActionSpec:
        """All six challenge actions, with the pitch bounds the method keeps itself to.

        Habitat does not clamp the camera pitch; ``min_pitch_deg`` /
        ``max_pitch_deg`` are what the target closing and the camera control
        read from the spec, the same bounds as on Gibson.
        """
        return DiscreteActionSpec(
            forward_step_m=self.forward_step_m,
            turn_angle_deg=self.turn_angle_deg,
            tilt_angle_deg=self.tilt_angle_deg,
            min_pitch_deg=self.min_pitch_deg, max_pitch_deg=self.max_pitch_deg,
            actions=ALL_ACTIONS)

    def kinematics(self) -> KinematicTolerance:
        """The shared default, with the climb bound HM3D's stairs need and the step filter's placement.

        Sliding is off here, so a blocked MOVE_FORWARD moves nothing at all and
        the horizontal bounds stand unchanged. ``climb_m`` (since 2026-10-20) is
        0.60 m, as Gibson's: the default 0.20 m is one stair riser, but the
        navmesh habitat-sim recomputes here (``agent_max_climb`` 0.20 m,
        ``agent_max_slope`` 45 degrees) lets one 0.25 m stride cross a riser
        AND run along a sloped tread, and the two add. The first v1 sweep lost
        ``00878-XB4GS9ShBRE/000000`` to it: action 170 moved 0.204 m across a
        tread and 0.210 m down, which the contract refused as "a pose left in
        the Y-up frame" and, through the runner, ended the scene (LESSONS.md,
        2026-10-20). The trade is the same as Gibson accepted: a step that
        reads entirely as height is no longer refused by this bound alone,
        and the independent path-length and heading checks remain.

        ``reproject_m`` (since 2026-10-26) is 0.15 m: habitat-sim's step
        filter starts every body action from the agent's projection onto the
        navmesh and ends it on the mesh, so an agent it finds a few
        centimetres off the recomputed mesh is carried back onto it by its
        next turn or step. The full v1 sweep of 2026-10-22 (1,871 episodes,
        some half a million actions) met it twice, and each ended a
        99-episode scene: ``00878-XB4GS9ShBRE/000001`` action 156, a
        TURN_LEFT that "moved" 0.071 m across the floor at -52 degrees from
        its heading; ``00873-bxsVRursffK/000068`` action 315, a MOVE_FORWARD
        that went 0.094 m at -53 degrees and 0.057 m up on the stairs. Both
        refusals' "likely ..." clauses named frame and sliding faults that
        1,871 episodes of agreeing path lengths rule out. The bound stays well
        under a stride: a simulator that slides along a wall, or a pose
        stream in the wrong frame, still fails the heading check on its first
        real step, LOOKs and STOP get no slack at all, and the independent
        path-length check remains.
        """
        return KinematicTolerance(climb_m=0.60, reproject_m=0.15)

    def evaluation_settings(self) -> EvaluationSettings:
        """The harness options this protocol requires, stated explicitly."""
        return EvaluationSettings(
            require_stop_for_success=self.require_stop_for_success,
            path_length_dimension=self.path_length_dimension,
            path_length_epsilon_m=self.path_length_epsilon_m,
            kinematics=self.kinematics())

    def with_split(self, split: str) -> "HM3DProtocol":
        """The same protocol on another split, for development.

        Only ``val`` is a published benchmark, so the protocol id says which
        split produced a number and the harness files rows under
        ``(benchmark, split)`` — a train result can never be aggregated into a
        validation one by accident. The loader refuses to call a non-``val``
        split complete, because there is no published count to check it against.
        """
        if not isinstance(split, str) or not split.strip():
            raise ValueError("Split must be a non-blank name")
        if split == self.split:
            return self
        # Swap the split out of the id rather than appending to it: an id that
        # still said "val" while running train would be actively misleading in
        # a manifest, which is the one place it is read.
        name, separator, revision = self.protocol_id.partition("/")
        suffix = "-" + self.split
        name = (name[:-len(suffix)] + "-" + split if name.endswith(suffix)
                else name + "-" + split)
        return replace(self, split=split, protocol_id=name + separator + revision)

    def with_success_distance(self, metres: float) -> "HM3DProtocol":
        """The same protocol under a different, explicitly named success radius.

        The radius goes into the ``protocol_id`` without rounding, so two
        different radii can never produce one id -- a fixed two-decimal format
        would render 1 mm and 2 mm both as ``0.00m``, i.e. as a radius this
        class refuses outright.
        """
        return replace(self, success_distance_m=float(metres),
                       protocol_id="%s+success%gm" % (self.protocol_id, float(metres)))


#: HM3D-v1 validation: the 2022 challenge episode dataset on HM3D-Semantics v0.1.
HM3D_V1 = HM3DProtocol(dataset_version="v1",
                       protocol_id="hm3d-objectnav-v1-val/2",
                       benchmark="hm3d_v1")

#: HM3D-v2 validation: the v2 episode dataset on HM3D-Semantics v0.2.
HM3D_V2 = HM3DProtocol(dataset_version="v2",
                       protocol_id="hm3d-objectnav-v2-val/2",
                       benchmark="hm3d_v2")

#: Selectable by name on the command line.
PROTOCOLS = {"v1": HM3D_V1, "v2": HM3D_V2}


def protocol_for(version: str) -> HM3DProtocol:
    """The published protocol for ``"v1"`` or ``"v2"``.

    Raises:
        KeyError: Any other version, naming what is available.
    """
    if version not in PROTOCOLS:
        raise KeyError("Unknown HM3D dataset version %r. Available: %s"
                       % (version, ", ".join(sorted(PROTOCOLS))))
    return PROTOCOLS[version]
