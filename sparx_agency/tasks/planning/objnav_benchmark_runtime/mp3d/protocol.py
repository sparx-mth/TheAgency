"""Versioned MP3D ObjectNav evaluation settings, outside the lightweight harness.

**This file is the MP3D simulation driver's one source of benchmark numbers.**
The search algorithm is shared with Gibson and HM3D and knows nothing of the
benchmark it is run on; everything that is specific to Matterport3D ObjectNav
-- the stride and the turn, the camera's resolution, field of view, mounting
height and depth range, the body, the sliding rule, the step budget, and the
success rule (STOP within 0.1 m geodesic of a published goal *view point*) --
is read off habitat-lab's own ``objectnav_mp3d`` benchmark configuration and
written here once. Changing any field is a different experiment, so the whole
dataclass is serialised into the run manifest and into the configuration lock.

Sources, transcribed rather than remembered:

* habitat-lab ``configs/tasks/objectnav_mp3d.yaml`` (v0.2.1/v0.2.2) and
  ``habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml`` (v0.2.4):
  500 actions; 640x480 RGB and depth at HFOV 79 degrees mounted at
  ``[0, 0.88, 0]``; depth clipped to 0.5-5.0 m; agent height 0.88 m and
  radius 0.18 m; MOVE_FORWARD 0.25 m, TURN 30 degrees, LOOK 30 degrees;
  sliding off; ``SUCCESS_DISTANCE`` 0.1; ``DISTANCE_TO: VIEW_POINTS``; success
  needs STOP.
* habitat-challenge 2021 ``configs/challenge_objectnav2021.local.rgbd.yaml``,
  which agrees with the above and is the protocol ApexNav and SG-Nav state
  for their MP3D rows (11 val scenes, 21 goal categories, 2,195 episodes).

One deliberate deviation is offered explicitly rather than applied silently:
:data:`APEXNAV_SUCCESS_DISTANCE_M`. ApexNav's released configs set the success
radius to 0.2 m where habitat-lab's ``objectnav_mp3d`` uses 0.1 m. We evaluate
under the official 0.1 m and re-score the *same* episodes at 0.2 m for a
like-for-like row under their table; see ``report.py``.

**Revision ``/2``** carries the two sensor/actuation choices the Gibson and
HM3D protocols share so that the *same* algorithm runs on all three: the camera
pitch the method keeps itself to (level to 30 degrees up, 60 degrees down --
habitat-sim never clamps a LOOK, so the bound is the method's, read from the
action spec) and the enclosed depth-hole fill (``depth_hole_*``; the rule is
documented on ``GibsonProtocol``). Neither touches the scoring: a LOOK is an
action against the same 500-action budget, and the depth fill changes what the
policy sees, not what the evaluator measures.

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

#: The success radius ApexNav's released configs use, in metres. It is not the
#: official one; it exists here only so the deviation can be named, scored
#: separately and disclosed.
APEXNAV_SUCCESS_DISTANCE_M = 0.2

#: Dataset version -> the Matterport3D release its episodes were generated on.
#: There is one: the Habitat-prepared Matterport3D scenes (``mp3d_habitat.zip``,
#: 90 buildings, each ``<scene>/<scene>.glb`` beside ``<scene>.navmesh``,
#: ``<scene>.house`` and ``<scene>_semantic.ply``).
SCENE_RELEASES = {"v1": "mp3d-habitat"}

#: Dataset version -> the first path component every episode's ``scene_id``
#: begins with, and therefore the directory name the scenes must be installed
#: under ``data/scene_datasets/``. Every MP3D ObjectNav v1 row says
#: ``mp3d/<scene>/<scene>.glb``.
SCENE_ROOT_NAMES = {"v1": "mp3d"}

#: What each published validation split is documented to contain (habitat-lab's
#: dataset table; ApexNav Sec. V-A; SG-Nav Sec. 4.1). The loader refuses a
#: release that disagrees rather than quietly scoring a different number of
#: episodes.
PUBLISHED_VAL = {"v1": {"scenes": 11, "episodes": 2195}}


@dataclass(frozen=True)
class MP3DProtocol:
    """One published MP3D ObjectNav configuration.

    Attributes:
        dataset_version: ``"v1"``; selects the episode dataset. There is one
            published MP3D ObjectNav release, kept versioned so the manifest
            and the results directory say which one a number came from.
        benchmark: The benchmark key results are filed under (``"mp3d_v1"``).
        success_distance_m: Geodesic distance to the nearest goal **view
            point** below which a STOP succeeds. habitat-lab's
            ``objectnav_mp3d`` uses 0.1 m.
        distance_to: What the distance-to-goal is measured to. habitat-lab's
            ObjectNav measures to ``view_points``, never to the object centre.
        camera_height_m: Where the RGB-D sensor sits above the floor.
        agent_height_m: The body's physical height, which is *not* the sensor
            mounting height in general even though MP3D happens to use 0.88 for
            both.
        navmesh: Which navmesh the episode is executed and measured on.
            ``"agent"`` is the reference behaviour and the default:
            habitat-sim's ``Simulator._config_pathfinder`` loads the shipped
            ``<scene>.navmesh`` and then **recomputes** it whenever its recorded
            settings differ from the agent's radius and height -- which they do
            here, because the shipped Matterport meshes are built at the
            library defaults (radius 0.10, height 1.50) and ObjectNav's agent is
            0.18 / 0.88; habitat-lab copies ``AGENT_0.RADIUS``/``HEIGHT``
            straight into the agent configuration, so every published MP3D
            distance and collision test comes from the recomputed mesh.
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
    # sensor handling. Matterport scans have more of these than HM3D's (glass,
    # mirrors, windows), so this matters more here, and it still changes only
    # what the policy sees.
    depth_hole_fill: bool = True
    depth_hole_max_fraction: float = 0.5
    depth_hole_min_rim: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset_version not in SCENE_RELEASES:
            raise ValueError("MP3D dataset version must be one of %r, got %r"
                             % (sorted(SCENE_RELEASES), self.dataset_version))
        if not self.camera_tilt:
            raise ValueError("MP3D ObjectNav's action space is the six actions; camera_tilt cannot be off")
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
        """The Matterport3D scene release these episodes were generated against."""
        return SCENE_RELEASES[self.dataset_version]

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
        read from the spec, the same bounds as on Gibson and HM3D.
        """
        return DiscreteActionSpec(
            forward_step_m=self.forward_step_m,
            turn_angle_deg=self.turn_angle_deg,
            tilt_angle_deg=self.tilt_angle_deg,
            min_pitch_deg=self.min_pitch_deg, max_pitch_deg=self.max_pitch_deg,
            actions=ALL_ACTIONS)

    def kinematics(self) -> KinematicTolerance:
        """The shared default with the two bounds the recomputed navmesh needs: the same as HM3D's.

        Both are derived from habitat-sim 0.2.4's step filter and the navmesh
        it recomputes at this agent's dimensions, not from either dataset, so
        they carry over unchanged from ``HM3DProtocol.kinematics`` (where the
        measurements behind them are recorded):

        ``climb_m`` 0.85 m: with ``cell_height`` 0.20 m and ``agent_max_climb``
        0.20 m the voxel grid connects neighbouring cells whose quantised
        floors differ by one cell (a riser of anything under 2 x 0.20 m), a
        45-degree run over the rest of a 0.25 m stride adds 0.25 m, and the
        detail mesh is sampled to within one ``cell_height``; 0.40 + 0.25 +
        0.20 is the most height one stride can legitimately read as on this
        mesh. Matterport's buildings are multi-storey with open stairwells
        and split levels, so the bound is exercised here at least as much as
        on HM3D; a storey (2.5 m and more) or a teleport is still refused, and
        the path-length and heading checks are untouched.

        ``reproject_m`` 0.15 m: ``tryStep`` starts every body action from the
        agent's projection onto the mesh and ends it on the mesh, so an agent
        it finds a few centimetres off the recomputed mesh is carried back
        onto it by its next turn or step. The environment also reports the
        projection itself (``MP3DEnv.placed_pose``) so the check judges each
        body action from where the filter started it.

        Sliding is off here, so a blocked MOVE_FORWARD moves nothing at all
        and the horizontal bounds stand unchanged.
        """
        return KinematicTolerance(climb_m=0.85, reproject_m=0.15)

    def evaluation_settings(self) -> EvaluationSettings:
        """The harness options this protocol requires, stated explicitly."""
        return EvaluationSettings(
            require_stop_for_success=self.require_stop_for_success,
            path_length_dimension=self.path_length_dimension,
            path_length_epsilon_m=self.path_length_epsilon_m,
            kinematics=self.kinematics())

    def with_split(self, split: str) -> "MP3DProtocol":
        """The same protocol on another split, for development.

        Only ``val`` is a published benchmark, so the protocol id says which
        split produced a number and the harness files rows under
        ``(benchmark, split)`` -- a train or val_mini result can never be
        aggregated into a validation one by accident. The loader refuses to
        call a non-``val`` split complete, because there is no published count
        to check it against.
        """
        if not isinstance(split, str) or not split.strip():
            raise ValueError("Split must be a non-blank name")
        if split == self.split:
            return self
        name, separator, revision = self.protocol_id.partition("/")
        suffix = "-" + self.split
        name = (name[:-len(suffix)] + "-" + split if name.endswith(suffix)
                else name + "-" + split)
        return replace(self, split=split, protocol_id=name + separator + revision)

    def with_success_distance(self, metres: float) -> "MP3DProtocol":
        """The same protocol under a different, explicitly named success radius.

        The radius goes into the ``protocol_id`` without rounding, so two
        different radii can never produce one id.
        """
        return replace(self, success_distance_m=float(metres),
                       protocol_id="%s+success%gm" % (self.protocol_id, float(metres)))


#: MP3D ObjectNav v1 validation: the 2021 Habitat challenge episode dataset on
#: the Habitat-prepared Matterport3D scenes.
MP3D_V1 = MP3DProtocol(dataset_version="v1",
                       protocol_id="mp3d-objectnav-v1-val/2",
                       benchmark="mp3d_v1")

#: Selectable by name on the command line.
PROTOCOLS = {"v1": MP3D_V1}


def protocol_for(version: str) -> MP3DProtocol:
    """The published protocol for ``"v1"``.

    Raises:
        KeyError: Any other version, naming what is available.
    """
    if version not in PROTOCOLS:
        raise KeyError("Unknown MP3D dataset version %r. Available: %s"
                       % (version, ", ".join(sorted(PROTOCOLS))))
    return PROTOCOLS[version]
