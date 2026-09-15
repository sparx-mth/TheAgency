"""Versioned RoboTHOR evaluation settings, outside the lightweight harness.

Every constant here is transcribed from the published challenge, not chosen:
``allenai/robothor-challenge``'s ``challenge_config.yaml`` and the runner in
``robothor_challenge/challenge.py``. Changing a field is a different
experiment, and the whole dataclass is written into the run manifest so a
resumed or frozen run cannot quietly disagree with the one it claims to
continue.

Three of these deserve more than a line, because each is a place where a
plausible wrong number is easy to reach:

* ``fieldOfView`` in the challenge config is **63.453048374758716**, and that
  is the **vertical** FOV -- the config's own comment says it "corresponds to
  a horizontal FOV of 79". At 640x480 the two give the same pinhole to within
  1e-14, which :mod:`...tests.test_thor_bridge` pins. Reading it as horizontal
  would shrink every focal length by a quarter and scale every mapped point
  with it.
* **Success is not a distance test.** ``challenge.py`` computes
  ``stopped and target_obj["visible"]`` and nothing else; the one-metre rule
  is carried entirely by ``visibilityDistance: 1.0``, because AI2-THOR's
  ``visible`` flag folds proximity, the camera frustum and an occlusion ray
  together. So :attr:`success_radius_m` is recorded for the manifest and for
  the method's own standoff reasoning, and is never used to judge an episode.
* **``l`` is shipped, not measured.** The official SPL numerator is
  ``path_distance(episode["shortest_path"])`` over corners precomputed with
  AI2-THOR's ``GetShortestPath`` and stored in the episode file. The evaluator
  never asks the live simulator for it, so neither do we -- a locally
  recomputed geodesic would be a different benchmark.

Python 3.8 syntax.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS, DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import KinematicTolerance

#: The 15 published validation scenes, in the publisher's order.
SCENES = tuple("FloorPlan_Val%d_%d" % (group, index)
               for group in (1, 2, 3) for index in (1, 2, 3, 4, 5))

#: Episodes the published validation split holds, and per scene. Used only to
#: check a claim of "the complete split"; nothing selects episodes by count.
VAL_EPISODES = 1800
VAL_EPISODES_PER_SCENE = 120


@dataclass(frozen=True)
class RobothorProtocol:
    """The RoboTHOR ObjectNav challenge profile, as the challenge publishes it.

    Geometry is the simulator's rendered RGB-D and its ground-truth base pose.
    Semantics remain predicted: no goal position, distance or object metadata
    reaches the policy.
    """

    protocol_id: str = "robothor-objectnav-challenge-2021-val/1"
    benchmark: str = "robothor"
    split: str = "val"
    max_steps: int = 500
    width: int = 640
    height: int = 480
    hfov_deg: float = 79.0
    #: The number the challenge config actually carries, kept so the manifest
    #: records what was passed to AI2-THOR rather than our restatement of it.
    vfov_deg: float = 63.453048374758716
    #: Camera height above :attr:`~...types.pose.AgentPose.z`, which is the
    #: floor -- so it is ``origin_height_m - 0.0312``, the LoCoBot's camera
    #: offset below its transform origin, and **not** ``0.901 - 0.0312``. The
    #: two differ by the 1 mm that the published start sits above the floor,
    #: and the live smoke check measures 0.8688 against a build.
    camera_height_m: float = 0.8688
    #: How far above the floor AI2-THOR reports the agent's transform origin.
    #: Every published episode starts at ``initial_position.y == 0.901`` and
    #: the LoCoBot capsule hangs 0.9 m below its origin.
    origin_height_m: float = 0.9
    agent_radius_m: float = 0.175
    body_height_m: float = 0.9
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    #: The Unity camera's own clip planes. AI2-THOR does not return ``inf`` or
    #: ``NaN`` where it sees nothing: a pixel with no geometry comes back as a
    #: finite far-plane sentinel. :attr:`max_depth_m` must stay below it, or
    #: the sky would map as a wall at the far plane. Read back from the live
    #: ``Initialize`` return and checked rather than assumed.
    camera_near_plane_m: float = 0.1
    camera_far_plane_m: float = 20.0
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    tilt_angle_deg: float = 30.0
    min_pitch_deg: float = -30.0
    max_pitch_deg: float = 30.0
    visibility_distance_m: float = 1.0
    success_radius_m: float = 1.0
    agent_mode: str = "locobot"
    agent_type: str = "stochastic"
    continuous_mode: bool = True
    snap_to_grid: bool = False
    require_stop_for_success: bool = True
    path_length_dimension: str = "3d"
    path_length_epsilon_m: float = 0.0
    #: The Unity build the challenge and AllenAct both pin. The pip version
    #: drifted (2.7.2 -> >=3.2.0); this commit is what fixes the physics,
    #: visibility and rendering, so it is the number to reproduce.
    #:
    #: It is also **not interchangeable with a modern one**, in two ways that
    #: were read out of its own source tree:
    #:
    #: * its LoCoBot branch sets ``maxDownwardLookAngle = maxUpwardLookAngle =
    #:   30f`` (``BaseFPSAgentController.cs``, the ``bot`` mode), which is
    #:   where :attr:`max_pitch_deg` comes from. ai2thor 5.0.0 moved the
    #:   LoCoBot into its own controller and raised the downward limit to
    #:   60 degrees, so a newer build has a different legal pitch range and a
    #:   ``LookDown`` that upstream would have refused now succeeds;
    #: * its depth shader returns ``Linear01Depth`` packed into three 8-bit
    #:   channels, where 5.0.0's returns float32 metres. The Python client
    #:   decodes what the build sends, so the client version and the build
    #:   commit are one choice, not two. Pair this build with the client the
    #:   challenge pins, or move both together and record that you did.
    thor_build_id: str = "bad5bc2b250615cb766ffb45d455c211329af17e"
    reference_ai2thor_version: str = "2.7.2"

    def camera(self) -> CameraSpec:
        """The registered RGB-D pinhole, built from the horizontal FOV."""
        return CameraSpec(
            intrinsics_from_hfov(self.width, self.height, self.hfov_deg),
            self.camera_height_m, self.min_depth_m, self.max_depth_m)

    def actions(self) -> DiscreteActionSpec:
        """All six challenge actions, with the LoCoBot's hard horizon clamp.

        The clamp is not decoration. AI2-THOR *fails* a LOOK that would pass
        +/-30 degrees: the step is spent and the camera does not move, and
        nothing downstream can see that, because only MOVE_FORWARD's blocked
        state is detectable from a pose. Left unset, the action ladder would
        re-ask for the same refused LOOK until the budget was gone. Every
        published episode starts at ``initial_horizon = 30`` -- already fully
        pitched down -- so this is reached on the first step, not rarely.
        """
        return DiscreteActionSpec(
            forward_step_m=self.forward_step_m,
            turn_angle_deg=self.turn_angle_deg,
            tilt_angle_deg=self.tilt_angle_deg,
            min_pitch_deg=self.min_pitch_deg,
            max_pitch_deg=self.max_pitch_deg,
            actions=ALL_ACTIONS)

    def initialize(self) -> dict:
        """The controller's ``initialize`` block, verbatim from the challenge."""
        return {"rotateStepDegrees": self.turn_angle_deg,
                "visibilityDistance": self.visibility_distance_m,
                "gridSize": self.forward_step_m,
                "agentType": self.agent_type,
                "continuousMode": self.continuous_mode,
                "snapToGrid": self.snap_to_grid,
                "agentMode": self.agent_mode,
                "fieldOfView": self.vfov_deg}

    def kinematics(self) -> KinematicTolerance:
        """The shared defaults, which are already AI2-THOR's noise.

        ``KinematicTolerance``'s docstring sizes ``turn_deg`` at six sigmas of
        AI2-THOR's N(0, 0.5) deg rotation noise and ``forward_overshoot_m`` at
        its N(0.001, 0.005) m step noise, precisely so that a full RoboTHOR
        sweep does not abort on a sound simulator. With ``snapToGrid`` off --
        which is what the challenge configures -- the realised step matches
        the converter's model, so nothing needs widening here. A run that
        turns snapping *on* would have to widen these and say why: a snapped
        0.25 m step off-axis can land 0.43 m away and read as a phantom wall.
        """
        return KinematicTolerance()


PROTOCOL = RobothorProtocol()
