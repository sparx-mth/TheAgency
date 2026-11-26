"""Habitat's 2-D episodic GPS and compass contract, at the agent/scorer boundary."""
from dataclasses import replace
import math

from sparx_agency.core.common.math.se2 import world_to_body_2d
from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.relative_height import RelativeHeight
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import STANDARD_DETECTOR


STANDARD_INPUTS = "rgbd-objectgoal-episodic-gps2d-compass/v1"


def uses_standard_inputs(config, policy, *, record):
    """Check the prepared run before creating output or handing an observation to the agent."""
    mode = config.get("policy_inputs")
    if mode is None:
        return False
    if mode != STANDARD_INPUTS:
        raise ValueError("Unknown policy input contract: %r" % mode)
    method, actual = config.get("method", {}), config["policy"]
    frames = actual.get("frame_log") or method.get("frame_log")
    if (record or config.get("perception") == "ground-truth" or method.get("perception") == "ground-truth"
            or actual.get("ground_truth_semantics") or method.get("llm_answers")
            or (frames and frames["store"]["kind"] != "memory")):
        raise ValueError("Standard inputs forbid ground-truth perception, recordings and answer stores")
    if "detector" in method or hasattr(policy, "detector"):
        backend = method.get("detector", {}).get("metadata", {}).get("backend")
        if backend != STANDARD_DETECTOR:
            raise ValueError("Standard inputs require a verified %s detector, not %r" % (STANDARD_DETECTOR, backend))
    return True


class StandardInputAgent(ObjNavAgent):
    """Keep absolute state in the evaluator, including for its 3-D SPL cross-check.

    GPS is expressed in the start frame, forward/left (Habitat GPS uses
    forward/right). Both the search policy and its action converter see this
    same frame. Relative height is estimated from RGB-D, never read from
    the world pose. LOOK pitch is action history.
    """

    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name
        self.origin = None

    def reset(self, episode):
        self.origin = None
        self.pitch = 0.0
        self.relative_height = RelativeHeight()
        self.previous_forward = False
        self.tilt = math.radians(episode.action_spec.tilt_angle_deg)
        self.agent.reset(replace(episode, episode_id="episode", scene_id="scene",
                                 benchmark="objectnav", split="unspecified", metadata={}))

    def act(self, observation):
        pose = observation.pose
        origin = self.origin
        if origin is None:
            if observation.step != 0:
                raise ValueError("Episodic GPS needs the reset observation")
            origin = self.origin = (pose.x, pose.y, pose.yaw)
        x, y = world_to_body_2d(pose.x, pose.y, *origin)
        planar = AgentPose(x, y, 0.0, normalize_angle(pose.yaw - origin[2]), self.pitch)
        height = self.relative_height.update(observation.rgb, observation.depth_m, observation.camera,
                                             planar, self.previous_forward)
        local = replace(planar, z=height)
        decision = self.agent.act(replace(observation, pose=local))
        self.previous_forward = decision.action == DiscreteAction.MOVE_FORWARD
        if decision.action == DiscreteAction.LOOK_DOWN:
            self.pitch += self.tilt
        elif decision.action == DiscreteAction.LOOK_UP:
            self.pitch -= self.tilt
        return decision

    def episode_info(self):
        estimator = self.relative_height
        return dict(self.agent.episode_info(), rgbd_height={
            "estimator": estimator.NAME, "relative_m": estimator.height,
            "accepted_strides": estimator.accepted, "unmeasured_strides": estimator.unmeasured})
