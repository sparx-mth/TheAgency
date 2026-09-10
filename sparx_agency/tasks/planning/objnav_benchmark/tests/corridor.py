"""A scripted corridor and a scripted agent for the runner's tests, each able to break one contract clause on request.

An adapter bug does not crash a benchmark; it produces a number. So each flaw
of :class:`CorridorEnv` breaks exactly one clause the runner checks, and each
misbehaviour of :class:`ScriptedAgent` is one the agent-error policy must
handle. Geodesic distance is straight-line distance along the corridor, so
every metric has a closed form.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import (
    DiscreteAction,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    EpisodeMeasurement,
)
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_episode

A = DiscreteAction
F, STOP = A.MOVE_FORWARD, A.STOP
CAMERA = CameraSpec(Intrinsics(width=4, height=3, fx=2.0, fy=2.0, cx=1.5, cy=1.0),
                    height_m=0.88, min_depth_m=0.1, max_depth_m=5.0)
OTHER_CAMERA = CameraSpec(CAMERA.intrinsics, height_m=1.5, min_depth_m=0.1,
                          max_depth_m=5.0)
SUCCESS_M = 0.1
#: Four 0.25 m steps to a goal 1 m ahead, then STOP.
WALK_AND_STOP = (F, F, F, F, STOP)
#: The script reaches the first two goals and stops 0.5 m short of the third.
GOALS = {"e1": 1.0, "e2": 1.0, "e3": 1.5}
CONFIG = {"env": "corridor", "agent": "scripted"}


class CorridorEnv(ObjNavEnv):
    """A corridor along world +x, the goal ``goal`` metres ahead; each flaw breaks one contract clause.

    Geodesic distance is straight-line distance, so every metric has a
    closed form. A wall at ``wall_x`` blocks MOVE_FORWARD atomically.
    """

    name = "corridor"

    def __init__(self, goals, max_steps=20, actions=None, wall_x=None,
                 flaws=(), unreachable_end=False):
        self.goals = dict(goals)
        self.spec = (DiscreteActionSpec() if actions is None
                     else DiscreteActionSpec(actions=actions))
        self.max_steps, self.wall_x = max_steps, wall_x
        self.flaws, self.unreachable_end = frozenset(flaws), unreachable_end
        self.resets = []

    def episode_ids(self):
        return tuple(self.goals)

    def reset(self, episode_id):
        self.goal = self.goals[episode_id]
        self.resets.append(episode_id)
        self.pose, self.steps, self.stopped, self.path = AgentPose(0.0, 0.0), 0, False, 0.0
        episode = ObjNavEpisode(
            episode_id="other" if "reset_wrong_id" in self.flaws else episode_id,
            scene_id="corridor", benchmark="corridor", split="test",
            target_category="chair", camera=CAMERA, action_spec=self.spec,
            max_steps=self.max_steps)
        return episode, self.observation(int("reset_step_one" in self.flaws), "reset")

    def observation(self, step, phase):
        """The frame at ``step``; a ``<phase>_wrong_*`` flaw corrupts it."""
        return ObjNavObservation(
            rgb=np.zeros((3, 4, 3), np.uint8), depth_m=np.full((3, 4), np.inf),
            pose=self.pose,
            camera=OTHER_CAMERA if phase + "_wrong_camera" in self.flaws else CAMERA,
            target_category="bed" if phase + "_wrong_target" in self.flaws else "chair",
            step=step)

    @property
    def episode_over(self):
        if "over_at_reset" in self.flaws and self.steps == 0:
            return True
        if "ends_early" in self.flaws and self.steps >= 2:
            return True
        budget = self.steps >= self.max_steps and "runs_past_budget" not in self.flaws
        return budget or (self.stopped and "stop_does_not_end" not in self.flaws)

    def step(self, action):
        if self.episode_over or not self.spec.allows(action):
            raise EnvContractError("corridor cannot execute %s now" % action.name)
        before, pose = self.pose, self.pose
        turn = math.radians(self.spec.turn_angle_deg)
        if action == STOP:
            self.stopped = True
        elif action == F:
            x = pose.x + 0.25 * math.cos(pose.yaw)
            if self.wall_x is None or x <= self.wall_x + 1e-9:
                pose = AgentPose(x, pose.y + 0.25 * math.sin(pose.yaw), 0.0,
                                 pose.yaw, pose.camera_pitch)
        elif action in (A.TURN_LEFT, A.TURN_RIGHT):
            sign = 1.0 if action == A.TURN_LEFT else -1.0
            pose = AgentPose(pose.x, pose.y, 0.0, pose.yaw + sign * turn, pose.camera_pitch)
        else:
            sign = 1.0 if action == A.LOOK_DOWN else -1.0
            pose = AgentPose(pose.x, pose.y, 0.0, pose.yaw, pose.camera_pitch + sign * turn)
        self.pose = pose
        self.path += math.dist(before.position(), pose.position())
        self.steps += 1
        return self.observation(self.steps + int("step_miscount" in self.flaws), "step")

    def measure(self):
        if not self.episode_over:
            raise EnvContractError("corridor is still running")
        final = (math.inf if self.unreachable_end
                 else math.hypot(self.goal - self.pose.x, self.pose.y))
        stopped = self.stopped != ("measure_wrong_stop" in self.flaws)
        success = stopped and final <= SUCCESS_M
        return EpisodeMeasurement(
            success=success, stop_called=stopped,
            termination=TERMINATION_STOP if stopped else TERMINATION_STEP_LIMIT,
            steps=self.steps + int("measure_miscount" in self.flaws),
            shortest_path_m=self.goal, start_distance_to_goal_m=self.goal,
            final_distance_to_goal_m=final,
            path_length_m=self.path * (100.0 if "path_in_cm" in self.flaws else 1.0),
            native_metrics={"success": float(success), "distance_to_goal": final
                            + (0.5 if "wrong_native_distance" in self.flaws else 0.0)})


def boom():
    raise RuntimeError("boom")


class ScriptedAgent(ObjNavAgent):
    """Plays a script, repeating its last action; can misbehave at one step, in one episode or all."""

    name = "scripted"

    def __init__(self, script=WALK_AND_STOP, misbehave_at=None, misbehaviour=boom,
                 in_episode=None, info=None, info_error=False, reset_error=None):
        self.script, self.misbehave_at, self.misbehaviour = tuple(script), misbehave_at, misbehaviour
        self.in_episode, self.info, self.info_error = in_episode, info, info_error
        self.reset_error = reset_error
        self.seen_steps = []

    def reset(self, episode):
        if self.reset_error is not None:
            raise self.reset_error
        self.episode_id, self.index = episode.episode_id, 0

    def act(self, observation):
        self.seen_steps.append(observation.step)
        index, self.index = self.index, self.index + 1
        if index == self.misbehave_at and self.in_episode in (None, self.episode_id):
            return self.misbehaviour()
        return AgentDecision(self.script[min(index, len(self.script) - 1)])

    def episode_info(self):
        if self.info_error:
            raise ValueError("no diagnostics")
        return {"acts": self.index} if self.info is None else self.info


def one_episode(env=None, agent=None, **options):
    """Run episode ``e1`` of ``env`` (a 1 m corridor by default) with ``agent`` (the walk-and-stop script)."""
    return run_episode(env or CorridorEnv({"e1": 1.0}), agent or ScriptedAgent(),
                       "e1", **options)
