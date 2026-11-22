"""A rig for the frame log's tests: one episode on a small camera, and frames that can be told apart.

Nothing here builds a policy: the log is exercised on its own, through the
contract types it reads. Each frame is a pattern of its step -- a constant
red level, a left-to-right green ramp and the complementary blue -- so a
frame filed under another's name, or a swapped channel order, changes a
number a test can assert.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec, NAVIGATION_ACTIONS
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose

WIDTH, HEIGHT = 48, 32


def episode(episode_id="scene-a/000003", target="chair", metadata=None) -> ObjNavEpisode:
    """One episode on a 48 x 32 camera with the benchmark's four body actions."""
    camera = CameraSpec(intrinsics_from_hfov(WIDTH, HEIGHT, 79.0), 0.88, 0.5, 5.0)
    actions = DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0, tilt_angle_deg=30.0,
                                 actions=NAVIGATION_ACTIONS)
    return ObjNavEpisode(episode_id, episode_id.split("/")[0], "synthetic", "val", target, camera, actions, 500,
                         metadata=dict(metadata or {}))


def red_level(step: int) -> int:
    """The constant red value of the frame of ``step``: distinct for the first 255 steps."""
    return (17 + 31 * step) % 256


def pattern(step: int) -> np.ndarray:
    """The ``(H, W, 3)`` uint8 RGB frame of ``step``."""
    rgb = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    rgb[..., 0] = red_level(step)
    rgb[..., 1] = np.linspace(0, 255, WIDTH).astype(np.uint8)[None, :]
    rgb[..., 2] = 255 - red_level(step)
    return rgb


def pose(step: int) -> AgentPose:
    """A pose of its step with no round numbers in it: position, heading and camera pitch all differ per step."""
    return AgentPose(1.0 / 3.0 + 0.25 * step, -2.0 / 7.0 * step, 0.1 * (step % 2),
                     0.5235987755982988 * step - 3.0, 0.5235987755982988 * (step % 3 - 1))


def frame(ep: ObjNavEpisode, step: int, at: AgentPose = None, rgb: np.ndarray = None) -> ObjNavObservation:
    """The observation of ``step``: :func:`pattern` over a flat depth image at :func:`pose` (or ``at`` / ``rgb``)."""
    return ObjNavObservation(pattern(step) if rgb is None else rgb, np.full((HEIGHT, WIDTH), 2.0, np.float32),
                             pose(step) if at is None else at, ep.camera, ep.target_category, step)
