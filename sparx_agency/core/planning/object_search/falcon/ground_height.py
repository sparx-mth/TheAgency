"""FALCON is written for aerial robots; on a ground robot it plans at one fixed height with a level camera."""
from __future__ import annotations

from dataclasses import replace


def level_camera(obs):
    """``obs`` with the camera pitch zeroed: FALCON sees every cell at the one body height, from a level camera."""
    return replace(obs, pose=replace(obs.pose, camera_pitch=0.0))
