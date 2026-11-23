"""Rotate at one position to cover the camera's horizontal field on a new storey."""
import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


def rotation_command(p, obs):
    budget = getattr(p, "warmup_turns", p.settings.warmup_steps)
    if p.warmup_actions >= budget:
        return None
    p._action_owner = "warmup"
    here = (float(obs.pose.x), float(obs.pose.y))
    anchor = getattr(p, "_warmup_xy", None)
    if anchor is not None and p.warmup_actions > 0 and math.dist(here, anchor) > 0.5:
        p.warmup_actions = 0  # a takeover moved the camera mid-rotation
    if p.warmup_actions == 0:
        p._warmup_xy = here
    return NavigationCommand.hold(final_yaw=normalize_angle(obs.pose.yaw + p.episode.action_spec.turn_angle_rad),
                                  info={"kind": "warmup", "warmup_step": p.warmup_actions + 1, "warmup_budget": budget})
