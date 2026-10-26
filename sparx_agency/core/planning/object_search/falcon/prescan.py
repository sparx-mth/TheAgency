"""A goal-free FALCON prescan on a coarse copy of the map, before the room reasoning begins."""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.planning.object_search.falcon.ground_height import level_camera
from sparx_agency.core.planning.object_search.world_model.coarse_map import UNKNOWN_RULES, downsample_world


@dataclass(frozen=True)
class PrescanSettings:
    """``steps``: actions the prescan may take, counted in the episode budget (0 turns it off); ``resolution_factor``: fine cells per planning cell."""

    steps: int = 0
    resolution_factor: int = 3
    unknown_rule: str = "all"

    def __post_init__(self):
        for name in ("steps", "resolution_factor"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("prescan.%s must be a non-negative integer" % name)
        if self.unknown_rule not in UNKNOWN_RULES:
            raise ValueError("prescan.unknown_rule must be one of %s" % ", ".join(UNKNOWN_RULES))
        if self.resolution_factor < 1:
            raise ValueError("prescan.resolution_factor must be at least 1")


class FalconPrescan:
    """Goal-free planar FALCON plans on the coarse copy while the fine map keeps every frame; then the loop reasons."""

    def __init__(self, policy, settings):
        from sparx_agency.core.planning.object_search.baselines.falcon_yolo import BaselineSettings, GoalFreeFalcon
        self.policy = policy
        self.settings = settings
        policy.baseline = BaselineSettings()
        self.explorer = GoalFreeFalcon(policy)
        self.actions = 0
        self.ended = None
        self.rotation_fallback = False
        self._step = None

    @property
    def active(self):
        return self.ended is None

    def plan(self, obs, world):
        """The next prescan command, or None once the budget is spent or FALCON has nothing left to explore."""
        if not self.active:
            return None
        if self.actions >= self.settings.steps:
            return self._end("budget")
        command = self.explorer.plan(level_camera(obs), downsample_world(world, self.settings.resolution_factor, self.settings.unknown_rule,
                                                          (obs.pose.x, obs.pose.y)))
        if command.stop:
            return self._end("exhausted")
        if obs.step != self._step:
            self._step = obs.step
            self.actions += 1
        return command

    def filter_action(self, obs, action):
        return self.explorer.filter_action(obs, action)

    def blocked(self, obs):
        self.explorer.blocked(obs)

    def suspend(self, reason):
        self.explorer.suspend(reason)

    def _end(self, why):
        """Release FALCON's route; a FALCON warm-up that gave up before a rotation's worth of actions hands over to the rotation."""
        p = self.policy
        self.ended = why
        p.route_memory.clear("prescan_" + why)
        p._route = p._goal = None
        if why == "exhausted" and p.settings.scan_kind == "falcon" and self.actions < p.warmup_turns:
            p.warmup_actions, p._warmup_pending = 0, True
            self.rotation_fallback = True
        return None

    def diagnostics(self):
        return {"steps": self.settings.steps, "actions": self.actions, "ended": self.ended,
                "rotation_fallback": self.rotation_fallback, "resolution_factor": self.settings.resolution_factor,
                "stats": dict(self.explorer.stats)}
