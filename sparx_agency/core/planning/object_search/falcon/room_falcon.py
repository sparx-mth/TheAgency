"""Goal-free FALCON confined to one room's mask on the fine map: viewpoint after viewpoint until the room has none left."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

ROOM_MARGIN_M = 0.5


def room_scope(world, room_mask, margin_m=ROOM_MARGIN_M):
    """The room's cells plus a margin: the watershed erodes a room's mask, so its frontier cells sit just outside it."""
    cells = int(np.ceil(margin_m / world.resolution))
    return binary_dilation(np.asarray(room_mask, dtype=bool), iterations=cells)


def _scoped_falcon_class():
    from sparx_agency.core.planning.object_search.baselines.falcon_yolo import GoalFreeFalcon

    class RoomScopedFalcon(GoalFreeFalcon):
        """The goal-free explorer with the room as its scope and no relocation to frontiers elsewhere on the storey."""

        room_scope = None

        def _disc(self, world, pose):
            return self.room_scope

        def _relocate(self, obs, world):
            return None

    return RoomScopedFalcon


class RoomFalcon:
    """A room visit driven by FALCON's own viewpoints, planned on the observed fine map inside the room."""

    def __init__(self, policy):
        from sparx_agency.core.planning.object_search.baselines.falcon_yolo import BaselineSettings
        if getattr(policy, "baseline", None) is None:
            policy.baseline = BaselineSettings()
        self.policy = policy
        self.explorer = _scoped_falcon_class()(policy)
        self.visits = 0

    def plan(self, obs, world, scope):
        """The next command inside ``scope``, or None when FALCON has no viewpoint left in it."""
        self.explorer.room_scope = scope
        command = self.explorer.plan(obs, world)
        if command.stop or str(command.info.get("kind", "")).startswith("exploration_exhausted"):
            return None
        return command

    def end(self):
        """The room is done: no viewpoint, route or exhausted-decision count carries into the next one."""
        self.visits += 1
        self.explorer.current = None
        self.explorer.exhausted = 0
        self.policy.route_memory.clear("room_falcon_end")
        self.policy._route = self.policy._goal = None

    def filter_action(self, obs, action):
        return self.explorer.filter_action(obs, action)

    def blocked(self, obs):
        self.explorer.blocked(obs)

    def suspend(self, reason):
        self.explorer.suspend(reason)

    def diagnostics(self):
        return {"visits": self.visits, "stats": dict(self.explorer.stats)}
