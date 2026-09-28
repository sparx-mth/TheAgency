"""Inactive local-route storage, kept separate from target/transit execution."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


class LocalRouteBank:
    """Store actual route cursors; scope/obstacle validation still owns reuse."""

    def __init__(self):
        self.saved = []
        self.suspended = self.resumed = 0

    def suspend(self, mask, view, policy, action):
        if view is None or policy.route_memory.path is None:
            return
        self.saved.append((mask.copy(), view, deepcopy(policy.route_memory), policy._route, policy._goal, action))
        self.suspended += 1

    def restore(self, mask, policy, action):
        matches = []
        for i, (previous, view, _, _, _, _) in enumerate(self.saved):
            overlap = np.count_nonzero(mask & previous)
            denominator = min(np.count_nonzero(mask), np.count_nonzero(previous))
            if denominator and overlap / denominator >= 0.25 and mask[view.cell[1], view.cell[0]]:
                matches.append((overlap, i))
        if not matches:
            return None
        _, index = max(matches)
        _, view, memory, route, goal, paused = self.saved.pop(index)
        memory.resume_after_pause(action - paused)
        policy.route_memory, policy._route, policy._goal = memory, route, goal
        self.resumed += 1
        return view

    def diagnostics(self):
        return {"suspended": self.suspended, "resumed": self.resumed, "pending": len(self.saved)}

