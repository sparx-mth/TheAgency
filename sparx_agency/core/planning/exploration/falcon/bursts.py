"""Action-budget orchestration, independent of simulator, ROS, LLM and planners."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


EXPLORE, REASON, SELECT, TRANSIT = "local_exploration", "room_reasoning", "room_selection", "transit"
VERIFY, RECOVER, DONE = "target_verification", "recovery", "done"


@dataclass
class Burst:
    token: int
    region: str
    start_action: int
    actions: int = 0
    status: str = "partial"
    reason: str = "active"


class BurstMachine:
    """Only explicit room selection can authorize a new burst.

    Room labels/doorway crossings cannot reset the current token. A rejected
    target resumes that same token and remaining allowance. Verification and
    recovery inside a burst also consume its allowance; every emitted action
    consumes the global allowance, including STOP and failed forward motion.
    """

    def __init__(self, params, global_limit):
        self.params, self.global_limit = params, global_limit
        self.phase = EXPLORE
        self.actions = 0
        self.burst = None
        self.history, self.transitions = [], []
        self.allocation = Counter()
        self.resume_phase = None
        self.phase_actions = 0
        self.last_step = -1
        self._entry_streak = 0
        self._entry_step = -1

    def transition(self, phase, reason):
        if self.phase != phase:
            self.transitions.append({"action": self.actions, "from": self.phase, "to": phase, "reason": reason})
            self.phase, self.phase_actions = phase, 0

    def begin(self, region):
        if self.burst is not None and self.burst.status == "partial":
            raise RuntimeError("An active/interrupted burst cannot be refilled")
        if self.phase not in (EXPLORE, SELECT, TRANSIT):
            raise RuntimeError("Only startup or selected-room entry can start exploration")
        self.burst = Burst(len(self.history), str(region), self.actions)
        self.history.append(self.burst)
        self.transition(EXPLORE, "selected region entered")

    def charge(self, step):
        if step == self.last_step:
            return
        if step < self.last_step or self.actions >= self.global_limit:
            raise RuntimeError("Action clock regressed or global budget exceeded")
        local = self.burst is not None and self.burst.status == "partial" and self.phase in (EXPLORE, VERIFY, RECOVER)
        if local and self.burst.actions >= self.params.burst_actions:
            raise RuntimeError("Local burst action budget exceeded")
        self.last_step = step
        self.allocation[self.phase] += 1
        self.actions += 1
        self.phase_actions += 1
        if local:
            self.burst.actions += 1

    def check_limits(self):
        if self.actions >= self.global_limit:
            self.end("global_budget")
            self.transition(DONE, "global action limit")
            return
        if self.burst is not None and self.burst.status == "partial" and self.burst.actions >= self.params.burst_actions:
            self.end("action_budget")

    def end(self, reason):
        if self.burst is not None and self.burst.status == "partial":
            self.burst.reason = reason
            self.burst.status = ("budget_exhausted" if reason in ("action_budget", "global_budget")
                                 else "no_useful_frontiers" if reason in ("frontiers_exhausted", "no_useful_reachable_frontiers")
                                 else "partial_ended")
        if self.phase != VERIFY:
            self.transition(REASON, reason)

    def interrupt(self):
        if self.phase == VERIFY:
            return
        self.resume_phase = self.phase
        if self.burst is not None and self.burst.status == "partial":
            self.burst.reason = "target_interruption"
        self.transition(VERIFY, "target candidate")

    def resume(self):
        phase = self.resume_phase or EXPLORE
        self.resume_phase = None
        if self.burst is not None and self.burst.status != "partial" and phase in (EXPLORE, RECOVER):
            phase = REASON
        self.transition(phase, "target rejected or lost")

    def entered(self, inside, step):
        """Debounce entry by actions, never by repeated calls or a new room ID."""
        if step != self._entry_step:
            self._entry_streak = self._entry_streak + 1 if inside else 0
            self._entry_step = step
        return self._entry_streak >= self.params.entry_confirm_actions

    def snapshot(self):
        from dataclasses import asdict
        return {"phase": self.phase, "global_actions": self.actions, "global_limit": self.global_limit,
                "burst": asdict(self.burst) if self.burst is not None else None,
                "bursts": [asdict(b) for b in self.history], "transitions": list(self.transitions),
                "actions_by_phase": dict(self.allocation)}

