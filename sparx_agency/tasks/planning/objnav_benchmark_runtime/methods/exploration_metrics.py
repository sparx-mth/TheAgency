"""Observed-only diagnostics; never consumes evaluator success or goal coordinates."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np


class ExplorationMetrics:
    """Coverage is ever-observed ternary-map area, not ground-truth coverage.

    Initial coverage is excluded from gained-per-action. Samples correspond to
    the observations at decision indices (the terminal observation is not fused
    just to improve a metric). Floor-local masks cannot overlay different floors.
    """

    def __init__(self):
        self.curve = []
        self.latencies = defaultdict(list)
        self.actions = Counter()
        self.allocation = Counter()
        self.seen = None
        self.floor = None
        self.previous_floor_area = 0.0
        self.floor_masks = {}
        self.floor_areas = {}
        self.last_pose = None
        self.last_action = None
        self.repeated_turns = self.turn_reversals = self.revisits = 0
        self.no_progress_actions = 0
        self.no_progress_decision_ms = 0.0
        self.visited = set()
        self.last_cell = None
        self.phase = "startup"

    def observe(self, observation, world, floor):
        if self.floor != floor:
            self.seen = self.floor_masks.setdefault(floor, np.zeros(world.grid.shape, bool))
            self.floor = floor
        self.seen |= world.grid != world.values.unknown
        self.floor_areas[floor] = float(self.seen.sum()) * world.resolution ** 2
        area = sum(self.floor_areas.values())
        gain = 0.0 if not self.curve else max(0.0, area - self.curve[-1]["observed_m2"])
        pose = observation.pose
        cell = (floor, int(math.floor(pose.x / 0.5)), int(math.floor(pose.y / 0.5)))
        if cell != self.last_cell:
            self.revisits += int(cell in self.visited)
            self.visited.add(cell)
            self.last_cell = cell
        if self.last_pose is not None and math.dist((pose.x, pose.y), self.last_pose) < 0.05 and gain < 0.01:
            self.no_progress_actions += 1
            if self.latencies["policy_decision"]:
                self.no_progress_decision_ms += self.latencies["policy_decision"][-1]
        self.last_pose = (pose.x, pose.y)
        self.curve.append({"action": observation.step, "floor": floor, "observed_m2": round(area, 4),
                           "new_m2": round(gain, 4), "phase": self.phase})
        return gain

    def emitted(self, observation, action, phase):
        name = action.name
        self.actions[name] += 1
        self.allocation[phase] += 1
        turn = "TURN" in name
        previous_turn = self.last_action is not None and "TURN" in self.last_action
        self.repeated_turns += int(turn and previous_turn and name == self.last_action)
        self.turn_reversals += int(turn and previous_turn and name != self.last_action)
        self.last_action = name
        if self.curve:
            self.curve[-1].update(emitted_action=name, phase=phase)

    def report(self):
        samples = {}
        for stage, values in self.latencies.items():
            samples[stage] = {"count": len(values), "total_ms": float(sum(values)),
                              "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                              "p99_ms": float(np.percentile(values, 99)), "max_ms": float(max(values))}
        gain = self.curve[-1]["observed_m2"] - self.curve[0]["observed_m2"] if self.curve else 0.0
        actions = sum(self.actions.values())
        return {"coverage_definition": "ever-observed occupancy area proxy; stable floors deduplicated; transition surfaces excluded; initial excluded from gain; decision observations only",
                "observed_area_by_floor_m2": dict(self.floor_areas),
                "observed_gain_m2": gain, "coverage_gain_m2_per_action": gain / max(1, actions),
                "coverage_curve": list(self.curve), "actions": dict(self.actions),
                "actions_by_phase": dict(self.allocation), "latency": samples,
                "revisits_0_5m_cells": self.revisits, "consecutive_same_turns": self.repeated_turns,
                "immediate_turn_reversals": self.turn_reversals,
                "no_progress_decision_wall_s": self.no_progress_decision_ms / 1000,
                "no_translation_or_coverage_actions": self.no_progress_actions}

