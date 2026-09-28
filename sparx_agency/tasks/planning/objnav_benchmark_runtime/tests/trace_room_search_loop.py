"""Headless trace of the room-search loop with a crude executor that FOLLOWS the commands.

Not a test: a development probe for reading the loop's decisions on the
production map size with fake services. The agent steps 0.25 m toward the
next waypoint of each ``follow`` command (or turns for a ``hold``), so
arrivals and releases happen the way the action converter would make them
happen, instead of a scripted straight line that ignores what it was told.

Run from the repo root::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/python -m \
        sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.trace_room_search_loop
"""
from __future__ import annotations

import math
import statistics
from dataclasses import replace

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeDetector, FakeLLM, observation


def execute(pose, command, step_m, turn_rad):
    """One discrete action's worth of the command, on a point agent."""
    if command.waypoints:
        ahead = [w for w in command.waypoints if math.dist((pose.x, pose.y), w) > 0.05]
        if ahead:
            target = ahead[0]
            heading = math.atan2(target[1] - pose.y, target[0] - pose.x)
            if abs(normalize_angle(heading - pose.yaw)) > turn_rad / 2:
                return replace(pose, yaw=normalize_angle(pose.yaw + math.copysign(turn_rad, normalize_angle(heading - pose.yaw)))), DiscreteAction.TURN_LEFT
            reach = min(step_m, math.dist((pose.x, pose.y), target))
            return replace(pose, x=pose.x + reach * math.cos(pose.yaw), y=pose.y + reach * math.sin(pose.yaw)), DiscreteAction.MOVE_FORWARD
    if command.final_yaw is not None and abs(normalize_angle(command.final_yaw - pose.yaw)) > 1e-3:
        return replace(pose, yaw=normalize_angle(pose.yaw + math.copysign(turn_rad, normalize_angle(command.final_yaw - pose.yaw)))), DiscreteAction.TURN_LEFT
    return pose, DiscreteAction.TURN_LEFT


def main(steps=60):
    episode = ObjNavEpisode("synthetic/0", "synthetic", "gibson", "val", "chair",
                            PROTOCOL.camera(), PROTOCOL.actions(), 500)
    policy = RPTSearchPolicy(FakeDetector("bed"), FakeLLM(), RPTSettings())
    policy.reset(episode, gibson_label_mapper().target_labels("chair"))
    pose = AgentPose(0.0, 0.0, 0.0, 0.0)
    spec = episode.action_spec
    kinds = []
    for step in range(steps):
        obs = replace(observation(episode, step, depth=3.0), pose=pose)
        command = policy.plan(obs)
        pose, action = execute(pose, command, spec.forward_step_m, spec.turn_angle_rad)
        policy.notify_action(obs, action)
        kinds.append(command.info.get("kind"))
        goal = policy.route_memory.goal
        print("step %2d (%5.2f,%5.2f) yaw %5.2f  sup=%-8s room=%s local=%2d kind=%-10s facts=%s goal=%s" % (
            step, obs.pose.x, obs.pose.y, obs.pose.yaw, policy.supervisor.state, policy.supervisor.room_id,
            policy.loop.local_steps, command.info.get("kind"),
            {pid: f.frontier_clusters for pid, f in policy.graph.facts.items()},
            None if goal is None else tuple(round(v, 2) for v in goal)))
    print("history", policy.supervisor.history)
    for event in policy.loop.events:
        print("  ", event)
    print("loop stats", {k: v for k, v in policy.loop.stats.items() if v})
    print("sweep stats", policy.sweep.stats)
    print("llm queries", policy.graph.queries, "oracle reuses", policy.graph.oracle_reuses,
          "holds", kinds.count(None), "rooms", len(policy.graph.registry.rooms))
    for key in ("mapping", "scene_graph", "room_reasoning", "room_selection", "room_confinement",
                "astar", "policy_decision"):
        values = policy.telemetry.latencies.get(key, [])
        if values:
            print("%-18s n=%3d  median %6.1f ms  max %6.1f ms" % (key, len(values), statistics.median(values), max(values)))


if __name__ == "__main__":
    main()

