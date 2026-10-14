"""Headless trace of the room-search loop with the real action converter on a point agent.

Not a test: a development probe for reading the loop's decisions on the
production map size with fake services. Each command goes through the same
:class:`DiscreteActionConverter` the headless agent uses, and the chosen
action moves a collision-free point agent by the benchmark's own geometry,
so arrivals, releases and the turns a route costs happen the way the
converter makes them happen -- an idle result is spent as the headless
agent's idle TURN_LEFT. (The first version aimed at the first waypoint by
hand and dithered for forty actions at a route whose first point lay behind
the agent; the converter projects onto the path and does not.)

Run from the repo root::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m \
        sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.trace_room_search_loop
"""
from __future__ import annotations

import statistics
from dataclasses import replace

from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeDetector, FakeLLM, observation


def execute(converter, pose, command, spec):
    """One discrete action's worth of the command, on a point agent, by the real converter."""
    result = converter.step(pose, command)
    action = result.action if result.action is not None else DiscreteAction.TURN_LEFT
    if action == DiscreteAction.STOP:
        return pose, action
    return apply_action(pose, action, spec), action


def main(steps=60):
    episode = ObjNavEpisode("synthetic/0", "synthetic", "gibson", "val", "chair",
                            PROTOCOL.camera(), PROTOCOL.actions(), 500)
    policy = RPTSearchPolicy(FakeDetector("bed"), FakeLLM(), RPTSettings())
    policy.reset(episode, gibson_label_mapper().target_labels("chair"))
    pose = AgentPose(0.0, 0.0, 0.0, 0.0)
    spec = episode.action_spec
    converter = DiscreteActionConverter(spec, policy.converter_params)
    kinds, actions = [], []
    for step in range(steps):
        obs = replace(observation(episode, step, depth=3.0), pose=pose)
        command = policy.plan(obs)
        pose, action = execute(converter, pose, command, spec)
        policy.notify_action(obs, action)
        kinds.append(command.info.get("kind"))
        actions.append(action.name)
        goal = policy.route_memory.goal
        print("step %2d (%5.2f,%5.2f) yaw %5.2f %-12s sup=%-8s room=%s local=%2d kind=%-10s facts=%s goal=%s" % (
            step, obs.pose.x, obs.pose.y, obs.pose.yaw, action.name, policy.supervisor.state, policy.supervisor.room_id,
            policy.loop.local_steps, command.info.get("kind"),
            {pid: f.frontier_clusters for pid, f in policy.graph.facts.items()},
            None if goal is None else tuple(round(v, 2) for v in goal)))
    print("history", policy.supervisor.history)
    for event in policy.loop.events:
        print("  ", event)
    print("loop stats", {k: v for k, v in policy.loop.stats.items() if v})
    print("sweep stats", policy.sweep.stats)
    print("actions", {name: actions.count(name) for name in sorted(set(actions))})
    print("llm queries", policy.graph.queries, "oracle reuses", policy.graph.oracle_reuses,
          "label queries", policy.graph.label_tracker.queries, "reasoning calls", getattr(policy.llm_client, "reasoning_calls", "?"),
          "holds", kinds.count(None), "rooms", len(policy.graph.registry.rooms))
    for key in ("mapping", "scene_graph", "room_reasoning", "room_relabel", "room_selection",
                "room_confinement", "astar", "policy_decision"):
        values = policy.telemetry.latencies.get(key, [])
        if values:
            print("%-18s n=%3d  median %6.1f ms  max %6.1f ms" % (key, len(values), statistics.median(values), max(values)))


if __name__ == "__main__":
    main()

