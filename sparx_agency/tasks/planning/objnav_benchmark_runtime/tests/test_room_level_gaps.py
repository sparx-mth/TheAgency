"""Regressions for room-level GAPs (2026-10-21): exits are a room's attributes, never nodes; the room's visit walks them.

Built on the two-room world of ``test_room_search_loop`` under the default
``visit="explore"``: room A (west) with an unexplored west end (one exit),
room B (east) with a small unknown patch by the door and an unexplored east
end (two exits). The scene graph is installed by hand, the frontier inventory
is computed for real from it, the oracle is the fixture's counter (it records
the exits it was handed per room), the solver is fixed, and A* is real.

The decision hierarchy under test: the GLOBAL loop values rooms (and stairs)
-- an exit gets no LLM number and is no RPT* node, however many a room has --
and a chosen room's LOCAL phase walks its GAPs until none is left or the
visit's bound runs out, after which control returns to the global loop. A
room a rotation finished with GAPs still unwalked is offered again for them
(reopened), walked and not re-scanned; the deprecated ``exits_as_nodes`` mode
keeps the 2026-10-04 behaviour (``test_opening_nodes``, ``test_explore_visit``).
"""
from __future__ import annotations

import pytest

from sparx_agency.core.mapping.topology.search_node_oracle import OPENING, ROOM, format_node, system_prompt
from sparx_agency.core.planning.exploration.object_search_supervisor import BUDGET_SPENT, EXHAUSTED, SEARCH, TRANSIT
from sparx_agency.core.planning.object_search.rooms.opening_nodes import (
    LANDMARK, Opening, OpeningSettings, RoomExits, opening_options, room_exits)
from sparx_agency.core.planning.object_search.rooms.room_scans import SCAN_POINT_INSIDE
from sparx_agency.core.planning.object_search.rooms.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.search_panel import render_search_panel
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_explore_visit import (
    LEGACY, explore_policy, resolve_around, walk)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import openings_of, refresh, west_gap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import IN_A, IN_B, cell_in, obs_at
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot


# -- the settings ----------------------------------------------------------------------------------------
def test_room_level_gaps_is_the_explore_visits_default_and_the_old_design_is_a_deprecated_knob():
    assert LoopSettings().room_level_gaps and not LoopSettings().openings.exits_as_nodes
    assert not LoopSettings(openings=OpeningSettings(exits_as_nodes=True)).room_level_gaps
    assert not LoopSettings(openings={"exits_as_nodes": True}).room_level_gaps, "the dict form of --policy-config"
    assert not LoopSettings(visit="scan").room_level_gaps and not LoopSettings(visit="sweep").room_level_gaps, (
        "no local walk to fold an exit into: those visits keep the exits as peek nodes")
    assert not LoopSettings(openings=OpeningSettings(enabled=False)).room_level_gaps, "no openings read at all"
    with pytest.raises(ValueError):
        OpeningSettings(exits_as_nodes=1)


# -- the per-room summary and the room's node line ---------------------------------------------------------
def test_room_exits_sums_the_exits_per_room_and_leaves_landmarks_and_roomless_exits_out():
    def exit_(index, pid, door=None, glimpsed=(), kind="exit"):
        return Opening(index=index, xy=(0.0, 0.0), heading=0.0, size_cells=8, geodesic_m=1.0, room_pid=pid,
                       door_id=door, glimpsed=tuple(glimpsed), kind=kind)
    summary = room_exits([exit_(0, 7, door=3, glimpsed=("toilet",)), exit_(1, 7, glimpsed=("sink", "toilet")),
                          exit_(2, 2), exit_(3, None), exit_(50_004, 7, glimpsed=("sofa",), kind=LANDMARK)])
    assert summary == {7: RoomExits(count=2, doorways=1, glimpsed=("sink", "toilet")), 2: RoomExits(count=1)}
    assert room_exits([]) == {}


def test_the_oracle_is_shown_rooms_only_each_with_its_exits_and_no_exit_gets_a_number():
    """Room A is valued under the solver's floor -- under exits-as-nodes its gap would have stayed a node of its
    own at the provisional value; here it is room A's attribute and nothing else."""
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    before = openings_of(policy, episode, world)
    assert len(before) == 3, "room A's west end; room B's patch by the door and its east end"
    policy.graph.probs = {0: 0.0, 1: 0.6}
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert reasoned == [0] and policy.reasoned_with[0]["stairs"] == [], "the oracle saw the rooms, not the gaps"
    assert policy.reasoned_with[0]["exits"] == {0: RoomExits(count=1), 1: RoomExits(count=2)}
    assert policy.loop._folded_now == {o.node_id for o in before} and policy.loop.stats["openings_folded"] == 3
    assert set(policy.loop._openings) == {o.node_id for o in before}, "still on record for the HUD and the landmark lookups"
    assert not any(k >= 200_000 for k in policy.loop.estimates), "no opening in the estimate: nothing RPT* was handed"
    assert all(pid < 200_000 for pid in policy.supervisor.calls[-1]["instance"].index_to_pid if pid >= 0)
    assert policy.loop.stats["openings_provisional"] == 0 and policy.loop._exits_at_call == {0: 1, 1: 2}
    assert policy.loop.order == (1,) and command.info["kind"] == "transit/1"
    assert policy.loop.estimate_events[0]["openings"] == [] and policy.loop.estimate_events[0]["exits"] == {
        "0": {"count": 1, "doorways": 0, "glimpsed": []}, "1": {"count": 2, "doorways": 0, "glimpsed": []}}
    assert policy.loop.estimate_events[0]["reopened"] == []
    assert "3 openings not yet looked into" in policy.reasoned_with[0]["context"].house, "the HOUSE line counts every exit"
    # The real scene graph renders the attributes on the room's line, as the prompt describes them.
    nodes = {n.id: n for n in policy.graph.nodes(world, 0, exits=policy.reasoned_with[0]["exits"])}
    assert nodes[1].kind == ROOM and nodes[1].exits == 2 and nodes[1].doorways == 0 and nodes[1].glimpsed == ()
    assert "  exits=2  " in format_node(nodes[1]) and "  exits=1  " in format_node(nodes[0])
    assert OPENING not in {n.kind for n in nodes.values()} and "OPENING" not in system_prompt(list(nodes.values()))
    # The frozen configuration says so.
    configuration = policy.configuration()
    assert configuration["node_oracle"]["nodes"].endswith("each room's exits are attributes on its line, not nodes")
    assert configuration["openings"].startswith("every exit of the storey") and "landmark" in configuration["openings"]
    assert "room-level GAPs" in configuration["room_visit"]
    # The HUD shows every live exit, folded into its room, and renders.
    policy.last_world = world
    search = search_snapshot(policy)
    assert {o["id"] for o in search["openings"]} == {o.node_id for o in before} and all(o["folded"] for o in search["openings"])
    assert render_search_panel(search, (320, 720)).shape == (720, 320, 3)


def test_a_doorway_and_a_glimpse_through_it_are_the_rooms_attributes():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    gap = west_gap(openings_of(policy, episode, world))
    door = Opening(index=gap.index, xy=gap.xy, heading=gap.heading, size_cells=gap.size_cells, geodesic_m=gap.geodesic_m,
                   room_pid=0, door_id=4, glimpsed=("toilet",))
    [option] = opening_options(policy, [door])
    summary = room_exits([option])
    assert summary == {0: RoomExits(count=1, doorways=1, glimpsed=("toilet",))}
    [node] = [n for n in policy.graph.nodes(world, 0, exits=summary) if n.id == 0]
    line = format_node(node)
    assert "exits=1 (1 doorway)" in line and line.endswith("glimpsed through its exits: toilet")


# -- the local phase: the GAPs of the chosen room, to the end or the bound; then the global loop -------------
def test_a_room_a_rotation_finished_with_gaps_left_is_reopened_for_them_entered_at_a_frontier_and_walked():
    """The warm-up spun a circle in the spawn room whose west end was never looked at. Under exits-as-nodes the
    room was finished and its gap carried the search on as a node; here the room itself is the node again."""
    policy, episode, world, rooms, reasoned = explore_policy(order=(0, 1))
    policy.scans.record(obs_at(episode, 0, IN_A), rooms[0], source="warmup")
    obs = obs_at(episode, 0, IN_A)
    command = policy.loop.plan(obs, world)
    assert policy.loop._finished[0][0] == SCAN_POINT_INSIDE, "the ledger has room A finished"
    assert policy.loop._reopened == {0} and 0 not in policy.loop._excluded and policy.loop._revisits == set()
    assert policy.loop.stats["rooms_reopened"] == 1
    [event] = [e for e in policy.loop.events if e["event"] == "room_reopened"]
    assert event["room"] == 0 and event["how"] == SCAN_POINT_INSIDE and event["frontier"] == 1 and event["exits"] == 1
    assert policy.reasoned_with[0]["exclude"] == () and policy.reasoned_with[0]["scanned"][0][0] == "full rotation", (
        "shown to the oracle as scanned, with its exits, not hidden")
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0 and command.info["kind"] == "transit/0"
    assert policy.loop.events[-1]["event"] == "transit" and policy.loop.events[-1]["entry"] == "frontier"
    assert policy.loop.estimates[0]["entry"] == "frontier", "entered at its best frontier, not the vantage point"
    settings = policy.loop.settings
    assert policy.loop._service_steps_for(0, obs, world) == settings.openings.service_steps + settings.explore_frontier_steps
    assert policy.loop._service_last[0] == pytest.approx(8.0), "priced by its frontiers, not as a second-pass scan"
    assert not any(e["event"] == "finished_in_transit" for e in policy.loop.events), "the walk is the point of going"
    # Arrival a step inside the gap: the local phase walks the GAPs left, no circle, and ends when none is left.
    entry = policy.route_memory.goal
    inside = (entry[0] + 0.5, entry[1])
    assert cell_in(world, rooms[0], inside)
    policy.sight.observe(obs_at(episode, 1, inside), world)
    first = policy.loop.plan(obs_at(episode, 1, inside), world)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 0 and policy.loop._explore["room"] == 0
    assert first.info["kind"] == "frontier" and first.waypoints and policy.loop._explore["frontier_left"] == 1
    policy.loop.charge()
    commands, last, where = walk(policy, episode, world, 2, inside)
    assert {c.info.get("kind") for c in commands[:-1]} <= {"frontier"} and not any(
        c.info.get("kind") == "room_scan" for c in commands)
    assert policy.supervisor.history[-1][:2] == (0, EXHAUSTED)
    assert not any(e["event"] == "finished_by_rotation" for e in policy.loop.events)
    # Every GAP walked: the room is finished for good, control went back to the global loop, room B is next.
    assert policy.loop._excluded == {0: "scanned:" + SCAN_POINT_INSIDE} and policy.loop._reopened == set()
    assert len(reasoned) == 2 and policy.loop.stats["loop_points"] == 1
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 1 and commands[-1].info["kind"] == "transit/1"
    # The deprecated mode: the room is withheld and its gap is the node.
    legacy, episode_l, world_l, rooms_l, _ = explore_policy(order=(0, 1), **LEGACY)
    legacy.scans.record(obs_at(episode_l, 0, IN_A), rooms_l[0], source="warmup")
    gap = west_gap(openings_of(legacy, episode_l, world_l))
    legacy.loop.plan(obs_at(episode_l, 0, IN_A), world_l)
    assert legacy.loop._excluded == {0: "scanned:" + SCAN_POINT_INSIDE} and legacy.loop._reopened == set()
    assert gap.node_id in legacy.reasoned_with[0]["stairs"] and legacy.reasoned_with[0]["exits"] == {1: RoomExits(count=2)}


def test_a_glance_mid_visit_finishes_the_room_in_the_ledger_but_the_visit_walks_its_gaps_to_the_end():
    """The counterpart of ``test_a_full_rotation_turned_on_the_way_finishes_the_room_and_ends_the_visit_at_once``:
    the rotation saw room B's floor, not what lies behind its east end, and the exits are nobody's nodes now."""
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    entry = policy.route_memory.goal
    inside = (entry[0] - 0.3, entry[1]) if cell_in(world, rooms[1], (entry[0] - 0.3, entry[1])) else entry
    first = policy.loop.plan(obs_at(episode, 1, inside), world)
    assert policy.supervisor.state == SEARCH and first.info["kind"] == "frontier"
    policy.loop.charge()
    policy.scans.record(obs_at(episode, 3, IN_B), rooms[1], source="glance")      # a glance completed in room B
    policy.sight.observe(obs_at(episode, 4, IN_B), world)
    command = policy.loop.plan(obs_at(episode, 4, IN_B), world)
    assert policy.supervisor.state == SEARCH and policy.loop.room_id == 1, "the visit goes on"
    assert command.info["kind"] == "frontier" and command.waypoints
    assert policy.loop.stats["finished_by_rotation"] == 0 and policy.loop._explore["frontier_left"] == 2
    assert policy.scans.status(world, 1, rooms[1]) == SCAN_POINT_INSIDE, "finished in the ledger all the same"
    policy.loop.charge()
    commands, last, where = walk(policy, episode, world, 5, IN_B)
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.loop._excluded == {1: "scanned:" + SCAN_POINT_INSIDE}, "no GAP left: finished, and not reopened"
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0 and len(reasoned) == 2
    # With frontier that is nobody's exit (a gap too narrow to count) the rotation ends the visit as before.
    policy2, episode2, world2, rooms2, _ = explore_policy(order=(1, 0), openings=OpeningSettings(min_cells=1000))
    assert openings_of(policy2, episode2, world2) == [], "no exit of the storey clears the bar"
    policy2.loop.plan(obs_at(episode2, 0, IN_A), world2)
    entry2 = policy2.route_memory.goal
    inside2 = (entry2[0] - 0.3, entry2[1]) if cell_in(world2, rooms2[1], (entry2[0] - 0.3, entry2[1])) else entry2
    policy2.loop.plan(obs_at(episode2, 1, inside2), world2)
    assert policy2.supervisor.state == SEARCH and policy2.loop._explore["frontier_left"] >= 1
    policy2.loop.charge()
    policy2.scans.record(obs_at(episode2, 3, IN_B), rooms2[1], source="glance")
    policy2.loop.plan(obs_at(episode2, 4, IN_B), world2)
    assert policy2.supervisor.history[-1][:2] == (1, EXHAUSTED) and policy2.loop.stats["finished_by_rotation"] == 1
    assert policy2.loop._excluded == {1: "scanned:" + SCAN_POINT_INSIDE}, "frontier without an exit does not reopen it"


def test_the_local_phase_ends_on_its_bound_and_control_returns_to_the_global_loop_on_that_action():
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0), explore_visit_steps=3)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    entry = policy.route_memory.goal
    policy.loop.plan(obs_at(episode, 1, entry), world)
    assert policy.supervisor.state == SEARCH and policy.loop.visit_budget() == 3 and reasoned == [0]
    for _ in range(3):
        policy.loop.charge()
    command = policy.loop.plan(obs_at(episode, 2, entry), world)
    assert policy.supervisor.history[-1][:2] == (1, BUDGET_SPENT)
    assert reasoned == [0, 2] and policy.loop.stats["loop_points"] == 1, "the release is the loop point: re-valued, re-solved"
    assert 1 not in policy.loop._excluded and 1 not in policy.loop._reopened, "unfinished, with GAPs left: a plain node"
    assert policy.supervisor.state == TRANSIT and command.info["kind"].startswith("transit/")


def test_a_finished_room_with_gaps_left_is_released_in_transit_only_once_the_way_there_showed_the_last_of_them():
    policy, episode, world, rooms, _ = explore_policy(order=(0, 1))
    policy.scans.record(obs_at(episode, 0, IN_A), rooms[0], source="warmup")
    policy.loop.plan(obs_at(episode, 0, IN_B), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0 and policy.loop._reopened == {0}
    # The west end gets mapped on the way (a look through the door): nothing is left to walk to.
    gap = west_gap(openings_of(policy, episode, world))
    resolve_around(world, gap.xy, 3.0)
    refresh(policy, world, IN_B)
    policy.loop.plan(obs_at(episode, 1, IN_B), world)
    assert policy.loop.stats["finished_in_transit"] == 1 and policy.supervisor.history[-1][:2] == (0, EXHAUSTED)
    assert policy.loop._excluded == {0: "scanned:" + SCAN_POINT_INSIDE} and policy.supervisor.room_id == 1


def test_a_new_exit_on_an_offered_room_buys_the_oracle_a_call_at_most_every_revalue_actions():
    """The room-level counterpart of the opening-node throttle: a room's line changed (one more exit), so the
    model is re-asked -- not on every re-snap of the map, and never when the room is no node anyway."""
    policy, episode, world, rooms, reasoned = explore_policy(order=(1, 0), openings=OpeningSettings(revalue_actions=5))
    for pid in (0, 1):                                              # both rooms cooling: nothing to choose, the fallback carries
        policy.supervisor.inner._cooling[pid] = 0.0
        policy.supervisor.inner._cooling_verdict[pid] = EXHAUSTED
    policy.graph.probs = {0: 0.6, 1: 0.4}
    policy.loop._needs_reason = False
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert reasoned == [0] and policy.loop._exits_at_call == {0: 1, 1: 2}, "exits the estimate has never seen: valued once"
    policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert reasoned == [0], "an unchanged map: nothing to ask"
    policy.loop._exits_at_call[1] = 1                               # as if room B's east end appeared after the call
    policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert reasoned == [0], "two actions later: not yet"
    policy.loop.plan(obs_at(episode, 5, IN_A), world)
    assert reasoned == [0, 5] and policy.loop._exits_at_call == {0: 1, 1: 2}, "five actions later: valued"
    policy.loop.plan(obs_at(episode, 9, IN_A), world)
    assert reasoned == [0, 5], "and the unchanged map asks nothing more"
