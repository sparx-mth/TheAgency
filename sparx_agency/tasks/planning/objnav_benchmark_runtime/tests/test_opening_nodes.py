"""Regressions for openings as nodes (2026-10-04): detection, the weak-type rule, the peek, the record.

Built on the two-room world of ``test_room_search_loop`` under ``visit="scan"``.
Room A (west) has an unexplored west end: a gap at x ~ 1.15 leading west. Room
B (east) has a small unknown patch near the door and an unexplored east end:
two openings. The scene graph is installed by hand and the frontier inventory
is computed for real from it; the oracle is the fixture's counter (every extra
node gets ``stair_prob``), the solver is fixed, and A* is real.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.object_search_supervisor import EXHAUSTED, SEARCH, SELECT, TRANSIT, UNREACHABLE
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import FloorPanels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.opening_nodes import (
    OPENING_NODE_BASE, OpeningRegistry, OpeningSettings, detect_openings, is_opening_node, opening_index_of,
    opening_node_id, unknown_heading)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_nodes import is_stair_node, stair_node_id
from sparx_agency.tasks.planning.objnav_benchmark_runtime.search_panel import _name, render_search_panel
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, IN_B, OCC, UNK, NamingLLM, loop_policy, obs_at, see, two_room_world)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot


def opening_policy(order=(1, 0), llm=None, **overrides):
    """The scan-mode fixture with a live frontier inventory, so the floor's openings can be read."""
    policy, episode, world, rooms, reasoned = loop_policy(order=order, visit="scan", llm=llm, **overrides)
    refresh(policy, world, IN_A)
    return policy, episode, world, rooms, reasoned


def refresh(policy, world, xy, yaw=0.0):
    """The background refresh's accessibility half: the inventory the openings are read from."""
    cost = policy.navigation_cost(world)
    policy.graph.refresh_accessibility(world, cost, xy, yaw, policy.sweep.settings.ranking)
    return cost


def openings_of(policy, episode, world, step=0, xy=IN_A):
    return detect_openings(policy, obs_at(episode, step, xy), world, policy.loop.settings.openings, policy.openings)


def west_gap(openings):
    [gap] = [o for o in openings if o.room_pid == 0]
    return gap


def peek_through(policy, episode, world, xy, start_step, start_yaw, max_turns=12):
    """Feed the loop its own peek turns until the peek ends; return (commands, last step)."""
    commands, yaw, step = [], start_yaw, start_step
    for _ in range(max_turns):
        command = policy.loop.plan(obs_at(episode, step, xy, yaw), world)
        commands.append(command)
        policy.loop.charge()
        if command.info.get("kind") != "peek" or command.waypoints:
            return commands, step
        yaw = float(command.final_yaw)
        step += 1
    raise AssertionError("the peek never ended")


# -- ids -------------------------------------------------------------------------------
def test_opening_ids_live_in_their_own_band_above_rooms_and_stairs():
    assert opening_node_id(3) == OPENING_NODE_BASE + 3 and opening_index_of(opening_node_id(3)) == 3
    assert is_opening_node(opening_node_id(0)) and not is_opening_node(stair_node_id(0)) and not is_opening_node(7)
    assert not is_opening_node(None)
    assert is_stair_node(stair_node_id(5)) and not is_stair_node(opening_node_id(5)), "the stair band ends where the openings begin"
    assert _name({"id": opening_node_id(3), "kind": "opening"}) == "O3"
    assert _name({"id": opening_node_id(3)}) == "O3" and _name({"id": stair_node_id(2)}) == "S2" and _name({"id": 4}) == "R4"


def test_the_registry_keeps_an_openings_id_as_its_frontier_re_snaps_and_retires_what_was_peeked():
    registry = OpeningRegistry(match_m=1.0, done_m=1.0)
    first = registry.identify(0, (1.0, 3.0))
    assert registry.identify(0, (1.4, 3.3)) == first, "the same opening, its frontier moved a little"
    assert registry.identify(0, (4.0, 3.0)) == first + 1, "another opening"
    assert registry.identify(1, (1.0, 3.0)) == first + 2, "the same place on another storey is another opening"
    assert not registry.peeked(0, (1.0, 3.0))
    registry.mark_peeked(0, (1.2, 3.1), step=40, why="looked")
    assert registry.peeked(0, (1.0, 3.0)) and not registry.peeked(0, (4.0, 3.0)) and not registry.peeked(1, (1.0, 3.0))
    diagnostics = registry.diagnostics()
    assert diagnostics["known"]["0"][str(first)] == [1.4, 3.3] and diagnostics["peeked"]["0"][0]["why"] == "looked"


# -- detection --------------------------------------------------------------------------
def test_the_floors_openings_are_its_exits_merged_into_one_node_each_with_a_heading_into_the_unknown():
    policy, episode, world, rooms, _ = opening_policy()
    openings = openings_of(policy, episode, world)
    assert len(openings) == 3, "room A's west end; room B's patch by the door and its east end"
    gap = west_gap(openings)
    assert gap.label == "gap" and gap.door_id is None and gap.glimpsed == ()
    assert 1.0 < gap.xy[0] < 1.6, "the threshold is the frontier's nearest passable cell, at the edge of the known floor"
    assert abs(abs(gap.heading) - math.pi) < 0.4, "the unknown lies west"
    assert gap.size_cells >= 30, "the whole west end is one opening (58 frontier cells, merged)"
    assert gap.geodesic_m == pytest.approx(math.dist(IN_A, gap.xy), abs=0.6)
    east = [o for o in openings if o.room_pid == 1]
    assert len(east) == 2 and all(o.size_cells >= policy.loop.settings.openings.min_cells for o in east)
    far = max(east, key=lambda o: o.xy[0])
    assert far.xy[0] > 10.0 and abs(far.heading) < 0.4, "room B's east end, the unknown to the east"
    ids = sorted(o.index for o in openings)
    assert ids == [0, 1, 2] and sorted(o.index for o in openings_of(policy, episode, world, step=1)) == ids, "sticky ids"
    assert unknown_heading(world, world.world_to_grid(3.0, 3.0)) is None, "no unknown around the middle of room A"


def test_a_peeked_opening_a_narrow_gap_and_an_objects_shadow_are_not_openings():
    policy, episode, world, rooms, _ = opening_policy()
    gap = west_gap(openings_of(policy, episode, world))
    policy.openings.mark_peeked(0, gap.xy, step=5, why="looked")
    assert not [o for o in openings_of(policy, episode, world, step=6) if o.room_pid == 0], "looked into: retired"
    policy, episode, world, rooms, _ = opening_policy()
    wide = len(openings_of(policy, episode, world))
    narrow = OpeningSettings(min_cells=100)
    assert len(detect_openings(policy, obs_at(episode, 0, IN_A), world, narrow, OpeningRegistry())) < wide, (
        "an exit the body cannot pass is not a way anywhere")
    # The shadow test is the fallback's own: a confirmed cabinet whose footprint the patch's frontier
    # sits beside hides it -- the strip behind a cabinet is not a way anywhere either. The patch's
    # ring is 22 cells (2.2 m); a 0.5 m cabinet plus the 0.3 m margin casts up to 2.4 m.
    policy, episode, world, rooms, _ = opening_policy()
    patch = min((o for o in openings_of(policy, episode, world) if o.room_pid == 1), key=lambda o: o.xy[0])
    assert patch.size_cells == 22
    for frame in range(3):
        policy.landmarks.observe("cabinet", patch.xy, frame_id=frame, radius_m=0.5)
    assert [lm.class_name for lm in policy.landmarks.confirmed()] == ["cabinet"]
    openings = openings_of(policy, episode, world, step=3)
    assert not any(math.dist(o.xy, patch.xy) < 0.5 for o in openings), "the patch is the cabinet's shadow now"
    assert len(openings) == 2


def test_a_door_landmark_at_the_threshold_makes_a_doorway_and_objects_beyond_it_are_glimpsed():
    policy, episode, world, rooms, _ = opening_policy()
    gap = west_gap(openings_of(policy, episode, world))
    doors = policy.doors
    for _ in range(doors.settings.min_observations):             # a confirmed door frame at the gap
        door = doors.landmarks.observe("door", (gap.xy[0] - 0.2, gap.xy[1]))
    doors.separated.add(door.id)
    assert [d.id for d in doors.confirmed()] == [door.id]
    again = west_gap(openings_of(policy, episode, world, step=5))
    assert again.label == "doorway" and again.door_id == door.id and again.index == gap.index
    # Glimpsed: a confirmed landmark beyond the threshold, within glimpse_m, inside the heading's cone.
    for frame in range(3):
        policy.landmarks.observe("toilet", (gap.xy[0] - 1.5, gap.xy[1]), frame_id=frame, radius_m=0.3)   # straight through
        policy.landmarks.observe("chair", (gap.xy[0] + 1.5, gap.xy[1]), frame_id=frame, radius_m=0.3)    # behind the agent
        policy.landmarks.observe("door", (gap.xy[0] - 1.0, gap.xy[1] + 1.5), frame_id=frame, radius_m=0.3)   # beyond, a door
        policy.landmarks.observe("bed", (gap.xy[0] - 0.5, gap.xy[1] - 1.3), frame_id=frame, radius_m=0.4)    # beside: 69 deg off
        policy.landmarks.observe("sink", (gap.xy[0] - 0.1, gap.xy[1]), frame_id=frame, radius_m=0.2)     # at the threshold itself
    assert sorted(lm.class_name for lm in policy.landmarks.confirmed()) == ["bed", "chair", "door", "sink", "toilet"]
    again = west_gap(openings_of(policy, episode, world, step=6))
    assert again.glimpsed == ("toilet",), ("the toilet is through the gap; the chair is behind the agent; a door is not an "
                                           "object; the bed stands beside the gap in this room; the sink is on the threshold")
    assert again.size_cells == gap.size_cells, "the toilet's footprint lies beyond the frontier, so it hides none of it"
    wide = OpeningSettings(glimpse_cone_deg=80.0)
    assert west_gap(detect_openings(policy, obs_at(episode, 6, IN_A), world, wide, OpeningRegistry())).glimpsed == ("bed", "toilet"), (
        "a wider cone takes the bed beside the gap for something behind it -- attempt 8's bedroom door that was the stairs")


def test_the_heading_points_away_from_the_known_floor_not_at_the_mean_of_every_unknown_cell():
    """A frontier cell has unknown on several sides: a shadow strip beside it as well as the gap it stands
    in. From the known floor toward the unknown is through the opening; the mean of the unknown is not."""
    world, labels = two_room_world()
    gap_cell = world.world_to_grid(1.15, 2.95)
    assert abs(abs(unknown_heading(world, gap_cell)) - math.pi) < 0.2, "room A's west end: the unknown lies west"
    g = world.grid.copy()
    g[20:36, 12:16] = UNK                                           # a strip of unknown just EAST of the gap's cell, inside room A
    shadowed = OccupancyGrid2D(g, world.params, values=world.values)
    heading = unknown_heading(shadowed, gap_cell)
    assert heading is not None and abs(abs(heading) - math.pi) < 0.6, "still west: the free floor is east, so away from it is west"
    assert unknown_heading(world, world.world_to_grid(3.0, 3.0)) is None, "no unknown in the middle of the room"
    all_unknown = OccupancyGrid2D(np.full(world.grid.shape, UNK, np.int8), world.params, values=world.values)
    assert unknown_heading(all_unknown, gap_cell) is None, "no known floor to point away from"


# -- the loop: openings as nodes ----------------------------------------------------
def test_openings_are_offered_to_the_oracle_and_the_solver_priced_as_a_peek_and_recorded():
    policy, episode, world, rooms, reasoned = opening_policy(order=(1, 0))
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    ids = sorted(policy.loop._openings)
    assert len(ids) == 3 and all(is_opening_node(i) for i in ids)
    assert sorted(policy.reasoned_with[0]["stairs"]) == ids, "the openings went to the oracle beside the rooms (as extra nodes)"
    assert policy.loop.stats["openings_offered"] == 3
    assert policy.loop.estimate_events[0]["openings"] == ids
    gap_id = next(i for i, o in policy.loop._openings.items() if o.opening.room_pid == 0)
    estimate = policy.loop.estimates[gap_id]
    assert estimate["kind"] == "opening" and estimate["label"] == "gap" and estimate["prob"] == 0.3
    assert estimate["via"] == "room 0 (type=unknown)" and estimate["room_pid"] == 0 and estimate["glimpsed"] == []
    assert estimate["entry"] == "peek" and 1.0 < estimate["distance_m"] < 2.6, (
        "the leg the order was charged is the walk to the threshold; the peek's service time is taken back out")
    assert policy.loop.order == (1, 0), "the fixed solver knows only the rooms: the openings are candidates, not chosen"
    assert command.info["kind"] == "transit/1"
    diagnostics = policy.loop.diagnostics()
    assert diagnostics["openings_offered"] == ids and diagnostics["peek"] is None
    assert policy.configuration()["node_oracle"]["nodes"].endswith("+ its openings")
    assert "peek" in policy.configuration()["openings"]
    assert "openings" in policy.episode_info()


def test_the_rpt_instance_charges_an_opening_a_peek_not_a_scan():
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    instance = policy.supervisor.calls[-1]["instance"]
    index = {pid: i for i, pid in enumerate(instance.index_to_pid)}
    gap_id = next(i for i, o in policy.loop._openings.items() if o.opening.room_pid == 0)
    s = policy.settings
    cruise = policy.episode.action_spec.forward_step_m / s.action_time_s
    gap_xy = policy.loop._openings[gap_id].opening.xy
    walk_s = math.dist(IN_A, gap_xy) / cruise
    peek_s = policy.loop.settings.openings.service_steps * s.action_time_s
    scan_s = policy.loop.settings.service_steps() * s.action_time_s
    assert instance.units == "seconds"
    assert instance.C[instance.depot, index[gap_id]] == pytest.approx(walk_s + peek_s, rel=0.35), "walk plus a peek"
    assert instance.C[instance.depot, index[0]] >= scan_s, "a room's arc carries the scan's service time"
    assert peek_s < scan_s


def test_when_the_order_puts_an_opening_first_the_agent_walks_to_the_threshold_and_peeks():
    policy, episode, world, rooms, reasoned = opening_policy(order=(1, 0))
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == gap.node_id
    assert policy.loop.stats["openings_chosen"] == 1 and policy.loop.stats["entry_peek"] == 1
    assert policy.loop.events[-1]["event"] == "transit" and policy.loop.events[-1]["entry"] == "peek"
    assert command.info["kind"] == "peek" and command.waypoints, "the walk to the threshold"
    assert list(command.waypoints[-1]) == pytest.approx(list(gap.xy), abs=0.3)
    assert policy.loop.peek_state()["node"] == gap.node_id and policy.loop.peek_state()["xy"] == pytest.approx(list(gap.xy), abs=0.01)
    policy.loop.charge()
    assert policy.loop._peek["approach_actions"] == 1, "the walk counts toward the peek's bound"
    # Arrival at the threshold: the look begins -- face the unknown (west) from the side nearer the agent's yaw.
    yaw0 = 2.0                                                     # facing north-west: the heading (west) is to the left
    commands, last = peek_through(policy, episode, world, gap.xy, 1, yaw0)
    assert policy.supervisor.stats["arrivals"] == 1
    looks = [c for c in commands if c.info.get("kind") == "peek" and not c.waypoints]
    assert 3 <= len(looks) <= 5, "face the first heading, then one turn per heading: a few actions, not a scan"
    turn = episode.action_spec.turn_angle_rad
    assert all(c.info["of"] == 3 and abs(c.info["heading_deg"] - math.degrees(gap.heading)) < 0.1 for c in looks)
    assert [c.info["look"] for c in looks] == sorted(c.info["look"] for c in looks), "the headings are taken in order"
    previous = yaw0
    for c in looks:
        swung = math.atan2(math.sin(float(c.final_yaw) - previous), math.cos(float(c.final_yaw) - previous))
        assert 0.0 < swung <= turn + 1e-6, "every look is one in-place turn to the left, never a jump"
        previous = float(c.final_yaw)
    assert policy.loop.stats["peeks_completed"] == 1 and policy.loop.stats["peeks_abandoned"] == 0
    assert policy.supervisor.history[-1][:2] == (gap.node_id, EXHAUSTED)
    assert policy.openings.peeked(0, gap.xy), "retired for the storey"
    complete = [e for e in policy.loop.events if e["event"] == "peek_complete"][0]
    assert complete["node"] == gap.node_id and complete["turns"] == len(looks) and complete["glimpsed"] == []
    assert [e["event"] for e in policy.loop.events if e["event"] in ("peek_look", "peek_complete")] == ["peek_look", "peek_complete"]
    # The loop point after the peek: the opening is not a node any more; the rooms are.
    assert gap.node_id not in policy.loop._openings and len(policy.loop._openings) == 2
    assert commands[-1].info["kind"] == "transit/1", "the same action moves on to the next node of the fresh order"
    assert policy.loop._peek is None and policy.loop.peek_state() is None
    assert policy.scans.records == [], "a peek finishes an opening, not a room"


def test_a_peek_whose_walk_overruns_its_bound_looks_from_nearby_or_retires_the_opening():
    """The bound grows with the distance: at least ``approach_steps``, else the distance's worth of forward
    steps times ``approach_factor`` plus six turns. Attempt 8's first peek spent 20 actions on a 4 m
    walk and gave up 1.8 m short."""
    settings = OpeningSettings(approach_steps=2, approach_factor=1.0, visit_steps=8)
    assert settings.approach_bound(0.0, 0.25) == 6 and settings.approach_bound(4.0, 0.25) == 22
    assert OpeningSettings().approach_bound(1.9, 0.25) == 20 and OpeningSettings().approach_bound(4.0, 0.25) == 32
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), openings=settings)
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.room_id == gap.node_id
    bound = policy.loop._peek["approach_bound"]
    assert bound == settings.approach_bound(gap.geodesic_m, episode.action_spec.forward_step_m) == 14, "1.94 m: 8 steps + 6"
    assert policy.loop.peek_state()["approach_bound"] == bound
    for step in range(1, bound + 1):                             # walking, but never getting anywhere
        policy.loop.plan(obs_at(episode, step, IN_A), world)
        policy.loop.charge()
    assert policy.loop._peek["approach_actions"] == bound
    command = policy.loop.plan(obs_at(episode, bound + 1, IN_A), world)    # 1.85 m short: too far to look from here
    assert policy.supervisor.history[-1][:2] == (gap.node_id, UNREACHABLE)
    assert policy.loop.stats["peeks_abandoned"] == 1 and policy.openings.peeked(0, gap.xy), "not offered again"
    assert [e for e in policy.loop.events if e["event"] == "peek_abandoned"][0]["reason"].startswith("approach budget of 14 spent")
    assert command.info["kind"] == "transit/1" and gap.node_id not in policy.loop._openings
    # Within merge_m of the threshold when the bound runs out: the look is the point, so look from here.
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), openings=OpeningSettings(approach_steps=1, approach_factor=0.1, visit_steps=8))
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    near = (gap.xy[0] + 1.0, gap.xy[1])
    policy.loop.plan(obs_at(episode, 0, near), world)
    for step in range(1, policy.loop._peek["approach_bound"] + 1):
        policy.loop.charge()
    command = policy.loop.plan(obs_at(episode, 1, near, yaw=math.pi), world)
    assert policy.supervisor.state == SEARCH and command.info["kind"] == "peek" and not command.waypoints
    assert [e for e in policy.loop.events if e["event"] == "peek_from_here"]


def test_the_approach_bound_grows_to_the_route_the_planner_actually_adopted():
    """Hanson 2026-10-04, actions 166-196: the inventory said 3.5 m along the floor (walked at the body
    radius), A* planned 12.5 m around a squeeze (at the preferred clearance), and the 30-action bound
    sized on the former ran out 5.8 m short while the route was still being followed."""
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    peek = policy.loop._peek
    assert peek["approach_bound"] == 20, "the straight 1.9 m walk: the floor of 20 actions"
    assert not [e for e in policy.loop.events if e["event"] == "peek_approach_extended"], "a route as long as the geodesic extends nothing"
    # The planner's route is four times the geodesic: the bound follows it.
    long_route = [(IN_A[0] - 0.25 * i, IN_A[1] + (0.0 if i % 2 else 0.2)) for i in range(50)]      # ~12 m of zig-zag
    policy.route_memory.path = long_route
    policy.route_memory.reason = "new_goal_or_invalid_route"
    peek["approach_actions"] = 3
    policy.loop._size_approach_to_route(obs_at(episode, 3, IN_A), peek)
    length = sum(math.dist(long_route[i], long_route[i + 1]) for i in range(len(long_route) - 1))
    assert peek["approach_bound"] == 3 + policy.loop.settings.openings.approach_bound(length, episode.action_spec.forward_step_m) > 60
    extended = [e for e in policy.loop.events if e["event"] == "peek_approach_extended"]
    assert len(extended) == 1 and extended[0]["was"] == 20 and extended[0]["route_m"] == pytest.approx(length, abs=0.01)
    assert policy.loop.stats["peek_approach_extended"] == 1
    # A route kept from the last action (not newly adopted) is not re-measured.
    policy.route_memory.reason = "committed_safe_route"
    before = peek["approach_bound"]
    policy.route_memory.path = long_route * 2
    policy.loop._size_approach_to_route(obs_at(episode, 4, IN_A), peek)
    assert peek["approach_bound"] == before


def test_a_refused_threshold_is_re_aimed_once_at_a_passable_cell_on_the_agents_side_then_retired():
    """The cell went occupied as the map grew around it: one re-aim within merge_m, then give up."""
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.room_id == gap.node_id
    gx, gy = world.world_to_grid(*gap.xy)
    g = world.grid.copy()
    g[gy - 6:gy + 7, gx - 6:gx + 5] = OCC                          # a metre of wall grows over the threshold (beyond the snap radius)
    walled = OccupancyGrid2D(g, world.params, values=world.values)
    command = policy.loop.plan(obs_at(episode, 1, IN_A), walled)
    assert policy.supervisor.room_id == gap.node_id and command.info["kind"] == "peek"
    reaimed = [e for e in policy.loop.events if e["event"] == "peek_reaimed"]
    assert len(reaimed) == 1 and policy.loop._peek["reaimed"]
    goal = policy.loop._peek["goal"]
    assert 0.4 <= math.dist(goal, gap.xy) <= 1.5 and goal[0] > gap.xy[0], "a passable cell just east of the wall, the agent's side"
    g[gy - 8:gy + 9, gx - 1:gx + 14] = OCC                         # ... and the whole west end
    sealed = OccupancyGrid2D(g, world.params, values=world.values)
    command = policy.loop.plan(obs_at(episode, 2, IN_A), sealed)
    assert policy.supervisor.history[-1][:2] == (gap.node_id, UNREACHABLE)
    assert [e for e in policy.loop.events if e["event"] == "peek_abandoned"][-1]["reason"] == "unreachable"
    assert policy.openings.peeked(0, gap.xy) and command.info["kind"] == "transit/1"


def test_a_peek_that_overruns_the_visit_bound_is_released_budget_spent_and_the_opening_retired():
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), openings=OpeningSettings(visit_steps=4))
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    policy.loop.plan(obs_at(episode, 1, gap.xy, yaw=0.0), world)   # arrived, facing east: the look needs ~6 turns to face west
    assert policy.supervisor.state == SEARCH and policy.loop.visit_budget() == 4, "a peek's bound, not a scan's"
    yaw = 0.0
    for step in range(2, 8):
        command = policy.loop.plan(obs_at(episode, step, gap.xy, yaw), world)
        policy.loop.charge()
        if policy.supervisor.room_id != gap.node_id:
            break
        yaw = float(command.final_yaw)
    assert policy.supervisor.history[-1][:2] == (gap.node_id, "budget_spent")
    assert policy.loop.stats["budget_releases"] == 1


def test_openings_count_as_rooms_left_so_the_way_back_upstairs_waits():
    """The storey is looked at before the way back is a choice -- its unpeeked doors included."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import STAIRS, stair_policy
    policy, episode, world, rooms, _ = stair_policy(order=(1, 0, STAIRS), visit="scan")
    refresh(policy, world, IN_A)
    policy.building.arrived_by, policy.building.arrived_step = 0, 0
    grace = policy.settings.multifloor.arrival_grace_actions
    for pid in rooms:                                              # every room finished: only the openings are left
        policy.scans.record(obs_at(episode, 1, rooms[pid].centroid), rooms[pid])
    policy.loop.plan(obs_at(episode, grace + 5, IN_A), world)
    assert policy.loop._excluded.keys() == {0, 1} and policy.loop._openings
    assert STAIRS not in policy.loop._stairs, "three openings to look into: the way back is held"
    assert [e for e in policy.loop.events if e["event"] == "way_back_held"]


def test_the_unknown_at_the_foot_of_a_seen_staircase_is_not_an_opening():
    """The stairs are a node of their own; the unknown beyond a flight is the other storey."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import stair_policy
    policy, episode, world, rooms, _ = stair_policy(order=(1, 0), visit="scan")
    stairs = OpeningRegistry()
    g = world.grid.copy()
    foot = policy.building.portals[0]["entry"]
    gx, gy = world.world_to_grid(foot[0], foot[1])
    g[gy - 1:gy + 2, gx - 8:gx + 8] = -1                           # unknown across the foot of the flight: a frontier there
    from sparx_agency.core.planning.environment import OccupancyGrid2D
    stepped = OccupancyGrid2D(g, world.params, values=world.values)
    refresh(policy, stepped, IN_A)
    goals = policy.graph.frontier_inventory.goals
    assert any(math.dist(goal.xy, (foot[0], foot[1])) < 1.0 for goal in goals), "the map does show a frontier at the foot"
    openings = detect_openings(policy, obs_at(episode, 0, IN_A), stepped, OpeningSettings(), stairs)
    assert not any(math.dist(o.xy, (foot[0], foot[1])) < 1.5 for o in openings), "... but it is the stairs, not an opening"
    assert [o for o in openings if o.room_pid == 0], "room A's west gap still is"


def test_a_new_opening_while_nothing_is_in_force_buys_the_oracle_a_call_at_most_every_revalue_actions():
    """Attempt 8 asked the oracle thirty times in sixty fallback actions, once per door that appeared or
    re-snapped. A new room or staircase is still valued at once."""
    policy, episode, world, rooms, reasoned = opening_policy(order=(1, 0), openings=OpeningSettings(revalue_actions=5))
    for pid in (0, 1):                                              # both rooms cooling: nothing to choose, the fallback carries
        policy.supervisor.inner._cooling[pid] = 0.0
        policy.supervisor.inner._cooling_verdict[pid] = EXHAUSTED
    policy.graph.probs = {0: 0.6, 1: 0.4}
    policy.loop._needs_reason = False
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert reasoned == [0], "three openings the estimate has never seen: valued once"
    policy.graph.stair_probs.pop(next(iter(policy.loop._openings)))  # as if one re-snapped to a new id
    policy.loop.plan(obs_at(episode, 2, IN_A), world)
    assert reasoned == [0], "two actions later: not yet"
    policy.loop.plan(obs_at(episode, 5, IN_A), world)
    assert reasoned == [0, 5], "five actions later: valued"
    from sparx_agency.core.mapping.topology.room_registry import TrackedRoom
    from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
    from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
    new_room = TrackedRoom(id=2, mask=rooms[0].mask, n_cells=rooms[0].n_cells, centroid=rooms[0].centroid)
    policy.graph.registry.rooms[2] = new_room
    policy.graph.options.append(RoomOption(room_id=2, prob=0.0, xy=new_room.centroid, label="R2"))
    policy.graph.facts[2] = RoomFacts(2, 1, 0.0, new_room.n_cells)
    policy.loop.plan(obs_at(episode, 6, IN_A), world)
    assert reasoned == [0, 5, 6], "a room the estimate has never seen is valued at once, throttle or no throttle"


# -- the weak-type rule --------------------------------------------------------------
def test_a_weak_type_label_never_rules_a_room_out():
    """Ranchester step 75: a toilet glimpsed through a door named the hallway a bathroom and every door
    off it was demoted with it. Hanson step 321: a sofa made the room with the sink a 'living room' and
    ruled it out for the toilet. One kind of object is a guess the oracle sees as `type=bathroom?`."""
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), llm=NamingLLM())
    assert policy.target.query == "chair"
    see(policy, {1: ["toilet"]}, step=0)
    policy.graph.relabel(1, 0)
    info = policy.graph.label_info(1)
    assert info["label"] == "bathroom" and info["strength"] == "weak"
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert 1 not in policy.loop._excluded and policy.loop.stats["weak_type_kept"] == 1
    kept = [e for e in policy.loop.events if e["event"] == "weak_type_kept"][0]
    assert kept["room"] == 1 and kept["label"] == "bathroom" and kept["openings"] == 2 and kept["objects"] == ["toilet"]
    assert policy.supervisor.room_id == 1 and command.info["kind"] == "transit/1", "still a node, still chosen by the order"
    policy.loop.plan(obs_at(episode, 1, IN_A), world)
    assert policy.loop.stats["weak_type_kept"] == 1, "logged once per room"
    # The former openings allowance is not consulted any more: kept whatever the bound says.
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), llm=NamingLLM(), weak_type_max_openings=2)
    see(policy, {1: ["toilet"]}, step=0)
    policy.graph.relabel(1, 0)
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert 1 not in policy.loop._excluded and policy.loop.stats["weak_type_kept"] == 1
    # A strong label (two kinds of object) rules the room out whatever its openings.
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), llm=NamingLLM())
    see(policy, {1: ["toilet", "sink"]}, step=0)
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["strength"] == "strong"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {1: "type:bathroom"}


def test_a_home_object_of_the_target_keeps_a_strongly_labelled_room_a_node():
    """Hanson 2026-10-04: the bathroom's sink was merged into a room of sofas. For a toilet, a sink is
    where it lives; the room stays a node whatever its label says."""
    from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), llm=NamingLLM())
    policy.target = gibson_label_mapper().target_labels("toilet")
    see(policy, {1: ["sofa", "bed"]}, step=0)                       # NamingLLM: a bedroom, strongly
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["label"] == "bedroom" and policy.graph.label_info(1)["strength"] == "strong"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._excluded == {1: "type:bedroom"}, "a toilet is not searched for in a bedroom"
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), llm=NamingLLM())
    policy.target = gibson_label_mapper().target_labels("toilet")
    see(policy, {1: ["sofa", "bed", "shower"]}, step=0)             # still a bedroom to the classifier
    policy.graph.relabel(1, 0)
    assert policy.graph.label_info(1)["label"] == "bedroom" and policy.graph.label_info(1)["strength"] == "strong"
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert 1 not in policy.loop._excluded and policy.loop.stats["home_object_kept"] == 1
    kept = [e for e in policy.loop.events if e["event"] == "home_object_kept"][0]
    assert kept["room"] == 1 and kept["home_objects"] == ["shower"]


# -- target landmarks as nodes -----------------------------------------------------------
def test_a_confirmed_landmark_of_the_targets_class_is_a_node_at_a_fixed_probability_without_the_oracle():
    """Ranchester attempt 9: the downstairs warm-up put a `sofa` on the map 3.8 m away and the search walked
    west for forty actions. The fixture's target is a chair: a confirmed chair in room B is the first node."""
    policy, episode, world, rooms, reasoned = opening_policy(order=(1, 0))
    chair = (8.5, 1.5)
    for frame in range(3):
        policy.landmarks.observe("chair", chair, frame_id=frame, radius_m=0.3)
        policy.landmarks.observe("bed", (9.5, 4.0), frame_id=frame, radius_m=0.8)             # not the target's class
    assert sorted(lm.class_name for lm in policy.landmarks.confirmed()) == ["bed", "chair"]
    cost = policy.navigation_cost(world)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.opening_nodes import LANDMARK, LANDMARK_INDEX_BASE, landmark_openings
    nodes = landmark_openings(policy, obs_at(episode, 0, IN_A), world, cost, policy.loop.settings.openings, set())
    [node] = nodes
    lid = next(lm.id for lm in policy.landmarks.confirmed() if lm.class_name == "chair")
    assert node.kind == LANDMARK and node.landmark_id == lid and node.index == LANDMARK_INDEX_BASE + lid
    assert node.glimpsed == ("chair",) and node.label == "chair landmark" and node.room_pid == 1
    assert math.dist(node.xy, chair) == pytest.approx(1.5, abs=0.15), "the standoff ring"
    assert node.xy[0] < chair[0], "... on the agent's side (the agent is west, in room A)"
    assert abs(normalize(node.heading - math.atan2(chair[1] - node.xy[1], chair[0] - node.xy[0]))) < 1e-6, "facing the chair"
    assert landmark_openings(policy, obs_at(episode, 0, IN_A), world, cost, policy.loop.settings.openings, {lid}) == [], "inspected: not again"
    policy.closing.rejected.append(((chair[0] + 0.3, chair[1], 0.0), policy.mapping.floor_id, 7, 1.0))
    assert landmark_openings(policy, obs_at(episode, 0, IN_A), world, cost, policy.loop.settings.openings, set()) == [], (
        "the takeover verified a candidate there and released it: not worth another look")
    policy.closing.rejected.clear()
    # In the loop: offered beside the exits at the fixed probability, never shown to the oracle, first in the order.
    policy.supervisor.inner._solver = lambda candidates, instance=None: sorted((c.room_id for c in candidates),
                                                                                key=lambda nid: -next(c.prob for c in candidates if c.room_id == nid))
    command = policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert node.node_id in policy.loop._openings and policy.loop.stats["landmarks_offered"] == 1
    assert node.node_id not in policy.reasoned_with[0]["stairs"], "the oracle was not asked about it"
    assert len(policy.reasoned_with[0]["stairs"]) == 3, "... only about the three exits"
    assert policy.loop.estimate_events[0]["landmarks"] == [node.node_id]
    estimate = policy.loop.estimates[node.node_id]
    assert estimate["kind"] == "landmark" and estimate["prob"] == 0.85 and estimate["landmark_id"] == lid
    assert estimate["why"].startswith("confirmed chair on the map") and estimate["entry"] == "peek"
    assert policy.supervisor.room_id == node.node_id and policy.loop.stats["landmarks_chosen"] == 1
    assert policy.loop.events[-1]["entry"] == "landmark" and command.info["kind"] == "peek"
    assert policy.loop.peek_state()["kind"] == LANDMARK
    # Arrival at the standoff, facing the chair: the look, then the landmark is inspected for the storey.
    commands, last = peek_through(policy, episode, world, node.xy, 1, node.heading)
    assert policy.loop.stats["peeks_completed"] == 1 and policy.loop.stats["landmarks_inspected"] == 1
    assert lid in policy.loop._inspected and node.node_id not in policy.loop._openings
    assert policy.openings.diagnostics()["peeked"] == {}, "a landmark is inspected by id, not retired at a threshold"
    complete = [e for e in policy.loop.events if e["event"] == "peek_complete"][0]
    assert complete["kind"] == LANDMARK and complete["glimpsed"] == ["chair"]
    search = search_snapshot(policy)
    assert all(o["kind"] != "landmark" for o in search["openings"]), "not offered again"
    assert _name({"id": node.node_id, "kind": "landmark", "landmark_id": lid}) == "T%d" % lid
    assert _name({"id": node.node_id}) == "T%d" % lid, "the index band alone names it"
    assert "confirmed landmark of the target's class is a node at p=0.85" in policy.configuration()["openings"]


def test_a_landmark_the_takeover_rejects_mid_look_ends_the_look():
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    chair = (8.5, 1.5)
    for frame in range(3):
        policy.landmarks.observe("chair", chair, frame_id=frame, radius_m=0.3)
    policy.supervisor.inner._solver = lambda candidates, instance=None: sorted((c.room_id for c in candidates),
                                                                                key=lambda nid: -next(c.prob for c in candidates if c.room_id == nid))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    node = policy.loop._peek["opening"]
    policy.loop.plan(obs_at(episode, 1, node.xy, yaw=node.heading), world)          # arrived: the look begins
    assert policy.supervisor.state == SEARCH and policy.loop._peek["targets"]
    policy.closing.rejected.append(((chair[0], chair[1], 0.0), policy.mapping.floor_id, 2, 1.0))   # the takeover ran and released
    command = policy.loop.plan(obs_at(episode, 2, node.xy, yaw=node.heading), world)
    assert policy.supervisor.history[-1][:2] == (node.node_id, EXHAUSTED)
    assert node.landmark_id in policy.loop._inspected
    assert [e for e in policy.loop.events if e["event"] == "peek_abandoned"][-1]["reason"] == "rejected by the takeover"
    assert command.info["kind"] == "transit/0", "the same action moves on: room A (p 0.6) heads the probability order"


def normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# -- settings and the record --------------------------------------------------------------
def test_room_numbers_are_unique_across_the_building():
    """Every storey keeps its own registry; a new storey's starts after the highest pid any storey
    has handed out, so the recording never shows two rooms called R0."""
    policy, episode, world, rooms, _ = opening_policy()
    assert policy.mapping.floor_id == 0 and policy.floors.active == 0
    policy.graph.registry._next = 2                                # rooms 0 and 1 were handed out upstairs
    policy.floors.activate(1)
    assert policy.graph.registry.rooms == {} and policy.graph.registry.next_pid == 2, "the storey below starts at R2"
    policy.graph.registry._next = 5                                # three rooms found down here: R2, R3, R4
    policy.floors.activate(0)
    assert policy.graph.registry.next_pid == 2 and set(policy.graph.registry.rooms) == {0, 1}, "back upstairs: its own registry"
    policy.floors.activate(2)
    assert policy.graph.registry.next_pid == 5, "a third storey starts after every room so far"
    assert "unique across the building" in policy.configuration()["room_partition"]


@pytest.mark.parametrize("kwargs", [{"min_cells": 0}, {"look_turns": -1}, {"look_turns": 2, "visit_steps": 5},
                                    {"merge_m": 0.0}, {"enabled": 1}, {"service_steps": 0}, {"glimpse_cone_deg": 120.0},
                                    {"glimpse_min_m": 3.0}, {"approach_factor": 0.0}, {"revalue_actions": 0},
                                    {"landmark_prob": 1.0}, {"landmarks_enabled": "yes"}, {"landmark_standoff_m": 0.0}])
def test_opening_settings_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        OpeningSettings(**kwargs)


def test_loop_settings_take_the_openings_as_a_dict_and_check_the_weak_type_allowance():
    settings = LoopSettings(openings={"enabled": False, "look_turns": 2})
    assert isinstance(settings.openings, OpeningSettings) and not settings.openings.enabled and settings.openings.look_turns == 2
    for kwargs in ({"weak_type_max_openings": -1}, {"weak_type_max_openings": 1.5}, {"openings": "yes"}):
        with pytest.raises(ValueError):
            LoopSettings(**kwargs)


def test_openings_off_is_the_ablation_where_the_exits_are_the_fallbacks_alone():
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0), openings=OpeningSettings(enabled=False))
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.loop._openings == {} and policy.reasoned_with[0]["stairs"] == []
    assert policy.configuration()["openings"].startswith("none") and policy.configuration()["node_oracle"]["nodes"].endswith("staircases")


def test_the_snapshot_and_the_panels_carry_the_openings():
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    policy.last_world = world
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    search = search_snapshot(policy)
    assert len(search["openings"]) == 3 and search["peek"] is None
    gap = next(o for o in search["openings"] if o["room_pid"] == 0)
    assert gap["kind"] == "opening" and gap["label"] == "gap" and gap["prob"] == 0.3 and gap["via"] == "room 0 (type=unknown)"
    assert gap["glimpsed"] == [] and len(gap["centroid"]) == 2 and abs(abs(gap["heading_deg"]) - 180.0) < 25.0
    assert gap["distance_m"] is not None and gap["why"] == ""
    column = render_search_panel(search, (320, 720))
    assert column.shape == (720, 320, 3) and (column > 60).any()
    obs = obs_at(episode, 0, IN_A)
    policy.mapping.update(obs)
    panels = FloorPanels(1, policy.settings, obs.pose)
    panels.capture(policy, obs)
    plain = panels.render(0, size=(640, 720))
    marked = panels.render(0, size=(640, 720), search=search)
    assert (plain != marked).any(), "the openings are drawn on the active floor"
    # Mid-peek the snapshot says so.
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    policy.last_world = world
    gap_opening = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap_opening.node_id, 1, 0]
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    assert policy.supervisor.room_id == gap_opening.node_id
    search = search_snapshot(policy)
    assert search["next_room"] == gap_opening.node_id and search["peek"]["node"] == gap_opening.node_id
    assert search["peek"]["xy"] == pytest.approx(list(gap_opening.xy), abs=0.01) and search["peek"]["of"] == 0
    assert render_search_panel(search, (320, 720)).shape == (720, 320, 3)
    policy.loop.plan(obs_at(episode, 1, gap_opening.xy, yaw=2.0), world)        # at the threshold: the look
    search = search_snapshot(policy)
    assert search["room_in_force"] == gap_opening.node_id and search["peek"]["of"] == 3 and search["local_budget"] == 30
    assert render_search_panel(search, (320, 720)).shape == (720, 320, 3)









