"""The search economy rules of 2026-10-13 (evening): what is worth a walk, what is worth less, and what is not re-walked.

Written from the per-episode traces of the toilet campaign's re-run
(``runs/benchmark_toilet_15episodes_20261013T184541Z``): an exit valued by
the unseen floor behind it, not by the model's flat number; an entered room
worth less than an unexplored one; a room's slivers not walked and not
keeping it open; a room the planner could not reach cooling briefly, not
for a hundred actions; a peek retired when the stairs are placed under it
and ended when the camera resolves it on the way; and the planner that
squeezes rather than walks round the house.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    EXHAUSTED, SEARCH, TRANSIT, UNREACHABLE, ObjectSearchParams, ObjectSearchSupervisor)
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.core.planning.object_search.rooms.opening_nodes import (
    Opening, OpeningSettings, detect_openings, opening_value_factor)
from sparx_agency.core.planning.object_search.rooms.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_explore_visit import explore_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import (
    opening_policy, openings_of, refresh, west_gap)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    FREE, IN_A, IN_B, OCC, RES, UNK, VALUES, loop_policy, obs_at)


# -- an exit is worth what is behind it ------------------------------------------------------------------
def test_an_exits_gain_is_the_unseen_floor_through_it_and_scales_the_oracles_number_a_doorway_never():
    """The two-room world: room A's west end hides a strip of unknown ONE metre deep behind a 58-cell gap (the cone
    from the threshold reaches 0.8 m2 of it before the wall); room B's patch by the door is 0.3 m2 behind a ring of
    frontier. The model values both "nothing glimpsed" alike; a wing four metres deep behind the same gap is full."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    openings = openings_of(policy, episode, world)
    gap = west_gap(openings)
    patch = min((o for o in openings if o.room_pid == 1), key=lambda o: o.xy[0])
    settings = policy.loop.settings.openings
    assert 0.5 < gap.gain_m2 < 1.5, "a metre of unknown, then the wall: what the look from the threshold would show"
    assert 0.0 < patch.gain_m2 < 1.0, "a 0.3 m2 patch: a fraction of a square metre"
    assert opening_value_factor(gap, settings) == pytest.approx(max(settings.gain_min_factor, gap.gain_m2 / settings.gain_full_m2))
    assert opening_value_factor(patch, settings) == pytest.approx(max(settings.gain_min_factor, patch.gain_m2 / settings.gain_full_m2))
    # The same room's west end onto a wing: four metres of unknown before the wall.
    g = world.grid.copy()
    g[1:59, 1:41] = UNK
    deep = OccupancyGrid2D(g, world.params, values=world.values)
    refresh(policy, deep, (5.3, 1.0))                            # read from the room's far corner: the gap is 1 m from IN_A now
    wing = west_gap(openings_of(policy, episode, deep, xy=(5.3, 1.0)))
    assert wing.gain_m2 >= settings.gain_full_m2 and opening_value_factor(wing, settings) == 1.0, "a wing: in full"
    assert opening_value_factor(replace(patch, door_id=7), settings) == 1.0, "a doorway is a room's door whatever the rays count"
    assert opening_value_factor(patch, replace(settings, gain_full_m2=0.0)) == 1.0, "0 disables the scaling"
    with pytest.raises(ValueError):
        OpeningSettings(gain_min_factor=0.0)
    with pytest.raises(ValueError):
        OpeningSettings(gain_full_m2=-1.0)


def test_the_loop_hands_rpt_star_the_scaled_numbers_and_an_entered_room_half_of_its():
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    openings = openings_of(policy, episode, world)
    gap = west_gap(openings)
    patch = min((o for o in openings if o.room_pid == 1), key=lambda o: o.xy[0])
    policy.graph.last_inside[1] = 7                              # the agent has stood in room B
    policy.graph.last_inside[0] = 9                              # ... and stands in room A now
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    factors = policy.loop._value_factors
    assert factors.get(1) == pytest.approx(LoopSettings().entered_discount), "room B: entered and left, not finished"
    assert 0 not in factors, "room A: the room the agent stands in -- the oracle's number stands"
    assert factors.get(patch.node_id, 1.0) < 1.0 if patch.node_id in policy.loop._openings_unfolded and not policy.loop._folded_now else True
    estimates = policy.loop.estimates
    assert estimates[1]["factor"] == pytest.approx(LoopSettings().entered_discount) and estimates[0]["factor"] == 1.0
    # The options the supervisor filtered on carry the scaled probability.
    options = policy.loop._scaled_values([RoomOption(room_id=1, prob=0.6, xy=IN_B, label="R1"),
                                          RoomOption(room_id=0, prob=0.6, xy=IN_A, label="R0")], [], here=0)
    assert [round(o.prob, 3) for o in options] == [0.3, 0.6]
    assert policy.loop.diagnostics()["value_factors"] == {"1": 0.5}
    with pytest.raises(ValueError):
        LoopSettings(entered_discount=0.0)


# -- a room's slivers are not walked and do not keep it open -------------------------------------------
def test_a_frontier_sliver_is_not_a_goal_of_the_explore_visit_and_does_not_keep_the_room_open():
    """Room B with a strip of unknown one cell deep along two metres of its north wall -- the shadow a cabinet
    leaves: a 20-cell boundary revealing 0.2 m2. Not worth a walk, not a live frontier -- the room is finished
    the moment the agent stands in it (a long boundary is not a large unknown)."""
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), room_b_resolved=True)
    g = world.grid.copy()
    g[58, 80:100] = UNK                                         # the strip: 2 m x 0.1 m of unknown under the north wall
    notched = OccupancyGrid2D(g, world.params, values=world.values)
    cost = refresh(policy, notched, IN_A)
    obs = obs_at(episode, 0, IN_A)
    raw = policy.graph.frontier_inventory.by_room.get(2, ())
    assert raw and max(g.size_cells for g in raw) >= policy.loop.settings.openings.min_cells, (
        "the inventory does show the strip as room B's frontier, an exit's width and more")
    assert policy.loop._explore_goals(obs, notched, cost, rooms[1]) == [], "... but the visit has nothing worth a walk"
    assert policy.loop._live_frontier(1, obs, notched) == 0
    assert not policy.loop._significant_frontier_left(obs, notched) or [
        g for g in policy.graph.frontier_inventory.goals if g.size_cells >= policy.loop.settings.openings.min_cells
        or math.isfinite(g.geodesic_m) and g.xy[0] < 5.0], "room A's west end is still worth a walk; the notch is not"
    # With the rule off every admissible frontier counts again.
    policy.loop.settings = replace(policy.loop.settings, explore_min_gain_m2=0.0)
    assert policy.loop._explore_goals(obs, notched, cost, rooms[1]), "0 restores the former population"
    assert policy.loop._live_frontier(1, obs, notched) >= 1


# -- a failed turn cools briefly ------------------------------------------------------------------------
def test_a_room_released_unreachable_cools_for_the_failure_cooldown_not_the_visit_cooldown():
    from sparx_agency.core.planning.exploration.tests.test_object_search_supervisor import facts as room_facts
    params = ObjectSearchParams(visit_cooldown_s=120.0, failure_cooldown_s=30.0, plan_grace_s=5.0, resolve_on_release=True)
    rooms = [RoomOption(room_id=1, prob=0.5, xy=(3.0, 0.0), label="a"), RoomOption(room_id=2, prob=0.4, xy=(6.0, 0.0), label="b")]
    facts = room_facts(r1=2, r2=2)
    sup = ObjectSearchSupervisor(params, solver=lambda candidates, instance=None: [c.room_id for c in sorted(candidates, key=lambda c: c.room_id)])
    sup.update(rooms, facts, (0.0, 0.0), now=0.0, last_plan_s=None)
    out = sup.update(rooms, facts, (0.0, 0.0), now=6.0, last_plan_s=None)
    assert out.action.verdict == UNREACHABLE and sup._is_cooling(1, 6.0)
    assert sup._is_cooling(1, 35.0) and not sup._is_cooling(1, 37.0), "30 s, not 120"
    sup.update(rooms, facts, (0.0, 0.0), now=37.0, last_plan_s=None)
    assert sup.room_id == 1, "chosen again once the failure cooldown is over"
    assert LoopSettings().supervisor_params(0).failure_cooldown_s == float(LoopSettings().failure_cooldown_steps)
    with pytest.raises(ValueError):
        ObjectSearchParams(failure_cooldown_s=-1.0)
    with pytest.raises(ValueError):
        LoopSettings(failure_cooldown_steps=-1)


# -- the peek's walk: the stairs placed under it, the opening resolved on the way ------------------------
def _peek_toward_west_gap(policy, episode, world):
    """Start a peek transit toward room A's west gap from room B (the scan-mode fixture: no fold hides the gap)."""
    gap = west_gap(openings_of(policy, episode, world))
    policy.supervisor.inner._solver = lambda candidates, instance=None: [gap.node_id, 1, 0]
    command = policy.loop.plan(obs_at(episode, 0, IN_B), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == gap.node_id and command.waypoints
    return gap


def test_a_peek_whose_threshold_the_stairs_are_placed_under_is_retired_and_the_walk_ends():
    """Corozal/000076, actions 168-255: the staircase seen at 203, the gap at its foot walked round for 50 more."""
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    gap = _peek_toward_west_gap(policy, episode, world)
    mask = np.zeros(world.grid.shape, bool)
    gx, gy = world.world_to_grid(*gap.xy)
    mask[gy - 3:gy + 4, gx - 8:gx - 4] = True                   # the capsule, half a metre west of the threshold
    policy.floor_guard.mask = mask
    policy.loop.charge()
    policy.loop.plan(obs_at(episode, 1, IN_B), world)
    assert policy.supervisor.room_id != gap.node_id, "the peek is over"
    retired = [e for e in policy.loop.events if e["event"] == "peek_abandoned"]
    assert retired and retired[-1]["node"] == gap.node_id and "staircase" in retired[-1]["reason"]
    assert policy.openings.peeked(0, gap.xy), "retired for the storey: never offered again"
    assert gap.node_id not in {o.node_id for o in openings_of(policy, episode, world, step=2, xy=IN_B)}, (
        "and the inventory itself drops an exit beside the mask")


def test_a_peek_whose_opening_the_camera_resolves_on_the_way_ends_there_without_the_walk():
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    gap = _peek_toward_west_gap(policy, episode, world)
    settings = policy.loop.settings.openings
    resolved = OccupancyGrid2D(world.grid.copy(), world.params, values=world.values)
    resolved.grid[1:59, 1:11] = FREE                            # the west end seen: room A has no frontier left
    for step in range(1, settings.vanish_actions + 1):
        refresh(policy, resolved, IN_B)
        policy.loop.charge()
        policy.loop.plan(obs_at(episode, step, IN_B), resolved)
        if step < settings.vanish_actions:
            assert policy.supervisor.room_id == gap.node_id, "one or two actions gone is a re-snap, not a resolution"
    assert policy.supervisor.room_id != gap.node_id
    assert policy.loop.stats["peeks_resolved_en_route"] == 1
    event = [e for e in policy.loop.events if e["event"] == "peek_resolved"][-1]
    assert event["node"] == gap.node_id and event["distance_m"] > settings.merge_m
    assert policy.supervisor.history[-1][:2] == (gap.node_id, EXHAUSTED), "productive: the look was had"
    assert not policy.openings.peeked(0, gap.xy), "nothing to retire: if the frontier re-snaps it is offered again"


def test_the_record_shows_every_live_opening_and_the_one_a_peek_walks_to():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot
    policy, episode, world, rooms, _ = opening_policy(order=(1, 0))
    gap = _peek_toward_west_gap(policy, episode, world)
    shown = {o["id"]: o for o in search_snapshot(policy)["openings"]}
    assert shown[gap.node_id]["targeted"] and shown[gap.node_id]["live"]
    assert all("gain_m2" in o and "factor" in o and "folded" in o for o in shown.values())
    # The inventory loses the opening: the record keeps it, flagged, until the loop decides.
    policy.loop._openings = {k: v for k, v in policy.loop._openings.items() if k != gap.node_id}
    shown = {o["id"]: o for o in search_snapshot(policy)["openings"]}
    assert shown[gap.node_id]["targeted"] and not shown[gap.node_id]["live"]


# -- the planner squeezes rather than walks round the house -------------------------------------------------
def _pinch_world():
    """A 10 x 6 m room split by a wall with a 0.5 m pinch; the way round the wall's north end is 1.2 m wide."""
    g = np.zeros((60, 100), np.int8)
    g[0, :] = g[-1, :] = OCC
    g[:, 0] = g[:, -1] = OCC
    g[1:48, 50] = OCC                                           # the wall, open at the north end (rows 48-58)
    g[12:17, 50] = FREE                                         # the pinch: 5 cells = 0.5 m, at y ~ 1.4
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def test_a_route_round_the_house_at_the_preferred_standoff_loses_to_the_squeeze_by_the_detour_factor():
    """Corozal/000076, actions 394-486: 12.7 m round the room because a 0.5 m pinch on the 7.5 m way read as
    0.25 m short of the preferred 0.30 m standoff; the body is 0.18 m."""
    world = _pinch_world()
    request = PlanRequest(Pose2D(2.0, 1.4, 0.0), Pose2D(8.0, 1.4), frame_id="world")
    strict = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.30, inflate_floor_m=0.18, unknown_blocked=True,
                                                        goal_snap_radius_m=0.3, waypoint_spacing_m=0.25, start_skip_m=0.0,
                                                        search_margin_m=20.0))
    detour = strict.plan(request, world)
    assert detour.ok and detour.path.length() > 10.0 and strict.last_inflate_m == pytest.approx(0.30), "round the north end"
    squeezing = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.30, inflate_floor_m=0.18, detour_factor=1.5,
                                                           unknown_blocked=True, goal_snap_radius_m=0.3,
                                                           waypoint_spacing_m=0.25, start_skip_m=0.0, search_margin_m=20.0))
    squeeze = squeezing.plan(request, world)
    assert squeeze.ok and squeeze.path.length() < 7.5 and squeezing.last_inflate_m == pytest.approx(0.18)
    assert squeeze.artifacts["detour_avoided_m"] == pytest.approx(detour.path.length())
    assert squeeze.artifacts["detour_factor"] == 1.5
    # A short detour is kept: the squeeze is for a route more than the factor longer.
    near = PlanRequest(Pose2D(4.0, 5.0, 0.0), Pose2D(6.0, 5.0), frame_id="world")
    kept = squeezing.plan(near, world)
    assert kept.ok and "detour_avoided_m" not in kept.artifacts and squeezing.last_inflate_m == pytest.approx(0.30)


# -- 2026-10-14: a room lingered in is as good as scanned; finished rooms are no nodes of the oracle's -------
def test_a_room_lingered_in_with_almost_no_frontier_left_keeps_the_lingered_discount_not_the_entered_one():
    """The user's rule: a room spent a long time in with almost nothing left to see is as good as scanned,
    whatever the model writes. Twenty actions inside and at most one significant frontier: 0.2 of the number."""
    from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0), room_b_resolved=True)
    settings = policy.loop.settings
    policy.graph.last_inside[1] = 7                              # entered and left
    policy.graph.last_inside[0] = 9
    fact = policy.graph.facts[1]
    obs = obs_at(episode, 0, IN_A)
    assert policy.loop._live_frontier(1, obs, world) == 0, "room B resolved: nothing significant left"
    # Entered briefly: half, decayed a little by the five actions inside (the dwell decay, 2026-10-21).
    policy.graph.facts[1] = RoomFacts(1, fact.frontier_clusters, 5.0, fact.cells)
    options = policy.loop._scaled_values([RoomOption(room_id=1, prob=0.6, xy=IN_B, label="R1")], [], here=0,
                                         obs=obs, world=world)
    assert options[0].prob == pytest.approx(0.6 * settings.entered_discount * 0.5 ** (5.0 / settings.dwell_half_life_steps))
    # Lingered: twenty actions inside, nothing left -- a fifth (the lingered cap is lower than the decayed half).
    policy.graph.facts[1] = RoomFacts(1, fact.frontier_clusters, settings.lingered_steps * policy.settings.action_time_s, fact.cells)
    options = policy.loop._scaled_values([RoomOption(room_id=1, prob=0.6, xy=IN_B, label="R1")], [], here=0,
                                         obs=obs, world=world)
    assert options[0].prob == pytest.approx(0.6 * settings.lingered_discount)
    assert policy.loop._value_factors == {1: settings.lingered_discount}
    # The room the agent stands in is never discounted, lingered or not.
    policy.graph.facts[0] = RoomFacts(0, 0, 99.0, policy.graph.facts[0].cells)
    options = policy.loop._scaled_values([RoomOption(room_id=0, prob=0.6, xy=IN_A, label="R0")], [], here=0,
                                         obs=obs, world=world)
    assert options[0].prob == pytest.approx(0.6)
    assert LoopSettings().lingered_discount < LoopSettings().entered_discount and LoopSettings().second_pass is False
    with pytest.raises(ValueError):
        LoopSettings(lingered_discount=0.0)
    with pytest.raises(ValueError):
        LoopSettings(lingered_steps=0)


# -- 2026-10-21: the unvisited are raised, and the entered decay with the time spent inside ------------------
def test_a_room_never_stood_in_is_raised_and_floored_and_an_entered_one_decays_with_its_dwell():
    """The user's recalibration: a never-entered room is where discovery is (``frontier=0`` on it measures the
    map, not the room), so its number is raised by ``unvisited_boost`` and read at no less than
    ``unvisited_floor``; a room entered and left loses half of its entered factor per ``dwell_half_life_steps``
    inside, down to ``dwell_floor``. The room the agent stands in and a room the model gave 0 keep their numbers."""
    from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
    policy, episode, world, rooms, _ = explore_policy(order=(1, 0))
    settings = policy.loop.settings
    obs = obs_at(episode, 0, IN_A)
    policy.graph.last_inside = {0: 9}                            # room A stood in (here); room B never
    fact = policy.graph.facts[1]
    policy.graph.facts[1] = RoomFacts(1, fact.frontier_clusters, 0.0, fact.cells)
    scale = lambda prob, here=0: policy.loop._scaled_values([RoomOption(room_id=1, prob=prob, xy=IN_B, label="R1")],
                                                            [], here=here, obs=obs, world=world)[0].prob
    assert settings.unvisited_boost == 1.5 and settings.unvisited_floor == 0.15
    assert scale(0.4) == pytest.approx(0.6), "never entered: raised by the boost"
    assert scale(0.04) == pytest.approx(settings.unvisited_floor), "... and read at no less than the floor"
    assert scale(0.9) == pytest.approx(1.0), "a probability, capped"
    assert scale(0.0) == 0.0, "a room the model (or the type prior) gave nothing stays nothing"
    assert policy.loop._value_factors == {}, "a zero is not a factor"
    assert scale(0.4, here=1) == pytest.approx(0.4), "the room the agent stands in keeps its number"
    policy.loop._finished[1] = ("peek", 3)
    assert scale(0.4) == pytest.approx(0.4), "a room finished from outside is not raised"
    del policy.loop._finished[1]
    # Entered and left: the entered factor decays with the time spent inside.
    policy.graph.last_inside[1] = 7
    half = settings.dwell_half_life_steps * policy.settings.action_time_s
    for inside, decay in ((0.0, 1.0), (half, 0.5), (2 * half, 0.25), (10 * half, settings.dwell_floor / settings.entered_discount)):
        policy.graph.facts[1] = RoomFacts(1, 3, inside, fact.cells)   # three frontiers: the lingered rule does not hold
        assert scale(0.6) == pytest.approx(0.6 * max(settings.dwell_floor, settings.entered_discount * decay)), inside
    assert policy.loop._value_factors == {1: settings.dwell_floor}
    # The decay off: the plain entered factor, as before 2026-10-21.
    policy.loop.settings = replace(settings, dwell_half_life_steps=0)
    assert scale(0.6) == pytest.approx(0.6 * settings.entered_discount)
    policy.loop.settings = replace(settings, unvisited_boost=1.0, unvisited_floor=0.0)
    policy.graph.last_inside = {0: 9}
    assert scale(0.04) == pytest.approx(0.04), "the boost and the floor off: the model's number stands"
    for bad in ({"dwell_half_life_steps": -1}, {"dwell_floor": 0.0}, {"unvisited_boost": 0.9}, {"unvisited_floor": 1.5}):
        with pytest.raises(ValueError):
            LoopSettings(**bad)


def test_finished_rooms_are_left_off_the_oracles_node_list_unless_asked_for():
    """A scanned room is not a node (the second pass is off), so it is not shown to the model either: its type
    is on the HOUSE line and its probability reads 0.0 downstream. ``scanned_to_oracle`` restores the lines."""
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), visit="explore", openings={"enabled": False})
    policy.scans.record(obs_at(episode, 5, IN_A), rooms[0], source="warmup")
    policy.loop.plan(obs_at(episode, 6, IN_A), world)
    assert policy.loop._excluded.get(0, "").startswith("scanned")
    assert policy.reasoned_with[-1]["exclude"] == (0,) and policy.reasoned_with[-1]["context"].house.startswith("rooms found")
    shown, episode2, world2, rooms2, _ = loop_policy(order=(1, 0), visit="explore", openings={"enabled": False},
                                                     scanned_to_oracle=True)
    shown.scans.record(obs_at(episode2, 5, IN_A), rooms2[0], source="warmup")
    shown.loop.plan(obs_at(episode2, 6, IN_A), world2)
    assert shown.reasoned_with[-1]["exclude"] == () and shown.reasoned_with[-1]["scanned"] == {0: ("full rotation", 1.0)}
    with pytest.raises(ValueError):
        LoopSettings(scanned_to_oracle=1)
